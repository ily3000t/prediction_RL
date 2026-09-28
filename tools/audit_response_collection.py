"""Development-only collector: three inspected episodes, no formal executor."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import re
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from diagnose_branches import environment_metadata
from prediction_rl.data.collection_store import (
    read_json as read, file_hash as sha, write_once, run_lock, seal_episode, verify_episode,
)
from prediction_rl.data.dataset_contract import digest, episode_id
from prediction_rl.data.branching import ReplayBrancher, fingerprint

PINNED = {
    'mechanism': 'de70e24603f1647769e557c2649a27e845e34b581139f8900ccec2c138bec4ca',
    'label': '7f4e239d40872262189bf527b366764bd818f69c680e8debe252b595bcbd27c2',
    'training': '174146a8d78962929e18bcd426f9c136db26f0f94deda5bfc449c488459dc3b5',
}


def now():
    return datetime.now(timezone.utc).isoformat()


def project_path(value):
    path = (ROOT / value.replace('\\', '/')).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError('Input path escaped project')
    return path


def validate_config(c):
    fixed = {'schema_version': 1, 'purpose': 'development_collector_parity_only',
             'simulator_seeds': [0, 1, 100], 'history_steps': 11, 'neighbor_capacity': 12,
             'horizon_steps': 25, 'tick_s': .2, 'probe_jerks': [-5, -2.5, 0, 2.5, 5],
             'continuation_jerk': 0, 'branch_repeats': 2, 'worker_wall_limit_s': 300}
    keys = set(fixed) | {'network', 'network_sha256'}
    keys |= {f'source_{k}_{suffix}' for k in PINNED for suffix in ('report', 'sha256')}
    if not isinstance(c, dict) or set(c) != keys:
        raise ValueError('Unknown/missing development collector fields')
    for key, expected in fixed.items():
        if c[key] != expected or fingerprint(c[key]) != fingerprint(expected):
            raise ValueError('Frozen development bound mismatch: ' + key)
    for kind, expected in PINNED.items():
        if c[f'source_{kind}_sha256'] != expected:
            raise ValueError('Use accepted development evidence')
    if c['network_sha256'] != '47874cfb6faa044253f065bb9bf1ab56b24c9ba6aeccd24b1449af21dfffe9f1':
        raise ValueError('Network changed')


def verify_inputs(hashes):
    for name, expected in hashes.items():
        if name.startswith('python_lf:'):
            actual = fingerprint(project_path(name[len('python_lf:'):]).read_text(encoding='utf-8'))
        elif name.startswith('json_canonical:'):
            actual = digest(read(project_path(name[len('json_canonical:'):])))
        else:
            actual = sha(project_path(name))
        if actual != expected:
            raise ValueError('Input/source changed: ' + name)


def prerequisites(config_path):
    c = read(config_path)
    validate_config(c)
    hashes = {'json_canonical:' + str(Path(config_path).resolve().relative_to(ROOT)): digest(c),
              c['network']: c['network_sha256']}
    reports = {}
    for kind in PINNED:
        name = c[f'source_{kind}_report']
        hashes[name] = PINNED[kind]
        if sha(project_path(name)) != PINNED[kind]:
            raise ValueError('Prerequisite evidence changed')
        reports[kind] = read(project_path(name))
    p3, labels, training = (reports[k] for k in ('mechanism', 'label', 'training'))
    if not (p3['engineering_gate'] and labels['label_contract_gate'] and training['training_interface_gate']):
        raise ValueError('Prior engineering gate not accepted')
    if any(r['status'] != 'complete' for r in reports.values()) or training['formal_training_ready']:
        raise ValueError('Require complete development-only prerequisites')
    upstream = {k: v for k, v in p3['source_sha256'].items() if k.startswith('RL-MPC-LaneMerging-master')}
    if digest(upstream) != labels['environment_identity_sha256']:
        raise ValueError('Episode namespace changed')
    hashes.update(upstream)
    policy = p3['plan']['reference_policy']
    hashes[policy['checkpoint']] = policy['sha256']
    hashes.update(labels['source_file_hashes'])
    for record in training['history_inputs'] + training['label_inputs']:
        hashes[record['path']] = record['sha256']
    label_dir = project_path(c['source_label_report']).parent
    manifest_path = label_dir / 'split_manifest.json'
    hashes[str(manifest_path.relative_to(ROOT))] = training['split_manifest_sha256']
    if (p3['plan']['simulator_seeds'] != c['simulator_seeds'] or
            p3['plan']['horizon_steps'] != c['horizon_steps'] or p3['plan']['intervention_steps'] != 1):
        raise ValueError('Development episode/probe definitions changed')
    # Bind all project Python implementations, not only the CLI entry file.
    for folder in ('src', 'tools'):
        for path in sorted((ROOT / folder).rglob('*.py')):
            # Python normalizes source newlines itself; Git CRLF checkout must
            # not invalidate an otherwise identical Windows resume request.
            hashes['python_lf:' + str(path.relative_to(ROOT))] = fingerprint(path.read_text(encoding='utf-8'))
    verify_inputs(hashes)
    return c, reports, hashes, read(manifest_path)


def collect_seed(c, reports, seed, output, binding):
    from prediction_rl.data.episode_collector import collect_episode
    p3, labels, training = (reports[k] for k in ('mechanism', 'label', 'training'))
    expected_child = next(child for child in p3['children'] if child['seed'] == seed)
    reference = p3['plan']['reference_policy']
    baseline = {'upstream_config': p3['plan']['upstream_config'],
                'reference_checkpoint': reference['checkpoint'], 'reference_sha256': reference['sha256'],
                'episode_group_namespace_sha256': labels['environment_identity_sha256']}
    collection = {k: c[k] for k in ('history_steps', 'neighbor_capacity', 'horizon_steps',
                  'tick_s', 'probe_jerks', 'continuation_jerk', 'branch_repeats')}
    collection.update(episode_index=0, backend=ReplayBrancher.backend,
                      root_targets=p3['plan']['root_targets'], discovery_max_steps=p3['plan']['discovery_max_steps'])
    job = {'split': 'development', 'simulator_seed': seed,
           'episode_id': episode_id(labels['environment_identity_sha256'], seed, 0)}

    def audit(stage, value):
        if stage == 'discovery':
            if len(value['targets']) != len(expected_child['roots']):
                raise AssertionError('Predeclared roots missing')
            for planned, expected in zip(value['targets'], expected_child['roots']):
                if any(planned[k] != expected[k] for k in planned):
                    raise AssertionError('Accepted development discovery changed')
            return
        number, root = value['number'], value['root']
        expected = expected_child['roots'][number]
        if (root.signature != expected['signature'] or root.prefix_trace_hash != expected['prefix_trace_hash']
                or root.traffic != expected['root_traffic']):
            raise AssertionError('Accepted original root changed')
        record = next(r for r in training['history_inputs'] if r['root_id'] == value['pack']['root_id'])
        old_history = read(project_path(record['path']))
        if any(value['history'][k] != old_history[k] for k in value['history']):
            raise AssertionError('History/input construction changed')
        if value['pack'] != read(project_path(old_history['label_pack'])):
            raise AssertionError('Independent label construction changed')
        raw = read(project_path(c['source_mechanism_report']).parent /
                   f"seed_{seed}/root_{len(root.prefix)}_branches.json")
        if value['branches'] != raw['branches'] or value['branches'] != raw['reverse_order_repeats']:
            raise AssertionError('Original transition/reward/observation changed')
        print(f'[p4f] seed={seed} root={number} original/history/labels=exact', flush=True)

    return collect_episode(ROOT, baseline, collection, c['network'], job, output, binding, audit=audit)


def aggregate(output, request, reports):
    from prediction_rl.prediction.interface import PredictorConfig
    from prediction_rl.prediction.supervision import align_supervision
    histories, packs, children = [], [], []
    for index, seed in enumerate(request['config']['simulator_seeds']):
        directory = output / f'e{index:03d}'
        child = verify_episode(directory, {'request_hash': digest(request), 'seed': seed})
        children.append(child['metadata'])
        for number, root in enumerate(child['metadata']['roots']):
            if root['status'] == 'complete':
                histories.append(read(directory / f'r{number}/history.json'))
                packs.append(read(directory / f'r{number}/labels.json'))
    batch = align_supervision(histories, packs, request['split_manifest'], PredictorConfig(), split='development')
    if batch.fingerprint() != reports['training']['data_hash']:
        raise AssertionError('Complete input/target/mask batch changed relative to accepted P4d')
    roots = [r for child in children for r in child['roots']]
    if len(roots) != 9 or any(r['status'] != 'complete' for r in roots):
        raise AssertionError('Accepted development coverage changed; keep every missing root recorded')
    return {'status': 'complete', 'collector_gate': True, 'children': children,
            'root_count': len(roots), 'candidate_count': sum(r['candidate_count'] for r in roots),
            'branch_executions': sum(r['branch_executions'] for r in roots),
            'terminal_branches': sum(r['terminal_branches'] for r in roots),
            'future_observed_frames': sum(r['future_observed_frames'] for r in roots),
            'supervision_data_hash': batch.fingerprint(), 'valid_actor_time_cells': int(batch.target_mask.sum())}


def run(args):
    c, reports, hashes, manifest = prerequisites(Path(args.config).resolve())
    if not re.fullmatch('[A-Za-z0-9_-]{1,32}', args.run_id):
        raise ValueError('Use a short unique run ID')
    git = lambda *a: subprocess.check_output(['git', '-C', str(ROOT), *a], text=True, encoding='utf-8').strip()
    if git('status', '--porcelain'):
        raise ValueError('Commit collector source before bounded acceptance')
    request = {'schema_version': 1, 'config': c, 'input_hashes': hashes,
               'environment': environment_metadata(), 'split_manifest': manifest,
               'execution_contract': 'simulation_blocking_exact_v1', 'formal_collection_authorized': False}
    output = ROOT / 'artifacts/p4' / args.run_id
    if args.resume:
        if not output.is_dir():
            raise ValueError('Cannot resume absent run')
    else:
        output.mkdir(parents=True, exist_ok=False)
    with run_lock(output):
        if args.resume:
            if read(output / 'request.json') != request:
                raise ValueError('Resume request/config/environment/source mismatch')
        else:
            write_once(output / 'request.json', request)
        invocation = output / 'invocations' / uuid.uuid4().hex[:12]
        invocation.mkdir(parents=True, exist_ok=False)
        report = {'status': 'running', 'collector_gate': False, 'git_commit': git('rev-parse', 'HEAD'),
                  'working_tree_dirty': False, 'started_at': now(), 'request_hash': digest(request),
                  'command': subprocess.list2cmdline([sys.executable, *sys.argv]),
                  'executed_episodes': [], 'reused_episodes': [], 'formal_training_ready': False}
        write_once(invocation / 'started.json', report)
        started = time.perf_counter()
        try:
            for index, seed in enumerate(c['simulator_seeds']):
                directory = output / f'e{index:03d}'
                binding = {'request_hash': digest(request), 'seed': seed}
                if directory.exists():
                    verify_episode(directory, binding)
                    report['reused_episodes'].append(seed)
                    print(f'[p4f] seed={seed} action=verified_reuse', flush=True)
                    continue
                print(f'[p4f] seed={seed} action=collect max_roots=3 max_branches=30', flush=True)
                report['executed_episodes'].append(seed)
                with (invocation / f'e{index:03d}.log').open('x', encoding='utf-8') as log:
                    subprocess.run([sys.executable, str(Path(__file__).resolve()), '--worker',
                                    '--request', str(output / 'request.json'), '--seed', str(seed)],
                                   cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True,
                                   timeout=c['worker_wall_limit_s'])
                verify_episode(directory, binding)
            verify_inputs(hashes)
            report.update(aggregate(output, request, reports))
        except BaseException as error:
            report.update(status='failed', collector_gate=False, failure_reason=f'{type(error).__name__}: {error}')
            raise
        finally:
            report.update(finished_at=now(), elapsed_s=time.perf_counter()-started)
            write_once(invocation / 'report.json', report)
            print('collector_report=' + str(invocation / 'report.json'), flush=True)
    return invocation / 'report.json'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config'); p.add_argument('--run-id'); p.add_argument('--resume', action='store_true')
    p.add_argument('--worker', action='store_true'); p.add_argument('--request'); p.add_argument('--seed', type=int)
    args = p.parse_args()
    if args.worker:
        request = read(Path(args.request))
        c = request['config']; validate_config(c)
        if type(args.seed) is not int or args.seed not in c['simulator_seeds']:
            raise ValueError('Only the three inspected development seeds are permitted')
        verify_inputs(request['input_hashes'])
        if request['environment'] != environment_metadata():
            raise ValueError('Worker environment changed')
        reports = {k: read(project_path(c[f'source_{k}_report'])) for k in PINNED}
        output = Path(args.request).resolve().parent / f"e{c['simulator_seeds'].index(args.seed):03d}"
        if not (output.parent / 'writer.lock').is_file():
            raise ValueError('Worker must belong to a locked parent invocation')
        collect_seed(c, reports, args.seed, output, {'request_hash': digest(request), 'seed': args.seed})
    else:
        if not args.config or not args.run_id:
            p.error('--config and --run-id required')
        run(args)


if __name__ == '__main__':
    main()
