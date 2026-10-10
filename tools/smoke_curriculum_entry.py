"""IC02b approved, bounded entry parity smoke. NEVER starts a DDPG learner."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'tools')]

import numpy as np
import torch

from prediction_rl.curriculum.entry_smoke import (
    PROTOCOL, FeaturePair, compare_traces, jobs, json_safe, rng_fingerprint, validate_protocol, validate_trace)
from prediction_rl.curriculum.train_reader import BASELINE_COMMIT, git, inside, strict_json
from prediction_rl.data.collection_store import file_hash as sha, read_json, run_lock, seal_episode, verify_episode, write_once
from prediction_rl.data.dataset_contract import digest
from prediction_rl.data.merge_geometry import MergeGeometry
from prediction_rl.envs.fresh_task import FreshGym20TaskEnv, native_packet, runtime_description, space_description
from audit_curriculum_interaction import output_directory


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return strict_json(Path(path).read_bytes())


def read_sealed_trace(directory, binding):
    """Compare persisted evidence, never a mixture of JSON and live objects."""
    receipt = verify_episode(directory, binding)
    raw = (Path(directory)/'trace.json').read_bytes()
    if hashlib.sha256(raw).hexdigest() != receipt['files'].get('trace.json'):
        raise ValueError('Sealed trace bytes changed before aggregation')
    return strict_json(raw), receipt['metadata']


def runtime():
    binary = shutil.which('sumo.exe') or shutil.which('sumo')
    if binary is None:
        raise FileNotFoundError('SUMO executable is unavailable')
    binary = Path(binary).resolve()
    extensions = list((ROOT/'RL-MPC-LaneMerging-master').glob('st_cy*.pyd'))
    if len(extensions) != 1:
        raise ValueError('Build the original extension in THIS worktree; do not disable Cython')
    return {'python_executable': sys.executable, 'python': sys.version, 'os': platform.platform(),
            'sumo_executable': str(binary), 'sumo_sha256': sha(binary),
            'sumo_version': subprocess.check_output([str(binary), '--version'], text=True).splitlines()[0],
            'SUMO_HOME': os.environ.get('SUMO_HOME'),
            'extension_path': str(extensions[0]), 'extension_sha256': sha(extensions[0]),
            'extension_build_command': 'python -B setup.py build_ext --inplace',
            'extension_build_cwd': str(ROOT/'RL-MPC-LaneMerging-master'),
            'packages': {name: importlib.metadata.version(name) for name in
                         ('torch', 'numpy', 'gym', 'traci', 'Cython', 'setuptools', 'autonomous-learning-library')}}


def inputs(baseline, config):
    baseline = Path(baseline).resolve()
    refs_path = inside(ROOT, config['baseline_references'])
    refs = read(refs_path)
    if (baseline != Path(refs['baseline_repository']).resolve() or ROOT != Path(refs['worktree']).resolve()
            or refs['baseline_commit'] != BASELINE_COMMIT or baseline == ROOT
            or git(baseline, 'rev-parse', 'HEAD') != BASELINE_COMMIT or git(baseline, 'status', '--porcelain')):
        raise ValueError('Pinned read-only baseline or isolated worktree identity changed')
    hashes = {'worktree:'+str(refs_path.relative_to(ROOT)).replace('\\', '/'): sha(refs_path)}
    for ref in refs['references']:
        path = inside(baseline, ref['relative_path'])
        if sha(path) != ref['sha256']:
            raise ValueError('Frozen baseline asset changed: '+ref['relative_path'])
        hashes['baseline:'+ref['relative_path']] = ref['sha256']
    report_path = inside(ROOT, config['ic02a_report'])
    report = read(report_path)
    if (sha(report_path) != config['ic02a_report_sha256'] or report['static_inspection_gate'] is not True
            or report['training_ready'] is not False or report['test_opened'] is not False):
        raise ValueError('Need the accepted immutable offline entry audit')
    hashes['worktree:'+config['ic02a_report']] = sha(report_path)
    source = ROOT/'RL-MPC-LaneMerging-master'
    originals = [p for pattern in ('*.py', '*.pyx', '*.xml', '*.sumocfg') for p in source.glob(pattern)]
    originals.append(inside(ROOT, config['upstream_config']))
    for path in originals:
        relative = path.relative_to(ROOT).as_posix()
        original = inside(baseline, relative)
        if path.read_bytes().replace(b'\r\n', b'\n') != original.read_bytes().replace(b'\r\n', b'\n'):
            raise ValueError('Original environment/configuration changed: '+relative)
        hashes['worktree:'+relative] = sha(path)
        hashes['baseline:'+relative] = sha(original)
    for path in (ROOT/'src/prediction_rl').rglob('*.py'):
        hashes['worktree:'+path.relative_to(ROOT).as_posix()] = sha(path)
    for relative in ('tools/smoke_curriculum_entry.py', 'configs/development/ic02b_entry_smoke_v1.json',
                     'provenance/ic02b_smoke_authorization.json'):
        hashes['worktree:'+relative] = sha(inside(ROOT, relative))
    authorization = read(ROOT/'provenance/ic02b_smoke_authorization.json')
    if (authorization['entry_contract'] != config['entry_contract'] or authorization['maximum_episodes'] != 12
            or authorization['maximum_control_steps'] != 144 or authorization['training_authorized'] is not False
            or authorization['new_counterfactual_branches'] != 0):
        raise ValueError('Approved smoke scope changed')
    historical = next(ref for ref in refs['references'] if ref['role'] == 'historical_120k_matched_request')
    parent = read(inside(baseline, historical['relative_path']))
    for mode in config['predictor_modes']:
        ensemble = parent['predictors'][mode]
        ref = next(r for r in refs['references'] if r['role'] == mode+'_predictor_frozen_three_member_ensemble')
        if (ensemble['sha256'] != ref['sha256']
                or Path(ensemble['path']).as_posix() != Path(ref['relative_path']).as_posix()):
            raise ValueError('Predictor metadata/path differs from the frozen baseline reference')
    return hashes, parent['predictors'], parent['predictor_training_request']


def require_clean():
    if git(ROOT, 'status', '--porcelain') or git(ROOT, 'branch', '--show-current') != 'codex/interaction-curriculum-v1':
        raise ValueError('Commit tested IC02b source on the isolated curriculum branch before running')


def prepare(baseline, run_id):
    require_clean()
    config = validate_protocol(read(ROOT/'configs/development/ic02b_entry_smoke_v1.json'))
    hashes, ensembles, request = inputs(baseline, config)
    out = output_directory(ROOT, Path(baseline).resolve(), run_id)
    q = {'version': 'ic02b_bounded_entry_request_v1', 'run_id': run_id, 'prepared_at': now(),
         'git_commit_sha': git(ROOT, 'rev-parse', 'HEAD'), 'working_tree_dirty': False,
         'launch_command': subprocess.list2cmdline(sys.orig_argv), 'protocol': config,
         'baseline_root': str(Path(baseline).resolve()), 'input_hashes': hashes, 'runtime': runtime(),
         'predictors': ensembles, 'predictor_training_request': request, 'jobs': jobs(config),
         'approval': {'source': 'human_continue_after_explicit_ic02b_12_episode_144_step_question',
                      'date': '2026-10-10', 'training_authorized': False}}
    out.mkdir(parents=True, exist_ok=False)
    write_once(out/'request.json', q)
    write_once(out/'preparation.json', {'request_hash': digest(q), 'status': 'prepared_not_executed'})
    print('entry_smoke_request='+str(out/'request.json'), flush=True)
    print('confirm_request_hash='+digest(q), flush=True)
    return out/'request.json', digest(q)


def load(path, expected_hash):
    require_clean()
    path = Path(path).resolve()
    if not path.is_relative_to(ROOT/'artifacts/ic') or path.name != 'request.json':
        raise ValueError('Request must be a local immutable IC artifact')
    q = read(path)
    if (q['version'] != 'ic02b_bounded_entry_request_v1' or digest(q) != expected_hash
            or read(path.parent/'preparation.json')['request_hash'] != expected_hash
            or q['git_commit_sha'] != git(ROOT, 'rev-parse', 'HEAD') or q['working_tree_dirty'] is not False):
        raise ValueError('Request hash or clean-source commit changed')
    config = validate_protocol(q['protocol'])
    hashes, ensembles, request = inputs(q['baseline_root'], config)
    if (q['input_hashes'] != hashes or q['runtime'] != runtime() or q['jobs'] != jobs(config)
            or q['predictors'] != ensembles or q['predictor_training_request'] != request):
        raise ValueError('Frozen inputs/environment/configuration changed; no silent adaptation')
    return path, q


def providers(q):
    from prediction_rl.features.probe_features import FrozenProbeFeatures
    from prediction_rl.prediction.interface import PredictorConfig
    from prediction_rl.prediction.kinematic_reference import LaneConstantVelocity
    from prediction_rl.prediction.trained_artifacts import load_trained_ensemble
    result = {}
    config = PredictorConfig.from_dict(q['predictor_training_request']['model'])
    geometry = LaneConstantVelocity(ROOT/'RL-MPC-LaneMerging-master/merge.net.xml')
    for mode, ensemble in q['predictors'].items():
        model = load_trained_ensemble(inside(Path(q['baseline_root']), ensemble['path']),
            q['predictor_training_request'], mode, ensemble['sha256'], ensemble['selected'])
        if len(model.members) != 3:
            raise ValueError('Keep the frozen three-member ensembles')
        result[mode] = FrozenProbeFeatures(geometry, config, model)
    return result


def tensor_hash(provider):
    value = hashlib.sha256()
    for name, tensor in sorted(provider.model.state_dict().items()):
        value.update(name.encode())
        value.update(str((tensor.dtype, tuple(tensor.shape))).encode())
        value.update(tensor.detach().cpu().numpy().tobytes())
    return value.hexdigest()


def stop_reason(packets):
    return 'native_terminal' if packets[-1]['done'] else 'engineering_cap_not_terminal'


def direct(path, expected_hash, job_id):
    path, q = load(path, expected_hash)
    job = next(j for j in q['jobs'] if j['id'] == job_id)
    if job['adapter'] != 'original_direct':
        raise ValueError('Direct worker cannot substitute for the new entry')
    started = read(path.parent/job_id/'started.json')
    if started['job'] != job or started['request_hash'] != expected_hash:
        raise ValueError('Direct episode must have its parent-owned start receipt')
    # A crash leaves this claim. No same-job retry can consume extra smoke episodes.
    write_once(path.parent/job_id/'direct_claim.json', {'job': job, 'request_hash': expected_hash,
                                                      'pid': os.getpid(), 'started_at': now()})
    torch.set_num_threads(q['protocol']['torch_threads'])
    from prediction_rl.envs.upstream import AuditedJerkEnv, upstream_session
    from prediction_rl.evaluation.terminal_events import StepObserver
    source = ROOT/'RL-MPC-LaneMerging-master'
    packets, cursors = [], (0, 0, 0)
    with upstream_session(source, inside(ROOT, PROTOCOL['upstream_config']), job['simulator_seed']) as settings:
        import control
        import traci
        with StepObserver(control, traci, {'id': f"fresh_{job['simulator_seed']}", 'condition_id': 'gym20'}) as observer:
            observer.cap = 114
            env = AuditedJerkEnv()
            try:
                description = {'spaces': space_description(env), 'runtime': runtime_description(
                    source, settings, MergeGeometry(source/'merge.net.xml'))}
                packet, cursors = native_packet(env, observer, cursors, 'reset', 0)
                packets.append(packet)
                for index, jerk in enumerate(PROTOCOL['requested_jerks'], 1):
                    if packets[-1]['done']:
                        break
                    packet, cursors = native_packet(env, observer, cursors, 'step', index, np.asarray([jerk], np.float32))
                    packets.append(packet)
            finally:
                env.close()
    write_once(path.parent/job_id/'native_trace.json', json_safe({
        'job': job, **description, 'packets': packets, 'stop_reason': stop_reason(packets)}))


def enrich(trace, pair):
    for packet in trace['packets']:
        packet['features'] = pair.accept(packet)
    return trace


def worker_trace(q, job, out, frozen_providers):
    source = ROOT/'RL-MPC-LaneMerging-master'
    env = FreshGym20TaskEnv(source, inside(ROOT, PROTOCOL['upstream_config']), job['simulator_seed'],
        out/'environment.log', control_cap=PROTOCOL['max_control_steps_per_episode'])
    try:
        description = {'spaces': space_description(env), 'runtime': env.runtime}
        pair = FeaturePair(description['spaces'], frozen_providers)
        env.reset()
        packets = [deepcopy_packet(env.last_packet)]
        packets[-1]['features'] = pair.accept(packets[-1])
        for jerk in PROTOCOL['requested_jerks']:
            if packets[-1]['done']:
                break
            env.step(np.asarray([jerk], np.float32))
            packets.append(deepcopy_packet(env.last_packet))
            packets[-1]['features'] = pair.accept(packets[-1])
        # At cap the last native done remains FALSE; do not call an extra tick.
        return {'job': job, **description, 'packets': packets, 'stop_reason': stop_reason(packets)}
    finally:
        env.close()


def deepcopy_packet(packet):
    from copy import deepcopy
    return deepcopy(packet)


def execute(path, expected_hash, resume=False):
    path, q = load(path, expected_hash)
    out = path.parent
    if (out/'report.json').exists() or (out/'failure.json').exists():
        raise FileExistsError('Complete/failed report is immutable; do not silently rerun the smoke')
    torch.set_num_threads(PROTOCOL['torch_threads'])
    report = {'status': 'running', 'started_at': now(), 'request_hash': expected_hash,
              'git_commit_sha': q['git_commit_sha'], 'working_tree_dirty': False,
              'launch_command': subprocess.list2cmdline(sys.orig_argv), 'protocol': PROTOCOL,
              'optimizer_updates': 0, 'new_counterfactual_branches': 0, 'ddpg_started': False,
              'test_opened': False, 'training_ready': False, 'runtime_parity_verified': False,
              'rows': [], 'comparisons': []}
    with run_lock(out):
        if not (out/'started.json').exists():
            write_once(out/'started.json', report)
        elif not resume:
            raise ValueError('Interrupted invocation preserved; resume can reuse COMPLETE jobs only')
        try:
            before = rng_fingerprint()
            frozen_providers = providers(q)
            predictor_before = {mode: tensor_hash(provider) for mode, provider in frozen_providers.items()}
            if rng_fingerprint() != before:
                raise ValueError('Frozen predictor loading consumed parent RNG')
            traces = {}
            geometry = MergeGeometry(ROOT/'RL-MPC-LaneMerging-master/merge.net.xml')
            for job in q['jobs']:
                directory = out/job['id']
                binding = {'request_hash': expected_hash, 'job': job}
                if directory.exists():
                    if not resume:
                        raise FileExistsError('Existing episode cannot be overwritten')
                else:
                    directory.mkdir()
                    write_once(directory/'started.json', {'job': job, 'request_hash': expected_hash,
                        'git_commit_sha': q['git_commit_sha'], 'started_at': now(), 'status': 'running'})
                    witness = rng_fingerprint()
                    if job['adapter'] == 'original_direct':
                        command = [sys.executable, '-B', str(Path(__file__).resolve()), 'direct',
                                   '--request', str(path), '--confirm-request-hash', expected_hash, '--job', job['id']]
                        write_once(directory/'launch.json', {'command': command})
                        with (directory/'direct.log').open('x', encoding='utf-8') as log:
                            subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
                        trace = read(directory/'native_trace.json')
                        trace = enrich(trace, FeaturePair(trace['spaces'], frozen_providers))
                    else:
                        trace = worker_trace(q, job, directory, frozen_providers)
                    trace = json_safe(trace)
                    row = {'job': job, **validate_trace(trace, job, geometry),
                           'parent_rng_unchanged': witness == rng_fingerprint(), 'finished_at': now()}
                    if row['parent_rng_unchanged'] is not True:
                        raise ValueError('Task entry/feature calls mutated persistent parent RNG')
                    write_once(directory/'trace.json', trace)
                    seal_episode(directory, binding, row)
                # JSON converts numeric settings keys to strings. Both fresh and
                # resumed jobs must use the same verified on-disk representation.
                trace, row = read_sealed_trace(directory, binding)
                validate_trace(trace, job, geometry)
                traces[job['id']] = trace
                report['rows'].append(row)
                print(f"[ic02b] job={job['id']} steps={row['actual_control_steps']} stop={trace['stop_reason']}", flush=True)
            for seed in PROTOCOL['simulator_seeds']:
                for left, right in ((f'd{seed}r0', f'd{seed}r1'), (f'w{seed}r0', f'w{seed}r1'),
                                    (f'd{seed}r0', f'w{seed}r0'), (f'd{seed}r1', f'w{seed}r1')):
                    report['comparisons'].append({'left': left, 'right': right,
                        **compare_traces(traces[left], traces[right])})
            report['actual_episodes'] = len(report['rows'])
            report['actual_control_steps'] = sum(row['actual_control_steps'] for row in report['rows'])
            report['waiting_insertion_ticks'] = sum(row['warmup_and_insertion_ticks'] for row in report['rows'])
            report['raw_ticks'] = sum(row['raw_ticks'] for row in report['rows'])
            report['feature_calls_per_mode'] = sum(row['feature_calls_per_mode'] for row in report['rows'])
            if (report['actual_episodes'] > 12 or report['actual_control_steps'] > 144
                    or report['waiting_insertion_ticks'] > 1212 or report['feature_calls_per_mode'] > 156):
                raise ValueError('Approved engineering budget exceeded')
            predictor_after = {mode: tensor_hash(provider) for mode, provider in frozen_providers.items()}
            if (predictor_before != predictor_after or rng_fingerprint() != before or any(
                    p.grad is not None or p.requires_grad for provider in frozen_providers.values()
                    for p in provider.model.parameters())):
                raise ValueError('Frozen predictor or persistent parent RNG changed')
            load(path, expected_hash)  # Verify immutable source, baseline and asset bytes after execution.
            report.update(status='complete', finished_at=now(), runtime_parity_verified=True,
                engineering_gate=True, parent_rng_unchanged=True, predictors_frozen=True,
                predictor_tensor_hashes=predictor_after, simulator_scene_ids=3,
                learner_replay_instantiated=False, task_classifier=None, target_distribution_pi=None,
                remaining_gate='task_layer_coverage_and_target_distribution_not_approved')
            write_once(out/'report.json', report)
            print('ic02b_report='+str(out/'report.json'), flush=True)
        except BaseException as error:
            report.update(status='failed', finished_at=now(), engineering_gate=False,
                          failure_type=type(error).__name__, failure_reason=str(error), traceback=traceback.format_exc())
            write_once(out/'failure.json', report)
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('prepare')
    p.add_argument('--baseline-root', required=True)
    p.add_argument('--run-id', required=True)
    p = commands.add_parser('run')
    p.add_argument('--request', required=True)
    p.add_argument('--confirm-request-hash', required=True)
    p.add_argument('--resume', action='store_true')
    p = commands.add_parser('direct')
    p.add_argument('--request', required=True)
    p.add_argument('--confirm-request-hash', required=True)
    p.add_argument('--job', required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(args.baseline_root, args.run_id)
    elif args.command == 'run':
        execute(args.request, args.confirm_request_hash, args.resume)
    else:
        direct(args.request, args.confirm_request_hash, args.job)


if __name__ == '__main__':
    main()
