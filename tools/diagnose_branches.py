"""P3 bounded development-only mechanism diagnostics; no model training."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from audit_environment import sha, write
from prediction_rl.envs.upstream import upstream_session, validate_seed
from prediction_rl.data.branching import (
    ReplayBrancher, RootUnavailable, native_snapshot_probe, response_summary, restore_rng,
)

PLAN_KEYS = {'schema_version', 'upstream_config', 'simulator_seeds', 'root_prefix_steps',
             'prefix_jerk', 'probe_fractions', 'intervention_steps', 'horizon_steps',
             'continuation_jerk', 'response_speed_threshold_mps', 'response_position_threshold_m',
             'worker_wall_limit_s'}


def environment_metadata():
    binary = shutil.which('sumo.exe' if sys.platform == 'win32' else 'sumo')
    if binary is None:
        raise FileNotFoundError('SUMO executable is not on PATH')
    return {'python_executable': sys.executable,
            'packages': {name: importlib.metadata.version(name) for name in
                         ('torch', 'numpy', 'gym', 'traci', 'Cython', 'cvxopt',
                          'autonomous-learning-library')},
            'sumo_binary': binary,
            'sumo_version': subprocess.check_output([binary, '--version'], text=True).splitlines()[0]}


def load_plan(path):
    import math
    plan = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(plan, dict) or set(plan) != PLAN_KEYS or plan['schema_version'] != 1:
        raise ValueError('Unknown/missing P3 fields or schema')
    seeds = plan['simulator_seeds']
    if not isinstance(seeds, list) or not 1 <= len(seeds) <= 3 or len(set(seeds)) != len(seeds):
        raise ValueError('Use one to three distinct predeclared development seeds')
    for seed in seeds:
        validate_seed(seed)
    prefixes = plan['root_prefix_steps']
    if not isinstance(prefixes, list) or not 1 <= len(prefixes) <= 2 or len(set(prefixes)) != len(prefixes):
        raise ValueError('Use at most two distinct fixed roots per seed')
    for value in prefixes:
        if type(value) is not int or not 0 <= value <= 100:
            raise ValueError('Root prefix must be an integer in [0,100]')
    if plan['probe_fractions'] != [0.0, 0.25, 0.5, 0.75, 1.0]:
        raise ValueError('This diagnostic freezes five equally spaced bound-derived probes')
    if type(plan['horizon_steps']) is not int or not 1 <= plan['horizon_steps'] <= 50:
        raise ValueError('Bounded horizon must be 1..50 steps')
    if type(plan['intervention_steps']) is not int or plan['intervention_steps'] != 1:
        raise ValueError('v1 intervention lasts one original decision period')
    for key in ('prefix_jerk', 'continuation_jerk', 'response_speed_threshold_mps',
                'response_position_threshold_m', 'worker_wall_limit_s'):
        value = plan[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f'Invalid finite number: {key}')
    if not 1 <= plan['worker_wall_limit_s'] <= 300:
        raise ValueError('Worker watchdog must not exceed 300 seconds')
    if min(plan['response_speed_threshold_mps'], plan['response_position_threshold_m']) <= 0:
        raise ValueError('Response thresholds must be strictly positive')
    config = Path(plan['upstream_config'])
    if config.is_absolute() or '..' in config.parts:
        raise ValueError('Use a project-relative upstream config')
    return plan


def run_worker(args, plan):
    if args.seed not in plan['simulator_seeds']:
        raise ValueError('Worker seed not in predeclared plan')
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = {'status': 'running', 'seed': args.seed, 'roots': [],
              'backend': ReplayBrancher.backend,
              'started_at': datetime.now(timezone.utc).isoformat()}
    started = time.perf_counter()
    try:
        with upstream_session(ROOT / 'RL-MPC-LaneMerging-master', ROOT / plan['upstream_config'], args.seed) as settings:
            write(output / 'resolved_config.json', settings.export_settings())
            import control
            manager = ReplayBrancher()
            try:
                low, high = settings.MINIMUM_NEGATIVE_JERK, settings.MAXIMUM_POSITIVE_JERK
                probes = [low + f * (high - low) for f in plan['probe_fractions']]
                if not low <= plan['prefix_jerk'] <= high or not low <= plan['continuation_jerk'] <= high:
                    raise ValueError('Prefix/continuation outside upstream jerk bounds')
                report['probe_jerks'] = probes
                report['intervention_s'] = settings.TICK_LENGTH
                report['horizon_s'] = settings.TICK_LENGTH * plan['horizon_steps']
                for prefix_steps in plan['root_prefix_steps']:
                    prefix = [plan['prefix_jerk']] * prefix_steps
                    try:
                        root = manager.capture(prefix)
                    except RootUnavailable as error:
                        report['roots'].append({'prefix_steps': prefix_steps, 'status': 'unavailable', 'reason': str(error)})
                        continue
                    branches, repeated = {}, {}
                    root_result = {'prefix_steps': prefix_steps, 'status': 'complete',
                                   'signature': root.signature, 'prefix_trace_hash': root.prefix_trace_hash,
                                   'root_traffic': root.traffic}
                    for index in list(range(5)) + list(reversed(range(5))):
                        key = str(index)
                        actions = [probes[index]] + [plan['continuation_jerk']] * (plan['horizon_steps'] - 1)
                        trace = manager.branch(root, actions)
                        if key not in branches:
                            branches[key] = trace
                        else:
                            repeated[key] = trace
                    if branches != repeated:
                        raise AssertionError('Repeated reverse-order branches differ')
                    pairs = response_summary(branches, set(root.traffic['vehicles']) - {'ego'},
                                             plan['response_speed_threshold_mps'],
                                             plan['response_position_threshold_m'])
                    root_result.update(repeated_reverse_order_exact=True, comparisons=pairs,
                                       neighbor_response_detected=any(p['neighbor_response_detected'] for p in pairs),
                                       max_ego_position_delta_m=max(p['max_ego_position_delta_m'] for p in pairs),
                                       max_neighbor_speed_delta_mps=max(p['max_neighbor_speed_delta_mps'] for p in pairs),
                                       max_neighbor_position_delta_m=max(p['max_neighbor_position_delta_m'] for p in pairs))
                    write(output / f'root_{prefix_steps}_branches.json', {
                        'root': root_result, 'branches': branches, 'reverse_order_repeats': repeated})
                    report['roots'].append(root_result)
                    print(f"[p3] seed={args.seed} root={prefix_steps} repeat_exact=true neighbor_response={root_result['neighbor_response_detected']}", flush=True)
                    # One native probe, with the same exact root/continuation as replay.
                    if args.seed == plan['simulator_seeds'][0] and prefix_steps == plan['root_prefix_steps'][-1]:
                        manager.close()
                        restore_rng(manager.initial_rng)
                        control.delay = manager.initial_delay
                        try:
                            native = native_snapshot_probe(output / 'native_probe', prefix,
                                                           [plan['continuation_jerk']] * plan['horizon_steps'])
                            native['root_matches_original_replay'] = native['root_before'] == root.traffic
                            # Fraction 0.5 is zero only for the present bounds: find it, don't assume symmetry.
                            matching = [str(i) for i, j in enumerate(probes) if j == plan['continuation_jerk']]
                            native['direct_matches_replay'] = bool(matching) and native['direct'] == branches[matching[0]]
                            native['native_eligible'] &= native['root_matches_original_replay'] and native['direct_matches_replay']
                            write(output / 'native_probe' / 'comparison.json', native)
                            report['native_probe'] = {k: v for k, v in native.items()
                                                      if k not in ('root_before', 'root_after', 'direct', 'restored')}
                        except Exception:
                            report['native_probe'] = {'native_eligible': False, 'status': 'error',
                                                      'failure_reason': traceback.format_exc()}
                report['status'] = 'complete'
            finally:
                manager.close()
    except BaseException:
        report.update(status='failed', failure_reason=traceback.format_exc())
        raise
    finally:
        report['elapsed_s'] = time.perf_counter() - started
        report['finished_at'] = datetime.now(timezone.utc).isoformat()
        write(output / 'report.json', report)


def parent(args, plan):
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,40}', args.run_id):
        raise ValueError('Use a short unique run ID')
    output = ROOT / 'artifacts/p3' / args.run_id
    output.mkdir(parents=True, exist_ok=False)
    git = lambda *a: subprocess.check_output(['git', '-C', str(ROOT), *a], text=True, encoding='utf-8')
    report = {'status': 'running', 'git_commit': git('rev-parse', 'HEAD').strip(),
              'working_tree_dirty': bool(git('status', '--porcelain').strip()),
              'command': subprocess.list2cmdline([sys.executable, *sys.argv]),
              'python': sys.version, 'os': platform.platform(),
              'started_at': datetime.now(timezone.utc).isoformat(),
              'execution_contract': 'simulation_blocking_exact_v1', 'formal_experiment': False,
              'plan': plan, 'plan_sha256': sha(Path(args.config)),
              'environment_versions': environment_metadata(),
              'optimizer_seed': None, 'predictor_checkpoint_hash': None,
              'policy_checkpoint_hash': None, 'dataset_manifest_hash': None,
              'backend': ReplayBrancher.backend}
    write(output / 'resolved_diagnostic_config.json', plan)
    (output / 'working_tree.diff').write_text(git('diff', 'HEAD', '--'), encoding='utf-8')
    paths = [*ROOT.joinpath('src').rglob('*.py'), *ROOT.joinpath('tools').glob('*.py'),
             *ROOT.joinpath('RL-MPC-LaneMerging-master').glob('*.py'),
             *ROOT.joinpath('RL-MPC-LaneMerging-master').glob('*.pyx'),
             *ROOT.joinpath('RL-MPC-LaneMerging-master').glob('*.xml'),
             *ROOT.joinpath('RL-MPC-LaneMerging-master').glob('*.sumocfg'), ROOT / plan['upstream_config']]
    for path in paths:
        destination = output / 'source_snapshot' / path.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    report['source_sha256'] = {str(p.relative_to(ROOT)): sha(p) for p in paths}
    started = time.perf_counter()
    try:
        children = []
        for seed in plan['simulator_seeds']:
            print(f'[p3] seed={seed}: fixed roots and ten branch rollouts/root', flush=True)
            with (output / f'seed_{seed}.log').open('w', encoding='utf-8') as log:
                subprocess.run([sys.executable, str(Path(__file__).resolve()), '--worker',
                                '--config', str(Path(args.config).resolve()), '--seed', str(seed),
                                '--output', str(output / f'seed_{seed}')], cwd=ROOT,
                               stdout=log, stderr=subprocess.STDOUT, check=True,
                               timeout=plan['worker_wall_limit_s'])
            children.append(json.loads((output / f'seed_{seed}/report.json').read_text()))
        roots = [r for c in children for r in c['roots']]
        complete = [r for r in roots if r['status'] == 'complete']
        affected_seeds = [c['seed'] for c in children if any(r.get('neighbor_response_detected') for r in c['roots'])]
        report.update(status='complete', children=children,
                      engineering_gate=bool(complete) and all(r['repeated_reverse_order_exact'] for r in complete),
                      expected_root_count=len(plan['simulator_seeds']) * len(plan['root_prefix_steps']),
                      evaluated_root_count=len(complete),
                      response_root_count=sum(r['neighbor_response_detected'] for r in complete),
                      response_seeds=affected_seeds)
        # A development signal, NOT statistical evidence or a tunable safety threshold.
        report['mechanism_gate'] = len(complete) == report['expected_root_count'] and len(affected_seeds) >= 2
        report['next_stage'] = 'P4_design_review' if report['mechanism_gate'] else 'pause_for_mechanism_diagnosis'
    except BaseException:
        report.update(status='failed', engineering_gate=False, failure_reason=traceback.format_exc())
        raise
    finally:
        report['elapsed_s'] = time.perf_counter() - started
        report['finished_at'] = datetime.now(timezone.utc).isoformat()
        write(output / 'report.json', report)
        print('p3_report=' + str(output / 'report.json'), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=str(ROOT / 'configs/development/p03_mechanism_v1.json'))
    parser.add_argument('--run-id')
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--output')
    args = parser.parse_args()
    plan = load_plan(args.config)
    if args.worker:
        if args.seed is None or args.output is None:
            parser.error('worker requires seed and output')
        run_worker(args, plan)
    else:
        if args.run_id is None:
            parser.error('run-id required')
        parent(args, plan)
