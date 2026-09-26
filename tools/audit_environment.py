"""P2 bounded process-isolated comparison of upstream and audited environment."""
import argparse
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
import re
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from prediction_rl.envs.upstream import (
    AuditedJerkEnv, observation_at_policy_dtype, traffic_snapshot, upstream_session,
)


def write(path, data):
    path.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False), encoding='utf-8')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def worker(args):
    import numpy as np
    source = ROOT / 'RL-MPC-LaneMerging-master'
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = {'status': 'running', 'kind': args.worker, 'seed': args.seed,
              'sequence': args.sequence, 'episodes': args.episodes,
              'started_at': datetime.now(timezone.utc).isoformat()}
    start = time.perf_counter()
    try:
        config = source / 'configs/train_default_1.json'
        with upstream_session(source, config, args.seed) as settings:
            write(output / 'resolved_config.json', settings.export_settings())
            import merge_gym
            from legacy_gym_compat import LegacyTimeLimit
            import control
            import traci
            env = AuditedJerkEnv() if args.worker == 'adapted' else LegacyTimeLimit(merge_gym.ContinuousJerkEnv({}), 500)
            traces, audits = [], []
            try:
                for episode in range(args.episodes):
                    obs = env.reset()
                    if args.worker == 'adapted':
                        obs, info = obs
                        initial = info['traffic']
                    else:
                        initial = traffic_snapshot()
                    initial_obs = observation_at_policy_dtype(obs, env.observation_space)
                    trace = {'reset_observation': initial_obs.tolist(), 'reset_traffic': initial,
                             'reset_delay': float(control.delay), 'steps': []}
                    episode_audits = []
                    for index in range(500):
                        value = {'positive': [5.], 'negative': [-5.],
                                 'mixed': [-5., -2.5, 0., 2.5, 5.]}[args.sequence]
                        action = np.array([value[index % len(value)]], dtype=np.float32)
                        obs, reward, done, info = env.step(action)
                        info = dict(info)
                        if args.worker == 'adapted':
                            episode_audits.append(info.pop('execution_audit'))
                        trace['steps'].append({
                            'action': action.tolist(),
                            'observation': observation_at_policy_dtype(obs, env.observation_space).tolist(),
                            'reward': float(reward), 'done': bool(done), 'info': info,
                            'traffic': traffic_snapshot(), 'delay': float(control.delay),
                        })
                        if done:
                            break
                    else:
                        raise AssertionError('Upstream registered 500-step cap not respected')
                    traces.append(trace)
                    audits.append(episode_audits)
            finally:
                env.close()
            write(output / 'traces.json', traces)
            write(output / 'execution_audits.json', audits)
            report.update(status='complete', steps=[len(t['steps']) for t in traces],
                          config_sha256=sha(config),
                          resolved_config_sha256=sha(output / 'resolved_config.json'),
                          traces_sha256=sha(output / 'traces.json'),
                          versions={n: importlib.metadata.version(n) for n in
                                    ('torch', 'numpy', 'gym', 'traci')},
                          sumo_version=subprocess.check_output(
                              ['sumo.exe' if sys.platform == 'win32' else 'sumo', '--version'], text=True).splitlines()[0])
    except BaseException:
        report.update(status='failed', failure_reason=traceback.format_exc())
        raise
    finally:
        report['elapsed_s'] = time.perf_counter() - start
        report['finished_at'] = datetime.now(timezone.utc).isoformat()
        write(output / 'report.json', report)


def parent(args):
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,40}', args.run_id):
        raise ValueError('Use a short unique run ID')
    output = ROOT / 'artifacts' / 'p2' / args.run_id
    output.mkdir(parents=True, exist_ok=False)
    git = lambda *a: subprocess.check_output(['git', '-C', str(ROOT), *a], text=True, encoding='utf-8')
    plan = [
        ('raw_stream', 'raw', 0, 'positive', 2),
        ('adapted_stream', 'adapted', 0, 'positive', 2),
        ('repeat_stream', 'adapted', 0, 'positive', 2),
        ('other_seed', 'adapted', 1, 'positive', 2),
        ('raw_limit', 'raw', 0, 'negative', 1),
        ('adapted_limit', 'adapted', 0, 'negative', 1),
        ('raw_mixed', 'raw', 100, 'mixed', 1),
        ('adapted_mixed', 'adapted', 100, 'mixed', 1),
    ]
    report = {'status': 'running', 'git_commit': git('rev-parse', 'HEAD').strip(),
              'working_tree_dirty': bool(git('status', '--porcelain').strip()),
              'command': subprocess.list2cmdline([sys.executable, *sys.argv]),
              'python': sys.version, 'python_executable': sys.executable,
              'os': platform.platform(), 'optimizer_seed': None,
              'predictor_checkpoint_hash': None, 'policy_checkpoint_hash': None,
              'dataset_manifest_hash': None,
              'started_at': datetime.now(timezone.utc).isoformat(),
              'plan': plan, 'execution_contract': 'simulation_blocking_exact_v1',
              'formal_experiment': False}
    (output / 'working_tree.diff').write_text(git('diff', 'HEAD', '--'), encoding='utf-8')
    source_paths = [Path(__file__), *ROOT.joinpath('src').rglob('*.py'),
                    *ROOT.joinpath('RL-MPC-LaneMerging-master').glob('*.py'),
                    *ROOT.joinpath('RL-MPC-LaneMerging-master').glob('*.pyx'),
                    *ROOT.joinpath('RL-MPC-LaneMerging-master').glob('*.xml'),
                    *ROOT.joinpath('RL-MPC-LaneMerging-master').glob('*.sumocfg')]
    import shutil
    for path in source_paths:
        copy = output / 'source_snapshot' / path.relative_to(ROOT)
        copy.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, copy)
    report['source_sha256'] = {str(p.relative_to(ROOT)): sha(p) for p in source_paths}
    start = time.perf_counter()
    try:
        for name, kind, seed, sequence, episodes in plan:
            command = [sys.executable, str(Path(__file__).resolve()), '--worker', kind,
                       '--output', str(output / name), '--seed', str(seed),
                       '--sequence', sequence, '--episodes', str(episodes)]
            print(f'[p2] {name}: episodes<={episodes}, steps/episode<=500', flush=True)
            # Per-worker wall-time limit aborts a DIAGNOSTIC; never changes an observation.
            with (output / (name + '.log')).open('w', encoding='utf-8') as log:
                subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                               check=True, timeout=120)
        read = lambda name: json.loads((output / name / 'traces.json').read_text(encoding='utf-8'))
        checks = {}
        for raw, adapted in [('raw_stream', 'adapted_stream'), ('raw_limit', 'adapted_limit'), ('raw_mixed', 'adapted_mixed')]:
            checks[raw + '_exact_match'] = read(raw) == read(adapted)
        checks['fresh_process_seed_repeatable'] = read('adapted_stream') == read('repeat_stream')
        checks['different_seed_changes_scene'] = read('adapted_stream')[0]['reset_traffic'] != read('other_seed')[0]['reset_traffic']
        limits = json.loads((output / 'adapted_limit/execution_audits.json').read_text())
        checks['500_step_timeout_preserved'] = len(limits[0]) == 500 and limits[0][-1]['termination_reason'] == 'upstream_time_limit'
        checks['upstream_terminal_extra_step_visible'] = abs(limits[0][-1]['simulation_elapsed_s'] - 0.4) < 1e-9
        stream = read('adapted_stream')
        checks['traffic_continues_between_resets'] = stream[1]['reset_traffic']['simulation_time_s'] > stream[0]['steps'][-1]['traffic']['simulation_time_s']
        report['checks'] = checks
        report['children'] = {p[0]: json.loads((output / p[0] / 'report.json').read_text()) for p in plan}
        report['status'] = 'complete' if all(checks.values()) else 'failed'
        if not all(checks.values()):
            raise AssertionError(f'P2 contract checks failed: {checks}')
    except BaseException:
        report.update(status='failed', failure_reason=traceback.format_exc())
        raise
    finally:
        report['elapsed_s'] = time.perf_counter() - start
        report['finished_at'] = datetime.now(timezone.utc).isoformat()
        write(output / 'report.json', report)
        print('p2_report=' + str(output / 'report.json'), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id')
    parser.add_argument('--worker', choices=['raw', 'adapted'])
    parser.add_argument('--output')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--sequence', choices=['positive', 'negative', 'mixed'])
    parser.add_argument('--episodes', type=int, choices=[1, 2])
    args = parser.parse_args()
    if args.worker:
        if any(v is None for v in (args.output, args.seed, args.sequence, args.episodes)):
            parser.error('worker requires output, seed, sequence and episodes')
        worker(args)
    else:
        if args.run_id is None:
            parser.error('run-id is required')
        parent(args)
