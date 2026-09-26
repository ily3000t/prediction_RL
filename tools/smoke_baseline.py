"""Bounded P1 diagnostics; NOT formal training or paper reproduction.

Run from project root with the pytorch interpreter. Each output ID is immutable.
"""
import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import re
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'RL-MPC-LaneMerging-master'


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, value):
        for stream in self.streams:
            stream.write(value)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(*args):
    return subprocess.check_output(['git', '-C', str(ROOT), *args], text=True, encoding='utf-8')


def run(mode, output):
    import numpy as np
    import torch
    from config import Settings
    config = SOURCE / 'configs' / ('train_default_1.json' if mode in ('train', 'train_entry', 'semantics') else 'combined_default_1.json')
    Settings.load_from_file(config)
    Settings.SYSTEM = 'Windows' if os.name == 'nt' else 'Linux'
    if mode == 'train_entry':
        Settings.FULL_LOG_DIR = str(output / 'training_logs')
    # Keep upstream Cython, reward, action period and seed. Diagnostic budget only.
    assert Settings.USE_CYTHON
    random.seed(Settings.SEED)
    np.random.seed(Settings.SEED)
    torch.manual_seed(Settings.SEED)
    if Settings.CUDA:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    resolved = Settings.export_settings()
    (output / 'resolved_config.json').write_text(json.dumps(resolved, indent=2, sort_keys=True), encoding='utf-8')
    result = {'config': str(config.relative_to(ROOT)), 'config_sha256': digest(config),
              'resolved_config_sha256': digest(output / 'resolved_config.json'), 'seed': Settings.SEED,
              'execution_contract': 'simulation_blocking_exact_v1', 'cython': True}
    import merge_gym
    import ddpg
    import control
    import prediction
    import st
    import sumo
    import traci
    import st_cy
    result['versions'] = {name: importlib.metadata.version(name) for name in
                          ('torch','numpy','gym','Cython','cvxopt','autonomous-learning-library','tensorboardX','traci')}
    result['python'] = sys.version
    result['python_executable'] = sys.executable
    result['os'] = platform.platform()
    result['sumo_binary'] = shutil.which('sumo.exe' if os.name == 'nt' else 'sumo')
    result['sumo_version'] = subprocess.check_output([result['sumo_binary'], '--version'], text=True).splitlines()[0]
    result['traci_module'] = traci.__file__
    result['cython_extension_sha256'] = digest(Path(st_cy.__file__))
    merge_gym.register_environments()
    agent = None
    env = None
    try:
        if mode == 'semantics':
            traces = []
            for wrapped in (False, True):
                # Recreate the module's fresh-process traffic state for BOTH traces.
                # start_sumo() itself does not reset this upstream global.
                control.delay = 0
                random.seed(Settings.SEED)
                np.random.seed(Settings.SEED)
                torch.manual_seed(Settings.SEED)
                env = (ddpg.DDPGAgent().env if wrapped else merge_gym.ContinuousJerkEnv({}))
                reset = env.reset()
                # ALL has always converted observations to the declared space dtype.
                dtype = env.state_space.dtype if wrapped else env.observation_space.dtype
                initial = reset.features.detach().cpu().numpy().reshape(-1) if wrapped else np.asarray(reset, dtype=dtype)
                trace = [initial.tolist()]
                for jerk in [-5., -2.5, 0., 2.5, 5.] * 4:
                    if wrapped:
                        state, reward = env.step(torch.tensor([jerk], device=env.device))
                        obs = state.features.detach().cpu().numpy().reshape(-1)
                        done = env.done
                    else:
                        obs, reward, done, _ = env.step(np.array([jerk], dtype=np.float32))
                    trace.append([np.asarray(obs, dtype=dtype).tolist(), float(reward), bool(done)])
                    if done:
                        break
                traces.append(trace)
                env.close()
                env = None
            (output / 'semantics_traces.json').write_text(json.dumps(
                {'raw_at_declared_dtype':traces[0], 'adapted':traces[1]}, indent=2), encoding='utf-8')
            assert traces[0] == traces[1], 'Raw/adapted observations, rewards or done flags differ'
            result.update(exact_trace_match=True, steps=len(traces[0])-1)
        elif mode == 'train_entry':
            # The original entry checks its frame budget between episodes:
            # requesting one frame runs exactly one episode, bounded at 500 steps.
            ddpg.DDPGAgent.train(1)
            events = list((output / 'training_logs').rglob('events.out.tfevents.*'))
            assert events, 'Original training entry did not preserve TensorBoard logs'
            result.update(requested_frames=1, max_episode_steps=500,
                          tensorboard_files=len(events),
                          note='Original warmup and minibatch unchanged; optimizer test is separate')
        elif mode == 'train':
            agent = ddpg.DDPGAgent()
            env = agent.env
            from all.presets.continuous import ddpg as preset
            policy = preset(device=agent.device, lr_q=Settings.LEARNING_RATE, lr_pi=Settings.LEARNING_RATE,
                            replay_start_size=8, minibatch_size=8)(env)
            base = policy.agent
            before_q = [p.detach().clone() for p in base.q.model.parameters()]
            before_pi = [p.detach().clone() for p in base.policy.model.parameters()]
            state = env.reset()
            for step in range(64):
                action = policy.act(state, env.reward)
                assert torch.isfinite(action).all()
                state, reward = env.step(action)
                assert np.isfinite(reward)
                if env.done:
                    policy.act(state, reward)
                    state = env.reset()
            q_change = sum((a-b).abs().sum().item() for a,b in zip(before_q, base.q.model.parameters()))
            pi_change = sum((a-b).abs().sum().item() for a,b in zip(before_pi, base.policy.model.parameters()))
            assert q_change > 0 and pi_change > 0
            assert all(torch.isfinite(p).all() for p in list(base.q.model.parameters()) + list(base.policy.model.parameters()))
            result.update(steps=64, critic_parameter_l1_change=q_change, actor_parameter_l1_change=pi_change,
                          diagnostic_overrides={'replay_start_size':8,'minibatch_size':8},
                          note='Optimizer smoke only; reduced warmup/batch are NOT formal training settings')
        else:
            if mode == 'st':
                sumo.start_sumo()
                controller = st.do_st_control
                callback = None
            else:
                model = SOURCE / 'pretrained_models' / 'ddpg_default1_extended'
                result['checkpoint_sha256'] = {n:digest(model/n) for n in ('policy.pt','q.pt')}
                agent = ddpg.DDPGAgent.load(str(model))
                controller = agent.do_control if mode == 'rl' else agent.do_combined_control
                callback = agent.end_episode_callback
            episode = control.run_episode(controller, prediction.HighwayState.from_sumo,
                                          max_episode_length=100, end_episode_callback=callback)
            result.update({k:episode[k] for k in ('crashed','merged','simulation_time_taken')})
            result['steps'] = len(episode['control_history'])
            result['max_episode_simulation_s'] = 100
            if mode == 'combined':
                result['st_takeover_steps'] = sum(agent.takeover_history)
        return result
    finally:
        if env is not None:
            env.close()
        elif agent is not None:
            agent.env.close()
        elif traci.isLoaded():
            traci.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['rl','st','combined','train','train_entry','semantics'], required=True)
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,48}', args.run_id):
        parser.error('Use a short alphanumeric run ID')
    output = ROOT / 'artifacts' / 'smoke' / args.run_id
    output.mkdir(parents=True, exist_ok=False)
    report = {'mode':args.mode, 'run_id':args.run_id, 'git_commit':git('rev-parse','HEAD').strip(),
              'working_tree_dirty':bool(git('status','--porcelain').strip()),
              'command':subprocess.list2cmdline([sys.executable, *sys.argv]),
              'started_at':datetime.now(timezone.utc).isoformat(), 'status':'running'}
    (output / 'working_tree.diff').write_text(git('diff','HEAD','--'), encoding='utf-8')
    # Include untracked diagnostic source as well as the tracked dirty diff.
    sources = [*SOURCE.glob('*.py'), *SOURCE.glob('*.pyx'), Path(__file__)]
    snapshot = output / 'source_snapshot'
    snapshot.mkdir()
    for path in sources:
        shutil.copy2(path, snapshot / path.name)
    report['source_sha256'] = {str(p.relative_to(ROOT)):digest(p) for p in sources}
    os.chdir(SOURCE)
    sys.path.insert(0, str(SOURCE))
    started = time.perf_counter()
    try:
        with (output / 'console.log').open('w', encoding='utf-8') as log:
            with contextlib.redirect_stdout(Tee(sys.stdout, log)), contextlib.redirect_stderr(Tee(sys.stderr, log)):
                report['result'] = run(args.mode, output)
        report['status'] = 'complete'
    except BaseException:
        report['status'] = 'failed'
        report['failure_reason'] = traceback.format_exc()
        raise
    finally:
        report['elapsed_s'] = time.perf_counter()-started
        report['finished_at'] = datetime.now(timezone.utc).isoformat()
        (output / 'report.json').write_text(json.dumps(report, indent=2, sort_keys=True), encoding='utf-8')
        print('smoke_report=' + str(output/'report.json'), flush=True)


if __name__ == '__main__':
    main()
