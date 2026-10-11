"""No SUMO, real predictors or experiment training; synthetic ALL parity only."""
from copy import deepcopy
from pathlib import Path
import random
import sys
from types import SimpleNamespace

import gym
import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from prediction_rl.curriculum.interaction_intensity import OfflineEpisodeSampler
from prediction_rl.envs.curriculum_stream import CurriculumTaskStream
from prediction_rl.envs.fresh_task import FreshGym20TaskEnv, ENTRY_CONTRACT, EXECUTION_CONTRACT
from prediction_rl.envs.prediction_features import PredictionFeatureEnv
from prediction_rl.training.all_ddpg import make_all_environment, preset_factory, experiment_type
from prediction_rl.training.curriculum_replay import PersistentTaskReplayAudit


PURPOSE = 'offline_integration_test_only'
TASKS = [{'task_id': 'test_'+level, 'simulator_seed': seed, 'level': level}
         for seed, level in enumerate(('L', 'M', 'H'))]


def sampler(mode='curriculum'):
    return OfflineEpisodeSampler(TASKS, mode, 17003, purpose='offline_algorithm_test_only')


class SyntheticWorker(FreshGym20TaskEnv):
    """Exercise the real parent packet/reset/step path without any IPC or SUMO."""
    def __init__(self, selection, events, *, length=3):
        self.seed = selection['simulator_seed']
        self.events, self.length = events, length
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (20,), np.float32)
        self.action_space = gym.spaces.Box(-5., 5., (1,), np.float32)
        self.control_cap = 500
        self._sequence = self._steps = 0
        self._ready = self._reset_used = self._faulted = self._closed = False
        self.last_packet = None
        self.events.append(('create', self.seed))

    def frame(self, tick):
        return {'simulation_time_s': float(tick)*.2,
                'vehicles': {'ego': {'speed': 7., 'acceleration': 0.,
                    'position': [float(tick), 0.], 'lane_id': 'ramp_0'}}}

    def _exchange(self, op, action=None):
        tick = 0 if op == 'reset' else self._steps+1
        done = tick == self.length
        reason = ('upstream_arrival', 'upstream_collision', 'upstream_time_limit')[self.seed%3] if done else None
        reward = None if op == 'reset' else 10. if reason == 'upstream_arrival' else -10. if reason == 'upstream_collision' else -.02
        observation = np.full(20, float(self.seed)+tick+(0. if action is None else float(action[0])), np.float32)
        frame = self.frame(tick)
        info = {'traffic': deepcopy(frame)} if op == 'reset' else {
            'execution_audit': {'after': deepcopy(frame), 'before': self.frame(tick-1),
                'requested_jerk': float(action[0]), 'measured_jerk': 0.,
                'termination_reason': reason, 'invalid_action_reward': 0.}}
        self.events.append((op, self.seed, tick))
        return {'phase': op, 'observation': observation, 'reward': reward, 'done': done,
                'info': info, 'current_frame': frame, 'control_step': tick,
                'raw_step_count': 101+tick, 'raw_events': {}}

    def close(self):
        self.events.append(('close', self.seed))
        self._closed = True


class Provider:
    def __init__(self, mode):
        self.model = SimpleNamespace(mode=mode)
        self.histories = []
    def __call__(self, history):
        self.histories.append(deepcopy(list(history)))
        return np.full(149, len(history)*.01, np.float32)


def build(mode='conditional', sampling_mode='curriculum', *, worker_length=3, on_event=None):
    events = []
    def factory(selection): return SyntheticWorker(selection, events, length=worker_length)
    spaces = SyntheticWorker(TASKS[0], [])
    stream = CurriculumTaskStream(sampler(sampling_mode), factory, spaces.observation_space,
                                  spaces.action_space, purpose=PURPOSE)
    audit = PersistentTaskReplayAudit(stream, purpose=PURPOSE, on_event=on_event)
    provider = Provider(mode)
    feature = PredictionFeatureEnv(stream, mode, provider, stream.current_frame)
    env = make_all_environment(feature, 'cpu', audit.record)
    return stream, audit, env, provider, events


class DirectReference:
    """Independent uninstrumented task fixture using the same boundary schedule."""
    execution_contract, reset_contract = EXECUTION_CONTRACT, ENTRY_CONTRACT
    def __init__(self, sampling_mode):
        self.sampler = sampler(sampling_mode)
        self.worker = None
        spaces = SyntheticWorker(TASKS[0], [])
        self.observation_space, self.action_space = spaces.observation_space, spaces.action_space
    def reset(self):
        if self.worker is not None: self.worker.close()
        self.worker = SyntheticWorker(self.sampler.begin_episode(), [])
        return self.worker.reset()
    def step(self, action):
        result = self.worker.step(action)
        if result[2]: self.sampler.end_episode(self.worker._steps)
        return result
    def current_frame(self): return self.worker.current_frame()
    def close(self):
        if self.worker is not None: self.worker.close()


def train_synthetic(path, mode, sampling_mode, watched):
    torch.manual_seed(555)
    np.random.seed(555)
    random.seed(555)
    saved_events = []
    if watched:
        stream, audit, env, provider, worker_events = build(mode, sampling_mode,
            on_event=lambda kind, value: saved_events.append((kind, value)))
    else:
        stream = DirectReference(sampling_mode)
        provider = Provider(mode)
        feature = PredictionFeatureEnv(stream, mode, provider, stream.current_frame)
        env = make_all_environment(feature, 'cpu')
        audit = None
        worker_events = []
    experiment = experiment_type(path)(preset_factory('cpu', 2e-4, smoke=True), env, quiet=True)
    if watched: audit.install(experiment._agent)
    experiment.train(frames=35, episodes=12)
    experiment._writer.close()
    env.close()
    return experiment, audit, provider, worker_events, saved_events


@pytest.mark.parametrize('mode', ['ordinary', 'conditional'])
@pytest.mark.parametrize('sampling_mode', ['mix', 'curriculum'])
def test_original_loop_actions_rng_optimizers_targets_and_replay_exact(tmp_path, mode, sampling_mode):
    torch.set_num_threads(1)
    plain, _, provider0, _, _ = train_synthetic(tmp_path/'plain', mode, sampling_mode, False)
    rng0 = (torch.get_rng_state().clone(), deepcopy(np.random.get_state()), random.getstate())
    watched, audit, provider1, events, rows = train_synthetic(tmp_path/'watched', mode, sampling_mode, True)
    assert torch.equal(rng0[0], torch.get_rng_state())
    np.testing.assert_array_equal(rng0[1][1], np.random.get_state()[1])
    assert rng0[1][2:] == np.random.get_state()[2:] and rng0[2] == random.getstate()
    a, b = plain._agent.agent, watched._agent.agent
    assert a._frames_seen == b._frames_seen == 48  # Reset/terminal calls are NOT control steps.
    assert plain.frame == watched.frame == 37
    for name in ('policy', 'q'):
        left, right = getattr(a, name), getattr(b, name)
        for key, value in left.model.state_dict().items():
            assert torch.equal(value, right.model.state_dict()[key])
        for key, value in left._target._target.state_dict().items():
            assert torch.equal(value, right._target._target.state_dict()[key])
        assert left._scheduler.state_dict() == right._scheduler.state_dict()
        for state, other in zip(left._optimizer.state.values(), right._optimizer.state.values()):
            for key, value in state.items(): assert torch.equal(value, other[key])
    for left, right in zip(a.replay_buffer, b.replay_buffer):
        assert torch.equal(left[0].features, right[0].features)
        assert torch.equal(left[1], right[1]) and left[2] == right[2]
        assert torch.equal(left[3].features, right[3].features)
        assert torch.equal(left[3].mask, right[3].mask)
    result = audit.snapshot()
    assert result['actual_control_steps'] == result['original_replay_size'] == 36
    assert len(result['completed_episodes']) == 12 and result['active_episode'] is None
    assert result['observed_minibatches'] > 0 and result['reset_bridge_transitions'] == 0
    assert len([x for x in rows if x[0] == 'transition']) == 36
    assert len([x for x in rows if x[0] == 'episode']) == 12
    assert all(len(p) in (1, 2, 3) for p in provider1.histories)  # New task begins at ONE frame.
    assert provider0.histories == provider1.histories
    for index, event in enumerate(events):
        if event[0] == 'create' and index: assert events[index-1][0] == 'close'
    assert all(info is None or not any(k in info for k in ('task_id', 'level', 'collection_stage'))
               for item in b.replay_buffer for state in (item[0], item[3]) for info in state.info)
    base = audit.base
    audit.detach()
    assert b.replay_buffer is audit.inner and base._reshape == audit._original_reshape


def test_no_worker_spawn_before_reset_and_no_switch_before_terminal_store(tmp_path):
    stream, audit, env, provider, events = build()
    assert events == []
    experiment = experiment_type(tmp_path)(preset_factory('cpu', 2e-4), env, quiet=True)
    audit.install(experiment._agent)
    env.reset()
    action = experiment._agent.act(env.state, env.reward)
    with pytest.raises(RuntimeError): stream.reset()
    for _ in range(2):
        env.step(action)
        action = experiment._agent.act(env.state, env.reward)
    env.step(action)
    assert env.done and audit.insertions == 2
    with pytest.raises(RuntimeError): stream.reset()
    experiment._agent.act(env.state, env.reward)  # Terminal sample stored FIRST.
    assert audit.insertions == 3 and stream.last_completion['actual_end_step'] == 3
    env.reset()
    assert provider.histories[-1][0]['simulation_time_s'] == 0.
    experiment._agent.act(env.state, env.reward)
    assert len(experiment._agent.agent.replay_buffer) == 3
    experiment._writer.close()
    env.close()


def test_actual_sample_observer_supports_real_ring_overwrite(tmp_path):
    stream, audit, env, _, _ = build()
    experiment = experiment_type(tmp_path)(preset_factory('cpu', 2e-4, smoke=True), env, quiet=True)
    # Tiny synthetic ring exercises the observer; NOT an experimental preset.
    experiment._agent.agent.replay_buffer.inner.capacity = 10
    audit.install(experiment._agent)
    experiment.train(frames=35)
    result = audit.snapshot()
    assert result['actual_control_steps'] == 36 and result['original_replay_size'] == 10
    assert result['original_replay_position'] == 6 and sum(result['replay_slot_counts'].values()) == 10
    assert sum(result['minibatch_transition_counts'].values()) == result['observed_minibatches']*8
    assert len(audit._by_state) == 10
    experiment._writer.close()
    env.close()


def test_stage_crossing_waits_for_native_terminal_and_keeps_original_replay(tmp_path):
    stream, audit, env, _, _ = build()
    experiment = experiment_type(tmp_path)(preset_factory('cpu', 2e-4), env, quiet=True)
    audit.install(experiment._agent)
    # Shift ONLY the offline fixture's clock, without inventing transitions.
    stream.actual_control_steps = stream._sampler._step = audit.audit._steps = 19999
    audit.insertions = 19999
    experiment._run_training_episode()
    assert stream.last_completion['actual_end_step'] == 20002
    assert stream.last_completion['collection_stage'] == 0
    assert stream.last_completion['crossed_boundaries'] == [
        {'boundary': 20000, 'actual_next_reset_step': 20002, 'overrun': 2}]
    original = experiment._agent.agent
    replay, optimizer = original.replay_buffer, original.q._optimizer
    experiment._run_training_episode()
    assert stream.selection['collection_stage'] == 1 and stream.selection['stage_boundary_overrun'] == 2
    assert original.replay_buffer is replay and original.q._optimizer is optimizer and len(replay) == 6
    assert audit.snapshot()['collected_transitions_by_stage_and_level']['0'] != {'L': 0, 'M': 0, 'H': 0}
    experiment._writer.close()
    env.close()


@pytest.mark.parametrize('fault', ['reset', 'step', 'space', 'entry', 'close', 'metadata'])
def test_worker_faults_are_invalid_not_done_retry_or_replacement(tmp_path, fault):
    stream, audit, env, _, events = build()
    original = stream._factory
    def factory(selection):
        worker = original(selection)
        if fault in ('reset', 'step', 'close'):
            setattr(worker, fault, lambda *a: (_ for _ in ()).throw(RuntimeError('synthetic failure')))
        elif fault == 'space': worker.action_space = gym.spaces.Box(-4., 4., (1,), np.float32)
        elif fault == 'entry': worker.reset_contract = 'continuing_traffic'
        else:
            exchange = worker._exchange
            def altered(*args):
                value = exchange(*args); value['info']['level'] = 'H'; return value
            worker._exchange = altered
        return worker
    stream._factory = factory
    experiment = experiment_type(tmp_path)(preset_factory('cpu', 2e-4), env, quiet=True)
    audit.install(experiment._agent)
    if fault in ('reset', 'space', 'entry', 'metadata'):
        with pytest.raises((RuntimeError, ValueError)): env.reset()
    elif fault == 'step':
        env.reset(); action = experiment._agent.act(env.state, env.reward)
        with pytest.raises(RuntimeError): env.step(action)
    else:
        experiment._run_training_episode()
        with pytest.raises(RuntimeError): env.reset()
    assert stream._faulted
    with pytest.raises(RuntimeError): env.reset()
    assert len([x for x in events if x[0] == 'create']) == 1
    experiment._writer.close()
    if fault != 'close': env.close()


@pytest.mark.parametrize('purpose', [None, 'formal_training', 'bounded_sumo_smoke'])
def test_core_cannot_silently_authorize_live_execution(purpose):
    worker = SyntheticWorker(TASKS[0], [])
    with pytest.raises(ValueError):
        CurriculumTaskStream(sampler(), lambda s: worker, worker.observation_space,
                            worker.action_space, purpose=purpose)


def test_replay_audit_cannot_install_after_first_action_or_install_twice(tmp_path):
    stream, audit, env, _, _ = build()
    experiment = experiment_type(tmp_path)(preset_factory('cpu', 2e-4), env, quiet=True)
    audit.install(experiment._agent)
    with pytest.raises(ValueError): audit.install(experiment._agent)
    env.reset(); experiment._agent.act(env.state, env.reward)
    with pytest.raises(ValueError): audit.detach()
    experiment._writer.close()
    env.close()


@pytest.mark.parametrize('fault', ['action', 'reward', 'time', 'mask', 'metadata'])
def test_alignment_fault_cannot_insert_or_acknowledge_a_transition(tmp_path, fault):
    stream, audit, env, _, _ = build()
    experiment = experiment_type(tmp_path)(preset_factory('cpu', 2e-4), env, quiet=True)
    audit.install(experiment._agent)
    env.reset()
    action = experiment._agent.act(env.state, env.reward)
    env.step(action)
    reward = env.reward
    if fault == 'action': experiment._agent.agent._action = torch.zeros_like(action)+4.99
    elif fault == 'reward': reward += 1.
    elif fault == 'time': experiment._agent.body.timestep += 1.
    elif fault == 'mask': env.state.mask[0] = 0
    else: env.state.info[0]['curriculum_level'] = 'H'
    with pytest.raises(ValueError): experiment._agent.act(env.state, reward)
    assert audit._faulted and len(audit.inner) == 0 and stream.last_completion is None
    with pytest.raises(RuntimeError): audit.sample(1)
    with pytest.raises(RuntimeError): stream.reset()
    experiment._writer.close()
    env.close()
