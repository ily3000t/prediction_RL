from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'tools')]

from prediction_rl.curriculum.entry_smoke import (PROTOCOL, FeaturePair, compare_traces,
    comparable_trace, jobs, rng_fingerprint, validate_protocol, validate_trace)
from prediction_rl.envs.fresh_task import FreshGym20TaskEnv, native_packet, observer_delta


def spaces():
    return {name: {'low': np.full(size, low, np.float32), 'high': np.full(size, high, np.float32),
                   'dtype': 'float32', 'shape': [size]}
            for name, size, low, high in (('observation', 20, -100, 100), ('action', 1, -5, 5))}


def packet(phase='reset', step=0, done=False, reason=None):
    audit = {'requested_jerk': 0., 'termination_reason': reason}
    return {'phase': phase, 'control_step': step, 'observation': np.full(20, step, np.float32),
            'reward': -.1 if phase == 'step' else None, 'done': done,
            'info': {'execution_audit': audit} if phase == 'step' else {'reset_contract': 'original'},
            'current_frame': {'only_current': step}, 'raw_step_count': 101+step,
            'raw_events': {'steps': [], 'creations': [], 'explicit_removals': []}}


class Provider:
    def __init__(self, mode):
        self.model = SimpleNamespace(mode=mode)
        self.histories = []
    def __call__(self, history):
        self.histories.append(deepcopy(history))
        return np.full(149, len(history), np.float32)


def prototype(cap=12):
    from gym.spaces import Box
    env = FreshGym20TaskEnv.__new__(FreshGym20TaskEnv)
    env.control_cap = cap
    env._ready = env._reset_used = env._faulted = env._closed = False
    env._steps = 0
    env.last_packet = None
    env.observation_space = Box(-100, 100, (20,), np.float32)
    env.action_space = Box(-5, 5, (1,), np.float32)
    env.exchanges = []
    env.responses = [packet()] + [packet('step', i) for i in range(1, cap+1)]
    def exchange(op, action=None):
        env.exchanges.append((op, deepcopy(action)))
        return env.responses.pop(0)
    env._exchange = exchange
    return env


def test_versioned_approval_scope_and_exact_budget():
    value = json.loads((ROOT/'configs/development/ic02b_entry_smoke_v1.json').read_text())
    assert validate_protocol(value) == PROTOCOL
    assert len(jobs(value)) == 12
    assert {j['simulator_seed'] for j in jobs(value)} == {11000, 11001, 11002}
    assert not value['training_permitted'] and value['optimizer_updates'] == 0
    assert value['new_counterfactual_branches'] == 0 and value['target_distribution_pi'] is None
    assert 12*len(value['requested_jerks']) == value['control_steps_max'] == 144


@pytest.mark.parametrize('key,value', [('repeats', 3), ('simulator_seeds', [200]),
    ('max_control_steps_per_episode', 13), ('optimizer_updates', 1), ('new_counterfactual_branches', 1),
    ('training_permitted', True), ('prefill_predictor_history', True), ('reference_prefix_before_policy_steps', 1),
    ('device', 'cuda'), ('target_distribution_pi', [.3, .3, .4]), ('unknown', 0), ('schema_version', True)])
def test_scope_budget_and_unknown_fields_cannot_be_silently_changed(key, value):
    config = deepcopy(PROTOCOL)
    config[key] = value
    with pytest.raises(ValueError):
        validate_protocol(config)


def test_single_native_reset_and_cap_do_not_create_terminal_or_add_step():
    env = prototype(2)
    with pytest.raises(RuntimeError):
        env.step(np.array([0.], np.float32))
    observation, info = env.reset()
    assert not env.last_packet['done'] and env._ready and env._reset_used
    for _ in range(2):
        _, reward, done, _ = env.step(np.array([0.], np.float32))
        assert reward == -.1 and done is False
    with pytest.raises(RuntimeError):
        env.step(np.array([0.], np.float32))
    with pytest.raises(RuntimeError):
        env.reset()
    assert len(env.exchanges) == 3 and not env.last_packet['done']
    result = env.current_frame()
    result['only_current'] = 999
    assert env.current_frame() == {'only_current': 2}


@pytest.mark.parametrize('reason', ['upstream_arrival', 'upstream_collision', 'upstream_time_limit'])
def test_native_terminal_and_cleanup_packet_transport_without_relabel(reason):
    env = prototype()
    env.responses[1] = packet('step', 1, True, reason)
    if reason == 'upstream_time_limit':
        env.responses[1]['raw_step_count'] = 103
        env.responses[1]['raw_events']['steps'] = [{'control_tick': True}, {'cleanup_tick': True}]
        env.responses[1]['raw_events']['explicit_removals'] = [{'original_remove': True}]
    env.reset()
    observation, reward, done, info = env.step(np.array([0.], np.float32))
    assert done is True and reward == -.1 and info['execution_audit']['termination_reason'] == reason
    if reason == 'upstream_time_limit':
        assert len(env.last_packet['raw_events']['steps']) == 2
        assert env.last_packet['raw_events']['explicit_removals'] == [{'original_remove': True}]
    with pytest.raises(RuntimeError):
        env.step(np.array([0.], np.float32))
    with pytest.raises(RuntimeError):
        env.reset()


@pytest.mark.parametrize('fault', ['observation', 'reward', 'done', 'sequence', 'missing', 'unknown'])
def test_bad_packet_invalidates_task_without_stale_or_synthetic_done(fault):
    env = prototype()
    env.reset()
    value = env.responses[0]
    if fault == 'observation': value['observation'][0] = np.nan
    elif fault == 'reward': value['reward'] = True
    elif fault == 'done': value['done'] = 1
    elif fault == 'sequence': value['control_step'] = 2
    elif fault == 'missing': del value['current_frame']
    elif fault == 'unknown': value['curriculum_level'] = 'C0'
    with pytest.raises((ValueError, RuntimeError, KeyError)):
        env.step(np.array([0.], np.float32))
    assert env._faulted and env.last_packet['phase'] == 'reset'
    with pytest.raises(RuntimeError): env.reset()
    with pytest.raises(RuntimeError): env.current_frame()


@pytest.mark.parametrize('reply', [None, {'ok': True, 'sequence': 2, 'value': {}},
    {'ok': False, 'sequence': 1, 'error': 'model error'},
    {'ok': True, 'sequence': 1, 'value': {}, 'unknown': 1}])
def test_worker_reply_failures_are_errors_not_normal_transitions(reply):
    env = FreshGym20TaskEnv.__new__(FreshGym20TaskEnv)
    env.timeout = 1
    env._connection = SimpleNamespace(poll=lambda _: True, recv=lambda: reply)
    with pytest.raises(RuntimeError): env._receive(1)


def test_liveness_watchdog_never_returns_an_observation_or_terminal():
    env = FreshGym20TaskEnv.__new__(FreshGym20TaskEnv)
    env.timeout = 1
    env._connection = SimpleNamespace(poll=lambda _: False)
    with pytest.raises(TimeoutError): env._receive(1)


@pytest.mark.parametrize('done', [False, True])
def test_both_original_feature_paths_use_current_only_one_frame_not_prefilled(done):
    providers = {mode: Provider(mode) for mode in ('ordinary', 'conditional')}
    pair = FeaturePair(spaces(), providers)
    first = pair.accept(packet())
    assert {x['history_frames'] for x in first.values()} == {1}
    value = packet('step', 1, done, 'upstream_arrival' if done else None)
    second = pair.accept(value)
    for mode in providers:
        assert np.array_equal(second[mode]['observation'][:20], value['observation'])
        assert second[mode]['terminal_zero_block'] is done
        if done:
            assert not np.asarray(second[mode]['observation'][20:]).any()
            assert pair.calls[mode] == 1
        else:
            assert pair.calls[mode] == 2 and providers[mode].histories[-1] == [{'only_current': 0}, {'only_current': 1}]


def test_history_caps_at_original_11_and_metadata_stays_outside_input():
    providers = {mode: Provider(mode) for mode in ('ordinary', 'conditional')}
    pair = FeaturePair(spaces(), providers)
    pair.accept(packet())
    for index in range(1, 13):
        value = packet('step', index)
        value['info']['task_id'] = 'not_a_feature'
        result = pair.accept(value)
    assert all(v['history_frames'] == 11 and len(v['observation']) == 169 for v in result.values())
    assert all(len(p.histories[-1]) == 11 and all(set(f) == {'only_current'} for f in p.histories[-1]) for p in providers.values())


def test_raw_observer_delta_copies_cleanup_without_inserting_simulation_calls(monkeypatch):
    observer = SimpleNamespace(events=[{'index': 0}], creations=[{'ego': True}], removals=[])
    frame = {'current': True}
    monkeypatch.setattr('prediction_rl.data.merge_geometry.read_extended_traffic', lambda: deepcopy(frame))
    class Base:
        def reset(self): return np.zeros(20, np.float32), {'original': True}
        def step(self, action):
            self.action = action.copy()
            observer.events.extend([{'index': 1}, {'index': 2, 'cleanup': True}])
            observer.removals.append({'explicit_original_remove': True})
            return np.ones(20, np.float32), -.5, True, {'native_terminal': True}
    base = Base()
    first, cursors = native_packet(base, observer, (0, 0, 0), 'reset', 0)
    second, counts = native_packet(base, observer, cursors, 'step', 1, np.array([0.], np.float32))
    assert counts == (3, 1, 1) and first['raw_events']['creations'] == [{'ego': True}]
    assert second['done'] and second['reward'] == -.5 and len(second['raw_events']['steps']) == 2
    assert second['raw_events']['explicit_removals'] == [{'explicit_original_remove': True}]
    second['raw_events']['steps'][0]['index'] = 999
    assert observer.events[1]['index'] == 1


def test_exact_parity_ignores_latency_and_pid_only_not_task_reward_or_features():
    value = {'job': {'id': 'd'}, 'runtime': {'pid': 1, 'settings': {'SEED': 11000}},
             'packets': [{'features': {'ordinary': {'feature_latency_s': .1, 'observation': [0.]}}}]}
    other = deepcopy(value)
    other['job'] = {'id': 'w'}
    other['runtime']['pid'] = 2
    other['packets'][0]['features']['ordinary']['feature_latency_s'] = .9
    assert compare_traces(value, other)['exact_native_and_feature_parity']
    other['packets'][0]['features']['ordinary']['observation'][0] = 1.
    with pytest.raises(ValueError): compare_traces(value, other)
    assert value['runtime']['pid'] == 1


def test_rng_witness_observes_changes_without_consuming_randomness():
    before = rng_fingerprint()
    assert rng_fingerprint() == before
    with torch.random.fork_rng(devices=[]):
        torch.rand(1)
        assert rng_fingerprint()['torch_cpu'] != before['torch_cpu']
    assert rng_fingerprint() == before


def test_bad_caps_rejected_before_spawning(monkeypatch, tmp_path):
    monkeypatch.setattr('prediction_rl.envs.fresh_task.mp.get_context', lambda *a: pytest.fail('Must not spawn'))
    for cap in (0, True, 501):
        with pytest.raises(ValueError): FreshGym20TaskEnv(ROOT, ROOT/'config.json', 11000, tmp_path/'log', control_cap=cap)


@pytest.mark.parametrize('action', [np.array([6.], np.float32), np.array([np.nan], np.float32),
                                  np.array([0., 1.], np.float32)])
def test_invalid_action_faults_parent_without_advancing_the_environment(action):
    env = prototype()
    env.reset()
    with pytest.raises(ValueError): env.step(action)
    assert len(env.exchanges) == 1 and env._faulted


@pytest.mark.parametrize('mode', ['ordinary', 'conditional'])
def test_worker_proxy_can_use_unchanged_feature_wrapper_directly(mode):
    from prediction_rl.envs.prediction_features import PredictionFeatureEnv
    base = prototype()
    provider = Provider(mode)
    env = PredictionFeatureEnv(base, mode, provider, base.current_frame)
    obs, info = env.reset()
    assert obs.shape == (169,) and info['prediction_features']['history_frames'] == 1
    obs, reward, done, info = env.step(np.array([0.], np.float32))
    assert not done and reward == -.1 and len(base.exchanges) == 2
    assert provider.histories[-1] == [{'only_current': 0}, {'only_current': 1}]


def complete_trace():
    job = jobs(PROTOCOL)[0]
    values = []
    native = {'position': [1., 2.], 'speed': 7., 'acceleration': 0., 'lane_id': 'ramp_0'}
    traffic = lambda n: {'simulation_time_s': .2*n, 'vehicles': {'ego': deepcopy(native)}}
    for index in range(13):
        value = packet('reset' if not index else 'step', index)
        end = 101+index
        value['current_frame'] = traffic(end)
        value['raw_events']['steps'] = [{'index': n, 'before': {'simulation_time_s': .2*n},
            'after': {'simulation_time_s': .2*(n+1)}} for n in range(0 if not index else end-1, end)]
        if index == 0:
            value['info']['traffic'] = traffic(end)
            value['raw_events']['creations'] = [{'vehicle_id': 'ego', 'route_id': 'rampRoute',
                'depart_position_m': 40., 'arrival_position_m': 50., 'simulation_time_s': 20.}]
        else:
            value['info']['execution_audit'].update(requested_jerk=PROTOCOL['requested_jerks'][index-1],
                before=traffic(end-1), after=traffic(end), execution_contract='simulation_blocking_exact_v1')
        value['features'] = {mode: {'observation': value['observation'].tolist()+[0.]*149,
            'history_frames': min(11, index+1), 'terminal_zero_block': False, 'time_feature': index*.001,
            'feature_latency_s': .1} for mode in ('ordinary', 'conditional')}
        values.append(value)
    return {'job': job, 'packets': values, 'stop_reason': 'engineering_cap_not_terminal'}


def test_complete_engineering_trace_counts_real_ticks_without_terminal_fabrication():
    value = complete_trace()
    result = validate_trace(value, value['job'], SimpleNamespace(validate_frame=lambda *a, **kw: None))
    assert result == {'actual_control_steps': 12, 'raw_ticks': 113, 'warmup_and_insertion_ticks': 101,
                      'native_done': False, 'feature_calls_per_mode': 13}


@pytest.mark.parametrize('fault', ['tick', 'future_frame', 'current_traffic', 'mask', 'time', 'cap_done', 'action', 'missing_mode'])
def test_mutated_trace_cannot_pass_native_and_feature_contract(fault):
    value = complete_trace()
    last = value['packets'][-1]
    if fault == 'tick': last['raw_events']['steps'][0]['index'] += 1
    elif fault == 'future_frame': last['current_frame']['simulation_time_s'] += .2
    elif fault == 'current_traffic': last['current_frame']['vehicles']['ego']['speed'] += 1
    elif fault == 'mask': last['features']['ordinary']['history_frames'] = 12
    elif fault == 'time': last['features']['ordinary']['time_feature'] += .001
    elif fault == 'cap_done': last['done'] = True
    elif fault == 'action': last['info']['execution_audit']['requested_jerk'] = 2.5
    elif fault == 'missing_mode': del last['features']['conditional']
    with pytest.raises(ValueError):
        validate_trace(value, value['job'], SimpleNamespace(validate_frame=lambda *a, **kw: None))
