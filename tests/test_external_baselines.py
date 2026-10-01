from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'tools')]
import compare_external_baselines as runner
from prediction_rl.data.dataset_contract import digest
from prediction_rl.evaluation.external_baselines import (
    PROTOCOL, AUDIT, ARMS, ASSETS, jobs, validate_protocol, common_settings,
    outcome, parity, latency_summary, aggregate)
from prediction_rl.envs.upstream import upstream_session


def histories(n=3, arrival=True, collision=False):
    return {'control_history': [1.] * n, 'state_history': [None] * n,
            'speed_history': [1.] * n, 'jerk_history': [0.] * n,
            'acceleration_history': [0.] * n, 'position_history': [[1., 0.]] * n,
            'closest_vehicle_history': [], 'disruption_history': [],
            'simulation_time_taken': n * .2, 'merged': arrival, 'crashed': collision}


def rows(config=AUDIT):
    return [{**j, **outcome(histories(), 'a' * 64, None, [.01] * 3, 9.98)} for j in jobs(config)]


def test_config_and_roster():
    assert json.loads((ROOT/'configs/development/p07d_external_baselines_v1.json').read_text()) == PROTOCOL
    assert len(jobs(PROTOCOL)) == 140 and len(jobs(AUDIT)) == 5
    assert len({j['id'] for j in jobs(PROTOCOL)}) == 140
    assert {j['run_seed'] for j in jobs(PROTOCOL) if j['arm'] != 'conditional'} == {None}
    assert sum(j['arm'] == 'conditional' for j in jobs(PROTOCOL)) == 60


@pytest.mark.parametrize('key,value', [('simulator_seeds', [200, 201]), ('conditional_run_seeds', [0]),
    ('warmup_s', 20), ('max_control_calls', 500), ('workers', True), ('extra', 1),
    ('reward_score', 'new_reward'), ('budget_matched', True), ('author_training_replicates', 3)])
def test_design_cannot_silently_change(key, value):
    p = deepcopy(PROTOCOL); p[key] = value
    with pytest.raises(ValueError): validate_protocol(p)


def test_selected_author_configs_share_environment_and_reward():
    common = []
    for arm in ARMS:
        with upstream_session(runner.SOURCE, runner.SOURCE/'configs'/runner.CONFIGS[arm], 200) as settings:
            settings.CUDA = False; common.append(common_settings(settings.export_settings()))
    assert all(c == common[0] for c in common)
    assert common[0]['REWARD_FUNCTION'] == 'Slotted Jerk'
    assert common[0]['INVALID_ACTION_PENALTY'] == 0
    assert common[0]['TICK_LENGTH'] == .2
    p = deepcopy(common[0]); p['CRASH_REWARD'] += 1
    assert digest(p) != digest(common[0])


def test_reward_score_is_exact_upstream_history_convention_not_gym_return():
    with upstream_session(runner.SOURCE, runner.SOURCE/'configs/train_default_1.json', 200):
        import rl, dqn
        e = histories(); e['jerk_history'] = [0., 2., 3.]
        score = rl.get_rl_custom_episode_stats(e, dqn.get_reward_function())['total_reward']
        assert score == pytest.approx(-.02 - .1 * 4 * .2 + 10)
        e['merged'] = False; e['crashed'] = True
        assert rl.get_rl_custom_episode_stats(e, dqn.get_reward_function())['total_reward'] == pytest.approx(-.1 - 10)
        assert outcome(e, 'a' * 64, None, [.1] * 3, score)['author_history_reward_terms'] == 2


def test_negative_results_complete_and_success_only_time_not_pooled_with_failure():
    r = rows(); r[-1].update(arrival=0, collision=0, time_limit=1, simulation_duration_s=100.2)
    a = aggregate(AUDIT, r)
    assert a['status'] == 'complete' and a['method_effect_gate'] is None
    assert a['arms']['conditional']['summary']['successful_episode_time_s'] == {'count': 0, 'mean': None}
    assert a['comparisons']['author_st']['per_run_seed']['0']['time_s_both_successful']['count'] == 0
    assert a['comparisons']['author_ddpg']['per_run_seed']['0']['arrival']['mean'] == -1


def test_author_snapshot_is_not_three_replicates():
    a = aggregate(PROTOCOL, rows(PROTOCOL))
    assert a['arms']['author_ddpg']['episodes'] == 20
    assert a['arms']['author_ddpg']['independent_training_replicates'] == 1
    assert a['arms']['author_st']['independent_training_replicates'] == 0
    assert a['arms']['conditional']['independent_training_replicates'] == 3
    assert len(a['comparisons']['author_ddpg']['paired_rows']) == 60
    assert a['budget_matched'] is False and a['old_p7_episodes_reused'] is False


@pytest.mark.parametrize('change', [lambda r:r.pop(), lambda r:r.append(deepcopy(r[0])),
    lambda r:r[0].update(run_seed=1), lambda r:r[0].update(initial_traffic_sha256='b' * 64),
    lambda r:r[0].update(collision=1), lambda r:r[0].update(arrival=True),
    lambda r:r[0].update(mean_abs_jerk=float('nan'))])
def test_incomplete_unpaired_invalid_results_rejected(change):
    r = rows(); change(r)
    with pytest.raises(ValueError): aggregate(AUDIT, r)


@pytest.mark.parametrize('key', ['control_history', 'position_history', 'speed_history', 'acceleration_history',
    'jerk_history', 'closest_vehicle_history', 'disruption_history', 'simulation_time_taken', 'crashed', 'merged'])
def test_parity_is_exact_and_includes_execution_and_safety(key):
    left = histories(); right = deepcopy(left); right[key] = 'changed'
    with pytest.raises(ValueError): parity(left, right)
    assert parity(left, deepcopy(left)) is True


@pytest.mark.parametrize('values', [[], [float('nan')], [-.1], [[.1]], [float('inf')]])
def test_invalid_latency_rejected(values):
    with pytest.raises(ValueError): latency_summary(values)


def test_latency_not_effect_gate_and_preserves_tails():
    s = latency_summary([.1, .2, 2.]); assert s['max'] == 2 and s['p99'] > s['p95'] > s['p50']
    a = aggregate(AUDIT, rows()); assert a['method_effect_gate'] is None
    assert a['arms']['author_ddpg']['summary']['controller_latency_count'] == 3
    assert a['arms']['author_ddpg']['summary']['controller_latency_s']['max'] == .01


@pytest.mark.parametrize('n', [0, 502])
def test_original_501_call_bound(n):
    with pytest.raises(ValueError): outcome(histories(n), 'a' * 64, None, [.1] * n, 0.)


def test_masks_and_takeover_cannot_be_silently_filled():
    with pytest.raises(ValueError): outcome(histories(), 'a' * 64, [True], [.1] * 3, 0.)
    with pytest.raises(ValueError): outcome(histories(), 'a' * 64, [1] * 3, [.1] * 3, 0.)
    r = outcome(histories(501, False), 'a' * 64, [False] * 501, [.1] * 501, -10.)
    assert r['time_limit'] == 1 and r['simulation_duration_s'] == pytest.approx(100.2)


def test_unknown_pickle_file_rejected_before_load(tmp_path, monkeypatch):
    for name in ASSETS: (tmp_path/name).write_bytes(b'changed')
    monkeypatch.setattr(runner, 'MODEL', tmp_path)
    with pytest.raises(ValueError, match='checkpoint changed'): runner.assets()
    (tmp_path/'feature.pt').write_bytes(b'unknown')
    with pytest.raises(ValueError, match='Unexpected author'): runner.assets()


def test_conditional_terminal_does_not_predict_or_execute():
    c = object.__new__(runner.ConditionalController); c.last_obs = np.arange(169, dtype=np.float32)
    calls = []; c.env = SimpleNamespace(_make_state=lambda x,done:(x, done))
    c.checked = SimpleNamespace(eval=lambda state,reward:calls.append((state, reward)))
    c.end(None)
    (obs, terminal), reward = calls[0]
    assert terminal is True and reward == 0
    assert np.array_equal(obs[:20], c.last_obs[:20]) and not np.any(obs[20:])


def test_conditional_query_uses_only_current_observed_history_and_original_jerk(monkeypatch):
    c = object.__new__(runner.ConditionalController); c.history = __import__('collections').deque(maxlen=11); c.actions = []
    observed = {'actual': True}; monkeypatch.setattr(runner, 'read_extended_traffic', lambda:observed)
    seen = []; c.provider = lambda h: seen.append(deepcopy(h)) or np.zeros(149, np.float32)
    c.env = SimpleNamespace(_make_state=lambda x,done:(x,done))
    c.checked = SimpleNamespace(eval=lambda state,reward:SimpleNamespace(item=lambda:2.5))
    monkeypatch.setitem(sys.modules, 'dqn', SimpleNamespace(get_state_vector_from_base_state=lambda s:np.arange(20)))
    commands = []; monkeypatch.setitem(sys.modules, 'control', SimpleNamespace(set_ego_jerk=lambda j:commands.append(j) or 5.))
    assert c(None) == 5. and commands == [2.5] and c.actions == [2.5]
    assert seen == [[observed]] and c.last_obs.shape == (169,)
