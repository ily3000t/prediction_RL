from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import gym
import numpy as np
import pytest
import torch
from all.environments import State
from all.logging import DummyWriter

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'tools')]
import diagnose_driver_warmup as runner
from prediction_rl.data.dataset_contract import digest
from prediction_rl.evaluation.driver_warmup import (PROTOCOL, AUDIT, CONDITIONS, AUTHOR, B0, B3,
    validate_protocol, roster, jobs, is_new, canonical_frame, row_from_native, aggregate, prefix_comparison)
from prediction_rl.training.all_ddpg import preset_factory


def frame(warmup=20, index=0):
    return {'simulation_time_s': warmup+.2*(index+1),
            'vehicles': {'ego': {'position': [1., 2.], 'speed': 5., 'acceleration': 0., 'lane_id': 'ramp_0', 'length_m': 5.},
                         'traffic': {'position': [3., 4.], 'speed': 7., 'acceleration': 0., 'lane_id': 'highwayrear_0'}}}


def records(job, count=3):
    cond = CONDITIONS[job['condition_id']]
    historical_missing = not is_new(job) and job['condition_id'] == 'author50' and job['model_id'] in (AUTHOR, B3)
    dim = 169 if job['model_id'] == B3 else 20
    return [runner.decision(frame(cond['warmup_s'], i), 1., 5., None if historical_missing else [0.]*dim, i,
                time_source='reconstructed_original_single_query_sequence' if historical_missing else 'recorded_policy_query') for i in range(count)]


def results(config=AUDIT):
    rows, traces = [], {}
    for j in roster(config):
        trace = records(j)
        outcome = {'arrival': 1, 'collision': 0, 'time_limit': 0, 'steps': 3, 'simulation_duration_s': .6,
                   'return': 2., 'author_history_total_reward': 7., 'mean_abs_measured_jerk': .2, 'mean_abs_jerk': .3,
                   'policy_sha256': 'a'*64, 'predictor_sha256': 'b'*64 if j['model_id'] == B3 else None}
        rows.append(row_from_native(j, outcome, trace, {'kind': 'new' if is_new(j) else 'historical'}))
        traces[j['id']] = trace
    return rows, traces


def test_exact_design_two_new_conditions_and_no_training():
    assert json.loads((ROOT/'configs/development/p07f_driver_warmup_v1.json').read_text()) == PROTOCOL
    assert len(roster(PROTOCOL)) == 84 and len(jobs(PROTOCOL)) == 42
    assert len(roster(AUDIT)) == 12 and len(jobs(AUDIT)) == 6
    assert PROTOCOL['simulator_seeds'] == [200, 210, 219]
    assert {j['condition_id'] for j in jobs(PROTOCOL)} == {'gym50', 'author20'}
    assert {j['training_id'] for j in jobs(PROTOCOL) if j['model_id'] == B3} == {'conditional_s0', 'conditional_s1', 'conditional_s2'}
    assert PROTOCOL['training_permitted'] is False and PROTOCOL['cross_driver_reward_delta'] is False


@pytest.mark.parametrize('key,value', [('simulator_seeds', [201, 208, 212]), ('project_run_seeds', [0, 1]),
    ('conditions', {}), ('workers', True), ('training_permitted', True), ('extra', 1),
    ('cross_driver_reward_delta', True), ('test_opened', True)])
def test_design_changes_rejected(key, value):
    p = deepcopy(PROTOCOL); p[key] = value
    with pytest.raises(ValueError): validate_protocol(p)


def test_canonical_frame_only_actually_shared_fields():
    f = frame(); common = canonical_frame(f)
    assert 'length_m' not in common['vehicles']['ego']
    assert 'length_m' in f['vehicles']['ego']  # No mutation of historical sidecars.
    assert canonical_frame(common) == common
    f['vehicles']['ego']['speed'] = float('nan')
    with pytest.raises(ValueError): canonical_frame(f)


def test_aggregate_native_scores_separate_and_missing_observations_explicit():
    r, t = results(PROTOCOL); a = aggregate(PROTOCOL, r, t)
    assert a['new_episodes'] == a['verified_historical_episodes'] == 42
    assert a['total_matrix_cells'] == 84 and a['method_effect_gate'] is None
    for comparisons in a['same_warmup_driver_comparisons'].values():
        for c in comparisons.values():
            assert c['reward_delta_computed'] is False and 'native_score' not in c['mean_native_delta']
            assert all(p['recorded_prefix']['initial_common_traffic_equal'] for p in c['paired_rows'])
    for comparisons in a['same_driver_warmup_comparisons'].values():
        for c in comparisons.values(): assert c['reward_delta_computed'] is True
    author50 = a['same_warmup_driver_comparisons'][AUTHOR]['50']['paired_rows'][0]
    assert author50['recorded_prefix']['observation']['exact_equal_on_available_prefix'] is None
    assert author50['recorded_prefix']['common_state_action_sequence_equal'] is True


def test_difference_records_first_divergence_without_interpolation():
    j = jobs(AUDIT)[0]; l = records(j); r = deepcopy(l)
    r[1]['requested_jerk'] = 2.; r[2]['frame']['vehicles']['traffic']['speed'] = 8.
    c = prefix_comparison(l, r)
    assert c['requested_jerk']['first_difference_index'] == 1
    assert c['frame']['first_difference_index'] == 2
    assert not c['common_state_action_sequence_equal']


@pytest.mark.parametrize('fault', ['missing_observation', 'reconstructed_time'])
def test_new_runs_cannot_claim_historical_missing_input_limitations(fault):
    r, t = results(); row = next(x for x in r if is_new(x)); record = t[row['id']][0]
    if fault == 'missing_observation':
        record.update(observation=None, observation_source='not_recorded_in_historical_native_trace')
    else: record['time_feature_source'] = 'reconstructed_original_single_query_sequence'
    row['decisions_sha256'] = digest(t[row['id']])
    with pytest.raises(ValueError, match='New runs require'): aggregate(AUDIT, r, t)


def test_negative_outcomes_still_complete_success_denominator_not_parking():
    r, t = results()
    for row in r:
        if row['model_id'] == B3: row.update(arrival=0, time_limit=1, native_score=-10., native_mean_abs_jerk=0.)
    a = aggregate(AUDIT, r, t)
    assert a['status'] == 'complete' and a['method_effect_gate'] is None
    assert a['matrix'][B3]['gym50']['summary']['successful_episode_time_s']['count'] == 0


@pytest.mark.parametrize('fault', ['missing', 'duplicate', 'identity', 'weights', 'predictor', 'initial',
    'score', 'events', 'steps', 'kind', 'hash', 'trace', 'time', 'observation'])
def test_invalid_cells_or_records_rejected(fault):
    r, t = results(); j = r[0]['id']
    if fault == 'missing': r.pop()
    elif fault == 'duplicate': r.append(deepcopy(r[0]))
    elif fault == 'identity': r[0]['condition_id'] = 'author50'
    elif fault == 'weights': r[0]['policy_sha256'] = 'e'*64
    elif fault == 'predictor': r[0]['predictor_sha256'] = 'b'*64
    elif fault == 'initial': r[0]['common_initial_traffic_sha256'] = 'e'*64
    elif fault == 'score': r[0]['native_score_name'] = 'reward'
    elif fault == 'events': r[0]['time_limit'] = 1
    elif fault == 'steps': r[0]['steps'] = 501
    elif fault == 'kind': r[0]['evidence']['kind'] = 'new'
    elif fault == 'hash': r[0]['policy_sha256'] = 'z'*64
    elif fault == 'trace': t[j][0]['requested_jerk'] = 3.
    elif fault == 'time': t[j][0]['time_feature'] = .001
    elif fault == 'observation': t[j][0]['observation'] = [0.]
    with pytest.raises(ValueError): aggregate(AUDIT, r, t)


@pytest.mark.parametrize('dimension', [20, 169])
def test_frozen_query_exact_for_both_layouts_without_optimizer_or_replay(dimension):
    env = SimpleNamespace(feature_arm='baseline' if dimension == 20 else 'conditional', device='cpu',
        state_space=gym.spaces.Box(-np.inf, np.inf, (dimension,), np.float32),
        action_space=gym.spaces.Box(-5, 5, (1,), np.float32))
    checked = preset_factory('cpu', .0002)(env, DummyWriter())
    before = runner.smoke.tensor_hash(checked.agent.policy.model.state_dict())
    q = runner.FrozenQuery(checked.body, checked.agent.policy.model, dimension)
    def state(done=False):
        return State(torch.linspace(-1, 1, dimension)[None], torch.tensor([0 if done else 1], dtype=torch.uint8), [{}])
    for _ in range(3): q(state())
    q(state(True)); q.complete(3)
    assert [r['time_feature'] for r in q.records] == [float(np.float32(.001)*np.float32(i)) for i in range(3)]
    assert float(checked.body.timestep.item()) == 0.
    assert not len(checked.agent.replay_buffer) and not checked.agent.policy._optimizer.state
    assert runner.smoke.tensor_hash(checked.agent.policy.model.state_dict()) == before
    with pytest.raises(ValueError, match='Query count'): q.complete(4)


def test_warmup_only_instance_field_before_reset():
    raw = SimpleNamespace(wait_before_start=20, current_episode_ticks=0, max_episode_ticks=500, reward_function=object())
    env = SimpleNamespace(env=SimpleNamespace(feature_env=SimpleNamespace(base=SimpleNamespace(raw=raw))))
    before = dict(vars(raw)); runner.configure_warmup(env, 50)
    assert vars(raw) == {**before, 'wait_before_start': 50}
    with pytest.raises(ValueError): runner.configure_warmup(env, 20)


def test_resume_preserves_incomplete_job_no_child_retry(tmp_path, monkeypatch):
    from prediction_rl.data.collection_store import write_once
    j = jobs(AUDIT)[0]; q = {'jobs': [j], 'protocol': AUDIT}
    out = tmp_path/'evaluate'/j['id']; out.mkdir(parents=True); write_once(out/'started.json', {'status': 'incomplete'})
    monkeypatch.setattr(runner, 'ROOT', tmp_path)
    monkeypatch.setattr(runner, 'load', lambda *a:(tmp_path/'request.json', q))
    monkeypatch.setattr(runner.shared, 'metadata', lambda:{'git_commit': 'a'*40})
    monkeypatch.setattr(runner.subprocess, 'run', lambda *a, **k:pytest.fail('Must not retry incomplete jobs'))
    with pytest.raises(ValueError, match='Incomplete/failed'): runner.execute(tmp_path/'request.json', 'frozen', True)
    assert (out/'started.json').exists() and not (out/'failure.json').exists()
    report = next((tmp_path/'invocations').glob('*/report.json'))
    assert json.loads(report.read_text())['status'] == 'failed'
