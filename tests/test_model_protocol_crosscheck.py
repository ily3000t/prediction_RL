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
import crosscheck_model_protocols as runner
from prediction_rl.evaluation.model_protocol_crosscheck import (
    PROTOCOL, AUDIT, AUTHOR, B0, B3, GYM, AUTHOR_LOOP, SCORES,
    roster, jobs, is_new, validate_protocol, native_row, aggregate)
from prediction_rl.training.all_ddpg import preset_factory


def rows(config=AUDIT):
    return [{**j, 'arrival': 1, 'reported_collision': 0, 'time_limit': 0,
             'steps': 3, 'simulation_duration_s': .6, 'native_score_name': SCORES[j['driver_id']],
             'native_score': 1. if j['driver_id'] == GYM else 7., 'native_mean_abs_jerk': .1,
             'policy_sha256': 'a'*64, 'predictor_sha256': 'b'*64 if j['model_id'] == B3 else None,
             'initial_traffic_sha256': ('c' if j['driver_id'] == GYM else 'd')*64,
             'evidence': {'kind': 'new' if is_new(j) else 'historical'}} for j in roster(config)]


def test_exact_config_and_only_two_missing_cells():
    assert json.loads((ROOT/'configs/development/p07e_model_protocol_crosscheck_v1.json').read_text()) == PROTOCOL
    assert len(jobs(PROTOCOL)) == 80 and len(roster(PROTOCOL)) == 280
    assert len(jobs(AUDIT)) == 4 and len(roster(AUDIT)) == 14
    assert sum(j['model_id'] == AUTHOR for j in jobs(PROTOCOL)) == 20
    assert {j['training_id'] for j in jobs(PROTOCOL) if j['model_id'] != AUTHOR} == {'baseline_s0', 'baseline_s1', 'baseline_s2'}
    assert all(j['driver_id'] == (GYM if j['model_id'] == AUTHOR else AUTHOR_LOOP) for j in jobs(PROTOCOL))
    assert PROTOCOL['training_permitted'] is False


@pytest.mark.parametrize('key,value', [('simulator_seeds', [200]), ('project_run_seeds', [0, 1]),
    ('workers', True), ('training_permitted', True), ('external_request_hash', 'a'*64),
    ('extra', 1), ('cross_protocol_reward_delta', True), ('formal_claim', True)])
def test_design_changes_rejected(key, value):
    p = deepcopy(PROTOCOL); p[key] = value
    with pytest.raises(ValueError): validate_protocol(p)


def test_native_score_mapping_not_relabeling_reward():
    j = jobs(AUDIT)[0]
    row = {'arrival': 1, 'collision': 0, 'time_limit': 0, 'steps': 1,
           'simulation_duration_s': .2, 'return': 2., 'author_history_total_reward': 7.,
           'mean_abs_measured_jerk': .3, 'mean_abs_jerk': .4,
           'initial_traffic_sha256': 'a'*64, 'policy_sha256': 'b'*64}
    assert native_row(j, row, {})['native_score'] == 2.
    assert native_row(j, row, {})['native_mean_abs_jerk'] == .3
    j = jobs(AUDIT)[1]
    assert native_row(j, row, {})['native_score'] == 7.
    assert native_row(j, row, {})['native_mean_abs_jerk'] == .4


def test_cross_protocol_hash_difference_allowed_no_reward_delta():
    a = aggregate(PROTOCOL, rows(PROTOCOL))
    assert a['new_episodes'] == 80 and a['verified_historical_episodes'] == 200
    assert a['matrix'][AUTHOR][GYM]['independent_policy_snapshots'] == 1
    assert a['matrix'][B0][GYM]['independent_policy_snapshots'] == 3
    assert a['matrix'][AUTHOR][GYM]['native_score_name'] != a['matrix'][AUTHOR][AUTHOR_LOOP]['native_score_name']
    for report in a['protocol_sensitivity'].values():
        assert report['reward_delta_computed'] is False
        assert set(report['mean_native_event_delta']) == {'arrival', 'reported_collision', 'time_limit'}
    assert a['method_effect_gate'] is None and a['budget_matched'] is False


def test_native_failures_complete_no_success_conditioned_parking_benefit():
    r = rows()
    for row in r:
        if row['model_id'] == B0 and row['run_seed'] == 2:
            row.update(arrival=0, time_limit=1, simulation_duration_s=100., native_score=-10., native_mean_abs_jerk=0.)
    a = aggregate(AUDIT, r)
    s = a['matrix'][B0][GYM]['per_run_seed']['2']
    assert s['successful_episode_time_s'] == {'mean': None, 'count': 0}
    assert a['status'] == 'complete' and a['method_effect_gate'] is None


@pytest.mark.parametrize('fault', ['missing', 'duplicate', 'identity', 'scene', 'weights',
    'predictor', 'missing_predictor', 'score', 'missing_score', 'nonfinite', 'events', 'steps', 'source_kind', 'hash'])
def test_bad_results_cannot_be_reused_or_aggregated(fault):
    r = rows()
    if fault == 'missing': r.pop()
    elif fault == 'duplicate': r.append(deepcopy(r[0]))
    elif fault == 'identity': r[0]['model_id'] = B0
    elif fault == 'scene': r[0]['initial_traffic_sha256'] = 'e'*64
    elif fault == 'weights': r[0]['policy_sha256'] = 'e'*64
    elif fault == 'predictor': r[0]['predictor_sha256'] = 'e'*64
    elif fault == 'missing_predictor': r[-1]['predictor_sha256'] = None
    elif fault == 'score': r[0]['native_score_name'] = 'return'
    elif fault == 'missing_score': r[0]['native_score'] = None
    elif fault == 'nonfinite': r[0]['native_score'] = float('nan')
    elif fault == 'events': r[0]['reported_collision'] = 1
    elif fault == 'steps': r[0]['steps'] = 501
    elif fault == 'source_kind': r[0]['evidence']['kind'] = 'historical'
    else: r[0]['policy_sha256'] = 'z'*64
    with pytest.raises(ValueError): aggregate(AUDIT, r)


def env():
    return SimpleNamespace(feature_arm='baseline', device='cpu',
        state_space=gym.spaces.Box(-np.inf, np.inf, (20,), np.float32),
        action_space=gym.spaces.Box(-5, 5, (1,), np.float32))


def state(done=False):
    return State(torch.linspace(-1, 1, 20).view(1, 20), torch.tensor([0 if done else 1], dtype=torch.uint8), [{}])


def test_one_query_per_actual_step_and_terminal_reset_no_replay_update():
    checked = preset_factory('cpu', .0002)(env(), DummyWriter())
    before = runner.smoke.tensor_hash(checked.agent.policy.model.state_dict())
    query = runner.FrozenQuery(checked.body, checked.agent.policy.model)
    for _ in range(3): query(state(), 0.)
    query(state(True), 0.)
    assert query.complete(3)
    assert [r['time_feature'] for r in query.records] == [0., .001, .002]
    assert float(checked.body.timestep.item()) == 0.
    assert not len(checked.agent.replay_buffer) and not checked.agent.policy._optimizer.state
    assert runner.smoke.tensor_hash(checked.agent.policy.model.state_dict()) == before


def test_double_query_or_wrong_terminal_is_rejected():
    checked = preset_factory('cpu', .0002)(env(), DummyWriter())
    query = runner.FrozenQuery(checked.body, checked.agent.policy.model)
    query(state(), 0.)
    with pytest.raises(ValueError): query.complete(1)
    checked.body.eval(state(), 0.)
    with pytest.raises(ValueError, match='outside actual'): query(state(), 0.)


def test_recorded_network_parity_is_strict():
    checked = preset_factory('cpu', .0002)(env(), DummyWriter())
    query = runner.FrozenQuery(checked.body, lambda s: torch.ones(1, 1)*5)
    with pytest.raises(ValueError, match='parity'): query(state(), 0.)


def test_original_author_modules_load_whole_no_environment_construction():
    # Offline shape/state identity audit only; fake env has no SUMO methods.
    body, policy, critic = runner.author_policy(env())
    query = runner.FrozenQuery(body, policy)
    query(state(), 0.); query(state(True), 0.)
    assert query.complete(1)
    assert runner.external.assets() == runner.external.ASSETS
    assert all(torch.isfinite(v).all() for v in critic.state_dict().values())


def test_legacy_observation_layout_cannot_be_reordered(monkeypatch):
    monkeypatch.setitem(sys.modules, 'prediction', SimpleNamespace(HighwayState=SimpleNamespace(from_sumo=lambda: None)))
    monkeypatch.setitem(sys.modules, 'dqn', SimpleNamespace(get_state_vector_from_base_state=lambda s:np.linspace(-1, 1, 20, dtype=np.float32)))
    runner.state_vector_parity(State(torch.from_numpy(np.linspace(-1, 1, 20, dtype=np.float32)[None]), state().mask))
    with pytest.raises(ValueError): runner.state_vector_parity(State(torch.zeros(1, 20), state().mask))


def test_checkpoint_b0_controller_never_calls_gym_reset_step(monkeypatch):
    c = object.__new__(runner.ProjectBaselineController); commands = []
    c.env = SimpleNamespace(_make_state=lambda obs, done:(obs, done))
    c.query = lambda state, reward: torch.tensor([[2.5]])
    monkeypatch.setitem(sys.modules, 'dqn', SimpleNamespace(get_state_vector_from_base_state=lambda s:np.zeros(20)))
    monkeypatch.setitem(sys.modules, 'control', SimpleNamespace(set_ego_jerk=lambda a:commands.append(a) or 3.))
    assert c(None) == 3. and commands == [2.5]
    c.end(None); assert commands == [2.5]


def test_resume_preserves_incomplete_job_and_does_not_launch_child(tmp_path, monkeypatch):
    from prediction_rl.data.collection_store import write_once
    job = jobs(AUDIT)[0]; q = {'jobs': [job], 'protocol': AUDIT}
    out = tmp_path/'evaluate'/job['id']; out.mkdir(parents=True)
    write_once(out/'started.json', {'status': 'incomplete'})
    monkeypatch.setattr(runner, 'ROOT', tmp_path)
    monkeypatch.setattr(runner, 'load', lambda *a:(tmp_path/'request.json', q, None, None))
    monkeypatch.setattr(runner.shared, 'metadata', lambda:{'git_commit': 'a'*40})
    monkeypatch.setattr(runner.subprocess, 'run', lambda *a,**k:pytest.fail('Incomplete job must not be retried'))
    with pytest.raises(ValueError, match='Incomplete/failed'):
        runner.execute(tmp_path/'request.json', 'frozen', resume=True)
    assert (out/'started.json').exists() and not (out/'failure.json').exists()
    report = next((tmp_path/'invocations').glob('*/report.json'))
    assert json.loads(report.read_text())['status'] == 'failed'


def test_new_historical_verifier_accepts_additions_but_not_changed_old_code(monkeypatch):
    calls = []
    q = {'source_hashes': {'old.py': 'old'}, 'git_commit': 'a'*40,
         'environment': {}, 'runtime_identity': {}, 'author_assets': {}, 'upstream_config_hashes': {}}
    monkeypatch.setattr(runner, 'verify_historical_inputs', lambda *a:calls.append(a))
    monkeypatch.setattr(runner.shared, 'source_hashes', lambda:{'old.py': 'old', 'new.py': 'new'})
    monkeypatch.setattr(runner.shared, 'env', lambda:{})
    monkeypatch.setattr(runner.p7, 'runtime_identity', lambda:{})
    monkeypatch.setattr(runner.external, 'assets', lambda:{})
    monkeypatch.setattr(runner.external, 'configurations', lambda:{})
    runner.unchanged_historical_source(q); assert calls
    monkeypatch.setattr(runner.shared, 'source_hashes', lambda:{'old.py': 'changed', 'new.py': 'new'})
    with pytest.raises(ValueError, match='Historical source'):
        runner.unchanged_historical_source(q)
