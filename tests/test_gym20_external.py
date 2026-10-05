from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'tools')]
import compare_gym20_external as runner
from prediction_rl.envs.gym20_speed_controller import speed_raw_class
from prediction_rl.evaluation.gym20_external import (
    PROTOCOL, AUDIT, AUTHORS, PROJECT, jobs, project_job, aggregate, validate_protocol)
from prediction_rl.evaluation.matched_learning_curve import METRICS
from prediction_rl.evaluation.external_baselines import ASSETS


def rows(p=AUDIT):
    authors = []
    project = []
    for j in jobs(p) + [project_job(a, s, v) for a in PROJECT
            for s in p['project_run_seeds'] for v in p['simulator_seeds']]:
        r = {**j, **{m: 0. for m in METRICS}, 'steps': 2,
             'natural_arrival': 1, 'ego_collision': 0, 'time_limit': 0,
             'initial_traffic_sha256': str(j['simulator_seed']).zfill(64),
             'native_event_agrees': True, 'no_policy_updates': True, 'adapter_parity': True}
        if j['arm'] in AUTHORS:
            r['policy_sha256'] = None if j['arm'] == 'author_st' else ASSETS['policy.pt']
            r['action_saturation_fraction'] = None if j['arm'] != 'author_ddpg' else 0.
            authors.append(r)
        else:
            r['actual_training_steps'] = 60010
            project.append(r)
    return authors, project


def test_frozen_config_and_reuse_roster():
    assert json.loads((ROOT/'configs/development/p07l_gym20_external_v1.json').read_text()) == PROTOCOL
    assert len(jobs(PROTOCOL)) == 80 and len(jobs(AUDIT)) == 4
    assert {j['run_seed'] for j in jobs(PROTOCOL)} == {None}
    a = aggregate(PROTOCOL, *rows(PROTOCOL))
    assert a['new_author_episodes'] == 80 and a['reused_project_episodes'] == 240
    assert a['unique_traffic_scenarios'] == 20 and len(a['comparisons']) == 16
    assert a['matrix']['author_st']['independent_training_replicates'] == 0
    assert a['matrix']['author_ddpg']['independent_training_replicates'] == 1
    assert a['matrix']['conditional']['independent_training_replicates'] == 3
    assert a['author50_results_mixed'] is False and a['training_performed'] is False


@pytest.mark.parametrize('field,value', [('warmup_s', 50), ('max_control_calls', 501),
    ('project_run_seeds', [0]), ('checkpoint_frames', 40000), ('training_permitted', True),
    ('budget_matched', True), ('author50_results_mixed', True), ('unknown', 1)])
def test_no_silent_design_amendment(field, value):
    p = deepcopy(PROTOCOL); p[field] = value
    with pytest.raises(ValueError): validate_protocol(p)


def test_negative_result_is_not_an_engineering_failure_or_checkpoint_selection():
    authors, project = rows()
    for r in project:
        if r['arm'] == 'conditional':
            r.update(natural_arrival=0, ego_collision=1, **{'return': -10.})
    a = aggregate(AUDIT, authors, project)
    assert a['status'] == 'complete' and a['method_effect_gate'] is None
    c = a['comparisons']['author_rl_mpc_safety__vs__conditional']
    assert c['equal_run_seed_mean_delta']['ego_collision'] == 1.
    assert c['equal_run_seed_mean_delta']['duration_s_both_natural_arrival'] is None
    assert len(c['paired_rows']) == 3  # Same one scene, not three author trainings.


@pytest.mark.parametrize('fault', ['missing', 'duplicate', 'scene', 'nonfinite', 'events', 'parity', 'updates', 'native', 'weights', 'snapshot', 'saturation'])
def test_invalid_reuse_or_external_cells_rejected(fault):
    authors, project = rows()
    if fault == 'missing': authors.pop()
    elif fault == 'duplicate': project.append(deepcopy(project[0]))
    elif fault == 'scene': project[0]['initial_traffic_sha256'] = 'f'*64
    elif fault == 'nonfinite': authors[0]['return'] = float('nan')
    elif fault == 'events': authors[0]['time_limit'] = 1
    elif fault == 'parity': authors[0]['adapter_parity'] = False
    elif fault == 'updates': project[0]['no_policy_updates'] = False
    elif fault == 'native': authors[0]['native_event_agrees'] = False
    elif fault == 'weights': authors[0]['policy_sha256'] = 'f'*64
    elif fault == 'snapshot': project[0]['actual_training_steps'] = 40000
    else: authors[1]['action_saturation_fraction'] = 0.
    with pytest.raises(ValueError): aggregate(AUDIT, authors, project)


def test_native_speed_is_never_reintegrated_clipped_or_followed_by_an_extra_step(monkeypatch):
    calls = []
    highway = object()
    monkeypatch.setitem(sys.modules, 'prediction', SimpleNamespace(
        HighwayState=SimpleNamespace(from_sumo=lambda: highway)))
    settings = SimpleNamespace(REWARD_FUNCTION='Slotted Jerk', INVALID_ACTION_PENALTY=0, TICK_LENGTH=.2)
    class Base:
        def step(self, token):
            self._do_action(token)
            calls.append('one_original_step')
    raw = speed_raw_class(Base, settings)()
    def controller(state):
        assert state is highway
        calls.append('native_setSpeed')
        return 35.  # Even an out-of-range command is not silently repaired.
    raw.controller = controller
    raw.step(None)
    assert calls == ['native_setSpeed', 'one_original_step']
    assert raw.commanded_speed == 35. and raw.invalid_action_reward == 0.
    with pytest.raises(ValueError): raw.step(1.)
    raw.controller = lambda state: float('nan')
    with pytest.raises(ValueError): raw.step(None)


@pytest.mark.parametrize('field,value', [('REWARD_FUNCTION', 'ST'), ('INVALID_ACTION_PENALTY', -1), ('TICK_LENGTH', .1)])
def test_unsupported_speed_reward_contract_is_rejected(field, value):
    settings = SimpleNamespace(REWARD_FUNCTION='Slotted Jerk', INVALID_ACTION_PENALTY=0, TICK_LENGTH=.2)
    setattr(settings, field, value)
    with pytest.raises(ValueError): speed_raw_class(object, settings)


def test_only_recording_fields_may_differ_in_adapter_parity():
    x = {'actions': [1], 'query_counts': [5], 'takeovers': [False],
         'outcome': dict(zip(('return', 'steps', 'arrival', 'collision', 'time_limit', 'initial_traffic_sha256'),
                             [1., 1, 1, 0, 0, 'a'*64])), 'raw_events': None}
    y = deepcopy(x); y['raw_events'] = {'recorded': True}
    assert runner.assert_adapter_parity(x, y) is True
    y['query_counts'] = [1]
    with pytest.raises(ValueError): runner.assert_adapter_parity(x, y)
