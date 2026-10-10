from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'tools')]
from prediction_rl.curriculum.task_characterization import (PROTOCOL, BudgetLedger, scan_checkpoints,
    validate_protocol, validate_actual_budget, root_evidence)
from prediction_rl.curriculum.task_collector import CountedEnv, CountedObserver
import prediction_rl.curriculum.task_collector as engine
from prediction_rl.data.merge_geometry import MergeGeometry
from test_actor_adapter import actor, frame
from test_curriculum_interaction import example


@pytest.fixture
def design():
    return json.loads((ROOT/'configs/development/ic02c_task_design_v1.json').read_text())


def test_new_authorization_does_not_mutate_offline_proposal(design):
    protocol = json.loads((ROOT/'configs/development/ic02d_task_engineering_v1.json').read_text())
    assert validate_protocol(protocol, design) == PROTOCOL
    assert design['runtime_authorized'] is False
    assert protocol['predictor_inference_calls'] == protocol['ddpg_training_steps'] == 0


@pytest.mark.parametrize('field,value', [('simulator_seeds', [11000,11001,11002]),
    ('workers', 2), ('tasks_max', 32), ('raw_ticks_max', 99999), ('automatic_retry', True),
    ('training_ready', True), ('task_classifier', 'old_root_levels'), ('unknown', 0)])
def test_engineering_scope_cannot_expand_or_promote(design, field, value):
    protocol = deepcopy(PROTOCOL); protocol[field] = value
    with pytest.raises(ValueError): validate_protocol(protocol, design)


class Env:
    action_space = SimpleNamespace(dtype=np.float32)
    def __init__(self, positions, terminal_at=None):
        self.positions, self.index, self.terminal_at, self.actions = positions, 0, terminal_at, []
    def read(self):
        lane, position = self.positions[min(self.index,len(self.positions)-1)]
        return frame(20.2+.2*self.index, {'ego': actor(lane, position, speed=10)})
    def step(self, action):
        self.actions.append(float(action[0])); self.index += 1
        return np.zeros(20, np.float32), 0., self.index == self.terminal_at, {}


def scan(env, design):
    calls = []
    def policy(obs, step): calls.append(step); return 2.5
    result = scan_checkpoints(env, design, MergeGeometry(ROOT/'RL-MPC-LaneMerging-master/merge.net.xml'),
                              policy, np.zeros(20,np.float32), env.read)
    return result, calls


def test_same_lane_reset_and_early_and_approach_are_independent(design):
    env = Env([('ramp_0',40), ('ramp_0',80), ('ramp_0',182)])
    result, calls = scan(env, design)
    assert [r['prefix_steps'] for r in result['targets']] == [0,1,2]
    assert result['targets'][0]['prefix_actions'] == []
    assert result['targets'][2]['prefix_actions'] == [2.5,2.5]
    assert calls == [0,1] and result['stop_reason'] == 'all_checkpoints_reached'


def test_terminal_crossing_is_not_a_nonterminal_root_and_not_replaced(design):
    env = Env([('ramp_0',40), ('ramp_0',182)], terminal_at=1)
    result, calls = scan(env, design)
    assert result['targets'][0]['prefix_steps'] == 0
    assert all(r['prefix_steps'] is None for r in result['targets'][1:])
    assert calls == [0] and len(env.actions) == 1


def test_unreachable_checkpoints_keep_cap_nonterminal_and_original_budget(design):
    env = Env([('ramp_0',40)])
    result, calls = scan(env, design)
    assert len(calls) == 128 and len(env.actions) == 128
    assert all(r['prefix_steps'] is None for r in result['targets'][1:])
    assert result['stop_reason'] == 'discovery_budget_exhausted' and result['cap_is_not_terminal']


def test_skip_to_other_lane_does_not_forge_a_ramp_checkpoint(design):
    env = Env([('ramp_0',40), (':mergenode_1_0',20)], terminal_at=2)
    result, calls = scan(env, design)
    assert all(r['prefix_steps'] is None for r in result['targets'][1:])


@pytest.mark.parametrize('kind', ['session', 'control', 'tick'])
def test_caps_reject_before_any_additional_external_action(design, kind):
    ledger = BudgetLedger(design)
    if kind == 'session':
        for _ in range(38): ledger.session()
        with pytest.raises(ValueError): ledger.session()
        assert ledger.sessions == 38
    elif kind == 'control':
        for _ in range(256): ledger.control('reference_discovery')
        with pytest.raises(ValueError): ledger.control('reference_discovery')
        assert ledger.control_attempts == 256
    else:
        for _ in range(7954): ledger.tick()
        with pytest.raises(ValueError): ledger.tick()
        assert ledger.raw_tick_attempts == 7954


def test_failed_native_step_is_reserved_not_completed_or_free_retry(design):
    ledger = BudgetLedger(design)
    manager = SimpleNamespace(ledger=ledger, phase='candidate_future', session_controls={ 'candidate_future':0})
    class Failure:
        def step(self, action): raise RuntimeError('native error')
    env = CountedEnv(Failure(), manager)
    with pytest.raises(RuntimeError): env.step([0.])
    assert ledger.control_attempts == 1 and ledger.completed_controls == 0


def successful_budget():
    return {'fresh_sumo_sessions': 1, 'control_attempts': 3, 'completed_control_steps': 3,
        'raw_tick_attempts': 104, 'completed_raw_ticks': 104,
        'control_steps_by_phase': {'reference_discovery':3}}


def test_successful_budget_is_validated_not_a_receipt_boolean(design):
    assert validate_actual_budget(successful_budget(), design)


@pytest.mark.parametrize('fault', ['unknown', 'bool', 'incomplete', 'phase_over', 'raw_missing', 'raw_extra', 'sessions_over'])
def test_invalid_success_budget_rejected(design, fault):
    b = successful_budget()
    if fault == 'unknown': b['x'] = 1
    elif fault == 'bool': b['fresh_sumo_sessions'] = True
    elif fault == 'incomplete': b['completed_control_steps'] = 2
    elif fault == 'phase_over': b['control_steps_by_phase'] = {'candidate_future': 751}
    elif fault == 'raw_missing': b['raw_tick_attempts'] = b['completed_raw_ticks'] = 103
    elif fault == 'raw_extra': b['raw_tick_attempts'] = b['completed_raw_ticks'] = 106
    elif fault == 'sessions_over': b['fresh_sumo_sessions'] = 39
    with pytest.raises(ValueError): validate_actual_budget(b, design)


def test_root_uses_frozen_scales_with_no_layer_reclassification():
    g = MergeGeometry(ROOT/'RL-MPC-LaneMerging-master/merge.net.xml')
    c = json.loads((ROOT/'configs/development/ic01_train_interaction_v1.json').read_text())
    h,p,b = example(g,c,responsive=True)
    row = root_evidence(h,p,b,deepcopy(b),g,c,{'dv':2.,'ds':3.},
                        {'task_id':'t11000','target':'reset','simulator_seed':11000})
    assert row['I_with_frozen_IC01_scales'] > 0 and row['repeat_noise_with_frozen_IC01_scales'] == 0
    assert row['task_level'] is None and not row['classification_applied']
    assert row['task_mapping']['root_is_original_reset']
    assert row['root_context']['coverage']['history_frames'] == 1


@pytest.mark.parametrize('postflight_fails', [False, True])
def test_engine_failure_always_records_postflight_without_sealing(tmp_path, monkeypatch, design, postflight_fails):
    from contextlib import contextmanager
    @contextmanager
    def failed_session(*args):
        raise RuntimeError('session entry failed before SUMO'); yield
    monkeypatch.setattr(engine, 'upstream_session', failed_session)
    monkeypatch.setattr(engine, 'rng_fingerprint', lambda: 'unchanged')
    def postflight():
        if postflight_fails: raise ValueError('input postflight failure')
    with pytest.raises(RuntimeError, match='session entry failed'):
        engine.collect_task(ROOT,ROOT,design,{'id':'t11000','simulator_seed':11000},tmp_path,{}, {}, {}, postflight)
    post = json.loads((tmp_path/'postflight.json').read_text())
    assert post['worker_session_rng_restored'] and post['budget'] is None
    assert post['immutable_input_postflight_passed'] == (not postflight_fails)
    assert (tmp_path/'failure.json').is_file() and not (tmp_path/'complete.json').exists()
