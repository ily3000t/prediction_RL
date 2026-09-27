import json
from copy import deepcopy
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'tools'))
from prediction_rl.data.geometric_roots import scan_lane_roots, target_position, validate_targets
from diagnose_branches import load_plan

TARGET = {'id': 'approach', 'lane_id': 'ramp_0', 'reference': 'before_end_m', 'value': 20.0}


class FakeReference:
    action_space = SimpleNamespace(dtype=np.float32)

    def __init__(self, positions, done_at=None):
        self.positions, self.done_at = positions, done_at
        self.index = 0
        self.actions = []
        self.read_indices = []

    def progress(self):
        self.read_indices.append(self.index)
        lane, position, length = self.positions[self.index]
        return {'lane_id': lane, 'lane_position_m': position,
                'lane_length_m': length, 'traffic': {'index': self.index}}

    def step(self, action):
        self.actions.append(action.copy())
        self.index += 1
        return None, 0, self.index == self.done_at, {}


def test_first_threshold_crossing_no_future_result_selection():
    env = FakeReference([('ramp_0', 70, 100), ('ramp_0', 79, 100), ('ramp_0', 83, 100)])
    result = scan_lane_roots(env, [TARGET], 2, 0.0, env.progress)
    root = result['targets'][0]
    assert root['prefix_steps'] == 2
    assert root['overshoot_m'] == 3
    assert root['target_position_m'] == 80
    assert len(env.actions) == 2
    assert all(float(action[0]) == 0.0 for action in env.actions)


def test_target_requires_correct_lane():
    env = FakeReference([('other', 90, 100), ('ramp_0', 81, 100)])
    result = scan_lane_roots(env, [TARGET], 1, 0.0, env.progress)
    assert result['targets'][0]['prefix_steps'] == 1


def test_terminal_state_is_never_selected_or_replaced():
    env = FakeReference([('ramp_0', 70, 100), ('ramp_0', 79, 100), ('ramp_0', 83, 100)], done_at=2)
    result = scan_lane_roots(env, [TARGET], 2, 0.0, env.progress)
    assert result['targets'][0]['prefix_steps'] is None
    assert result['stop_reason'] == 'episode_terminated_before_remaining_targets'
    assert env.read_indices == [0, 1]


def test_budget_exhaustion_keeps_missing_target():
    env = FakeReference([('ramp_0', 0, 100), ('ramp_0', 1, 100)])
    result = scan_lane_roots(env, [TARGET], 1, 0.0, env.progress)
    assert result['targets'][0]['prefix_steps'] is None
    assert result['stop_reason'] == 'discovery_budget_exhausted'


def test_missed_lane_does_not_substitute_other_root():
    env = FakeReference([('ramp_0', 79, 100), ('downstream', 10, 100)])
    result = scan_lane_roots(env, [TARGET], 1, 0.0, env.progress)
    assert result['targets'][0]['prefix_steps'] is None


@pytest.mark.parametrize('reference,value,expected', [
    ('before_end_m', 20, 80), ('fraction', .5, 50), ('from_start_m', 5, 5)])
def test_target_geometry(reference, value, expected):
    assert target_position({**TARGET, 'reference': reference, 'value': value}, 100) == expected


def test_geometric_target_outside_lane_is_error():
    with pytest.raises(ValueError):
        target_position(TARGET, 10)


@pytest.mark.parametrize('change', [
    {'reference': 'best_response'}, {'value': float('nan')}, {'value': True},
    {'reference': 'fraction', 'value': 1.1}, {'reward_filter': 1},
])
def test_invalid_targets_rejected(change):
    with pytest.raises(ValueError):
        validate_targets([{**TARGET, **change}])


def test_duplicate_lane_targets_rejected():
    with pytest.raises(ValueError):
        validate_targets([TARGET, {**TARGET, 'id': 'another'}])


def test_v1_conditions_preserved_in_v2():
    old = load_plan(ROOT / 'configs/development/p03_mechanism_v1.json')
    new = load_plan(ROOT / 'configs/development/p03_mechanism_v2_merge_region.json')
    for key in set(old) & set(new) - {'schema_version'}:
        assert old[key] == new[key]
    assert len(new['root_targets']) == 3
    assert new['native_probe_enabled'] is False


@pytest.mark.parametrize('change', [
    {'discovery_max_steps': 251}, {'native_probe_enabled': True},
    {'root_prefix_steps': [0, 30]}, {'discovery_max_steps': True},
])
def test_invalid_v2_plan_rejected(tmp_path, change):
    plan = json.loads((ROOT / 'configs/development/p03_mechanism_v2_merge_region.json').read_text())
    plan.update(change)
    path = tmp_path / 'plan.json'
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError):
        load_plan(path)
