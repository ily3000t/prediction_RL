import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'tools'))
from prediction_rl.data.branching import ReplayBrancher, ReplayRoot, fingerprint, plain, response_summary
from diagnose_branches import load_plan, environment_metadata


def frame(ego_x, neighbor_x=10.0, neighbor_speed=7.0, time=1.0, with_neighbor=True):
    vehicles = {'ego': {'position': [ego_x, 0], 'speed': 1.0}}
    if with_neighbor:
        vehicles['n'] = {'position': [neighbor_x, 0], 'speed': neighbor_speed}
    return {'info': {'execution_audit': {'after': {'simulation_time_s': time, 'vehicles': vehicles}}}}


def test_relative_gap_change_is_not_neighbor_response():
    pair = response_summary({'a': [frame(0)], 'b': [frame(5)]}, {'n'}, .05, .1)[0]
    assert pair['max_ego_position_delta_m'] == 5
    assert pair['max_neighbor_position_delta_m'] == 0
    assert not pair['neighbor_response_detected']


def test_absolute_neighbor_response_detected():
    pair = response_summary({'a': [frame(0)], 'b': [frame(0, neighbor_speed=6)]}, {'n'}, .05, .1)[0]
    assert pair['neighbor_response_detected']
    assert pair['max_neighbor_speed_delta_mps'] == 1


def test_missing_actor_is_censored_not_zero_filled():
    pair = response_summary({'a': [frame(0)], 'b': [frame(0, with_neighbor=False)]}, {'n'}, .05, .1)[0]
    assert pair['paired_neighbor_observations'] == 0
    assert pair['root_actor_membership_mismatches'] == 1
    assert not pair['neighbor_response_detected']


def test_time_alignment_required():
    with pytest.raises(AssertionError):
        response_summary({'a': [frame(0)], 'b': [frame(0, time=2)]}, {'n'}, .05, .1)


def test_newly_arrived_actors_do_not_establish_root_response():
    pair = response_summary({'a': [frame(0)], 'b': [frame(0, neighbor_speed=6)]}, set(), .05, .1)[0]
    assert not pair['neighbor_response_detected']


def test_snapshot_serialization_is_stable_and_type_aware():
    assert fingerprint({'x': np.array([1.0])}) == fingerprint({'x': np.array([1.0])})
    assert fingerprint(np.array([1], dtype=np.float32)) != fingerprint(np.array([1], dtype=np.float64))
    with pytest.raises(ValueError):
        fingerprint(float('nan'))


def test_foreign_root_rejected_before_touching_sumo():
    manager = object.__new__(ReplayBrancher)
    manager.owner = object()
    root = ReplayRoot((), 'x', 'y', {}, object())
    with pytest.raises(ValueError):
        manager.restore(root)


def test_frozen_plan_valid():
    plan = load_plan(ROOT / 'configs/development/p03_mechanism_v1.json')
    assert plan['intervention_steps'] == 1
    assert plan['simulator_seeds'] == [0, 1, 100]


def test_environment_audit_fields(monkeypatch):
    import diagnose_branches as diagnostic
    monkeypatch.setattr(diagnostic.shutil, 'which', lambda binary: '/sumo')
    monkeypatch.setattr(diagnostic.importlib.metadata, 'version', lambda name: 'test')
    monkeypatch.setattr(diagnostic.subprocess, 'check_output', lambda *a, **k: 'SUMO test\n')
    metadata = environment_metadata()
    assert metadata['sumo_version'] == 'SUMO test'
    assert metadata['packages']['torch'] == 'test'


@pytest.mark.parametrize('change', [
    {'typo': 1}, {'intervention_steps': 2}, {'horizon_steps': 10000},
    {'root_prefix_steps': [0, 30, 50]}, {'simulator_seeds': [0, 0]},
    {'response_speed_threshold_mps': 0}, {'upstream_config': '../other.json'},
])
def test_invalid_or_unbounded_plans_rejected(tmp_path, change):
    plan = json.loads((ROOT / 'configs/development/p03_mechanism_v1.json').read_text())
    plan.update(change)
    path = tmp_path / 'plan.json'
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError):
        load_plan(path)
