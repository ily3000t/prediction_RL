import hashlib
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'tools'))
from prediction_rl.data.reference_policy import verify_checkpoint
from prediction_rl.data.geometric_roots import scan_lane_roots
from diagnose_branches import load_plan
from test_geometric_roots import FakeReference, TARGET


def test_v3_preserves_shared_conditions():
    old = load_plan(ROOT / 'configs/development/p03_mechanism_v2_merge_region.json')
    new = load_plan(ROOT / 'configs/development/p03_mechanism_v3_author_policy.json')
    for key in set(old) & set(new) - {'schema_version'}:
        assert old[key] == new[key]
    assert 'prefix_jerk' not in new


@pytest.mark.parametrize('change', [
    {'adapter': 'other'}, {'sha256': 'bad'}, {'checkpoint': '../policy.pt'},
    {'checkpoint': 'C:/policy.pt'}, {'checkpoint': 'q.pt'}, {'noise': .1},
])
def test_policy_contract_rejects_unknown_or_unpinned(tmp_path, change):
    plan = json.loads((ROOT / 'configs/development/p03_mechanism_v3_author_policy.json').read_text())
    plan['reference_policy'].update(change)
    path = tmp_path / 'plan.json'
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError):
        load_plan(path)


def test_hash_mismatch_rejected_without_deserialization(tmp_path):
    path = tmp_path / 'policy.pt'
    path.write_bytes(b'not a pickle')
    with pytest.raises(ValueError, match='SHA256'):
        verify_checkpoint(path, '0' * 64)
    assert verify_checkpoint(path, hashlib.sha256(path.read_bytes()).hexdigest()) == path


def test_discovery_records_policy_prefix_without_probing():
    env = FakeReference([('ramp_0', 70, 100), ('ramp_0', 79, 100), ('ramp_0', 83, 100)])
    calls = []
    def policy(obs, step):
        calls.append((obs, step))
        return [-2.5, 1.25][step]
    result = scan_lane_roots(env, [TARGET], 2, None, env.progress, policy, [1.0])
    assert result['targets'][0]['prefix_actions'] == [-2.5, 1.25]
    assert result['reference_actions'] == [-2.5, 1.25]
    assert calls == [([1.0], 0), (None, 1)]
    assert [float(x[0]) for x in env.actions] == [-2.5, 1.25]


def test_terminal_policy_path_keeps_missing_targets():
    env = FakeReference([('ramp_0', 70, 100), ('ramp_0', 83, 100)], done_at=1)
    result = scan_lane_roots(env, [TARGET], 2, None, env.progress, lambda obs, step: .5, [])
    assert result['targets'][0]['prefix_steps'] is None
    assert result['reference_actions'] == [.5]
