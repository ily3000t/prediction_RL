from copy import deepcopy
import json
from pathlib import Path
import random
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'tools')]
import diagnose_ddpg_stability as runner
from prediction_rl.evaluation.ddpg_stability import (
    PROTOCOL, AUDIT, METRICS, FrozenGaussian, validate_protocol, roster, aggregate,
    scalar_summary, training_windows, control_summary)
from prediction_rl.evaluation.exploration import PROTOCOL as P7, jobs


def parent():
    return {'protocol': P7, 'evaluation_jobs': jobs(P7, 'evaluate')}


def test_frozen_design_and_roster():
    assert json.loads((ROOT/'configs/development/p07i_ddpg_stability_v1.json').read_text()) == PROTOCOL
    full = roster(PROTOCOL, parent()); audit = roster(AUDIT, parent())
    assert len(full) == 144 and len(audit) == 8
    assert sum(j['condition'] == 'deterministic' for j in full) == 36
    assert sum(j['condition'] == 'upstream_noise' for j in full) == 108
    assert {j['run_seed'] for j in full} == {0, 1, 2}
    for key, value in [('run_seeds', [0, 1]), ('noise_repeats', [0]), ('simulator_seeds', [200]), ('workers', 8)]:
        changed = deepcopy(PROTOCOL); changed[key] = value
        with pytest.raises(ValueError): validate_protocol(changed)
    changed = parent(); changed['evaluation_jobs'] = changed['evaluation_jobs'][:-1]
    with pytest.raises(ValueError): roster(PROTOCOL, changed)


def test_noise_exact_clip_replay_common_prefix_and_rng_isolation():
    a = FrozenGaussian(-5., 5., .1, 200, 0); b = FrozenGaussian(-5., 5., .1, 200, 0)
    global_torch = torch.get_rng_state().clone(); py = random.getstate(); np_state = np.random.get_state()
    observed = []
    for jerk in [5., -5., 0., 1.] * 25:
        action = torch.tensor([[jerk]], dtype=torch.float32)
        out, fields = a.apply(action); repeated, again = b.apply(action)
        assert torch.equal(out, repeated) and fields == again
        assert out.item() == float(torch.max(torch.min(action+torch.tensor([[fields['noise']]]), torch.tensor(5.)), torch.tensor(-5.)).item())
        observed.append(fields['noise'])
    assert a.draws == b.draws == 100 and a.std == .5
    assert torch.equal(global_torch, torch.get_rng_state()) and py == random.getstate()
    later = np.random.get_state(); assert np_state[0] == later[0] and np.array_equal(np_state[1], later[1]) and np_state[2:] == later[2:]
    c = FrozenGaussian(-5., 5., .1, 200, 0)
    # Different policy actions share exactly the same noise prefix.
    assert [c.apply(torch.zeros((1, 1)))[1]['noise'] for _ in range(100)] == observed
    assert FrozenGaussian(-5., 5., .1, 200, 1).seed != a.seed
    assert FrozenGaussian(-5., 5., .1, 210, 0).seed != a.seed


@pytest.mark.parametrize('value', [torch.tensor([[float('nan')]]), torch.tensor([[6.]]), torch.zeros(1), torch.zeros((1, 1), dtype=torch.float64)])
def test_invalid_action_not_repaired(value):
    with pytest.raises(ValueError): FrozenGaussian(-5., 5., .1, 200, 0).apply(value)


def test_windows_keep_empty_intervals_and_signed_actor_loss():
    rows = [{'step': 5000, 'value': -2.}, {'step': 5001, 'value': -3.}]
    w = training_windows(rows, [0, 5000, 10000])
    assert w['0_5000']['count'] == 0 and w['0_5000']['mean'] is None
    assert w['5000_10000']['mean'] == -2.5
    with pytest.raises(ValueError): scalar_summary([{'step': 1, 'value': float('inf')}])
    with pytest.raises(ValueError): training_windows(list(reversed(rows)), [0, 10000])


def test_control_suffix_is_descriptive_and_preserves_low_speed_denominator():
    rows = [{'step': i, 'speed_before_mps': .0 if i != 1 else 2., 'actor_jerk': -1.,
             'requested_jerk': 1., 'noise': 2., 'clipped': False, 'reward': 1.,
             'q_actor': 2., 'q_requested': 3., 'done': i == 2} for i in range(3)]
    r = control_summary(rows, .5, .1)
    assert r['discounted_executed_return_from_start'] == 1.75
    assert r['low_speed_fraction'] == 2/3 and r['longest_low_speed_sample_grid_s'] == .2
    assert r['low_speed_actor_negative_jerk_fraction'] == 1.
    assert r['low_speed_requested_negative_jerk_fraction'] == 0.
    assert r['suffix_error_is_descriptive_not_bellman_loss_or_calibration'] is True
    for row in rows: row['speed_before_mps'] = 2.
    assert control_summary(rows, .5, .1)['low_speed_requested_negative_jerk_fraction'] is None
    rows[-1]['done'] = False
    with pytest.raises(ValueError): control_summary(rows, .5, .1)


def synthetic_rows():
    rows = []
    for j in roster(PROTOCOL, parent()):
        r = {**j, **{m: 0. for m in METRICS}, 'no_policy_updates': True,
             'exact_parent_parity': True if j['condition'] == 'deterministic' else None,
             'initial_traffic_sha256': str(j['simulator_seed']), 'natural_arrival': 0,
             'ego_collision': 0, 'time_limit': 1, 'other_terminal_event': 0}
        r['return'] = 0. if j['condition'] == 'deterministic' else [-9., -3., -6.][j['noise_repeat']]
        rows.append(r)
    return rows


def test_negative_result_aggregates_with_repeat_scene_seed_layers():
    rows = synthetic_rows(); r = aggregate(PROTOCOL, parent(), rows)
    assert r['episodes'] == 144 and r['independent_simulator_scenes'] == 3
    assert r['method_effect_gate'] is None and r['formal_claim'] is False
    assert r['arms']['conditional']['equal_run_seed_mean']['noise_minus_deterministic']['return'] == -6.
    assert len(r['paired_cells']) == 36
    for changed in [rows[:-1], rows+[rows[0]], [x for x in rows if x['run_seed'] != 2]]:
        with pytest.raises(ValueError): aggregate(PROTOCOL, parent(), changed)


def test_traffic_and_parity_mismatch_invalid_not_driving_failure():
    rows = synthetic_rows(); rows[0]['exact_parent_parity'] = False
    with pytest.raises(ValueError): aggregate(PROTOCOL, parent(), rows)
    rows = synthetic_rows(); rows[0]['initial_traffic_sha256'] = 'different'
    with pytest.raises(ValueError): aggregate(PROTOCOL, parent(), rows)
    rows = synthetic_rows(); rows[0]['ego_collision'] = 1
    with pytest.raises(ValueError): aggregate(PROTOCOL, parent(), rows)


def test_actual_runtime_is_bound_without_training_call():
    runtime = runner.noise_runtime()
    assert runtime['all_version'] == '0.5.3' and len(runtime['implementation_hashes']) == 5
    source = (ROOT/'tools/diagnose_ddpg_stability.py').read_text()
    assert 'checked.act(' not in source and 'experiment.train(' not in source
