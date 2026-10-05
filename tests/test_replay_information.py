from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'tools')]
from prediction_rl.data.collection_store import read_json, file_hash, write_once
from prediction_rl.data.dataset_contract import digest
from prediction_rl.features.probe_features import METADATA
from prediction_rl.evaluation.replay_information import (
    PROTOCOL, AUDIT, PERTURBATIONS, jobs, validate_protocol, validate_vector,
    perturb, select_states, analyze, aggregate, describe)
from prediction_rl.training.all_ddpg import preset_factory, checkpoint_contract, save_evaluation_weights
import diagnose_replay_information as cli


def row(i):
    state = np.arange(170, dtype=np.float32)/100
    state[160:169] = [1., .5, 1., .5, 1., .5, 1., .5, 1.]
    step = (i-1) % 12; state[169] = np.float32(step)*np.float32(.001)
    return {'transition_id': i, 'episode': (i-1)//12, 'step': step, 'state': state.tolist(),
            'physical_before': {'observation_sha256': digest(state[:169].tolist()),
                'feature_metadata': dict(zip(METADATA, map(float, state[160:169]))),
                'ego': {'speed': 0. if i % 5 == 0 else 20., 'lane_id': 'ramp_0' if i % 3 else 'highway_0'},
                'history_frames': 6 if step < 2 else 11}, 'reward': 10.}


def test_frozen_design_roster_and_budget():
    assert read_json(ROOT/'configs/development/p07m_replay_information_v1.json') == PROTOCOL
    assert [(j['arm'], j['run_seed']) for j in jobs(PROTOCOL)] == [
        (a, s) for a in ('ordinary', 'conditional') for s in (0, 1, 2)]
    assert PROTOCOL['sample']['uniform_cap']+4*PROTOCOL['sample']['stratum_cap'] == 1024
    assert PROTOCOL['training_permitted'] is False and PROTOCOL['simulation_permitted'] is False


@pytest.mark.parametrize('key', ['run_seeds', 'checkpoint_frames', 'unknown', 'training_permitted', 'sample'])
def test_config_changes_rejected(key):
    p = deepcopy(PROTOCOL)
    if key == 'sample': p[key]['uniform_cap'] = 9999
    else: p[key] = True
    with pytest.raises(ValueError): validate_protocol(p)


@pytest.mark.parametrize('fault', ['dimension', 'nan', 'time', 'observation_hash', 'metadata', 'presence', 'role_mask', 'pool_mask'])
def test_saved_input_contract_rejects_corruption(fault):
    r = row(1)
    if fault == 'dimension': r['state'].append(1.)
    elif fault == 'nan': r['state'][0] = float('nan')
    elif fault == 'time': r['state'][-1] = 1.
    elif fault == 'observation_hash': r['physical_before']['observation_sha256'] = '0'*64
    elif fault == 'metadata': r['physical_before']['feature_metadata']['front_present'] = 0.
    else:
        k = 'front_present' if fault != 'pool_mask' else 'pool_present'
        value = .5 if fault == 'presence' else 0.
        r['physical_before']['feature_metadata'][k] = value
        r['state'][160+METADATA.index(k)] = value
        r['physical_before']['observation_sha256'] = digest(r['state'][:169])
    with pytest.raises(ValueError): validate_vector(r)


def test_reservoir_outcome_blind_no_rng_bounded_and_reproducible():
    rs = [row(i) for i in range(1, 901)]; j = jobs(PROTOCOL)[0]
    rng = torch.get_rng_state().clone(); numpy_before = deepcopy(np.random.get_state())
    selected, s = select_states(iter(rs), j, PROTOCOL, 900)
    rs2 = deepcopy(rs)
    for r in rs2: r['reward'] = -1000.; r['done'] = True
    selected2, s2 = select_states(iter(rs2), j, PROTOCOL, 900)
    assert s == s2 and selected == selected2 and s['unique_selected_states'] <= 1024
    assert s['eligible_counts']['low_speed'] == 180 and len(s['selected_ids']['uniform']) == 512
    assert len(s['selected_ids']['short_history']) == 128
    assert torch.equal(rng, torch.get_rng_state())
    numpy_after = np.random.get_state(); np.testing.assert_array_equal(numpy_before[1], numpy_after[1])
    assert numpy_before[2:] == numpy_after[2:]


def test_audit_scan_stops_at_exact_prefix():
    def stream():
        for i in range(1, 129): yield row(i)
        raise AssertionError('Audit scanned beyond bound')
    selected, s = select_states(stream(), jobs(AUDIT)[0], AUDIT, 900)
    assert s['scanned_transitions'] == 128 and len(selected) <= 24


@pytest.mark.parametrize('fault', ['missing', 'duplicate', 'empty'])
def test_scan_rejects_incomplete_or_reordered_replay(fault):
    rs = [row(i) for i in range(1, 21)]
    if fault == 'missing': rs.pop()
    elif fault == 'duplicate': rs[-1]['transition_id'] = 19
    else: rs = []
    with pytest.raises(ValueError): select_states(rs, jobs(PROTOCOL)[0], PROTOCOL, 20)


@pytest.mark.parametrize('variant', PERTURBATIONS)
def test_perturbations_preserve_observation_time_and_correct_candidate_layout(variant):
    x = torch.tensor([row(1)['state'], row(2)['state']], dtype=torch.float32); copy = x.clone()
    y = perturb(x, variant)
    assert torch.equal(x, copy) and torch.equal(y[:, :20], x[:, :20]) and torch.equal(y[:, 169:], x[:, 169:])
    if variant == 'zero_all_prediction': assert (y[:, 20:169] == 0).all()
    else:
        assert torch.equal(y[:, 160:169], x[:, 160:169])
        if variant == 'zero_values_keep_metadata': assert (y[:, 20:160] == 0).all()
        else:
            role = y[:, 20:140].reshape(-1, 5, 2, 4, 3); pool = y[:, 140:160].reshape(-1, 5, 4)
            for c in range(5):
                assert torch.equal(role[:, c], x[:, 20:140].reshape(-1, 5, 2, 4, 3).mean(1))
                assert torch.equal(pool[:, c], x[:, 140:160].reshape(-1, 5, 4).mean(1))


def test_missing_role_stays_masked_and_constant_candidates_unchanged():
    x = torch.zeros(2, 170); x[:, 20:140] = torch.arange(24).float().repeat(5)
    x[:, 20:140].reshape(-1, 5, 2, 4, 3)[:, :, 0] = 0
    x[:, 140:160] = torch.arange(4).float().repeat(5); x[:, 162] = 1.
    assert torch.equal(perturb(x, 'candidate_mean_keep_metadata'), x)


def test_actual_strict_checkpoint_and_direct_queries_no_time_or_updates(tmp_path, monkeypatch):
    from all.logging import DummyWriter
    torch.set_num_threads(1); torch.manual_seed(56)
    env = cli.SpaceOnlyEnv('ordinary'); original = preset_factory('cpu', .0002)(env, DummyWriter())
    # A nontrivial deterministic actor, rather than all-zero ALL initialization.
    with torch.no_grad(): original.agent.policy.model.model[-1].weight.fill_(.001)
    contract = checkpoint_contract(env, original.preset_parameters, '1'*64, '2'*64)
    save_evaluation_weights(original, tmp_path/'n60000.pt', contract)
    model = {'arm': 'ordinary', 'directory': str(tmp_path), 'weights_contract': contract,
             'weights_sha256': file_hash(tmp_path/'n60000.pt')}
    monkeypatch.setattr(cli, 'inside', lambda root, path: Path(path))
    agent = cli.loaded_agent(model)
    monkeypatch.setattr(agent.body, 'eval', lambda *args: pytest.fail('Appended TimeFeature twice'))
    monkeypatch.setattr(agent.agent, 'act', lambda *args: pytest.fail('Called training agent'))
    selected, selection = select_states([row(i) for i in range(1, 21)], jobs(AUDIT)[0], AUDIT, 20)
    result = analyze(agent, selected, selection, AUDIT)
    assert result['no_updates_no_time_advance_no_rng_draws']
    assert result['groups']['uniform']['zero_values_keep_metadata']['mean'] > 0
    assert agent.body.timestep is None and not agent.agent.q._optimizer.state
    model['weights_contract'] = {**contract, 'arm': 'conditional'}
    with pytest.raises(ValueError): cli.loaded_agent(model)


def test_negative_or_zero_sensitivity_completes_without_method_gate():
    rows = [{'job': j, 'no_updates_no_time_advance_no_rng_draws': True,
             'groups': {'uniform': {v: describe([0.]*8, .1) for v in PERTURBATIONS}}} for j in jobs(PROTOCOL)]
    a = aggregate(PROTOCOL, rows)
    assert a['engineering_complete'] and a['method_effect_gate'] is None and a['simulation_episodes'] == 0
    assert a['automatic_budget_extension'] is False
    assert a['prediction_error_attribution'] == 'not_identifiable_from_saved_replay_vectors'
    with pytest.raises(ValueError): aggregate(PROTOCOL, rows[:-1])
    with pytest.raises(ValueError): aggregate(PROTOCOL, rows+[rows[0]])


def test_descriptive_threshold_not_method_success_rule():
    s = describe([0., .05, .2], .1)
    assert s['reference_exceedance_rate'] == 1/3 and s['mean'] == pytest.approx(.25/3)
    assert describe([], .1)['mean'] is None
    with pytest.raises(ValueError): describe([float('nan')], .1)


def test_partial_diagnostic_never_auto_retried(tmp_path):
    write_once(tmp_path/'failure.json', {'failed': True})
    with pytest.raises(ValueError): cli.model_run(tmp_path, {}, {}, False)
    with pytest.raises(ValueError): cli.model_run(tmp_path, {}, {}, True)
