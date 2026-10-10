from copy import deepcopy
import hashlib
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'tools')]

from prediction_rl.curriculum.entry_smoke import PROTOCOL, compare_traces, jobs, json_safe, validate_trace
from prediction_rl.data.collection_store import file_hash, seal_episode, write_once
from prediction_rl.data.dataset_contract import digest
from smoke_curriculum_entry import read, read_sealed_trace
import review_curriculum_entry as reviewer
from test_curriculum_fresh_entry import complete_trace


def seal_trace(directory, trace, binding, metadata):
    directory.mkdir()
    write_once(directory/'started.json', {'status': 'running', **binding,
        'git_commit_sha': reviewer.ORIGINAL_SOURCE_COMMIT})
    write_once(directory/'trace.json', json_safe(trace))
    seal_episode(directory, binding, metadata)


def test_numeric_settings_keys_are_canonicalized_only_by_verified_persistence(tmp_path):
    live = {'job': {'id': 'worker'}, 'runtime': {'pid': 2,
        'settings': {'JERK_VALUES_DQN': {0: -5, 1: -2.5, 2: 0, 3: 2.5, 4: 5}}},
        'packets': [{'features': {'ordinary': {'observation': [0.], 'feature_latency_s': .1}}}]}
    direct = deepcopy(live)
    direct['job'] = {'id': 'direct'}
    direct['runtime']['pid'] = 1
    direct['runtime']['settings']['JERK_VALUES_DQN'] = {'0': -5, '1': -2.5, '2': 0, '3': 2.5, '4': 5}
    with pytest.raises(ValueError, match='exact parity'):
        compare_traces(direct, live)
    binding = {'request_hash': 'frozen', 'job': live['job']}
    directory = tmp_path/'sealed'
    seal_trace(directory, live, binding, {'original': True})
    before = file_hash(directory/'trace.json')
    persisted, metadata = read_sealed_trace(directory, binding)
    assert compare_traces(direct, persisted)['exact_native_and_feature_parity']
    assert metadata == {'original': True}
    assert file_hash(directory/'trace.json') == before
    assert set(live['runtime']['settings']['JERK_VALUES_DQN']) == {0, 1, 2, 3, 4}


@pytest.mark.parametrize('mutation', ['setting', 'extra_file', 'binding', 'unsealed'])
def test_sealed_trace_reader_rejects_changed_or_unsealed_evidence(tmp_path, mutation):
    directory = tmp_path/'sealed'
    value = complete_trace()
    binding = {'request_hash': 'original', 'job': value['job']}
    seal_trace(directory, value, binding, {})
    if mutation == 'setting':
        # Simulate corruption without overwriting original files or re-sealing.
        actual_sha = file_hash(directory/'trace.json')
        monkey_receipt = read(directory/'complete.json')
        monkey_receipt['files']['trace.json'] = '0'*64
        from unittest.mock import patch
        with patch('smoke_curriculum_entry.verify_episode', return_value=monkey_receipt):
            with pytest.raises(ValueError, match='Sealed trace bytes'):
                read_sealed_trace(directory, binding)
        assert file_hash(directory/'trace.json') == actual_sha
        return
    if mutation == 'extra_file': write_once(directory/'unexpected.json', {})
    elif mutation == 'binding': binding = {**binding, 'request_hash': 'different'}
    elif mutation == 'unsealed': directory = tmp_path/'not_sealed'
    with pytest.raises(ValueError): read_sealed_trace(directory, binding)


def test_historical_runner_is_verified_but_other_source_drift_is_not_allowed(monkeypatch):
    original = b'original runner\n'
    before = hashlib.sha256(original).hexdigest()
    key = 'worktree:'+reviewer.RUNNER
    q = {'input_hashes': {key: before, 'worktree:runtime.py': 'frozen'}}
    monkeypatch.setattr(reviewer.subprocess, 'check_output', lambda command: original)
    current = {key: 'new runner', 'worktree:runtime.py': 'frozen'}
    result = reviewer.verify_historical_inputs(q, current)
    assert result['changed_input_names'] == [key] and result['historical_runner_verified_from_git']
    for invalid in ({**current, 'worktree:runtime.py': 'changed'}, {**current, 'extra': 'not_requested'}):
        with pytest.raises(ValueError): reviewer.verify_historical_inputs(q, invalid)
    monkeypatch.setattr(reviewer.subprocess, 'check_output', lambda command: b'wrong history')
    with pytest.raises(ValueError, match='Historical runner'): reviewer.verify_historical_inputs(q, current)


def original_failure():
    q = {'protocol': PROTOCOL, 'git_commit_sha': reviewer.ORIGINAL_SOURCE_COMMIT, 'jobs': jobs(PROTOCOL)}
    value = {'status': 'failed', 'engineering_gate': False, 'request_hash': digest(q),
        'git_commit_sha': q['git_commit_sha'], 'working_tree_dirty': False, 'protocol': PROTOCOL,
        'failure_type': 'ValueError', 'failure_reason': 'Native/feature trajectory mismatch; exact parity cannot be relaxed',
        'runtime_parity_verified': False, 'training_ready': False, 'ddpg_started': False,
        'test_opened': False, 'optimizer_updates': 0, 'new_counterfactual_branches': 0,
        'rows': [{'job': job} for job in q['jobs']]}
    return q, value


@pytest.mark.parametrize('field,value', [('failure_reason', 'unknown error'), ('training_ready', True),
    ('test_opened', True), ('optimizer_updates', 1), ('predictor_tensor_hashes', {'invented': 'postflight'}),
    ('parent_rng_unchanged', True), ('predictors_frozen', True), ('rows', [])])
def test_review_does_not_reinterpret_incomplete_or_other_failures(field, value):
    q, failure = original_failure()
    reviewer.validate_original_failure(failure, q, digest(q))
    failure[field] = value
    with pytest.raises(ValueError): reviewer.validate_original_failure(failure, q, digest(q))


def make_original_grid(tmp_path, monkeypatch, bad_rng=False):
    geometry = SimpleNamespace(validate_frame=lambda *a, **kw: None)
    monkeypatch.setattr(reviewer, 'MergeGeometry', lambda path: geometry)
    q = {'jobs': jobs(PROTOCOL), 'protocol': PROTOCOL}
    rows, traces = [], {}
    for index, job in enumerate(q['jobs']):
        trace = complete_trace()
        trace.update(job=job, runtime={'pid': index, 'settings': {'SEED': job['simulator_seed'],
            'JERK_VALUES_DQN': {0: -5, 1: -2.5, 2: 0, 3: 2.5, 4: 5}}})
        row = {'job': job, **validate_trace(trace, job, geometry),
            'parent_rng_unchanged': not (bad_rng and index == 0), 'finished_at': 'recorded'}
        seal_trace(tmp_path/job['id'], trace, {'request_hash': digest(q), 'job': job}, row)
        rows.append(row)
        traces[job['id']] = read(tmp_path/job['id']/'trace.json')
    prefix = [{'left': left, 'right': right, **compare_traces(traces[left], traces[right])}
              for left, right in (('d11000r0', 'd11000r1'), ('w11000r0', 'w11000r1'))]
    return q, {'rows': rows, 'comparisons': prefix}


def test_offline_grid_reuses_all_receipts_without_simulation_or_inference(tmp_path, monkeypatch):
    q, failure = make_original_grid(tmp_path, monkeypatch)
    monkeypatch.setattr(reviewer.subprocess, 'run', lambda *a, **kw: pytest.fail('No subprocess execution'))
    from smoke_curriculum_entry import FreshGym20TaskEnv
    monkeypatch.setattr(FreshGym20TaskEnv, '__init__', lambda *a, **kw: pytest.fail('No environment construction'))
    original = {job['id']: file_hash(tmp_path/job['id']/'trace.json') for job in q['jobs']}
    result = reviewer.review_jobs(tmp_path/'request.json', q, failure)
    assert result['counts'] == {'actual_episodes': 12, 'actual_control_steps': 144,
        'waiting_insertion_ticks': 1212, 'raw_ticks': 1356, 'feature_calls_per_mode': 156,
        'native_terminal_episodes': 0, 'unique_simulator_scenes': 3}
    assert len(result['comparisons']) == 12 and all(x['exact_native_and_feature_parity'] for x in result['comparisons'])
    assert result['representation_diagnostic']['source_based_reconstruction_not_original_memory_capture']
    assert result['representation_diagnostic']['mismatch_reproduced_with_unchanged_comparator']
    assert result['representation_diagnostic']['json_roundtrip_passes_unchanged_comparator']
    assert all(result['episode_file_hashes'][j]['trace.json'] == h for j, h in original.items())
    assert all(file_hash(tmp_path/j/'trace.json') == h for j, h in original.items())


def test_offline_review_preserves_rng_failures_instead_of_claiming_parity(tmp_path, monkeypatch):
    q, failure = make_original_grid(tmp_path, monkeypatch, bad_rng=True)
    with pytest.raises(ValueError, match='metadata/RNG'): reviewer.review_jobs(tmp_path/'request.json', q, failure)


def test_review_contract_cannot_authorize_more_work():
    assert reviewer.REVIEW_CONTRACT['new_episodes'] == reviewer.REVIEW_CONTRACT['new_control_steps'] == 0
    for flag in ('predictor_inference_run', 'sumo_started', 'ddpg_started', 'training_ready', 'rewrite_original_report'):
        assert reviewer.REVIEW_CONTRACT[flag] is False
