from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'tools')]
import recover_terminal_events as recovery
from prediction_rl.evaluation.terminal_events_v2 import classify, VERSION
from test_terminal_events import fixture, ego, snap


def author_limit(condition):
    warm = recovery.legacy.CONDITIONS[condition]['warmup_s']
    job = next(j for j in recovery.legacy.jobs(recovery.legacy.AUDIT)
               if j['model_id'] == recovery.legacy.previous.AUTHOR and j['condition_id'] == condition)
    events = []
    def add(before, after):
        events.append({'index': len(events), 'before': before, 'after': after,
                       'arrived_vehicle_ids': [], 'colliding_vehicle_ids': [],
                       'colliding_vehicle_number': 0, 'collisions': []})
    for i in range(warm*5): add(snap(.2*i), snap(.2*(i+1)))
    e = ego(lane='ramp_0', position=100.); e.update(route_index=0, speed=0.)
    add(snap(warm), snap(warm+.2, e))
    decisions = []
    for i in range(501):
        t = events[-1]['after']['simulation_time_s']
        decisions.append({'frame': {'simulation_time_s': t, 'vehicles': {'ego': {k:e[k] for k in
                          ('position', 'speed', 'acceleration', 'lane_id')}}}})
        if i != 500: add(snap(t, e), snap(t+.2, e))
    t = events[-1]['after']['simulation_time_s']
    removal = {'after_step_count': len(events), 'before': snap(t, e), 'after': snap(t),
               'source': 'original_control_remove_ego_car'}
    add(snap(t), snap(t+.2))
    native = {**job, 'arrival': 0, 'reported_collision': 0, 'time_limit': 1, 'steps': 501}
    raw = {'version': 'raw_terminal_events_v1', 'job': job, 'capture_boundary': recovery.legacy.PROTOCOL['capture_boundary'],
           'arrival_position_source': recovery.legacy.PROTOCOL['arrival_position_source'],
           'creations': [{**recovery.legacy.PROTOCOL['ego_creation_contract'], 'simulation_time_s': warm,
                         'depart_speed_mps': 7.}], 'explicit_removals': [removal], 'steps': events}
    return raw, native, decisions


@pytest.mark.parametrize('condition', ['author20', 'author50'])
def test_exact_original_501_command_time_limit_without_synthetic_step(condition):
    raw, row, trace = author_limit(condition); before = deepcopy((raw, row, trace))
    with pytest.raises(ValueError, match='Native cleanup/extra-step'): recovery.legacy.classify(raw, row, trace)
    result = classify(raw, row, trace)
    assert result['classification'] == 'time_limit' and result['native_event_agrees']
    assert result['actual_control_steps'] == 501
    assert result['raw_simulation_steps'] == recovery.legacy.CONDITIONS[condition]['warmup_s']*5+502
    assert raw['steps'][-1]['before']['ego'] is None
    assert (raw, row, trace) == before


@pytest.mark.parametrize('condition', list(recovery.legacy.CONDITIONS))
@pytest.mark.parametrize('label', ['arrival', 'collision', 'overlap', 'far', 'unknown'])
def test_unchanged_non_author_limit_classifications(condition, label):
    raw, row, trace = fixture(label, condition)
    assert classify(raw, row, trace) == recovery.legacy.classify(raw, row, trace)


@pytest.mark.parametrize('condition', ['gym20', 'gym50'])
def test_gym_time_limit_still_uses_original_post_action_cleanup(condition):
    raw, row, trace = fixture('limit', condition)
    assert classify(raw, row, trace) == recovery.legacy.classify(raw, row, trace)


@pytest.mark.parametrize('fault', ['extra_step', 'missing_step', 'no_removal', 'duplicate_removal',
    'bad_count', 'bad_final_state', 'bad_previous_state', 'unremoved_ego', 'wrong_native_steps',
    'arrival', 'collision', 'early_missing_ego', 'shifted_removal', 'missing_last_command'])
def test_author_limit_repair_does_not_relax_evidence(fault):
    raw, row, trace = author_limit('author20')
    if fault == 'extra_step': raw['steps'].append(deepcopy(raw['steps'][-1]))
    elif fault == 'missing_step': raw['steps'].pop()
    elif fault == 'no_removal': raw['explicit_removals'].clear()
    elif fault == 'duplicate_removal': raw['explicit_removals'].append(deepcopy(raw['explicit_removals'][0]))
    elif fault == 'bad_count': raw['explicit_removals'][0]['after_step_count'] -= 1
    elif fault == 'bad_final_state': trace[-1]['frame']['vehicles']['ego']['speed'] = 1.
    elif fault == 'bad_previous_state': raw['steps'][-2]['after']['ego']['speed'] = 1.
    elif fault == 'unremoved_ego': raw['explicit_removals'][0]['after'] = deepcopy(raw['explicit_removals'][0]['before'])
    elif fault == 'wrong_native_steps': row['steps'] -= 1
    elif fault == 'arrival': row['arrival'] = 1
    elif fault == 'collision': row['reported_collision'] = 1
    elif fault == 'early_missing_ego': raw['steps'][101]['before'] = snap(raw['steps'][101]['before']['simulation_time_s'])
    elif fault == 'shifted_removal': raw['explicit_removals'][0]['before']['simulation_time_s'] += .2
    elif fault == 'missing_last_command': trace.pop()
    with pytest.raises(ValueError): classify(raw, row, trace)


def test_recovery_config_is_an_explicit_unchanged_design():
    config = json.loads((ROOT/'configs/development/p07g_terminal_events_recovery_v1.json').read_text())
    recovery.require_design(config)
    assert config == recovery.RECOVERY and config['analysis_implementation'] == VERSION
    assert config['expected_completed_cells'] + len(config['recoverable_analysis_failure_ids']) + config['expected_new_simulations'] == 84
    assert not config['training_permitted'] and not config['reward_action_termination_changes']
    config['expected_new_simulations'] = 12
    with pytest.raises(ValueError): recovery.require_design(config)


def cell_fixture(tmp_path, monkeypatch, complete=False):
    monkeypatch.setattr(recovery, 'ROOT', tmp_path)
    job = next(j for j in recovery.legacy.jobs(recovery.legacy.PROTOCOL) if j['id'] == recovery.FAILED_IDS[0])
    q = {'protocol': recovery.legacy.PROTOCOL}
    directory = tmp_path/'source'/job['id']; directory.mkdir(parents=True)
    started = {'git_commit': recovery.RECOVERY['source_execution_commit'], 'working_tree_dirty': False}
    row = {**job, 'time_limit': 1, 'independent_events': {'classification': 'time_limit'}}
    ep = {'native': 'unchanged'}
    for name, payload in {'started.json': started, 'episode.json': ep, 'trace.json': {}, 'raw_events.json': {},
        'resolved_config.json': {}}.items(): recovery.write_once(directory/name, payload)
    if complete:
        recovery.write_once(directory/'analysis.json', row)
        recovery.seal_episode(directory, recovery.legacy.binding(q, job),
            {**started, 'no_policy_updates': True, 'exact_reference_parity': True, 'interface_audit_pass': True, 'outcome': ep})
    else:
        recovery.write_once(directory/'failure.json', {**started,
            'failure_reason': 'analysis = classify(\nValueError: Native cleanup/extra-step count changed\n'})
    monkeypatch.setattr(recovery, 'analyze', lambda *a: deepcopy(row))
    return q, job, directory, row


@pytest.mark.parametrize('complete', [True, False])
def test_read_only_revalidation_preserves_every_original_byte(tmp_path, monkeypatch, complete):
    q, job, directory, _ = cell_fixture(tmp_path, monkeypatch, complete)
    before = {p.name: p.read_bytes() for p in directory.iterdir()}
    cell = recovery.source_cell(q, job, directory, {})
    assert cell['kind'] == ('verified_complete' if complete else 'recovered_analysis_failure')
    assert cell['analysis']['independent_events']['classification'] == 'time_limit'
    assert {p.name: p.read_bytes() for p in directory.iterdir()} == before
    assert set(cell['files']) == set(before)


@pytest.mark.parametrize('fault', ['missing_raw', 'unapproved_job', 'other_error', 'extra_file', 'dirty', 'wrong_commit', 'positive_result'])
def test_failure_recovery_requires_complete_specific_defect_evidence(tmp_path, monkeypatch, fault):
    q, job, directory, row = cell_fixture(tmp_path, monkeypatch)
    if fault == 'missing_raw': (directory/'raw_events.json').unlink()
    elif fault == 'unapproved_job': job = {**job, 'id': 'unapproved'}
    elif fault == 'other_error':
        (directory/'failure.json').unlink(); recovery.write_once(directory/'failure.json', {'failure_reason': 'model crash'})
    elif fault == 'extra_file': recovery.write_once(directory/'extra.json', {})
    elif fault in ('dirty', 'wrong_commit'):
        s = recovery.read(directory/'started.json'); (directory/'started.json').unlink()
        s['working_tree_dirty' if fault == 'dirty' else 'git_commit'] = True if fault == 'dirty' else 'a'*40
        recovery.write_once(directory/'started.json', s)
    elif fault == 'positive_result': row['time_limit'] = 0
    before = {p.name: p.read_bytes() for p in directory.iterdir()}
    with pytest.raises(ValueError): recovery.source_cell(q, job, directory, {})
    assert {p.name: p.read_bytes() for p in directory.iterdir()} == before


def test_completed_receipt_corruption_cannot_be_recovered_as_failure(tmp_path, monkeypatch):
    q, job, directory, _ = cell_fixture(tmp_path, monkeypatch, True)
    recovery.write_once(directory/'unexpected.json', {})
    with pytest.raises(ValueError, match='artifacts changed'): recovery.source_cell(q, job, directory, {})


def test_prepare_only_freezes_ten_pending_jobs_without_running_sumo(tmp_path, monkeypatch):
    manifest = {'pending_jobs': [{'id': 'pending_'+str(i)} for i in range(10)], 'cells': {}}
    out = tmp_path/'prepared'
    monkeypatch.setattr(recovery.shared, 'clean', lambda: None)
    monkeypatch.setattr(recovery, 'source_snapshot', lambda: manifest)
    monkeypatch.setattr(recovery.shared, 'output_dir', lambda *a: out)
    monkeypatch.setattr(recovery.shared, 'metadata', lambda: {'git_commit': 'a'*40})
    monkeypatch.setattr(recovery.shared, 'env', lambda: {'device': 'cpu'})
    monkeypatch.setattr(recovery.legacy.previous.p7, 'runtime_identity', lambda: {})
    monkeypatch.setattr(recovery.legacy.previous.cross.external, 'assets', lambda: {})
    monkeypatch.setattr(recovery.subprocess, 'run', lambda *a, **kw: pytest.fail('prepare must not launch workers'))
    path, expected = recovery.prepare(recovery.RECOVERY, 'prepared')
    assert len(recovery.read(path)['jobs']) == 10 and recovery.digest(recovery.read(path)) == expected
    p = recovery.read(out/'preparation.json')
    assert p['verified_complete_reused'] == 72 and p['failed_raw_records_revalidated'] == 2
    assert p['status'] == 'prepared_not_simulated' and not (out/'evaluate').exists()


def test_incomplete_new_cell_is_preserved_without_automatic_retry(tmp_path, monkeypatch):
    job = {'id': 'new'}; q = {'jobs': [job], 'protocol': {'workers': 2}}
    directory = tmp_path/'evaluate'/'new'; directory.mkdir(parents=True)
    recovery.write_once(directory/'started.json', {})
    monkeypatch.setattr(recovery, 'load', lambda *a: (tmp_path/'request.json', q, {}))
    monkeypatch.setattr(recovery.shared, 'metadata', lambda: {'git_commit': 'a'*40})
    monkeypatch.setattr(recovery.subprocess, 'run', lambda *a, **kw: pytest.fail('must not retry incomplete'))
    with pytest.raises(ValueError, match='Incomplete/failed'): recovery.execute(tmp_path/'request.json', 'hash', True)
    assert (directory/'started.json').exists() and not (tmp_path/'writer.lock').exists()


def test_full_84_cell_aggregate_has_explicit_reuse_and_new_simulation_origins(tmp_path, monkeypatch):
    jobs = recovery.legacy.jobs(recovery.legacy.PROTOCOL)
    def row(job):
        return {**job, 'arrival': 0, 'reported_collision': 0, 'time_limit': 1,
                'parity': {'exact_native_outcome_parity': True, 'exact_available_trajectory_parity': True},
                'independent_events': {'classification': 'time_limit', 'native_arrival_with_ego_collision': False}}
    old = {j['id']: {'kind': 'verified_complete' if i < 72 else 'recovered_analysis_failure',
                    'directory': 'unchanged_source/'+j['id'], 'files': {'raw_events.json': 'a'*64}, 'analysis': row(j)}
           for i, j in enumerate(jobs[:74])}
    q = {'jobs': jobs[74:]}; manifest = {'cells': old}; out = tmp_path/'new'
    for j in q['jobs']:
        d = out/'evaluate'/j['id']; recovery.write_once(d/'episode.json', {})
        recovery.write_once(d/'analysis.json', row(j))
    monkeypatch.setattr(recovery.legacy, 'parent', lambda: (None, None, None, None, {}))
    monkeypatch.setattr(recovery, 'verify_episode', lambda *a: {'metadata': {'no_policy_updates': True,
        'interface_audit_pass': True, 'exact_reference_parity': True, 'analysis_implementation': VERSION, 'outcome': {}}})
    monkeypatch.setattr(recovery, 'analyze', lambda q, j, *a: row(j))
    result = recovery.collect(q, manifest, out)
    assert result['episodes'] == 84 and result['method_effect_gate'] is None and result['engineering_complete']
    assert [r['id'] for r in result['rows']] == [j['id'] for j in jobs]
    counts = {k: sum(r['recovery_origin']['kind'] == k for r in result['rows'])
              for k in ('verified_complete', 'recovered_analysis_failure', 'new_simulation')}
    assert list(counts.values()) == [72, 2, 10]
    assert all(r['recovery_origin']['source_request_hash'] == recovery.RECOVERY['source_request_hash']
               for r in result['rows'][:74])
    assert not result['source_artifacts_modified'] and result['analysis_implementation'] == VERSION
