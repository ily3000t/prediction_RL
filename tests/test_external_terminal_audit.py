from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'tools')]
from prediction_rl.evaluation.external_terminal_audit import (
    PROTOCOL, AUDIT, AUTHOR, B0, B3, MODELS, NATIVE_FIELDS, roster,
    validate_protocol, native_row, assert_parity, analyze, aggregate, paired)
from test_terminal_events import fixture


def cell(job):
    sidecar, native, decisions = fixture('overlap', 'author50')
    sidecar['job'] = job
    decisions[0].update(commanded_speed=7., requested_jerk=None, policy_queries=[])
    outcome = {k: 1. for k in NATIVE_FIELDS}
    outcome.update(arrival=1, collision=0, time_limit=0, steps=1, st_takeover_rate=None,
        policy_sha256=None if job['model_id'] == 'author_st' else 'a'*64,
        predictor_sha256='b'*64 if job['model_id'] == B3 else None)
    row = native_row(job, outcome, decisions)
    reference = {'row': row, 'decisions': deepcopy(decisions), 'histories': {'merged': True}, 'takeovers': None}
    return job, outcome, decisions, {'merged': True}, None, sidecar, reference


def rows(config=AUDIT):
    return [analyze(*cell(job)) for job in roster(config)]


def test_frozen_config_and_200_cells_keep_all_seeds_and_external_supervisors():
    assert json.loads((ROOT/'configs/development/p07h_external_terminal_audit_v1.json').read_text()) == PROTOCOL
    jobs = roster(PROTOCOL)
    assert len(jobs) == 200 and len(roster(AUDIT)) == 7
    assert len({j['id'] for j in jobs}) == 200
    assert {j['condition_id'] for j in jobs} == {'author50'}
    assert {j['run_seed'] for j in jobs if j['model_id'] == B3} == {0, 1, 2}
    assert {'author_st', 'author_rl_mpc_safety', 'author_rl_mpc_switching'} <= set(MODELS)


@pytest.mark.parametrize('field,value', [('warmup_s',20), ('simulator_seeds',[200]),
    ('project_run_seeds',[0,1]), ('workers',True), ('training_permitted',True),
    ('strict_native_trajectory_parity',False), ('native_reward_recomputed_with_new_labels',True), ('unknown',1)])
def test_amended_design_rejected(field,value):
    config = deepcopy(PROTOCOL); config[field] = value
    with pytest.raises(ValueError): validate_protocol(config)


def test_negative_event_finishes_without_rewriting_original_positive_reward():
    a = aggregate(AUDIT, rows())
    assert a['status'] == 'complete' and a['method_effect_gate'] is None
    assert a['episodes'] == 7 and a['independent_simulator_seed_ids'] == 1
    for entry in a['matrix'].values():
        assert entry['summary']['native_author_reward_mean'] == 1.
        assert entry['summary']['independent_rates']['ego_collision'] == 1.
        assert entry['summary']['natural_arrival_duration_s']['count'] == 0
        assert entry['summary']['natural_arrival_duration_s']['mean'] is None


@pytest.mark.parametrize('fault', ['missing','duplicate','identity','parity','initial','weights','unknown'])
def test_missing_or_corrupt_cells_do_not_become_valid_results(fault):
    r = rows(PROTOCOL)
    if fault == 'missing': r.pop()
    elif fault == 'duplicate': r[-1] = deepcopy(r[0])
    elif fault == 'identity': r[0]['simulator_seed'] = 201
    elif fault == 'parity': r[0]['parity']['exact_native_outcome_parity'] = False
    elif fault == 'initial': r[0]['common_initial_traffic_sha256'] = 'c'*64
    elif fault == 'weights': r[0]['policy_sha256'] = 'c'*64
    else: r[0]['independent_events']['classification'] = 'unknown'
    with pytest.raises(ValueError): aggregate(PROTOCOL,r)


@pytest.mark.parametrize('fault', ['reward','frame','command','queries','histories','takeover'])
def test_recorder_must_preserve_native_reward_controller_and_supervisor(fault):
    j,o,d,h,t,s,ref = cell(roster(AUDIT)[0]); row = native_row(j,o,d)
    if fault == 'reward': row['author_history_total_reward'] = 2.
    elif fault == 'frame': d[0]['frame']['vehicles']['ego']['speed'] = 9.
    elif fault == 'command': d[0]['commanded_speed'] = 9.
    elif fault == 'queries': d[0]['policy_queries'] = [1.,2.]
    elif fault == 'histories': h['merged'] = False
    else: t = [True]
    with pytest.raises(ValueError): assert_parity(row,d,h,t,ref)


def test_mpc_queries_keep_multiple_requests_and_st_has_no_checkpoint():
    args = cell(next(j for j in roster(AUDIT) if j['model_id'] == 'author_rl_mpc_safety'))
    j,o,d,h,t,s,ref = args
    d[0]['policy_queries'] = [1.,2.,3.]; ref['decisions'][0]['policy_queries'] = [1.,2.,3.]
    result = analyze(*args)
    assert 'policy_queries' in result['parity']['compared_decision_fields']
    st = cell(next(j for j in roster(AUDIT) if j['model_id'] == 'author_st'))
    assert analyze(*st)['policy_sha256'] is None


def test_missing_p7g_histories_are_explicit_and_not_fabricated():
    j,o,d,h,t,s,ref = cell(roster(AUDIT)[0]); ref['histories']['jerk_history'] = [1.]
    result = analyze(j,o,d,h,t,s,ref)
    assert result['parity']['historical_histories_not_present_in_raw_replay'] == ['jerk_history']


def test_time_efficiency_requires_both_independent_natural_arrivals():
    l,r = rows()[:2]; r['simulation_duration_s'] = 99.
    p = paired(l,r)
    assert p['right_minus_left']['duration_s_both_natural_arrival'] is None
    l['independent_events']['classification'] = 'natural_arrival'
    r['independent_events']['classification'] = 'natural_arrival'
    assert paired(l,r)['right_minus_left']['duration_s_both_natural_arrival'] == 98.


def test_source_snapshot_requires_exact_raw_identity_without_simulation(monkeypatch):
    import audit_external_terminal_events as runner
    jobs = roster(PROTOCOL)
    def ref(job): return {'row': job}
    reused = {j['id'] for j in jobs if j['model_id'] in (AUTHOR,B0,B3) and j['simulator_seed'] in (200,210,219)}
    monkeypatch.setattr(runner,'reference',ref)
    monkeypatch.setattr(runner,'raw_reuse',lambda j,r:{'exact':True} if j['id'] in reused else None)
    monkeypatch.setattr(runner,'identity',lambda:{'frozen':True})
    monkeypatch.setattr(runner.subprocess,'run',lambda *a,**k:pytest.fail('Preparing sources must not simulate'))
    m = runner.source_snapshot(PROTOCOL)
    assert len(m['references']) == 200 and len(m['reuse']) == 21 and len(m['pending_jobs']) == 179
    assert all(j['id'] not in reused for j in m['pending_jobs'])


def test_resume_cannot_retry_partial_or_failed_new_cells(tmp_path,monkeypatch):
    import audit_external_terminal_events as runner
    from prediction_rl.data.collection_store import write_once
    j = roster(AUDIT)[1]; q = {'jobs':[j],'protocol':AUDIT}
    directory = tmp_path/'evaluate'/j['id']; directory.mkdir(parents=True)
    write_once(directory/'started.json',{'status':'incomplete'})
    monkeypatch.setattr(runner,'ROOT',tmp_path)
    monkeypatch.setattr(runner,'load',lambda *a:(tmp_path/'request.json',q,{'references':{j['id']:{}}}))
    monkeypatch.setattr(runner.shared,'metadata',lambda:{'git_commit':'a'*40})
    monkeypatch.setattr(runner.subprocess,'run',lambda *a,**k:pytest.fail('Incomplete cells must never retry'))
    with pytest.raises(ValueError,match='Incomplete/failed'): runner.execute(tmp_path/'request.json','frozen',True)
    assert (directory/'started.json').exists() and not (directory/'complete.json').exists()


def test_stable_native_fields_exclude_wall_clock_and_include_takeover():
    assert 'st_takeover_rate' in NATIVE_FIELDS
    assert not any('latency' in field for field in NATIVE_FIELDS)


@pytest.mark.parametrize('model',[AUTHOR,B0,B3,'author_st','author_rl_mpc_safety','author_rl_mpc_switching'])
def test_original_decision_adapter_does_not_invent_mpc_jerk(model):
    import audit_external_terminal_events as runner
    j = next(j for j in roster(AUDIT) if j['model_id'] == model)
    _,_,d,*_ = cell(j)
    result = {'actual_observed_frames':[d[0]['frame']], 'histories':{'control_history':[7.]},
              'decision_policy_queries':[[1.]] if model == AUTHOR else [[1.,2.]],
              'requested_conditional_jerks':[.5] if model == B3 else None,
              'actual_query_records':[{'requested_jerk':.3}]}
    record = runner.records(result,j)[0]
    assert record['commanded_speed'] == 7.
    if model in ('author_st','author_rl_mpc_safety','author_rl_mpc_switching'):
        assert record['requested_jerk'] is None and record['policy_queries'] == [1.,2.]
    elif model == AUTHOR: assert record['requested_jerk'] == 1.
    elif model == B0: assert record['requested_jerk'] == .3
    else: assert record['requested_jerk'] == .5
