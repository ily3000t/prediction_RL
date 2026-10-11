from copy import deepcopy
import json
from pathlib import Path
import random
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'tools')]
from prediction_rl.curriculum.interaction_intensity import (
    PROTOCOL, ExposureAudit, OfflineEpisodeSampler, proposal, schedule_proposal, validate_protocol)
from test_curriculum_task_rule_proposal import family
import propose_interaction_intensity as cli


def rows():
    return [r for i,s in enumerate(PROTOCOL['task_seeds']) for r in family(s, score=float(i), closing=float(i))]


def roster():
    return [{'task_id':f't{i}', 'simulator_seed':11000+i, 'level':level}
            for level,start,n in [('L',0,11),('M',11,10),('H',21,11)] for i in range(start,start+n)]


def sampler(mode='curriculum', seed=3, tasks=None):
    return OfflineEpisodeSampler(roster() if tasks is None else tasks, mode, seed,
                                 purpose='offline_algorithm_test_only')


def test_independent_readable_config_and_no_training_authorization():
    assert validate_protocol(json.loads((ROOT/cli.CONFIG).read_text())) == PROTOCOL
    assert not PROTOCOL['training_ready'] and not PROTOCOL['ddpg_training_authorized']
    assert not PROTOCOL['schedule_execution_authorized'] and not PROTOCOL['task_layer_assignment_authorized']


@pytest.mark.parametrize('key,value', [('window_steps',25),('anchor','reset'),('normalization','new'),
    ('D_use','classifier'),('minimum_unique_tasks_per_level',1),('unknown',True),
    ('new_sumo_sessions_authorized',True),('ddpg_training_authorized',True),
    ('task_classifier',{}),('target_distribution_pi',{}),('schedule_execution_authorized',True),
    ('evaluation_seeds',[81000]),('requested_control_steps_per_model',200000)])
def test_unknown_or_expanded_contract_rejected(key,value):
    cfg = deepcopy(PROTOCOL); cfg[key] = value
    with pytest.raises(ValueError): validate_protocol(cfg)


def test_three_mutually_exclusive_intensity_layers_and_old_live_fields_unchanged():
    source = rows(); before = deepcopy(source); result = proposal(source,PROTOCOL)
    assert result['summary']['proposed_counts'] == {'L':11,'M':10,'H':11}
    assert result['summary']['coverage_gate'] is True
    assert result['summary']['fixed_three_level_schedule_mathematically_schedulable'] is True
    assert all(t['task_level'] is None and not t['classification_applied'] and not t['execution_authorized']
               and not t['training_ready'] for t in result['tasks'])
    assert result['summary']['target_distribution_pi'] is None and not result['summary']['distribution_frozen']
    assert source == before
    assert proposal(list(reversed(source)),PROTOCOL) == result


def test_D_does_not_determine_or_filter_labels():
    source = rows(); baseline = proposal(source,PROTOCOL)
    for i,r in enumerate(source):
        for rel in r['road_relations']['target_relations'].values():
            rel['route_projected_gap_m'] = -100. if i%2 else 100.
            rel['closing_speed_mps'] = 20. if i%3 else 0.
    result = proposal(source,PROTOCOL)
    assert [(t['task_id'],t['proposed_level']) for t in result['tasks']] == [
        (t['task_id'],t['proposed_level']) for t in baseline['tasks']]
    assert result['summary']['independent_D_and_coverage_by_level'] != baseline['summary']['independent_D_and_coverage_by_level']


@pytest.mark.parametrize('fault', ['missing_response','missing_slots','missing_noise','collapsed','high_noise'])
def test_unusable_response_or_boundaries_block_freeze_without_assigning_residual_M(fault):
    source = rows()
    if fault == 'missing_response':
        source[2]['common_prefix']['per_actor']['cross']['response'].update(status='unavailable')
    elif fault == 'missing_slots': source[2]['original_evidence']['root_context']['coverage']['key_actor_missing_slots'] = ['cross']
    elif fault == 'missing_noise': source[2]['original_evidence']['repeat_noise_with_frozen_IC01_scales'] = None
    elif fault == 'collapsed':
        for r in source:
            for a in r['common_prefix']['per_actor'].values(): a['response']['I_dense_with_frozen_IC01_scales'] = 0.
    else:
        for r in source: r['original_evidence']['repeat_noise_with_frozen_IC01_scales'] = 100.
    result = proposal(source,PROTOCOL)
    assert not result['summary']['coverage_gate'] and result['summary']['unclassified_task_ids']
    assert result['summary']['schedule_proposal'] is None
    assert not result['summary']['training_ready']


@pytest.mark.parametrize('fault',['missing','duplicate_root','duplicate_episode','wrong_seed'])
def test_full_roster_no_replacements_or_duplicate_source(fault):
    source = rows()
    if fault == 'missing': source = source[3:]
    elif fault == 'duplicate_root': source[2] = deepcopy(source[1])
    elif fault == 'duplicate_episode':
        for r in source[3:6]: r['original_evidence']['episode_id'] = source[0]['original_evidence']['episode_id']
    else: source[-1]['simulator_seed'] = 1000
    with pytest.raises(ValueError): proposal(source,PROTOCOL)


def test_new_schedule_and_matching_nominal_mix():
    result = schedule_proposal({'L':11/32,'M':10/32,'H':11/32})
    assert result['nominal_Mix_reset_mass'] == pytest.approx({'L':.3385416666666667,'M':.2729166666666667,'H':.3885416666666667})
    assert result['stages'][2] == {'start':60000,'end':120000,'mass':{'L':11/32,'M':10/32,'H':11/32}}
    assert not result['actual_exposure_equality_claim']


@pytest.mark.parametrize('mass', [{'L':1,'M':1,'H':1},{'L':-.1,'M':.5,'H':.6},
    {'L':.5,'M':.5}, {'L':float('nan'),'M':0,'H':1}])
def test_mass_errors_not_silently_renormalized(mass):
    with pytest.raises(ValueError): schedule_proposal(mass)


def test_same_seed_same_sampler_trace_and_no_global_rng_consumption():
    import numpy as np
    import torch
    py = random.getstate(); np_before = np.random.get_state(); torch_before = torch.get_rng_state().clone()
    first = sampler(); second = sampler()
    a, b = [], []
    for steps in [19999,2,39999,1,30000,30000]:
        a.append((first.begin_episode(),first.end_episode(steps)))
        b.append((second.begin_episode(),second.end_episode(steps)))
    assert a == b and random.getstate() == py
    np_after = np.random.get_state()
    assert np_before[0] == np_after[0] and np.array_equal(np_before[1],np_after[1]) and np_before[2:] == np_after[2:]
    assert torch.equal(torch_before,torch.get_rng_state())
    assert all(not selection['execution_authorized'] for selection,_ in a)


def test_episode_not_truncated_at_stage_or_budget_boundary_and_delay_not_elapsed_stage_length():
    s = sampler(); initial = s.begin_episode()
    with pytest.raises(ValueError): s.begin_episode()
    assert initial['collection_stage'] == 0
    finish = s.end_episode(20007)
    assert finish['crossed_boundaries'] == [{'boundary':20000,'actual_next_reset_step':20007,'overrun':7}]
    next_one = s.begin_episode()
    assert next_one['collection_stage'] == 1 and next_one['stage_boundary_overrun'] == 7
    assert next_one['sampling_mass'] == {'L':.2,'M':.2,'H':.6}
    s.end_episode(5)
    assert s.begin_episode()['stage_boundary_overrun'] == 7
    finish = s.end_episode(100001)
    assert finish['request_budget_overrun'] == 13
    with pytest.raises(ValueError): s.begin_episode()


def test_mix_mass_stays_fixed_while_curriculum_changes_at_legal_boundary():
    s = sampler('mix'); first = s.begin_episode(); s.end_episode(60001); last = s.begin_episode()
    assert first['sampling_mass'] == last['sampling_mass']
    assert last['collection_stage'] == 2
    assert first['sampling_mass'] != {'L':11/32,'M':10/32,'H':11/32}


@pytest.mark.parametrize('fault',['empty_layer','duplicate_id','duplicate_seed','bad_level','extra_input'])
def test_invalid_roster_cannot_become_a_sampler(fault):
    tasks = roster()
    if fault == 'empty_layer': tasks = [t for t in tasks if t['level'] != 'H']
    elif fault == 'duplicate_id': tasks[1]['task_id'] = tasks[0]['task_id']
    elif fault == 'duplicate_seed': tasks[1]['simulator_seed'] = tasks[0]['simulator_seed']
    elif fault == 'bad_level': tasks[0]['level'] = 'C0'
    else: tasks[0]['future_reward'] = 2.
    with pytest.raises(ValueError): sampler(tasks=tasks)


@pytest.mark.parametrize('steps',[0,-1,True,2.5])
def test_only_positive_actual_transition_counts_advance_clock(steps):
    s = sampler(); s.begin_episode()
    with pytest.raises(ValueError): s.end_episode(steps)


def selection(level, step=0, stage=0):
    return {'task_id':'task_'+level,'simulator_seed':11000+ord(level),'level':level,
            'episode_index':step,'collection_stage':stage,'actual_start_step':step,'execution_authorized':False}


def test_audit_tracks_collection_replay_and_actual_sample_lag_with_all_outcomes():
    audit = ExposureAudit(capacity=10,recent_window=2)
    audit.begin_episode(selection('L')); audit.record_transition(0,1); audit.record_transition(1,2); audit.end_episode('collision')
    audit.begin_episode(selection('H',2,1)); audit.record_transition(2,3); audit.record_transition(3,4); audit.end_episode('deadline')
    metadata = audit.record_minibatch([0,1,0])
    assert [m['level'] for m in metadata] == ['L','L','L']
    report = audit.snapshot()
    assert report['collected_transition_counts'] == {'L':2,'M':0,'H':2}
    assert report['recent_transition_counts'] == {'L':0,'M':0,'H':2}
    assert report['replay_slot_counts'] == {'L':2,'M':0,'H':2}
    assert report['minibatch_transition_counts'] == {'L':3,'M':0,'H':0}
    assert report['minibatch_transitions_by_stage_and_level']['0']['L'] == 3
    assert report['collected_transitions_by_stage_and_level']['1']['H'] == 2
    assert report['collected_transitions_by_task'] == {'task_L':2,'task_H':2}
    assert [e['outcome'] for e in report['completed_episodes']] == ['collision','deadline']
    assert not report['replay_resampled_or_cleared'] and not report['metadata_in_network_inputs']


def test_audit_observes_overwritten_slots_without_changing_replay_or_losing_collection_totals():
    audit = ExposureAudit(capacity=2,recent_window=3)
    audit.begin_episode(selection('L')); audit.record_transition(0,1); audit.record_transition(1,2); audit.end_episode('arrival')
    audit.begin_episode(selection('M',2,1)); audit.record_transition(0,3); audit.end_episode('invalid_run')
    assert audit.snapshot()['replay_slot_counts'] == {'L':1,'M':1,'H':0}
    assert audit.snapshot()['collected_transition_counts'] == {'L':2,'M':1,'H':0}
    assert audit.record_minibatch([0])[0]['actual_environment_step'] == 3


def test_audit_rejects_reset_pseudo_transitions_unknown_samples_and_bad_alignment():
    audit = ExposureAudit(capacity=2)
    with pytest.raises(ValueError): audit.record_transition(0,1)
    with pytest.raises(ValueError): audit.begin_episode(selection('L',1))
    audit.begin_episode(selection('L'))
    with pytest.raises(ValueError): audit.begin_episode(selection('H'))
    for index,step in [(0,0),(0,2),(2,1),(True,1)]:
        with pytest.raises(ValueError): audit.record_transition(index,step)
    for indices in [[],[0],[True]]:
        with pytest.raises(ValueError): audit.record_minibatch(indices)
    with pytest.raises(ValueError): audit.end_episode('arrival')


@pytest.mark.parametrize('fault',[None,'old_changed','old_missing','new_missing','unknown_extra'])
def test_frozen_source_allowlist_has_one_explicit_new_module_only(fault):
    sealed = {'old.py':'frozen'}
    current = {**sealed,'worktree:src/prediction_rl/curriculum/interaction_intensity.py':'new'}
    if fault == 'old_changed': current['old.py'] = 'edited'
    elif fault == 'old_missing': del current['old.py']
    elif fault == 'new_missing': del current['worktree:src/prediction_rl/curriculum/interaction_intensity.py']
    elif fault == 'unknown_extra': current['unreviewed.py'] = 'extra'
    if fault is None: cli.verify_prior_hashes(current,sealed)
    else:
        with pytest.raises(ValueError): cli.verify_prior_hashes(current,sealed)


def test_sealed_real_training_side_data_if_present():
    path = ROOT/'artifacts/ic/ic02f_audit_v1/roots.json'
    if not path.exists(): pytest.skip('Local immutable training audit not distributed with source')
    source = json.loads(path.read_text())['rows']; result = proposal(source,PROTOCOL)
    assert result['summary']['proposed_counts'] == {'L':11,'M':10,'H':11}
    assert result['threshold_proposal']['I_low'] == pytest.approx(.3976121680515756)
    assert result['threshold_proposal']['I_high'] == pytest.approx(2.6839018875244487)
    assert result['summary']['coverage_gate']
    assert result['summary']['independent_D_and_coverage_by_level']['H']['maximum_target_closing_speed_mps']['quantiles']['0.5'] == pytest.approx(9.011373519897461)
