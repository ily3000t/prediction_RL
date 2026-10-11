from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'),str(ROOT/'tools')]
from prediction_rl.curriculum.task_rule_proposal import PROTOCOL, proposal, proposed_decision, task_features, validate_protocol
import propose_curriculum_task_rules as cli


def family(seed=11000,score=1.,extra=0.,gap=2.,closing=1.):
    actors = ['front','rear','cross']
    roles = {'front':['target_front'],'rear':['target_rear'],'cross':['nearest_cross_path_arrival']}
    context = {'actor_ids':actors,'roles_by_actor':roles,
               'coverage':{'key_actor_missing_slots':[],'history_frames':11,'omitted_neighbor_count':3},
               'difficulty':{'critical_actor_count':3}}
    rows = []
    for target in PROTOCOL['checkpoints']:
        values = {'front':score,'rear':score/2,'cross':extra}
        window = {'steps':14,'sets':{'all_critical':{'status':'available'}},
            'per_actor':{a:{'roles':roles[a],'response':{'actor_ids':[a],'horizon_steps':14,'status':'available',
                'I_dense_with_frozen_IC01_scales':value,'interaction':{'full_common_horizon':True}}} for a,value in values.items()}}
        evidence = {'task_id':f't{seed}','simulator_seed':seed,'target':target,'task_level':None,
            'classification_applied':False,'training_ready':False,'episode_id':f'ep{seed}',
            'root_context':deepcopy(context),'repeat_noise_with_frozen_IC01_scales':0.}
        relation = {'target_front':{'actor_id':'front','route_projected_gap_m':gap,'closing_speed_mps':closing,'path_relation':'cross_path'},
                    'target_rear':{'actor_id':'rear','route_projected_gap_m':gap+1,'closing_speed_mps':0.,'path_relation':'cross_path'}}
        rows.append({'task_id':f't{seed}','simulator_seed':seed,'target':target,'root_id':f'{seed}-{target}',
            'original_evidence':evidence,'common_prefix':window,'source_request_hash':'frozen',
            'road_relations':{'target_relations':relation,'original_difficulty_unchanged':{'critical_actor_count':3}}})
    return rows


def feature(**kwargs):
    return task_features(family(**kwargs),PROTOCOL)


THRESHOLDS = {'I_low':1.,'I_high':3.,'closing_low':2.,'repeat_noise_floor':0.}


def test_readable_config_exactly_matches_single_proposal():
    assert validate_protocol(json.loads((ROOT/cli.CONFIG).read_text())) == PROTOCOL


@pytest.mark.parametrize('field,value',[('window_steps',10),('anchor','reset'),('scope','test'),
    ('task_layer_assignment_authorized',True),('ddpg_training_authorized',True),
    ('target_distribution_pi',{}),('minimum_unique_tasks_per_level',1),('unknown',0)])
def test_unknown_or_expanded_proposal_rejected(field,value):
    c = deepcopy(PROTOCOL); c[field] = value
    with pytest.raises(ValueError): validate_protocol(c)


def test_maximum_of_actor_means_preserves_extra_actor_response_and_ties():
    t = feature(score=.03,extra=1.5)
    assert t['I_peak'] == 1.5 and t['peak_actor_ids'] == ['cross']
    assert t['peak_actor_roles'] == {'cross':['nearest_cross_path_arrival']}
    t = feature(score=1.5,extra=1.5)
    assert t['peak_actor_ids'] == ['cross','front']
    assert t['per_actor_I']['rear'] == .75


def test_only_anchor_defines_task_no_cherry_picked_early_zero():
    rows = family(score=3.)
    for row in rows[:2]:
        for value in row['common_prefix']['per_actor'].values():
            value['response']['I_dense_with_frozen_IC01_scales'] = 0.
    assert task_features(rows,PROTOCOL)['I_peak'] == 3.


@pytest.mark.parametrize('fault',['order','task','seed','live_label','future_role','window','ego','count','nonfinite','partial'])
def test_mutated_or_invalid_root_contract_rejected(fault):
    rows = family(); r = rows[2]
    if fault == 'order': rows.reverse()
    elif fault == 'task': rows[0]['task_id'] = 'other'
    elif fault == 'seed': rows[0]['simulator_seed'] += 1
    elif fault == 'live_label': r['original_evidence']['task_level'] = 'C1'
    elif fault == 'future_role': r['common_prefix']['per_actor']['front']['roles'] = ['future_winner']
    elif fault == 'window': r['common_prefix']['steps'] = 15
    elif fault == 'ego': r['original_evidence']['root_context']['actor_ids'].append('ego')
    elif fault == 'count': r['road_relations']['original_difficulty_unchanged']['critical_actor_count'] = 2
    elif fault == 'nonfinite': r['common_prefix']['per_actor']['front']['response']['I_dense_with_frozen_IC01_scales'] = float('nan')
    else: r['common_prefix']['per_actor']['front']['response']['interaction']['full_common_horizon'] = False
    with pytest.raises(ValueError): task_features(rows,PROTOCOL)


def test_unavailable_actor_is_not_zero_or_dropped():
    rows = family(); response = rows[2]['common_prefix']['per_actor']['cross']['response']
    response.update(status='unavailable',I_dense_with_frozen_IC01_scales=None)
    t = task_features(rows,PROTOCOL)
    assert t['I_peak'] is None and not t['I_available']
    decision = proposed_decision(t,THRESHOLDS,PROTOCOL)
    assert decision['proposed_level'] == 'C2' and decision['reason'] == 'unavailable_response'


def test_C0_requires_low_response_and_both_weak_constraint_components():
    assert proposed_decision(feature(score=1.,gap=.01,closing=2.),THRESHOLDS,PROTOCOL)['proposed_level'] == 'C0'
    for t in [feature(score=1.,gap=0.), feature(score=1.,gap=-1.), feature(score=1.,closing=2.01),feature(score=2.)]:
        assert proposed_decision(t,THRESHOLDS,PROTOCOL)['proposed_level'] == 'C2'


def test_route_ttc_is_never_a_decision_variable():
    t = feature(score=1.,gap=2.,closing=1.)
    t['difficulty_components']['minimum_ttc_route_proxy_s'] = 0.
    assert proposed_decision(t,THRESHOLDS,PROTOCOL)['proposed_level'] == 'C0'


def test_C1_high_response_boundary_and_noise_gate():
    assert proposed_decision(feature(score=3.),THRESHOLDS,PROTOCOL)['proposed_level'] == 'C1'
    thresholds = {**THRESHOLDS,'repeat_noise_floor':3.}
    assert proposed_decision(feature(score=3.),thresholds,PROTOCOL)['proposed_level'] == 'C2'
    thresholds = {**THRESHOLDS,'I_low':0.,'I_high':0.}
    assert proposed_decision(feature(score=0.,gap=-1.),thresholds,PROTOCOL)['proposed_level'] == 'C2'


def test_C0_priority_resolves_quantile_ties_without_task_ids():
    thresholds = {**THRESHOLDS,'I_low':1.,'I_high':1.}
    decision = proposed_decision(feature(score=1.),thresholds,PROTOCOL)
    assert decision['matches_C0'] and decision['matches_C1'] and decision['proposed_level'] == 'C0'


def test_missing_target_gap_is_none_not_infinite_or_easy():
    rows = family(); rows[2]['road_relations']['target_relations']['target_rear'] = {'actor_id':None,'path_relation':'absent'}
    t = task_features(rows,PROTOCOL)
    assert t['minimum_projected_opportunity_gap_m'] is None
    assert proposed_decision(t,THRESHOLDS,PROTOCOL)['proposed_level'] == 'C2'


def test_proposal_coverage_failure_is_a_normal_complete_result_no_live_labels():
    rows = [r for i,s in enumerate(PROTOCOL['task_seeds']) for r in family(s,score=float(i),gap=-1.,closing=2.)]
    snapshot = deepcopy(rows); r = proposal(rows,PROTOCOL)
    assert r['summary']['proposed_counts']['C0'] == 0
    assert not r['summary']['coverage_gate'] and not r['summary']['fixed_three_level_schedule_mathematically_schedulable']
    assert r['summary']['proposed_Mix_nominal_layer_mass']['C0'] == .2
    assert sum(r['summary']['proposed_uniform_task_layer_mass'].values()) == pytest.approx(1.)
    assert sum(r['summary']['proposed_Mix_nominal_layer_mass'].values()) == pytest.approx(1.)
    assert all(t['task_level'] is None and not t['classification_applied'] and not t['execution_authorized'] for t in r['tasks'])
    assert not r['summary']['training_ready'] and r['summary']['target_distribution_pi'] is None
    assert len(r['tasks']) == 32 and rows == snapshot


@pytest.mark.parametrize('fault',['missing_task','duplicate_root','wrong_seed','duplicate_episode'])
def test_full_roster_and_unique_source_episodes_required(fault):
    rows = [r for s in PROTOCOL['task_seeds'] for r in family(s)]
    if fault == 'missing_task': rows = rows[3:]
    elif fault == 'duplicate_root': rows[2] = deepcopy(rows[1])
    elif fault == 'wrong_seed': rows[-1]['simulator_seed'] = 1000
    else:
        for row in rows[3:6]: row['original_evidence']['episode_id'] = rows[0]['original_evidence']['episode_id']
    with pytest.raises(ValueError): proposal(rows,PROTOCOL)


def test_row_order_cannot_affect_terciles_or_membership():
    rows = [r for i,s in enumerate(PROTOCOL['task_seeds']) for r in family(s,score=float(i),closing=float(i))]
    assert proposal(rows,PROTOCOL) == proposal(list(reversed(rows)),PROTOCOL)


@pytest.mark.parametrize('fault',[None,'changed','missing','unexpected','missing_new'])
def test_prior_source_bytes_preserved_with_exact_single_new_module(fault):
    old = {'frozen_science.py':'oldhash'}
    current = {**old,'worktree:src/prediction_rl/curriculum/task_rule_proposal.py':'newhash'}
    if fault == 'changed': current['frozen_science.py'] = 'changed'
    elif fault == 'missing': del current['frozen_science.py']
    elif fault == 'unexpected': current['unreviewed_module.py'] = 'extra'
    elif fault == 'missing_new': del current['worktree:src/prediction_rl/curriculum/task_rule_proposal.py']
    if fault is None: cli.verify_prior_hashes(current,old)
    else:
        with pytest.raises(ValueError): cli.verify_prior_hashes(current,old)
