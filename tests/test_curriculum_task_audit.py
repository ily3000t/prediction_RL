from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'tools'), str(ROOT/'tests')]
from prediction_rl.curriculum.task_audit import (actor_sets, dense_response, longest_dense_prefix,
    neighbor_timelines, rank_comparison, relation_audit, summarize, window_audit)
from prediction_rl.curriculum.interaction import root_context
from prediction_rl.data.actor_adapter import build_actor_inputs
from prediction_rl.data.merge_geometry import MergeGeometry
from test_curriculum_interaction import example
import audit_curriculum_tasks as cli


@pytest.fixture
def metric():
    return json.loads((ROOT/'configs/development/ic01_train_interaction_v1.json').read_text())


@pytest.fixture
def geometry():
    return MergeGeometry(ROOT/'RL-MPC-LaneMerging-master/merge.net.xml')


def lines(steps=25):
    # Third root-selected actor is unresponsive; only neighbors, absolute coordinates.
    return {cid: {t: {'a': (100.+t+cid, 5.+cid), 'b': (200.+t+2*cid, 10.+2*cid),
                     'extra': (300.+t, 15.)} for t in range(steps)} for cid in range(5)}


SCALES = {'dv': 1., 'ds': 1.}
CONTEXT = {'actor_ids': ['a','b','extra'], 'roles_by_actor': {
    'a':['target_front','lowest_same_path_ttc'], 'b':['target_rear'], 'extra':['nearest_cross_path_arrival']}}


def test_fixed_windows_and_frozen_config_are_explicit():
    config = cli.validate_protocol(json.loads((ROOT/cli.CONFIG).read_text()))
    assert config['fixed_window_grid_steps'] == [5,10,15,25]
    assert config['training_ready'] is False and config['task_classifier'] is None


@pytest.mark.parametrize('field,value', [('tasks',31),('scope','test'),('fixed_window_grid_steps',[14]),
    ('task_layer_assignment_authorized',True),('new_sumo_sessions_authorized',True),
    ('predictor_inference_authorized',True),('ddpg_training_authorized',True),('unknown',0)])
def test_protocol_cannot_expand_or_silently_ignore_unknown_fields(field,value):
    config = deepcopy(cli.PROTOCOL); config[field] = value
    with pytest.raises(ValueError): cli.validate_protocol(config)


def test_deduplication_does_not_weight_one_actor_twice():
    assert actor_sets(CONTEXT) == {'all_critical':['a','b','extra'], 'target_front_rear':['a','b']}


def test_extra_unresponsive_actor_dilutes_mean_not_target_response(metric):
    audit = window_audit(lines(),CONTEXT,10,metric,SCALES)
    all_i = audit['sets']['all_critical']['I_dense_with_frozen_IC01_scales']
    target_i = audit['sets']['target_front_rear']['I_dense_with_frozen_IC01_scales']
    assert all_i == pytest.approx(target_i*2/3)
    assert audit['per_actor']['extra']['response']['I_dense_with_frozen_IC01_scales'] == 0
    assert audit['sets']['all_critical']['interaction']['common_cells'] == 10*10*3


@pytest.mark.parametrize('fault',['actor','step','short'])
def test_missing_cells_make_long_window_unavailable_not_zero(metric,fault):
    data = lines()
    if fault == 'actor': del data[4][9]['a']
    elif fault == 'step': del data[4][9]
    else: data[4] = {t:v for t,v in data[4].items() if t<9}
    assert longest_dense_prefix(data,['a','b'],25) == 9
    value = dense_response(data,['a','b'],10,metric,SCALES)
    assert value['status'] == 'unavailable' and value['I_dense_with_frozen_IC01_scales'] is None
    assert value['missing_candidate_actor_time_cells'] > 0 and value['interaction'] is None
    assert dense_response(data,['a','b'],5,metric,SCALES)['status'] == 'available'


def test_missing_extra_actor_does_not_censor_target_set(metric):
    data = lines(); del data[4][4]['extra']
    result = window_audit(data,CONTEXT,10,metric,SCALES)
    assert result['sets']['all_critical']['status'] == 'unavailable'
    assert result['sets']['target_front_rear']['status'] == 'available'


def test_empty_set_is_uncomputable(metric):
    r = dense_response(lines(),[],5,metric,SCALES)
    assert r['status'] == 'unavailable' and r['unavailable_reason'] == 'empty_actor_set'
    assert r['I_dense_with_frozen_IC01_scales'] is None


@pytest.mark.parametrize('fault',['ego','missing_candidate','out_of_horizon','nan'])
def test_invalid_timelines_or_ego_response_rejected(fault):
    data = lines(); actors = ['a']
    if fault == 'ego': actors = ['ego']
    elif fault == 'missing_candidate': del data[4]
    elif fault == 'out_of_horizon': data[4][25] = data[4][0]
    else: data[4][0]['a'] = (float('nan'),0.)
    with pytest.raises(ValueError): longest_dense_prefix(data,actors,25)


@pytest.mark.parametrize('steps', [0,26,True,2.5])
def test_invalid_window_rejected(metric,steps):
    with pytest.raises(ValueError): dense_response(lines(),['a'],steps,metric,SCALES)


@pytest.mark.parametrize('scale', [0.,-1.,float('inf'),float('nan')])
def test_invalid_normalization_rejected(metric,scale):
    with pytest.raises(ValueError): dense_response(lines(),['a'],5,metric,{'dv':scale,'ds':1.})


def test_audit_is_pure_and_keeps_original_evidence(metric):
    data, context = lines(), deepcopy(CONTEXT); snapshot = deepcopy((data,context,metric,SCALES))
    window_audit(data,context,10,metric,SCALES)
    assert (data,context,metric,SCALES) == snapshot


def test_rank_comparison_ties_reversals_and_missing_values():
    r = rank_comparison({'a':1.,'b':1.,'c':3.,'d':None}, {'a':3.,'b':3.,'c':1.,'d':4.})
    assert r['matched_tasks'] == 3 and r['spearman'] == pytest.approx(-1.)
    assert r['pair_order_reversals'] == r['comparable_non_tied_pairs'] == 2
    assert r['maximum_absolute_rank_change'] == 2
    assert rank_comparison({'a':0.},{'a':0.})['spearman'] is None
    assert rank_comparison({}, {})['matched_tasks'] == 0


def test_terminal_step_never_becomes_dense_future(geometry,metric):
    h,p,b = example(geometry,metric,responsive=True,terminal_step=2)
    timeline = neighbor_timelines(p,b,geometry,metric)
    assert longest_dense_prefix(timeline,['a','b'],25) == 2
    assert dense_response(timeline,['a','b'],5,metric,SCALES)['status'] == 'unavailable'


@pytest.mark.parametrize('fault',['split','plan','candidate'])
def test_neighbor_reader_rejects_altered_family(geometry,metric,fault):
    h,p,b = example(geometry,metric)
    if fault == 'split': p['split'] = 'test'
    elif fault == 'plan': p['candidates'][0]['labels']['requested_plan'][0] += 1
    else: p['candidates'].append(deepcopy(p['candidates'][0]))
    with pytest.raises(ValueError): neighbor_timelines(p,b,geometry,metric)


def test_cross_path_negative_gap_is_not_same_path_overlap(geometry,metric):
    def state(lane,position,world,speed):
        return {'lane_id':lane,'lane_position_m':position,'position':world,'speed':speed,
                'acceleration':0.,'length_m':5.,'width_m':1.8}
    root = {'simulation_time_s':20.,'vehicles': {
        'ego':state('ramp_0',geometry.lengths['ramp_0']-10.,[0.,0.],10.),
        'a':state('highwayrear_0',geometry.lengths['highwayrear_0']-10.,[40.,0.],8.),
        'b':state('highwayrear_0',geometry.lengths['highwayrear_0']-20.,[50.,0.],12.)}}
    h = {'history':[root],'inputs':build_actor_inputs([root],geometry)}
    context = root_context(h,geometry,metric)
    result = relation_audit(h,{'root_context':context},geometry)
    relation = next(d for d in result['target_relations'].values() if d['actor_id']=='a')
    assert relation['path_relation'] == 'cross_path'
    assert relation['route_projected_gap_m'] < 0 and relation['old_route_ttc_proxy_s'] == 0
    assert relation['same_path_gap_m'] is relation['same_path_cv_ttc_s'] is None
    assert relation['front_bumper_euclidean_distance_m'] == 40
    assert not relation['route_proxy_is_immediate_collision_measure']
    assert result['original_difficulty_unchanged'] == context['difficulty']


def test_same_path_metrics_and_absent_roles_remain_explicit(geometry,metric):
    h,p,b = example(geometry,metric)
    context = root_context(h,geometry,metric)
    r = relation_audit(h,{'root_context':context},geometry)
    assert r['target_relations']['target_front']['path_relation'] == 'same_path'
    assert r['target_relations']['target_front']['same_path_gap_m'] == 10
    context['target_relations']['target_rear']['actor_id'] = None
    r = relation_audit(h,{'root_context':context},geometry)
    assert r['target_relations']['target_rear']['path_relation'] == 'absent'


def test_source_verification_rejects_report_or_receipt_mismatch(tmp_path,monkeypatch):
    directory = tmp_path/'source'; directory.mkdir(); monkeypatch.setattr(cli,'ROOT',tmp_path)
    job = {'id':'t11000'}; row = {'job':job}
    post = {'parent_rng_unchanged':True,'immutable_input_postflight_passed':True}
    q = {'protocol':{},'git_commit_sha':'source','input_hashes':{},'jobs':[job]}
    report = {'request_hash':cli.digest(q),'status':'complete','engineering_gate':True,
              'actual_tasks':1,'training_ready':False,'predictors_loaded':False,
              'parent_postflight':post,'tasks':[row]}
    for name,value in [('request.json',q),('report.json',report),('postflight.json',post)]:
        (directory/name).write_text(json.dumps(value))
    (directory/'t11000').mkdir(); (directory/'t11000/complete.json').write_text('{}')
    hashes = {name:cli.file_hash(directory/name) for name in ('request.json','report.json','postflight.json')}
    monkeypatch.setattr(cli,'verify_episode',lambda *a:{'metadata':row})
    assert cli.verify_source(directory,hashes,{},'source',{})[0] == q
    monkeypatch.setattr(cli,'verify_episode',lambda *a:{'metadata':{}})
    with pytest.raises(ValueError,match='sealed metadata'): cli.verify_source(directory,hashes,{},'source',{})
    (directory/'report.json').write_text('{}')
    with pytest.raises(ValueError,match='changed'): cli.verify_source(directory,hashes,{},'source',{})


def test_summary_retains_unavailable_windows_without_task_selection(metric):
    context = deepcopy(CONTEXT)
    context.update(difficulty={'critical_actor_count':3},target_relations={})
    data = lines(14)
    original = {'I_with_frozen_IC01_scales':8.,'interaction':{'full_common_horizon':False},'root_context':context}
    row = {'task_id':'t1','target':'approach','original_evidence':original,
           'windows':{str(n):window_audit(data,context,n,metric,SCALES) for n in [5,10,15,25]},
           'common_prefix':window_audit(data,context,14,metric,SCALES),
           'road_relations':{'target_relations':{},'original_difficulty_unchanged':context['difficulty']}}
    snapshot = deepcopy(row); summary = summarize([row],[5,10,15,25],14)
    assert summary['tasks'] == 1 and summary['by_target']['approach']['fixed_window_grid']['25']['all_critical']['unavailable'] == 1
    assert row == snapshot and 'task_level' not in row
