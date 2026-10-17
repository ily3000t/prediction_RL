from copy import deepcopy
import inspect
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'),str(ROOT/'tools')]
import validate_rcci_v2_cohort as cli
from prediction_rl.evaluation.rcci_v2_cohort import (
    SETTINGS,MODES,project_predictions,compare_native,mechanism,aggregate,cut,
)
from prediction_rl.features.road_consequence_geometry import RoadConsequenceMap
from test_road_consequences import scene,state
from test_rcci_frozen_audit import branch_from_forecast


@pytest.fixture
def road(): return RoadConsequenceMap(ROOT/'RL-MPC-LaneMerging-master/merge.net.xml')


def prepared(road):
    h,x,n,e = scene(road,{'ego':state(road,'ramp_0',195.,speed=15.),
                         'main':state(road,position=195.,speed=15.)})
    b = branch_from_forecast(road,h,x,n,e)
    return h,x,n,e,b


def test_exact_config_and_fixed_full_budget():
    assert json.loads((ROOT/cli.CONFIG_PATH).read_text()) == SETTINGS
    assert SETTINGS['expected_roots'] == {'train':713,'validation':177}
    assert SETTINGS['max_ensemble_forward_calls'] == 2*890
    assert SETTINGS['max_member_forward_calls'] == 3*1780
    assert not SETTINGS['simulation'] and not SETTINGS['training'] and not SETTINGS['test_access']


def test_perfect_matched_native_forecasts_and_fixed_denominators(road):
    h,x,n,e,b = prepared(road)
    p = project_predictions(h,{m:n for m in MODES},e,road)
    r = compare_native(h,b,0,p,road)
    assert r['common_follow_points'] == r['native_follow_points'] == 26
    assert r['common_cross_points'] == 26
    assert max(map(abs,r['ego_progress_errors_m'])) < 1e-8
    for mode in MODES:
        assert r['cross'][mode]['counts']['actor_entry_tp'] == 1
        for kind in ('following','cross'):
            assert all(abs(v) < 1e-8 for values in r[kind][mode]['errors'].values() for v in values)
        for v in r['fixed_horizons'].values():
            assert v['total_candidates'] == v['native_observed'] == v['joint_supported'] == 1


def test_common_prefix_never_recovers_and_fixed_5s_not_faked(road):
    h,x,n,e,b = prepared(road); damaged = n.copy()
    damaged[0,0,7,1] += 4.
    p = project_predictions(h,{'ordinary':n,'conditional':damaged},e,road)
    r = compare_native(h,b,0,p,road)
    assert r['common_follow_points'] == r['common_cross_points'] == 8  # 1.4s + root
    assert r['fixed_horizons']['1.0']['joint_supported'] == 1
    for time in ('3.0','5.0'):
        assert r['fixed_horizons'][time]['native_observed'] == 1
        assert r['fixed_horizons'][time]['joint_supported'] == 0
        assert not r['fixed_horizons'][time]['methods']
    for kind in ('following','cross'):
        for key in r[kind]['ordinary']['errors']:
            assert len(r[kind]['ordinary']['errors'][key]) == len(r[kind]['conditional']['errors'][key])


def test_no_native_future_support_does_not_become_zero_error(road):
    h,x,n,e,b = prepared(road); b['future_metadata']['trajectory_usable'][0] = False
    p = project_predictions(h,{m:n for m in MODES},e,road); r = compare_native(h,b,0,p,road)
    assert r['common_follow_points'] == r['common_cross_points'] == 1
    assert r['ego_progress_errors_m'] == []
    for m in MODES:
        assert r['following'][m]['errors']['front_combined_min_gap_m'] == []
        assert r['cross'][m]['errors'] == {}
    assert all(v['joint_supported'] == 0 for v in r['fixed_horizons'].values())


def test_one_missing_actor_censors_entire_follow_pool(road):
    h,x,n,e = scene(road,{'ego':state(road),'front':state(road,position=170.),
                         'rear':state(road,position=130.)})
    b = branch_from_forecast(road,h,x,n,e)
    b['future_metadata']['frames'][2]['vehicles'].pop('rear')
    p = project_predictions(h,{m:n for m in MODES},e,road); r = compare_native(h,b,0,p,road)
    assert r['native_follow_points'] == r['common_follow_points'] == 3
    assert all(v['native_observed'] == 0 for v in r['fixed_horizons'].values())


def test_ego_reference_error_is_separate_from_neighbor_prediction_error(road):
    h,x,n,e = scene(road,{'ego':state(road),'front':state(road,position=170.)})
    b = branch_from_forecast(road,h,x,n,e)
    changed = e.copy(); changed[:,:,0] += np.arange(1,26)*.2
    p = project_predictions(h,{m:n for m in MODES},changed,road); r = compare_native(h,b,0,p,road)
    for m in MODES:
        errors = r['following'][m]['errors']
        assert abs(errors['front_combined_min_gap_m'][0]) > 1.
        assert errors['front_neighbor_only_min_gap_m'] == pytest.approx([0.])
    assert max(r['ego_progress_errors_m']) == pytest.approx(5.)


def test_event_miss_not_hidden_in_time_mae(road):
    h,x,n,e,b = prepared(road); stopped = n.copy(); stopped[:,:,:,2] = 0
    stopped[:,:,:,:2] = n[:,:,:1,:2]
    p = project_predictions(h,{'ordinary':n,'conditional':stopped},e,road)
    r = compare_native(h,b,0,p,road)
    assert r['cross']['conditional']['counts']['actor_entry_fn'] == 1
    assert not r['cross']['conditional']['errors'].get('actor_entry_s')
    assert r['cross']['ordinary']['counts']['actor_entry_tp'] == 1


def test_root_occupancy_is_not_forecast_entry_success(road):
    h,x,n,e = scene(road,{'ego':state(road,':mergenode_1_0',20.,speed=1.),
                         'main':state(road,':mergenode_0_0',20.,speed=1.)})
    b = branch_from_forecast(road,h,x,n,e); p = project_predictions(h,{m:n for m in MODES},e,road)
    r = compare_native(h,b,0,p,road)
    for m in MODES:
        c = r['cross'][m]['counts']
        assert c['actor_entry_observed_root_or_passed'] == c['actor_occupies_entire_support'] == 1
        assert c['first_overlap_equals_ego_entry'] == c['overlap_saturated_at_T'] == 1
        assert not c.get('actor_entry_tp')


def test_mechanism_zero_for_identical_predictions_and_no_missing_zero_subtraction(road):
    h,x,n,e,b = prepared(road); p = project_predictions(h,{m:n for m in MODES},e,road)
    r = mechanism(h,p,road)
    for m in MODES:
        for cells in r[m]['variants'].values():
            for cell in cells:
                assert all(v['delta'] is None or abs(v['delta']) < 1e-8 for v in cell.values())
                assert not any(v['availability_changed'] for v in cell.values())
        assert r[m]['replacement_may_be_ood']


def test_neighbor_increment_is_not_ego_plan_difference(road):
    h,x,n,e = scene(road,{'ego':state(road),'front':state(road,position=170.)})
    changed = n.copy(); changed[4,0,:,0] -= 1.
    p = project_predictions(h,{'ordinary':n,'conditional':changed},e,road); r = mechanism(h,p,road)
    assert r['ordinary']['variants']['conditional_increment'][4]['front_gap_m']['delta'] == pytest.approx(0.)
    assert r['conditional']['variants']['conditional_increment'][4]['front_gap_m']['delta'] == pytest.approx(-1.)
    assert r['conditional']['variants']['ego_plan'][4]['front_gap_m']['delta'] == pytest.approx(0.)
    assert r['conditional']['variants']['neighbor_response'][4]['front_gap_m']['delta'] == pytest.approx(-1.)


def test_larger_future_gap_cannot_change_a_root_minimum(road):
    h,x,n,e = scene(road,{'ego':state(road),'front':state(road,position=170.)})
    changed = n.copy(); changed[4,0,:,0] += 2.
    r = mechanism(h,project_predictions(h,{'ordinary':n,'conditional':changed},e,road),road)
    assert r['conditional']['variants']['conditional_increment'][4]['front_gap_m']['delta'] == pytest.approx(0.)


def test_mechanism_all_candidates_models_use_same_shortest_support(road):
    h,x,n,e,b = prepared(road); damaged = n.copy(); damaged[4,0,4,1] += 4.
    p = project_predictions(h,{'ordinary':n,'conditional':damaged},e,road); r = mechanism(h,p,road)
    assert r['ordinary']['common_follow_horizon_s'] == pytest.approx(.8)
    assert r['conditional']['common_follow_horizon_s'] == pytest.approx(.8)
    assert r['ordinary']['common_cross_horizon_s'] == pytest.approx(.8)
    with pytest.raises(ValueError): cut(p['conditional'][4][1]['main'],26)


def test_future_poison_cannot_reach_feature_or_mechanism_inputs(road):
    h,x,n,e,b = prepared(road); expected = cli.evaluate_features(h,{m:n for m in MODES},e,road)
    hh = deepcopy(h); hh[0].update(future_actor_ids=['oracle'],labels=float('nan'),termination_mask=True)
    actual = cli.evaluate_features(hh,{m:n for m in MODES},e,road)
    assert expected[1:] == actual[1:]
    assert set(inspect.signature(cli.infer).parameters) == {'history','road','ego_geometry','cfg','models'}
    text = inspect.getsource(cli.run)
    assert text.index('evaluate_features(') < text.index("pack = read(hp.parent/'labels.json')")


def test_aggregate_separates_design_episodes_and_equal_weights_episodes(road):
    h,x,n,e,b = prepared(road); p,cov,fail,mech = cli.evaluate_features(h,{m:n for m in MODES},e,road)
    row = {'root_id':'r','episode_id':'one','split':'train','design_episode_reuse':False,
        'coverage':cov,'prefix_failures':fail,'mechanism':mech,'omitted_actor_count':0,
        'branches':[compare_native(h,b,c,p,road) for c in range(5)]}
    a = deepcopy(row); a['branches'][0]['cross']['ordinary']['errors']['combined_overlap_s'] = [2.]
    d = deepcopy(row); d.update(episode_id='design',design_episode_reuse=True)
    r = aggregate([row,a,d])
    g = r['train/other_existing_episodes']
    assert g['roots'] == 2 and g['unique_episodes'] == 1
    assert r['train/design_episode_reuse']['roots'] == 1
    assert r['validation/other_existing_episodes']['roots'] == 0
    assert g['methods']['ordinary']['episode_errors']['cross']['combined_overlap_s'][
        'equal_weight_episode_mae']['count'] == 1
    assert not g['independent_confirmation']


def test_unknown_fields_and_test_scope_fail_before_opening_data(monkeypatch):
    bad = deepcopy(SETTINGS); bad['test_access'] = True
    monkeypatch.setattr(cli,'read',lambda *a:pytest.fail('Opened source for invalid config'))
    with pytest.raises(ValueError): cli.prerequisites(bad,'unused')


def test_prepare_prerequisites_never_load_weights():
    text = inspect.getsource(cli.prerequisites)
    assert 'torch.load' not in text and 'review_training' not in text and 'load_trained_ensemble(' not in text


def test_model_review_is_inherited_from_hash_bound_original_parent_not_intermediate(tmp_path,monkeypatch):
    monkeypatch.setattr(cli,'ROOT',tmp_path)
    frozen = {'training_acceptance':{'ensembles':{'ordinary':'accepted'}},'input_hashes':{'x':'sha'},'baseline_root':'baseline'}
    p = tmp_path/'request.json'; p.write_text(json.dumps(frozen))
    intermediate = {'config':{'parent_request':'request.json','parent_request_hash':cli.digest(frozen)},
        'parent_artifact_hashes':{'request.json':cli.sha(p)},'input_hashes':{'x':'sha'},'baseline_root':'baseline'}
    assert cli.inherited_training_acceptance(intermediate) == frozen['training_acceptance']
    intermediate['parent_artifact_hashes']['request.json'] = 'changed'
    with pytest.raises(ValueError): cli.inherited_training_acceptance(intermediate)


def test_full_cohort_is_bound_to_report_request_and_roots_receipt(tmp_path,monkeypatch):
    monkeypatch.setattr(cli,'ROOT',tmp_path); directory = tmp_path/'full'; directory.mkdir()
    def write(name,value): (directory/name).write_text(json.dumps(value))
    request = {'scope':'full','baseline_root':'baseline'}; write('request.json',request); write('roots.json',[{'root_id':'r'}])
    report = {'status':'complete','request_hash':cli.digest(request),'roots_sha256':cli.sha(directory/'roots.json')}
    write('report.json',report); write('receipt.json',{'request_hash':cli.digest(request),
        'report_sha256':cli.sha(directory/'report.json'),'roots_sha256':cli.sha(directory/'roots.json')})
    frozen = {'config':{'prior_report':'full/report.json','prior_report_sha256':cli.sha(directory/'report.json')}}
    monkeypatch.setattr(cli,'frozen_review_parent',lambda ancestor:frozen)
    rows,bound = cli.accepted_full_cohort({'baseline_root':'baseline'})
    assert rows == [{'root_id':'r'}] and len(bound) == 4
    write('roots.json',[{'root_id':'substituted'}])
    with pytest.raises(ValueError): cli.accepted_full_cohort({'baseline_root':'baseline'})


def test_stale_request_rejected_before_asset_loading(tmp_path,monkeypatch):
    monkeypatch.setattr(cli,'ROOT',tmp_path); monkeypatch.setattr(cli.shared,'clean',lambda:None)
    monkeypatch.setattr(cli.shared,'sources',lambda:{'new':'sha'})
    monkeypatch.setattr(cli,'prerequisites',lambda *a:pytest.fail('Opened assets for stale request'))
    out = tmp_path/'artifacts/rcci02_v2_cohort/old'; out.mkdir(parents=True)
    q = {'version':cli.VERSION,'config':deepcopy(SETTINGS),'source_hashes':{'old':'sha'}}
    h = cli.digest(q)
    (out/'request.json').write_text(json.dumps(q)); (out/'preparation.json').write_text(json.dumps({'request_hash':h}))
    with pytest.raises(ValueError): cli.load_request(out/'request.json',h)


def test_empty_physical_pool_is_preserved_not_dropped(road):
    h,x,n,e = scene(road,{'ego':state(road)})
    b = branch_from_forecast(road,h,x,n,e); p,cov,fail,mech = cli.evaluate_features(h,{m:n for m in MODES},e,road)
    r = compare_native(h,b,0,p,road)
    assert r['common_follow_points'] == 26 and not r['eligible_cross_pairs']
    for m in MODES:
        assert not r['following'][m]['errors']['front_combined_min_gap_m']
        assert not r['cross'][m]['errors']
        assert all(x['front_gap_m']['delta'] is None for x in mech[m]['variants']['total'])


@pytest.mark.parametrize('fault',['nan','shape','native_reverse'])
def test_invalid_arrays_or_native_truth_stop_instead_of_skipping(road,fault):
    h,x,n,e,b = prepared(road)
    if fault == 'native_reverse':
        b['future_metadata']['frames'][1]['vehicles']['main']['lane_position_m'] -= 20.
        p = project_predictions(h,{m:n for m in MODES},e,road)
        with pytest.raises(ValueError): compare_native(h,b,0,p,road)
    else:
        if fault == 'nan': n[0,11,0,0] = np.nan
        else: n = n[:,:,:24]
        with pytest.raises(ValueError): project_predictions(h,{m:n for m in MODES},e,road)


def test_synthetic_runner_seals_inputs_outputs_and_does_not_load_actual_weights(tmp_path,monkeypatch,road):
    from prediction_rl.prediction.interface import PredictorConfig
    h,x,n,e,b = prepared(road); baseline = tmp_path/'baseline'; baseline.mkdir()
    directory = baseline/'r'; directory.mkdir(); out = tmp_path/'artifacts/rcci02_v2_cohort/synthetic'; out.mkdir(parents=True)
    pack = {'candidates':[{'candidate_id':c,'labels':{'requested_plan':[]}} for c in range(5)],'root_traffic':{}}
    (directory/'labels.json').write_text(json.dumps(pack))
    for c in range(5):
        branch = deepcopy(b); branch['trace'] = []
        (directory/f'c{c}.json').write_text(json.dumps(branch))
    hp = directory/'history.json'; hp.write_text(json.dumps({'history':h,'inputs':x,'root_id':'root','episode_id':'ep',
        'label_pack':'r/labels.json','label_pack_sha256':cli.sha(directory/'labels.json')}))
    network = tmp_path/cli.shared.CONFIG['network']; network.parent.mkdir(parents=True)
    network.write_bytes((ROOT/cli.shared.CONFIG['network']).read_bytes())
    cp = tmp_path/'config.json'; cp.write_text('{}'); design = tmp_path/'design.md'; design.write_text('frozen')
    cfg = deepcopy(SETTINGS); cfg.update(max_ensemble_forward_calls=2,max_member_forward_calls=6,design_contract='design.md')
    q = {'baseline_root':str(baseline),'config':cfg,'config_path':'config.json','config_sha256':cli.sha(cp),
        'design_contract_sha256':cli.sha(design),'worktree_map_sha256':cli.sha(network),'source_hashes':{},
        'environment':{},'input_hashes':{},'parent_artifact_hashes':{},'design_episodes':[],
        'training_acceptance':{'ensembles':{m:{'path':'fake.pt','sha256':'not_loaded','selected':[]} for m in MODES}},
        'roots':[{'history':'r/history.json','root_id':'root','episode_id':'ep','split':'train'}]}
    monkeypatch.setattr(cli,'ROOT',tmp_path)
    monkeypatch.setattr(cli,'load_request',lambda *a:(out,q,{'model':PredictorConfig().to_dict()}))
    monkeypatch.setattr(cli,'load_trained_ensemble',lambda *a:object())
    monkeypatch.setattr(cli,'infer',lambda *a:(x,{m:n for m in MODES},e))
    monkeypatch.setattr(cli,'build_branch_labels',lambda *a:{'requested_plan':[]})
    monkeypatch.setattr(cli.shared,'clean',lambda:None)
    monkeypatch.setattr(cli.shared,'sources',lambda:{})
    monkeypatch.setattr(cli.shared,'environment',lambda:{})
    monkeypatch.setattr(cli.shared,'git',lambda root,*args:cli.shared.BASE if args == ('rev-parse','HEAD') else '')
    monkeypatch.setattr(cli.torch,'load',lambda *a,**k:pytest.fail('Actual weight loading forbidden in synthetic test'))
    cli.run('unused','synthetic_hash')
    report = cli.read(out/'report.json'); receipt = cli.read(out/'receipt.json'); index = cli.read(out/'index.json')
    assert report['status'] == 'complete' and report['roots'] == 1 and report['ensemble_forward_calls'] == 2
    assert receipt['report_sha256'] == cli.sha(out/'report.json') and receipt['index_sha256'] == cli.sha(out/'index.json')
    assert all(cli.sha(out/name) == hsh for name,hsh in index['root_files'].items())
    assert not (out/'writer.lock').exists() and not report['native_targets_used_for_features']
