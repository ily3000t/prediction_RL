from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from prediction_rl.features.road_consequence_geometry import RoadConsequenceMap
from prediction_rl.features.road_consequences import summarize_road_consequences as v1
from prediction_rl.features.road_consequences_v2 import (
    DIMENSION,NAMES,PROBE_FIELDS,contract,supported_prefix,prefix_window,prefix_overlap,
    summarize_road_consequences_v2 as summarize,
)
from test_road_consequences import scene,state,track
from prediction_rl.evaluation.rcci_frozen_audit import observed_window


@pytest.fixture
def road(): return RoadConsequenceMap(ROOT/'RL-MPC-LaneMerging-master/merge.net.xml')


def test_layout_metadata_and_full_gap_support(road):
    h,x,n,e = scene(road); r = summarize(h,n,e,road); old = v1(h,n,e,road)
    assert DIMENSION == len(set(NAMES)) == 149 and len(PROBE_FIELDS) == 28
    assert r['features'].shape == (149,)
    assert r['features'][:7] == pytest.approx([.15,0,0,1,1,1,1])
    assert r['features'][7:14] == pytest.approx([.15,0,0,1,1,1,1])
    assert not r['features'][14:28].any()
    np.testing.assert_array_equal(r['features'][-9:],old['features'][-9:])
    assert not r['unknown_suffix_safe'] and not r['online_release']


def test_late_reversal_keeps_earlier_all_actor_information_not_full_horizon(road):
    h,x,n,e = scene(road); slot = x['actor_ids'].index('front')
    n[0,slot,20,0] = n[0,slot,19,0]-.2
    before = n.copy(); old = v1(h,n,e,road); r = summarize(h,n,e,road)
    assert old['features'][:3].tolist() == [0.,0.,0.]
    assert r['features'][:7] == pytest.approx([.15,0,0,1,1,.8,0])
    assert r['features'][7:14] == pytest.approx([.15,0,0,1,1,.8,0])
    d = r['candidates'][0]
    assert d['actor_prefixes']['front']['first_unavailable_reason'] == 'route_reversal'
    assert d['actor_prefixes']['front']['first_unavailable_time_s'] == pytest.approx(4.2)
    assert d['front']['all_pool_actor_ids'] == ['front','rear']
    assert r['features'][28+6] == 1.  # Other candidates independent.
    np.testing.assert_array_equal(n,before)


@pytest.mark.parametrize('reason',['negative_speed','off_path'])
def test_prefix_never_reenters_after_one_bad_point(road,reason):
    h,x,n,e = scene(road); slot = x['actor_ids'].index('front')
    if reason == 'negative_speed': n[0,slot,3,2] = -.1
    else: n[0,slot,3,1] += 4.
    r = summarize(h,n,e,road); d = r['candidates'][0]['actor_prefixes']['front']
    assert d['horizon_s'] == pytest.approx(.6) and not d['full_5s']
    assert d['first_unavailable_reason'] == reason
    assert r['features'][5] == pytest.approx(.6/5)


def test_bad_nonmandatory_actor_is_not_removed_from_pool(road):
    vehicles = {'ego':state(road),'front':state(road,position=170.),'rear':state(road,position=130.),
                'far':state(road,position=10.,speed=0.)}
    h,x,n,e = scene(road,vehicles); assert 'far' not in x['mandatory_roles'].values()
    n[0,x['actor_ids'].index('far'),20,0] -= .2
    r = summarize(h,n,e,road)
    assert r['features'][5] == pytest.approx(.8) and r['features'][6] == 0
    assert r['candidates'][0]['front']['all_pool_actor_ids'] == ['far','front','rear']


def test_root_only_is_not_a_forecast_hit_or_safe_zero(road):
    h,x,n,e = scene(road); n[0,x['actor_ids'].index('front'),0,1] += 4.
    r = summarize(h,n,e,road)
    assert r['features'][:7].tolist() == [0.,0.,0.,1.,0.,0.,0.]
    assert r['candidates'][0]['front']['shared_prefix_points'] == 1


def test_empty_pool_has_no_applicable_forecast(road):
    h,x,n,e = scene(road,{'ego':state(road)})
    r = summarize(h,n,e,road)
    assert r['features'][:7].tolist() == [0,0,0,0,0,1,1]
    assert r['features'][7:14].tolist() == [0,0,0,0,0,1,1]
    assert not r['features'][14:28].any()


def test_censored_occupancy_does_not_need_false_exits(road):
    h,x,n,e = scene(road,{'ego':state(road,':mergenode_1_0',20.,speed=1.),
                         'main':state(road,':mergenode_0_0',20.,speed=1.)})
    r = summarize(h,n,e,road); f = r['features'][14:28]
    assert f == pytest.approx([0,0,1,0,1,1,1,1,1,1,1,1,1,1])
    pair = r['candidates'][0]['conflict']['pairs']['main']
    assert pair['actor']['exit_s'] is None and pair['ego']['exit_s'] is None
    assert pair['supported_overlap_s'] == 5. and pair['first_overlap_s'] == 0.


def test_partial_cross_overlap_is_only_prefix_duration(road):
    h,x,n,e = scene(road,{'ego':state(road,':mergenode_1_0',20.,speed=1.),
                         'main':state(road,':mergenode_0_0',20.,speed=1.)})
    n[0,0,20,0] = n[0,0,19,0]-.2
    r = summarize(h,n,e,road); f = r['features'][14:28]
    assert f[2] == pytest.approx(.8) and f[5] == pytest.approx(.8)
    assert f[12:14].tolist() == [1.,0.]
    pair = r['candidates'][0]['conflict']['pairs']['main']
    assert pair['supported_overlap_s'] == pytest.approx(4.) and pair['scope'] == 'shared_prefix_only'
    assert pair['actor']['right_censored'] and pair['actor']['exit_s'] is None


def test_no_entry_is_censored_not_unavailable_or_infinite_eta(road):
    h,x,n,e = scene(road,{'ego':state(road,'ramp_0',100.,speed=0.),'main':state(road,position=100.,speed=0.)})
    r = summarize(h,n,e,road); f = r['features'][14:28]
    assert f == pytest.approx([0,0,0,0,1,1,0,0,0,0,1,1,1,1])
    p = r['candidates'][0]['conflict']['pairs']['main']
    assert p['first_overlap_s'] is None and p['actor']['entry_s'] is None


def test_root_only_cross_has_no_available_future_overlap(road):
    h,x,n,e = scene(road,{'ego':state(road,'ramp_0',100.),'main':state(road,position=100.)})
    n[0,0,0,1] += 4.
    r = summarize(h,n,e,road); f = r['features'][14:28]
    assert f[4] == 1 and f[5] == 0 and f[12] == 0 and f[13] == 0


def test_full_windows_retain_old_geometry_and_entry_times(road):
    h,x,n,e = scene(road,{'ego':state(road,'ramp_0',195.,speed=15.),'main':state(road,position=195.,speed=15.)})
    r = summarize(h,n,e,road); old = v1(h,n,e,road)
    assert r['features'][14] == pytest.approx(old['features'][12])
    assert r['features'][15] == pytest.approx(old['features'][14])
    assert r['features'][16] == pytest.approx(old['features'][16])
    assert r['features'][17] == pytest.approx(max(r['features'][14:16]))
    assert r['features'][26:28].tolist() == [1,1]


def test_prefix_window_full_parity_and_explicit_length_guard():
    p = supported_prefix(track(np.linspace(-2,8,26)))
    assert prefix_window(p,(0,4)) == observed_window(p.track.progress,(0,4))
    with pytest.raises(ValueError): prefix_window(p,(0,4),27)
    p.track.valid[4] = False
    cut = supported_prefix(p.track)
    assert cut.length == 4 and prefix_window(cut,(0,4))['horizon_s'] == pytest.approx(.6)
    with pytest.raises(ValueError): prefix_window(p,(0,4))


def test_boundary_contact_is_not_positive_overlap():
    a = {'horizon_s':1.,'reached':True,'entry_s':0.,'exit_s':.5}
    b = {'horizon_s':1.,'reached':True,'entry_s':.5,'exit_s':1.}
    p = prefix_overlap(a,b)
    assert p['supported_overlap_s'] == 0 and p['first_overlap_s'] is None and p['overlap_available']


def test_batch_and_single_candidates_identical(road):
    h,x,n,e = scene(road); r = summarize(h,n,e,road)
    singles = [summarize(h,n[c:c+1],e[c:c+1],road) for c in range(5)]
    np.testing.assert_array_equal(r['features'][:140],np.concatenate([s['features'][:28] for s in singles]))
    for c,s in enumerate(singles): assert r['candidates'][c] == s['candidates'][0]


@pytest.mark.parametrize('bad',['nan_padding','shape','root_xy'])
def test_engineering_errors_do_not_turn_into_partial_availability(road,bad):
    h,x,n,e = scene(road)
    if bad == 'nan_padding': n[0,11,0,0] = np.nan
    if bad == 'shape': n = n[:,:,:24]
    if bad == 'root_xy': h[0]['vehicles']['ego']['position'][0] += .1
    with pytest.raises(ValueError): summarize(h,n,e,road)


def test_id_ties_actor_order_and_future_poison(road):
    h,x,n,e = scene(road,{'ego':state(road,':mergenode_1_0',20.,speed=1.),
                         'z':state(road,':mergenode_0_0',20.,speed=1.),'a':state(road,':mergenode_0_0',20.,speed=1.)})
    r = summarize(h,n,e,road); assert r['candidates'][0]['conflict']['selected_actor'] == 'a'
    poisoned = deepcopy(h); poisoned[0]['vehicles'] = dict(reversed(list(poisoned[0]['vehicles'].items())))
    poisoned[0].update(labels={'future':float('nan')},termination_mask=[True],future_actor_ids=['oracle'])
    np.testing.assert_array_equal(r['features'],summarize(poisoned,n,e,road)['features'])


@pytest.mark.parametrize('partial',[False,True])
def test_postprocessor_rigid_transform_not_model_equivariance(road,partial):
    h,x,n,e = scene(road,{'ego':state(road,':mergenode_1_0',20.),'main':state(road,':mergenode_0_0',20.)})
    if partial: n[0,0,20,0] = n[0,0,19,0]-.2
    expected = summarize(h,n,e,road)
    angle = .7; rot = np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
    shift = np.array([100.,-35.]); moved = deepcopy(road)
    moved.points = {lane:p@rot.T+shift for lane,p in road.points.items()}; moved.zone = road.zone@rot.T+shift
    hh = deepcopy(h)
    for s in hh[0]['vehicles'].values(): s['position'] = (rot@s['position']+shift).tolist()
    nn,ee = n.copy(),e.copy(); nn[...,:2] = n[...,:2]@rot.T; ee[...,:2] = e[...,:2]@rot.T
    result = summarize(hh,nn,ee,moved)
    np.testing.assert_allclose(result['features'],expected['features'],atol=2e-7,rtol=0)


def test_omitted_tokens_are_not_locatable_actors(road):
    vehicles = {'ego':state(road)}
    vehicles.update({f'n{i:02}':state(road,position=20.+i*8.,speed=0.) for i in range(15)})
    h,x,n,e = scene(road,vehicles); r = summarize(h,n,e,road)
    assert len(r['root_omitted_ids']) == 3 and len(r['candidates'][0]['actor_prefixes']) == 12
    assert not set(r['root_omitted_ids']) & set(r['candidates'][0]['actor_prefixes'])


def test_serializable_contract_and_no_provider_registration():
    schema = contract(); json.dumps(schema,allow_nan=False)
    assert schema['dimension'] == 149 and not schema['forecast_repair'] and not schema['unknown_suffix_safe']
    schema['names'].clear(); assert len(contract()['names']) == 149
    assert 'road_consequences_v2' not in (ROOT/'src/prediction_rl/envs/prediction_features.py').read_text()
