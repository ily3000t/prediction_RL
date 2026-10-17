"""Offline RCCI02 invariants; no SUMO, checkpoints or driving-result tuning."""
from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from prediction_rl.data.actor_adapter import build_actor_inputs
from prediction_rl.features.road_consequence_geometry import (
    RoadConsequenceMap, ProjectedTrack, occupancy_window, window_pair,
)
from prediction_rl.features.road_consequences import (
    DIMENSION, NAMES, PROBE_FIELDS, contract, summarize_road_consequences,
)
from prediction_rl.features.probe_features import summarize as raw_summary


@pytest.fixture
def road():
    return RoadConsequenceMap(ROOT/'RL-MPC-LaneMerging-master/merge.net.xml')


def state(road, lane='highwayrear_0', position=150., speed=10., length=5., width=1.8):
    return {'position':road.xy(lane, position).tolist(), 'lane_id':lane,
            'lane_position_m':position, 'speed':speed, 'acceleration':0.,
            'length_m':length, 'width_m':width}


def trajectory(road, actor, origin):
    lane = actor['lane_id']; start = actor['lane_position_m']
    rows = []
    for i in range(1,26):
        distance = start+actor['speed']*i*.2
        path = road.legal_lanes(lane); index = 0
        while distance > road.geometry.lengths[path[index]] and index < len(path)-1:
            distance -= road.geometry.lengths[path[index]]; index += 1
        xy = road.xy(path[index], distance)-origin
        rows.append([*xy, actor['speed'], 0.])
    return np.asarray(rows)


def scene(road, vehicles=None):
    vehicles = vehicles or {'ego':state(road), 'front':state(road, position=170.),
                            'rear':state(road, position=130.)}
    h = [{'simulation_time_s':0., 'vehicles':vehicles}]
    x = build_actor_inputs(h, road.geometry)
    origin = np.asarray(vehicles['ego']['position'])
    neighbors = np.zeros((5,12,25,4)); ego = np.repeat(trajectory(road, vehicles['ego'], origin)[None],5,0)
    for slot,actor in enumerate(x['actor_ids']):
        if actor is not None: neighbors[:,slot] = trajectory(road, vehicles[actor], origin)
    return h, x, neighbors, ego


def output(road, *args):
    h,x,n,e = scene(road, *args)
    return summarize_road_consequences(h,n,e,road), (h,x,n,e)


def track(progress):
    p = np.asarray(progress, dtype=float)
    return ProjectedTrack(p, ('highwayrear_0',)*26, np.ones(26,dtype=bool), ('valid',)*26, np.ones(26))


def test_layout_signed_gaps_and_root_metadata(road):
    r,(h,x,n,e) = output(road)
    assert DIMENSION == len(set(NAMES)) == 149 and len(PROBE_FIELDS) == 28
    assert r['features'].shape == (149,)
    assert r['features'][:6] == pytest.approx([.15,0.,0.,1.,1.,.5])
    assert r['features'][6:12] == pytest.approx([.15,0.,0.,1.,1.,.5])
    np.testing.assert_array_equal(r['features'][-9:], raw_summary(x,torch.tensor(n),torch.tensor(e))[-9:])
    assert not r['features'][12:28].any()  # No cross-approach relation.
    assert r['online_release'] is False and not r['all_actor_safety_claim']


def test_overlap_is_negative_route_clearance_not_point_distance(road):
    r,_ = output(road, {'ego':state(road), 'overlap':state(road, position=152.)})
    assert r['features'][0] == pytest.approx(-.03)
    assert r['features'][3:5].tolist() == [1.,1.]


def test_closing_speed_and_full_26_sample_minimum(road):
    r,_ = output(road, {'ego':state(road, speed=10.), 'front':state(road, position=170.,speed=9.)})
    assert r['features'][:3] == pytest.approx([.1,1/30,1.])


def test_batch_and_single_are_identical(road):
    r,(h,x,n,e) = output(road)
    singles = [summarize_road_consequences(h,n[c:c+1],e[c:c+1],road) for c in range(5)]
    np.testing.assert_array_equal(r['features'][:140], np.concatenate([s['features'][:28] for s in singles]))
    for c,s in enumerate(singles): assert r['candidates'][c] == s['candidates'][0]


def test_empty_pool_is_not_a_safety_claim(road):
    r,_ = output(road, {'ego':state(road)})
    assert r['features'][:6].tolist() == [0,0,0,0,1,0]
    assert not r['features'][12:28].any()
    assert r['features'][-5] == 0


@pytest.mark.parametrize('bad', ['nan_padding','nan_ego','shape','bad_root'])
def test_engineering_errors_raise(road,bad):
    h,x,n,e = scene(road)
    if bad == 'nan_padding': n[0,11,0,0] = np.nan
    if bad == 'nan_ego': e[0,0,0] = np.nan
    if bad == 'shape': n = n[:,:,:24]
    if bad == 'bad_root': h[0]['vehicles']['ego']['position'][0] += .01
    with pytest.raises(ValueError): summarize_road_consequences(h,n,e,road)


@pytest.mark.parametrize('bad', ['off_path','endpoint','negative_speed','route_reversal'])
def test_invalid_forecast_is_masked_not_dropped(road,bad):
    h,x,n,e = scene(road); slot = x['actor_ids'].index('front')
    if bad == 'off_path': n[0,slot,:,1] += 4.
    if bad == 'endpoint': n[0,slot,-1,:2] = np.array([101.,-1.6])-h[0]['vehicles']['ego']['position']
    if bad == 'negative_speed': n[0,slot,3,2] = -1.
    if bad == 'route_reversal': n[0,slot,3,:2] = n[0,slot,1,:2]
    r = summarize_road_consequences(h,n,e,road)
    assert r['features'][:3].tolist() == [0.,0.,0.]
    assert r['features'][4] == 0.  # Complete pool, NOT valid-subset pooling.
    assert bad in r['candidates'][0]['actor_invalid_reasons']['front']
    assert r['features'][28+4] == 1.  # Candidate 1 remains independent.


def test_only_legal_successors_and_ambiguous_projection(road):
    assert road.project(road.xy(':mergenode_1_0',10.), 'highwayrear_0')['valid'] is False
    assert road.project(road.xy('highwayrear_0',100.), 'highwayahead_0')['valid'] is False
    assert road.project(road.xy('highwayahead_0',0.), 'highwayrear_0')['valid']
    broken = deepcopy(road)
    broken.points['highwayahead_0'] = road.points['highwayrear_0'].copy()
    assert broken.project(road.xy('highwayrear_0',100.),'highwayrear_0')['reason'] == 'ambiguous'


def test_junction_is_finite_and_uses_front_bumper_length_width(road):
    root = state(road)
    lo,hi = road.zone_interval(root)
    assert hi == pytest.approx(5.)
    assert hi < road.geometry.lengths['highwayahead_0']
    longer = deepcopy(root); longer['length_m'] = 8.
    assert road.zone_interval(longer)[1] == pytest.approx(8.)
    ramp = state(road,'ramp_0',150.)
    wider = deepcopy(ramp); wider['width_m'] = 3.
    a,b = road.zone_interval(ramp),road.zone_interval(wider)
    assert b[0] <= a[0] and b[1] >= a[1]
    assert lo < -52.


def test_window_interpolation_censoring_and_not_reached():
    w = occupancy_window(track(np.linspace(-2,8,26)), (0.,4.))
    assert w == {'valid':True,'reached':True,'entry_s':1.,'exit_s':3.,
                 'left_censored':False,'right_censored':False,'passed_before_root':False}
    left = occupancy_window(track(np.linspace(2,3,26)),(0,4))
    assert left['entry_s'] == 0 and left['exit_s'] is None
    assert left['left_censored'] and left['right_censored']
    pending = occupancy_window(track(np.linspace(-10,-5,26)),(0,4))
    assert not pending['reached'] and pending['right_censored'] and pending['entry_s'] is None
    past = occupancy_window(track(np.linspace(6,9,26)),(0,4))
    assert past['passed_before_root'] and not past['right_censored']
    assert window_pair(w,left)['signed_margin_s'] is None
    assert window_pair(w,left)['overlap_s'] == pytest.approx(2.)


def test_stopped_exactly_at_exit_not_falsely_declared_left():
    w = occupancy_window(track(np.full(26,4.)),(0,4))
    assert w['reached'] and w['exit_s'] is None and w['right_censored']


def test_short_event_between_original_four_sample_times_is_retained():
    w = occupancy_window(track(np.linspace(-1,9,26)),(0, .4))
    assert w['entry_s'] == pytest.approx(.5) and w['exit_s'] == pytest.approx(.7)
    assert not any(w['entry_s'] <= t <= w['exit_s'] for t in (.2,1.,3.,5.))


def test_cross_approach_windows_and_stable_id_tie_break(road):
    vehicles = {'ego':state(road,':mergenode_1_0',20.),
                'z':state(road,':mergenode_0_0',20.),'a':state(road,':mergenode_0_0',20.)}
    r,_ = output(road,vehicles)
    c = r['candidates'][0]['conflict']
    assert c['selected_actor'] == 'a' and c['complete_pool_projection']
    assert r['features'][18:20].tolist() == [1,1]
    assert r['features'][22:24].tolist() == [1,1]  # Both left-censored.
    assert r['features'][26] == 1 and r['features'][27] == 0
    assert c['pairs']['a']['overlap_s'] > 0


def test_no_arrival_is_not_encoded_as_zero_risk(road):
    r,_ = output(road, {'ego':state(road,'ramp_0',100.,speed=0.),
                       'main':state(road,position=100.,speed=0.)})
    f = r['features'][12:28]
    assert f[6:10].tolist() == [1,1,0,0]
    assert f[12:16].tolist() == [1,1,0,0]
    assert f[:6].tolist() == [0.]*6


def test_omitted_tokens_not_turned_into_vehicles(road):
    vehicles = {'ego':state(road)}
    vehicles.update({f'n{i:02}':state(road,position=20.+i*8.,speed=0.) for i in range(15)})
    r,(h,x,n,e) = output(road,vehicles)
    assert len(r['root_omitted_ids']) == 3 and len([a for a in r['actor_ids'] if a]) == 12
    assert r['features'][-1] == 1 and r['features'][-2] == pytest.approx(3/15)
    assert all(a in x['actor_ids'] for a in r['candidates'][0]['actor_invalid_reasons'])


def test_actor_order_future_poison_and_inputs_unchanged(road):
    h,x,n,e = scene(road); before_n = n.copy(); before_h = deepcopy(h)
    first = summarize_road_consequences(h,n,e,road)
    poison = deepcopy(h)
    poison[0]['vehicles'] = dict(reversed(list(poison[0]['vehicles'].items())))
    poison[0].update(future_lanes=['oracle'], labels={'reward':float('nan')}, terminal_mask=[False])
    second = summarize_road_consequences(poison,n,e,road)
    np.testing.assert_array_equal(first['features'],second['features'])
    np.testing.assert_array_equal(n,before_n); assert h == before_h


def test_postprocessor_rigid_transform_consistency_not_network_equivariance(road):
    r,(h,x,n,e) = output(road, {'ego':state(road,':mergenode_1_0',20.),
                              'main':state(road,':mergenode_0_0',20.)})
    angle = .73; rot = np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
    shift = np.array([123.,-74.]); transformed = deepcopy(road)
    transformed.points = {lane:p@rot.T+shift for lane,p in road.points.items()}
    transformed.zone = road.zone@rot.T+shift
    hh = deepcopy(h)
    for s in hh[0]['vehicles'].values(): s['position'] = (rot@s['position']+shift).tolist()
    nn,ee = n.copy(),e.copy(); nn[...,:2] = n[...,:2]@rot.T; ee[...,:2] = e[...,:2]@rot.T
    result = summarize_road_consequences(hh,nn,ee,transformed)
    np.testing.assert_allclose(result['features'],r['features'],atol=2e-7,rtol=0.)


def test_contract_is_serializable_and_not_mutable_via_return():
    import json
    first = contract(); json.dumps(first,allow_nan=False)
    assert first['dimension'] == len(first['names']) == 149
    first['names'].clear()
    assert len(contract()['names']) == 149 and not contract()['online_registration']


def test_incomplete_window_not_fabricated_and_margin_direction():
    t = track(np.linspace(-2,8,26)); t.valid[-1] = False
    invalid = occupancy_window(t,(0,4))
    assert not invalid['valid'] and invalid['entry_s'] is None and not invalid['right_censored']
    a = occupancy_window(track(np.linspace(-2,18,26)),(0,4))  # .5 to 1.5
    b = occupancy_window(track(np.linspace(-10,10,26)),(0,4))  # 2.5 to 3.5
    assert window_pair(a,b)['signed_margin_s'] == pytest.approx(1.)
    c = occupancy_window(track(np.linspace(-4,16,26)),(0,4))  # 1 to 2
    assert window_pair(a,c)['signed_margin_s'] == pytest.approx(-.5)


def test_conflict_invalid_pool_has_explicit_mask_not_valid_subset(road):
    h,x,n,e = scene(road, {'ego':state(road,':mergenode_1_0',20.),
                          'bad':state(road,':mergenode_0_0',20.),
                          'good':state(road,':mergenode_0_0',10.)})
    n[0,x['actor_ids'].index('bad'),1,1] += 4.
    r = summarize_road_consequences(h,n,e,road)
    f = r['features'][12:28]
    assert f[6:8].tolist() == [1.,0.]
    assert not f[:6].any() and not f[8:].any()
    assert r['candidates'][0]['conflict']['selected_actor'] == 'bad'


def test_candidate_conditioning_is_not_discarded_or_promoted_to_action(road):
    h,x,n,e = scene(road)
    slot = x['actor_ids'].index('front')
    n[4,slot] = trajectory(road,state(road,position=170.,speed=9.),h[0]['vehicles']['ego']['position'])
    r = summarize_road_consequences(h,n,e,road)
    assert r['features'][0] != r['features'][4*28]
    assert 'action' not in r and 'selected_candidate' not in r


def test_online_runtime_has_not_been_registered():
    wrapper = (ROOT/'src/prediction_rl/envs/prediction_features.py').read_text(encoding='utf-8')
    assert 'road_consequences' not in wrapper and 'rcci02' not in wrapper
