from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'),str(ROOT/'tools')]
from prediction_rl.evaluation.rcci_frozen_audit import (
    SETTINGS, MODES, observed_window, observed_pair, native_prefix, compare_branch,
    sensitivity, aggregate, error_summary,
)
from prediction_rl.features.road_consequence_geometry import RoadConsequenceMap, occupancy_window
from test_road_consequences import scene, state, track, trajectory
import audit_rcci_frozen_predictors as cli


@pytest.fixture
def road(): return RoadConsequenceMap(ROOT/'RL-MPC-LaneMerging-master/merge.net.xml')


def branch_from_forecast(road,h,x,n,e,c=0):
    origin = np.asarray(h[0]['vehicles']['ego']['position']); frames = []
    for i in range(25):
        vehicles = {}
        for actor,root in h[0]['vehicles'].items():
            p = e[c,i] if actor == 'ego' else n[c,x['actor_ids'].index(actor),i]
            projected = road.project(p[:2]+origin,root['lane_id'])
            value = deepcopy(root); value.update(position=(p[:2]+origin).tolist(),speed=float(p[2]),
                lane_id=projected['lane'],lane_position_m=projected['progress_m']-road.geometry.offsets[projected['lane']])
            vehicles[actor] = value
        frames.append({'simulation_time_s':(i+1)*.2,'vehicles':vehicles})
    return {'future_metadata':{'frames':frames,'trajectory_usable':[True]*25}}


def prepared(road):
    h,x,n,e = scene(road,{'ego':state(road,'ramp_0',195.,speed=15.),
                          'main':state(road,position=195.,speed=15.)})
    branch = branch_from_forecast(road,h,x,n,e)
    return h,x,n,e,branch


def test_independent_exact_config_and_limits():
    assert json.loads((ROOT/cli.CONFIG_PATH).read_text()) == SETTINGS
    assert SETTINGS['splits'] == ['train','validation'] and SETTINGS['roots_per_split'] == 3
    assert not SETTINGS['training'] and not SETTINGS['simulation'] and not SETTINGS['test_access']


def test_full_window_matches_frozen_adapter_not_new_event_definition():
    p = np.linspace(-2,8,26)
    a = observed_window(p,(0,4)); b = occupancy_window(track(p),(0,4))
    for key in a:
        if key != 'horizon_s': assert a[key] == b[key]
    assert a['horizon_s'] == 5.


def test_prefix_censoring_no_fake_terminal_tail():
    w = observed_window([-2.,-1.,0.],(0,4))
    assert w['entry_s'] == pytest.approx(.4) and w['exit_s'] is None
    assert w['right_censored'] and w['horizon_s'] == pytest.approx(.4)
    short = observed_window([-2.,-1.],(0,4))
    assert not short['reached'] and short['entry_s'] is None
    assert error_summary([])['mae'] is None


def test_overlap_comparison_is_observed_horizon_only():
    a = observed_window([0.,1.,2.],(0,4)); b = observed_window([1.,2.,3.],(0,4))
    p = observed_pair(a,b)
    assert p['observed_overlap_s'] == pytest.approx(.4)
    assert p['signed_margin_s'] is None and p['ego_entry_order'] is None
    with pytest.raises(ValueError): observed_pair(a,observed_window([0.,1.],(0,4)))


@pytest.mark.parametrize('fault',['nan','reverse','too_long'])
def test_invalid_prefix_fails_not_marked_safe(fault):
    p = {'nan':[0,np.nan],'reverse':[1,0],'too_long':list(range(27))}[fault]
    with pytest.raises(ValueError): observed_window(p,(0,4))


def test_terminal_or_actor_disappearance_stops_prefix(road):
    h,x,n,e,b = prepared(road)
    b['future_metadata']['trajectory_usable'][1] = False
    assert len(native_prefix(h[0],b,'main')) == 2
    b['future_metadata']['trajectory_usable'][1] = True
    b['future_metadata']['frames'][1]['vehicles'].pop('main')
    assert len(native_prefix(h[0],b,'main')) == 2


def test_matched_perfect_forecasts_have_zero_observed_errors(road):
    h,x,n,e,b = prepared(road)
    r = compare_branch(h,b,0,{m:n for m in MODES},e,road)['actors'][0]
    assert r['observed_future_steps'] == 25 and r['joint_window_support']
    for mode in MODES:
        errors = r['methods'][mode]['errors']
        assert len(errors['progress_m']) == 25
        assert errors['entry_s'] and errors['exit_s']
        assert all(abs(v) < 1e-8 for values in errors.values() for v in values)


def test_event_miss_is_counted_not_hidden_by_time_mae(road):
    h,x,n,e,b = prepared(road); stopped = n.copy()
    slot = x['actor_ids'].index('main')
    stationary = deepcopy(h[0]['vehicles']['main']); stationary['speed'] = 0.
    stopped[:,slot] = trajectory(road,stationary,h[0]['vehicles']['ego']['position'])
    r = compare_branch(h,b,0,{'ordinary':n,'conditional':stopped},e,road)['actors'][0]
    assert r['methods']['ordinary']['counts']['entry_tp'] == 1
    assert r['methods']['conditional']['counts']['entry_fn'] == 1
    assert r['methods']['conditional']['errors']['entry_s'] == []


def test_joint_projection_error_support_does_not_favour_one_model(road):
    h,x,n,e,b = prepared(road); damaged = n.copy()
    damaged[0,x['actor_ids'].index('main'),3,1] += 4.
    r = compare_branch(h,b,0,{'ordinary':n,'conditional':damaged},e,road)['actors'][0]
    assert not r['joint_window_support'] and r['joint_projected_points'] == 24
    assert len(r['methods']['ordinary']['errors']['progress_m']) == 24
    assert len(r['methods']['conditional']['errors']['progress_m']) == 24
    assert r['methods']['ordinary']['counts']['valid_forecast_points'] == 25
    assert r['methods']['conditional']['counts']['valid_forecast_points'] == 24


def test_no_future_support_is_not_25_imputed_errors(road):
    h,x,n,e,b = prepared(road); b['future_metadata']['trajectory_usable'][0] = False
    r = compare_branch(h,b,0,{m:n for m in MODES},e,road)['actors'][0]
    assert r['observed_future_steps'] == 0 and r['joint_projected_points'] == 0
    assert not r['joint_window_support']
    assert r['methods']['ordinary']['errors']['progress_m'] == []


def test_neighbor_replacement_is_separate_from_ego_probe_effect(road):
    h,x,n,e,b = prepared(road); conditional = n.copy()
    conditional[4,x['actor_ids'].index('main'),:,0] += 1.
    r = sensitivity(h,{'ordinary':n,'conditional':conditional},e,road)
    assert r['ordinary']['neighbor_candidate_xy_max_m'] == 0.
    assert r['ordinary']['neighbor_replacement_numeric_max'] == 0.
    assert r['conditional']['neighbor_candidate_xy_max_m'] == pytest.approx(1.)
    assert r['conditional']['replacement_may_be_ood']


def test_aggregate_preserves_episode_denominators_and_empty_support(road):
    h,x,n,e,b = prepared(road)
    row = {'split':'train','episode_id':'same','sensitivity':sensitivity(h,{m:n for m in MODES},e,road),
           'branches':[compare_branch(h,b,c,{m:n for m in MODES},e,road) for c in range(5)]}
    r = aggregate([row,row])
    assert r['train']['roots'] == 2 and r['train']['unique_episodes'] == 1
    assert r['train']['methods']['ordinary']['counts']['actor_tracks'] == 10
    assert r['validation']['methods']['ordinary']['errors']['progress_m']['count'] == 0
    assert r['validation']['methods']['ordinary']['errors']['progress_m']['mae'] is None


def test_unknown_config_rejected_before_opening_source(monkeypatch):
    bad = deepcopy(SETTINGS); bad['test_access'] = True
    monkeypatch.setattr(cli.shared,'inputs',lambda *a:pytest.fail('Source opened before config rejection'))
    with pytest.raises(ValueError): cli.prerequisites('not_opened',bad)


def test_stale_source_request_rejected_before_asset_loading(tmp_path,monkeypatch):
    monkeypatch.setattr(cli,'ROOT',tmp_path); monkeypatch.setattr(cli.shared,'clean',lambda:None)
    monkeypatch.setattr(cli.shared,'sources',lambda:{'x':'new'})
    monkeypatch.setattr(cli,'prerequisites',lambda *a:pytest.fail('Loaded assets for stale source'))
    out = tmp_path/'artifacts/rcci02/old'; out.mkdir(parents=True)
    q = {'version':cli.VERSION,'source_hashes':{'x':'old'}}; h = cli.digest(q)
    (out/'request.json').write_text(json.dumps(q)); (out/'preparation.json').write_text(json.dumps({'request_hash':h}))
    with pytest.raises(ValueError): cli.load_request(out/'request.json',h)
