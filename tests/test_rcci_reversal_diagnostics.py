from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'),str(ROOT/'tools')]
from prediction_rl.evaluation.rcci_reversal_diagnostics import (
    SETTINGS, MODES, track_diagnostics, numeric_support, margin_unavailable_reasons,
    diagnose_root, aggregate, describe,
)
from prediction_rl.features.road_consequence_geometry import RoadConsequenceMap, occupancy_window, window_pair
from prediction_rl.features.road_consequences import summarize_road_consequences
from test_road_consequences import scene, state, track
from test_rcci_frozen_audit import prepared, branch_from_forecast
import diagnose_rcci_reversals as cli


@pytest.fixture
def road(): return RoadConsequenceMap(ROOT/'RL-MPC-LaneMerging-master/merge.net.xml')


def test_exact_config_no_repair_authority():
    assert json.loads((ROOT/cli.CONFIG_PATH).read_text()) == SETTINGS
    assert not SETTINGS['change_adapter_or_thresholds'] and not SETTINGS['simulation']
    assert SETTINGS['max_model_forward_calls'] == 12 and SETTINGS['scope'] == 'same_six_roots_no_expansion'


def test_one_reversal_is_not_five_reverse_events(road):
    h,x,n,e = scene(road); slot = x['actor_ids'].index('front')
    p = n[0,slot].copy(); p[20,0] = p[19,0]-.2; saved = p.copy()
    native = [road.geometry.progress(h[0]['vehicles']['front'])+i*2 for i in range(26)]
    d = track_diagnostics(h[0]['vehicles']['front'],p,h[0]['vehicles']['ego']['position'],road,native_progress=native)
    assert d['actual_reverse_transition_count'] == 1
    assert d['v1_invalid_reasons'] == {'route_reversal':5}
    assert d['later_projectable_nonnegative_speed_points_after_reverse'] == 4
    event = d['reversal_events'][0]
    assert event['time_s'] == pytest.approx(4.2) and event['backward_step_m'] == pytest.approx(.2)
    assert event['predicted_speed_end_mps'] == 10. and event['observed_native_step_m'] == 2.
    assert event['ds_minus_end_speed_dt_m'] == pytest.approx(-2.2)
    assert event['raw_projection_valid_both'] and event['backward_over_coordinate_ulp'] > 1000
    np.testing.assert_array_equal(p,saved)


def test_reversal_after_observed_horizon_has_no_imputed_native_motion(road):
    h,x,n,e = scene(road); p = n[0,0].copy(); p[20,0] = p[19,0]-.2
    root = h[0]['vehicles'][x['actor_ids'][0]]
    d = track_diagnostics(root,p,h[0]['vehicles']['ego']['position'],road,native_progress=[road.geometry.progress(root)]*3)
    assert d['reversal_events'][0]['observed_native_step_m'] is None
    assert d['observed_future_steps'] == 2 and d['native_reverse_transition_count'] == 0


def test_negative_speed_does_not_invent_reversal_or_tail_propagation(road):
    h,x,n,e = scene(road); p = n[0,0].copy(); p[3,2] = -.1
    d = track_diagnostics(h[0]['vehicles'][x['actor_ids'][0]],p,h[0]['vehicles']['ego']['position'],road)
    assert not d['complete_v1'] and d['negative_speed_points'] == 1
    assert d['actual_reverse_transition_count'] == 0
    assert d['later_projectable_nonnegative_speed_points_after_reverse'] == 0
    assert d['minimum_predicted_speed_mps'] == pytest.approx(-.1)


def test_no_reversal_does_not_return_fake_zero_error_evidence(road):
    h,x,n,e = scene(road)
    d = track_diagnostics(h[0]['vehicles'][x['actor_ids'][0]],n[0,0],h[0]['vehicles']['ego']['position'],road)
    assert d['complete_v1'] and d['reversal_events'] == [] and d['first_reverse_time_s'] is None
    assert describe([])['median'] is None and describe([])['count'] == 0


@pytest.mark.parametrize('bad',['nan','shape'])
def test_invalid_input_is_an_error_not_mask_sensitivity(road,bad):
    h,x,n,e = scene(road); p = n[0,0].copy()
    if bad == 'nan': p[2,0] = np.nan
    else: p = p[:24]
    with pytest.raises(ValueError): track_diagnostics(h[0]['vehicles'][x['actor_ids'][0]],p,h[0]['vehicles']['ego']['position'],road)


def test_numeric_zero_can_have_valid_support_and_censoring_stays_separate():
    f = np.zeros(28); f[3]=f[4]=1
    assert all(numeric_support(f)[k] for k in (0,1,2))
    f[18:22] = 1; f[22:26] = 1; f[26]=1
    m = numeric_support(f)
    assert m[12] and m[14] and m[16]  # entry=0 observed/left-censored is real support.
    assert not m[13] and not m[15] and not m[17]
    f[19] = 0
    assert not any(numeric_support(f)[k] for k in (12,13,14,15,16,17))


def test_margin_reasons_distinguish_censoring_from_projection_invalidity():
    w = occupancy_window(track(np.linspace(1,2,26)),(0,4))
    p = {'ego':w,'actor':w,**window_pair(w,w)}
    assert set(margin_unavailable_reasons(p)) == {'ego_left_censored','ego_right_censored','actor_left_censored','actor_right_censored'}
    invalid = deepcopy(w); invalid.update(valid=False,left_censored=False,right_censored=False)
    assert margin_unavailable_reasons({'ego':invalid,'actor':w,**window_pair(invalid,w)})[0] == 'ego_projection_incomplete'


def test_non_left_censored_complete_margin_is_possible_not_dead_by_construction():
    a = occupancy_window(track(np.linspace(-2,18,26)),(0,4))
    b = occupancy_window(track(np.linspace(-3,17,26)),(0,4))
    assert margin_unavailable_reasons({'ego':a,'actor':b,**window_pair(a,b)}) == []


def test_diagnostics_preserve_v1_features_and_all_actors(road):
    h,x,n,e,b = prepared(road); predictions = {m:n for m in MODES}
    branches = [branch_from_forecast(road,h,x,n,e,c) for c in range(5)]
    d = diagnose_root(h,predictions,e,branches,road)
    for mode in MODES:
        original = summarize_road_consequences(h,n,e,road)
        np.testing.assert_array_equal(d['methods'][mode]['feature_vector'],original['features'])
        assert len(d['methods'][mode]['tracks']) == 5*sum(x['actor_mask'])
        assert all(c['all_pair_margin_valid'] > 0 for c in d['methods'][mode]['candidates'])
    json.dumps(d,allow_nan=False)


def test_field_pool_loss_and_mandatory_roles_remain_visible(road):
    h,x,n,e,b = prepared(road); bad = n.copy()
    bad[0,0,20,0] = bad[0,0,19,0]-.2
    branches = [branch_from_forecast(road,h,x,n,e,c) for c in range(5)]
    d = diagnose_root(h,{'ordinary':n,'conditional':bad},e,branches,road)
    row = {'split':'train','episode_id':'single',**d}
    a = aggregate([row,row]); c = a['train']['methods']['conditional']['counts']
    assert a['train']['roots'] == 2 and a['train']['unique_episodes'] == 1
    assert c['actual_reverse_transitions'] == 2 and c['invalid_mandatory_actor_tracks'] == 2
    assert c['cross_applicable_but_pool_incomplete'] == 2
    assert d['methods']['conditional']['candidates'][0]['invalid_actor_ids'] == [x['actor_ids'][0]]
    assert a['validation']['methods']['ordinary']['backward_step_m']['count'] == 0


def test_unknown_config_rejected_before_reading_parent(monkeypatch):
    monkeypatch.setattr(cli,'read',lambda *a:pytest.fail('Read parent before rejecting changed configuration'))
    bad = deepcopy(SETTINGS); bad['change_adapter_or_thresholds'] = True
    with pytest.raises(ValueError): cli.parent_evidence(bad)


def test_stale_diagnostic_source_never_loads_assets(tmp_path,monkeypatch):
    monkeypatch.setattr(cli,'ROOT',tmp_path); monkeypatch.setattr(cli.shared,'clean',lambda:None)
    monkeypatch.setattr(cli.shared,'sources',lambda:{'x':'new'})
    monkeypatch.setattr(cli,'prerequisites',lambda *a:pytest.fail('Loaded assets for changed source'))
    out = tmp_path/'artifacts/rcci02_diag/old'; out.mkdir(parents=True)
    q = {'version':cli.VERSION,'config':SETTINGS,'source_hashes':{'x':'old'}}; h = cli.digest(q)
    (out/'request.json').write_text(json.dumps(q)); (out/'preparation.json').write_text(json.dumps({'request_hash':h}))
    with pytest.raises(ValueError): cli.load_request(out/'request.json',h)
