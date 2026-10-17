from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'),str(ROOT/'tools')]
from prediction_rl.features.road_consequence_geometry import RoadConsequenceMap
from prediction_rl.features.road_consequences import summarize_road_consequences as v1
from prediction_rl.features.road_consequences_v2 import summarize_road_consequences_v2 as v2
from prediction_rl.evaluation.rcci_v2_cached_audit import SETTINGS, MODES, compare, aggregate
from test_road_consequences import scene,state
import audit_rcci_v2_cached as cli


@pytest.fixture
def road(): return RoadConsequenceMap(ROOT/'RL-MPC-LaneMerging-master/merge.net.xml')


def test_exact_config_forbids_inference_or_new_scenes():
    assert json.loads((ROOT/cli.CONFIG_PATH).read_text()) == SETTINGS
    assert SETTINGS['model_forward_calls'] == 0 and not SETTINGS['model_weight_loading']
    assert SETTINGS['splits'] == ['train','validation'] and not SETTINGS['native_targets_used_for_v2_features']


def test_partial_information_never_counts_as_full_coverage(road):
    h,x,n,e = scene(road); n[0,x['actor_ids'].index('front'),20,0] = n[0,x['actor_ids'].index('front'),19,0]-.2
    r = compare(v1(h,n,e,road),v2(h,n,e,road)); c = r['counts']
    assert c['full_follow_pool_5s'] == 4
    assert c['v1_front_min_available'] == 4 and c['v2_supported_front_min_available'] == 5
    assert c['v2_front_available_with_unknown_suffix'] == 1
    assert c['v2_front_min_at_observed_root'] == 5  # Availability is not prediction accuracy.
    assert r['full_5s_support_unchanged'] and not r['comparison_is_prediction_accuracy']


def test_fabricated_full_coverage_is_rejected(road):
    h,x,n,e = scene(road); n[0,0,3,2] = -.1
    old,new = v1(h,n,e,road),v2(h,n,e,road)
    new['features'][6] = 1
    with pytest.raises(ValueError,match='partial support'): compare(old,new)


def test_full_support_values_and_root_metadata_are_checked(road):
    h,x,n,e = scene(road); old,new = v1(h,n,e,road),v2(h,n,e,road)
    new['features'][0] += .01
    with pytest.raises(ValueError,match='front consequence'): compare(old,new)
    new = v2(h,n,e,road); new['features'][-1] = 1
    with pytest.raises(ValueError,match='Root metadata'): compare(old,new)


def test_zero_overlap_support_is_not_equated_to_old_window_availability(road):
    h,x,n,e = scene(road,{'ego':state(road,'ramp_0',100.,speed=0.),'main':state(road,position=100.,speed=0.)})
    c = compare(v1(h,n,e,road),v2(h,n,e,road))['counts']
    assert c['full_cross_pool_5s'] == c['cross_applicable'] == 5
    assert c['v1_complete_cross_overlap_available'] == 0
    assert c['v2_supported_overlap_available'] == 5 and c['v2_supported_positive_overlap'] == 0


def test_cached_comparison_never_loads_weights(road,monkeypatch):
    import torch
    monkeypatch.setattr(torch,'load',lambda *a,**k:pytest.fail('Weights loaded by cached replay'))
    h,x,n,e = scene(road)
    assert compare(v1(h,n,e,road),v2(h,n,e,road))['full_5s_support_unchanged']


def test_aggregation_keeps_episode_denominator(road):
    h,x,n,e = scene(road); compared = compare(v1(h,n,e,road),v2(h,n,e,road))
    row = {'split':'train','episode_id':'same','methods':{m:compared for m in MODES}}
    r = aggregate([row,row])
    assert r['train']['roots'] == 2 and r['train']['unique_episodes'] == 1
    assert r['train']['methods']['ordinary']['counts']['candidate_tables'] == 10
    assert r['validation']['methods']['ordinary']['following_prefix_horizon_s']['count'] == 0


def test_unknown_config_rejected_before_cache_read(monkeypatch):
    monkeypatch.setattr(cli,'read',lambda *a:pytest.fail('Opened parent before rejecting settings'))
    bad = deepcopy(SETTINGS); bad['model_forward_calls'] = 12
    with pytest.raises(ValueError): cli.parent_inputs(bad)


def test_stale_source_never_loads_parent_or_models(tmp_path,monkeypatch):
    monkeypatch.setattr(cli,'ROOT',tmp_path); monkeypatch.setattr(cli.shared,'clean',lambda:None)
    monkeypatch.setattr(cli.shared,'sources',lambda:{'x':'new'})
    monkeypatch.setattr(cli,'parent_inputs',lambda *a:pytest.fail('Opened parent for changed source'))
    out = tmp_path/'artifacts/rcci02_v2/old'; out.mkdir(parents=True)
    q = {'version':cli.VERSION,'config':SETTINGS,'source_hashes':{'x':'old'}}; h = cli.digest(q)
    (out/'request.json').write_text(json.dumps(q)); (out/'preparation.json').write_text(json.dumps({'request_hash':h}))
    with pytest.raises(ValueError): cli.load_request(out/'request.json',h)
