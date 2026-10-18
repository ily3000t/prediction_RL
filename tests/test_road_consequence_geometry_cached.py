from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from prediction_rl.features.road_consequence_geometry import RoadConsequenceMap
from prediction_rl.features.road_consequence_geometry_cached import CachedRoadConsequenceMap
from prediction_rl.features.road_consequences_v2 import summarize_road_consequences_v2
from prediction_rl.features.rcci_provider_cached import FrozenCachedRCCIV2Features
from test_road_consequences import scene, state
from test_rcci_provider import provider

NETWORK = ROOT/'RL-MPC-LaneMerging-master/merge.net.xml'


@pytest.mark.parametrize('lane', ['ramp_0', ':mergenode_1_0', 'highwayrear_0', ':mergenode_0_0', 'highwayahead_0'])
def test_exact_xy_projection_all_lanes_successors_boundaries_and_residuals(lane):
    old, new = RoadConsequenceMap(NETWORK), CachedRoadConsequenceMap(NETWORK)
    for future_lane in old.legal_lanes(lane):
        for position in np.linspace(0, old.geometry.lengths[future_lane], 23):
            point = old.xy(future_lane, position)
            np.testing.assert_array_equal(point, new.xy(future_lane, position))
            for delta in ((0, 0), (.5, 0), (0, -.5), (5, 3)):
                assert old.project(point+delta, lane) == new.project(point+delta, lane)
    for bad in ([np.nan, 1], [0, np.inf], [1], [[1, 2]]):
        for road in (old, new):
            with pytest.raises(ValueError): road.project(bad, lane)
    for bad in (-1., old.geometry.lengths[lane]+1., float('nan')):
        for road in (old, new):
            with pytest.raises(ValueError): road.xy(lane, bad)


def test_zone_cache_exact_keys_lru_invalid_not_cached():
    old, new = RoadConsequenceMap(NETWORK), CachedRoadConsequenceMap(NETWORK, zone_capacity=2)
    root = state(old, 'ramp_0', 100.)
    for width in (1.8, 2., 1.8, 2.1):
        root['width_m'] = width
        assert new.zone_interval(root) == old.zone_interval(root)
    assert new.cache_info() == {'hits': 1, 'misses': 3, 'size': 2, 'capacity': 2}
    root['width_m'] = 2.; new.zone_interval(root)
    assert new.cache_info()['misses'] == 4
    root['width_m'] = float('nan')
    with pytest.raises(ValueError): new.zone_interval(root)
    assert new.cache_info()['size'] == 2
    for lane in old.points:
        for length, width in ((5., 1.8), (8., 3.), (5.000000001, 1.8)):
            root.update(lane_id=lane, length_m=length, width_m=width)
            assert new.zone_interval(root) == old.zone_interval(root)
    for capacity in (0, 129, True, 2.):
        with pytest.raises(ValueError): CachedRoadConsequenceMap(NETWORK, zone_capacity=capacity)


def test_immutable_geometry_cannot_leave_stale_cache():
    road = CachedRoadConsequenceMap(NETWORK)
    for array in (road.zone, *road.points.values(), *road._lanes['ramp_0'][:-1]):
        with pytest.raises(ValueError): array.setflags(write=True)
    with pytest.raises(TypeError): road.points['ramp_0'] = np.zeros((2, 2))
    with pytest.raises(TypeError): road.geometry.offsets['ramp_0'] = 1.
    with pytest.raises(AttributeError): road.geometry.lengths = {}
    with pytest.raises(AttributeError): road.zone = road.zone.copy()
    with pytest.raises(AttributeError): road.points = {}


@pytest.mark.parametrize('mode', ['ordinary', 'conditional'])
def test_cached_provider_exact_and_opt_in_not_old_raw_or_default(mode):
    p = provider(mode)
    new = FrozenCachedRCCIV2Features(NETWORK, p.config, p.model)
    assert type(p.road) is RoadConsequenceMap
    assert isinstance(new.road, CachedRoadConsequenceMap)
    h, _, n, e = scene(p.road, {'ego': state(p.road, ':mergenode_1_0', 20.),
                              'main': state(p.road, ':mergenode_0_0', 20.)})
    for mutate in (False, True):
        nn = n.copy()
        if mutate: nn[0, 0, 2, 1] += 4.
        a = summarize_road_consequences_v2(h, nn, e, p.road)
        b = summarize_road_consequences_v2(h, nn, e, new.road)
        np.testing.assert_array_equal(a.pop('features'), b.pop('features'))
        assert a == b
    before = deepcopy(h)
    np.testing.assert_array_equal(p(h), new(h))
    assert h == before and p.calls == new.calls == 1
    assert p.last['result']['candidates'] == new.last['result']['candidates']
