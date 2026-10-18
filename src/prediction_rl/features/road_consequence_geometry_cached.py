"""Exact static-geometry caching; the original map remains the oracle.

No forecast, actor, time or validity-result cache. Geometry is immutable for the
lifetime of this instance; a different network requires a new instance.
"""
from collections import OrderedDict
import math
from types import MappingProxyType

import numpy as np

from prediction_rl.data.merge_geometry import MergeGeometry
from prediction_rl.features.road_consequence_geometry import (
    EPS, PROJECTION_RADIUS_M, RoadConsequenceMap,
)


def _readonly(value):
    a = np.asarray(value, dtype=np.float64)
    # Bytes-backed arrays cannot have WRITEABLE re-enabled through a view.
    return np.frombuffer(a.tobytes(), dtype=a.dtype).reshape(a.shape)


class _FrozenGeometry(MergeGeometry):
    def __init__(self, original):
        for name in ('lengths', 'offsets', 'groups'):
            object.__setattr__(self, name, MappingProxyType(dict(getattr(original, name))))

    def __setattr__(self, name, value):
        raise AttributeError('Pinned geometry is immutable; create a new map')


class CachedRoadConsequenceMap(RoadConsequenceMap):
    """Same scalar operations/tie order as v1; only static terms are reused."""
    def __init__(self, network, *, zone_capacity=128):
        if type(zone_capacity) is not int or not 1 <= zone_capacity <= 128:
            raise ValueError('Zone cache capacity must be an integer in [1,128]')
        super().__init__(network)
        self.geometry = _FrozenGeometry(self.geometry)
        self.points = MappingProxyType({lane: _readonly(p) for lane, p in self.points.items()})
        self.zone = _readonly(self.zone)
        lanes = {}
        for lane, p in self.points.items():
            delta = np.diff(p, axis=0)
            arc = np.linalg.norm(delta, axis=1)
            # Preserve the separate cumsum/sum expressions used by the oracle.
            lanes[lane] = (_readonly(delta), _readonly(arc),
                _readonly(np.r_[0., np.cumsum(arc)[:-1]]),
                _readonly(np.cumsum(arc)), _readonly(arc**2), arc.sum())
        self._lanes = MappingProxyType(lanes)
        self._zone_capacity = zone_capacity
        self._zone_cache = OrderedDict()
        self._zone_hits = self._zone_misses = 0
        self._locked = True

    def __setattr__(self, name, value):
        if getattr(self, '_locked', False) and name in (
                'geometry', 'points', 'zone', '_lanes', '_zone_capacity', '_locked'):
            raise AttributeError('Pinned geometry is immutable; create a new map')
        super().__setattr__(name, value)

    def xy(self, lane, position):
        length = self.geometry.lengths[lane]
        if not math.isfinite(position) or not 0 <= position <= length:
            raise ValueError('Position outside finite lane')
        p = self.points[lane]
        d, arc, _, ends, _, _ = self._lanes[lane]
        s = position/length*ends[-1]
        i = min(int(np.searchsorted(ends, s, side='right')), len(d)-1)
        start = 0 if i == 0 else ends[i-1]
        return p[i]+(s-start)/arc[i]*d[i]

    def project(self, xy, root_lane):
        xy = np.asarray(xy, dtype=np.float64)
        if xy.shape != (2,) or not np.isfinite(xy).all():
            raise ValueError('Nonfinite/malformed projection input')
        candidates = []
        for lane in self.legal_lanes(root_lane):
            p = self.points[lane]
            d, arc, starts, _, squared, total = self._lanes[lane]
            u = np.sum((xy-p[:-1])*d, axis=1)/squared
            clipped = np.clip(u, 0., 1.)
            dist = np.linalg.norm(p[:-1]+clipped[:, None]*d-xy, axis=1)
            if not np.isfinite(dist).all():
                raise ValueError('Projection arithmetic overflow')
            for i in np.flatnonzero(dist <= dist.min()+EPS):
                s = (starts[i]+clipped[i]*arc[i])/total*self.geometry.lengths[lane]
                candidates.append((float(dist[i]), self.geometry.offsets[lane]+s, lane,
                    bool((i == 0 and u[i] < -EPS) or (i == len(d)-1 and u[i] > 1+EPS))))
        best = min(c[0] for c in candidates)
        tied = [c for c in candidates if c[0] <= best+EPS]
        chosen = min(tied, key=lambda c: (c[2], c[1]))
        ambiguous = max(c[1] for c in tied)-min(c[1] for c in tied) > 1e-6
        reason = ('ambiguous' if ambiguous else 'endpoint' if chosen[3] else
                  'off_path' if best > PROJECTION_RADIUS_M else 'valid')
        return {'progress_m': chosen[1], 'lane': chosen[2], 'residual_m': best,
                'valid': reason == 'valid', 'reason': reason}

    def zone_interval(self, root):
        length, width = float(root['length_m']), float(root['width_m'])
        if not all(math.isfinite(v) and v > 0 for v in (length, width)):
            raise ValueError('Invalid observed vehicle extents')
        key = (root['lane_id'], length, width)
        if key in self._zone_cache:
            self._zone_hits += 1
            self._zone_cache.move_to_end(key)
            return self._zone_cache[key]
        self._zone_misses += 1
        value = super().zone_interval(root)
        self._zone_cache[key] = value
        if len(self._zone_cache) > self._zone_capacity:
            self._zone_cache.popitem(last=False)
        return value

    def cache_info(self):
        return {'hits': self._zone_hits, 'misses': self._zone_misses,
                'size': len(self._zone_cache), 'capacity': self._zone_capacity}
