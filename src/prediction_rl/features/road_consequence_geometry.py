"""RCCI02 finite merge-node control-volume geometry (not collision geometry).

Positions are front-bumper centres. Footprints are road-aligned rectangles;
the finite control volume is the convex hull of the pinned junction polygon.
Neither projection nor simultaneous occupancy is a safety certificate.
"""
from dataclasses import dataclass
import math
import xml.etree.ElementTree as ET

import numpy as np

from prediction_rl.data.merge_geometry import MergeGeometry

PROJECTION_RADIUS_M = .5  # Representation support, NOT a clearance threshold.
EPS = 1e-8
CHAINS = (
    ('ramp_0', ':mergenode_1_0', 'highwayahead_0'),
    ('highwayrear_0', ':mergenode_0_0', 'highwayahead_0'),
)


def convex_hull(points):
    points = sorted(set(map(tuple, points)))
    def cross(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1])-(a[1]-o[1])*(b[0]-o[0])
    def half(seq):
        result = []
        for p in seq:
            while len(result) > 1 and cross(result[-2], result[-1], p) <= 0:
                result.pop()
            result.append(p)
        return result
    hull = np.asarray(half(points)[:-1]+half(points[::-1])[:-1], dtype=np.float64)
    if len(hull) < 3 or not np.isfinite(hull).all():
        raise ValueError('Degenerate finite control volume')
    return hull


@dataclass(frozen=True)
class ProjectedTrack:
    progress: np.ndarray
    lanes: tuple
    valid: np.ndarray
    reasons: tuple
    speed: np.ndarray

    @property
    def complete(self):
        return bool(self.valid.all())


class RoadConsequenceMap:
    def __init__(self, network):
        self.geometry = MergeGeometry(network)
        tree = ET.parse(network).getroot()
        self.points = {x.get('id'): np.asarray([list(map(float, p.split(',')))
                       for p in x.get('shape').split()], dtype=np.float64)
                       for x in tree.findall('./edge/lane')}
        node = tree.find("./junction[@id='mergenode']")
        if node is None:
            raise ValueError('Missing pinned merge node')
        self.zone = convex_hull([list(map(float, p.split(','))) for p in node.get('shape').split()])
        for p in self.points.values():
            if (p.ndim != 2 or p.shape[1] != 2 or not np.isfinite(p).all()
                    or len(p) < 2 or np.any(np.linalg.norm(np.diff(p, axis=0), axis=1) <= 0)):
                raise ValueError('Invalid lane shape')

    def legal_lanes(self, root_lane):
        if root_lane not in self.points:
            raise ValueError('Unknown observed root lane')
        chain = next(c for c in CHAINS if root_lane in c)
        return chain[chain.index(root_lane):]

    def xy(self, lane, position):
        """No route-end extrapolation; SUMO length scales polyline arc length."""
        length = self.geometry.lengths[lane]
        if not math.isfinite(position) or not 0 <= position <= length:
            raise ValueError('Position outside finite lane')
        p = self.points[lane]; d = np.diff(p, axis=0)
        arc = np.linalg.norm(d, axis=1); ends = np.cumsum(arc)
        s = position/length*ends[-1]
        i = min(int(np.searchsorted(ends, s, side='right')), len(d)-1)
        start = 0 if i == 0 else ends[i-1]
        return p[i]+(s-start)/arc[i]*d[i]

    def project(self, xy, root_lane):
        """Only the observed lane and its unique successors; no label lanes."""
        xy = np.asarray(xy, dtype=np.float64)
        if xy.shape != (2,) or not np.isfinite(xy).all():
            raise ValueError('Nonfinite/malformed projection input')
        candidates = []
        for lane in self.legal_lanes(root_lane):
            p = self.points[lane]; d = np.diff(p, axis=0)
            arc = np.linalg.norm(d, axis=1); starts = np.r_[0., np.cumsum(arc)[:-1]]
            u = np.sum((xy-p[:-1])*d, axis=1)/arc**2
            clipped = np.clip(u, 0., 1.)
            dist = np.linalg.norm(p[:-1]+clipped[:, None]*d-xy, axis=1)
            if not np.isfinite(dist).all():
                raise ValueError('Projection arithmetic overflow')
            for i in np.flatnonzero(dist <= dist.min()+EPS):
                s = (starts[i]+clipped[i]*arc[i])/arc.sum()*self.geometry.lengths[lane]
                candidates.append((float(dist[i]), self.geometry.offsets[lane]+s, lane,
                                   bool((i == 0 and u[i] < -EPS) or (i == len(d)-1 and u[i] > 1+EPS))))
        best = min(c[0] for c in candidates)
        tied = [c for c in candidates if c[0] <= best+EPS]
        chosen = min(tied, key=lambda c:(c[2], c[1]))
        ambiguous = max(c[1] for c in tied)-min(c[1] for c in tied) > 1e-6
        reason = ('ambiguous' if ambiguous else 'endpoint' if chosen[3] else
                  'off_path' if best > PROJECTION_RADIUS_M else 'valid')
        return {'progress_m':chosen[1], 'lane':chosen[2], 'residual_m':best,
                'valid':reason == 'valid', 'reason':reason}

    def track(self, root, future, origin):
        """26 points: observed root plus 25 root-relative forecast points."""
        a = np.asarray(future, dtype=np.float64)
        if a.shape != (25, 4) or not np.isfinite(a).all():
            raise ValueError('Require 25 finite XY/speed/acceleration rows')
        xy = np.vstack((root['position'], a[:, :2]+origin))
        speed = np.r_[root['speed'], a[:, 2]]
        projected = [self.project(p, root['lane_id']) for p in xy]
        progress = np.asarray([p['progress_m'] for p in projected])
        valid = np.asarray([p['valid'] for p in projected], dtype=bool)
        reasons = [p['reason'] for p in projected]
        # Positive-speed forecasts that reverse along the route are not snapped
        # into a forward trajectory; the remainder loses event validity.
        reverse = np.flatnonzero(np.diff(progress) < -EPS)
        if len(reverse):
            valid[reverse[0]+1:] = False
            reasons[reverse[0]+1:] = ['route_reversal']*(25-reverse[0])
        for i in np.flatnonzero(speed < 0):
            valid[i] = False; reasons[i] = 'negative_speed'
        return ProjectedTrack(progress, tuple(p['lane'] for p in projected), valid, tuple(reasons), speed)

    def zone_interval(self, root):
        """Front-bumper progress interval whose road-aligned body touches zone.

        SAT clipping is exact on each straight centreline segment for this
        *approximation*. The union must be connected on this pinned road.
        Length/width come from the observed actor, never a lane-width default.
        """
        length, width = float(root['length_m']), float(root['width_m'])
        if not all(math.isfinite(v) and v > 0 for v in (length, width)):
            raise ValueError('Invalid observed vehicle extents')
        chain = next(c for c in CHAINS if root['lane_id'] in c)
        if root['lane_id'] == 'highwayahead_0':
            chain = ('highwayahead_0',)
        intervals = []
        edges = np.roll(self.zone, -1, axis=0)-self.zone
        for lane in chain:
            p = self.points[lane]; delta = np.diff(p, axis=0)
            arcs = np.linalg.norm(delta, axis=1); starts = np.r_[0., np.cumsum(arcs)[:-1]]
            for i, (a, d, arc) in enumerate(zip(p[:-1], delta, arcs)):
                tangent = d/arc; normal = np.array([-tangent[1], tangent[0]])
                offsets = np.asarray([t*tangent+w*normal for t in (-length, 0.) for w in (-width/2, width/2)])
                axes = np.vstack((tangent, normal, np.c_[-edges[:, 1], edges[:, 0]]))
                lower, upper = 0., 1.
                for axis in axes:
                    zone = self.zone@axis; extent = offsets@axis
                    lo, hi = zone.min()-extent.max(), zone.max()-extent.min()
                    position, movement = a@axis, d@axis
                    if abs(movement) <= EPS:
                        if position < lo-EPS or position > hi+EPS:
                            lower, upper = 1., 0.; break
                    else:
                        bounds = sorted(((lo-position)/movement, (hi-position)/movement))
                        lower, upper = max(lower, bounds[0]), min(upper, bounds[1])
                if lower <= upper+EPS:
                    scale = self.geometry.lengths[lane]/arcs.sum()
                    offset = self.geometry.offsets[lane]+starts[i]*scale
                    intervals.append((offset+lower*arc*scale, offset+upper*arc*scale))
        if not intervals:
            raise ValueError('Legal route misses pinned control volume')
        intervals.sort(); lo, hi = intervals[0]
        for left, right in intervals[1:]:
            if left > hi+1e-5:
                raise ValueError('Disconnected control-volume support requires a new contract')
            hi = max(hi, right)
        return float(lo), float(hi)


def occupancy_window(track, interval):
    """Piecewise-linear *projected progress*, not a continuous SUMO guarantee.

    None times have explicit status; no fake large ETA. Incomplete trajectories
    cannot supply a full-horizon window, including when only a late point fails.
    """
    if (track.progress.shape != (26,) or track.valid.shape != (26,)
            or not np.isfinite(track.progress).all() or len(interval) != 2
            or not all(math.isfinite(x) for x in interval) or interval[0] > interval[1]):
        raise ValueError('Malformed projected track/finite control-volume interval')
    if not track.complete:
        return {'valid':False, 'reached':False, 'entry_s':None, 'exit_s':None,
                'left_censored':False, 'right_censored':False, 'passed_before_root':False}
    if np.any(np.diff(track.progress) < -EPS):
        raise ValueError('Complete window requires non-reversing projected progress')
    s = track.progress; lo, hi = interval
    passed = bool(s[0] > hi+EPS)
    if passed:
        return {'valid':True, 'reached':False, 'entry_s':None, 'exit_s':None,
                'left_censored':False, 'right_censored':False, 'passed_before_root':True}
    def crossing(boundary):
        if s[0] >= boundary-EPS: return 0.
        indices = np.flatnonzero(s >= boundary-EPS)
        if not len(indices): return None
        i = int(indices[0]); fraction = (boundary-s[i-1])/(s[i]-s[i-1])
        return float((i-1+fraction)*.2)
    entry = crossing(lo)
    # At an exit boundary and stopped, body still touches the zone. It has not
    # demonstrably left; a positive subsequent displacement establishes exit.
    exit_time = crossing(hi) if np.any(s > hi+EPS) else None
    return {'valid':True, 'reached':entry is not None, 'entry_s':entry, 'exit_s':exit_time,
            'left_censored':bool(s[0] >= lo-EPS), 'right_censored':exit_time is None,
            'passed_before_root':False}


def window_pair(ego, actor):
    valid = ego['valid'] and actor['valid']
    overlap = margin = None
    if valid and ego['reached'] and actor['reached']:
        # Observed-horizon overlap is computable with a right-censored exit.
        ee = 5. if ego['exit_s'] is None else ego['exit_s']
        ae = 5. if actor['exit_s'] is None else actor['exit_s']
        overlap = max(0., min(ee, ae)-max(ego['entry_s'], actor['entry_s']))
        # Exact separation/order margin requires both exits and entries.
        if (not ego['left_censored'] and not actor['left_censored']
                and ego['exit_s'] is not None and actor['exit_s'] is not None):
            margin = max(ego['entry_s'], actor['entry_s'])-min(ee, ae)
    return {'valid':valid, 'overlap_s':overlap, 'signed_margin_s':margin}
