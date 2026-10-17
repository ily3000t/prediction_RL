"""Offline RCCI01 observations/metadata audit, NOT a conflict/safety adapter.

Known future lanes belong to offline labels only. Never use this diagnostic
to choose the projection lane of an online prediction.
"""
from collections import Counter
from array import array
import math
import xml.etree.ElementTree as ET

import numpy as np

from prediction_rl.data.merge_geometry import MergeGeometry

VERSION = 'rcci01_data_geometry_audit_v1'
CONTRACT = {
    'version':VERSION, 'purpose':'offline_feasibility_not_method_effect',
    'splits':['train','validation'], 'bounded_roots_per_split':3,
    'selection':'frozen_review_order_no_outcome_filter',
    'dataset_review':'artifacts/data_reviews/response_v1_review/review.json',
    'dataset_review_sha256':'0f59c02cb6daa075e65ab0c7461ca5ff5ca1bfb7f13af9d41ef85ae57632d312',
    'collection_request':'artifacts/data/response_v1/request.json',
    'predictor_request':'artifacts/training/predictor_v1/request.json',
    'network':'RL-MPC-LaneMerging-master/merge.net.xml',
    'candidate_ids':[0,1,2,3,4], 'horizon_steps':25, 'tick_s':.2,
    'diagnostic_distance_bins_m':[.001,.05,.25,1.,2.,5.],
    'projection_acceptance_threshold':None, 'predictor_inference':False,
    'simulation':False, 'training':False, 'online_adapter_released':False}
MEASUREMENTS = ('lane_position_xy_error_m','known_lane_projection_error_m','projected_lane_position_error_m')
SPARSE_INDICES = (0, 4, 14, 24)
HEADING_KEYS = ('angle', 'angle_deg', 'heading', 'heading_rad')


class AuditMap:
    """Constrained polyline projections; residuals are NOT validity decisions."""
    def __init__(self, network):
        self.geometry = MergeGeometry(network)
        self.points = {x.get('id'): np.array([list(map(float, p.split(',')))
                       for p in x.get('shape').split()], dtype=np.float64)
                       for x in ET.parse(network).getroot().findall('./edge/lane')}
        for lane, points in self.points.items():
            if (points.ndim != 2 or points.shape[1] != 2 or len(points) < 2
                    or not np.isfinite(points).all()
                    or np.any(np.linalg.norm(np.diff(points, axis=0), axis=1) <= 0)):
                raise ValueError('Malformed/degenerate lane polyline: '+lane)
            if not np.isfinite(np.linalg.norm(np.diff(points,axis=0),axis=1)).all():
                raise ValueError('Lane polyline arithmetic overflow: '+lane)

    def lane_xy(self, lane, position):
        if lane not in self.points or not math.isfinite(position):
            raise ValueError('Unknown lane/nonfinite position')
        length = self.geometry.lengths[lane]
        if not 0 <= position <= length:
            raise ValueError('Position outside lane; no silent extrapolation')
        points = self.points[lane]; delta = np.diff(points, axis=0)
        arc = np.linalg.norm(delta, axis=1); ends = np.cumsum(arc)
        distance = position/length*ends[-1]
        index = min(int(np.searchsorted(ends, distance, side='right')), len(arc)-1)
        start = 0. if index == 0 else ends[index-1]
        return points[index]+(distance-start)/arc[index]*delta[index]

    def project(self, xy, allowed_lanes):
        """Explicit lane whitelist, no nearest-lane search over the whole map.

        Endpoint clamps and alternate nearest projections are reported. They
        must NOT be interpreted as successful road occupancy or safety.
        """
        xy = np.asarray(xy, dtype=np.float64)
        lanes = tuple(allowed_lanes)
        if (xy.shape != (2,) or not np.isfinite(xy).all() or not lanes
                or len(set(lanes)) != len(lanes) or any(l not in self.points for l in lanes)):
            raise ValueError('Need finite XY and explicit unique legal lanes')
        choices = []
        for lane in lanes:
            p = self.points[lane]; delta = np.diff(p, axis=0)
            lengths = np.linalg.norm(delta, axis=1)
            fraction = np.sum((xy-p[:-1])*delta, axis=1)/lengths**2
            clamped = np.clip(fraction, 0., 1.)
            projected = p[:-1]+clamped[:, None]*delta
            distances = np.linalg.norm(projected-xy, axis=1)
            if not np.isfinite(distances).all(): raise ValueError('Projection arithmetic overflow')
            starts = np.concatenate(([0.], np.cumsum(lengths)[:-1]))
            for i in np.flatnonzero(distances <= distances.min()+1e-9):
                s = float((starts[i]+clamped[i]*lengths[i])/lengths.sum()*self.geometry.lengths[lane])
                choices.append({'lane': lane, 'lane_position_m': s, 'distance_m': float(distances[i]),
                    'progress_m': self.geometry.offsets[lane]+s,
                    'endpoint_clamped': bool((i == 0 and fraction[i] < 0)
                        or (i == len(delta)-1 and fraction[i] > 1))})
        best = min(c['distance_m'] for c in choices)
        tied = [c for c in choices if c['distance_m'] <= best+1e-9]
        chosen = min(tied, key=lambda c:(c['lane'], c['lane_position_m']))
        # Adjacent segments meeting at the same path coordinate are one point.
        ambiguous = max(c['progress_m'] for c in tied)-min(c['progress_m'] for c in tied) > 1e-6
        return {**chosen, 'ambiguous_progress': ambiguous,
                'co_nearest_lanes': sorted({c['lane'] for c in tied}),
                'diagnostic_only_no_acceptance_threshold': True}

    def audit_frame(self, frame):
        self.geometry.validate_frame(frame)
        records = []
        for actor, state in sorted(frame['vehicles'].items()):
            if any(k in state and state[k] is not None and not math.isfinite(state[k]) for k in HEADING_KEYS):
                raise ValueError('Nonfinite recorded heading')
            lane = state['lane_id']; xy = np.asarray(state['position'], dtype=np.float64)
            projected = self.project(xy, [lane])
            records.append({'actor': actor, 'lane': lane,
                'lane_position_xy_error_m': float(np.linalg.norm(self.lane_xy(lane, state['lane_position_m'])-xy)),
                'known_lane_projection_error_m': projected['distance_m'],
                'projected_lane_position_error_m': abs(projected['lane_position_m']-state['lane_position_m']),
                'ambiguous_progress': projected['ambiguous_progress'],
                'endpoint_clamped': projected['endpoint_clamped'],
                'heading_recorded': any(k in state and state[k] is not None for k in HEADING_KEYS),
                'route_id_recorded': 'route_id' in state,
                'length_m': state['length_m'], 'width_m': state['width_m']})
        return records


def audit_branch(branch, root, geometry):
    trace = branch['trace']; future = branch['future_metadata']
    frames = future['frames']; usable = future['trajectory_usable']
    if (not 1 <= len(trace) <= 25 or len(frames) != len(trace)
            or usable != [not s['done'] for s in trace]):
        raise ValueError('Future metadata length/terminal mask changed')
    previous = root; states = []; transitions = Counter(); terminal_seen = False
    new_ids = set(); missing_root_ids = set(); relations = Counter()
    for i, (step, frame) in enumerate(zip(trace, frames)):
        if terminal_seen: raise ValueError('Frames after terminal transition')
        if not math.isclose(frame['simulation_time_s']-previous['simulation_time_s'], .2, abs_tol=1e-8, rel_tol=0):
            raise ValueError('Future time grid changed')
        after = step['info']['execution_audit']['after']
        if (frame['simulation_time_s'] != after['simulation_time_s']
                or set(frame['vehicles']) != set(after['vehicles'])
                or any(any(frame['vehicles'][a].get(k) != v for k,v in s.items())
                       for a,s in after['vehicles'].items())):
            raise ValueError('Future geometry/base trace changed')
        terminal_seen = bool(step['done'])
        if not terminal_seen:
            states.extend(geometry.audit_frame(frame))
            vehicles = frame['vehicles']; ego = vehicles['ego']; groups = geometry.geometry.groups
            new_ids.update(set(vehicles)-set(root['vehicles']))
            missing_root_ids.update(set(root['vehicles'])-set(vehicles))
            for actor in sorted(vehicles):
                if actor in previous['vehicles'] and vehicles[actor]['lane_id'] != previous['vehicles'][actor]['lane_id']:
                    transitions['all'] += 1
                    if i not in SPARSE_INDICES: transitions['outside_four_sample_indices'] += 1
                if actor != 'ego':
                    a,b = groups[ego['lane_id']],groups[vehicles[actor]['lane_id']]
                    name = ('same_approach' if a == b and a != 'main_downstream' else
                            'both_downstream' if a == b else
                            'approach_vs_downstream' if 'main_downstream' in (a,b) else 'cross_approach')
                    relations[name] += 1
        previous = frame
    return {'states': states, 'observed_transitions': len(trace),
            'nonterminal_frames': sum(usable), 'horizon_censored': terminal_seen,
            'lane_transitions': dict(transitions), 'observed_relation_counts': dict(relations),
            'new_actor_ids': sorted(new_ids), 'missing_root_actor_ids': sorted(missing_root_ids)}


def distribution(values, bins):
    if any(not math.isfinite(b) or b<=0 for b in bins) or list(bins)!=sorted(set(bins)):
        raise ValueError('Diagnostic bins must be finite, positive and strictly increasing')
    if not values: return {'count': 0, 'mean': None, 'max': None, 'quantiles': None, 'histogram': []}
    a = np.asarray(values, dtype=np.float64)
    if not np.isfinite(a).all() or np.any(a < 0): raise ValueError('Invalid diagnostic measurement')
    edges = [0., *bins, math.inf]
    return {'count': len(values), 'mean': float(a.mean()), 'max': float(a.max()),
            'quantiles': {str(q):float(np.quantile(a,q)) for q in (.5,.95,.99)},
            'histogram': [{'lower_inclusive_m':l, 'upper_exclusive_m':None if math.isinf(u) else u,
                'count':int(((a>=l)&(a<u)).sum())} for l,u in zip(edges,edges[1:])]}


class AuditAccumulator:
    """Keep numeric vectors for exact quantiles, not millions of state dicts."""
    def __init__(self):
        self.counts=Counter(); self.episodes=set(); self.transitions=Counter(); self.relations=Counter()
        self.values={f:array('d') for f in MEASUREMENTS}

    def add(self,r):
        branches=r['branches']; states=list(r['root_states'])+[s for b in branches for s in b['states']]
        if not states: raise ValueError('Missing audit observations')
        self.episodes.add(r['episode_id'])
        self.counts.update(roots=1,branches=len(branches),actor_state_observations=len(states),
            roots_with_omitted_actors=int(r['omitted_actors']>0),
            roots_with_short_ego_history=int(r['ego_history_fraction']<1),
            heading_recorded_states=sum(s['heading_recorded'] for s in states),
            route_id_recorded_states=sum(s['route_id_recorded'] for s in states),
            endpoint_clamped_states=sum(s['endpoint_clamped'] for s in states),
            ambiguous_known_lane_progress_states=sum(s['ambiguous_progress'] for s in states),
            censored_branches=sum(b['horizon_censored'] for b in branches),
            branches_with_new_actors=sum(bool(b['new_actor_ids']) for b in branches),
            branches_with_missing_root_actors=sum(bool(b['missing_root_actor_ids']) for b in branches))
        for b in branches:
            self.transitions.update(b['lane_transitions']);self.relations.update(b['observed_relation_counts'])
        for f in MEASUREMENTS:self.values[f].extend(s[f] for s in states)

    def report(self,bins):
        if not self.counts['roots']:raise ValueError('Missing roots')
        result={**dict(self.counts),'unique_episodes':len(self.episodes),'root_slot_capacity':12,
            'lane_transitions':dict(self.transitions),'observed_relation_counts':dict(self.relations),
            'native_footprint_ground_truth_available':False,
            'event_extraction_implemented':False,'forecast_projection_evaluated':False,
            'sparse_transition_counts_are_not_missed_conflict_events':True,
            'shared_history_and_same_actors_repeated_not_independent_samples':True}
        for f,values in self.values.items():result[f]=distribution(values,bins)
        return result


def summarize(rows,bins):
    acc=AuditAccumulator()
    for r in rows:acc.add(r)
    return acc.report(bins)
