"""RCCI-A v2: descriptive consequences on shared, unrepaired prefixes.

All pool actors remain represented over the SAME prefix. Unknown suffixes
stay unknown; a supported minimum is not a full-horizon safety minimum.
Offline only: no registration, model loading, action choice or labels.
"""
from dataclasses import dataclass
import math

import numpy as np

from prediction_rl.data.actor_adapter import build_actor_inputs
from prediction_rl.features.road_consequence_geometry import RoadConsequenceMap, EPS, PROJECTION_RADIUS_M
from prediction_rl.features.road_consequences import array, root_metadata, GAP_TIE_EPS_M
from prediction_rl.features.probe_features import METADATA

VERSION = 'rcci02_a_supported_prefix_control_volume_v2'
TIME_TIE_EPS_S = 1e-8
FOLLOW_FIELDS = ('supported_min_route_gap_over_100m','closing_at_min_over_30mps',
    'time_at_min_over_5s','applicable','minimum_available','shared_prefix_horizon_over_5s','full_pool_5s')
CROSS_FIELDS = ('ego_entry_over_5s','actor_entry_over_5s','supported_overlap_over_5s',
    'first_overlap_over_5s','applicable','shared_prefix_horizon_over_5s',
    'ego_reached','actor_reached','ego_left_censored','actor_left_censored',
    'ego_right_censored','actor_right_censored','overlap_available','full_pool_5s')
PROBE_FIELDS = tuple(f'{role}.{field}' for role in ('front','rear') for field in FOLLOW_FIELDS
                    )+tuple('conflict.'+field for field in CROSS_FIELDS)
NAMES = tuple(f'probe_{c}.{field}' for c in range(5) for field in PROBE_FIELDS)+METADATA
DIMENSION = len(NAMES)


def contract():
    return {'version':VERSION,'release':'offline_prototype_only','dimension':DIMENSION,
        'names':list(NAMES),'probe_fields':list(PROBE_FIELDS),'root_metadata':list(METADATA),
        'tick_s':.2,'horizon_steps':25,'root_included':True,'neighbor_capacity_excluding_ego':12,
        'mean_tracks_first':True,'geometry':'unchanged_RoadConsequenceMap_v1',
        'projection_radius_m':PROJECTION_RADIUS_M,'progress_nonreversal_eps_m':EPS,
        'gap_tie_eps_m':GAP_TIE_EPS_M,'window_tie_eps_s':TIME_TIE_EPS_S,
        'prefix':'stop_before_first_v1_invalid_point_never_reenter',
        'following_pool':'ego_and_all_selected_physical_neighbors_same_prefix',
        'cross_pool':'ego_and_all_eligible_cross_approach_neighbors_same_prefix',
        'minimum_semantics':'sampled_supported_prefix_not_full_horizon_minimum',
        'overlap_semantics':'positive_duration_within_shared_prefix_not_collision',
        'censoring':'relative_to_shared_prefix_end_not_imputed_5s_exit',
        'cross_selection':'overlap_then_unreached_then_separation_stable_id',
        'numeric_missing_placeholder':0.,'clipping':False,'forecast_repair':False,
        'unknown_suffix_safe':False,'online_registration':False,'old_policy_weights_compatible':False,
        'all_actor_safety_claim':False}


@dataclass(frozen=True)
class Prefix:
    track: object
    length: int

    @property
    def horizon_s(self): return (self.length-1)*.2

    @property
    def complete(self): return self.length == 26

    def diagnostics(self):
        return {'supported_points_including_root':self.length,'horizon_s':self.horizon_s,
                'full_5s':self.complete,'first_unavailable_time_s':None if self.complete else self.length*.2,
                'first_unavailable_reason':None if self.complete else self.track.reasons[self.length]}


def supported_prefix(track):
    if (track.progress.shape != (26,) or track.valid.shape != (26,) or track.speed.shape != (26,)
            or track.valid.dtype != np.bool_
            or not np.isfinite(track.progress).all() or not np.isfinite(track.speed).all()
            or len(track.reasons) != 26 or len(track.lanes) != 26):
        raise ValueError('Malformed v1 projected track')
    if not track.valid[0]: raise ValueError('Observed root is invalid, not censored future')
    bad = np.flatnonzero(~track.valid)
    return Prefix(track,26 if not len(bad) else int(bad[0]))


def prefix_window(prefix, interval, length=None):
    n = prefix.length if length is None else length
    if type(n) is not int or not 1 <= n <= prefix.length:
        raise ValueError('Cannot extend beyond supported prefix')
    lo,hi = interval
    if not all(math.isfinite(v) for v in interval) or lo > hi:
        raise ValueError('Need finite control-volume interval')
    s = prefix.track.progress[:n]; horizon = (n-1)*.2
    if not prefix.track.valid[:n].all(): raise ValueError('Cannot use a known-invalid prefix point')
    if np.any(np.diff(s) < -EPS): raise ValueError('Prefix must not contain a known reversal')
    if s[0] > hi+EPS:
        return {'horizon_s':horizon,'entry_s':None,'exit_s':None,'reached':False,
                'left_censored':False,'right_censored':False,'passed_before_root':True}
    def crossing(boundary):
        if s[0] >= boundary-EPS: return 0.
        indices = np.flatnonzero(s >= boundary-EPS)
        if not len(indices): return None
        i = int(indices[0])
        return float((i-1+(boundary-s[i-1])/(s[i]-s[i-1]))*.2)
    entry = crossing(lo); exit_time = crossing(hi) if np.any(s > hi+EPS) else None
    return {'horizon_s':horizon,'entry_s':entry,'exit_s':exit_time,'reached':entry is not None,
            'left_censored':bool(s[0] >= lo-EPS),'right_censored':exit_time is None,'passed_before_root':False}


def prefix_overlap(ego, actor):
    if ego['horizon_s'] != actor['horizon_s']: raise ValueError('Need identical pair support')
    duration = 0.; first = None
    if ego['reached'] and actor['reached']:
        end_ego = ego['horizon_s'] if ego['exit_s'] is None else ego['exit_s']
        end_actor = actor['horizon_s'] if actor['exit_s'] is None else actor['exit_s']
        start = max(ego['entry_s'],actor['entry_s'])
        raw_duration = min(end_ego,end_actor)-start
        if raw_duration > TIME_TIE_EPS_S: duration,first = float(raw_duration),float(start)
    return {'supported_overlap_s':duration,'first_overlap_s':first,
            'overlap_available':ego['horizon_s'] > 0,'scope':'shared_prefix_only'}


def following(ego, actors, states, geometry, direction):
    n = min([ego.length]+[p.length for p in actors.values()]); horizon = (n-1)*.2
    rows = []
    for actor in sorted(actors):
        t = actors[actor].track; et = ego.track
        for i in range(n):
            if not geometry.same_path({'lane_id':et.lanes[i]},{'lane_id':t.lanes[i]}): continue
            delta = t.progress[i]-et.progress[i]
            if abs(delta) <= GAP_TIE_EPS_M: delta = 0.
            if (direction == 'front') != (delta >= 0): continue
            gap = delta-states[actor]['length_m'] if direction == 'front' else -delta-states['ego']['length_m']
            closing = et.speed[i]-t.speed[i] if direction == 'front' else t.speed[i]-et.speed[i]
            rows.append((float(gap),i,actor,float(closing)))
    best = min((r[0] for r in rows),default=None)
    chosen = None if best is None else min((r for r in rows if r[0] <= best+GAP_TIE_EPS_M),key=lambda r:(r[1],r[2]))
    available = chosen is not None and n > 1
    numeric = [0.,0.,0.] if not available else [chosen[0]/100.,chosen[3]/30.,chosen[1]*.2/5.]
    return numeric+[float(bool(rows)),float(available),horizon/5.,float(n == 26)], {
        'selected_actor':None if chosen is None else chosen[2],'minimum_available':available,
        'shared_prefix_points':n,'shared_prefix_horizon_s':horizon,'full_pool_5s':n == 26,
        'all_pool_actor_ids':sorted(actors),'scope':'supported_prefix_not_global_minimum'}


def conflict(ego, actors, states, road):
    group = road.geometry.group(states['ego'])
    eligible = sorted(a for a in actors if {group,road.geometry.group(states[a])} == {'ramp','main_approach'})
    if not eligible: return [0.]*14, {'selected_actor':None,'eligible_actors':[],'pairs':{}}
    n = min([ego.length]+[actors[a].length for a in eligible]); horizon = (n-1)*.2
    ego_interval = road.zone_interval(states['ego']); ew = prefix_window(ego,ego_interval,n)
    pairs = {}
    for actor in eligible:
        interval = road.zone_interval(states[actor]); aw = prefix_window(actors[actor],interval,n)
        deficit = max(0.,ego_interval[0]-ego.track.progress[n-1],interval[0]-actors[actor].track.progress[n-1])
        pairs[actor] = {'ego':ew,'actor':aw,**prefix_overlap(ew,aw),'entry_progress_deficit_m':float(deficit)}
    def score(actor):
        p = pairs[actor]
        if p['supported_overlap_s'] > 0: return 0,-p['supported_overlap_s'],p['first_overlap_s']
        if not (p['ego']['reached'] and p['actor']['reached']): return 1,p['entry_progress_deficit_m'],0.
        return 2,abs(p['ego']['entry_s']-p['actor']['entry_s']),0.
    category = min(score(a)[0] for a in eligible)
    primary = min(score(a)[1] for a in eligible if score(a)[0] == category)
    tolerance = GAP_TIE_EPS_M if category == 1 else TIME_TIE_EPS_S
    tied = [a for a in eligible if score(a)[0] == category and score(a)[1] <= primary+tolerance]
    secondary = min(score(a)[2] for a in tied)
    chosen = min(a for a in tied if score(a)[2] <= secondary+TIME_TIE_EPS_S)
    p = pairs[chosen]; aw = p['actor']
    values = [0. if v is None else v/5. for v in (ew['entry_s'],aw['entry_s'],p['supported_overlap_s'],p['first_overlap_s'])]
    values += [1.,horizon/5.,*map(float,(ew['reached'],aw['reached'],ew['left_censored'],aw['left_censored'],
        ew['right_censored'],aw['right_censored'],p['overlap_available'],n == 26))]
    return values, {'selected_actor':chosen,'eligible_actors':eligible,'pairs':pairs,
        'shared_prefix_points':n,'shared_prefix_horizon_s':horizon,'full_pool_5s':n == 26,
        'scope':'supported_occupancy_not_collision_or_safety'}


def summarize_road_consequences_v2(history, neighbors, ego, road):
    if not isinstance(road,RoadConsequenceMap): raise ValueError('Need unchanged pinned road geometry')
    inputs = build_actor_inputs(history,road.geometry); neighbors,ego = array(neighbors),array(ego)
    if (neighbors.ndim != 4 or neighbors.shape[1:] != (12,25,4) or not neighbors.shape[0]
            or ego.shape != (neighbors.shape[0],25,4) or not np.isfinite(neighbors).all() or not np.isfinite(ego).all()):
        raise ValueError('Malformed/nonfinite forecasts, including padding')
    states = history[-1]['vehicles']; origin = np.asarray(states['ego']['position'])
    for actor,state in states.items():
        if np.linalg.norm(road.xy(state['lane_id'],state['lane_position_m'])-state['position']) > 1e-5:
            raise ValueError('Observed lane position/XY mismatch: '+actor)
    values = []; candidates = []
    for c in range(len(ego)):
        ep = supported_prefix(road.track(states['ego'],ego[c],origin))
        actors = {a:supported_prefix(road.track(states[a],neighbors[c,k],origin))
                  for k,a in enumerate(inputs['actor_ids']) if a is not None}
        front,fd = following(ep,actors,states,road.geometry,'front')
        rear,rd = following(ep,actors,states,road.geometry,'rear')
        cross,cd = conflict(ep,actors,states,road)
        values.extend(front+rear+cross)
        candidates.append({'front':fd,'rear':rd,'conflict':cd,'ego_prefix':ep.diagnostics(),
                           'actor_prefixes':{a:p.diagnostics() for a,p in actors.items()}})
    features = np.asarray(values+root_metadata(inputs),dtype=np.float32)
    if not np.isfinite(features).all(): raise ValueError('Feature conversion overflow')
    return {'version':VERSION,'features':features,'candidates':candidates,'actor_ids':inputs['actor_ids'],
        'root_omitted_ids':inputs['root_omitted_ids'],'ensemble_reduction':'mean_tracks_before_consequences',
        'online_release':False,'all_actor_safety_claim':False,'unknown_suffix_safe':False}
