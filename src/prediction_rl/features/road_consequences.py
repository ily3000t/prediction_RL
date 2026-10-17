"""RCCI02-A mean-track, offline-only 149-channel consequence adapter.

No environment/provider registration, weights, labels, future masks or action
selection. Raw remains the default. This is not an online release or a shield.
"""
from collections import Counter

import numpy as np

from prediction_rl.data.actor_adapter import build_actor_inputs
from prediction_rl.features.road_consequence_geometry import (
    RoadConsequenceMap, PROJECTION_RADIUS_M, occupancy_window, window_pair,
)
from prediction_rl.features.probe_features import METADATA

VERSION = 'rcci02_a_mean_track_control_volume_v1'
FOLLOW_FIELDS = ('min_route_gap_over_100m', 'closing_at_min_over_30mps',
                 'time_at_min_over_5s', 'applicable', 'complete_pool_projection',
                 'relation_sample_fraction')
CONFLICT_FIELDS = ('ego_entry_over_5s', 'ego_exit_over_5s', 'actor_entry_over_5s',
    'actor_exit_over_5s', 'observed_overlap_over_5s', 'signed_margin_over_5s',
    'applicable', 'complete_pool_projection', 'ego_reached', 'actor_reached',
    'ego_left_censored', 'actor_left_censored', 'ego_right_censored',
    'actor_right_censored', 'overlap_valid', 'margin_valid')
PROBE_FIELDS = tuple(f'{role}.{name}' for role in ('front', 'rear') for name in FOLLOW_FIELDS
                    ) + tuple('conflict.'+name for name in CONFLICT_FIELDS)
NAMES = tuple(f'probe_{c}.{name}' for c in range(5) for name in PROBE_FIELDS)+METADATA
DIMENSION = len(NAMES)
GAP_TIE_EPS_M = 1e-6  # Numeric tie only, not a safety/effect-size threshold.


def contract():
    """Serializable frozen description, not a mutable experiment config."""
    return {'version':VERSION, 'release':'offline_prototype_only',
        'dimension':DIMENSION, 'names':list(NAMES), 'probe_fields':list(PROBE_FIELDS),
        'root_metadata':list(METADATA), 'tick_s':.2, 'horizon_steps':25, 'root_included':True,
        'neighbor_capacity_excluding_ego':12, 'mean_tracks_first':True,
        'projection_radius_m':PROJECTION_RADIUS_M, 'gap_numeric_tie_m':GAP_TIE_EPS_M,
        'window_pool_numeric_tie_s':1e-8,
        'projection_numeric_tie_m':1e-8, 'progress_ambiguity_m':1e-6,
        'zone':'convex_hull_of_pinned_merge_junction_not_shared_downstream',
        'position_reference':'front_bumper_center', 'footprint':'road_aligned_rectangle',
        'event_interpolation':'piecewise_linear_projected_progress_0_to_5s',
        'following_minimum':'26_grid_points_not_continuous_minimum',
        'numeric_missing_placeholder':0., 'clipping':False,
        'invalid_selected_track':'invalidate_numeric_pool_not_drop_actor',
        'cross_pair_pool':'invalid_then_overlap_then_unresolved_then_separation_stable_id',
        'unknown_prediction_errors':'raise', 'finite_off_path_prediction':'explicit_projection_mask',
        'online_registration':False, 'raw_policy_weights_compatible':False,
        'all_actor_safety_claim':False}


def array(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float64)


def root_metadata(inputs):
    result = []
    for role in ('target_front', 'target_rear'):
        actor = inputs['mandatory_roles'].get(role)
        result.extend((float(actor is not None), 0. if actor is None else
                       float(np.mean(inputs['actor_history_mask'][inputs['actor_ids'].index(actor)]))))
    selected = sum(inputs['actor_mask']); omitted = len(inputs['root_omitted_ids'])
    return result+[float(selected > 0), float(np.mean(inputs['ego_history_mask'])),
                   selected/12., omitted/(omitted+12.), float(omitted > 0)]


def following(ego, actors, states, geometry, direction):
    """Dynamic physical-slot pool; route gap is not Euclidean body clearance.

    Any invalid selected actor invalidates the numeric pool summary, rather
    than silently deleting that actor and presenting the remainder as safe.
    """
    rows = []; relation_cells = 0
    complete = ego.complete and all(t.complete for t in actors.values())
    for actor in sorted(actors):
        track = actors[actor]
        for i in range(26):
            if not ego.valid[i] or not track.valid[i]: continue
            if not geometry.same_path({'lane_id':ego.lanes[i]}, {'lane_id':track.lanes[i]}): continue
            delta = track.progress[i]-ego.progress[i]
            if abs(delta) <= GAP_TIE_EPS_M: delta = 0.
            if (direction == 'front') != (delta >= 0): continue
            relation_cells += 1
            gap = (delta-states[actor]['length_m'] if direction == 'front'
                   else -delta-states['ego']['length_m'])
            closing = (ego.speed[i]-track.speed[i] if direction == 'front'
                       else track.speed[i]-ego.speed[i])
            rows.append((float(gap), i, actor, float(closing)))
    best = min((r[0] for r in rows), default=None)
    selected = (None if best is None else min((r for r in rows if r[0] <= best+GAP_TIE_EPS_M),
                                             key=lambda r:(r[1],r[2])))
    numeric = [0., 0., 0.] if selected is None or not complete else [selected[0]/100., selected[3]/30., selected[1]*.2/5.]
    return numeric+[float(bool(rows)), float(complete), relation_cells/max(1, 26*len(actors))], {
        'selected_actor':None if selected is None else selected[2],
        'minimum_valid':bool(selected is not None and complete),
        'complete_pool_projection':bool(complete), 'relation_samples':relation_cells}


def conflict(ego, actors, states, road):
    eg = road.geometry.group(states['ego'])
    # Only distinct incoming approaches require crossing-order information.
    # A downstream actor is handled as connected-route following; its route
    # of origin is not invented from future labels.
    eligible = sorted(a for a in actors if {eg, road.geometry.group(states[a])} == {'ramp', 'main_approach'})
    if not eligible:
        return [0.]*16, {'selected_actor':None, 'eligible_actors':[], 'pairs':{}}
    ew = occupancy_window(ego, road.zone_interval(states['ego']))
    pairs = {}
    for actor in eligible:
        aw = occupancy_window(actors[actor], road.zone_interval(states[actor]))
        p = window_pair(ew, aw)
        pairs[actor] = {'ego':ew, 'actor':aw, **p}
    def key(actor):
        p = pairs[actor]; margin = p['signed_margin_s']; overlap = p['overlap_s']
        # Invalid > observed overlap > unresolved/censored > finite separation.
        # This is a deterministic descriptive pooling rule, not an action mask.
        category = (0 if not p['valid'] else 1 if overlap is not None and overlap > 0
                    else 2 if margin is None else 3)
        return category, (-(overlap or 0.) if category == 1 else margin or 0.), actor
    category = min(key(a)[0] for a in eligible)
    score = min(key(a)[1] for a in eligible if key(a)[0] == category)
    chosen = min(a for a in eligible if key(a)[0] == category and key(a)[1] <= score+1e-8)
    p = pairs[chosen]; aw = p['actor']
    complete = all(pair['valid'] for pair in pairs.values())
    values = [0.]*16
    values[6:8] = [1., float(complete)]
    if complete:
        times = [ew['entry_s'], ew['exit_s'], aw['entry_s'], aw['exit_s'],
                 p['overlap_s'], p['signed_margin_s']]
        values[:6] = [0. if t is None else t/5. for t in times]
        values[8:] = list(map(float, (ew['reached'], aw['reached'], ew['left_censored'],
            aw['left_censored'], ew['right_censored'], aw['right_censored'],
            p['overlap_s'] is not None, p['signed_margin_s'] is not None)))
    return values, {'selected_actor':chosen, 'eligible_actors':eligible, 'pairs':pairs,
                    'complete_pool_projection':complete}


def summarize_road_consequences(history, neighbors, ego, road):
    """Return features plus transparent per-candidate audit diagnostics.

    neighbors [candidate,12,25,4] must be the SAME ensemble-mean forecast as
    Raw; ego [candidate,25,4] is the unchanged fixed-probe reference rollout.
    Accept any positive candidate count for batch/single parity. Production
    five-probe validation stays in the existing rollout, not relaxed here.
    """
    if not isinstance(road, RoadConsequenceMap):
        raise ValueError('Need versioned pinned road geometry')
    inputs = build_actor_inputs(history, road.geometry)
    neighbors, ego = array(neighbors), array(ego)
    if (neighbors.ndim != 4 or neighbors.shape[1:] != (12,25,4)
            or not neighbors.shape[0] or ego.shape != (neighbors.shape[0],25,4)
            or not np.isfinite(neighbors).all() or not np.isfinite(ego).all()):
        raise ValueError('Malformed/nonfinite trajectories, including padding')
    root = history[-1]; states = root['vehicles']; origin = np.asarray(states['ego']['position'])
    # Root lane metadata must agree with the map before forecasting. Unknown
    # or inconsistent observations are an engineering error, not censored risk.
    for actor, state in states.items():
        if np.linalg.norm(road.xy(state['lane_id'], state['lane_position_m'])-state['position']) > 1e-5:
            raise ValueError('Observed lane position/XY mismatch: '+actor)
    ids = inputs['actor_ids']; values = []; diagnostics = []
    for c in range(len(ego)):
        et = road.track(states['ego'], ego[c], origin)
        tracks = {actor:road.track(states[actor], neighbors[c,slot], origin)
                  for slot,actor in enumerate(ids) if actor is not None}
        front, fd = following(et, tracks, states, road.geometry, 'front')
        rear, rd = following(et, tracks, states, road.geometry, 'rear')
        cross, cd = conflict(et, tracks, states, road)
        values.extend(front+rear+cross)
        diagnostics.append({'front':fd, 'rear':rd, 'conflict':cd,
            'ego_invalid_reasons':dict(Counter(r for r in et.reasons if r != 'valid')),
            'actor_invalid_reasons':{a:dict(Counter(r for r in t.reasons if r != 'valid'))
                                     for a,t in tracks.items() if not t.complete},
            'projection_complete_actors':sum(t.complete for t in tracks.values())})
    features = np.asarray(values+root_metadata(inputs), dtype=np.float32)
    if not np.isfinite(features).all():
        raise ValueError('Feature conversion overflow')
    return {'version':VERSION, 'features':features, 'candidates':diagnostics,
            'actor_ids':ids, 'root_omitted_ids':inputs['root_omitted_ids'],
            'all_actor_safety_claim':False, 'ensemble_reduction':'mean_tracks_before_consequences',
            'online_release':False}
