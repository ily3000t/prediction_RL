"""Explain frozen v1 invalidity; do not repair forecasts or change masks."""
from collections import Counter

import numpy as np

from prediction_rl.data.actor_adapter import build_actor_inputs
from prediction_rl.evaluation.rcci_frozen_audit import MODES, native_prefix
from prediction_rl.features.road_consequence_geometry import EPS
from prediction_rl.features.road_consequences import PROBE_FIELDS, summarize_road_consequences

VERSION = 'rcci02_reversal_and_field_diagnostics_v1'
SETTINGS = {'version':VERSION,'parent_request':'artifacts/rcci02/rcci02_bnd_v1/request.json',
    'parent_request_hash':'02d6730a98e3a3a1a3a3065fb9a2d66ff49967ceed10addd3d40a608313d3166',
    'parent_report_sha256':'6c5f8f9ba0b35089d4996fc3b5fdbf487cdbc65398fb4c0685730e1bb22415a4',
    'parent_roots_sha256':'f30cc77926e44d82648e8c227e4fe8306a5e86f86f4dcee909c3f7f938244741',
    'scope':'same_six_roots_no_expansion','splits':['train','validation'],
    'torch_threads':1,'max_model_forward_calls':12,'save_mean_forecasts':True,
    'change_adapter_or_thresholds':False,'training':False,'simulation':False,
    'calibration_access':False,'test_access':False}
NUMERIC = (0,1,2,6,7,8,12,13,14,15,16,17)


def describe(values):
    x = np.asarray(values,dtype=np.float64)
    if not len(x): return {'count':0,'min':None,'median':None,'p95':None,'max':None,'mean':None}
    if not np.isfinite(x).all(): raise ValueError('Nonfinite diagnostic statistic')
    return {'count':len(x),'min':float(x.min()),'median':float(np.median(x)),
            'p95':float(np.quantile(x,.95)),'max':float(x.max()),'mean':float(x.mean())}


def track_diagnostics(root, future, origin, road, *, native_progress=None):
    a = np.asarray(future,dtype=np.float64)
    track = road.track(root,a,origin)  # Unmodified v1, including tail masking.
    xy = np.vstack((root['position'],a[:,:2]+origin))
    raw = [road.project(p,root['lane_id']) for p in xy]
    raw_valid = np.asarray([p['valid'] for p in raw])
    delta = np.diff(track.progress); reverse = delta < -EPS
    intrinsic_bad = ~raw_valid | (track.speed < 0)
    direct_reverse = np.r_[False,reverse]
    # Later projectable points are NOT declared recovered/usable: the v1
    # full-window non-reversal assumption has already failed.
    after_reverse_only = ~track.valid & ~intrinsic_bad & ~direct_reverse
    ulp = np.spacing(np.abs(a[:,:2].astype(np.float32))).astype(np.float64)
    if not np.isfinite(ulp).all(): raise ValueError('Require finite float32 forecast storage scale')
    ulp_max = float(ulp.max())
    native = None if native_progress is None else np.asarray(native_progress,dtype=np.float64)
    if native is not None and (native.ndim != 1 or not 1 <= len(native) <= 26 or not np.isfinite(native).all()):
        raise ValueError('Invalid actual observed prefix')
    events = []
    for index in np.flatnonzero(reverse):
        step = int(index+1); native_delta = None
        if native is not None and step < len(native): native_delta = float(native[step]-native[step-1])
        events.append({'step':step,'time_s':step*.2,'backward_step_m':float(-delta[index]),
            'raw_projection_valid_both':bool(raw_valid[index] and raw_valid[step]),
            'world_xy_displacement_m':float(np.linalg.norm(xy[step]-xy[index])),
            'predicted_speed_start_mps':float(track.speed[index]),
            'predicted_speed_end_mps':float(track.speed[step]),
            'ds_minus_end_speed_dt_m':float(delta[index]-track.speed[step]*.2),
            'observed_native_step_m':native_delta,
            'coordinate_storage_ulp_max_m':ulp_max,
            'backward_over_coordinate_ulp':None if ulp_max == 0 else float(-delta[index]/ulp_max)})
    consistent_support = raw_valid[:-1] & raw_valid[1:] & (track.speed[1:] >= 0)
    return {'complete_v1':track.complete,'root_speed_mps':float(root['speed']),
        'root_lane':root['lane_id'],'projected_progress_m':track.progress.tolist(),
        'predicted_speed_mps':track.speed.tolist(),'v1_valid':track.valid.tolist(),
        'v1_invalid_reasons':dict(Counter(r for r in track.reasons[1:] if r != 'valid')),
        'raw_projection_invalid_reasons':dict(Counter(p['reason'] for p in raw[1:] if not p['valid'])),
        'actual_reverse_transition_count':len(events),'reversal_events':events,
        'first_reverse_time_s':None if not events else events[0]['time_s'],
        'max_drawdown_m':float((np.maximum.accumulate(track.progress)-track.progress).max()),
        'sum_backward_steps_m':float(-delta[reverse].sum()),
        'forecast_net_progress_m':float(track.progress[-1]-track.progress[0]),
        'later_projectable_nonnegative_speed_points_after_reverse':int(after_reverse_only[1:].sum()),
        'negative_speed_points':int((track.speed[1:] < 0).sum()),
        'minimum_predicted_speed_mps':float(track.speed[1:].min()),
        'observed_future_steps':None if native is None else len(native)-1,
        'native_reverse_transition_count':None if native is None else int((np.diff(native) < -EPS).sum()),
        'ds_minus_end_speed_dt_m':(delta-track.speed[1:]*.2)[consistent_support].tolist()}


def numeric_support(row):
    f = np.asarray(row)
    if f.shape != (28,) or not np.isfinite(f).all(): raise ValueError('Require v1 candidate fields')
    support = {k:False for k in NUMERIC}
    for k in (0,1,2): support[k] = bool(f[3] and f[4])
    for k in (6,7,8): support[k] = bool(f[9] and f[10])
    if f[18] and f[19]:
        support.update({12:bool(f[20]),13:bool(f[20] and not f[24]),
                        14:bool(f[21]),15:bool(f[21] and not f[25]),
                        16:bool(f[26]),17:bool(f[27])})
    return support


def margin_unavailable_reasons(pair):
    if pair['signed_margin_s'] is not None: return []
    reasons = []
    for role in ('ego','actor'):
        w = pair[role]
        if not w['valid']:
            reasons.append(role+'_projection_incomplete'); continue
        if not w['reached']: reasons.append(role+'_not_reached_in_horizon')
        if w['left_censored']: reasons.append(role+'_left_censored')
        if w['right_censored']: reasons.append(role+'_right_censored')
        if w['passed_before_root']: reasons.append(role+'_passed_before_root')
    if not reasons: raise ValueError('Unexplained v1 margin invalidity')
    return reasons


def diagnose_root(history, predictions, ego, branches, road):
    if set(predictions) != set(MODES) or len(branches) != 5:
        raise ValueError('Need both frozen forecasts and all five native branches')
    inputs = build_actor_inputs(history,road.geometry); root = history[-1]
    states = root['vehicles']; origin = np.asarray(states['ego']['position']); modes = {}
    intervals = {actor:road.zone_interval(states[actor]) for actor in
                 ['ego']+[a for a in inputs['actor_ids'] if a is not None]}
    for mode in MODES:
        features = summarize_road_consequences(history,predictions[mode],ego,road)
        fields = features['features'][:140].reshape(5,28); tracks = []; candidates = []
        for c in range(5):
            invalid_ids = []
            for slot,actor in enumerate(inputs['actor_ids']):
                if actor is None: continue
                frames = native_prefix(root,branches[c],actor)
                progress = [road.geometry.progress(f['vehicles'][actor]) for f in frames]
                diag = track_diagnostics(states[actor],predictions[mode][c,slot],origin,road,native_progress=progress)
                diag.update(actor_id=actor,candidate_id=c,
                    mandatory_roles=sorted(k for k,v in inputs['mandatory_roles'].items() if v == actor),
                    control_volume_progress_interval_m=list(intervals[actor]),
                    control_volume_span_m=intervals[actor][1]-intervals[actor][0])
                if not diag['complete_v1']: invalid_ids.append(actor)
                tracks.append(diag)
            conflict = features['candidates'][c]['conflict']; pairs = conflict['pairs']
            chosen = conflict['selected_actor']
            all_reasons = Counter(reason for p in pairs.values() for reason in margin_unavailable_reasons(p))
            support = numeric_support(fields[c])
            candidates.append({'candidate_id':c,'invalid_actor_ids':invalid_ids,
                'front_applicable':bool(fields[c,3]),'rear_applicable':bool(fields[c,9]),
                'following_pool_complete':bool(fields[c,4]),
                'ego_invalid_reasons':features['candidates'][c]['ego_invalid_reasons'],
                'front_selected_actor':features['candidates'][c]['front']['selected_actor'],
                'rear_selected_actor':features['candidates'][c]['rear']['selected_actor'],
                'numeric_support':{PROBE_FIELDS[k]:v for k,v in support.items()},
                'numeric_values':{PROBE_FIELDS[k]:float(fields[c,k]) for k in NUMERIC},
                'cross_applicable':chosen is not None,'complete_cross_pool':bool(fields[c,19]) if chosen else None,
                'eligible_cross_pairs':len(pairs),
                'all_pair_margin_valid':sum(p['signed_margin_s'] is not None for p in pairs.values()),
                'all_pair_margin_unavailable_reasons':dict(all_reasons),
                'selected_cross_actor':chosen,
                'selected_pair_margin_unavailable_reasons':[] if chosen is None else margin_unavailable_reasons(pairs[chosen]),
                'selected_pair':None if chosen is None else pairs[chosen]})
        modes[mode] = {'tracks':tracks,'candidates':candidates,'feature_vector':features['features'].tolist()}
    ego_tracks = []
    for c in range(5):
        frames = native_prefix(root,branches[c],'ego')
        progress = [road.geometry.progress(f['vehicles']['ego']) for f in frames]
        ego_tracks.append(track_diagnostics(states['ego'],ego[c],origin,road,native_progress=progress))
    return {'methods':modes,'ego_tracks':ego_tracks,'root_actor_ids':inputs['actor_ids'],
            'root_omitted_ids':inputs['root_omitted_ids']}


def aggregate(rows):
    output = {}
    for split in SETTINGS['splits']:
        group = [r for r in rows if r['split'] == split]; methods = {}
        for mode in MODES:
            tracks = [t for r in group for t in r['methods'][mode]['tracks']]
            candidates = [c for r in group for c in r['methods'][mode]['candidates']]
            events = [e for t in tracks for e in t['reversal_events']]
            counts = Counter(actor_tracks=len(tracks),invalid_actor_tracks=sum(not t['complete_v1'] for t in tracks),
                actual_reverse_transitions=len(events),
                reverse_tracks=sum(bool(t['reversal_events']) for t in tracks),
                reverse_transitions_with_valid_raw_projection=sum(e['raw_projection_valid_both'] for e in events),
                reverse_transitions_with_positive_reported_end_speed=sum(e['predicted_speed_end_mps'] > 0 for e in events),
                reverse_transitions_inside_native_support=sum(e['observed_native_step_m'] is not None for e in events),
                reverse_transitions_outside_native_support=sum(e['observed_native_step_m'] is None for e in events),
                native_reverse_transitions=sum(t['native_reverse_transition_count'] for t in tracks),
                negative_speed_points=sum(t['negative_speed_points'] for t in tracks),
                later_projectable_points_after_reverse=sum(t['later_projectable_nonnegative_speed_points_after_reverse'] for t in tracks),
                invalid_mandatory_actor_tracks=sum(not t['complete_v1'] and bool(t['mandatory_roles']) for t in tracks),
                candidate_tables=len(candidates),cross_applicable=sum(c['cross_applicable'] for c in candidates),
                front_applicable=sum(c['front_applicable'] for c in candidates),
                front_applicable_but_pool_incomplete=sum(c['front_applicable'] and not c['following_pool_complete'] for c in candidates),
                rear_applicable=sum(c['rear_applicable'] for c in candidates),
                rear_applicable_but_pool_incomplete=sum(c['rear_applicable'] and not c['following_pool_complete'] for c in candidates),
                cross_applicable_but_pool_incomplete=sum(c['cross_applicable'] and not c['complete_cross_pool'] for c in candidates),
                all_eligible_cross_pairs=sum(c['eligible_cross_pairs'] for c in candidates),
                all_pair_margin_valid=sum(c['all_pair_margin_valid'] for c in candidates))
            reasons = Counter(reason for c in candidates for reason in c['selected_pair_margin_unavailable_reasons'])
            all_reasons = Counter()
            for c in candidates: all_reasons.update(c['all_pair_margin_unavailable_reasons'])
            field_support = {name:sum(c['numeric_support'][name] for c in candidates)
                             for name in (PROBE_FIELDS[k] for k in NUMERIC)}
            methods[mode] = {'counts':dict(counts),'backward_step_m':describe([e['backward_step_m'] for e in events]),
                'reverse_time_s':describe([e['time_s'] for e in events]),
                'first_reverse_time_per_track_s':describe([t['first_reverse_time_s'] for t in tracks if t['reversal_events']]),
                'backward_step_over_coordinate_ulp':describe([e['backward_over_coordinate_ulp'] for e in events if e['backward_over_coordinate_ulp'] is not None]),
                'max_drawdown_per_reversing_track_m':describe([t['max_drawdown_m'] for t in tracks if t['reversal_events']]),
                'abs_ds_minus_end_speed_dt_m':describe([abs(v) for t in tracks for v in t['ds_minus_end_speed_dt_m']]),
                'negative_predicted_speeds_mps':describe([v for t in tracks for v in t['predicted_speed_mps'][1:] if v < 0]),
                'numeric_field_support_counts':field_support,
                'selected_margin_unavailable_reasons_nonexclusive':dict(reasons),
                'all_pair_margin_unavailable_reasons_nonexclusive':dict(all_reasons)}
        output[split] = {'roots':len(group),'unique_episodes':len({r['episode_id'] for r in group}),
            'methods':methods,'counts_are_not_independent_scenarios':True}
    return output
