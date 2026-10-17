"""Bounded frozen-forecast diagnostics, never an online feature provider.

Native future lanes/existence are OFFLINE labels only. Errors use identical
observed, projectable support for both models; missing events retain counts.
"""
from collections import Counter
import math

import numpy as np

from prediction_rl.data.actor_adapter import build_actor_inputs
from prediction_rl.features.road_consequences import summarize_road_consequences

VERSION = 'rcci02_frozen_predictor_bounded_audit_v1'
MODES = ('ordinary', 'conditional')
SETTINGS = {'version':VERSION, 'splits':['train','validation'], 'roots_per_split':3,
    'cohort':'frozen_review_prefix_no_outcome_filter', 'scope':'bounded_only',
    'training_report':'artifacts/training/predictor_v1/invocations/648e96637f5e/report.json',
    'prior_report':'artifacts/rcci01/rcci01_full_v1/report.json',
    'prior_report_sha256':'47665549cb455cd214a878aa8cae8985cbcb82c5aebf47bc401c59421d623fe3',
    'methods':list(MODES), 'torch_threads':1, 'simulation':False, 'training':False,
    'calibration_access':False, 'test_access':False, 'online_release':False,
    'comparison':'same_observed_prefix_and_joint_projection_support',
    'counterfactual_sensitivity':'repeat_zero_jerk_neighbor_response_keep_all_ego_probes',
    'numeric_event_tie_s':1e-8}


def observed_window(progress, interval):
    """Same v1 geometry/interpolation, but explicitly shorter observed support.

    No terminal/padded tail is invented. Exits after this horizon are unknown.
    Kept independent from the fixed-26 online prototype to avoid relaxing it.
    """
    s = np.asarray(progress,dtype=np.float64); lo,hi = interval
    if (s.ndim != 1 or not 1 <= len(s) <= 26 or not np.isfinite(s).all()
            or np.any(np.diff(s) < -1e-8) or not all(math.isfinite(v) for v in interval) or lo > hi):
        raise ValueError('Invalid observed non-reversing prefix')
    horizon = (len(s)-1)*.2
    if s[0] > hi+1e-8:
        return {'entry_s':None,'exit_s':None,'left_censored':False,'right_censored':False,
                'reached':False,'passed_before_root':True,'horizon_s':horizon}
    def crossing(bound):
        if s[0] >= bound-1e-8: return 0.
        indices = np.flatnonzero(s >= bound-1e-8)
        if not len(indices): return None
        i = int(indices[0])
        return float((i-1+(bound-s[i-1])/(s[i]-s[i-1]))*.2)
    entry = crossing(lo); exit_time = crossing(hi) if np.any(s > hi+1e-8) else None
    return {'entry_s':entry, 'exit_s':exit_time, 'left_censored':bool(s[0] >= lo-1e-8),
            'right_censored':exit_time is None, 'reached':entry is not None,
            'passed_before_root':False,'horizon_s':horizon}


def observed_pair(ego, actor):
    if ego['horizon_s'] != actor['horizon_s']:
        raise ValueError('Pair windows must share observed horizon')
    overlap = 0.; margin = order = None
    if ego['reached'] and actor['reached']:
        end_ego = ego['horizon_s'] if ego['exit_s'] is None else ego['exit_s']
        end_actor = actor['horizon_s'] if actor['exit_s'] is None else actor['exit_s']
        overlap = max(0.,min(end_ego,end_actor)-max(ego['entry_s'],actor['entry_s']))
        if (not ego['left_censored'] and not actor['left_censored']
                and ego['exit_s'] is not None and actor['exit_s'] is not None):
            margin = max(ego['entry_s'],actor['entry_s'])-min(end_ego,end_actor)
            delta = actor['entry_s']-ego['entry_s']
            order = 0 if abs(delta) <= 1e-8 else (1 if delta > 0 else -1)
    return {'observed_overlap_s':float(overlap),'signed_margin_s':margin,'ego_entry_order':order}


def native_prefix(root, branch, actor):
    """Root-known actor only, contiguous usable future, no fill after absence."""
    frames = [root]
    future = branch['future_metadata']
    if len(future['frames']) != len(future['trajectory_usable']):
        raise ValueError('Native metadata lengths differ')
    for i,(frame,usable) in enumerate(zip(future['frames'],future['trajectory_usable']),1):
        if not math.isclose(frame['simulation_time_s']-root['simulation_time_s'],i*.2,abs_tol=1e-8,rel_tol=0):
            raise ValueError('Native observed grid changed')
        if not usable or actor not in frame['vehicles'] or 'ego' not in frame['vehicles']:
            break
        frames.append(frame)
    return frames


def confusion(truth, prediction):
    return ('tp' if truth else 'fp') if prediction else ('fn' if truth else 'tn')


def compare_branch(history, branch, candidate, predictions, ego_rollout, road):
    """Return small audit rows, not raw trajectories. Labels never reach model."""
    if set(predictions) != set(MODES) or not 0 <= candidate < 5:
        raise ValueError('Require both matched frozen forecasts and candidate ID')
    root = history[-1]; states = root['vehicles']; origin = np.asarray(states['ego']['position'])
    inputs = build_actor_inputs(history,road.geometry)
    egos = {mode:road.track(states['ego'],ego_rollout[candidate],origin) for mode in MODES}
    native_ego = native_prefix(root,branch,'ego'); ne = len(native_ego)
    actual_ego_progress = np.array([road.geometry.progress(f['vehicles']['ego']) for f in native_ego])
    et = egos['ordinary']; ego_valid = et.valid[:ne].copy(); ego_valid[0] = False
    ego_reference = {'complete_5s_projection':et.complete,'forecast_points':25,
        'valid_forecast_points':int(et.valid[1:].sum()),'observed_future_steps':ne-1,
        'invalid_point_reasons':dict(Counter(r for r in et.reasons[1:] if r != 'valid')),
        'progress_errors_m':(et.progress[:ne]-actual_ego_progress)[ego_valid].tolist()}
    rows = []
    for slot,actor in enumerate(inputs['actor_ids']):
        if actor is None: continue
        tracks = {mode:road.track(states[actor],predictions[mode][candidate,slot],origin) for mode in MODES}
        frames = native_prefix(root,branch,actor); n = len(frames)
        progress = np.array([road.geometry.progress(f['vehicles'][actor]) for f in frames])
        ego_progress = np.array([road.geometry.progress(f['vehicles']['ego']) for f in frames])
        interval = road.zone_interval(states[actor]); ego_interval = road.zone_interval(states['ego'])
        truth = observed_window(progress,interval); ego_truth = observed_window(ego_progress,ego_interval)
        cross = {road.geometry.group(states['ego']),road.geometry.group(states[actor])} == {'ramp','main_approach'}
        common_points = np.logical_and.reduce([tracks[m].valid[:n] for m in MODES])
        common_points[0] = False  # Identical observed root must not inflate accuracy.
        common_window = n > 1 and all(tracks[m].valid[:n].all() for m in MODES)
        common_pair = common_window and all(egos[m].valid[:n].all() for m in MODES)
        row = {'actor_id':actor,'observed_future_steps':n-1,'native_right_censored':truth['right_censored'],
            'native_window':truth,'root_cross_approach':cross,'joint_projected_points':int(common_points.sum()),
            'joint_window_support':bool(common_window),'joint_pair_support':bool(common_pair), 'methods':{}}
        for mode in MODES:
            track = tracks[mode]
            counts = Counter({'actor_tracks':1,'complete_actor_tracks':int(track.complete),
                'forecast_points':25,'valid_forecast_points':int(track.valid[1:].sum()),
                'ego_actor_complete_tracks':int(track.complete and egos[mode].complete),
                'observed_future_points':n-1,'valid_observed_points':int(track.valid[1:n].sum())})
            if truth['entry_s'] is not None and truth['exit_s'] is not None and not truth['left_censored']:
                counts['native_complete_future_windows'] += 1
                counts['native_window_outside_raw_four_times'] += int(not any(
                    truth['entry_s'] <= t <= truth['exit_s'] for t in (.2,1.,3.,5.)))
            for reason in track.reasons[1:]:
                if reason != 'valid': counts['invalid_point_'+reason] += 1
            errors = {'progress_m':(track.progress[:n]-progress)[common_points].tolist(),
                      'entry_s':[],'exit_s':[],'overlap_s':[],'margin_s':[]}
            pw = None
            if common_window:
                counts['joint_window_support'] += 1
                pw = observed_window(track.progress[:n],interval)
                # Current occupancy is observed and identical, not a forecast hit.
                if not truth['left_censored'] and not truth['passed_before_root']:
                    counts['entry_'+confusion(truth['reached'],pw['reached'])] += 1
                if not truth['passed_before_root']:
                    counts['exit_'+confusion(truth['exit_s'] is not None,pw['exit_s'] is not None)] += 1
                for event in ('entry_s','exit_s'):
                    if truth[event] is not None and pw[event] is not None and not (event == 'entry_s' and truth['left_censored']):
                        errors[event].append(pw[event]-truth[event])
            if common_pair and cross:
                counts['joint_cross_pair_support'] += 1
                ep = observed_window(egos[mode].progress[:n],ego_interval)
                truth_pair = observed_pair(ego_truth,truth); pred_pair = observed_pair(ep,pw)
                errors['overlap_s'].append(pred_pair['observed_overlap_s']-truth_pair['observed_overlap_s'])
                counts['overlap_'+confusion(truth_pair['observed_overlap_s'] > 1e-8,
                                              pred_pair['observed_overlap_s'] > 1e-8)] += 1
                if truth_pair['signed_margin_s'] is not None and pred_pair['signed_margin_s'] is not None:
                    errors['margin_s'].append(pred_pair['signed_margin_s']-truth_pair['signed_margin_s'])
                    counts['joint_finite_margin_support'] += 1
                    counts['entry_order_correct'] += int(truth_pair['ego_entry_order'] == pred_pair['ego_entry_order'])
            row['methods'][mode] = {'counts':dict(counts),'errors':errors,'predicted_observed_window':pw}
        rows.append(row)
    return {'candidate_id':candidate,'ego_reference':ego_reference,'actors':rows}


def sensitivity(history, predictions, ego, road):
    """Diagnostic replacement only: not causal identification or useful control."""
    inputs = build_actor_inputs(history,road.geometry); mask = np.asarray(inputs['actor_mask'])
    result = {}
    numeric = [0,1,2,6,7,8,12,13,14,15,16,17]
    flags = [i for i in range(28) if i not in numeric]
    for mode in MODES:
        p = np.asarray(predictions[mode]); frozen = np.repeat(p[2:3],5,axis=0)
        actual = summarize_road_consequences(history,p,ego,road)
        held = summarize_road_consequences(history,frozen,ego,road)
        full = actual['features'][:140].reshape(5,28); reference = held['features'][:140].reshape(5,28)
        neighbor_delta = p[:,mask]-frozen[:,mask]
        result[mode] = {'features':actual['features'].tolist(),
            'complete_following_candidates':sum(bool(row[4]) for row in full),
            'cross_applicable_candidates':sum(bool(row[18]) for row in full),
            'complete_cross_candidates':sum(bool(row[18] and row[19]) for row in full),
            'valid_overlap_candidates':sum(bool(row[26]) for row in full),
            'valid_margin_candidates':sum(bool(row[27]) for row in full),
            'neighbor_candidate_xy_max_m':float(np.linalg.norm(neighbor_delta[...,:2],axis=-1).max()) if mask.any() else 0.,
            'neighbor_candidate_speed_max_mps':float(np.abs(neighbor_delta[...,2]).max()) if mask.any() else 0.,
            'neighbor_replacement_numeric_max':float(np.abs(full[:,numeric]-reference[:,numeric]).max()),
            'neighbor_replacement_flag_changes':int(np.count_nonzero(full[:,flags] != reference[:,flags])),
            'all_candidate_numeric_spread_max':float(np.abs(full[:,numeric]-full[2:3,numeric]).max()),
            'all_candidate_flag_changes_from_zero':int(np.count_nonzero(full[:,flags] != full[2:3,flags])),
            'replacement_may_be_ood':True}
    return result


def error_summary(values):
    a = np.asarray(values,dtype=np.float64)
    if not len(a): return {'count':0,'mae':None,'bias':None,'p95_absolute':None,'max_absolute':None}
    if not np.isfinite(a).all(): raise ValueError('Nonfinite audit statistic')
    return {'count':len(a),'mae':float(np.abs(a).mean()),'bias':float(a.mean()),
            'p95_absolute':float(np.quantile(np.abs(a),.95)),'max_absolute':float(np.abs(a).max())}


def aggregate(rows):
    output = {}
    for split in SETTINGS['splits']:
        group = [r for r in rows if r['split'] == split]; methods = {}
        ego_counts = Counter(); ego_errors = []
        for r in group:
            for b in r['branches']:
                e = b['ego_reference']; ego_counts['candidate_rollouts'] += 1
                ego_counts['complete_5s_projection'] += int(e['complete_5s_projection'])
                for key in ('forecast_points','valid_forecast_points','observed_future_steps'):
                    ego_counts[key] += e[key]
                for reason,count in e['invalid_point_reasons'].items(): ego_counts['invalid_point_'+reason] += count
                ego_errors.extend(e['progress_errors_m'])
        for mode in MODES:
            counts = Counter(); errors = {k:[] for k in ('progress_m','entry_s','exit_s','overlap_s','margin_s')}
            for r in group:
                s = r['sensitivity'][mode]
                counts.update({k:s[k] for k in ('complete_following_candidates','cross_applicable_candidates',
                    'complete_cross_candidates','valid_overlap_candidates','valid_margin_candidates')})
                counts['candidate_tables'] += 5
                for b in r['branches']:
                    for a in b['actors']:
                        counts.update(a['methods'][mode]['counts'])
                        for k,v in a['methods'][mode]['errors'].items(): errors[k].extend(v)
            methods[mode] = {'counts':dict(counts),'errors':{k:error_summary(v) for k,v in errors.items()}}
        output[split] = {'roots':len(group),'unique_episodes':len({r['episode_id'] for r in group}),
            'methods':methods,'ego_reference':{'counts':dict(ego_counts),'progress_errors_m':error_summary(ego_errors)},
            'independent_samples_are_episodes_not_actor_candidate_time_cells':True}
    return output
