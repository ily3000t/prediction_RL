"""Frozen-cohort v2 diagnostics. Future labels never construct policy features.

All errors use identical B2/B3/native support. Cells are descriptive; episodes
are the traffic units. This module never loads weights, trains or runs SUMO.
"""
from collections import Counter

import numpy as np

from prediction_rl.data.actor_adapter import build_actor_inputs
from prediction_rl.features.road_consequence_geometry import ProjectedTrack, EPS
from prediction_rl.features.road_consequences_v2 import (
    Prefix, supported_prefix, prefix_window, prefix_overlap, following, conflict,
)
from prediction_rl.evaluation.rcci_frozen_audit import MODES, native_prefix, error_summary, confusion
from prediction_rl.evaluation.rcci_v2_cached_audit import describe

VERSION = 'rcci02_v2_full_cohort_validation_v1'
SETTINGS = {
    'version':VERSION, 'splits':['train','validation'], 'scope':'all_previously_reviewed_roots',
    'expected_roots':{'train':713,'validation':177}, 'expected_episodes':320,
    'design_contract':'docs/design/rcci02_v2_cohort_validation_v1.md',
    'parent_request':'artifacts/rcci02_v2/rcci02_v2_bnd_v1/request.json',
    'parent_request_hash':'82cb25d7eb8e26d14c23d2fec15654f1ca1058fec5d633546eb3a760c55f5a69',
    'parent_report_sha256':'bbc6c3494e917bc707064f4ff6152cd38ba8277ed4db70ecba3d64988b7d4362',
    'parent_roots_sha256':'4b592620ef035006e193582a3f44f7732290f91a318649511918264d0093ea9b',
    'methods':list(MODES), 'torch_threads':1, 'device':'cpu', 'dtype':'float32',
    'max_ensemble_forward_calls':1780, 'max_member_forward_calls':5340,
    'fixed_horizons_s':[1.,3.,5.], 'simulation':False, 'training':False,
    'calibration_access':False, 'test_access':False, 'online_release':False,
    'resume':False, 'skip_failed_roots':False,
}


def project_predictions(history, predictions, ego, road):
    if set(predictions) != set(MODES): raise ValueError('Need matched B2/B3 forecasts')
    if (np.shape(ego) != (5,25,4) or not np.isfinite(ego).all()
            or any(np.shape(p) != (5,12,25,4) or not np.isfinite(p).all() for p in predictions.values())):
        raise ValueError('Require complete finite five-probe forecast arrays')
    states = history[-1]['vehicles']; origin = np.asarray(states['ego']['position'])
    inputs = build_actor_inputs(history,road.geometry)
    return {mode:[(
        supported_prefix(road.track(states['ego'],ego[c],origin)),
        {a:supported_prefix(road.track(states[a],predictions[mode][c,k],origin))
         for k,a in enumerate(inputs['actor_ids']) if a is not None},
    ) for c in range(5)] for mode in MODES}


def pool_length(ego, actors): return min([ego.length]+[p.length for p in actors.values()])
def cut(prefix, n):
    if not 1 <= n <= prefix.length: raise ValueError('Cannot invent support')
    return Prefix(prefix.track,n)


def _following_values(ego, actors, states, road, n):
    return {role:following(cut(ego,n),{a:cut(p,n) for a,p in actors.items()},states,road.geometry,role)[0]
            for role in ('front','rear')}


def _native_pool(root, branch, actor_ids, road):
    """Only offline truth: known future lanes, no projection or tail padding."""
    result = {}
    for actor in ['ego',*actor_ids]:
        frames = native_prefix(root,branch,actor)
        progress = np.array([road.geometry.progress(f['vehicles'][actor]) for f in frames])
        speed = np.array([f['vehicles'][actor]['speed'] for f in frames],dtype=np.float64)
        if (not np.isfinite(progress).all() or not np.isfinite(speed).all()
                or np.any(np.diff(progress) < -EPS) or np.any(speed < 0)):
            raise ValueError('Native observed support must be finite and non-reversing')
        track = ProjectedTrack(progress,tuple(f['vehicles'][actor]['lane_id'] for f in frames),
            np.ones(len(frames),dtype=bool),tuple('valid' for _ in frames),speed)
        result[actor] = Prefix(track,len(frames))
    return result['ego'],{a:result[a] for a in actor_ids}


def _following_comparison(projected, native, states, road, n):
    truth = _following_values(*native,states,road,n)
    reference_truth = _following_values(projected[MODES[0]][0],native[1],states,road,n)
    output = {}
    for mode in MODES:
        prediction = _following_values(*projected[mode],states,road,n)
        out = {'counts':{},'errors':{}}
        for role in ('front','rear'):
            p = prediction[role]
            for label,t in (('combined',truth[role]),('neighbor_only',reference_truth[role])):
                key = role+'_'+label
                out['counts'][key+'_applicable_'+confusion(bool(t[3]),bool(p[3]))] = 1
                available = bool(t[4] and p[4])
                out['errors'][key+'_min_gap_m'] = [(p[0]-t[0])*100.] if available else []
                out['counts'][key+'_joint_minimum_available'] = int(available)
                out['counts'][key+'_prediction_min_at_root'] = int(p[4] and p[2] == 0)
        output[mode] = out
    return output


def _cross_comparison(projected, native, states, road, eligible, n):
    if n <= 1: return {m:{'counts':{},'errors':{}} for m in MODES}
    ew_true = prefix_window(native[0],road.zone_interval(states['ego']),n)
    ew_ref = prefix_window(projected[MODES[0]][0],road.zone_interval(states['ego']),n)
    output = {m:{'counts':Counter(),'errors':{}} for m in MODES}
    for actor in eligible:
        aw_true = prefix_window(native[1][actor],road.zone_interval(states[actor]),n)
        for mode in MODES:
            out = output[mode]; ew = prefix_window(projected[mode][0],road.zone_interval(states['ego']),n)
            aw = prefix_window(projected[mode][1][actor],road.zone_interval(states[actor]),n)
            pair = prefix_overlap(ew,aw)
            out['counts']['eligible_pair_comparisons'] += 1
            for event,truth,prediction in (('actor_entry',aw_true['entry_s'],aw['entry_s']),
                                            ('actor_exit',aw_true['exit_s'],aw['exit_s'])):
                if event == 'actor_entry' and (aw_true['left_censored'] or aw_true['passed_before_root']):
                    out['counts']['actor_entry_observed_root_or_passed'] += 1
                    continue
                out['counts'][event+'_'+confusion(truth is not None,prediction is not None)] += 1
                if truth is not None and prediction is not None:
                    out['errors'].setdefault(event+'_s',[]).append(prediction-truth)
            for label,truth_ego in (('combined',ew_true),('neighbor_only',ew_ref)):
                truth_pair = prefix_overlap(truth_ego,aw_true)
                out['errors'].setdefault(label+'_overlap_s',[]).append(
                    pair['supported_overlap_s']-truth_pair['supported_overlap_s'])
                out['counts'][label+'_overlap_'+confusion(truth_pair['supported_overlap_s'] > 0,
                                                         pair['supported_overlap_s'] > 0)] += 1
                if pair['first_overlap_s'] is not None and truth_pair['first_overlap_s'] is not None:
                    out['errors'].setdefault(label+'_first_overlap_s',[]).append(
                        pair['first_overlap_s']-truth_pair['first_overlap_s'])
            out['counts']['actor_occupies_entire_support'] += int(aw['left_censored'] and aw['right_censored'])
            out['counts']['first_overlap_equals_ego_entry'] += int(pair['first_overlap_s'] is not None
                and ew['entry_s'] is not None and abs(pair['first_overlap_s']-ew['entry_s']) <= 1e-8)
            out['counts']['overlap_saturated_at_T'] += int(abs(pair['supported_overlap_s']-(n-1)*.2) <= 1e-8)
    return output


def compare_native(history, branch, candidate, projected, road):
    """Called after features exist. No native masks/lanes return to an adapter."""
    root = history[-1]; states = root['vehicles']
    local = {m:projected[m][candidate] for m in MODES}; actors = sorted(local[MODES[0]][1])
    if any(sorted(local[m][1]) != actors for m in MODES): raise ValueError('Matched actor pool changed')
    native = _native_pool(root,branch,actors,road)
    native_n = pool_length(*native)
    common_n = min([native_n]+[pool_length(*local[m]) for m in MODES])
    result = {'candidate_id':candidate,'native_follow_points':native_n,'common_follow_points':common_n,
        'following':_following_comparison(local,native,states,road,common_n), 'fixed_horizons':{}}
    for seconds in SETTINGS['fixed_horizons_s']:
        n = 1+round(seconds/.2); observed = native_n >= n; supported = common_n >= n
        result['fixed_horizons'][str(seconds)] = {'total_candidates':1,'native_observed':int(observed),
            'joint_supported':int(supported), 'methods':_following_comparison(local,native,states,road,n) if supported else {}}
    # Report ego-reference error separately, excluding the observed root.
    ne = min(native[0].length,local[MODES[0]][0].length)
    result['ego_progress_errors_m'] = (local[MODES[0]][0].track.progress[1:ne]-native[0].track.progress[1:ne]).tolist()
    eligible = [a for a in actors if {road.geometry.group(states['ego']),road.geometry.group(states[a])}
                == {'ramp','main_approach'}]
    nt_cross = pool_length(native[0],{a:native[1][a] for a in eligible})
    np_cross = [pool_length(local[m][0],{a:local[m][1][a] for a in eligible}) for m in MODES]
    nc = min([nt_cross]+np_cross)
    result.update(eligible_cross_pairs=len(eligible),native_cross_points=nt_cross,common_cross_points=nc,
        cross=_cross_comparison(local,native,states,road,eligible,nc))
    return result


def _diagnostic_values(ego, actors, states, road, follow_n, cross_n):
    follow = _following_values(ego,actors,states,road,follow_n)
    group = road.geometry.group(states['ego'])
    eligible = {a:p for a,p in actors.items() if {group,road.geometry.group(states[a])} == {'ramp','main_approach'}}
    cross = conflict(cut(ego,cross_n),{a:cut(p,cross_n) for a,p in eligible.items()},states,road)[0]
    result = {role+'_gap_m':float(v[0]*100.) if v[4] else None for role,v in follow.items()}
    result.update(ego_entry_s=cross[0]*5. if cross[4] and cross[6] else None,
        actor_entry_s=cross[1]*5. if cross[4] and cross[7] else None,
        overlap_s=cross[2]*5. if cross[12] else None,
        first_overlap_s=cross[3]*5. if cross[12] and cross[2] > 0 else None)
    return result


def mechanism(history, projected, road):
    """Support-matched substitution, not causal or policy-performance evidence."""
    states = history[-1]['vehicles']; group = road.geometry.group(states['ego'])
    all_pairs = [p for m in MODES for p in projected[m]]
    nf = min(pool_length(*p) for p in all_pairs)
    nc = min(pool_length(e,{a:p for a,p in aa.items() if {group,road.geometry.group(states[a])}
                           == {'ramp','main_approach'}}) for e,aa in all_pairs)
    result = {}
    for mode in MODES:
        ze,za = projected[mode][2]
        reference = _diagnostic_values(ze,za,states,road,nf,nc)
        variants = {'total':[],'ego_plan':[],'neighbor_response':[],'conditional_increment':[]}
        for e,aa in projected[mode]:
            total = _diagnostic_values(e,aa,states,road,nf,nc)
            held = _diagnostic_values(e,za,states,road,nf,nc)
            neighbor = _diagnostic_values(ze,aa,states,road,nf,nc)
            for name,first,second in (('total',total,reference),('ego_plan',held,reference),
                                       ('neighbor_response',neighbor,reference),('conditional_increment',total,held)):
                variants[name].append({k:{'joint_available':first[k] is not None and second[k] is not None,
                    'availability_changed':(first[k] is None) != (second[k] is None),
                    'delta':None if first[k] is None or second[k] is None else first[k]-second[k]}
                    for k in reference})
        result[mode] = {'common_follow_horizon_s':(nf-1)*.2,'common_cross_horizon_s':(nc-1)*.2,
            'replacement_may_be_ood':True,'quality_support_fixed_across_all_models_candidates':True,'variants':variants}
    return result


def _merge_records(records):
    counts = Counter(); values = {}
    for r in records:
        counts.update(r['counts'])
        for key,v in r['errors'].items(): values.setdefault(key,[]).extend(v)
    return {'counts':dict(counts),'errors':{k:error_summary(v) for k,v in sorted(values.items())}}


def aggregate(rows):
    """Separate reused design episodes; episode equal-weight MAEs are explicit."""
    output = {}
    for split in SETTINGS['splits']:
        for reused in (False,True):
            group = [r for r in rows if r['split'] == split and r['design_episode_reuse'] == reused]
            episodes = sorted({r['episode_id'] for r in group}); methods = {}
            for mode in MODES:
                coverage = Counter(); reasons = Counter()
                for r in group:
                    coverage.update(r['coverage'][mode]['counts']); reasons.update(r['prefix_failures'][mode])
                method = {'coverage':dict(coverage),'first_invalid_reasons':dict(reasons),
                    'following_prefix_horizon_s':describe([t for r in group for t in r['coverage'][mode]['following_prefix_horizons_s']]),
                    'cross_prefix_horizon_s':describe([t for r in group for t in r['coverage'][mode]['cross_prefix_horizons_s']]),
                    'following':_merge_records([b['following'][mode] for r in group for b in r['branches']]),
                    'cross':_merge_records([b['cross'][mode] for r in group for b in r['branches']]),
                    'fixed_horizons':{},'episode_errors':{},'mechanism':{}}
                for time in SETTINGS['fixed_horizons_s']:
                    entries = [b['fixed_horizons'][str(time)] for r in group for b in r['branches']]
                    method['fixed_horizons'][str(time)] = {**{k:sum(e[k] for e in entries) for k in
                        ('total_candidates','native_observed','joint_supported')},
                        **_merge_records([e['methods'][mode] for e in entries if e['joint_supported']])}
                for kind in ('following','cross'):
                    records = {ep:_merge_records([b[kind][mode] for r in group if r['episode_id'] == ep
                                                  for b in r['branches']]) for ep in episodes}
                    keys = sorted({k for rec in records.values() for k in rec['errors']})
                    method['episode_errors'][kind] = {key:{'equal_weight_episode_mae':error_summary([
                        rec['errors'][key]['mae'] for rec in records.values()
                        if key in rec['errors'] and rec['errors'][key]['mae'] is not None]),
                        'episodes_total':len(episodes)} for key in keys}
                for variant in ('total','ego_plan','neighbor_response','conditional_increment'):
                    cells = [x for r in group for x in r['mechanism'][mode]['variants'][variant]]
                    method['mechanism'][variant] = {key:{'comparisons':len(cells),
                        'joint_available':sum(x[key]['joint_available'] for x in cells),
                        'availability_changes':sum(x[key]['availability_changed'] for x in cells),
                        'signed_deltas':error_summary([x[key]['delta'] for x in cells if x[key]['delta'] is not None])}
                        for key in ('front_gap_m','rear_gap_m','ego_entry_s','actor_entry_s','overlap_s','first_overlap_s')}
                methods[mode] = method
            output[f'{split}/'+('design_episode_reuse' if reused else 'other_existing_episodes')] = {
                'roots':len(group),'unique_episodes':len(episodes),'methods':methods,
                'omitted_actor_count':describe([r['omitted_actor_count'] for r in group]),
                'native_follow_horizon_s':describe([(b['native_follow_points']-1)*.2 for r in group for b in r['branches']]),
                'joint_follow_horizon_s':describe([(b['common_follow_points']-1)*.2 for r in group for b in r['branches']]),
                'native_cross_horizon_s':describe([(b['native_cross_points']-1)*.2 for r in group for b in r['branches'] if b['eligible_cross_pairs']]),
                'joint_cross_horizon_s':describe([(b['common_cross_points']-1)*.2 for r in group for b in r['branches'] if b['eligible_cross_pairs']]),
                'ego_progress_errors_m':error_summary([v for r in group for b in r['branches'] for v in b['ego_progress_errors_m']]),
                'independent_units':'episodes_not_roots_candidates_actors_or_times',
                'independent_confirmation':False}
    return output
