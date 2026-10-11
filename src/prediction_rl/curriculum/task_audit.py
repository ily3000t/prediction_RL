"""Exploratory train-task comparability diagnostics, not a task classifier."""
from collections import Counter
from itertools import combinations
import math

from prediction_rl.curriculum.interaction import distribution, merge_pairs, pair_response, validated_timeline


def neighbor_timelines(pack, branches, geometry, metric):
    if pack['split'] != 'train':
        raise ValueError('Only sealed train branches may enter this audit')
    labels = {row['candidate_id']: row['labels'] for row in pack['candidates']}
    if len(pack['candidates']) != 5 or set(labels) != set(range(5)) or set(branches) != set(range(5)):
        raise ValueError('Require the unchanged five-candidate family')
    for cid in range(5):
        if labels[cid]['requested_plan'] != [metric['probe_jerks'][cid]] + [0.]*(metric['horizon_steps']-1):
            raise ValueError('Candidate intervention changed')
    return {cid: validated_timeline(branches[cid], pack['root_traffic'], labels[cid], geometry, metric)
            for cid in range(5)}


def longest_dense_prefix(timelines, actors, horizon):
    """All candidates, all selected stable IDs, every tick; never fill a gap."""
    actors = sorted(set(actors))
    if set(timelines) != set(range(5)) or 'ego' in actors:
        raise ValueError('Five candidates and neighbor-only stable IDs required')
    if type(horizon) is not int or horizon <= 0:
        raise ValueError('Invalid horizon')
    for line in timelines.values():
        if any(type(t) is not int or not 0 <= t < horizon for t in line):
            raise ValueError('Timeline outside the original horizon')
        for frame in line.values():
            for actor in actors:
                if actor in frame and (len(frame[actor]) != 2 or not all(math.isfinite(v) for v in frame[actor])):
                    raise ValueError('Nonfinite or malformed neighbor response')
    if not actors:
        return 0
    for t in range(horizon):
        if any(t not in line or any(actor not in line[t] for actor in actors) for line in timelines.values()):
            return t
    return horizon


def dense_response(timelines, actors, steps, metric, scales):
    if (type(steps) is not int or not 1 <= steps <= metric['horizon_steps']
            or any(not math.isfinite(scales[k]) or scales[k] <= 0 for k in ('dv', 'ds'))):
        raise ValueError('Invalid window or frozen normalization scales')
    actors = sorted(set(actors))
    prefix = longest_dense_prefix(timelines, actors, metric['horizon_steps'])
    missing = sum(t not in line or actor not in line[t]
                  for line in timelines.values() for t in range(steps) for actor in actors)
    available = bool(actors) and prefix >= steps
    result = {'status': 'available' if available else 'unavailable', 'actor_ids': actors,
              'horizon_steps': steps, 'duration_s': steps*metric['tick_s'],
              'longest_dense_prefix_steps': prefix, 'missing_candidate_actor_time_cells': missing,
              'unavailable_reason': None if available else 'empty_actor_set' if not actors else 'incomplete_common_prefix',
              'I_dense_with_frozen_IC01_scales': None, 'interaction': None}
    if available:
        cut = {cid: {t: line[t] for t in range(steps)} for cid, line in timelines.items()}
        stats = merge_pairs([pair_response(cut[a], cut[b], actors, steps, steps)
                             for a, b in combinations(range(5), 2)])
        if not stats['full_common_horizon'] or stats['supported_pairs'] != 10:
            raise AssertionError('Dense support cannot use partial-cell averaging')
        result.update(interaction=stats, I_dense_with_frozen_IC01_scales=(
            stats['mean_abs_dv_mps']/scales['dv'] + stats['mean_abs_ds_m']/scales['ds']))
    return result


def actor_sets(context):
    all_ids = context['actor_ids']
    targets = sorted(actor for actor in all_ids if any(role in ('target_front', 'target_rear')
                    for role in context['roles_by_actor'][actor]))
    return {'all_critical': sorted(all_ids), 'target_front_rear': targets}


def window_audit(timelines, context, steps, metric, scales):
    sets = actor_sets(context)
    return {'steps': steps, 'duration_s': steps*metric['tick_s'],
            'sets': {name: dense_response(timelines, ids, steps, metric, scales) for name, ids in sets.items()},
            'per_actor': {actor: {'roles': context['roles_by_actor'][actor],
                'response': dense_response(timelines, [actor], steps, metric, scales)}
                for actor in sets['all_critical']}}


def relation_audit(history, evidence, geometry):
    """Separate route-projected opportunities from same-path closing proxies."""
    root = history['history'][-1]; geometry.validate_frame(root)
    ego = root['vehicles']['ego']; context = evidence['root_context']
    relations = {}
    for role, old in context['target_relations'].items():
        actor = old['actor_id']
        if actor is None:
            relations[role] = {'actor_id': None, 'path_relation': 'absent'}
            continue
        other = root['vehicles'][actor]; same = geometry.same_path(ego, other)
        pair = geometry.pair(ego, other)
        relations[role] = {'actor_id': actor, 'path_relation': 'same_path' if same else 'cross_path',
            'ego_lane_id': ego['lane_id'], 'other_lane_id': other['lane_id'],
            'ego_speed_mps': ego['speed'], 'other_speed_mps': other['speed'],
            'route_projected_gap_m': old['gap_route_proxy_m'],
            'old_route_ttc_proxy_s': old['ttc_s'], 'closing_speed_mps': old['closing_mps'],
            'same_path_gap_m': pair['same_path_gap_m'], 'same_path_cv_ttc_s': pair['cv_ttc_s'],
            'front_bumper_euclidean_distance_m': pair['front_bumper_distance_m'],
            'ego_merge_end_eta_s': geometry.eta(ego), 'other_merge_end_eta_s': geometry.eta(other),
            'cross_path_eta_difference_s': pair['cross_path_eta_difference_s'],
            'route_proxy_is_immediate_collision_measure': False,
            'euclidean_front_bumper_distance_is_surface_gap': False}
    return {'original_difficulty_unchanged': context['difficulty'], 'target_relations': relations,
            'critical_actor_path_relations': {actor: 'same_path' if geometry.same_path(ego, root['vehicles'][actor])
                else 'cross_path' for actor in context['actor_ids']},
            'no_new_D_score_or_safety_threshold': True}


def ranks(values):
    result = {}; ordered = sorted(values, key=lambda k: (values[k], k))
    index = 0
    while index < len(ordered):
        end = index+1
        while end < len(ordered) and values[ordered[end]] == values[ordered[index]]:
            end += 1
        for key in ordered[index:end]:
            result[key] = (index+1+end)/2
        index = end
    return result


def rank_comparison(left, right):
    """Descriptive paired ranks only; no independent-sample significance claim."""
    keys = sorted(k for k in left.keys() & right.keys() if left[k] is not None and right[k] is not None)
    if not keys:
        return {'matched_tasks': 0, 'spearman': None, 'pair_order_reversals': 0,
                'comparable_non_tied_pairs': 0, 'maximum_absolute_rank_change': None, 'rank_changes': {}}
    a, b = ranks({k: left[k] for k in keys}), ranks({k: right[k] for k in keys})
    mean = (len(keys)+1)/2
    denominator = math.sqrt(math.fsum((a[k]-mean)**2 for k in keys)*math.fsum((b[k]-mean)**2 for k in keys))
    pairs = [(left[x]-left[y], right[x]-right[y]) for x, y in combinations(keys, 2)]
    comparable = [(x, y) for x, y in pairs if x != 0 and y != 0]
    changes = {k: b[k]-a[k] for k in keys}
    return {'matched_tasks': len(keys), 'spearman': math.fsum((a[k]-mean)*(b[k]-mean) for k in keys)/denominator
            if denominator else None, 'pair_order_reversals': sum(x*y < 0 for x, y in comparable),
            'comparable_non_tied_pairs': len(comparable),
            'maximum_absolute_rank_change': max(abs(v) for v in changes.values()), 'rank_changes': changes}


def summarize(rows, grid, common_steps):
    result = {'roots': len(rows), 'tasks': len({r['task_id'] for r in rows}),
              'global_dense_prefix_ceiling_steps': common_steps,
              'ceiling_is_observed_support_not_a_frozen_training_window': True, 'by_target': {}}
    for target in sorted({r['target'] for r in rows}):
        subset = [r for r in rows if r['target'] == target]
        old = {r['task_id']: r['original_evidence']['I_with_frozen_IC01_scales'] for r in subset}
        windows = {}
        for steps in grid:
            windows[str(steps)] = {name: distribution(r['windows'][str(steps)]['sets'][name]['I_dense_with_frozen_IC01_scales']
                for r in subset) for name in ('all_critical', 'target_front_rear')}
        all_i = {r['task_id']: r['common_prefix']['sets']['all_critical']['I_dense_with_frozen_IC01_scales'] for r in subset}
        target_i = {r['task_id']: r['common_prefix']['sets']['target_front_rear']['I_dense_with_frozen_IC01_scales'] for r in subset}
        extras = []
        for r in subset:
            sets = actor_sets(r['original_evidence']['root_context'])
            extra_ids = sorted(set(sets['all_critical'])-set(sets['target_front_rear']))
            if extra_ids:
                extras.append({'task_id': r['task_id'], 'extra_actor_ids': extra_ids,
                    'extra_actor_I': {a: r['common_prefix']['per_actor'][a]['response']['I_dense_with_frozen_IC01_scales'] for a in extra_ids},
                    'all_critical_I': all_i[r['task_id']], 'target_front_rear_I': target_i[r['task_id']]})
        relations = [d for r in subset for d in r['road_relations']['target_relations'].values() if d['actor_id'] is not None]
        projected_nonpositive = [d for d in relations if d['path_relation']=='cross_path' and d['route_projected_gap_m'] <= 0]
        groups = {}
        for count in sorted({len(r['original_evidence']['root_context']['actor_ids']) for r in subset}):
            group = [r for r in subset if len(r['original_evidence']['root_context']['actor_ids']) == count]
            groups[str(count)] = {'roots': len(group), 'original_full_horizon': sum(r['original_evidence']['interaction']['full_common_horizon'] for r in group),
                'original_I': distribution(r['original_evidence']['I_with_frozen_IC01_scales'] for r in group),
                'common_all_critical_I': distribution(all_i[r['task_id']] for r in group),
                'common_target_front_rear_I': distribution(target_i[r['task_id']] for r in group)}
        result['by_target'][target] = {'roots': len(subset), 'original_I': distribution(old.values()),
            'original_partial_horizon_roots': sum(not r['original_evidence']['interaction']['full_common_horizon'] for r in subset),
            'fixed_window_grid': windows, 'common_all_critical_I': distribution(all_i.values()),
            'common_target_front_rear_I': distribution(target_i.values()),
            'original_vs_common_ranks': rank_comparison(old, all_i),
            'common_all_vs_target_ranks': rank_comparison(all_i, target_i),
            'extra_critical_actor_effect': extras, 'critical_actor_count_groups': groups,
            'path_relation_counts': dict(Counter(d['path_relation'] for d in relations)),
            'cross_path_nonpositive_projected_gap_count': len(projected_nonpositive),
            'front_bumper_distance_for_cross_nonpositive_gap_m': distribution(d['front_bumper_euclidean_distance_m'] for d in projected_nonpositive),
            'difficulty_distributions': {k: distribution(r['road_relations']['original_difficulty_unchanged'][k] for r in subset)
                for k in subset[0]['road_relations']['original_difficulty_unchanged']}}
    return result
