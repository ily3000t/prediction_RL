"""Train-side neighbor response analysis, with no policy or reward inputs."""
from collections import Counter
from itertools import combinations
import math

from prediction_rl.data.actor_adapter import select_actors
from prediction_rl.data.dataset_contract import build_branch_labels


def quantile(values, q):
    values = sorted(float(v) for v in values if v is not None)
    if not values:
        return None
    if not 0 <= q <= 1 or not all(math.isfinite(v) for v in values):
        raise ValueError('Invalid finite quantile inputs')
    index = q * (len(values) - 1)
    low, high = math.floor(index), math.ceil(index)
    return values[low] + (values[high] - values[low]) * (index - low)


def distribution(values):
    values = list(values)
    finite = [v for v in values if v is not None]
    return {'count': len(finite), 'unavailable': len(values)-len(finite),
            'min': min(finite, default=None), 'max': max(finite, default=None),
            'mean': math.fsum(finite)/len(finite) if finite else None,
            'quantiles': {str(q): quantile(finite, q) for q in (0, 1/3, .5, 2/3, .95, 1)}}


def root_context(history, geometry, config):
    """Roles fixed using only root state, before reading any candidate future."""
    root = history['history'][-1]
    selected, roles, metrics = select_actors(root, geometry, 12)
    if (roles != history['inputs']['mandatory_roles']
            or selected != [x for x in history['inputs']['actor_ids'] if x is not None]):
        raise ValueError('Recorded root-only actor selection changed')
    ego = root['vehicles']['ego']
    critical = {}
    for role in config['critical_roles']:
        actor = roles.get(role)
        if actor is None:
            continue
        metric = metrics[actor]
        if metric['front_bumper_distance_m'] > config['sensor_radius_m']:
            continue
        if role == 'nearest_cross_path_arrival' and (
                metric['cross_path_eta_difference_s'] is None
                or metric['cross_path_eta_difference_s'] > config['conflict_horizon_s']):
            continue
        if role == 'lowest_same_path_ttc' and (
                metric['cv_ttc_s'] is None or metric['cv_ttc_s'] > config['conflict_horizon_s']):
            continue
        critical.setdefault(actor, []).append(role)
    gaps, closing, ttcs, details = [], [], [], {}
    for role in ('target_front', 'target_rear'):
        actor = roles.get(role)
        if actor is None or actor not in critical:
            details[role] = {'actor_id': None, 'gap_route_proxy_m': None,
                             'closing_mps': None, 'ttc_s': None, 'ttc_status': 'absent'}
            continue
        other = root['vehicles'][actor]
        delta = metrics[actor]['progress_delta_m']
        gap = delta-other['length_m'] if role == 'target_front' else -delta-ego['length_m']
        speed = max(ego['speed']-other['speed'] if role == 'target_front'
                    else other['speed']-ego['speed'], 0.)
        ttc = 0. if gap <= 0 else gap/speed if speed > 1e-6 else None
        gaps.append(gap); closing.append(speed)
        if ttc is not None:
            ttcs.append(ttc)
        details[role] = {'actor_id': actor, 'gap_route_proxy_m': gap, 'closing_mps': speed,
                         'ttc_s': ttc, 'ttc_status': 'finite' if ttc is not None else 'not_closing'}
    conflicts = [m['cross_path_eta_difference_s'] for key, m in metrics.items()
                 if m['cross_path_eta_difference_s'] is not None
                 and m['cross_path_eta_difference_s'] <= config['conflict_horizon_s']
                 and m['front_bumper_distance_m'] <= config['sensor_radius_m']]
    slot = {actor: i for i, actor in enumerate(history['inputs']['actor_ids']) if actor is not None}
    return {'actor_ids': sorted(critical), 'roles_by_actor': critical, 'target_relations': details,
            'difficulty': {'minimum_gap_route_proxy_m': min(gaps, default=None),
                'maximum_closing_mps': max(closing, default=None),
                'minimum_ttc_route_proxy_s': min(ttcs, default=None),
                'remaining_merge_distance_m': max(-geometry.progress(ego), 0.),
                'critical_actor_count': len(critical), 'cv_conflict_proxy_count': len(conflicts),
                'minimum_cross_path_eta_difference_s': min(conflicts, default=None)},
            'coverage': {'key_actor_missing_slots': [a for a in critical if a not in slot],
                'key_history_steps': {a: sum(history['inputs']['actor_history_mask'][slot[a]])
                                      if a in slot else 0 for a in critical},
                'history_frames': len(history['history']),
                'root_neighbor_count': len(root['vehicles'])-1,
                'omitted_neighbor_count': len(history['inputs']['root_omitted_ids'])}}


def validated_timeline(branch, root, labels, geometry, config):
    """Reject moving-origin labels, mismatched sidecars and post-terminal tails."""
    trace = branch['trace']; meta = branch['future_metadata']
    if (meta['contract'] != 'lane_position_length_width_v1'
            or len(trace) != len(meta['frames'])
            or meta['trajectory_usable'] != [not row['done'] for row in trace]):
        raise ValueError('Invalid geometry/terminal support metadata')
    plan = labels['requested_plan']
    if len(plan) != config['horizon_steps']:
        raise ValueError('Unexpected prediction horizon')
    if labels != build_branch_labels(root, trace, plan, config['tick_s']):
        raise ValueError('Branch labels differ from common fixed-root coordinates or masks')
    result = {}
    for index, (step, frame) in enumerate(zip(trace, meta['frames'])):
        geometry.validate_frame(frame, require_ego=not step['done'])
        base = {'simulation_time_s': frame['simulation_time_s'], 'vehicles': {
            actor: {k: value[k] for k in ('position', 'speed', 'acceleration', 'lane_id')}
            for actor, value in frame['vehicles'].items()}}
        if base != step['info']['execution_audit']['after']:
            raise ValueError('Sidecar not aligned with audited absolute traffic')
        if not step['done']:
            if not math.isclose(frame['simulation_time_s'],
                    root['simulation_time_s']+(index+1)*config['tick_s'], abs_tol=1e-8):
                raise ValueError('Nonterminal time grid changed')
            result[index] = {actor: (geometry.progress(value), value['speed'])
                             for actor, value in frame['vehicles'].items() if actor != 'ego'}
    return result


def pair_response(left, right, actors, horizon, minimum_common_steps):
    if any(type(t) is not int or not 0 <= t < horizon for t in (*left, *right)):
        raise ValueError('Response timeline escaped the fixed horizon')
    actors = sorted(set(actors)-{'ego'})
    dv, ds, times = [], [], set()
    actor_counts = Counter()
    for t in sorted(left.keys() & right.keys()):
        for actor in actors:
            if actor not in left[t] or actor not in right[t]:
                continue
            x1, v1 = left[t][actor]; x2, v2 = right[t][actor]
            if not all(math.isfinite(v) for v in (x1, v1, x2, v2)):
                raise ValueError('Nonfinite neighbor response')
            dv.append(abs(v1-v2)); ds.append(abs(x1-x2)); times.add(t); actor_counts[actor] += 1
    n = len(dv)
    possible = len(actors)*horizon
    return {'common_cells': n, 'possible_cells': possible, 'common_steps': len(times),
            'actor_common_steps': dict(actor_counts),
            'supported': len(times) >= minimum_common_steps,
            'mean_abs_dv_mps': math.fsum(dv)/n if n else None,
            'mean_abs_ds_m': math.fsum(ds)/n if n else None,
            'max_abs_dv_mps': max(dv, default=None), 'max_abs_ds_m': max(ds, default=None)}


def merge_pairs(pairs):
    n = sum(p['common_cells'] for p in pairs)
    possible = sum(p['possible_cells'] for p in pairs)
    return {'common_cells': n, 'possible_cells': possible,
            'available_pairs': sum(p['common_cells'] > 0 for p in pairs),
            'supported_pairs': sum(p['supported'] for p in pairs), 'pair_count': len(pairs),
            'common_cell_fraction': n/possible if possible else None,
            'full_common_horizon': bool(possible and n == possible),
            'mean_abs_dv_mps': math.fsum(p['mean_abs_dv_mps']*p['common_cells']
                                      for p in pairs if p['common_cells'])/n if n else None,
            'mean_abs_ds_m': math.fsum(p['mean_abs_ds_m']*p['common_cells']
                                     for p in pairs if p['common_cells'])/n if n else None,
            'max_abs_dv_mps': max((p['max_abs_dv_mps'] for p in pairs if p['common_cells']), default=None),
            'max_abs_ds_m': max((p['max_abs_ds_m'] for p in pairs if p['common_cells']), default=None)}


def analyze_root(history, pack, branches, repeated, geometry, config, binding):
    if (history['split'] != 'train' or pack['split'] != 'train'
            or history['root_id'] != pack['root_id'] or history['episode_id'] != pack['episode_id']):
        raise ValueError('IC01 only accepts aligned train root families')
    root = pack['root_traffic']
    last = history['history'][-1]
    if root != {'simulation_time_s': last['simulation_time_s'], 'vehicles': {
            actor: {k: value[k] for k in ('position', 'speed', 'acceleration', 'lane_id')}
            for actor, value in last['vehicles'].items()}}:
        raise ValueError('History and branch root differ')
    context = root_context(history, geometry, config)
    labels = {c['candidate_id']: c['labels'] for c in pack['candidates']}
    expected = set(range(5))
    if (len(pack['candidates']) != 5 or set(labels) != expected
            or set(branches) != expected or set(repeated) != expected):
        raise ValueError('Require five distinct candidates and five verified repeats')
    timelines, repeats = {}, {}
    for cid in sorted(expected):
        if labels[cid]['requested_plan'] != [config['probe_jerks'][cid]]+[0.]*(config['horizon_steps']-1):
            raise ValueError('Candidate plan changed')
        timelines[cid] = validated_timeline(branches[cid], root, labels[cid], geometry, config)
        # Identical bytes are verified by the reader; avoid parsing the same large
        # JSON twice. Non-identical diagnostic fixtures still compare real grids.
        repeats[cid] = timelines[cid] if repeated[cid] is branches[cid] else validated_timeline(
            repeated[cid], root, labels[cid], geometry, config)
    args = (context['actor_ids'], config['horizon_steps'], config['minimum_common_steps'])
    pairs = [{'candidates': [a, b], **pair_response(timelines[a], timelines[b], *args)}
             for a, b in combinations(sorted(expected), 2)]
    noise = [pair_response(timelines[cid], repeats[cid], *args) for cid in sorted(expected)]
    return {**binding, 'root_id': history['root_id'], 'episode_id': history['episode_id'],
            'root_lane': last['vehicles']['ego']['lane_id'], 'root_time_s': root['simulation_time_s'],
            'root_context': context, 'interaction': merge_pairs(pairs), 'repeat_noise': merge_pairs(noise),
            'candidate_pairs': pairs,
            'censoring': {'terminal_candidates': sum(branches[c]['trace'][-1]['done'] for c in expected),
                'observed_nonterminal_steps': {str(c): len(timelines[c]) for c in sorted(expected)}},
            'task_mapping': {'executable_as_original_reset': False,
                'reference_prefix_steps': len(pack['prefix_actions']),
                'prefix_elapsed_s': len(pack['prefix_actions'])*config['tick_s'],
                'reason': 'geometric_root_along_reference_policy_not_episode_reset'}}


def stratify(rows, config):
    """Provisional train-only boundaries. Never a DDPG release gate."""
    q = config['upper_quantile']; low = config['lower_quantile']
    scales = {}; sources = {}
    for channel, unit in (('dv', 'mps'), ('ds', 'm')):
        key = f'mean_abs_{channel}_{unit}'
        positive = [r['interaction'][key] for r in rows if r['interaction'][key] is not None
                    and r['interaction'][key] > config['response_epsilon']]
        scales[channel] = quantile(positive, config['scale_quantile']) if positive else 1.
        sources[channel] = 'positive_train_root_quantile' if positive else 'unit_fallback_no_positive_support'
    noise_floor = max((r['repeat_noise']['max_abs_dv_mps']/scales['dv']
                       +r['repeat_noise']['max_abs_ds_m']/scales['ds'] for r in rows
                       if r['repeat_noise']['common_cells']), default=0.)
    for row in rows:
        stats = row['interaction']
        row['I'] = (stats['mean_abs_dv_mps']/scales['dv']+stats['mean_abs_ds_m']/scales['ds']
                    if stats['common_cells'] else None)
        row['support_qualified'] = (stats['supported_pairs']/stats['pair_count']
                                   >= config['minimum_supported_pair_fraction'])
    thresholds = {'I_low': quantile([r['I'] for r in rows], low),
                  'I_high': quantile([r['I'] for r in rows], q), 'repeat_noise_floor': noise_floor}
    for name, percentile in (('minimum_gap_route_proxy_m', q), ('maximum_closing_mps', low),
                             ('remaining_merge_distance_m', q), ('cv_conflict_proxy_count', low),
                             ('minimum_ttc_route_proxy_s', q)):
        thresholds[name] = quantile([r['root_context']['difficulty'][name] for r in rows], percentile)
    for row in rows:
        d = row['root_context']['difficulty']; value = row['I']; stats = row['interaction']
        weak = all(d[k] is not None and thresholds[k] is not None and comparison(d[k], thresholds[k])
                   for k, comparison in (
                       ('minimum_gap_route_proxy_m', lambda a, b: a >= b),
                       ('maximum_closing_mps', lambda a, b: a <= b),
                       ('remaining_merge_distance_m', lambda a, b: a >= b),
                       ('cv_conflict_proxy_count', lambda a, b: a <= b)))
        if d['minimum_ttc_route_proxy_s'] is not None and thresholds['minimum_ttc_route_proxy_s'] is not None:
            weak = weak and d['minimum_ttc_route_proxy_s'] >= thresholds['minimum_ttc_route_proxy_s']
        high = (value is not None and thresholds['I_high'] is not None and value >= thresholds['I_high']
                and value > noise_floor+config['response_epsilon'])
        if (high and row['support_qualified']
                and 0 < d['critical_actor_count'] <= config['C1_maximum_critical_actors']):
            row['level'] = 'C1'; row['level_reason'] = 'clear_supported_key_neighbor_response'
        elif (value is not None and thresholds['I_low'] is not None and value <= thresholds['I_low']
              and weak and stats['full_common_horizon']):
            row['level'] = 'C0'; row['level_reason'] = 'weak_constraints_low_response_full_support'
        else:
            row['level'] = 'C2'
            row['level_reason'] = ('uncomputable_response' if value is None else
                'insufficient_common_support' if not row['support_qualified'] else 'remaining_complete_task')
    levels = {level: {'roots': sum(r['level'] == level for r in rows),
                     'unique_episodes': len({r['episode_id'] for r in rows if r['level'] == level})}
              for level in ('C0', 'C1', 'C2')}
    return {'scales': scales, 'scale_sources': sources, 'thresholds': thresholds,
            'levels': levels, 'root_quantiles_are_provisional_not_task_distribution': True,
            'nondegenerate_root_pool': all(v['unique_episodes'] >= config['minimum_layer_episodes']
                                          for v in levels.values()),
            'response_available': any(r['I'] is not None and r['I'] > noise_floor+config['response_epsilon']
                                      for r in rows), 'training_ready': False}


def summarize(rows, episodes, unavailable, classification):
    names = tuple(rows[0]['root_context']['difficulty']) if rows else ()
    return {'episodes': len(episodes), 'reached_roots': len(rows),
            'unavailable_roots': len(unavailable), 'unavailable_reasons': dict(Counter(r['reason'] for r in unavailable)),
            'I': distribution(r['I'] for r in rows),
            'I_zero_roots': sum(r['I'] == 0 for r in rows),
            'I_positive_roots': sum(r['I'] is not None and r['I'] > 0 for r in rows),
            'support_qualified_roots': sum(r['support_qualified'] for r in rows),
            'full_common_horizon_roots': sum(r['interaction']['full_common_horizon'] for r in rows),
            'difficulty': {k: distribution(r['root_context']['difficulty'][k] for r in rows) for k in names},
            'by_location': {target: {'roots': sum(r['target'] == target for r in rows),
                'levels': dict(Counter(r['level'] for r in rows if r['target'] == target)),
                'I': distribution(r['I'] for r in rows if r['target'] == target)}
                for target in sorted({r['target'] for r in rows})},
            'key_actor_missing_slots': sum(len(r['root_context']['coverage']['key_actor_missing_slots']) for r in rows),
            'roots_with_incomplete_key_history': sum(any(n < 11 for n in
                r['root_context']['coverage']['key_history_steps'].values()) for r in rows),
            'classification': classification,
            'executable_original_reset_tasks': 0, 'training_ready': False,
            'independent_unit': 'training_episodes_not_roots_candidates_or_cells'}
