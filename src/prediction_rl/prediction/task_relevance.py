"""Retrospective task diagnosis, not a new score, reward, or policy feature."""
from collections import Counter
import itertools
import math

import torch

from .candidate_ranking import ego_labels, ranking_rows, TIE_EPS_M
from .offline_diagnostics import average


def branch_task(branch, labels):
    trace = branch['trace']; frames = branch['future_metadata']['frames']
    if len(trace) != labels['observed_steps'] or len(frames) != len(trace):
        raise ValueError('Trace/label length differs')
    components = []; discrepancies = []; unchecked = 0
    for step in trace:
        audit = step['info']['execution_audit']; reason = audit['termination_reason']
        actual = step['reward']; invalid = audit['invalid_action_reward']
        if not all(math.isfinite(x) for x in (actual, invalid)):
            raise ValueError('Nonfinite upstream reward')
        terminal = -10. if reason == 'upstream_collision' else 10. if reason == 'upstream_arrival' else 0.
        time_cost = 0. if terminal else -.1*.2
        jerk = audit['measured_jerk']
        # Collision/arrival reward ignores jerk. A timeout may remove ego before
        # the read-only audit; retain an explicitly unverified residual then.
        jerk_cost = 0. if terminal else None if jerk is None else -.1*jerk**2*.2
        expected = None if jerk_cost is None else terminal+time_cost+jerk_cost+invalid
        if expected is None: unchecked += 1
        else: discrepancies.append(abs(actual-expected))
        components.append({'terminal': terminal, 'time': time_cost, 'jerk': jerk_cost,
                           'invalid_action': invalid, 'actual': actual})
    if discrepancies and max(discrepancies) > 1e-6:
        raise ValueError('Recorded reward differs from frozen upstream Slotted Jerk semantics')
    if trace[-1]['info']['execution_audit']['termination_reason'] != labels['termination_reason']:
        raise ValueError('Terminal label differs from trace')
    return {'observed_return': sum(s['reward'] for s in trace), 'steps': len(trace),
            'elapsed_s': sum(s['info']['execution_audit']['simulation_elapsed_s'] for s in trace),
            'done': trace[-1]['done'], 'full_horizon': labels['full_time_horizon_observed'],
            'reason': labels['termination_reason'], 'events': labels['events'],
            'reward_components': {k: sum(c[k] for c in components) if all(c[k] is not None for c in components) else None
                                  for k in ('terminal', 'time', 'jerk', 'invalid_action')},
            'reward_unverified_steps': unchecked, 'reward_max_abs_error': max(discrepancies, default=None)}


def coverage(branch, selected, root_ids, geometry):
    nearest = None; min_gap = None; seen = 0
    for t, (step, frame) in enumerate(zip(branch['trace'], branch['future_metadata']['frames'])):
        if step['done']: continue  # Never fabricate geometry on a terminal frame.
        geometry.validate_frame(frame); ego = frame['vehicles']['ego']
        for actor, state in frame['vehicles'].items():
            if actor == 'ego': continue
            pair = geometry.pair(ego, state); seen += 1
            role = 'selected' if actor in selected else 'omitted_root' if actor in root_ids else 'new_actor'
            detail = {'actor_id': actor, 'time_index': t, 'horizon_s': (t+1)*.2, 'role': role,
                      'ego_lane': ego['lane_id'], 'actor_lane': state['lane_id']}
            value = pair['front_bumper_distance_m']
            if nearest is None or value < nearest['distance_m']:
                nearest = {**detail, 'distance_m': value}
            gap = pair['same_path_gap_m']
            if gap is not None and (min_gap is None or gap < min_gap['gap_m']):
                min_gap = {**detail, 'gap_m': gap}
    return {'nonterminal_actor_time_cells': seen, 'nearest_front_bumper': nearest, 'minimum_same_path_gap': min_gap,
            'causal_collision_actor_identified': False}


def pair_task_agreement(scores, tasks):
    result = {}
    for name in ('observed_return', 'upstream_collision', 'upstream_arrival'):
        counts = Counter(); values = []
        for a, b in itertools.combinations(range(5), 2):
            if name == 'observed_return':
                comparable = (tasks[a]['done'] and tasks[b]['done']) or (tasks[a]['full_horizon'] and tasks[b]['full_horizon'])
                difference = tasks[a]['observed_return']-tasks[b]['observed_return']
            else:
                left, right = tasks[a]['events'][name], tasks[b]['events'][name]
                comparable = left['mask'] and right['mask']
                difference = (left['value']-right['value'])*(-1 if name == 'upstream_collision' else 1)
            if not comparable: counts['censored_or_incomparable'] += 1; continue
            if abs(difference) <= 1e-6: counts['task_tied'] += 1; continue
            if scores is None: counts['no_geometry'] += 1; continue
            delta = scores[a]-scores[b]
            counts['informative'] += 1
            if abs(delta) <= TIE_EPS_M: counts['proxy_tied'] += 1
            correct = abs(delta) > TIE_EPS_M and delta*difference > 0
            counts['concordant' if correct else 'not_concordant'] += 1; values.append(float(correct))
        result[name] = {'agreement': average(values), 'counts': dict(counts)}
    return result


def diagnose_root(predictions, batch, history, pack, branches, geometry):
    if len(batch.root_ids) != 1 or len(branches) != 5:
        raise ValueError('One complete root family required')
    ranked = ranking_rows(predictions, batch, [history], [pack], geometry)[0]
    candidates = sorted(pack['candidates'], key=lambda c: c['candidate_id'])
    tasks = [branch_task(b, c['labels']) for b, c in zip(branches, candidates)]
    selected = {a for a in history['inputs']['actor_ids'] if a is not None}
    root_ids = set(history['history'][-1]['vehicles'])
    ranked.update(root_lane=history['history'][-1]['vehicles']['ego']['lane_id'],
                  tasks=tasks, coverage=[coverage(b, selected, root_ids, geometry.geometry) for b in branches])
    scores = {m: v['predicted_scores_m'] for m, v in ranked['methods'].items()}
    scores['observed_proxy'] = next(iter(ranked['methods'].values()))['observed_scores_m'] if ranked['methods'] else None
    ranked['task_agreement'] = {m: pair_task_agreement(scores.get(m), tasks) for m in [*predictions, 'observed_proxy']}
    observed_ego, ego_mask = ego_labels(pack)
    common = (batch.target_mask[0] & ego_mask[:, None, :]).all(0)
    ranked['critical_cells'] = []; ranked['wrong_order_pairs'] = []
    for cid in range(5):
        if not common.any(): continue
        distance = (batch.targets[0, cid, ..., :2].double()-observed_ego[cid, None, :, :2]).norm(dim=-1)
        index = int(distance.masked_fill(~common, float('inf')).argmin()); k, t = divmod(index, 25)
        actor = history['inputs']['actor_ids'][k]
        frame = branches[cid]['future_metadata']['frames'][t]
        ranked['critical_cells'].append({'candidate_id': cid, 'actor_id': actor, 'slot': k, 'time_index': t,
            'horizon_s': (t+1)*.2, 'ego_lane': frame['vehicles']['ego']['lane_id'], 'actor_lane': frame['vehicles'][actor]['lane_id'],
            'neighbor_position_error_m': {m: float((v[0, cid, k, t, :2]-batch.targets[0, cid, k, t, :2]).norm()) for m, v in predictions.items()},
            'response_delta_error_m': {m: average([float(((v[0,cid,k,t,:2]-v[0,other,k,t,:2])-
                (batch.targets[0,cid,k,t,:2]-batch.targets[0,other,k,t,:2])).norm()) for other in range(5) if other != cid]) for m,v in predictions.items()}})
    for m, values in ranked['methods'].items():
        truth, predicted = values['observed_scores_m'], values['predicted_scores_m']
        for a, b in itertools.combinations(range(5), 2):
            delta, estimate = truth[a]-truth[b], predicted[a]-predicted[b]
            if abs(delta) > TIE_EPS_M and (abs(estimate) <= TIE_EPS_M or delta*estimate <= 0):
                ranked['wrong_order_pairs'].append({'method': m, 'left': a, 'right': b, 'true_difference_m': delta,
                    'predicted_difference_m': estimate, 'left_reason': tasks[a]['reason'], 'right_reason': tasks[b]['reason']})
    return ranked


def aggregate(rows, roster, methods):
    if len({r['root_id'] for r in rows}) != len(rows) or not {r['episode_id'] for r in rows} <= set(roster):
        raise ValueError('Duplicate root/wrong episode roster')
    result = {'roots': len(rows), 'episodes': len(roster), 'branches': 5*len(rows),
        'empty_geometry_roots': sum(not r['methods'] for r in rows),
        'termination_counts': dict(Counter(t['reason'] or 'horizon_nonterminal' for r in rows for t in r['tasks'])),
        'reward_unverified_steps': sum(t['reward_unverified_steps'] for r in rows for t in r['tasks']),
        'nearest_actor_role_counts': dict(Counter(c['nearest_front_bumper']['role'] if c['nearest_front_bumper'] else 'no_geometry'
            for r in rows for c in r['coverage'])), 'methods': {}}
    for method in [*methods, 'observed_proxy']:
        output = {}
        for task in ('observed_return', 'upstream_collision', 'upstream_arrival'):
            metrics = [r['task_agreement'][method][task] for r in rows]
            episodes = [{'episode_id': e, 'agreement': average([r['task_agreement'][method][task]['agreement'] for r in rows if r['episode_id'] == e])} for e in roster]
            counts = Counter()
            for v in metrics: counts.update(v['counts'])
            output[task] = {'root_equal_agreement': average([v['agreement'] for v in metrics]),
                'episode_equal_agreement': average([e['agreement'] for e in episodes]), 'pair_counts_descriptive_only': dict(counts),
                'eligible_roots': sum(v['agreement'] is not None for v in metrics),
                'eligible_episodes': sum(e['agreement'] is not None for e in episodes), 'episode_metrics': episodes}
        result['methods'][method] = output
    result['by_root_lane'] = {lane: {'roots': sum(r['root_lane'] == lane for r in rows),
        'wrong_order_counts': dict(Counter(p['method'] for r in rows if r['root_lane'] == lane for p in r['wrong_order_pairs']))}
        for lane in sorted({r['root_lane'] for r in rows})}
    return result
