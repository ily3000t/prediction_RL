"""Observed-support proximity ranking diagnostic, NOT collision or policy safety.

The root-only ego rollout is shared by all predictors. Future ego and censoring
masks are evaluation labels only, never prediction inputs. No calibrated bands
are repurposed as action guarantees.
"""
from collections import Counter
import itertools
import math

import torch

from .offline_diagnostics import average

CONTROL = {'TICK_LENGTH': .2, 'MAX_POSITIVE_ACCELERATION': 4.5,
           'MAX_NEGATIVE_ACCELERATION': -6., 'MAX_SPEED': 30.}
PARITY = {'position_m': .01, 'speed_mps': .001, 'acceleration_mps2': .001}
FIELDS = ('score_mae_m', 'pair_delta_mae_m', 'pair_order_accuracy',
          'chosen_proxy_regret_m', 'best_proxy_tie_hit_rate')
TIE_EPS_M = 1e-6  # Numeric equality only; not a safety/practical-effect threshold.


def command_speed(speed, acceleration, jerk):
    if not all(math.isfinite(x) for x in (speed, acceleration, jerk)) or not 0 <= speed <= 30:
        raise ValueError('Invalid ego state/command')
    if not -5 <= jerk <= 5:
        raise ValueError('Jerk outside frozen upstream bounds')
    acceleration = min(4.5, max(-6., acceleration + jerk * .2))
    return min(30., max(0., speed + acceleration * .2))


def rollout_ego(root, plans, geometry):
    """Input whitelist: one observed frame and plans only, no future metadata.

    Semi-implicit distance and measured-acceleration recurrence are a diagnostic
    approximation of this pinned execution path, not a general SUMO emulator.
    Actual execution parity is measured separately on every evaluated root.
    """
    geometry.geometry.validate_frame(root)
    if plans.shape != (5, 25) or not torch.isfinite(plans).all():
        raise ValueError('Expected five finite frozen probe plans')
    expected = plans.new_zeros(5, 25); expected[:, 0] = plans.new_tensor([-5, -2.5, 0, 2.5, 5])
    if not torch.equal(plans, expected):
        raise ValueError('Different intervention or continuation rule')
    ego = root['vehicles']['ego']; lane = ego['lane_id']; position = ego['lane_position_m']
    anchor = geometry.xy(lane, position); result = torch.zeros(5, 25, 4, dtype=torch.float64)
    for c, plan in enumerate(plans.tolist()):
        speed, acceleration, distance = ego['speed'], ego['acceleration'], 0.
        for t, jerk in enumerate(plan):
            new_speed = command_speed(speed, acceleration, jerk)
            acceleration = (new_speed - speed) / .2; speed = new_speed
            distance += speed * .2
            x, y = geometry.xy(lane, position + distance)
            result[c, t] = result.new_tensor([x-anchor[0], y-anchor[1], speed, acceleration])
    return result


def ego_labels(pack):
    candidates = sorted(pack['candidates'], key=lambda c: c['candidate_id'])
    if [c['candidate_id'] for c in candidates] != list(range(5)):
        raise ValueError('Missing/duplicate ego candidate labels')
    values, masks = [], []
    for candidate in candidates:
        labels = candidate['labels']; i = labels['actor_ids'].index('ego')
        values.append(labels['future'][i]); masks.append(labels['future_mask'][i])
    values = torch.tensor(values, dtype=torch.float64); masks = torch.tensor(masks)
    if values.shape != (5, 25, 4) or masks.shape != (5, 25) or masks.dtype != torch.bool or not torch.isfinite(values).all():
        raise ValueError('Invalid ego labels')
    return values, masks


def parity_metrics(predicted, observed, mask):
    error = predicted - observed; valid = error[mask]
    row = {'observed_frames': int(mask.sum()), 'gate': False}
    if len(valid):
        position = valid[:, :2].square().sum(-1).sqrt()
        row.update(position_max_m=float(position.max()), position_mean_m=float(position.mean()),
                   speed_max_mps=float(valid[:, 2].abs().max()), acceleration_max_mps2=float(valid[:, 3].abs().max()))
        row['gate'] = (row['position_max_m'] <= PARITY['position_m'] and
                       row['speed_max_mps'] <= PARITY['speed_mps'] and
                       row['acceleration_max_mps2'] <= PARITY['acceleration_mps2'])
    return row


def score_metrics(prediction, truth):
    """Higher minimum FRONT-BUMPER distance is the sole proxy utility here."""
    if prediction.shape != (5,) or truth.shape != (5,) or not torch.isfinite(prediction).all() or not torch.isfinite(truth).all():
        raise ValueError('Require five finite comparable proxy scores')
    pairs, correct, informative = [], [], 0
    for a, b in itertools.combinations(range(5), 2):
        actual = float(truth[a]-truth[b]); estimated = float(prediction[a]-prediction[b])
        pairs.append(abs(estimated-actual))
        if abs(actual) > TIE_EPS_M:
            informative += 1
            correct.append(abs(estimated) > TIE_EPS_M and actual*estimated > 0)
    best = float(truth.max()); maximum = float(prediction.max())
    # Fixed lowest-ID numeric-tie rule; never choose using oracle scores.
    choice = next(i for i, value in enumerate(prediction) if maximum-float(value) <= TIE_EPS_M)
    return {'predicted_scores_m': prediction.tolist(), 'observed_scores_m': truth.tolist(),
            'chosen_candidate_id': choice, 'observed_spread_m': float(truth.max()-truth.min()),
            'informative_pairs': informative, 'score_mae_m': float((prediction-truth).abs().mean()),
            'pair_delta_mae_m': average(pairs), 'pair_order_accuracy': average(correct),
            'chosen_proxy_regret_m': max(0., best-float(truth[choice])),
            'best_proxy_tie_hit_rate': float(best-float(truth[choice]) <= TIE_EPS_M)}


def ranking_rows(predictions, batch, histories, packs, geometry):
    if any(p.shape != batch.targets.shape or not torch.isfinite(p).all() for p in predictions.values()):
        raise ValueError('Invalid neighbor predictions')
    if len(histories) != len(batch.root_ids) or len(packs) != len(histories):
        raise ValueError('Root count mismatch')
    rows = []
    for i, (h, pack) in enumerate(zip(histories, packs)):
        if (h['root_id'], pack['root_id'], h['episode_id'], pack['episode_id']) != (
                batch.root_ids[i], batch.root_ids[i], batch.episode_ids[i], batch.episode_ids[i]):
            raise ValueError('Root/episode order mismatch')
        predicted_ego = rollout_ego(h['history'][-1], batch.plans[i], geometry)
        observed_ego, ego_mask = ego_labels(pack)
        # Exactly the same actor-time support for ALL five branches and methods.
        # This is retrospective censoring, never an online availability mask.
        common = (batch.target_mask[i] & ego_mask[:, None, :]).all(0)
        actor_mask = batch.observations['actor_mask'][i]
        full = bool(actor_mask.any() and common[actor_mask].all())
        row = {'root_id': h['root_id'], 'episode_id': h['episode_id'],
               'common_actor_time_cells': int(common.sum()), 'common_time_steps': int(common.any(0).sum()),
               'selected_actors': int(actor_mask.sum()),
               'omitted_root_actors': len(h['history'][-1]['vehicles'])-1-int(actor_mask.sum()),
               'full_5s_all_selected_actors': full,
               'ego_parity': parity_metrics(predicted_ego, observed_ego, ego_mask),
               'new_actor_ids_by_candidate': {str(c['candidate_id']): c['labels'].get('new_actor_ids', []) for c in pack['candidates']},
               'termination_counts': dict(Counter(c['labels']['termination_reason'] or 'none' for c in pack['candidates'])),
               'methods': {}}
        if common.any():
            actual_distance = (batch.targets[i, ..., :2].double()-observed_ego[:, None, :, :2]).norm(dim=-1)
            truth = actual_distance[:, common].min(-1).values
            for name, prediction in predictions.items():
                distance = (prediction[i, ..., :2].double()-predicted_ego[:, None, :, :2]).norm(dim=-1)
                row['methods'][name] = score_metrics(distance[:, common].min(-1).values, truth)
        rows.append(row)
    return rows


def summarize_ranking(rows, roster, methods):
    if len({r['root_id'] for r in rows}) != len(rows) or not set(r['episode_id'] for r in rows) <= set(roster):
        raise ValueError('Duplicate roots/wrong episode roster')
    output = {'roots': len(rows), 'episodes': len(roster),
              'empty_support_roots': sum(not r['methods'] for r in rows),
              'ego_parity_gate': bool(rows) and all(r['ego_parity']['gate'] for r in rows),
              'ego_parity_failed_roots': [r['root_id'] for r in rows if not r['ego_parity']['gate']],
              'roots_with_omitted_actors': sum(r['omitted_root_actors'] > 0 for r in rows),
              'roots_with_new_actors': sum(any(r['new_actor_ids_by_candidate'].values()) for r in rows),
              'support_views': {}}
    for view in ('all_common_observed_support', 'complete_5s_all_selected_actors'):
        selected = [r for r in rows if r['methods'] and
                    (view == 'all_common_observed_support' or r['full_5s_all_selected_actors'])]
        values = {}; paired = {}
        for method in methods:
            episodes = [{'episode_id': e, **{k: average([r['methods'][method][k] for r in selected
                         if r['episode_id'] == e]) for k in FIELDS}} for e in roster]
            values[method] = {'root_equal': {k: average([r['methods'][method][k] for r in selected]) for k in FIELDS},
                              'episode_equal': {k: average([e[k] for e in episodes]) for k in FIELDS},
                              'root_denominators': {k: sum(r['methods'][method][k] is not None for r in selected) for k in FIELDS},
                              'episode_denominators': {k: sum(e[k] is not None for e in episodes) for k in FIELDS},
                              'episode_metrics': episodes}
        for method in methods:
            if method == 'conditional': continue
            paired[method+'__to__conditional'] = {}
            for key in FIELDS:
                delta = [(r['episode_id'], r['methods']['conditional'][key]-r['methods'][method][key])
                         for r in selected if r['methods']['conditional'][key] is not None and r['methods'][method][key] is not None]
                per_episode = [average([v for e, v in delta if e == episode]) for episode in roster]
                paired[method+'__to__conditional'][key] = {'paired_roots': len(delta),
                    'paired_episodes': sum(v is not None for v in per_episode),
                    'root_mean_right_minus_left': average([v for _, v in delta]),
                    'episode_mean_right_minus_left': average(per_episode)}
        output['support_views'][view] = {'roots': len(selected), 'methods': values, 'paired_differences': paired,
            'observed_score_spread_m': [r['methods'][methods[0]]['observed_spread_m'] for r in selected]}
    return output
