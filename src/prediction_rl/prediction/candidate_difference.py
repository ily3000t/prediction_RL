"""Opt-in same-root response-difference regression; legacy MSE stays unchanged.

Inputs must already be aligned by stable actor ID in the fixed ROOT ego frame.
Masks are supervision metadata, never online policy inputs. This objective does
not infer missing futures, rank actions, or force different actions to disagree.
"""
from copy import deepcopy
from itertools import combinations
import math

import torch

from .losses import LOSS_CONTRACT, trajectory_loss


DIFFERENCE_LOSS_CONTRACT = {
    'version': 'root_candidate_response_difference_v1',
    'trajectory_contract': deepcopy(LOSS_CONTRACT),
    'pairing': 'all_unordered_candidates_within_aligned_root',
    'coordinate_frame': 'fixed_root_ego',
    'mask': 'same_actor_same_time_intersection',
    'reduction': 'common_cells_then_valid_pairs_then_trajectory_supervised_roots',
    'root_without_common_pairs': 'zero_difference_keep_trajectory_and_count',
    'empty_trajectory_batch': 'error',
    'selection_metric': 'validation_root_equal_scaled_mse',
    'ordinary_mode': 'not_trainable_with_difference_objective',
}
DIFFERENCE_TRAINING_VERSION = 'root_family_epoch_cpu_difference_v1'


def checked_lambda(value):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError('difference_lambda must be a finite nonnegative number, not Boolean')
    return float(value)


def difference_training(baseline, weight):
    """Only loss identity/weight and the scheduled method set change.

    The baseline trajectory-MSE selection/patience rule deliberately remains
    frozen, so an objective AND selection-rule change are not confounded.
    """
    if (baseline['loss_contract'] != LOSS_CONTRACT['version']
            or baseline['channel_scales'] != LOSS_CONTRACT['channel_scales']
            or baseline['channel_weights'] != LOSS_CONTRACT['channel_weights']
            or baseline['selection_metric'] != DIFFERENCE_LOSS_CONTRACT['selection_metric']
            or 'difference_lambda' in baseline):
        raise ValueError('Require the frozen trajectory-MSE baseline training settings')
    training = deepcopy(baseline)
    training.update(methods=['conditional'],
                    loss_contract=DIFFERENCE_LOSS_CONTRACT['version'],
                    difference_lambda=checked_lambda(weight))
    return training


def _difference_loss(prediction, target, mask):
    # Public callers validate via the unchanged trajectory_loss first.
    root_valid = mask.flatten(1).any(1)
    scales = prediction.new_tensor(LOSS_CONTRACT['channel_scales'])
    weights = prediction.new_tensor(LOSS_CONTRACT['channel_weights'])
    pair_losses, pair_counts = [], []
    for left, right in combinations(range(prediction.shape[1]), 2):
        common = mask[:, left] & mask[:, right]
        residual = torch.where(common[..., None],
            (prediction[:, left] - prediction[:, right])
            - (target[:, left] - target[:, right]), 0.) / scales
        squared = (residual.square() * weights).sum(-1) / weights.sum()
        counts = common.sum((-1, -2))
        pair_losses.append(squared.sum((-1, -2)) / counts.clamp_min(1))
        pair_counts.append(counts)
    if pair_losses:
        counts = torch.stack(pair_counts, dim=1)
        valid = counts > 0
        valid_per_root = valid.sum(-1)
        per_root = torch.stack(pair_losses, dim=1).sum(-1) / valid_per_root.clamp_min(1)
        loss = per_root[root_valid].mean()
        pair_roots = valid_per_root > 0
        common_cells, valid_pairs = int(counts.sum()), int(valid.sum())
    else:
        # A differentiable zero without a potentially overflowing tensor sum.
        loss = prediction.reshape(-1)[0] * 0.
        pair_roots = torch.zeros_like(root_valid)
        common_cells = valid_pairs = 0
    if not torch.isfinite(loss):
        raise ValueError('Nonfinite candidate response-difference loss')
    candidates = prediction.shape[1]
    coverage = {
        'valid_candidate_pairs': valid_pairs,
        'possible_candidate_pairs': int(root_valid.sum()) * candidates * (candidates - 1) // 2,
        'pair_common_actor_time_cells': common_cells,
        'roots_with_valid_pairs': int((root_valid & pair_roots).sum()),
        'roots_without_valid_pairs': int((root_valid & ~pair_roots).sum()),
        'difference_root_denominator': int(root_valid.sum()),
    }
    return loss, coverage


def response_difference_loss(prediction, target, mask):
    trajectory_loss(prediction, target, mask)  # same shape/dtype/finite/empty checks
    return _difference_loss(prediction, target, mask)


def candidate_difference_objective(prediction, target, mask, weight):
    weight = checked_lambda(weight)
    absolute, counts = trajectory_loss(prediction, target, mask)
    difference, coverage = _difference_loss(prediction, target, mask)
    # Exact legacy value/gradient/update path when lambda=0: no added zero graph.
    objective = absolute if weight == 0 else absolute + weight * difference
    if not torch.isfinite(objective):
        raise ValueError('Nonfinite combined trajectory/difference objective')
    return objective, {
        **counts, **coverage,
        'trajectory_loss': float(absolute.detach()),
        'difference_loss': float(difference.detach()),
        'objective_loss': float(objective.detach()),
    }
