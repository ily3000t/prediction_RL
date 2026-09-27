"""Explicit development trajectory objective; not a viability/safety objective."""
import torch

LOSS_CONTRACT = {'version': 'root_candidate_masked_mse_v1', 'channel_scales': [10.,3.,5.,2.],
                 'channel_weights': [1.,1.,1.,1.], 'reduction': 'cells_then_candidates_then_roots',
                 'empty_branch': 'exclude_and_count', 'empty_batch': 'error'}


def trajectory_loss(prediction, target, mask):
    if (prediction.ndim != 5 or prediction.shape != target.shape or prediction.shape[-1] != 4 or
            mask.shape != target.shape[:-1] or mask.dtype != torch.bool or
            prediction.dtype != torch.float32 or target.dtype != torch.float32 or
            prediction.device != target.device or target.device != mask.device):
        raise ValueError('Expected matching float32 [root,candidate,actor,time,4] and Boolean masks')
    if not torch.isfinite(prediction).all() or not torch.isfinite(target).all():
        raise ValueError('Nonfinite trajectories are errors, even at masked positions')
    scales = prediction.new_tensor(LOSS_CONTRACT['channel_scales'])
    weights = prediction.new_tensor(LOSS_CONTRACT['channel_weights'])
    residual = torch.where(mask[...,None], prediction-target, 0.) / scales
    squared = (residual.square() * weights).sum(-1) / weights.sum()
    counts = mask.sum((-1,-2))
    branches = squared.sum((-1,-2)) / counts.clamp_min(1)
    branch_valid = counts > 0
    candidates_per_root = branch_valid.sum(-1)
    root_valid = candidates_per_root > 0
    if not root_valid.any():
        raise ValueError('No supervised trajectory in batch; do not perform an optimizer step')
    per_root = branches.sum(-1) / candidates_per_root.clamp_min(1)
    loss = per_root[root_valid].mean()
    if not torch.isfinite(loss): raise ValueError('Nonfinite trajectory loss')
    counts_report = {'valid_actor_time_cells': int(mask.sum()), 'supervised_branches': int(branch_valid.sum()),
                     'empty_branches': int((~branch_valid).sum()), 'supervised_roots': int(root_valid.sum()),
                     'empty_roots': int((~root_valid).sum())}
    return loss, counts_report
