"""Observed-root prediction channels, not an action selector or safety certificate."""
import numpy as np
import torch

from prediction_rl.data.actor_adapter import build_actor_inputs
from prediction_rl.prediction.interface import PredictorConfig, collate_inputs, probe_plans
from prediction_rl.prediction.candidate_ranking import rollout_ego

VERSION = 'prediction_probe_features_v1'
TIMES = (.2, 1., 3., 5.)
INDICES = (0, 4, 14, 24)
ROLES = ('target_front', 'target_rear')
METADATA = ('front_present', 'front_history_fraction', 'rear_present',
            'rear_history_fraction', 'pool_present', 'ego_history_fraction',
            'selected_fraction', 'omitted_count_fraction', 'omitted_present')
NAMES = tuple(f'probe_{c}.{role}.t{t:g}.{field}' for c in range(5)
              for role in ROLES for t in TIMES
              for field in ('dx_over_100m', 'dy_over_100m', 'dv_over_30mps')) + tuple(
    f'probe_{c}.pool.t{t:g}.min_front_bumper_distance_over_100m'
    for c in range(5) for t in TIMES) + METADATA
DIMENSION = len(NAMES)  # 120 relative states + 20 pooled distances + 9 root channels.


def summarize(inputs, neighbors, ego):
    """One root, ANY candidate count; pooling follows ensemble MEAN trajectories.

    No labels, future masks, future IDs, reward or action choice are consulted.
    Coordinates are world-axis offsets, NOT signed road gaps/body clearance.
    """
    cfg = PredictorConfig()
    batch = collate_inputs([inputs], cfg)
    if (neighbors.ndim != 4 or neighbors.shape[1:] != (12, 25, 4)
            or ego.shape != (neighbors.shape[0], 25, 4) or neighbors.shape[0] < 1
            or not torch.isfinite(neighbors).all() or not torch.isfinite(ego).all()):
        raise ValueError('Invalid finite neighbor/ego trajectories')
    ids = inputs['actor_ids']; mask = batch['actor_mask'][0]
    if (len(ids) != 12 or len(set(x for x in ids if x is not None)) != int(mask.sum())
            or any((x is not None) != bool(mask[i]) for i,x in enumerate(ids))):
        raise ValueError('Actor identities/masks disagree')
    roles = inputs['mandatory_roles']; values = []; metadata = []
    # Convert without silently clipping physically implausible model values.
    neighbors = neighbors.detach().cpu().double(); ego = ego.detach().cpu().double()
    for c in range(len(ego)):
        for role in ROLES:
            actor = roles.get(role)
            if actor is not None and actor not in ids:
                raise ValueError('Mandatory actor missing from physical slots')
            if actor is None:
                values.extend([0.] * 12)
            else:
                state = neighbors[c, ids.index(actor), list(INDICES), :3]
                relative = (state - ego[c, list(INDICES), :3]) / state.new_tensor([100.,100.,30.])
                values.extend(relative.flatten().tolist())
    for c in range(len(ego)):
        if mask.any():
            distance = (neighbors[c, mask][:, list(INDICES), :2]
                        - ego[c, list(INDICES), :2]).norm(dim=-1).min(0).values / 100.
            values.extend(distance.tolist())
        else:
            values.extend([0.] * 4)
    for role in ROLES:
        actor = roles.get(role)
        metadata.extend([float(actor is not None), 0. if actor is None else
                         float(batch['actor_history_mask'][0, ids.index(actor)].float().mean())])
    omitted = len(inputs['root_omitted_ids'])
    metadata.extend([float(mask.any()), float(batch['ego_history_mask'][0].float().mean()),
                     float(mask.sum()) / 12., omitted / (omitted + 12.), float(omitted > 0)])
    result = np.asarray(values + metadata, dtype=np.float32)
    if not np.isfinite(result).all():
        raise ValueError('Feature conversion overflow; invalidate rather than substitute')
    return result


class FrozenProbeFeatures:
    """CPU inference on observed history only; independent of simulator advancement."""
    def __init__(self, geometry, config, model):
        if (model.config != config or model.mode not in ('ordinary','conditional')
                or model.training or any(p.requires_grad or p.device.type != 'cpu' for p in model.parameters())):
            raise ValueError('Require strict loaded, frozen CPU evaluation ensemble')
        self.geometry, self.config, self.model = geometry, config, model

    def __call__(self, history):
        cfg = self.config
        inputs = build_actor_inputs(history, self.geometry.geometry, cfg.neighbor_capacity,
                                    cfg.history_steps, cfg.tick_s)
        batch = collate_inputs([inputs], cfg); plans = probe_plans(1, cfg)
        with torch.inference_mode():
            predicted = self.model(batch, plans)['mean'][0]
            ego = rollout_ego(history[-1], plans[0], self.geometry)
            result = summarize(inputs, predicted, ego)
        if result.shape != (DIMENSION,):
            raise ValueError('Feature layout changed')
        return result
