"""Strict P4c development contract; no simulator, future labels or policy imports."""
from dataclasses import asdict, dataclass, fields

import torch

from prediction_rl.data.actor_adapter import FEATURES, SUMMARY_FEATURES
from prediction_rl.data.merge_geometry import GROUPS


@dataclass(frozen=True)
class PredictorConfig:
    schema_version: int = 1
    neighbor_capacity: int = 12  # Ego is separate.
    history_steps: int = 11
    horizon_steps: int = 25
    tick_s: float = 0.2
    hidden_dim: int = 64
    attention_heads: int = 4
    ensemble_size: int = 3
    jerk_min: float = -5.0
    jerk_max: float = 5.0

    def __post_init__(self):
        integers = ('schema_version', 'neighbor_capacity', 'history_steps',
                    'horizon_steps', 'hidden_dim', 'attention_heads', 'ensemble_size')
        if any(type(getattr(self, k)) is not int for k in integers):
            raise ValueError('Integer model fields must be integers, not booleans')
        if (self.schema_version != 1 or self.neighbor_capacity != 12 or
                self.history_steps != 11 or self.horizon_steps != 25 or
                self.ensemble_size != 3):
            raise ValueError('Unsupported development tensor/ensemble contract')
        if self.hidden_dim < 4 or self.attention_heads < 1 or self.hidden_dim % self.attention_heads:
            raise ValueError('Invalid attention dimensions')
        if any(type(getattr(self, k)) not in (int, float) for k in ('tick_s', 'jerk_min', 'jerk_max')):
            raise ValueError('Invalid numeric model fields')
        if (self.tick_s, self.jerk_min, self.jerk_max) != (0.2, -5.0, 5.0):
            raise ValueError('Do not change the original control grid/bounds')

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict) or set(value) != {f.name for f in fields(cls)}:
            raise ValueError('Missing/unknown predictor configuration fields')
        return cls(**value)

    def to_dict(self):
        return asdict(self)


def tensor_shapes(config):
    k, t = config.neighbor_capacity, config.history_steps
    return {
        'ego_features': (t, 10), 'ego_history_mask': (t,),
        'actor_features': (k, t, 10), 'actor_history_mask': (k, t), 'actor_mask': (k,),
        'summary_features': (3, t, 7), 'summary_history_mask': (3, t),
        'summary_feature_mask': (3, t, 7), 'summary_mask': (3,),
    }


def validate_batch(batch, config):
    shapes = tensor_shapes(config)
    if not isinstance(batch, dict) or set(batch) != set(shapes):
        raise ValueError('Predictor accepts only fixed observation tensors')
    first = batch['ego_features']
    if not isinstance(first, torch.Tensor) or first.ndim != 3 or first.shape[0] < 1:
        raise ValueError('Require a nonempty root batch')
    size, device = first.shape[0], first.device
    for key, shape in shapes.items():
        value = batch[key]
        dtype = torch.bool if 'mask' in key else torch.float32
        if (not isinstance(value, torch.Tensor) or tuple(value.shape) != (size, *shape)
                or value.dtype != dtype or value.device != device):
            raise ValueError('Invalid tensor shape/dtype/device: ' + key)
        if not torch.isfinite(value).all():
            raise ValueError('Nonfinite input, including masked cells: ' + key)
    if not batch['ego_history_mask'][:, -1].all():
        raise ValueError('Root ego must be observed')
    for name in ('actor', 'summary'):
        if not torch.equal(batch[name + '_mask'], batch[name + '_history_mask'][:, :, -1]):
            raise ValueError('Root/history mask disagreement: ' + name)
    if (batch['summary_feature_mask'] & ~batch['summary_history_mask'].unsqueeze(-1)).any():
        raise ValueError('Summary scalar cannot be valid in a missing token')


def collate_inputs(inputs, config, device='cpu'):
    """Whitelist tensors; variable audit IDs/metrics and labels are never channels."""
    if not inputs:
        raise ValueError('Empty root batch')
    for row in inputs:
        required = {'schema_version': 1, 'neighbor_capacity': config.neighbor_capacity,
                    'history_steps': config.history_steps, 'tick_s': config.tick_s,
                    'ego_separate': True, 'feature_names': list(FEATURES),
                    'summary_feature_names': list(SUMMARY_FEATURES), 'summary_groups': list(GROUPS)}
        if any(row.get(k) != value for k, value in required.items()):
            raise ValueError('Actor adapter metadata/layout mismatch')
    result = {}
    for key in tensor_shapes(config):
        raw = torch.as_tensor([row[key] for row in inputs])
        # Do not silently cast invalid integer masks to bool.
        if 'mask' in key and raw.dtype != torch.bool:
            raise ValueError('Masks must contain booleans: ' + key)
        result[key] = raw.to(device=device, dtype=torch.bool if 'mask' in key else torch.float32)
    validate_batch(result, config)
    return result


def validate_plans(plans, batch_size, device, config):
    if (not isinstance(plans, torch.Tensor) or plans.ndim != 3 or
            plans.shape[0] != batch_size or plans.shape[1] < 1 or
            plans.shape[2] != config.horizon_steps or plans.dtype != torch.float32 or
            plans.device != device or not torch.isfinite(plans).all()):
        raise ValueError('Plans must be finite float32 [root,candidate,horizon] on the scene device')
    if ((plans < config.jerk_min) | (plans > config.jerk_max)).any():
        raise ValueError('Candidate exceeds upstream jerk bounds; never clip silently')
    if torch.count_nonzero(plans[:, :, 1:]):
        raise ValueError('v1 intervenes one original step, then fixed zero-jerk reference')


def probe_plans(batch_size, config, device='cpu'):
    plans = torch.zeros((batch_size, 5, config.horizon_steps), device=device, dtype=torch.float32)
    plans[:, :, 0] = torch.linspace(config.jerk_min, config.jerk_max, 5, device=device)
    return plans
