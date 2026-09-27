"""Light shared history/interaction encoder with optional candidate conditioning."""
import torch
from torch import nn

from .interface import validate_batch, validate_plans


class MaskedHistory(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.cell = nn.GRUCell(input_dim, hidden_dim)

    def forward(self, values, mask):
        h = values.new_zeros((*values.shape[:-2], self.cell.hidden_size))
        h = h.reshape(-1, self.cell.hidden_size)
        sequence = values.reshape(-1, values.shape[-2], values.shape[-1])
        valid = mask.reshape(-1, mask.shape[-1])
        for t in range(sequence.shape[1]):
            x = torch.where(valid[:, t, None], sequence[:, t], 0.0)
            updated = self.cell(x, h)
            h = torch.where(valid[:, t, None], updated, h)
        return h.reshape(*values.shape[:-2], -1)


class SceneEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        d = config.hidden_dim
        # Fixed unit conversions, not statistics fitted on development/test data.
        self.register_buffer('physical_scale', torch.tensor([100, 10, 30, 5, 100, 5, 2, 1, 1, 1.]))
        self.register_buffer('summary_scale', torch.tensor([12, 100, 10, 10, 30, 30, 1.]))
        self.physical_history = MaskedHistory(11, d)  # SI-scaled features + fixed-grid time.
        self.summary_history = MaskedHistory(15, d)  # values + scalar masks + time.
        self.summary_group = nn.Embedding(3, d)
        self.ego_role = nn.Parameter(torch.zeros(d))
        self.attention = nn.MultiheadAttention(d, config.attention_heads, dropout=0.0, batch_first=True)
        self.norm = nn.LayerNorm(d)

    def forward(self, batch):
        physical = torch.cat([batch['ego_features'][:, None], batch['actor_features']], dim=1)
        physical_mask = torch.cat([batch['ego_history_mask'][:, None], batch['actor_history_mask']], dim=1)
        t = physical.shape[-2]
        # Relative index encodes where holes occur without feeding candidate/future data.
        time = torch.linspace(-1, 0, t, device=physical.device).view(1, 1, t, 1).expand(*physical.shape[:-1], 1)
        encoded = self.physical_history(torch.cat([physical / self.physical_scale, time], -1), physical_mask)
        ego = encoded[:, :1] + self.ego_role
        scalar_mask = batch['summary_feature_mask']
        summaries = torch.where(scalar_mask, batch['summary_features'] / self.summary_scale, 0.0)
        summary_time = time[:, :3]
        summary_input = torch.cat([summaries, scalar_mask.to(summaries.dtype), summary_time], dim=-1)
        summary = self.summary_history(summary_input, batch['summary_history_mask'])
        summary = summary + self.summary_group.weight[None]
        tokens = torch.cat([ego, encoded[:, 1:], summary], dim=1)
        valid = torch.cat([torch.ones_like(batch['actor_mask'][:, :1]), batch['actor_mask'], batch['summary_mask']], dim=1)
        # The always-valid ego prevents all-key-masked attention NaNs.
        attended, _ = self.attention(tokens, tokens, tokens, key_padding_mask=~valid, need_weights=False)
        context = self.norm(tokens + attended)
        return torch.where(valid[..., None], context, 0.0)


class ResponsePredictor(nn.Module):
    """One trajectory per root physical neighbor; summaries are not target vehicles.

    Ordinary mode explicitly ablates plan information with a constant reference.
    Architecture/parameter counts match; effective information does not.
    """
    def __init__(self, config, mode):
        super().__init__()
        if mode not in ('ordinary', 'conditional'):
            raise ValueError('Unknown prediction mode')
        self.config, self.mode = config, mode
        d = config.hidden_dim
        self.scene_encoder = SceneEncoder(config)
        # The frozen plan has one free scalar: first-step jerk, then 24 zeros.
        self.plan_encoder = nn.Sequential(nn.Linear(1, d), nn.Tanh())
        self.response_head = nn.Sequential(nn.Linear(3 * d, d), nn.Tanh(), nn.Linear(d, config.horizon_steps * 4))
        self.register_buffer('response_scale', torch.tensor([10., 3., 5., 2.]))

    def forward(self, batch, plans):
        validate_batch(batch, self.config)
        validate_plans(plans, batch['ego_features'].shape[0], batch['ego_features'].device, self.config)
        scene = self.scene_encoder(batch)  # Once per root batch, never once per candidate.
        b, c, _ = plans.shape
        k, h = self.config.neighbor_capacity, self.config.horizon_steps
        # No plan value reaches the ordinary model; validation only checks legality.
        conditioning = plans if self.mode == 'conditional' else torch.zeros_like(plans)
        plan = self.plan_encoder(conditioning[:, :, :1] / max(abs(self.config.jerk_min), abs(self.config.jerk_max)))
        actor = scene[:, None, 1:k + 1].expand(b, c, k, -1)
        ego = scene[:, None, :1].expand(b, c, k, -1)
        plan = plan[:, :, None].expand(b, c, k, -1)
        response = self.response_head(torch.cat([actor, ego, plan], dim=-1)).reshape(b, c, k, h, 4)
        # x/y are relative to the FIXED root ego frame, speed/acceleration in SI.
        root = batch['actor_features'][:, None, :, -1, :4]
        trajectory = root[..., None, :] + response * self.response_scale
        mask = batch['actor_mask'][:, None, :, None, None]
        trajectory = torch.where(mask, trajectory, 0.0)
        if not torch.isfinite(trajectory).all():
            raise ValueError('Nonfinite predictor output')
        return trajectory


class PredictorEnsemble(nn.Module):
    def __init__(self, config, mode, initialization_seeds):
        super().__init__()
        seeds = tuple(initialization_seeds)
        if (len(seeds) != config.ensemble_size or len(set(seeds)) != len(seeds) or
                any(type(s) is not int or s < 0 or s >= 2**32 for s in seeds)):
            raise ValueError('Require three distinct uint32 member initialization seeds')
        self.config, self.mode, self.initialization_seeds = config, mode, seeds
        members = []
        # Local initialization must not consume caller CPU RNG or mutate CUDA RNG.
        with torch.random.fork_rng(devices=[]):
            for seed in seeds:
                torch.random.default_generator.manual_seed(seed)
                members.append(ResponsePredictor(config, mode))
        self.members = nn.ModuleList(members)

    def forward(self, batch, plans):
        trajectories = torch.stack([member(batch, plans) for member in self.members])
        return {'member_trajectories': trajectories, 'mean': trajectories.mean(0),
                'disagreement': trajectories.std(0, unbiased=False), 'actor_mask': batch['actor_mask']}
