"""Frozen CPU RCCI v2 provider for bounded integration, not policy training."""
import numpy as np
import torch

from prediction_rl.data.actor_adapter import build_actor_inputs
from prediction_rl.prediction.interface import collate_inputs, probe_plans
from prediction_rl.prediction.candidate_ranking import rollout_ego
from prediction_rl.prediction.kinematic_reference import LaneConstantVelocity
from prediction_rl.features.road_consequence_geometry import RoadConsequenceMap
from prediction_rl.features.road_consequences_v2 import DIMENSION, summarize_road_consequences_v2


class FrozenRCCIV2Features:
    release = 'bounded_native_engineering_only_not_training'

    def __init__(self, network, config, model):
        if (model.config != config or model.mode not in ('ordinary','conditional') or model.training
                or any(p.requires_grad or p.device.type != 'cpu' for p in model.parameters())
                or any(b.device.type != 'cpu' for b in model.buffers())):
            raise ValueError('Require strict loaded frozen CPU ensemble')
        self.geometry = LaneConstantVelocity(network)
        self.road = RoadConsequenceMap(network)
        self.config,self.model = config,model
        self.calls = 0
        self.last = None

    def __call__(self, history):
        self.last = None  # An exception must never leave usable stale diagnostics.
        cfg = self.config
        inputs = build_actor_inputs(history,self.road.geometry,cfg.neighbor_capacity,cfg.history_steps,cfg.tick_s)
        batch = collate_inputs([inputs],cfg); plans = probe_plans(1,cfg)
        with torch.inference_mode():
            self.calls += 1
            predicted = self.model(batch,plans)['mean'][0]
            ego = rollout_ego(history[-1],plans[0],self.geometry)
            result = summarize_road_consequences_v2(history,predicted,ego,self.road)
        features = result['features']
        if features.shape != (DIMENSION,) or features.dtype != np.float32 or not np.isfinite(features).all():
            raise ValueError('Invalid RCCI block; no repairs/stale fallback')
        self.last = {'result':result,'neighbors':predicted.detach().cpu().numpy().copy(),
            'ego':ego.detach().cpu().numpy().copy(),'observation_time_s':history[-1]['simulation_time_s']}
        return features.copy()
