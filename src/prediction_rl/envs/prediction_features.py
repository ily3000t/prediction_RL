"""Blocking observation augmentation only. Does not connect DDPG training yet."""
from collections import deque
import time

import numpy as np

from prediction_rl.features.probe_features import DIMENSION, VERSION


class PredictionFeatureEnv:
    execution_contract = 'simulation_blocking_exact_v1'

    def __init__(self, base, mode, provider=None, read_frame=None):
        if mode not in ('baseline','zero','ordinary','conditional'):
            raise ValueError('Unknown comparison arm')
        learned = mode in ('ordinary','conditional')
        if (learned != (provider is not None and read_frame is not None)
                or (not learned and (provider is not None or read_frame is not None))):
            raise ValueError('Only learned arms accept prediction/state readers')
        if learned and provider.model.mode != mode:
            raise ValueError('Predictor mode mismatch')
        if base.execution_contract != self.execution_contract:
            raise ValueError('Need blocking base environment')
        from gym.spaces import Box
        self.base, self.mode, self.provider, self.read_frame = base, mode, provider, read_frame
        self.action_space = base.action_space
        self.base_dimension = base.observation_space.shape[0]
        if len(base.observation_space.shape) != 1 or base.observation_space.dtype != np.float32:
            raise ValueError('Require original flat float32 observation')
        if mode == 'baseline':
            self.observation_space = base.observation_space
        else:
            self.observation_space = Box(
                np.concatenate([base.observation_space.low, np.full(DIMENSION, -np.inf, np.float32)]),
                np.concatenate([base.observation_space.high, np.full(DIMENSION, np.inf, np.float32)]),
                dtype=np.float32)
        self.history = deque(maxlen=11)
        self._ready = self._faulted = self._closed = False

    def _augment(self, obs, info, terminal=False):
        started = time.perf_counter()
        obs = np.asarray(obs)
        if obs.shape != (self.base_dimension,) or obs.dtype != np.float32 or not np.isfinite(obs).all():
            raise ValueError('Invalid original observation')
        if self.mode == 'baseline':
            result = obs.copy()
        else:
            block = np.zeros(DIMENSION, np.float32)
            if self.provider is not None and not terminal:
                frame = self.read_frame()
                self.history.append(frame)
                block = self.provider(list(self.history))
                if block.shape != (DIMENSION,) or block.dtype != np.float32 or not np.isfinite(block).all():
                    raise ValueError('Invalid prediction features; no stale fallback')
            result = np.concatenate([obs, block])
        return result, {**info, 'prediction_features': {
            'version': VERSION, 'mode': self.mode, 'terminal_zero_block': terminal and self.mode != 'baseline',
            'history_frames': len(self.history), 'latency_s': time.perf_counter()-started}}

    def reset(self):
        if self._ready or self._closed or self._faulted:
            raise RuntimeError('Reset requires healthy finished environment')
        try:
            self.history.clear()
            obs, info = self.base.reset()
            result = self._augment(obs, info)
            self._ready = True
            return result
        except BaseException:
            self._faulted = True
            raise

    def step(self, action):
        if not self._ready or self._closed or self._faulted:
            raise RuntimeError('Reset healthy environment before stepping')
        try:
            obs, reward, done, info = self.base.step(action)  # Exactly once; no action modifications.
            result, info = self._augment(obs, info, terminal=done)
            self._ready = not done
            return result, reward, done, info
        except BaseException:
            self._faulted = True
            raise

    def close(self):
        if not self._closed:
            self._closed = True
            self._ready = False
            self.base.close()
