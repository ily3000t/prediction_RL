"""P2: upstream-preserving, process-owned SUMO session and audit adapter.

No prediction, action filtering, reward shaping, or wall-clock deadline logic.
A session owns upstream globals/CWD and must not share a process with training
or another SUMO session. reset() continues upstream traffic; new seeded scenes
belong in new worker processes, not per-episode reseeding.
"""
from contextlib import contextmanager
from copy import deepcopy
import importlib
import json
import os
from pathlib import Path
import random
import sys
import threading

import numpy as np

_SESSION_LOCK = threading.Lock()


def validate_settings(payload, defaults):
    if not isinstance(payload, dict):
        raise ValueError('Configuration must be a JSON object')
    unknown = set(payload) - set(defaults)
    if unknown:
        raise ValueError(f'Unknown upstream settings: {sorted(unknown)}')
    for key, value in payload.items():
        if key == 'SEED':
            if value == 'Random':
                continue
            validate_seed(value)
        elif isinstance(defaults[key], bool):
            if not isinstance(value, bool):
                raise ValueError(f'{key} requires boolean')
        elif isinstance(defaults[key], (int, float)):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
                raise ValueError(f'{key} requires a finite number')
        elif not isinstance(value, type(defaults[key])):
            raise ValueError(f'{key} has an incompatible type')


def validate_seed(seed):
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError('Use an explicit integer run seed in [0, 2**32)')


@contextmanager
def upstream_session(source, config, seed):
    """Initialize like a fresh upstream main.py, restoring globals on exit.

    For sequential diagnostics, restoring control.delay=0 at session entry is
    equivalent to a fresh module import. It is NOT done between reset calls.
    Policy training must use a dedicated process: RNG seeding occurs only here.
    """
    validate_seed(seed)
    source, config = Path(source).resolve(), Path(config).resolve()
    if not (source / 'merge_gym.py').is_file():
        raise FileNotFoundError(source / 'merge_gym.py')
    if not _SESSION_LOCK.acquire(blocking=False):
        raise RuntimeError('Only one upstream session per process is supported')
    old_cwd, old_path = Path.cwd(), list(sys.path)
    settings = None
    defaults = None
    rng = None
    control = None
    old_delay = None
    traci = None
    owns_connection = False
    try:
        sys.path.insert(0, str(source))
        # Generic upstream module names cannot safely coexist with other repos.
        for name in ('config', 'control', 'merge_gym', 'sumo', 'prediction', 'dqn', 'st'):
            module = sys.modules.get(name)
            if module is not None and Path(module.__file__).resolve().parent != source:
                raise RuntimeError(f'Foreign module already loaded: {name}')
        settings = importlib.import_module('config').Settings
        defaults = deepcopy(settings.export_settings())
        payload = json.loads(config.read_text(encoding='utf-8'))
        validate_settings(payload, defaults)
        for key, value in payload.items():
            setattr(settings, key, {int(k): v for k, v in value.items()} if isinstance(value, dict) else value)
        settings.SEED = seed
        settings.SYSTEM = 'Windows' if os.name == 'nt' else 'Linux'
        if settings.GYM_ENVIRONMENT != 'sumo-jerk-continuous-v0':
            raise ValueError('P2 supports original continuous-jerk DDPG only')
        os.chdir(source)
        control = importlib.import_module('control')
        traci = importlib.import_module('traci')
        if traci.isLoaded():
            raise RuntimeError('Existing TraCI connection must not be reused')
        owns_connection = True
        old_delay = control.delay
        control.delay = 0
        import torch
        rng = (random.getstate(), np.random.get_state(), torch.get_rng_state(),
               torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
               torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if settings.CUDA:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        yield settings
    finally:
        try:
            if owns_connection and traci is not None and traci.isLoaded():
                traci.close()
        finally:
            if rng is not None:
                import torch
                random.setstate(rng[0])
                np.random.set_state(rng[1])
                torch.set_rng_state(rng[2])
                if rng[3]:
                    torch.cuda.set_rng_state_all(rng[3])
                torch.backends.cudnn.deterministic = rng[4]
                torch.backends.cudnn.benchmark = rng[5]
            if control is not None and old_delay is not None:
                control.delay = old_delay
            if defaults is not None:
                for key, value in defaults.items():
                    setattr(settings, key, value)
            os.chdir(old_cwd)
            sys.path[:] = old_path
            _SESSION_LOCK.release()


def observation_at_policy_dtype(observation, space):
    result = np.asarray(observation, dtype=space.dtype)
    if result.shape != space.shape or not np.isfinite(result).all():
        raise ValueError('Invalid upstream observation')
    return result.copy()


def validate_action(action, space):
    value = np.asarray(action, dtype=space.dtype)
    if value.shape != space.shape or not np.isfinite(value).all():
        raise ValueError('Jerk action must be a finite one-element vector')
    if np.any(value < space.low) or np.any(value > space.high):
        raise ValueError('Jerk outside original action bounds; no clipping is applied')
    return value.copy()


def termination_reason(raw, done):
    if not done:
        return None
    if raw.crashed:
        return 'upstream_collision'
    if raw.merged:
        return 'upstream_arrival'
    if raw.current_episode_ticks >= raw.max_episode_ticks:
        return 'upstream_time_limit'
    return 'registered_time_limit'


def traffic_snapshot():
    """Read-only full traffic state. Absent ego is represented by absence."""
    import traci
    return {
        'simulation_time_s': float(traci.simulation.getTime()),
        'vehicles': {vehicle: {
            'position': list(traci.vehicle.getPosition(vehicle)),
            'speed': float(traci.vehicle.getSpeed(vehicle)),
            'acceleration': float(traci.vehicle.getAcceleration(vehicle)),
            'lane_id': traci.vehicle.getLaneID(vehicle),
        } for vehicle in sorted(traci.vehicle.getIDList())},
    }


class AuditedJerkEnv:
    """Four-result API with the same float32 policy inputs as original ALL.

    The adapter does not change ContinuousJerkEnv or connect itself to upstream
    DDPG training. Each valid step delegates exactly once. Errors invalidate the
    adapter, rather than returning stale or synthetic observations.
    """
    execution_contract = 'simulation_blocking_exact_v1'
    reset_contract = 'upstream_continuing_traffic_v1'

    def __init__(self):
        if not _SESSION_LOCK.locked():
            raise RuntimeError('Use inside upstream_session')
        import traci
        if traci.isLoaded():
            raise RuntimeError('An environment already owns the default TraCI connection')
        import merge_gym
        from legacy_gym_compat import LegacyTimeLimit
        self.raw = merge_gym.ContinuousJerkEnv({})
        self.env = LegacyTimeLimit(self.raw, 500)
        self.observation_space = self.raw.observation_space
        self.action_space = self.raw.action_space
        self._ready = False
        self._faulted = False
        self._closed = False

    def reset(self):
        if self._closed or self._faulted or self._ready:
            raise RuntimeError('Reset requires an open, healthy, finished environment')
        try:
            obs = observation_at_policy_dtype(self.env.reset(), self.observation_space)
            self._ready = True
            return obs, {'reset_contract': self.reset_contract, 'traffic': traffic_snapshot()}
        except BaseException:
            self._faulted = True
            raise

    def step(self, action):
        if not self._ready or self._closed or self._faulted:
            raise RuntimeError('Call reset on a healthy environment before step')
        try:
            action = validate_action(action, self.action_space)
            import control
            from config import Settings
            before = traffic_snapshot()
            ego = before['vehicles']['ego']
            # Reconstructed from the SAME upstream helper; not a wire-level
            # interception and never mislabeled as a measured command.
            expected_speed = float(control.get_ego_speed_from_jerk(
                ego['speed'], ego['acceleration'], float(action[0])))
            obs, reward, done, info = self.env.step(action)
            obs = observation_at_policy_dtype(obs, self.observation_space)
            if not np.isfinite(reward):
                raise ValueError('Nonfinite upstream reward')
            after = traffic_snapshot()
            after_ego = after['vehicles'].get('ego')
            elapsed = after['simulation_time_s'] - before['simulation_time_s']
            measured_jerk = None
            if after_ego is not None and abs(elapsed - Settings.TICK_LENGTH) < 1e-9:
                measured_jerk = (after_ego['acceleration'] - ego['acceleration']) / Settings.TICK_LENGTH
            audit = {
                'requested_jerk': float(action[0]),
                'commanded_speed_reconstructed': expected_speed,
                'measured_post_step_speed': None if after_ego is None else after_ego['speed'],
                'measured_post_step_acceleration': None if after_ego is None else after_ego['acceleration'],
                'measured_jerk': measured_jerk,
                'upstream_reward_projected_jerk': float(np.asarray(self.raw.projected_jerk).item()),
                'invalid_action_reward': float(self.raw.invalid_action_reward),
                'simulation_elapsed_s': elapsed,
                'termination_reason': termination_reason(self.raw, done),
                'execution_contract': self.execution_contract,
                'before': before, 'after': after,
            }
            self._ready = not done
            return obs, reward, done, {**info, 'execution_audit': audit}
        except BaseException:
            self._faulted = True
            raise

    def close(self):
        if not self._closed:
            self._closed = True
            self._ready = False
            import traci
            if traci.isLoaded():
                self.env.close()
