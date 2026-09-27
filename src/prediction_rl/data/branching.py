"""Exact replay-backed roots and an experimental native SUMO snapshot probe.

Replay is deliberately the trusted development backend. Native loadState must
pass continuation parity before it can be used as a dataset backend.
"""
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random

import numpy as np

from prediction_rl.envs.upstream import AuditedJerkEnv, traffic_snapshot


def plain(value):
    if isinstance(value, np.ndarray):
        return {'dtype': str(value.dtype), 'shape': list(value.shape), 'values': value.tolist()}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (tuple, list)):
        return [plain(x) for x in value]
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if value.__class__.__name__ == 'HighwayState':
        return plain(vars(value))
    raise TypeError(f'Unexpected snapshot value: {type(value)}')


def fingerprint(value):
    return hashlib.sha256(json.dumps(plain(value), sort_keys=True, allow_nan=False).encode()).hexdigest()


def capture_rng():
    import torch
    return (deepcopy(random.getstate()), deepcopy(np.random.get_state()),
            torch.get_rng_state().clone(),
            [s.clone() for s in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else [])


def restore_rng(state):
    import torch
    random.setstate(deepcopy(state[0]))
    np.random.set_state(deepcopy(state[1]))
    torch.set_rng_state(state[2].clone())
    if state[3]:
        torch.cuda.set_rng_state_all([s.clone() for s in state[3]])


def rng_fingerprint():
    state = capture_rng()
    return fingerprint((state[0], state[1], state[2].tolist(), [x.tolist() for x in state[3]]))


def root_signature(env):
    import control
    from config import Settings
    raw = {k: v for k, v in vars(env.raw).items()
           if k not in ('start_time', 'reward_function', 'action_space', 'observation_space')}
    return fingerprint({
        'traffic': traffic_snapshot(), 'raw': raw, 'delay': control.delay,
        'limit_steps': env.env._elapsed_steps, 'ready': env._ready,
        'faulted': env._faulted, 'settings': Settings.export_settings(),
        'rng': rng_fingerprint(), 'reward_function': env.raw.reward_function.__qualname__,
    })


def rollout(env, actions):
    result = []
    for jerk in actions:
        obs, reward, done, info = env.step(np.array([jerk], dtype=env.action_space.dtype))
        result.append({'observation': obs.tolist(), 'reward': float(reward),
                       'done': bool(done), 'info': plain(info)})
        if done:
            break
    return result


class RootUnavailable(RuntimeError):
    """A predeclared root is beyond episode termination; do not replace it."""


@dataclass(frozen=True)
class ReplayRoot:
    prefix: tuple
    signature: str
    prefix_trace_hash: str
    traffic: dict
    owner: object


class ReplayBrancher:
    """Fresh SUMO plus full action-prefix replay; no approximate restoration."""
    backend = 'fresh_sumo_prefix_replay_v1'

    def __init__(self):
        import control
        import traci
        import merge_gym  # Complete upstream imports before freezing RNG state.
        from config import Settings
        if traci.isLoaded():
            raise RuntimeError('Construct replay manager before opening an environment')
        self.initial_rng = capture_rng()
        self.initial_delay = control.delay
        self.settings_hash = fingerprint(Settings.export_settings())
        self.owner = object()
        self.env = None

    def _fresh(self):
        from config import Settings
        import control
        if fingerprint(Settings.export_settings()) != self.settings_hash:
            raise RuntimeError('Configuration changed after snapshot manager initialization')
        self.close()
        restore_rng(self.initial_rng)
        control.delay = self.initial_delay
        self.env = AuditedJerkEnv()
        obs, _ = self.env.reset()
        return obs

    def _replay(self, prefix):
        observation = self._fresh()
        trace = rollout(self.env, prefix)
        if trace and trace[-1]['done']:
            raise RootUnavailable('Predeclared root reaches/passes episode termination')
        if len(trace) != len(prefix):
            raise RootUnavailable('Incomplete root action prefix')
        return observation, trace

    def capture(self, prefix):
        prefix = tuple(float(x) for x in prefix)
        _, trace = self._replay(prefix)
        return ReplayRoot(prefix, root_signature(self.env), fingerprint(trace),
                          deepcopy(traffic_snapshot()), self.owner)

    def discover_lane_roots(self, targets, max_steps, jerk, policy=None):
        from prediction_rl.data.geometric_roots import scan_lane_roots, target_position, validate_targets
        import traci
        validate_targets(targets)
        observation = self._fresh()
        for target in targets:
            target_position(target, traci.lane.getLength(target['lane_id']))
        if policy is not None:
            policy.bind(self.env)
        return scan_lane_roots(self.env, targets, max_steps, jerk,
                               action_source=policy, initial_observation=observation)

    def restore(self, root):
        if root.owner is not self.owner:
            raise ValueError('Snapshot belongs to a different run/manager')
        _, trace = self._replay(root.prefix)
        if fingerprint(trace) != root.prefix_trace_hash or root_signature(self.env) != root.signature:
            self.close()
            raise AssertionError('Root replay differs; refuse contaminated counterfactual labels')
        return self.env

    def branch(self, root, actions):
        return rollout(self.restore(root), actions)

    def close(self):
        if self.env is not None:
            self.env.close()
            self.env = None


def native_snapshot_probe(output, prefix, actions):
    """Isolated experiment, never selected automatically for formal data.

    Additional SUMO flags only enable RNG serialization and 17-digit precision.
    A fresh replay root must also match this probe's root outside this function.
    """
    import control
    import traci
    from config import Settings
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    env = AuditedJerkEnv()
    try:
        # No ticks have elapsed. Reload original inputs with save-only options.
        args = ['-c', 'ramp.sumocfg', '--step-length', str(Settings.TICK_LENGTH),
                '--seed', str(Settings.SEED), '--save-state.rng', 'true',
                '--save-state.precision', '17']
        if Settings.USE_ALTERNATE_TRAFFIC_DISTRIBUTION:
            route = {'low': 'merge2.rou.xml', 'medium': 'merge2b.rou.xml', 'high': 'merge2c.rou.xml'}[Settings.TRAFFIC_DENSITY]
            args.extend(['--route-files', route])
        elif Settings.USE_SIMPLE_TRAFFIC_DISTRIBUTION:
            args.extend(['--route-files', 'merge_impossible.rou.xml'])
        traci.load(args)
        if Settings.USE_SIMPLE_TRAFFIC_DISTRIBUTION:
            traci.vehicletype.setMaxSpeed('normal', Settings.OTHER_CAR_SPEED)
        env.reset()
        prefix_trace = rollout(env, prefix)
        if prefix_trace and prefix_trace[-1]['done']:
            raise RootUnavailable('Native probe root is unavailable')
        root_before = traffic_snapshot()
        state = output / 'sumo.xml'
        traci.simulation.saveState(str(state))
        saved_raw = deepcopy(vars(env.raw))
        saved_limit, saved_delay = env.env._elapsed_steps, control.delay
        saved_rng = capture_rng()
        before_signature = root_signature(env)
        direct = rollout(env, actions)
        traci.simulation.loadState(str(state))
        vars(env.raw).clear()
        vars(env.raw).update(deepcopy(saved_raw))
        env.env._elapsed_steps = saved_limit
        control.delay = saved_delay
        restore_rng(saved_rng)
        env._ready, env._faulted = True, False
        root_after = traffic_snapshot()
        after_signature = root_signature(env)
        restored = rollout(env, actions)
        import xml.etree.ElementTree as ET
        xml = ET.parse(state)
        rng_tags = [node.tag for node in xml.iter() if 'rng' in node.tag.lower()]
        report = {
            'sumo_load_options': args, 'rng_tags': rng_tags,
            'root_before': root_before, 'root_after': root_after,
            'direct': direct, 'restored': restored,
            'root_exact': before_signature == after_signature,
            'continuation_exact': direct == restored,
            'state_sha256': hashlib.sha256(state.read_bytes()).hexdigest(),
        }
        report['native_eligible'] = bool(rng_tags) and report['root_exact'] and report['continuation_exact']
        return report
    finally:
        env.close()


def response_summary(branches, root_ids, speed_threshold, position_threshold):
    """Absolute same-actor, same-time motion changes; NOT ego-relative gaps."""
    from itertools import combinations
    summary = []
    for left, right in combinations(branches, 2):
        max_neighbor_position = max_neighbor_speed = max_ego_position = 0.0
        paired_neighbors = comparable_steps = membership_mismatches = 0
        for a, b in zip(branches[left], branches[right]):
            ta = a['info']['execution_audit']['after']
            tb = b['info']['execution_audit']['after']
            if ta['simulation_time_s'] != tb['simulation_time_s']:
                raise AssertionError('Counterfactual times are not aligned')
            comparable_steps += 1
            va, vb = ta['vehicles'], tb['vehicles']
            membership_mismatches += len((set(va) ^ set(vb)) & set(root_ids))
            for actor in set(va) & set(vb):
                distance = float(np.linalg.norm(np.array(va[actor]['position']) - vb[actor]['position']))
                if actor == 'ego':
                    max_ego_position = max(max_ego_position, distance)
                elif actor in root_ids:
                    paired_neighbors += 1
                    max_neighbor_position = max(max_neighbor_position, distance)
                    max_neighbor_speed = max(max_neighbor_speed, abs(va[actor]['speed'] - vb[actor]['speed']))
        summary.append({
            'left': left, 'right': right, 'comparable_steps': comparable_steps,
            'paired_neighbor_observations': paired_neighbors,
            'root_actor_membership_mismatches': membership_mismatches,
            'max_ego_position_delta_m': max_ego_position,
            'max_neighbor_position_delta_m': max_neighbor_position,
            'max_neighbor_speed_delta_mps': max_neighbor_speed,
            'neighbor_response_detected': paired_neighbors > 0 and
                (max_neighbor_position >= position_threshold or max_neighbor_speed >= speed_threshold),
            'left_observed_steps': len(branches[left]), 'right_observed_steps': len(branches[right]),
        })
    return summary
