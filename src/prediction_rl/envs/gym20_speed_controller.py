"""Native speed control under the unchanged upstream Gym reset/step clock.

Never convert an ST speed into a clipped jerk. This adapter is evaluation-only
and supports ONLY Slotted Jerk with INVALID_ACTION_PENALTY=0: terminal reward
does not depend on projected jerk; running reward uses measured acceleration.
"""
import math

import numpy as np

from prediction_rl.envs.upstream import (
    observation_at_policy_dtype, traffic_snapshot, termination_reason)


def speed_raw_class(upstream_class, settings):
    if (settings.REWARD_FUNCTION != 'Slotted Jerk' or settings.INVALID_ACTION_PENALTY != 0
            or settings.TICK_LENGTH != .2):
        raise ValueError('Speed adapter cannot infer a jerk-action penalty or another reward')

    class NativeSpeedGym(upstream_class):
        def _do_action(self, token):
            if token is not None or self.controller is None:
                raise ValueError('Only the bound original speed controller may issue a command')
            import prediction
            # Original ST/RL+MPC executes setSpeed itself. No second setSpeed,
            # jerk conversion, clipping, simulation step or planning rewrite.
            speed = float(self.controller(prediction.HighwayState.from_sumo()))
            if not math.isfinite(speed):
                raise ValueError('Nonfinite original commanded speed')
            self.commanded_speed = speed
            self.invalid_action_reward = 0.
            # Not a submitted jerk. The pinned terminal reward ignores it.
            self.projected_jerk = 0.

    return NativeSpeedGym


class Gym20SpeedControllerEnv:
    execution_contract = 'simulation_blocking_exact_v1'

    def __init__(self, settings):
        import merge_gym
        self.raw = speed_raw_class(merge_gym.ContinuousJerkEnv, settings)({})
        self.raw.controller = None
        self.observation_space = self.raw.observation_space
        self.action_space = self.raw.action_space  # Original author network transform only.
        self.ready = self.closed = self.faulted = False
        if self.raw.wait_before_start != 20 or self.raw.max_episode_ticks != 500:
            self.close()
            raise ValueError('Gym20 reset/episode bound changed')

    def bind(self, controller):
        if self.raw.controller is not None or self.ready or not callable(controller):
            raise ValueError('Bind exactly one frozen native controller before reset')
        self.raw.controller = controller

    def reset(self):
        if self.closed or self.ready or self.faulted or self.raw.controller is None:
            raise RuntimeError('Reset needs a healthy bound speed controller')
        try:
            obs = observation_at_policy_dtype(self.raw.reset(), self.observation_space)
            self.ready = True
            return obs, {'traffic': traffic_snapshot()}
        except BaseException:
            self.faulted = True
            raise

    def step(self, token=None):
        if token is not None or not self.ready or self.closed or self.faulted:
            raise RuntimeError('Only one native-controller step on a healthy environment is allowed')
        try:
            before = traffic_snapshot(); ego = before['vehicles']['ego']
            obs, reward, done, info = self.raw.step(None)
            obs = observation_at_policy_dtype(obs, self.observation_space)
            if not np.isfinite(reward):
                raise ValueError('Nonfinite original Gym reward')
            after = traffic_snapshot(); following = after['vehicles'].get('ego')
            elapsed = after['simulation_time_s']-before['simulation_time_s']
            measured = ((following['acceleration']-ego['acceleration'])/.2
                        if following is not None and abs(elapsed-.2) < 1e-9 else None)
            self.ready = not done
            audit = {'before': before, 'after': after, 'requested_jerk': None,
                'commanded_speed_native': self.raw.commanded_speed,
                'measured_jerk': measured, 'invalid_action_reward': self.raw.invalid_action_reward,
                'upstream_reward_projected_jerk': 0., 'simulation_elapsed_s': elapsed,
                'termination_reason': termination_reason(self.raw, done),
                'execution_contract': self.execution_contract,
                'action_contract': 'native_speed_no_jerk_conversion_or_clipping'}
            return obs, reward, done, {**info, 'execution_audit': audit}
        except BaseException:
            self.faulted = True
            raise

    def close(self):
        if not self.closed:
            self.closed = True; self.ready = False
            import traci
            if traci.isLoaded():
                self.raw.close()
