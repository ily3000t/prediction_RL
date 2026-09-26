"""Keep the upstream Gym 0.23 four-result contract on newer Gym.

Only the local SUMO environments use this adapter. No site-packages monkey
patches, reward changes, action conversions or additional safety checks.
Time-limit semantics follow Gym 0.23.1 (the upstream pinned requirement).
"""
from copy import deepcopy

import gym
from gym.envs.registration import load


class LegacyTimeLimit(gym.Wrapper):
    def __init__(self, env, max_episode_steps):
        super().__init__(env)
        if not isinstance(max_episode_steps, int) or max_episode_steps <= 0:
            raise ValueError('Expected a positive registered episode step limit')
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = None

    def reset(self, **kwargs):
        self._elapsed_steps = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        if self._elapsed_steps is None:
            raise RuntimeError('Call reset before step')
        transition = self.env.step(action)
        if not isinstance(transition, tuple) or len(transition) != 4:
            raise TypeError('Upstream SUMO step must return four values')
        obs, reward, done, info = transition
        self._elapsed_steps += 1
        if self._elapsed_steps >= self._max_episode_steps:
            info = dict(info)
            info['TimeLimit.truncated'] = not done
            done = True
        return obs, reward, done, info


def make_legacy_gym_environment(env_id, device):
    """Instantiate the registered upstream constructor with a legacy TimeLimit.

    Bypass only modern Gym's API checker/TimeLimit, which assume five outputs.
    Keep the registered kwargs, observation/action spaces and 500-step limit.
    """
    if env_id not in ('sumo-jerk-continuous-v0', 'sumo-jerk-v0', 'sumo-accel-v0'):
        raise ValueError('Compatibility adapter is restricted to upstream SUMO environments')
    spec = deepcopy(gym.spec(env_id))
    constructor = load(spec.entry_point) if isinstance(spec.entry_point, str) else spec.entry_point
    if constructor.__module__ != 'merge_gym':
        raise ValueError('Unexpected registered environment constructor')
    raw = constructor(**deepcopy(spec.kwargs))
    try:
        raw.spec = spec
        legacy = LegacyTimeLimit(raw, spec.max_episode_steps)
        from all.environments import GymEnvironment
        wrapped = GymEnvironment(legacy, device=device)
        wrapped._name = env_id
        return wrapped
    except BaseException:
        raw.close()
        raise
