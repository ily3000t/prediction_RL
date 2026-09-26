import sys
from pathlib import Path

import gym
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'RL-MPC-LaneMerging-master'))
from legacy_gym_compat import LegacyTimeLimit, make_legacy_gym_environment


class FakeEnv(gym.Env):
    observation_space = gym.spaces.Box(-1, 1, (1,), dtype=np.float32)
    action_space = gym.spaces.Box(-5, 5, (1,), dtype=np.float32)

    def __init__(self, done_at=999):
        self.done_at = done_at

    def reset(self):
        self.t = 0
        return np.zeros(1, dtype=np.float32)

    def step(self, action):
        self.last_action = action
        self.t += 1
        return np.array([self.t], dtype=np.float32), float(action[0]), self.t == self.done_at, {'original': True}


@pytest.mark.parametrize('done_at', [2, 3, 999])
def test_legacy_limit_matches_original_done_and_reward(done_at):
    raw = FakeEnv(done_at)
    env = LegacyTimeLimit(raw, 3)
    np.testing.assert_array_equal(env.reset(), [0])
    for tick in range(1, 4):
        action = np.array([-1.75], dtype=np.float32)
        obs, reward, done, info = env.step(action)
        assert raw.last_action is action
        assert reward == -1.75
        np.testing.assert_array_equal(obs, [tick])
        assert done == (tick == done_at or tick >= 3)
        assert info['original'] is True
        if tick >= 3:
            assert info['TimeLimit.truncated'] == (tick != done_at)
        else:
            assert 'TimeLimit.truncated' not in info
        if done:
            break


def test_reset_clears_time_limit():
    env = LegacyTimeLimit(FakeEnv(), 1)
    env.reset()
    assert env.step([0])[2]
    env.reset()
    assert env._elapsed_steps == 0


def test_step_before_reset_rejected():
    with pytest.raises(RuntimeError):
        LegacyTimeLimit(FakeEnv(), 3).step([0])


@pytest.mark.parametrize('limit', [0, -1, None])
def test_invalid_limit_rejected(limit):
    with pytest.raises(ValueError):
        LegacyTimeLimit(FakeEnv(), limit)


def test_five_result_env_rejected():
    raw = FakeEnv()
    raw.step = lambda action: (None, 0, False, False, {})
    env = LegacyTimeLimit(raw, 1)
    env.reset()
    with pytest.raises(TypeError):
        env.step([0])


def test_other_environments_not_affected():
    with pytest.raises(ValueError):
        make_legacy_gym_environment('CartPole-v1', 'cpu')
    assert not hasattr(np, 'bool8')
