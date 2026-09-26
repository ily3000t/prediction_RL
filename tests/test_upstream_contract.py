import sys
from pathlib import Path
from types import SimpleNamespace

import gym
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from prediction_rl.envs.upstream import (
    AuditedJerkEnv, observation_at_policy_dtype, termination_reason,
    upstream_session, validate_action, validate_seed, validate_settings,
)

SPACE = gym.spaces.Box(-5, 5, (1,), dtype=np.float32)


@pytest.mark.parametrize('seed', [-1, 2**32, True, 1.5, '0'])
def test_invalid_seed(seed):
    with pytest.raises(ValueError):
        validate_seed(seed)


@pytest.mark.parametrize('action', [0, [1, 2], [float('nan')], [float('inf')], [5.001], [-5.001]])
def test_invalid_actions_are_not_clipped(action):
    with pytest.raises(ValueError):
        validate_action(action, SPACE)


@pytest.mark.parametrize('action', [[-5], [0], [5]])
def test_valid_actions_keep_original_policy_dtype(action):
    result = validate_action(action, SPACE)
    assert result.dtype == np.float32
    np.testing.assert_array_equal(result, action)


def test_observation_uses_original_all_cast_and_copy():
    raw = np.array([0.123456789], dtype=np.float64)
    result = observation_at_policy_dtype(raw, SPACE)
    np.testing.assert_array_equal(result, raw.astype(np.float32))
    assert not np.shares_memory(raw, result)


@pytest.mark.parametrize('obs', [[float('nan')], [0, 0]])
def test_invalid_observation(obs):
    with pytest.raises(ValueError):
        observation_at_policy_dtype(obs, SPACE)


def test_unknown_settings_are_not_ignored():
    with pytest.raises(ValueError, match='Unknown'):
        validate_settings({'TICK_LENGHT': 0.2}, {'TICK_LENGTH': 0.2})


@pytest.mark.parametrize('payload', [{'CUDA': 'false'}, {'TICK_LENGTH': True}, {'TICK_LENGTH': float('inf')}])
def test_invalid_setting_types(payload):
    with pytest.raises(ValueError):
        validate_settings(payload, {'CUDA': True, 'TICK_LENGTH': 0.2})


@pytest.mark.parametrize('crashed,merged,ticks,expected', [
    (True, False, 1, 'upstream_collision'),
    (False, True, 500, 'upstream_arrival'),
    (False, False, 500, 'upstream_time_limit'),
    (False, False, 2, 'registered_time_limit'),
])
def test_end_reason_respects_upstream_precedence(crashed, merged, ticks, expected):
    raw = SimpleNamespace(crashed=crashed, merged=merged, current_episode_ticks=ticks, max_episode_ticks=500)
    assert termination_reason(raw, True) == expected
    assert termination_reason(raw, False) is None


def test_invalid_step_faults_adapter_and_cannot_be_retried():
    env = object.__new__(AuditedJerkEnv)
    env._ready, env._closed, env._faulted = True, False, False
    env.action_space = SPACE
    with pytest.raises(ValueError):
        env.step([float('nan')])
    assert env._faulted
    with pytest.raises(RuntimeError):
        env.step([0])
    with pytest.raises(RuntimeError):
        env.reset()


def test_mid_episode_reset_is_rejected():
    env = object.__new__(AuditedJerkEnv)
    env._ready, env._closed, env._faulted = True, False, False
    with pytest.raises(RuntimeError):
        env.reset()


def test_failed_session_restores_process_and_releases_lock(monkeypatch):
    import os
    root = Path(__file__).resolve().parents[1]
    before_cwd, before_path = os.getcwd(), list(sys.path)
    monkeypatch.setitem(sys.modules, 'config', SimpleNamespace(__file__=str(root / 'foreign/config.py')))
    # The second attempt must fail for the same module conflict, not a stuck lock.
    for _ in range(2):
        with pytest.raises(RuntimeError, match='Foreign module'):
            with upstream_session(root / 'RL-MPC-LaneMerging-master',
                                  root / 'RL-MPC-LaneMerging-master/configs/train_default_1.json', 0):
                pytest.fail('Foreign module must be rejected')
        assert os.getcwd() == before_cwd
        assert sys.path == before_path
