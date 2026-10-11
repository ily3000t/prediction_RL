"""Episode-boundary task transport core, not a training authorization or CLI.

The injected factory has the FreshGym20TaskEnv interface. IC03b adds only the
explicitly approved two-episode smoke scope; formal training is not authorized.
"""
from copy import deepcopy

import numpy as np

from prediction_rl.curriculum.interaction_intensity import OfflineEpisodeSampler
from prediction_rl.envs.fresh_task import ENTRY_CONTRACT, EXECUTION_CONTRACT


METADATA_KEYS = frozenset(('task_id', 'simulator_seed', 'level', 'curriculum_level',
    'collection_stage', 'actual_environment_step', 'episode_index', 'sampling_mass'))


def same_space(left, right):
    return (left.shape == right.shape and left.dtype == right.dtype
            and np.array_equal(left.low, right.low) and np.array_equal(left.high, right.high))


class CurriculumTaskStream:
    """Keep learner-facing spaces stable while replacing terminal task workers.

    No worker is constructed before reset. A terminal task remains owned until
    the ORIGINAL replay store has acknowledged its last transition. All task
    metadata stays on this object, not in native observations or info.
    """
    execution_contract = EXECUTION_CONTRACT
    reset_contract = ENTRY_CONTRACT

    def __init__(self, sampler, worker_factory, observation_space, action_space, *, purpose):
        from prediction_rl.curriculum.integration_smoke import SCOPE, SmokeEpisodeSampler
        permitted = ((purpose == 'offline_integration_test_only' and type(sampler) is OfflineEpisodeSampler)
                     or (purpose == SCOPE and type(sampler) is SmokeEpisodeSampler))
        if not permitted or not callable(worker_factory):
            raise ValueError('Require an offline or approved bounded-smoke binding; no matched training')
        self.purpose = purpose
        if (observation_space.shape != (20,) or observation_space.dtype != np.float32
                or action_space.shape != (1,) or action_space.dtype != np.float32
                or not np.array_equal(action_space.low, [-5.])
                or not np.array_equal(action_space.high, [5.])):
            raise ValueError('Require the original raw observation and continuous jerk spaces')
        self.observation_space = deepcopy(observation_space)
        self.action_space = deepcopy(action_space)
        self._sampler, self._factory = sampler, worker_factory
        self._worker = self._owner = self._selection = None
        self._terminal = self._acknowledged = self._faulted = self._closed = False
        self.actual_control_steps = self.episode_control_steps = 0
        self.last_completion = None

    @property
    def selection(self):
        if self._selection is None:
            raise RuntimeError('No current task selection')
        return deepcopy(self._selection)

    def bind_replay_owner(self, owner):
        if self._owner is not None or self._worker is not None or owner is None:
            raise RuntimeError('Bind the single persistent replay audit before the first task')
        self._owner = owner

    def _info(self, info):
        if not isinstance(info, dict) or METADATA_KEYS.intersection(info):
            raise ValueError('Task labels must not enter native info or replay State.info')

    def reset(self):
        if (self._closed or self._faulted or self._owner is None
                or (self._worker is not None and not (self._terminal and self._acknowledged))):
            raise RuntimeError('Reset requires a healthy boundary and stored terminal transition')
        try:
            if self._worker is not None:
                self._worker.close()
                self._worker = None
            self._selection = self._sampler.begin_episode()
            if self._selection['actual_start_step'] != self.actual_control_steps:
                raise ValueError('Sampler clock differs from actual control transitions')
            self._worker = self._factory(deepcopy(self._selection))
            if (self._worker.reset_contract != ENTRY_CONTRACT
                    or self._worker.execution_contract != EXECUTION_CONTRACT
                    or not same_space(self.observation_space, self._worker.observation_space)
                    or not same_space(self.action_space, self._worker.action_space)):
                raise ValueError('Task worker changed the approved native entry or spaces')
            self.episode_control_steps = 0
            self._terminal = self._acknowledged = False
            result = self._worker.reset()
            self._info(result[1])
            return result
        except BaseException:
            self._faulted = True
            raise

    def step(self, action):
        if self._closed or self._faulted or self._worker is None or self._terminal:
            raise RuntimeError('No control step before reset or after terminal/failure')
        try:
            result = self._worker.step(action)
            if len(result) != 4 or type(result[2]) is not bool:
                raise ValueError('Native reward/done interface changed')
            self._info(result[3])
            self.episode_control_steps += 1
            self.actual_control_steps += 1
            self._terminal = result[2]
            return result
        except BaseException:
            self._faulted = True
            raise

    def acknowledge_terminal(self, owner):
        if (owner is not self._owner or not self._terminal or self._acknowledged
                or self._faulted or self._closed):
            raise RuntimeError('Only the bound replay observer can acknowledge a terminal store')
        completion = self._sampler.end_episode(self.episode_control_steps)
        if completion['actual_end_step'] != self.actual_control_steps:
            self._faulted = True
            raise ValueError('Completed task control clock mismatch')
        self.last_completion = deepcopy(completion)
        self._acknowledged = True
        return completion

    def current_frame(self):
        if self._closed or self._faulted or self._worker is None:
            raise RuntimeError('Current frame requires a healthy task worker')
        return self._worker.current_frame()

    def close(self):
        if self._closed:
            return
        self._closed = True
        if self._worker is not None:
            self._worker.close()
