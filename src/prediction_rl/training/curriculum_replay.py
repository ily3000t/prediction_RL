"""Observe one persistent ALL replay; do not replace its sampling or train loop."""
from copy import deepcopy

from prediction_rl.curriculum.interaction_intensity import ExposureAudit
from prediction_rl.evaluation.training_contract import TransitionWitness
from prediction_rl.envs.curriculum_stream import METADATA_KEYS
from prediction_rl.training.all_ddpg import CheckedReplayBuffer


class PersistentTaskReplayAudit:
    """Recorder plus instance-local hook AFTER original uniform random sampling.

    Bind record to make_all_environment, then install once on a freshly created
    agent. Retain the same SingleEnvExperiment/TimeFeature/agent for every task.
    Optional events can be streamed to a run ledger; no full trajectories are
    retained here. IC03b additionally permits its explicit zero-update smoke;
    neither scope authorizes formal training.
    """
    def __init__(self, stream, *, purpose, on_event=None, recent_window=1024):
        from prediction_rl.curriculum.integration_smoke import SCOPE
        if (purpose not in ('offline_integration_test_only', SCOPE) or stream.purpose != purpose
                or (on_event is not None and not callable(on_event))):
            raise ValueError('IC03 is not a live training authorization')
        self.purpose = purpose
        self.stream = stream
        self.on_event = on_event
        self.recent_window = recent_window
        self.witness = TransitionWitness()
        self.inner = self.base = self.audit = None
        self._by_state = {}
        self.insertions = 0
        self.sample_calls = 0
        self._faulted = self._detached = False
        stream.bind_replay_owner(self)

    def install(self, checked):
        core = checked.agent
        inner = core.replay_buffer
        if (self.inner is not None or checked.evaluation_only or type(inner) is not CheckedReplayBuffer
                or len(inner) or core._state is not None or core._frames_seen != 0):
            raise ValueError('Install once before the fresh learner has acted; no weights-only resume')
        from prediction_rl.curriculum.integration_smoke import SCOPE
        if self.purpose == SCOPE:
            from prediction_rl.training.all_ddpg import resolved_preset
            if checked.preset_parameters != resolved_preset('cpu', .0002, smoke=False):
                raise ValueError('Real smoke must retain original warmup, batch and optimizer preset')
        base = inner.inner
        if type(base).__name__ != 'ExperienceReplayBuffer' or base.pos != 0:
            raise ValueError('Require the unchanged original uniform replay')
        self.inner, self.base = inner, base
        self.audit = ExposureAudit(capacity=base.capacity, recent_window=self.recent_window)
        self._original_reshape = base._reshape
        self._reshape_hook = self.reshape
        base._reshape = self._reshape_hook  # This ONE buffer, not the installed class.
        core.replay_buffer = self
        self._core = core
        return self

    def _healthy(self):
        if self.inner is None or self._faulted or self._detached:
            raise RuntimeError('Replay observer is uninstalled, detached or invalid')

    def _emit(self, kind, value):
        if self.on_event is not None:
            self.on_event(kind, deepcopy(value))

    def record(self, event, obs, reward, done, info):
        self._healthy()
        try:
            if METADATA_KEYS.intersection(info):
                raise ValueError('Curriculum labels leaked into environment info')
            if event == 'reset':
                self.audit.begin_episode(self.stream.selection)
            self.witness.record(event, obs, reward, done, info)
        except BaseException:
            self._faulted = True
            raise

    def store(self, state, action, reward, next_state):
        self._healthy()
        try:
            for item in (state, next_state):
                # Original ALL reset State.info contains None, not an empty dict.
                if item is not None and any(info is not None and METADATA_KEYS.intersection(info)
                                            for info in item.info):
                    raise ValueError('Curriculum labels leaked into replay State.info')
            inserted = self.witness.on_store(state, action, reward, next_state)
            before, slot = len(self.inner), self.base.pos
            old = self.base.buffer[slot] if before == self.base.capacity else None
            result = self.inner.store(state, action, reward, next_state)  # Exactly once.
            if (len(self.inner) != min(before+int(inserted), self.base.capacity)
                    or self.base.pos != (slot+int(inserted)) % self.base.capacity):
                raise ValueError('Original replay insertion/position differs from observed control')
            if inserted:
                actual = self.base.buffer[slot]
                if actual[0] is not state or actual[3] is not next_state:
                    raise ValueError('Stored replay tuple does not match the real transition')
                if old is not None:
                    del self._by_state[id(old[0])]
                self._by_state[id(state)] = slot
                self.insertions += 1
                metadata = self.audit.record_transition(slot, self.stream.actual_control_steps)
                row = self.witness.rows[-1]
                self._emit('transition', {'metadata': metadata, 'transition': row})
                if row['done']:
                    if row['reason'] not in ('upstream_arrival', 'upstream_collision', 'upstream_time_limit'):
                        raise ValueError('Unexplained terminal event cannot complete a task')
                    self.audit.end_episode(row['reason'])
                    completion = self.stream.acknowledge_terminal(self)
                    self._emit('episode', {**completion, 'outcome': row['reason']})
            # Retain only the previous tensor and pending event, not full traffic.
            for values in (self.witness.rows, self.witness.states, self.witness.resets, self.witness.stores):
                values.clear()
            return result
        except BaseException:
            self._faulted = True
            raise

    def reshape(self, minibatch, weights):
        self._healthy()
        try:
            indices = [self._by_state[id(sample[0])] for sample in minibatch]
            if any(self.base.buffer[i] is not sample for i, sample in zip(indices, minibatch)):
                raise ValueError('Observed minibatch is not from the current original replay slots')
            metadata = self.audit.record_minibatch(indices)
            self.sample_calls += 1
            self._emit('minibatch', {'sample_call': self.sample_calls,
                'actual_control_steps': self.insertions, 'original_replay_indices': indices,
                'transition_metadata': metadata})
            return self._original_reshape(minibatch, weights)
        except BaseException:
            self._faulted = True
            raise

    def sample(self, n):
        self._healthy()
        return self.inner.sample(n)  # No audit RNG draws or extra sample call.

    def update_priorities(self, *args):
        return self.inner.update_priorities(*args)

    def __len__(self): return len(self.inner)
    def __iter__(self): return iter(self.inner)

    def snapshot(self):
        self._healthy()
        if self.witness.pending is not None or self.audit._steps != self.insertions:
            raise ValueError('Unconsumed environment event or incomplete actual replay audit')
        return {**self.audit.snapshot(), 'original_replay_size': len(self.inner),
            'original_replay_position': self.base.pos, 'reset_bridge_transitions': 0,
            'terminal_transition_acknowledged_before_reset': True,
            'original_train_loop_retained': True, 'audit_scope': self.purpose,
            'bounded_smoke_execution_authorized':self.purpose == 'ic03b_approved_bounded_integration_smoke_v1',
            'formal_training_authorized': False}

    def detach(self):
        """Restore only this buffer's hook; learner/replay/optimizer are not reset."""
        self._healthy()
        if self.witness.pending is not None or self.audit._active is not None:
            raise ValueError('Do not detach in the middle of a task or recorder/store pair')
        if self.base._reshape is not self._reshape_hook or self._core.replay_buffer is not self:
            raise ValueError('Replay hook ownership changed')
        self.base._reshape = self._original_reshape
        self._core.replay_buffer = self.inner
        self._detached = True
