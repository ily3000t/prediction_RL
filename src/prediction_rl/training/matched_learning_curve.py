"""Streaming, observation-only training instrumentation. No extra RNG draws or updates."""
import gzip
import json
import math

import numpy as np

from prediction_rl.evaluation.training_contract import TransitionWitness, state_measurement
from prediction_rl.features.probe_features import METADATA


class Jsonl:
    def __init__(self, path):
        self.handle = gzip.open(path, 'xt', encoding='utf-8')
    def append(self, row):
        self.handle.write(json.dumps(row, sort_keys=True, allow_nan=False, separators=(',', ':'))+'\n')
    def flush(self): self.handle.flush()
    def close(self): self.handle.close()


class Coverage:
    """Cumulative physical-state moments; minibatch counts include repeated draws."""
    names = ('low_speed', 'early_ramp', 'short_history', 'speed_mps', 'acceleration_mps2',
             'background_vehicle_count', 'history_frames') + METADATA
    def __init__(self):
        self.count = 0; self.total = np.zeros(len(self.names)); self.minimum = np.full(len(self.names), np.inf)
        self.maximum = np.full(len(self.names), -np.inf); self.available = np.zeros(len(self.names), dtype=np.int64)
    @classmethod
    def vector(cls, row):
        e = row['ego']
        if e is None: raise ValueError('Replay current state cannot have absent ego')
        history = row['history_frames']; metadata = row['feature_metadata']
        return np.asarray([float(e['speed'] <= .1), float(e['lane_id'] == 'ramp_0'),
                           np.nan if history == 0 else float(history < 11), e['speed'], e['acceleration'],
                           row['background_vehicle_count'], history] +
                          [np.nan if metadata is None or history == 0 else metadata[k] for k in METADATA], np.float64)
    def add(self, vectors):
        values = np.asarray(vectors, np.float64).reshape(-1, len(self.names))
        valid = ~np.isnan(values)
        if np.isinf(values).any(): raise ValueError('Infinite coverage measurement')
        self.count += len(values); self.available += valid.sum(0)
        self.total += np.where(valid, values, 0).sum(0)
        self.minimum = np.minimum(self.minimum, np.where(valid, values, np.inf).min(0))
        self.maximum = np.maximum(self.maximum, np.where(valid, values, -np.inf).max(0))
    def result(self, scope):
        return {'scope': scope, 'observed_states': self.count,
            'moments': {k: {'count': int(n), 'mean': float(self.total[i]/n) if n else None,
                           'min': float(self.minimum[i]) if n else None,
                           'max': float(self.maximum[i]) if n else None}
                        for i, (k, n) in enumerate(zip(self.names, self.available))}}


class StreamingReplay:
    """Observe ORIGINAL selected tuples at _reshape; never resample or modify State.info."""
    def __init__(self, checked, directory, recorder):
        self.inner = checked.agent.replay_buffer
        self.recorder = recorder; self.witness = TransitionWitness()
        self.transitions = Jsonl(directory/'replay.jsonl.gz'); self.samples = Jsonl(directory/'minibatches.jsonl.gz')
        self.losses = Jsonl(directory/'losses.jsonl.gz')
        self.by_state = {}; self.insertions = 0; self.sample_calls = 0
        self.replay_coverage = Coverage(); self.sample_coverage = Coverage()
        self.previous_measurement = None; self.low_to_moving = 0
        self.last_row = None
        # Scoped to this one original buffer, after original random choice.
        base = self.inner.inner
        if base.capacity <= 60499 or type(base).__name__ != 'ExperienceReplayBuffer':
            raise ValueError('Expected original non-wrapping replay for bounded 60k experiment')
        self.original_reshape = base._reshape
        base._reshape = self.reshape
        checked.agent.replay_buffer = self
        for name in ('q', 'policy'):
            approximation = getattr(checked.agent, name)
            original = approximation.reinforce
            def watched(loss, *, original=original, name=name):
                value = float(loss.detach())
                if not math.isfinite(value): raise ValueError('Nonfinite original optimizer loss')
                self.losses.append({'component': name, 'training_steps': self.insertions,
                                    'sample_call': self.sample_calls, 'value': value})
                return original(loss)
            approximation.reinforce = watched

    def record(self, event, obs, reward, done, info):
        self.witness.record(event, obs, reward, done, info)

    def store(self, state, action, reward, next_state):
        event = self.witness.pending['event']
        measurement = self.witness.states[-1]
        inserted = self.witness.on_store(state, action, reward, next_state)
        if inserted:
            if self.previous_measurement is None: raise ValueError('Missing observed replay state')
            row = self.witness.rows[-1]
            self.insertions += 1
            vector = Coverage.vector(self.previous_measurement)
            self.by_state[id(state)] = (self.insertions, vector)
            self.replay_coverage.add([vector])
            previous, current = self.previous_measurement['ego'], measurement['ego']
            self.low_to_moving += int(current is not None and previous['speed'] <= .1 < current['speed'])
            self.recorder.verify_row(row)
            self.last_row = row
            self.transitions.append({k: row[k] for k in ('episode', 'step', 'state', 'next_state',
                'requested_jerk', 'measured_jerk', 'reward', 'done', 'next_mask', 'reason')} | {
                'transition_id': self.insertions, 'physical_before': self.previous_measurement,
                'physical_after': measurement})
        before = len(self.inner); result = self.inner.store(state, action, reward, next_state)
        if len(self.inner) != before+int(inserted): raise ValueError('Replay insertion mismatch')
        self.previous_measurement = measurement
        # Full traffic witness is bounded to one transition, not kept for 720k steps.
        self.witness.rows.clear(); self.witness.states.clear(); self.witness.resets.clear(); self.witness.stores.clear()
        return result

    def reshape(self, minibatch, weights):
        identities = [self.by_state[id(sample[0])] for sample in minibatch]
        self.sample_calls += 1
        self.sample_coverage.add([v for _, v in identities])
        self.samples.append({'sample_call': self.sample_calls, 'training_steps': self.insertions,
                             'transition_ids': [i for i, _ in identities]})
        return self.original_reshape(minibatch, weights)

    def sample(self, n): return self.inner.sample(n)
    def update_priorities(self, *args): return self.inner.update_priorities(*args)
    def __len__(self): return len(self.inner)
    def __iter__(self): return iter(self.inner)
    def flush(self):
        for s in (self.transitions, self.samples, self.losses): s.flush()
    def close(self):
        for s in (self.transitions, self.samples, self.losses): s.close()
    def summary(self):
        if self.witness.pending is not None: raise ValueError('Incomplete recorder/store pair')
        return {'transitions': self.insertions, 'sample_calls': self.sample_calls,
                'replay_states': self.replay_coverage.result('actual_inserted_current_states'),
                'sampled_states': self.sample_coverage.result('actual_minibatch_draws_with_replacement'),
                'low_speed_to_moving_transitions': self.low_to_moving,
                'sample_indices_saved': True, 'reset_bridge_transitions': 0,
                'measurements_do_not_claim_historical_20k_coverage': True}


def train_nodes(experiment, nodes, snapshot):
    """Same original experiment/agent/replay/RNG across absolute frame targets."""
    for n in nodes:
        experiment.train(frames=n, episodes=np.inf)
        actual = experiment.frame-1
        if not n <= actual <= n+499: raise ValueError('Original episode-boundary budget changed')
        snapshot(n, actual)
