"""P7j read-only witnesses around original ALL storage and TD loss.

These hooks do not choose actions, draw random numbers, or change transitions.
Requested jerk is the MDP action; measured jerk is an environment response.
"""
from copy import deepcopy
import math
import statistics

import numpy as np
import torch
from torch.nn.functional import mse_loss

from prediction_rl.data.dataset_contract import digest
from prediction_rl.features.probe_features import METADATA

VERSION = 'p07j_training_contract_v1'
ARMS = ('baseline', 'zero', 'ordinary', 'conditional')


def summary(values):
    values = list(values)
    if any(not math.isfinite(float(x)) for x in values):
        raise ValueError('Nonfinite descriptive measurement')
    return {'count': len(values), 'mean': statistics.mean(values) if values else None,
            'min': min(values) if values else None, 'max': max(values) if values else None}


def state_measurement(obs, info):
    frame = info['traffic'] if 'traffic' in info else info['execution_audit']['after']
    ego = frame['vehicles'].get('ego')
    metadata = None if len(obs) == 20 else dict(zip(METADATA, map(float, obs[-len(METADATA):])))
    nearest = None if ego is None else min((math.dist(ego['position'], v['position'])
        for k,v in frame['vehicles'].items() if k != 'ego'), default=None)
    return {'simulation_time_s': frame['simulation_time_s'], 'traffic_sha256': digest(frame),
        'background_vehicle_count': len(frame['vehicles']) - int(ego is not None),
        'ego': deepcopy(ego), 'history_frames': info['prediction_features']['history_frames'],
        'feature_metadata': metadata, 'nearest_observed_center_distance_m': nearest,
        'observation_sha256': digest(obs.tolist())}


def coverage(rows):
    live = [r for r in rows if r['ego'] is not None]
    low = [r for r in live if r['ego']['speed'] <= .1]
    pairs = [(a,b) for a,b in zip(rows,rows[1:]) if a['ego'] is not None and b['ego'] is not None
             and b.get('event') != 'reset' and a.get('episode') == b.get('episode')]
    return {'observed_states': len(rows), 'live_ego_states': len(live),
        'low_speed_fraction': len(low)/len(live) if live else None,
        'early_ramp_fraction': statistics.mean(r['ego']['lane_id'] == 'ramp_0' for r in live) if live else None,
        'short_history_fraction': statistics.mean(r['history_frames'] < 11 for r in live)
            if live and all(r['history_frames'] > 0 for r in live) else None,
        'speed_mps': summary(r['ego']['speed'] for r in live),
        'acceleration_mps2': summary(r['ego']['acceleration'] for r in live),
        'background_vehicle_count': summary(r['background_vehicle_count'] for r in live),
        'history_frames': summary(r['history_frames'] for r in live),
        'low_speed_to_moving_transitions': sum(a['ego']['speed'] <= .1 < b['ego']['speed'] for a,b in pairs),
        'low_speed_positive_acceleration_states': sum(r['ego']['acceleration'] > 0 for r in low),
        'nearest_observed_center_distance_m': summary(r['nearest_observed_center_distance_m'] for r in live
            if r['nearest_observed_center_distance_m'] is not None),
        'feature_metadata': {k: summary(r['feature_metadata'][k] for r in live if r['feature_metadata'] is not None)
                             for k in METADATA},
        'scope': 'new_bounded_trajectory_not_historical_20k_replay'}


class TransitionWitness:
    """Pair recorder events with actual ALL store calls, including episode resets."""
    def __init__(self):
        self.pending = None
        self.previous = None
        self.episode = -1
        self.tick = 0
        self.rows = []
        self.resets = []
        self.stores = []
        self.states = []

    def record(self, event, obs, reward, done, info):
        if self.pending is not None or event not in ('reset', 'step'):
            raise ValueError('Recorder/store events not one-to-one')
        if event == 'reset':
            self.episode += 1; self.tick = 0
            self.resets.append(state_measurement(obs, info))
        else:
            self.tick += 1
        self.pending = {'event': event, 'observation': np.array(obs, copy=True),
                        'reward': 0. if reward is None else float(reward), 'done': bool(done),
                        'info': deepcopy(info), 'episode': self.episode, 'tick': self.tick}
        self.states.append({**state_measurement(obs, info), 'event': event, 'episode': self.episode})

    def on_store(self, state, action, reward, next_state):
        e = self.pending
        if e is None:
            raise ValueError('Store without an observed reset/step')
        expected = torch.tensor(np.r_[e['observation'], np.float32(.001*e['tick'])], dtype=torch.float32).view(1, -1)
        # TimeFeature multiplies the integer counter in float32, not in float64.
        expected[0, -1] = torch.tensor(float(e['tick']), dtype=torch.float32) * .001
        if (not torch.equal(next_state.features.cpu(), expected) or
                bool(next_state.done) != e['done'] or float(reward) != e['reward']):
            raise ValueError('next-state/reward/time/mask alignment mismatch')
        inserted = state is not None and not bool(state.done)
        if (e['event'] == 'step') != inserted:
            raise ValueError('Terminal transition lost or reset bridge inserted')
        if inserted:
            if (self.previous is None or not torch.equal(state.features.cpu(), self.previous) or
                    float(action.item()) != e['info']['execution_audit']['requested_jerk']):
                raise ValueError('Previous observation or submitted command mismatches replay')
            a = e['info']['execution_audit']
            self.rows.append({'episode': e['episode'], 'step': e['tick']-1,
                'state': state.features.cpu().tolist()[0], 'next_state': next_state.features.cpu().tolist()[0],
                'requested_jerk': float(action.item()), 'measured_jerk': a['measured_jerk'],
                'reward': e['reward'], 'done': e['done'], 'next_mask': int(next_state.mask.item()),
                'reason': a['termination_reason'], 'execution_audit': a})
        self.stores.append({'event': e['event'], 'inserted': inserted, 'done': e['done'],
                            'episode': e['episode'], 'tick': e['tick']})
        self.previous = next_state.features.cpu().clone(); self.pending = None
        return inserted

    def finish(self):
        if self.pending is not None or not self.rows or not self.rows[-1]['done']:
            raise ValueError('Incomplete witness')
        return {'transitions': len(self.rows), 'reset_calls': len(self.resets),
                'terminal_transitions': sum(r['done'] for r in self.rows),
                'reset_bridge_transitions': 0, 'aligned': True}


class ReplayWitness:
    def __init__(self, inner, witness, core):
        self.inner, self.witness, self.core = inner, witness, core
        self.samples = []; self.losses = []; self.expected_loss = None

    def store(self, state, action, reward, next_state):
        inserted = self.witness.on_store(state, action, reward, next_state)
        before = len(self.inner)
        result = self.inner.store(state, action, reward, next_state)
        if len(self.inner) != before + int(inserted):
            raise ValueError('Actual replay insertion disagrees with observed transition')
        return result

    def sample(self, n):
        batch = self.inner.sample(n)  # Exactly one original random sample.
        states, actions, rewards, following, _ = batch
        with torch.no_grad():
            boot = self.core.q.target(following, self.core.policy.target(following))
            targets = rewards + self.core.discount_factor * boot
            self.expected_loss = mse_loss(self.core.q.eval(states, actions), targets)
        terminal = following.mask == 0
        if torch.any(boot[terminal] != 0) or not torch.equal(targets[terminal], rewards[terminal]):
            raise ValueError('Terminal TD target bootstraps')
        self.samples.append({'size': n, 'terminal_samples': int(terminal.sum()),
            'target_min': float(targets.min()), 'target_max': float(targets.max())})
        return batch

    def reinforce(self, loss):
        if self.expected_loss is None or not torch.equal(loss.detach(), self.expected_loss):
            raise ValueError('Actual critic TD MSE differs from original masked target')
        self.losses.append(float(loss.detach())); self.expected_loss = None
        return self.original_reinforce(loss)

    def __len__(self): return len(self.inner)
    def __iter__(self): return iter(self.inner)
    def update_priorities(self, *a): return self.inner.update_priorities(*a)


def install_witness(checked, witness):
    core = checked.agent
    wrapper = ReplayWitness(core.replay_buffer, witness, core)
    wrapper.original_reinforce = core.q.reinforce
    core.replay_buffer = wrapper
    core.q.reinforce = wrapper.reinforce
    return wrapper


def verify_reward(rows, settings):
    """Recompute the frozen Slotted Jerk implementation, never replace rewards."""
    if settings.REWARD_FUNCTION != 'Slotted Jerk':
        raise ValueError('Unapproved reward contract')
    errors = []
    for row in rows:
        a = row['execution_audit']; reason = row['reason']
        if reason == 'upstream_collision': base = settings.CRASH_REWARD
        elif reason == 'upstream_arrival': base = settings.SUCCESS_REWARD
        else:
            # Deadline has a second cleanup step; measured_jerk is intentionally
            # None there. Its reward uses the PRE-cleanup state's physical jerk.
            jerk = a['measured_jerk']
            if jerk is None:
                jerk = row.get('reward_jerk_before_cleanup')
            if jerk is None: raise ValueError('Missing pre-cleanup reward witness')
            base = settings.TIME_REWARD*settings.TICK_LENGTH - settings.ALT_J_WEIGHT*jerk**2*settings.TICK_LENGTH
        errors.append(abs(row['reward'] - (base + a['invalid_action_reward'])))
    if max(errors, default=0.) > 1e-7:
        raise ValueError('Reward differs from frozen upstream implementation')
    return {'checked_transitions': len(rows), 'max_absolute_reward_error': max(errors, default=0.),
            'time_limit_has_extra_terminal_penalty': False}
