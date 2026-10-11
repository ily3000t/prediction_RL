"""IC03b explicit bounded execution scope; never authorizes optimizer updates."""
from copy import deepcopy
import math

import numpy as np

from prediction_rl.curriculum.interaction_intensity import OfflineEpisodeSampler
from prediction_rl.data.dataset_contract import digest

SCOPE = 'ic03b_approved_bounded_integration_smoke_v1'
PROTOCOL = {
    'schema_version': 1, 'protocol_id': SCOPE, 'run_id': 'ic03b_smoke_v1',
    'scope': 'approved_engineering_smoke_not_formal_training',
    'arms': ['b2_mix', 'b2_curriculum', 'b3_mix', 'b3_curriculum'],
    'optimizer_seed': 0, 'sampling_seed': 17003, 'episodes_per_arm': 2,
    'native_control_cap': 500, 'sumo_sessions_max': 8, 'control_steps_max': 4000,
    'raw_ticks_max': 4816, 'feature_calls_max': 4008,
    'optimizer_updates': 0, 'predictor_training_steps': 0, 'new_counterfactual_branches': 0,
    'torch_threads': 1, 'device': 'cpu', 'learning_rate': .0002,
    'original_replay_warmup': 5000, 'original_minibatch_size': 100,
    'entry_contract': 'seeded_fresh_gym20_task_v1',
    'execution_contract': 'simulation_blocking_exact_v1',
    'reference_prefix_steps': 0, 'native_terminal_required': True,
    'strata_use': 'approved_smoke_only_from_sealed_IC02h_proposals',
    'new_sumo_sessions_authorized': True, 'frozen_predictor_inference_authorized': True,
    'fresh_learner_act_authorized': True, 'optimizer_updates_authorized': False,
    'formal_training_authorized': False, 'training_ready': False, 'test_opened': False,
    'automatic_retry': False, 'resume_incomplete_job': False,
}


def validate_protocol(value):
    if not isinstance(value, dict) or digest(value) != digest(PROTOCOL):
        raise ValueError('Changed approved eight-episode smoke; no silent budget/version change')
    return value


def task_roster(proposals):
    tasks = proposals['tasks']
    if (len(tasks) != 32 or [t['simulator_seed'] for t in tasks] != list(range(11000, 11249, 8))
            or len({t['task_id'] for t in tasks}) != 32
            or any(t['task_level'] is not None or t['classification_applied'] is not False
                   or t['execution_authorized'] is not False for t in tasks)
            or {k:sum(t['proposed_level'] == k for t in tasks) for k in ('L','M','H')}
                   != {'L':11, 'M':10, 'H':11}):
        raise ValueError('Need the complete unchanged IC02h proposal, not a selected live pool')
    return [{'task_id':t['task_id'], 'simulator_seed':t['simulator_seed'], 'level':t['proposed_level']}
            for t in tasks]


def jobs(tasks):
    result = []
    for arm in PROTOCOL['arms']:
        model, mode = arm.split('_', 1)
        algorithm = OfflineEpisodeSampler(tasks, mode, PROTOCOL['sampling_seed'],
                                          purpose='offline_algorithm_test_only')
        planned = []
        for _ in range(2):
            choice = algorithm.begin_episode()
            planned.append({k:choice[k] for k in ('task_id','simulator_seed','level')})
            # Plan RNG choices only. All actual episodes end before 20k, so the
            # number below does not affect stage/mass. Never relabel it a real step.
            algorithm.end_episode(1)
        result.append({'id':arm, 'predictor_mode':'ordinary' if model == 'b2' else 'conditional',
                       'sampling_mode':mode, 'planned_tasks':planned})
    return result


class SmokeEpisodeSampler:
    """Pure sampler composition plus an explicit two-episode authorization cap."""
    scope = SCOPE
    def __init__(self, tasks, job, request_hash, protocol):
        validate_protocol(protocol)
        if job not in jobs(tasks) or not isinstance(request_hash, str) or len(request_hash) != 64:
            raise ValueError('Invalid frozen smoke binding')
        if any(c not in '0123456789abcdef' for c in request_hash):
            raise ValueError('Invalid request digest')
        self.job, self.request_hash = deepcopy(job), request_hash
        self.algorithm = OfflineEpisodeSampler(tasks, job['sampling_mode'], PROTOCOL['sampling_seed'],
                                              purpose='offline_algorithm_test_only')
        self.started = self.completed = 0

    def begin_episode(self):
        if self.started != self.completed or self.started >= PROTOCOL['episodes_per_arm']:
            raise ValueError('No task retry, mid-episode switch or third smoke session')
        choice = self.algorithm.begin_episode()
        if {k:choice[k] for k in ('task_id','simulator_seed','level')} != self.job['planned_tasks'][self.started]:
            raise ValueError('Actual sampler task differs from the frozen request')
        self.started += 1
        return {**choice, 'smoke_scope':SCOPE, 'smoke_request_hash':self.request_hash,
                'smoke_execution_authorized':True, 'formal_training_authorized':False}

    def end_episode(self, actual_control_steps):
        if type(actual_control_steps) is not int or not 1 <= actual_control_steps <= 500:
            raise ValueError('Native episode outside approved smoke control cap')
        result = self.algorithm.end_episode(actual_control_steps)
        self.completed += 1
        return {**result, 'smoke_scope':SCOPE, 'smoke_request_hash':self.request_hash,
                'smoke_execution_authorized':True, 'formal_training_authorized':False}


class NativePacketAudit:
    """Validate current native evidence; no simulator calls or policy inputs."""
    def __init__(self, geometry, settings):
        self.geometry, self.settings = geometry, settings
        self.steps = self.raw_ticks = self.feature_calls = 0
        self.previous_acceleration = 0.
        self.finished = False
        self.reward_error_max = 0.

    def accept(self, packet, obs, reward, done, info):
        phase = 'reset' if self.raw_ticks == 0 else 'step'
        if self.finished or packet['phase'] != phase or packet['control_step'] != self.steps:
            raise ValueError('Native packet sequence changed')
        if (type(done) is not bool or packet['done'] != done
                or not np.array_equal(np.asarray(packet['observation'], np.float32), obs[:20])):
            raise ValueError('Feature wrapper changed native observation/done')
        events = packet['raw_events']['steps']
        reason = None if phase == 'reset' else packet['info']['execution_audit']['termination_reason']
        if reason not in (None, 'upstream_arrival', 'upstream_collision', 'upstream_time_limit'):
            raise ValueError('Unknown native terminal reason')
        expected = 101 if phase == 'reset' else 2 if reason == 'upstream_time_limit' else 1
        if len(events) != expected:
            raise ValueError('Extra/missing preheat, insertion, control or cleanup tick')
        for event in events:
            if (event['index'] != self.raw_ticks
                    or abs(event['before']['simulation_time_s']-.2*self.raw_ticks) > 1e-7
                    or abs(event['after']['simulation_time_s']-event['before']['simulation_time_s']-.2) > 1e-7):
                raise ValueError('Raw tick clock mismatch')
            self.raw_ticks += 1
        frame = packet['current_frame']
        self.geometry.validate_frame(frame, require_ego=not done)
        projection = {'simulation_time_s':frame['simulation_time_s'], 'vehicles': {
            actor:{k:v[k] for k in ('position','speed','acceleration','lane_id')}
            for actor,v in frame['vehicles'].items()}}
        current = packet['info']['traffic'] if phase == 'reset' else packet['info']['execution_audit']['after']
        if (projection != current or packet['raw_step_count'] != self.raw_ticks
                or frame['simulation_time_s'] != events[-1]['after']['simulation_time_s']):
            raise ValueError('Predictor frame differs from native current state')
        if phase == 'reset':
            created = packet['raw_events']['creations']
            if (reward is not None or done or packet['reward'] is not None or len(created) != 1
                    or created[0]['vehicle_id'] != 'ego' or created[0]['route_id'] != 'rampRoute'
                    or created[0]['simulation_time_s'] != 20.
                    or created[0]['depart_position_m'] != 40. or created[0]['arrival_position_m'] != 50.):
                raise ValueError('Original reset/ego insertion semantics changed')
        else:
            a = packet['info']['execution_audit']
            if (reward != packet['reward'] or a != info['execution_audit']
                    or a['execution_contract'] != PROTOCOL['execution_contract']
                    or packet['raw_events']['creations'] or (reason is not None) != done):
                raise ValueError('Native execution/terminal/reward changed')
            if self.settings['REWARD_FUNCTION'] != 'Slotted Jerk':
                raise ValueError('Unapproved reward implementation')
            if reason == 'upstream_arrival': base = self.settings['SUCCESS_REWARD']
            elif reason == 'upstream_collision': base = self.settings['CRASH_REWARD']
            else:
                # The original deadline reward precedes the removal/cleanup tick.
                physical = events[0]['after']['ego']
                if physical is None: raise ValueError('Missing pre-cleanup physical reward witness')
                jerk = (physical['acceleration']-self.previous_acceleration)/.2
                base = self.settings['TIME_REWARD']*.2-self.settings['ALT_J_WEIGHT']*jerk**2*.2
                self.previous_acceleration = physical['acceleration']
            error = abs(reward-(base+a['invalid_action_reward']))
            self.reward_error_max = max(self.reward_error_max, error)
            if not math.isfinite(error) or error > 1e-7:
                raise ValueError('Reward differs from the original Slotted Jerk formula')
        features = info['prediction_features']
        expected_history = min(11, self.steps+1) if not done else min(11, self.steps)
        if (features['history_frames'] != expected_history or features['terminal_zero_block'] != done
                or obs.shape != (169,) or not np.isfinite(obs).all() or (done and obs[20:].any())):
            raise ValueError('Per-task history or terminal feature contract changed')
        self.feature_calls += int(not done)
        if phase == 'step': self.finished = done
        # Next expected native packet step (reset has no transition).
        self.steps += 1

    def result(self):
        actual = self.steps-1
        if not self.finished or not 1 <= actual <= 500 or self.raw_ticks > 602:
            raise ValueError('Native full episode absent or budget exceeded; no synthetic done')
        return {'actual_control_steps':actual, 'raw_ticks':self.raw_ticks,
                'feature_calls':self.feature_calls, 'reward_max_absolute_error':self.reward_error_max,
                'native_terminal_verified':True}
