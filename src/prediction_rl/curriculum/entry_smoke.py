"""Frozen IC02b engineering budget and exact native/feature comparisons."""
from collections import deque
from copy import deepcopy
import hashlib
import pickle

import numpy as np

from prediction_rl.data.dataset_contract import digest
from prediction_rl.envs.fresh_task import ENTRY_CONTRACT, EXECUTION_CONTRACT

PROTOCOL = {
    'schema_version': 1, 'protocol_id': 'ic02b-entry-parity-smoke-v1',
    'authorization_scope': 'entry_prototype_and_bounded_engineering_smoke_only',
    'entry_contract': ENTRY_CONTRACT, 'execution_contract': EXECUTION_CONTRACT,
    'baseline_references': 'provenance/interaction_curriculum_baseline_v1.json',
    'ic02a_report': 'artifacts/ic/ic02a_v1/report.json',
    'ic02a_report_sha256': 'c47dd2134446892361fa84321cfb228d155effbc07638b239dd13e19b02fb888',
    'upstream_config': 'RL-MPC-LaneMerging-master/configs/train_default_1.json',
    'simulator_seeds': [11000, 11001, 11002], 'repeats': 2,
    'adapters': ['original_direct', 'worker_entry'],
    'requested_jerks': [0, 2.5, -2.5, 5, -5, 0, 0, 2.5, -2.5, 0, 0, 0],
    'max_control_steps_per_episode': 12, 'episodes_max': 12, 'control_steps_max': 144,
    'warmup_and_insertion_ticks_max': 1212, 'feature_calls_per_mode_max': 156,
    'predictor_modes': ['ordinary', 'conditional'], 'torch_threads': 1, 'device': 'cpu',
    'reference_prefix_before_policy_steps': 0, 'prefill_predictor_history': False,
    'bounded_cap_is_not_mdp_terminal': True, 'optimizer_updates': 0,
    'new_counterfactual_branches': 0, 'training_permitted': False, 'formal_claim': False,
    'test_opened': False, 'task_classifier': None, 'target_distribution_pi': None,
}


def validate_protocol(value):
    if not isinstance(value, dict) or digest(value) != digest(PROTOCOL):
        raise ValueError('Changed approved IC02b scope/budget; create and approve a new version')
    return value


def jobs(value):
    validate_protocol(value)
    return [{'id': f'{"d" if adapter == "original_direct" else "w"}{seed}r{repeat}',
             'adapter': adapter, 'simulator_seed': seed, 'repeat': repeat}
            for seed in value['simulator_seeds'] for adapter in value['adapters']
            for repeat in range(value['repeats'])]


def json_safe(value):
    if isinstance(value, np.ndarray):
        if not np.isfinite(value).all():
            raise ValueError('Nonfinite native array')
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, deque)):
        return [json_safe(item) for item in value]
    return value


def rng_fingerprint():
    import torch
    values = {'python': hashlib.sha256(pickle.dumps(__import__('random').getstate())).hexdigest(),
              'numpy': hashlib.sha256(pickle.dumps(np.random.get_state())).hexdigest(),
              'torch_cpu': hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
              'cuda_initialized': torch.cuda.is_initialized()}
    values['torch_cuda'] = ([hashlib.sha256(x.cpu().numpy().tobytes()).hexdigest()
                             for x in torch.cuda.get_rng_state_all()] if values['cuda_initialized'] else [])
    return values


class PacketView:
    """CURRENT packet only; original feature wrapper never receives the trace/future."""
    execution_contract = EXECUTION_CONTRACT

    def __init__(self, description):
        from gym.spaces import Box
        self.observation_space = Box(np.asarray(description['observation']['low'], np.float32),
                                     np.asarray(description['observation']['high'], np.float32), dtype=np.float32)
        self.action_space = Box(np.asarray(description['action']['low'], np.float32),
                                np.asarray(description['action']['high'], np.float32), dtype=np.float32)
        self.packet = None

    def accept(self, packet):
        self.packet = deepcopy(packet)

    def frame(self):
        return deepcopy(self.packet['current_frame'])

    def reset(self):
        if self.packet['phase'] != 'reset':
            raise ValueError('Packet view reset boundary mismatch')
        return np.asarray(self.packet['observation'], np.float32), deepcopy(self.packet['info'])

    def step(self, action):
        if self.packet['phase'] != 'step' or float(action[0]) != self.packet['info']['execution_audit']['requested_jerk']:
            raise ValueError('Packet view action/boundary mismatch')
        return (np.asarray(self.packet['observation'], np.float32), self.packet['reward'],
                self.packet['done'], deepcopy(self.packet['info']))

    def close(self):
        pass


class FeaturePair:
    """Reuse BOTH unchanged PredictionFeatureEnv paths without a second SUMO tick."""
    def __init__(self, spaces, providers):
        from prediction_rl.envs.prediction_features import PredictionFeatureEnv
        if set(providers) != {'ordinary', 'conditional'}:
            raise ValueError('Require both frozen predictors')
        self.views = {mode: PacketView(spaces) for mode in providers}
        self.envs = {mode: PredictionFeatureEnv(self.views[mode], mode, provider, self.views[mode].frame)
                     for mode, provider in providers.items()}
        self.calls = dict.fromkeys(providers, 0)

    def accept(self, packet):
        result = {}
        for mode, env in self.envs.items():
            self.views[mode].accept(packet)
            if packet['phase'] == 'reset':
                observation, info = env.reset()
            else:
                observation, reward, done, info = env.step(np.asarray(
                    [packet['info']['execution_audit']['requested_jerk']], np.float32))
                if reward != packet['reward'] or done != packet['done']:
                    raise ValueError('Feature wrapper changed native reward/done')
            metadata = info['prediction_features']
            if not packet['done']:
                self.calls[mode] += 1
            result[mode] = {'observation': observation.tolist(), 'history_frames': metadata['history_frames'],
                            'terminal_zero_block': metadata['terminal_zero_block'],
                            'time_feature': packet['control_step']*.001,
                            'feature_latency_s': metadata['latency_s']}
        return result


def comparable_trace(trace):
    # Only measured latency and process identity are non-semantic differences.
    values = deepcopy(trace)
    values.pop('job', None)
    runtime = values.get('runtime', {})
    runtime.pop('pid', None)
    for packet in values['packets']:
        for value in packet.get('features', {}).values():
            value.pop('feature_latency_s', None)
    return json_safe(values)


def compare_traces(left, right):
    if comparable_trace(left) != comparable_trace(right):
        raise ValueError('Native/feature trajectory mismatch; exact parity cannot be relaxed')
    return {'exact_native_and_feature_parity': True, 'states_compared': len(left['packets']),
            'control_steps_compared': len(left['packets'])-1}


def validate_trace(trace, job, geometry):
    if trace['job'] != job or len(trace['packets']) < 1 or len(trace['packets']) > 13:
        raise ValueError('Invalid bounded episode identity/count')
    count = 0
    for index, packet in enumerate(trace['packets']):
        phase = 'reset' if index == 0 else 'step'
        if packet['phase'] != phase or packet['control_step'] != index or type(packet['done']) is not bool:
            raise ValueError('Packet sequence/terminal changed')
        if index == 0 and (packet['done'] or packet['reward'] is not None):
            raise ValueError('Reset is not a transition')
        observation = np.asarray(packet['observation'])
        if observation.shape != (20,) or not np.isfinite(observation).all():
            raise ValueError('Invalid original observation')
        geometry.validate_frame(packet['current_frame'], require_ego=not packet['done'])
        events = packet['raw_events']['steps']
        expected = 101 if index == 0 else 2 if packet['info']['execution_audit']['termination_reason'] == 'upstream_time_limit' else 1
        if len(events) != expected:
            raise ValueError('Lost/extra native waiting/insertion/control/cleanup tick')
        for event in events:
            if (event['index'] != count or abs(event['before']['simulation_time_s']-.2*count) > 1e-7
                    or abs(event['after']['simulation_time_s']-event['before']['simulation_time_s']-.2) > 1e-7):
                raise ValueError('Raw tick sequence mismatch')
            count += 1
        if packet['raw_step_count'] != count:
            raise ValueError('Raw cumulative tick mismatch')
        if packet['current_frame']['simulation_time_s'] != events[-1]['after']['simulation_time_s']:
            raise ValueError('Current frame is not at the last native tick')
        frame = packet['current_frame']
        projection = {'simulation_time_s': frame['simulation_time_s'], 'vehicles': {
            actor: {key: state[key] for key in ('position', 'speed', 'acceleration', 'lane_id')}
            for actor, state in frame['vehicles'].items()}}
        original_current = packet['info']['traffic'] if index == 0 else packet['info']['execution_audit']['after']
        if projection != original_current:
            raise ValueError('Transported predictor frame differs from the native current traffic')
        if index == 0:
            created = packet['raw_events']['creations']
            if (len(created) != 1 or created[0]['vehicle_id'] != 'ego' or created[0]['route_id'] != 'rampRoute'
                    or created[0]['depart_position_m'] != 40 or created[0]['arrival_position_m'] != 50
                    or created[0]['simulation_time_s'] != 20.):
                raise ValueError('Original reset/insertion changed')
        else:
            if (isinstance(packet['reward'], bool) or not np.isfinite(packet['reward']) or packet['info']['execution_audit']['requested_jerk']
                    != PROTOCOL['requested_jerks'][index-1]):
                raise ValueError('Native reward/requested action changed')
            if packet['raw_events']['creations']:
                raise ValueError('Unexpected ego reinsertion')
            audit = packet['info']['execution_audit']
            if (audit['before']['simulation_time_s'] != events[0]['before']['simulation_time_s']
                    or audit['after']['simulation_time_s'] != events[-1]['after']['simulation_time_s']
                    or audit['execution_contract'] != EXECUTION_CONTRACT
                    or (audit['termination_reason'] is not None) != packet['done']):
                raise ValueError('Action/native-event terminal boundary mismatch')
        if packet['done'] and index != len(trace['packets'])-1:
            raise ValueError('Transition after native terminal')
        for mode, feature in packet['features'].items():
            result = np.asarray(feature['observation'])
            expected_history = min(11, index+1) if not packet['done'] else min(11, index)
            if (mode not in PROTOCOL['predictor_modes'] or result.shape != (169,)
                    or not np.isfinite(result).all() or not np.array_equal(result[:20], observation)
                    or feature['history_frames'] != expected_history
                    or feature['time_feature'] != index*.001
                    or feature['terminal_zero_block'] != packet['done']
                    or (packet['done'] and result[20:].any())):
                raise ValueError('History/prediction/time/terminal observation contract mismatch')
        if set(packet['features']) != set(PROTOCOL['predictor_modes']):
            raise ValueError('Missing predictor mode')
    expected_stop = 'native_terminal' if trace['packets'][-1]['done'] else 'engineering_cap_not_terminal'
    if trace['stop_reason'] != expected_stop or (expected_stop != 'native_terminal' and len(trace['packets']) != 13):
        raise ValueError('Engineering cap cannot be converted into an MDP terminal')
    return {'actual_control_steps': len(trace['packets'])-1, 'raw_ticks': count,
            'warmup_and_insertion_ticks': 101, 'native_done': trace['packets'][-1]['done'],
            'feature_calls_per_mode': sum(not p['done'] for p in trace['packets'])}
