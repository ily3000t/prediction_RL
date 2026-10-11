"""Finite, train-only task characterization; no task classifier or learner."""
from collections import Counter
from copy import deepcopy

import numpy as np

from prediction_rl.curriculum.task_design import replay_budget, validate_design
from prediction_rl.data.dataset_contract import digest
from prediction_rl.data.response_collection import base_traffic

PROTOCOL = {
    'version': 'ic02d-task-engineering-v1', 'run_id': 'ic02d_eng_v1',
    'design_config': 'configs/development/ic02c_task_design_v1.json',
    'design_report': 'artifacts/ic/ic02c_design_v1/report.json',
    'design_report_sha256': '6b54b5b96b4fe975132d197ae4f71f357ea63fea1515397005d3f28cc353d487',
    'authorization': 'provenance/ic02d_engineering_authorization.json',
    'simulator_seeds': [11000, 11008, 11016], 'workers': 1, 'torch_threads': 1,
    'split': 'train', 'entry_contract': 'seeded_fresh_gym20_task_v1',
    'execution_contract': 'simulation_blocking_exact_v1',
    'tasks_max': 3, 'root_slots_max': 9, 'candidate_branches_max': 90,
    'fresh_sumo_sessions_max': 114, 'control_steps_max': 12234, 'raw_ticks_max': 23862,
    'predictor_inference_calls': 0, 'predictor_training_steps': 0, 'ddpg_training_steps': 0,
    'learner_instantiated': False, 'task_classifier': None, 'target_distribution_pi': None,
    'test_opened': False, 'training_ready': False, 'automatic_retry': False,
}

# Execution identity only. The failed v1 request and scientific design stay fixed.
REPAIRED_PROTOCOL = {**PROTOCOL,
    'version': 'ic02d-task-engineering-v2', 'run_id': 'ic02d_eng_v2',
    'authorization': 'provenance/ic02d_engineering_v2_authorization.json'}


def validate_protocol(value, design):
    validate_design(design)
    if not isinstance(value, dict) or digest(value) not in (digest(PROTOCOL), digest(REPAIRED_PROTOCOL)):
        raise ValueError('Changed approved three-task engineering scope')
    budget = replay_budget(3, design)
    if (value['simulator_seeds'] != design['engineering_seeds']
            or budget['fresh_sumo_sessions_max'] != value['fresh_sumo_sessions_max']
            or budget['total_control_steps_max'] != value['control_steps_max']
            or budget['total_raw_ticks_max'] != value['raw_ticks_max']):
        raise ValueError('Engineering budget differs from approved design')
    return value


def scan_checkpoints(env, design, geometry, policy, observation, reader):
    """Independent observer supporting reset and multiple SAME-lane checkpoints."""
    validate_design(design)
    targets = design['checkpoint_proposal']
    selected, actions, visited = {}, [], []
    stop = 'discovery_budget_exhausted'
    for step in range(design['discovery_max_steps']+1):
        frame = reader()
        geometry.validate_frame(frame)
        ego = frame['vehicles']['ego']
        lane = ego['lane_id']
        if not visited or visited[-1] != lane:
            visited.append(lane)
        for target in targets:
            if target['id'] in selected:
                continue
            reset = target['kind'] == 'original_reset'
            position = (None if reset else target['value'] if target['reference'] == 'from_start_m'
                        else geometry.lengths[target['lane_id']]-target['value'])
            qualifies = step == 0 if reset else lane == target['lane_id'] and ego['lane_position_m'] >= position
            if qualifies:
                selected[target['id']] = {'target': deepcopy(target), 'prefix_steps': step,
                    'prefix_actions': list(actions), 'selection': {'traffic': base_traffic(frame),
                        'frame': deepcopy(frame), 'observation': observation.tolist()},
                    'target_position_m': position,
                    'overshoot_m': None if reset else ego['lane_position_m']-position}
        if len(selected) == len(targets):
            stop = 'all_checkpoints_reached'; break
        if step == design['discovery_max_steps']:
            break
        jerk = policy(observation, step)
        observation, reward, done, info = env.step(np.array([jerk], dtype=env.action_space.dtype))
        actions.append(float(jerk))
        if done:
            stop = 'native_terminal_before_remaining_checkpoints'; break
    return {'targets': [selected.get(t['id'], {'target': deepcopy(t), 'prefix_steps': None,
                'reason': stop}) for t in targets], 'reference_actions': actions,
            'stop_reason': stop, 'visited_lane_ids': visited,
            'cap_is_not_terminal': True}


class BudgetLedger:
    """Reserve before an external action; failed attempts are never free retries."""
    PHASES = ('reference_discovery', 'root_capture_prefix', 'branch_replay_prefix', 'candidate_future')
    def __init__(self, design):
        self.budget = replay_budget(1, validate_design(design))
        self.sessions = self.control_attempts = self.raw_tick_attempts = 0
        self.completed_controls = self.completed_raw_ticks = 0
        self.controls_by_phase = Counter()
        self.phase_caps = dict(zip(self.PHASES, (self.budget[k] for k in (
            'reference_discovery_control_steps_max', 'root_capture_prefix_control_steps_max',
            'branch_replay_prefix_control_steps_max', 'candidate_future_control_steps_max'))))

    def session(self):
        if self.sessions >= self.budget['fresh_sumo_sessions_max']:
            raise ValueError('Fresh SUMO session budget exhausted before start')
        self.sessions += 1

    def control(self, phase):
        if (phase not in self.phase_caps or self.controls_by_phase[phase] >= self.phase_caps[phase]
                or self.control_attempts >= self.budget['total_control_steps_max']):
            raise ValueError('Control phase budget exhausted before step')
        self.controls_by_phase[phase] += 1; self.control_attempts += 1

    def tick(self):
        if self.raw_tick_attempts >= self.budget['total_raw_ticks_max']:
            raise ValueError('Raw SUMO tick budget exhausted before step')
        self.raw_tick_attempts += 1

    def export(self):
        return {'fresh_sumo_sessions': self.sessions, 'control_attempts': self.control_attempts,
            'completed_control_steps': self.completed_controls,
            'raw_tick_attempts': self.raw_tick_attempts, 'completed_raw_ticks': self.completed_raw_ticks,
            'control_steps_by_phase': dict(self.controls_by_phase)}


def root_evidence(history, pack, branches, repeats, geometry, config, scales, binding):
    from prediction_rl.curriculum.interaction import analyze_root
    row = analyze_root(history, pack, branches, repeats, geometry, config, binding)
    stats = row['interaction']; noise = row['repeat_noise']
    row['I_with_frozen_IC01_scales'] = (stats['mean_abs_dv_mps']/scales['dv']
        + stats['mean_abs_ds_m']/scales['ds'] if stats['common_cells'] else None)
    row['repeat_noise_with_frozen_IC01_scales'] = (noise['max_abs_dv_mps']/scales['dv']
        + noise['max_abs_ds_m']/scales['ds'] if noise['common_cells'] else None)
    row['normalization_scales'] = scales
    row['support_qualified'] = stats['supported_pairs']/stats['pair_count'] >= config['minimum_supported_pair_fraction']
    row['task_mapping'] = {'task_id': binding['task_id'], 'root_is_original_reset': not pack['prefix_actions'],
        'reference_prefix_steps': len(pack['prefix_actions']),
        'reference_prefix_used_for_characterization_only': True,
        'reference_prefix_before_training_policy_steps': 0}
    row.update(task_level=None, training_ready=False, classification_applied=False)
    return row


def validate_actual_budget(value, design):
    maximum = replay_budget(1, validate_design(design))
    keys = {'fresh_sumo_sessions', 'control_attempts', 'completed_control_steps', 'raw_tick_attempts',
            'completed_raw_ticks', 'control_steps_by_phase'}
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError('Unknown/missing actual budget field')
    phases = value['control_steps_by_phase']
    ledger = BudgetLedger(design)
    if (not isinstance(phases, dict) or set(phases)-set(ledger.PHASES)
            or any(type(v) is not int or not 0 <= v <= ledger.phase_caps[k] for k,v in phases.items())
            or any(type(value[k]) is not int or value[k] < 0 for k in keys-{'control_steps_by_phase'})
            or value['fresh_sumo_sessions'] > maximum['fresh_sumo_sessions_max']
            or value['control_attempts'] != value['completed_control_steps']
            or value['control_attempts'] != sum(phases.values())
            or value['control_attempts'] > maximum['total_control_steps_max']
            or value['raw_tick_attempts'] != value['completed_raw_ticks']
            or value['raw_tick_attempts'] > maximum['total_raw_ticks_max']):
        raise ValueError('Invalid, incomplete or over-budget successful task')
    extra = value['completed_raw_ticks']-value['completed_control_steps']-101*value['fresh_sumo_sessions']
    if not 0 <= extra <= value['fresh_sumo_sessions']:
        raise ValueError('Missing/extra waiting, insertion or cleanup ticks')
    return value
