"""Unified descriptive event analysis; unchanged author50 control and reward."""
from copy import deepcopy
import math
import statistics

from prediction_rl.data.dataset_contract import digest
from .driver_warmup import AUTHOR, B0, B3, canonical_frame
from .terminal_events_v2 import VERSION as EVENT_VERSION, classify

EXTERNAL = {AUTHOR: 'author_ddpg', 'author_st': 'author_st',
    'author_rl_mpc_safety': 'author_rl_mpc_safety',
    'author_rl_mpc_switching': 'author_rl_mpc_switching'}
MODELS = (*EXTERNAL, B0, B3)
LABELS = ('natural_arrival', 'ego_collision', 'time_limit', 'background_collision_stop',
          'arrived_far_before_endpoint', 'unexplained_ego_disappearance', 'unexplained_terminal')
PROTOCOL = {
    'version': 'p07h_external_terminal_audit_v1',
    'purpose': 'development_external_baselines_with_independent_terminal_evidence',
    'models': list(MODELS), 'project_run_seeds': [0, 1, 2],
    'simulator_seeds': list(range(200, 220)), 'condition_id': 'author50',
    'warmup_s': 50, 'max_episode_s': 100, 'max_control_calls': 501,
    'execution_contract': 'simulation_blocking_exact_v1',
    'scene_execution': 'fresh_process_first_episode_per_simulator_seed',
    'workers': 2, 'torch_threads': 1, 'device': 'cpu',
    'event_implementation': EVENT_VERSION,
    'endpoint_rule': 'recorded_arrival_position_with_original_30mps_tick_bound_and_1m_tolerance',
    'collision_rule': 'recorded_ego_participant_overrides_native_arrived_flag_not_fault_attribution',
    'historical_raw_reuse': 'verified_p7g_author50_200_210_219_only',
    'strict_native_trajectory_parity': True, 'native_reward_recomputed_with_new_labels': False,
    'reward_action_termination_changes': False, 'training_permitted': False,
    'checkpoint_selection': 'unchanged_author_snapshot_and_project_p7_final',
    'budget_matched': False, 'formal_claim': False, 'test_opened': False,
    'analysis': 'paired_descriptive_events_per_training_seed_no_winner_or_significance_gate',
    'recovery_request': 'artifacts/p7g/p7g_recovery_v1/request.json',
    'recovery_request_hash': 'a8fcf644880e745f5b6cb57f25b926d13b37cf474c08fcc280da18470fcd8988',
    'recovery_aggregate_sha256': 'bf6a85b3c560aa6a403d08966a6c5e7ed185d97f0175ac325271181f9dd0c2f1',
}
AUDIT = deepcopy(PROTOCOL)
AUDIT.update(version='p07h_external_terminal_interface_v1', purpose='bounded_external_event_interface_audit',
             simulator_seeds=[200], workers=1,
             selection='all_external_and_b0_seed2_b3_seeds0_2_not_model_selection')

# Wall-clock profiles are deliberately excluded from exact native parity.
NATIVE_FIELDS = ('arrival', 'collision', 'time_limit', 'steps', 'simulation_duration_s',
    'author_history_total_reward', 'author_history_reward_terms', 'mean_speed_mps', 'mean_abs_jerk',
    'author_postmerge_closest_distance_m', 'mean_disruption_decel_mps2', 'st_takeover_rate',
    'policy_sha256', 'predictor_sha256')


def validate_protocol(config):
    if digest(config) not in (digest(PROTOCOL), digest(AUDIT)):
        raise ValueError('Changed P7h design; approve a new immutable version')
    return config


def roster(config):
    validate_protocol(config)
    result = [{'id': f'{model}_author50_r{seed if seed is not None else "asset"}_v{scene}',
        'model_id': model, 'condition_id': 'author50', 'run_seed': seed, 'simulator_seed': scene,
        'training_id': None if model in EXTERNAL else f'{"baseline" if model == B0 else "conditional"}_s{seed}'}
        for model in MODELS for seed in ([None] if model in EXTERNAL else config['project_run_seeds'])
        for scene in config['simulator_seeds']]
    if config == AUDIT:
        result = [j for j in result if j['model_id'] in EXTERNAL or
                  (j['model_id'], j['run_seed']) in ((B0, 2), (B3, 0), (B3, 2))]
    return result


def native_row(job, outcome, decisions):
    if not decisions or len(decisions) != outcome['steps']:
        raise ValueError('Missing original decision frames')
    row = {**job, **{k: outcome[k] for k in NATIVE_FIELDS},
           'reported_collision': outcome['collision'],
           'common_initial_traffic_sha256': digest(decisions[0]['frame'])}
    if (any(type(row[k]) is not int or row[k] not in (0, 1) for k in ('arrival', 'collision', 'time_limit'))
            or sum(row[k] for k in ('arrival', 'collision', 'time_limit')) != 1
            or type(row['steps']) is not int or not 1 <= row['steps'] <= 501):
        raise ValueError('Invalid original exclusive outcome')
    for key in NATIVE_FIELDS[4:12]:
        value = row[key]
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)):
            raise ValueError('Invalid original finite measurement: '+key)
    for key in ('policy_sha256', 'predictor_sha256'):
        value = row[key]
        if value is not None and (not isinstance(value, str) or len(value) != 64 or
                                  any(c not in '0123456789abcdef' for c in value)):
            raise ValueError('Invalid checkpoint binding')
    if (row['policy_sha256'] is None) != (job['model_id'] == 'author_st'):
        raise ValueError('ST has no learned policy; other methods require an exact checkpoint')
    if (row['predictor_sha256'] is not None) != (job['model_id'] == B3):
        raise ValueError('Project predictor binding changed')
    for index, decision in enumerate(decisions):
        frame = canonical_frame(decision['frame'])
        if (frame != decision['frame'] or 'ego' not in frame['vehicles'] or
                not math.isclose(frame['simulation_time_s'], 50+.2*(index+1), abs_tol=1e-7)
                or not math.isfinite(decision['commanded_speed']) or decision['commanded_speed'] < 0):
            raise ValueError('Original author50 state/control boundary changed')
    return row


def assert_parity(row, decisions, histories, takeovers, reference):
    """Exact native scores and all available states/commands/queries/histories.

MPC query lists are not collapsed to a fictitious single jerk. Missing P7g
history fields are reported explicitly, never reconstructed from predictions.
"""
    old = reference['row']; previous = reference['decisions']
    if any(row[k] != old[k] for k in (*NATIVE_FIELDS, 'common_initial_traffic_sha256')) or len(decisions) != len(previous):
        raise ValueError('Recorder changed native outcome/reward/model identity')
    compared = set()
    for now, before in zip(decisions, previous):
        for field in ('frame', 'commanded_speed', 'requested_jerk', 'policy_queries'):
            if before.get(field) is not None:
                if now.get(field) != before[field]:
                    raise ValueError('Recorder changed original decision: '+field)
                compared.add(field)
    old_histories = reference['histories']
    if not set(histories) <= set(old_histories) or any(histories[k] != old_histories[k] for k in histories):
        raise ValueError('Recorder changed original native histories')
    if takeovers != reference['takeovers']:
        raise ValueError('Recorder changed original ST takeover sequence')
    return {'exact_native_outcome_parity': True, 'exact_available_trajectory_parity': True,
        'compared_control_steps': len(decisions), 'compared_decision_fields': sorted(compared),
        'compared_native_histories': sorted(histories),
        'historical_histories_not_present_in_raw_replay': sorted(set(old_histories)-set(histories)),
        'unavailable_fields_not_reconstructed': True}


def analyze(job, outcome, decisions, histories, takeovers, sidecar, reference):
    row = native_row(job, outcome, decisions)
    parity = assert_parity(row, decisions, histories, takeovers, reference)
    return {**row, 'parity': parity, 'independent_events': classify(sidecar, row, decisions)}


def mean(values):
    values = list(values)
    return statistics.mean(values) if values else None


def summarize(rows):
    counts = {label: sum(r['independent_events']['classification'] == label for r in rows) for label in LABELS}
    if sum(counts.values()) != len(rows):
        raise ValueError('Unknown terminal event category')
    success = [r for r in rows if r['independent_events']['classification'] == 'natural_arrival']
    return {'episodes': len(rows), 'independent_counts': counts,
        'independent_rates': {k: v/len(rows) for k, v in counts.items()},
        'native_arrivals': sum(r['arrival'] for r in rows),
        'native_arrival_with_ego_collision': sum(r['independent_events']['native_arrival_with_ego_collision'] for r in rows),
        'native_author_reward_mean': mean(r['author_history_total_reward'] for r in rows),
        'natural_arrival_duration_s': {'count': len(success), 'mean': mean(r['simulation_duration_s'] for r in success)},
        'natural_arrival_mean_abs_jerk': {'count': len(success), 'mean': mean(r['mean_abs_jerk'] for r in success)}}


def paired(left, right):
    l = left['independent_events']['classification']; r = right['independent_events']['classification']
    return {'simulator_seed': right['simulator_seed'], 'run_seed': right['run_seed'],
        'left_event': l, 'right_event': r,
        'right_minus_left': {
            'natural_arrival': int(r == 'natural_arrival')-int(l == 'natural_arrival'),
            'ego_collision': int(r == 'ego_collision')-int(l == 'ego_collision'),
            'time_limit': int(r == 'time_limit')-int(l == 'time_limit'),
            'duration_s_both_natural_arrival': right['simulation_duration_s']-left['simulation_duration_s']
                if l == r == 'natural_arrival' else None,
            'mean_abs_jerk_both_natural_arrival': right['mean_abs_jerk']-left['mean_abs_jerk']
                if l == r == 'natural_arrival' else None}}


def aggregate(config, rows):
    expected = roster(config); index = {r['id']: r for r in rows}
    if len(rows) != len(index) or set(index) != {j['id'] for j in expected}:
        raise ValueError('Missing/extra/duplicate P7h cells; do not remove failed seeds')
    for job in expected:
        row = index[job['id']]
        if (any(row.get(k) != v for k, v in job.items()) or
                any(row['parity'].get(k) is not True for k in
                    ('exact_native_outcome_parity', 'exact_available_trajectory_parity'))):
            raise ValueError('P7h cell identity/parity changed')
    for scene in config['simulator_seeds']:
        if len({r['common_initial_traffic_sha256'] for r in rows if r['simulator_seed'] == scene}) != 1:
            raise ValueError('Methods did not start from the same observed traffic')
    matrix = {}
    for model in MODELS:
        selected = [r for r in rows if r['model_id'] == model]
        seeds = sorted({r['run_seed'] for r in selected}, key=str)
        for seed in seeds:
            if len({(r['policy_sha256'], r['predictor_sha256']) for r in selected if r['run_seed'] == seed}) != 1:
                raise ValueError('Frozen model changed across scenes')
        matrix[model] = {'summary': summarize(selected), 'per_run_seed':
            {str(s): summarize([r for r in selected if r['run_seed'] == s]) for s in seeds}}
    comparisons = {}
    for left_model in (*EXTERNAL, B0):
        values = []
        for right in rows:
            if right['model_id'] != B3: continue
            left = next((r for r in rows if r['model_id'] == left_model and
                r['simulator_seed'] == right['simulator_seed'] and
                (left_model in EXTERNAL or r['run_seed'] == right['run_seed'])), None)
            if left is not None: values.append(paired(left, right))
        comparisons[left_model] = {'left': left_model, 'right': B3, 'paired_rows': values,
            'per_run_seed': {str(s): {k: {'count': len(v := [p['right_minus_left'][k] for p in values
                    if p['run_seed'] == s and p['right_minus_left'][k] is not None]), 'mean': mean(v)}
                for k in values[0]['right_minus_left']} for s in sorted({p['run_seed'] for p in values})} if values else {},
            'reused_author_snapshot_not_independent_training_replicates': left_model in EXTERNAL}
    return {'status': 'complete', 'engineering_complete': True, 'method_effect_gate': None,
        'episodes': len(rows), 'independent_simulator_seed_ids': len(config['simulator_seeds']),
        'matrix': matrix, 'comparisons': comparisons, 'rows': rows,
        'native_reward_recomputed_with_new_labels': False, 'reward_action_termination_changes': False,
        'budget_matched': False, 'formal_claim': False, 'test_opened': False,
        'event_implementation': EVENT_VERSION,
        'interpretation': 'development_diagnostic_not_formal_method_ranking_or_collision_fault_attribution'}
