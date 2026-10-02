"""P7f: descriptive driver x warmup diagnosis, not a method-selection gate."""
from copy import deepcopy
import math

from prediction_rl.data.dataset_contract import digest
from prediction_rl.evaluation.model_protocol_crosscheck import AUTHOR, B0, B3, MODELS, mean, summary

CONDITIONS = {
    'gym20': {'driver': 'gym', 'warmup_s': 20, 'max_control_calls': 500,
              'score': 'gym_transition_return', 'collision': 'original_just_had_collision_and_ego_absent'},
    'gym50': {'driver': 'gym', 'warmup_s': 50, 'max_control_calls': 500,
              'score': 'gym_transition_return', 'collision': 'original_just_had_collision_and_ego_absent'},
    'author20': {'driver': 'author', 'warmup_s': 20, 'max_control_calls': 501,
                 'score': 'author_history_total_reward', 'collision': 'original_any_vehicle_collision_check'},
    'author50': {'driver': 'author', 'warmup_s': 50, 'max_control_calls': 501,
                 'score': 'author_history_total_reward', 'collision': 'original_any_vehicle_collision_check'},
}
PROTOCOL = {
    'version': 'p07f_driver_warmup_v1', 'purpose': 'bounded_development_driver_x_warmup_diagnosis',
    'crosscheck_request': 'artifacts/p7e/p7e_cross_v1/request.json',
    'crosscheck_request_hash': 'f50f32e772fa47c8195e39ecf93fb8153cf989720f2e05be5260840e82bab5de',
    'crosscheck_aggregate_sha256': 'f16aac96b72654e2f3b3a0c60b396596c4c61458bff1f53c67cd3570b461ae86',
    'recording_request': 'artifacts/p7b/p7b_diag_v1/request.json',
    'recording_request_hash': '98d221e3b8ee24aab965189cc2e2acd5db1f3dd4f7a6b6bf1789dc9fcae91f5f',
    'recording_summary_sha256': '9cd83601e61ac2dd736cc48293d479b9c2dca8f1b72f50b870645ed9c051b040',
    'models': list(MODELS), 'project_run_seeds': [0, 1, 2], 'simulator_seeds': [200, 210, 219],
    'scene_selection': 'unchanged_pre_result_p7b_first_midpoint_last',
    'conditions': deepcopy(CONDITIONS), 'episodes_per_seed': 1,
    'scene_execution': 'fresh_process_first_episode_per_simulator_seed',
    'execution_contract': 'simulation_blocking_exact_v1', 'inference_device': 'cpu',
    'workers': 2, 'torch_threads': 1, 'policy': 'frozen_deterministic_no_exploration',
    'checkpoint_selection': 'unchanged_author_asset_and_p7_final_no_selection',
    'training_permitted': False, 'formal_claim': False, 'test_opened': False, 'budget_matched': False,
    'cross_driver_reward_delta': False,
    'analysis': 'native_events_and_common_recorded_state_action_prefix_not_formal_method_ranking',
    'historical_missing_observations': 'explicit_null_no_synthetic_reconstruction',
}
AUDIT = deepcopy(PROTOCOL)
AUDIT.update(version='p07f_driver_warmup_audit_v1', purpose='bounded_interface_audit',
             project_run_seeds=[0], simulator_seeds=[200], workers=1)


def validate_protocol(config):
    if digest(config) not in (digest(PROTOCOL), digest(AUDIT)):
        raise ValueError('Changed driver/warmup design; approve a new version')
    return config


def roster(config):
    validate_protocol(config)
    return [{'id': f'{m}_{c}_r{s if s is not None else "asset"}_v{v}',
             'model_id': m, 'condition_id': c, 'run_seed': s, 'simulator_seed': v,
             'training_id': None if m == AUTHOR else f'{"baseline" if m == B0 else "conditional"}_s{s}'}
            for m in MODELS for c in CONDITIONS
            for s in ([None] if m == AUTHOR else config['project_run_seeds'])
            for v in config['simulator_seeds']]


def is_new(job):
    return job['condition_id'] in ('gym50', 'author20')


def jobs(config):
    return [j for j in roster(config) if is_new(j)]


def canonical_frame(frame):
    """Shared ACTUAL fields, not equality of incompatible historical encodings."""
    result = {'simulation_time_s': frame['simulation_time_s'], 'vehicles': {}}
    if not math.isfinite(result['simulation_time_s']) or not frame['vehicles']:
        raise ValueError('Invalid observed traffic frame')
    for name, vehicle in sorted(frame['vehicles'].items()):
        value = {k: deepcopy(vehicle[k]) for k in ('position', 'speed', 'acceleration', 'lane_id')}
        if (not isinstance(name, str) or len(value['position']) != 2
                or not isinstance(value['lane_id'], str)
                or any(not math.isfinite(x) for x in (*value['position'], value['speed'], value['acceleration']))):
            raise ValueError('Invalid common vehicle fields')
        result['vehicles'][name] = value
    return result


def validate_decisions(decisions, row):
    dimension = 169 if row['model_id'] == B3 else 20
    if len(decisions) != row['steps']:
        raise ValueError('Missing actual decision records')
    for i, record in enumerate(decisions):
        frame = canonical_frame(record['frame'])
        obs = record['observation']
        if (frame != record['frame'] or 'ego' not in frame['vehicles']
                or not math.isclose(frame['simulation_time_s'],
                    CONDITIONS[row['condition_id']]['warmup_s']+.2*(i+1), abs_tol=1e-7)
                or not math.isfinite(record['requested_jerk']) or not -5 <= record['requested_jerk'] <= 5
                or not math.isfinite(record['commanded_speed']) or record['commanded_speed'] < 0
                or not math.isclose(record['time_feature'], .001*i, abs_tol=1e-7)
                or record['time_feature_source'] not in ('recorded_policy_query', 'reconstructed_original_single_query_sequence')
                or record['commanded_speed_source'] not in ('original_controller_return', 'upstream_helper_reconstruction')
                or (obs is not None and (len(obs) != dimension or any(not math.isfinite(x) for x in obs)))):
            raise ValueError('Invalid recorded state/action/time contract')
        if obs is None and record['observation_source'] != 'not_recorded_in_historical_native_trace':
            raise ValueError('Missing observation requires explicit historical limitation')
        if obs is not None and record['observation_source'] != 'recorded_policy_input':
            raise ValueError('Synthetic observations are not permitted')
        if is_new(row) and (obs is None or record['time_feature_source'] != 'recorded_policy_query'):
            raise ValueError('New runs require actual policy input/time records')
    if row['common_initial_traffic_sha256'] != digest(decisions[0]['frame']):
        raise ValueError('Initial common traffic binding mismatch')
    if row['decisions_sha256'] != digest(decisions):
        raise ValueError('Common decision trace changed')


def row_from_native(job, outcome, decisions, evidence):
    gym = CONDITIONS[job['condition_id']]['driver'] == 'gym'
    row = {**job, 'arrival': outcome['arrival'], 'reported_collision': outcome['collision'],
           'time_limit': outcome['time_limit'], 'steps': outcome['steps'],
           'simulation_duration_s': outcome['simulation_duration_s'],
           'native_score_name': CONDITIONS[job['condition_id']]['score'],
           'native_score': outcome['return'] if gym else outcome['author_history_total_reward'],
           'native_mean_abs_jerk': outcome['mean_abs_measured_jerk'] if gym else outcome['mean_abs_jerk'],
           'policy_sha256': outcome['policy_sha256'], 'predictor_sha256': outcome.get('predictor_sha256'),
           'common_initial_traffic_sha256': digest(decisions[0]['frame']),
           'decisions_sha256': digest(decisions), 'evidence': evidence}
    validate_decisions(decisions, row)
    return row


def prefix_comparison(left, right):
    """No alignment/interpolation after states diverge; missing fields stay missing."""
    result = {'left_steps': len(left), 'right_steps': len(right), 'common_index_count': min(len(left), len(right))}
    for field in ('frame', 'requested_jerk', 'commanded_speed', 'observation', 'time_feature'):
        compared = [(i, l[field], r[field]) for i, (l, r) in enumerate(zip(left, right))
                    if l[field] is not None and r[field] is not None]
        different = [i for i, l, r in compared if l != r]
        result[field] = {'compared_indices': len(compared), 'different_indices': len(different),
                         'first_difference_index': different[0] if different else None,
                         'exact_equal_on_available_prefix': not different if compared else None}
    result['initial_common_traffic_equal'] = left[0]['frame'] == right[0]['frame']
    result['all_common_states_equal'] = result['frame']['exact_equal_on_available_prefix']
    result['common_state_action_sequence_equal'] = (len(left) == len(right)
        and all(result[k]['exact_equal_on_available_prefix'] is True
                for k in ('frame', 'requested_jerk', 'commanded_speed', 'time_feature')))
    result['historical_observation_unavailable'] = result['observation']['compared_indices'] == 0
    result['interpretation'] = 'recorded_index_prefix_only_not_causal_attribution_after_divergence'
    return result


def aggregate(config, rows, traces):
    expected = roster(config); index = {r['id']: r for r in rows}
    if len(index) != len(rows) or set(index) != {j['id'] for j in expected} or set(traces) != set(index):
        raise ValueError('Missing/extra/duplicate diagnostic cells; no seed removal')
    for job in expected:
        r = index[job['id']]
        if any(r.get(k) != v for k, v in job.items()):
            raise ValueError('Driver/warmup model/seed identity changed')
        if (any(type(r[k]) is not int or r[k] not in (0, 1) for k in ('arrival', 'reported_collision', 'time_limit'))
                or sum(r[k] for k in ('arrival', 'reported_collision', 'time_limit')) != 1
                or type(r['steps']) is not int or not 1 <= r['steps'] <= CONDITIONS[job['condition_id']]['max_control_calls']
                or r['simulation_duration_s'] <= 0 or r['native_score_name'] != CONDITIONS[job['condition_id']]['score']
                or r['native_score'] is None or any(r[k] is not None and not math.isfinite(r[k])
                    for k in ('native_score', 'simulation_duration_s', 'native_mean_abs_jerk'))
                or (r['native_mean_abs_jerk'] is not None and r['native_mean_abs_jerk'] < 0)
                or r['evidence']['kind'] != ('new' if is_new(job) else 'historical')):
            raise ValueError('Invalid native diagnostic result')
        for key in ('policy_sha256', 'common_initial_traffic_sha256', 'decisions_sha256'):
            h = r[key]
            if not isinstance(h, str) or len(h) != 64 or any(c not in '0123456789abcdef' for c in h):
                raise ValueError('Invalid diagnostic identity hash')
        h = r['predictor_sha256']
        if ((job['model_id'] != B3 and h is not None) or (job['model_id'] == B3
                and (not isinstance(h, str) or len(h) != 64 or any(c not in '0123456789abcdef' for c in h)))):
            raise ValueError('Predictor binding changed')
        validate_decisions(traces[job['id']], r)
    for condition in CONDITIONS:
        for scene in config['simulator_seeds']:
            selected = [r for r in rows if r['condition_id'] == condition and r['simulator_seed'] == scene]
            if len({r['common_initial_traffic_sha256'] for r in selected}) != 1:
                raise ValueError('Models start from different traffic within one condition')
    matrix = {}; driver_pairs = {}; warmup_pairs = {}
    for model in MODELS:
        seeds = [None] if model == AUTHOR else config['project_run_seeds']
        for seed in seeds:
            selected = [r for r in rows if r['model_id'] == model and r['run_seed'] == seed]
            if len({(r['policy_sha256'], r['predictor_sha256']) for r in selected}) != 1:
                raise ValueError('Frozen model changed across conditions')
        matrix[model] = {c: {'episodes': len(selected := [r for r in rows if r['model_id'] == model and r['condition_id'] == c]),
                            'summary': summary(selected), 'native_score_name': CONDITIONS[c]['score'],
                            'per_run_seed': {str(s): summary([r for r in selected if r['run_seed'] == s]) for s in seeds}}
                         for c in CONDITIONS}
        def paired(left, right, reward_delta):
            values = []
            for seed in seeds:
                for scene in config['simulator_seeds']:
                    def cell(c):
                        return next(r for r in rows if r['model_id'] == model and r['run_seed'] == seed
                                    and r['simulator_seed'] == scene and r['condition_id'] == c)
                    l, r = cell(left), cell(right)
                    delta = {k: r[k]-l[k] for k in ('arrival', 'reported_collision', 'time_limit')}
                    if reward_delta: delta['native_score'] = r['native_score']-l['native_score']
                    values.append({'run_seed': seed, 'simulator_seed': scene, 'right_minus_left': delta,
                                   'recorded_prefix': prefix_comparison(traces[l['id']], traces[r['id']])})
            return {'left': left, 'right': right, 'paired_rows': values,
                    'mean_native_delta': {k: mean(v['right_minus_left'][k] for v in values) for k in values[0]['right_minus_left']},
                    'reward_delta_computed': reward_delta, 'not_a_formal_method_effect': True}
        driver_pairs[model] = {str(w): paired(f'gym{w}', f'author{w}', False) for w in (20, 50)}
        warmup_pairs[model] = {d: paired(d+'20', d+'50', True) for d in ('gym', 'author')}
    return {'status': 'complete', 'engineering_complete': True, 'method_effect_gate': None,
            'new_episodes': len(jobs(config)), 'verified_historical_episodes': len(rows)-len(jobs(config)),
            'total_matrix_cells': len(rows), 'independent_simulator_seed_ids': len(config['simulator_seeds']),
            'matrix': matrix, 'same_warmup_driver_comparisons': driver_pairs,
            'same_driver_warmup_comparisons': warmup_pairs, 'rows': rows,
            'cross_driver_reward_delta': False, 'native_collision_definitions_not_identical': True,
            'analysis': config['analysis'], 'budget_matched': False, 'formal_claim': False, 'test_opened': False}
