"""Frozen model x native protocol; no cross-protocol reward deltas."""
from copy import deepcopy
import math
import statistics

from prediction_rl.data.dataset_contract import digest
from prediction_rl.evaluation.closed_loop_diagnostics import PROTOCOL as P7_PARENT

AUTHOR = 'author_pretrained_ddpg'
B0 = 'project_20k_b0_ddpg'
B3 = 'project_20k_b3_conditional_ddpg'
GYM = 'p7_gym20'
AUTHOR_LOOP = 'p7d_author50'
MODELS = (AUTHOR, B0, B3)
DRIVERS = (GYM, AUTHOR_LOOP)
SCORES = {GYM: 'gym_transition_return', AUTHOR_LOOP: 'author_history_total_reward'}
PROTOCOL = {
    'version': 'p07e_model_protocol_crosscheck_v1',
    'purpose': 'development_model_origin_x_protocol_sensitivity',
    **{k: P7_PARENT[k] for k in ('parent_request', 'parent_request_hash', 'parent_aggregate_sha256')},
    'external_request': 'artifacts/p7d/p7d_compare_v1/request.json',
    'external_request_hash': '18a2434c49a9f44ff27b0b0e27221b2d1914af2703373bd606609febac0a67e3',
    'external_aggregate_sha256': 'fd1b48fd262aa676e4c1cb1e56b64adc17e921a58924562503eb104fdd2aaeec',
    'models': list(MODELS), 'drivers': list(DRIVERS), 'project_run_seeds': [0, 1, 2],
    'simulator_seeds': list(range(200, 220)), 'episodes_per_seed': 1,
    'scene_execution': 'fresh_process_first_episode_per_simulator_seed',
    'driver_contracts': {
        GYM: {'driver': 'unchanged_p7_gym', 'warmup_s': 20, 'max_control_calls': 500,
              'score': SCORES[GYM], 'collision': 'original_just_had_collision_and_ego_absent'},
        AUTHOR_LOOP: {'driver': 'unchanged_upstream_control_run_episode', 'warmup_s': 50,
                      'max_control_calls': 501, 'score': SCORES[AUTHOR_LOOP],
                      'collision': 'original_any_vehicle_collision_check'}},
    'execution_contract': 'simulation_blocking_exact_v1', 'inference_device': 'cpu',
    'torch_threads': 1, 'workers': 2,
    'checkpoint_selection': 'unchanged_author_asset_and_p7_final_no_selection',
    'training_permitted': False, 'budget_matched': False, 'formal_claim': False,
    'test_opened': False,
    'cross_protocol_analysis': 'same_seed_native_event_sensitivity_not_identical_traffic_or_causal_warmup_effect',
    'cross_protocol_reward_delta': False,
}
AUDIT = deepcopy(PROTOCOL)
AUDIT.update(version='p07e_crosscheck_engineering_smoke_v1',
             purpose='bounded_crosscheck_interface_audit', simulator_seeds=[200], workers=1)


def validate_protocol(value):
    if digest(value) not in (digest(PROTOCOL), digest(AUDIT)):
        raise ValueError('Changed crosscheck design; create and approve a new version')
    return value


def roster(config):
    validate_protocol(config)
    return [{'id': f'{model}_{driver}_r{seed if seed is not None else "asset"}_v{scene}',
             'model_id': model, 'driver_id': driver, 'run_seed': seed, 'simulator_seed': scene,
             'training_id': None if model == AUTHOR else f'{"baseline" if model == B0 else "conditional"}_s{seed}'}
            for model in MODELS for driver in DRIVERS
            for seed in ([None] if model == AUTHOR else config['project_run_seeds'])
            for scene in config['simulator_seeds']]


def is_new(job):
    return (job['model_id'], job['driver_id']) in ((AUTHOR, GYM), (B0, AUTHOR_LOOP))


def jobs(config):
    return [j for j in roster(config) if is_new(j)]


def native_row(job, row, evidence):
    gym = job['driver_id'] == GYM
    return {**job, 'arrival': row['arrival'], 'reported_collision': row['collision'],
            'time_limit': row['time_limit'], 'steps': row['steps'],
            'simulation_duration_s': row['simulation_duration_s'],
            'native_score_name': SCORES[job['driver_id']],
            'native_score': row['return'] if gym else row['author_history_total_reward'],
            'native_mean_abs_jerk': row['mean_abs_measured_jerk'] if gym else row['mean_abs_jerk'],
            'initial_traffic_sha256': row['initial_traffic_sha256'],
            'policy_sha256': row['policy_sha256'], 'predictor_sha256': row.get('predictor_sha256'),
            'evidence': evidence}


def mean(values):
    values = list(values)
    return statistics.mean(values) if values else None


def summary(rows):
    result = {k: {'mean': mean(r[k] for r in rows if r[k] is not None),
                  'count': sum(r[k] is not None for r in rows)}
              for k in ('arrival', 'reported_collision', 'time_limit', 'native_score',
                        'simulation_duration_s', 'native_mean_abs_jerk')}
    for key, field in (('successful_episode_time_s', 'simulation_duration_s'),
                       ('successful_episode_native_mean_abs_jerk', 'native_mean_abs_jerk')):
        valid = [r[field] for r in rows if r['arrival'] and r[field] is not None]
        result[key] = {'mean': mean(valid), 'count': len(valid)}
    return result


def aggregate(config, rows):
    expected = roster(config)
    index = {r['id']: r for r in rows}
    if len(index) != len(rows) or set(index) != {j['id'] for j in expected}:
        raise ValueError('Missing/duplicate/extra crosscheck rows; no seed exclusion')
    for job in expected:
        r = index[job['id']]
        if any(r.get(k) != v for k, v in job.items()):
            raise ValueError('Crosscheck model/driver/seed identity mismatch')
        if (any(type(r[k]) is not int or r[k] not in (0, 1)
                for k in ('arrival', 'reported_collision', 'time_limit'))
                or sum(r[k] for k in ('arrival', 'reported_collision', 'time_limit')) != 1
                or type(r['steps']) is not int
                or not 1 <= r['steps'] <= config['driver_contracts'][job['driver_id']]['max_control_calls']
                or r['simulation_duration_s'] <= 0
                or r['native_score_name'] != SCORES[job['driver_id']]
                or r['native_score'] is None
                or (r['native_mean_abs_jerk'] is not None and r['native_mean_abs_jerk'] < 0)
                or any(r[k] is not None and not math.isfinite(r[k]) for k in
                       ('native_score', 'simulation_duration_s', 'native_mean_abs_jerk'))):
            raise ValueError('Invalid native crosscheck outcome')
        for key in ('policy_sha256', 'initial_traffic_sha256'):
            h = r[key]
            if not isinstance(h, str) or len(h) != 64 or any(c not in '0123456789abcdef' for c in h):
                raise ValueError('Missing native identity hash')
        predictor = r['predictor_sha256']
        if ((job['model_id'] != B3 and predictor is not None)
                or (job['model_id'] == B3 and (not isinstance(predictor, str) or len(predictor) != 64
                    or any(c not in '0123456789abcdef' for c in predictor)))):
            raise ValueError('Missing/inapplicable predictor binding')
        if r['evidence'].get('kind') != ('new' if is_new(job) else 'historical'):
            raise ValueError('Historical result cannot substitute a missing crosscheck job')
    # Hash encodings/initial conditions differ across drivers: compare ONLY within.
    for driver in DRIVERS:
        for scene in config['simulator_seeds']:
            selected = [r for r in rows if r['driver_id'] == driver and r['simulator_seed'] == scene]
            if len({r['initial_traffic_sha256'] for r in selected}) != 1:
                raise ValueError('Models did not start from identical traffic within driver')
    for model in MODELS:
        for seed in ([None] if model == AUTHOR else config['project_run_seeds']):
            selected = [r for r in rows if r['model_id'] == model and r['run_seed'] == seed]
            if len({r['policy_sha256'] for r in selected}) != 1:
                raise ValueError('Crosscheck did not use the same frozen checkpoint')
            if len({r['predictor_sha256'] for r in selected}) != 1:
                raise ValueError('Predictor binding changed across protocols')
    matrix = {}
    for model in MODELS:
        matrix[model] = {}
        for driver in DRIVERS:
            selected = [r for r in rows if r['model_id'] == model and r['driver_id'] == driver]
            matrix[model][driver] = {
                'episodes': len(selected), 'native_score_name': SCORES[driver],
                'independent_policy_snapshots': 1 if model == AUTHOR else len(config['project_run_seeds']),
                'summary': summary(selected),
                'per_run_seed': {str(s): summary([r for r in selected if r['run_seed'] == s])
                                 for s in ([None] if model == AUTHOR else config['project_run_seeds'])}}
    within = {}
    for driver in DRIVERS:
        for left, right in ((AUTHOR, B0), (AUTHOR, B3), (B0, B3)):
            pairs = []
            for seed in config['project_run_seeds']:
                for scene in config['simulator_seeds']:
                    l = next(r for r in rows if r['model_id'] == left and r['driver_id'] == driver
                             and r['run_seed'] == (None if left == AUTHOR else seed) and r['simulator_seed'] == scene)
                    r = next(r for r in rows if r['model_id'] == right and r['driver_id'] == driver
                             and r['run_seed'] == seed and r['simulator_seed'] == scene)
                    d = {k: r[k] - l[k] for k in ('arrival', 'reported_collision', 'time_limit', 'native_score')}
                    d['time_s_both_successful'] = r['simulation_duration_s'] - l['simulation_duration_s'] if l['arrival'] and r['arrival'] else None
                    pairs.append({'run_seed': seed, 'simulator_seed': scene, 'right_minus_left': d})
            per_seed = {str(s): {k: {'mean': mean(p['right_minus_left'][k] for p in pairs
                                                if p['run_seed'] == s and p['right_minus_left'][k] is not None),
                                    'count': sum(p['run_seed'] == s and p['right_minus_left'][k] is not None for p in pairs)}
                                for k in pairs[0]['right_minus_left']} for s in config['project_run_seeds']}
            within[f'{driver}:{right}-minus-{left}'] = {
                'left': left, 'right': right, 'driver_id': driver, 'native_score_name': SCORES[driver],
                'per_run_seed': per_seed, 'paired_rows': pairs,
                'equal_run_seed_mean_delta': {k: mean(v[k]['mean'] for v in per_seed.values() if v[k]['mean'] is not None)
                                             for k in pairs[0]['right_minus_left']},
                'author_snapshot_reused_not_independent_replicates': left == AUTHOR}
    sensitivity = {}
    for model in MODELS:
        pairs = []
        for seed in ([None] if model == AUTHOR else config['project_run_seeds']):
            for scene in config['simulator_seeds']:
                selected = {r['driver_id']: r for r in rows if r['model_id'] == model
                            and r['run_seed'] == seed and r['simulator_seed'] == scene}
                pairs.append({'run_seed': seed, 'simulator_seed': scene,
                              'author50_minus_gym20_native_event_delta': {
                                  k: selected[AUTHOR_LOOP][k] - selected[GYM][k]
                                  for k in ('arrival', 'reported_collision', 'time_limit')}})
        sensitivity[model] = {'paired_rows': pairs,
            'mean_native_event_delta': {k: mean(p['author50_minus_gym20_native_event_delta'][k] for p in pairs)
                                       for k in ('arrival', 'reported_collision', 'time_limit')},
            'interpretation': config['cross_protocol_analysis'], 'reward_delta_computed': False}
    return {'status': 'complete', 'engineering_complete': True, 'method_effect_gate': None,
            'new_episodes': len(jobs(config)), 'verified_historical_episodes': len(rows) - len(jobs(config)),
            'total_matrix_cells': len(rows), 'independent_simulator_seed_ids': len(config['simulator_seeds']),
            'matrix': matrix, 'within_protocol_comparisons': within, 'protocol_sensitivity': sensitivity,
            'rows': rows, 'budget_matched': False, 'formal_claim': False, 'test_opened': False,
            'native_collision_definitions_not_identical': True, 'cross_protocol_reward_delta': False,
            'analysis': 'descriptive_development_crosscheck_not_predictor_causal_effect_or_convergence_test'}
