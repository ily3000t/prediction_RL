"""Author-loop external comparison. Negative outcomes are valid observations."""
from copy import deepcopy
import json
import math
import statistics
import numpy as np
from prediction_rl.data.dataset_contract import digest
from prediction_rl.evaluation.closed_loop_diagnostics import PROTOCOL as PARENT_AUDIT

ARMS = ('author_ddpg', 'author_st', 'author_rl_mpc_safety', 'author_rl_mpc_switching', 'conditional')
CONFIGS = dict(zip(ARMS, ('train_default_1.json', 'st_default.json', 'combined_default_1.json', 'combined_default_1b.json', 'train_default_1.json')))
ASSETS = {'policy.pt': '548663282e348356f658fe8204b44b7ad589d935b05a1f18fc3d8927b3dd7750',
          'q.pt': 'a17d7a891ca773b78f0096e1c1f176f15a2eee51adcb9f981e3f496f6efc5213'}
PROTOCOL = {
    'version': 'p07d_external_baselines_v1', 'purpose': 'development_external_comparison',
    **{k: PARENT_AUDIT[k] for k in ('parent_request', 'parent_request_hash', 'parent_aggregate_sha256')},
    'arms': list(ARMS), 'conditional_run_seeds': [0, 1, 2], 'simulator_seeds': list(range(200, 220)),
    'episodes_per_seed': 1, 'scene_execution': 'fresh_process_first_episode_per_simulator_seed',
    'episode_driver': 'unchanged_upstream_control_run_episode_v1',
    'warmup_s': 50, 'max_episode_s': 100, 'max_control_calls': 501,
    'execution_contract': 'simulation_blocking_exact_v1',
    'inference_device': 'cpu', 'torch_threads': 1, 'workers': 2,
    'checkpoint_selection': 'frozen_author_snapshot_and_p7_final_no_selection',
    'reward_score': 'unchanged_rl_get_rl_custom_episode_stats_slotted_jerk',
    'author_training_replicates': 1, 'budget_matched': False, 'formal_claim': False, 'test_opened': False}
AUDIT = deepcopy(PROTOCOL)
AUDIT.update(version='p07d_external_engineering_smoke_v1', purpose='bounded_external_controller_parity_audit',
             conditional_run_seeds=[0], simulator_seeds=[200], workers=1)


def validate_protocol(config):
    if digest(config) not in (digest(PROTOCOL), digest(AUDIT)):
        raise ValueError('Changed external design: create and approve a new version')
    return config


def jobs(config):
    validate_protocol(config)
    return [{'id': f'{arm}_r{seed if seed is not None else "asset"}_v{scene}',
             'arm': arm, 'run_seed': seed, 'simulator_seed': scene, 'config': CONFIGS[arm],
             'training_id': None if seed is None else f'conditional_s{seed}'}
            for arm in ARMS
            for seed in (config['conditional_run_seeds'] if arm == 'conditional' else [None])
            for scene in config['simulator_seeds']]


def common_settings(settings):
    """Only orchestration/model path and declared supervisor choices may differ."""
    ignored = {'TASK', 'LOG_DIR', 'FULL_LOG_DIR', 'MODEL_NAME', 'NUM_EPISODES', 'SEED',
               'ROLLOUT_LENGTH', 'ST_TEST_ROLLOUTS', 'USE_MIN_ALLOWED_DISTANCE_IN_COMBINED_SOLVER',
               'LIMIT_DQN_SPEED', 'TEST_ST_STRICTLY_BETTER', 'TEST_ROLLOUT_STATE', 'CHECK_ROLLOUT_CRASH',
               'COMBINATION_MIN_DISTANCE', 'STOP_X', 'REMEMBER_LAST_CHOICE_FOR_SWITCHING_COMBINED'}
    # Settings includes integer-keyed action dictionaries. Freeze JSON semantics
    # BEFORE hashing: reading a receipt converts those keys to strings.
    return json.loads(json.dumps({k: v for k, v in settings.items() if k not in ignored}, allow_nan=False))


def finite_mean(values):
    values = list(values)
    if any(not math.isfinite(float(x)) for x in values): raise ValueError('Nonfinite evaluation measurement')
    return statistics.mean(values) if values else None


def latency_summary(values):
    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or not len(x) or not np.isfinite(x).all() or (x < 0).any(): raise ValueError('Invalid controller latencies')
    return dict(zip(('mean', 'p50', 'p95', 'p99', 'max'), map(float, [x.mean(), *np.quantile(x, [.5, .95, .99]), x.max()])))


def outcome(episode, initial_hash, takeover, latencies, original_score):
    n = len(episode['control_history'])
    if not 1 <= n <= 501 or any(len(episode[k]) != n for k in
            ('state_history', 'speed_history', 'jerk_history', 'acceleration_history', 'position_history')):
        raise ValueError('Invalid original episode history/bound')
    arrived, crashed = bool(episode['merged']), bool(episode['crashed'])
    if arrived and crashed or len(latencies) != n: raise ValueError('Inconsistent original outcome')
    if takeover is not None and (len(takeover) != n or any(type(x) is not bool for x in takeover)):
        raise ValueError('Supervisor decision count mismatch')
    if not math.isfinite(original_score): raise ValueError('Nonfinite author reward score')
    return {'arrival': int(arrived), 'collision': int(crashed), 'time_limit': int(not arrived and not crashed),
            'steps': n, 'simulation_duration_s': float(episode['simulation_time_taken']),
            'author_history_total_reward': float(original_score), 'author_history_reward_terms': max(n - 1, 0),
            'mean_speed_mps': finite_mean(episode['speed_history']),
            'mean_abs_jerk': finite_mean(abs(x) for x in episode['jerk_history']),
            'author_postmerge_closest_distance_m': min(episode['closest_vehicle_history']) if episode['closest_vehicle_history'] else None,
            'mean_disruption_decel_mps2': finite_mean(episode['disruption_history']),
            'st_takeover_rate': None if takeover is None else sum(takeover) / n,
            'initial_traffic_sha256': initial_hash, 'controller_latency_s': latency_summary(latencies),
            'controller_latency_samples_s': list(latencies)}


def parity(left, right):
    fields = ('crashed', 'merged', 'control_history', 'position_history', 'speed_history', 'acceleration_history',
              'jerk_history', 'closest_vehicle_history', 'disruption_history', 'simulation_time_taken')
    if any(left[k] != right[k] for k in fields): raise ValueError('Recording changed original author episode')
    return True


METRICS = ('arrival', 'collision', 'time_limit', 'author_history_total_reward', 'simulation_duration_s',
           'mean_speed_mps', 'mean_abs_jerk', 'author_postmerge_closest_distance_m', 'mean_disruption_decel_mps2', 'st_takeover_rate')


def summarize(rows):
    return {m: {'mean': finite_mean(r[m] for r in rows if r[m] is not None), 'count': sum(r[m] is not None for r in rows)}
            for m in METRICS} | {
        'successful_episode_time_s': {'mean': finite_mean(r['simulation_duration_s'] for r in rows if r['arrival']), 'count': sum(r['arrival'] for r in rows)},
        'successful_episode_mean_abs_jerk': {'mean': finite_mean(r['mean_abs_jerk'] for r in rows if r['arrival']), 'count': sum(r['arrival'] for r in rows)},
        'controller_latency_s': latency_summary([x for r in rows for x in r['controller_latency_samples_s']]),
        'controller_latency_count': sum(len(r['controller_latency_samples_s']) for r in rows)}


def aggregate(config, rows):
    expected = jobs(config)
    if len(rows) != len(expected) or len({r['id'] for r in rows}) != len(rows): raise ValueError('Missing or duplicate external episodes')
    index = {r['id']: r for r in rows}
    for j in expected:
        r = index.get(j['id'])
        if r is None or any(r[k] != v for k, v in j.items()): raise ValueError('Episode roster/lineage changed')
        if any(type(r[k]) is not int or r[k] not in (0, 1) for k in ('arrival', 'collision', 'time_limit')) or sum(
                r[k] for k in ('arrival', 'collision', 'time_limit')) != 1: raise ValueError('Invalid exclusive episode outcome')
    for scene in config['simulator_seeds']:
        if len({r['initial_traffic_sha256'] for r in rows if r['simulator_seed'] == scene}) != 1: raise ValueError('Methods did not start from identical scenes')
    arms = {}
    for arm in ARMS:
        selected = [r for r in rows if r['arm'] == arm]
        arms[arm] = {'independent_training_replicates': len(config['conditional_run_seeds']) if arm == 'conditional' else 0 if arm == 'author_st' else 1,
                     'episodes': len(selected), 'summary': summarize(selected),
                     'per_run_seed': {str(s): summarize([r for r in selected if r['run_seed'] == s])
                                      for s in (config['conditional_run_seeds'] if arm == 'conditional' else [None])}}
    comparisons = {}
    for arm in ARMS[:-1]:
        pairs = []
        for seed in config['conditional_run_seeds']:
            for scene in config['simulator_seeds']:
                left = next(r for r in rows if r['arm'] == arm and r['simulator_seed'] == scene)
                right = next(r for r in rows if r['arm'] == 'conditional' and r['run_seed'] == seed and r['simulator_seed'] == scene)
                delta = {m: None if left[m] is None or right[m] is None else right[m] - left[m] for m in METRICS if m != 'st_takeover_rate'}
                delta['time_s_both_successful'] = right['simulation_duration_s'] - left['simulation_duration_s'] if left['arrival'] and right['arrival'] else None
                pairs.append({'run_seed': seed, 'simulator_seed': scene, 'right_minus_left': delta})
        per_seed = {str(s): {m: {'mean': finite_mean(p['right_minus_left'][m] for p in pairs if p['run_seed'] == s and p['right_minus_left'][m] is not None),
                    'count': sum(p['run_seed'] == s and p['right_minus_left'][m] is not None for p in pairs)}
                    for m in pairs[0]['right_minus_left']} for s in config['conditional_run_seeds']}
        comparisons[arm] = {'left': arm, 'right': 'conditional', 'per_run_seed': per_seed, 'paired_rows': pairs,
            'equal_run_seed_mean_delta': {m: finite_mean(v[m]['mean'] for v in per_seed.values() if v[m]['mean'] is not None) for m in pairs[0]['right_minus_left']},
            'author_snapshot_reused_not_independent_replicates': arm != 'author_st'}
    return {'status': 'complete', 'engineering_complete': True, 'method_effect_gate': None,
            'analysis': 'development_paired_descriptive_no_winner_or_significance_gate',
            'episodes': len(rows), 'scenarios': len(config['simulator_seeds']), 'arms': arms, 'comparisons': comparisons,
            'budget_matched': False, 'formal_claim': False, 'test_opened': False, 'old_p7_episodes_reused': False}
