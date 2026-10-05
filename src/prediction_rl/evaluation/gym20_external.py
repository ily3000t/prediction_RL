"""P7l: frozen author controllers versus the existing 60k Gym20 endpoint.

This is an adapted development comparison, not Author50 reproduction or a
budget-matched estimate of the action-conditioning effect.
"""
from copy import deepcopy
import math
import statistics

from prediction_rl.data.dataset_contract import digest
from prediction_rl.evaluation.external_baselines import CONFIGS, ASSETS
from prediction_rl.evaluation.matched_learning_curve import METRICS

AUTHORS = ('author_ddpg', 'author_st', 'author_rl_mpc_safety', 'author_rl_mpc_switching')
PROJECT = ('baseline', 'zero', 'ordinary', 'conditional')
PROTOCOL = {
    'version': 'p07l_gym20_external_v1', 'purpose': 'adapted_external_development_comparison',
    'parent_request': 'artifacts/p7k/p7k_60k_v1/request.json',
    'parent_request_hash': 'c7e352fb4e760cfc918e7b69fbe81d6ac4cba1b6b93641ad8a12afe236c5c09a',
    'parent_aggregate_sha256': 'c26b5b8ac3371376097bd26c901b9a4409ed9cc6ae510a7080713a1ccb788b91',
    'author_controllers': list(AUTHORS), 'project_arms': list(PROJECT),
    'project_run_seeds': [0, 1, 2], 'simulator_seeds': list(range(200, 220)),
    'checkpoint_frames': 60000, 'checkpoint_selection': 'unchanged_60k_primary_no_selection',
    'episode_driver': 'upstream_gym20_with_native_speed_controller_adapter_v1',
    'warmup_s': 20, 'max_control_calls': 500, 'tick_s': .2,
    'scene_execution': 'fresh_process_first_episode_per_simulator_seed',
    'reward_score': 'upstream_gym_transition_slotted_jerk_invalid_penalty_zero',
    'event_contract': 'raw_ego_participant_and_route_endpoint_v2',
    'planner_action_contract': 'original_commanded_speed_no_jerk_projection_or_clipping',
    'rl_mpc_queries': 'original_virtual_rollout_order_and_TimeFeature_queries',
    'execution_contract': 'simulation_blocking_exact_v1', 'device': 'cpu',
    'torch_threads': 1, 'workers': 2, 'training_permitted': False,
    'budget_matched': False, 'author50_results_mixed': False,
    'formal_claim': False, 'test_opened': False,
}
AUDIT = deepcopy(PROTOCOL)
AUDIT.update(version='p07l_gym20_external_audit_v1',
             purpose='bounded_adapter_parity_not_method_evidence',
             simulator_seeds=[200], workers=1)


def validate_protocol(value):
    if digest(value) not in (digest(PROTOCOL), digest(AUDIT)):
        raise ValueError('Changed external Gym20 design: approve a new version')
    return value


def jobs(protocol):
    validate_protocol(protocol)
    return [{'id': f'{a}_v{v}', 'arm': a, 'run_seed': None,
             'simulator_seed': v, 'config': CONFIGS[a]}
            for a in AUTHORS for v in protocol['simulator_seeds']]


def project_job(arm, seed, scene):
    return {'id': f'project_{arm}_s{seed}_v{scene}', 'arm': arm,
            'run_seed': seed, 'simulator_seed': scene, 'checkpoint_frames': 60000}


def summarize(rows):
    def avg(values):
        values = [v for v in values if v is not None]
        return statistics.mean(values) if values else None
    return {m: avg(r[m] for r in rows) for m in METRICS} | {
        'natural_arrival_duration_s': avg(r['simulation_duration_s'] for r in rows if r['natural_arrival']),
        'natural_arrival_duration_count': sum(r['natural_arrival'] for r in rows),
        'st_takeover_rate': avg(r.get('st_takeover_rate') for r in rows),
    }


def aggregate(protocol, authors, project):
    validate_protocol(protocol)
    expected = jobs(protocol) + [project_job(a, s, v) for a in PROJECT
        for s in protocol['project_run_seeds'] for v in protocol['simulator_seeds']]
    rows = authors + project
    index = {r['id']: r for r in rows}
    if len(rows) != len(index) or set(index) != {j['id'] for j in expected}:
        raise ValueError('Missing, duplicate or extra cells; never discard failures')
    scenes = {}
    for j in expected:
        r = index[j['id']]
        if (any(r.get(k) != v for k, v in j.items()) or r.get('no_policy_updates') is not True
                or r.get('native_event_agrees') is not True or r.get('adapter_parity') is not True):
            raise ValueError('Changed identity, unaudited event or controller parity')
        flags = [r[k] for k in ('natural_arrival', 'ego_collision', 'time_limit')]
        if (any(type(v) is not int or v not in (0, 1) for v in flags) or sum(flags) != 1
                or type(r['steps']) is not int or not 1 <= r['steps'] <= 500
                or any(r[m] is not None and not math.isfinite(r[m]) for m in METRICS)
                or not 0 <= r['low_speed_fraction'] <= 1):
            raise ValueError('Invalid outcome')
        if j['arm'] in AUTHORS:
            expected_policy = None if j['arm'] == 'author_st' else ASSETS['policy.pt']
            if r.get('policy_sha256') != expected_policy:
                raise ValueError('Author policy snapshot changed')
        elif (type(r.get('actual_training_steps')) is not int
                or not 60000 <= r['actual_training_steps'] <= 60499):
            raise ValueError('Reused snapshot is not the frozen 60k endpoint')
        saturation = r['action_saturation_fraction']
        if (saturation is None) != (j['arm'] in AUTHORS and j['arm'] != 'author_ddpg'):
            raise ValueError('Do not invent jerk-action saturation for speed controllers')
        if saturation is not None and not 0 <= saturation <= 1:
            raise ValueError('Invalid action saturation')
        h = r['initial_traffic_sha256']; v = j['simulator_seed']
        if not isinstance(h, str) or len(h) != 64 or (v in scenes and scenes[v] != h):
            raise ValueError('Methods did not begin in identical Gym20 traffic')
        scenes[v] = h
    matrix = {}
    for a in (*AUTHORS, *PROJECT):
        selected = [r for r in rows if r['arm'] == a]
        matrix[a] = {'episodes': len(selected), 'summary': summarize(selected),
            'independent_training_replicates': 0 if a == 'author_st' else 1 if a in AUTHORS else 3,
            'per_run_seed': {str(s): summarize([r for r in selected if r['run_seed'] == s])
                for s in ([None] if a in AUTHORS else protocol['project_run_seeds'])}}
    comparisons = {}
    for a in AUTHORS:
        for b in PROJECT:
            pairs = []
            for s in protocol['project_run_seeds']:
                for v in protocol['simulator_seeds']:
                    l = index[f'{a}_v{v}']; r = index[project_job(b, s, v)['id']]
                    delta = {m: None if l[m] is None or r[m] is None else r[m]-l[m] for m in METRICS}
                    delta['duration_s_both_natural_arrival'] = (
                        r['simulation_duration_s']-l['simulation_duration_s']
                        if l['natural_arrival'] and r['natural_arrival'] else None)
                    pairs.append({'run_seed': s, 'simulator_seed': v, 'right_minus_left': delta})
            per_seed = {str(s): {m: statistics.mean(x['right_minus_left'][m] for x in pairs
                if x['run_seed'] == s and x['right_minus_left'][m] is not None)
                if any(x['run_seed'] == s and x['right_minus_left'][m] is not None for x in pairs) else None
                for m in pairs[0]['right_minus_left']} for s in protocol['project_run_seeds']}
            comparisons[f'{a}__vs__{b}'] = {'left': a, 'right': b, 'paired_rows': pairs,
                'per_run_seed_delta': per_seed,
                'equal_run_seed_mean_delta': {m: statistics.mean(x[m] for x in per_seed.values() if x[m] is not None)
                    if any(x[m] is not None for x in per_seed.values()) else None for m in pairs[0]['right_minus_left']},
                'author_snapshot_repeated_not_independent_replicates': a != 'author_st'}
    return {'status': 'complete', 'engineering_complete': True, 'method_effect_gate': None,
        'matrix': matrix, 'comparisons': comparisons, 'unique_traffic_scenarios': len(scenes),
        'new_author_episodes': len(authors), 'reused_project_episodes': len(project),
        'budget_matched': False, 'author50_results_mixed': False,
        'analysis': 'adapted_Gym20_paired_development_descriptive_not_formal_significance',
        'training_performed': False, 'formal_claim': False, 'test_opened': False}
