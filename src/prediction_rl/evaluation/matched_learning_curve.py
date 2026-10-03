"""One bounded, matched development experiment; checkpoints are NOT selected by results."""
from copy import deepcopy
import math
import statistics

from prediction_rl.data.dataset_contract import digest

PROTOCOL = {
    'version': 'p07k_matched_60k_v1',
    'purpose': 'bounded_matched_development_learning_curve',
    'arms': ['baseline', 'zero', 'ordinary', 'conditional'],
    'training': {'run_seeds': [0, 1, 2], 'checkpoint_frames': [20000, 40000, 60000],
                 'requested_frames': 60000, 'workers': 1, 'device': 'cpu',
                 'torch_threads': 1, 'learning_rate': .0002, 'warm_start': False,
                 'replay_start_size': 5000, 'minibatch_size': 100},
    'evaluation': {'simulator_seeds': list(range(200, 220)), 'workers': 2,
                   'episodes_per_seed': 1, 'policy': 'deterministic_no_exploration',
                   'protocol': 'gym20_fresh_process_first_episode'},
    'checkpoint_rule': 'first_completed_episode_at_or_after_requested_node',
    'selection': 'final_60k_primary_earlier_nodes_diagnostic_no_best_selection',
    'reset_contract': 'upstream_continuing_traffic_v1',
    'execution_contract': 'simulation_blocking_exact_v1',
    'event_contract': 'raw_ego_participant_and_route_endpoint_v2',
    'action_saturation_threshold': 4.99,
    'low_speed_mps': .1,
    'comparisons': [
        {'id': 'input_capacity', 'left': 'baseline', 'right': 'zero'},
        {'id': 'ordinary_information', 'left': 'zero', 'right': 'ordinary'},
        {'id': 'action_conditioning', 'left': 'ordinary', 'right': 'conditional'},
        {'id': 'total_augmentation', 'left': 'baseline', 'right': 'conditional'}],
    'aggregation': 'scene_paired_within_training_seed_then_equal_training_seed_mean',
    'from_scratch': True, 'automatic_budget_extension': False,
    'formal_claim': False, 'test_opened': False,
}
SMOKE = deepcopy(PROTOCOL)
SMOKE.update(version='p07k_workflow_smoke_v1', purpose='bounded_engineering_not_learning_evidence',
             selection='all_smoke_nodes_engineering_only')
SMOKE['training'].update(run_seeds=[0], checkpoint_frames=[16, 32, 64], requested_frames=64,
                         replay_start_size=8, minibatch_size=8)
SMOKE['evaluation'].update(simulator_seeds=[200], workers=1)
METRICS = ('return', 'natural_arrival', 'ego_collision', 'time_limit',
           'simulation_duration_s', 'low_speed_fraction', 'action_saturation_fraction',
           'mean_abs_measured_jerk', 'minimum_observed_front_bumper_distance_m')


def validate_protocol(value):
    if digest(value) not in (digest(PROTOCOL), digest(SMOKE)):
        raise ValueError('Changed matched design: freeze and approve a new version')
    return value


def jobs(protocol, stage):
    validate_protocol(protocol)
    training = [{'id': f'{a}_s{s}', 'arm': a, 'run_seed': s}
                for a in protocol['arms'] for s in protocol['training']['run_seeds']]
    if stage == 'train': return training
    if stage != 'evaluate': raise ValueError('Unknown stage')
    return [{**j, 'training_id': j['id'], 'checkpoint_frames': n,
             'simulator_seed': v, 'id': f'{j["id"]}_n{n}_v{v}'}
            for j in training for n in protocol['training']['checkpoint_frames']
            for v in protocol['evaluation']['simulator_seeds']]


def snapshot_steps(actual, requested):
    if type(actual) is not int or not requested <= actual <= requested+499:
        raise ValueError('Snapshot must retain original completed-episode budget')


def mean(values):
    values = [v for v in values if v is not None]
    return statistics.mean(values) if values else None


def aggregate(protocol, rows):
    validate_protocol(protocol)
    expected = jobs(protocol, 'evaluate'); lookup = {r['id']: r for r in rows}
    if len(rows) != len(lookup) or set(lookup) != {j['id'] for j in expected}:
        raise ValueError('Missing, duplicate or extra cells; never exclude seeds')
    roots = {}
    for j in expected:
        r = lookup[j['id']]
        if any(r.get(k) != v for k, v in j.items()) or r.get('no_policy_updates') is not True:
            raise ValueError('Evaluation identity or frozen policy changed')
        snapshot_steps(r['actual_training_steps'], j['checkpoint_frames'])
        flags = [r[k] for k in ('natural_arrival', 'ego_collision', 'time_limit')]
        if (any(type(v) is not int or v not in (0, 1) for v in flags) or sum(flags) != 1
                or type(r['steps']) is not int or not 1 <= r['steps'] <= 500
                or any(r[m] is not None and not math.isfinite(r[m]) for m in METRICS)
                or any(not 0 <= r[k] <= 1 for k in ('low_speed_fraction', 'action_saturation_fraction'))
                or r.get('native_event_agrees') is not True):
            raise ValueError('Invalid or unaudited outcome')
        h = r['initial_traffic_sha256']; v = j['simulator_seed']
        if not isinstance(h, str) or len(h) != 64: raise ValueError('Missing initial scene hash')
        if v in roots and roots[v] != h: raise ValueError('Same seed has different initial traffic')
        roots[v] = h
    nodes = []
    seeds = protocol['training']['run_seeds']
    for n in protocol['training']['checkpoint_frames']:
        selected = [r for r in rows if r['checkpoint_frames'] == n]
        arms = {}
        for a in protocol['arms']:
            per_seed = {str(s): {m: mean(r[m] for r in selected if r['arm'] == a and r['run_seed'] == s)
                                for m in METRICS} for s in seeds}
            arms[a] = {'per_run_seed': per_seed,
                       'equal_run_seed_mean': {m: mean(v[m] for v in per_seed.values()) for m in METRICS},
                       'actual_steps_by_run_seed': {str(s): sorted({r['actual_training_steps'] for r in selected
                                                                  if r['arm'] == a and r['run_seed'] == s}) for s in seeds}}
        comparisons = {}
        for c in protocol['comparisons']:
            pairs = []
            for s in seeds:
                for v in protocol['evaluation']['simulator_seeds']:
                    l = lookup[f'{c["left"]}_s{s}_n{n}_v{v}']; r = lookup[f'{c["right"]}_s{s}_n{n}_v{v}']
                    pairs.append({'run_seed': s, 'simulator_seed': v, 'right_minus_left': {
                        m: None if l[m] is None or r[m] is None else r[m]-l[m] for m in METRICS}})
            per_seed = {str(s): {m: mean(p['right_minus_left'][m] for p in pairs if p['run_seed'] == s)
                                 for m in METRICS} for s in seeds}
            comparisons[c['id']] = {**c, 'paired_rows': pairs, 'per_run_seed_delta': per_seed,
                'equal_run_seed_mean_delta': {m: mean(v[m] for v in per_seed.values()) for m in METRICS},
                'run_seed_std': {m: statistics.stdev(v[m] for v in per_seed.values() if v[m] is not None)
                                if sum(v[m] is not None for v in per_seed.values()) > 1 else None for m in METRICS}}
        nodes.append({'requested_training_steps': n, 'arms': arms, 'comparisons': comparisons})
    return {'status': 'complete', 'engineering_complete': True, 'method_effect_gate': None,
            'nodes': nodes, 'primary_node': protocol['training']['requested_frames'],
            'checkpoint_selection_by_development_results': False,
            'training_replicates': len(seeds), 'unique_traffic_scenarios': len(roots),
            'evaluation_episodes': len(rows), 'repeated_nodes_are_not_independent_scenes': True,
            'author50_results_mixed': False, 'formal_claim': False, 'test_opened': False,
            'analysis': 'paired_development_descriptive_not_formal_significance_or_convergence_proof'}
