"""P7n: one approved 120k development endpoint, never best-node selection."""
from copy import deepcopy
import math
import statistics

from prediction_rl.data.dataset_contract import digest
from prediction_rl.evaluation.matched_learning_curve import (
    PROTOCOL as PREVIOUS, SMOKE, METRICS, mean, snapshot_steps)

PROTOCOL = deepcopy(PREVIOUS)
PROTOCOL.update(version='p07n_matched_120k_v1',
                selection='final_120k_primary_earlier_nodes_diagnostic_no_best_selection')
PROTOCOL['training'].update(checkpoint_frames=[20000, 40000, 60000, 120000], requested_frames=120000)
DECISION = {
    'request': 'artifacts/p7m/p7m_info_v1/request.json',
    'request_hash': '80d3c140e81445c4a50ef1103fb1cb504836263adb16d1198790e43fbe924318',
    'aggregate_sha256': 'c566bc97c86cde432259ec85f2b8a4a9b0f072c5b72dce29457ea9b91ee1598e',
}


def validate_protocol(value):
    # Reuse the EXACT historical engineering-only smoke design/kernel. It is
    # never relabeled as a 120k learning result and lives under a new request.
    if digest(value) not in (digest(PROTOCOL), digest(SMOKE)):
        raise ValueError('Fixed 120k design changed; no automatic extension or seed selection')
    return value


def jobs(protocol, stage):
    validate_protocol(protocol)
    training = [{'id': f'{a}_s{s}', 'arm': a, 'run_seed': s}
                for a in protocol['arms'] for s in protocol['training']['run_seeds']]
    if stage == 'train': return training
    if stage != 'evaluate': raise ValueError('Unknown stage')
    return [{**j, 'training_id': j['id'], 'checkpoint_frames': n, 'simulator_seed': v,
             'id': f'{j["id"]}_n{n}_v{v}'} for j in training
            for n in protocol['training']['checkpoint_frames'] for v in protocol['evaluation']['simulator_seeds']]


def validate_replay_capacity(protocol, preset):
    validate_protocol(protocol)
    # Historical instrumentation guards >60,499. Add the stronger new budget
    # preflight WITHOUT altering that frozen source or actual replay behavior.
    if (preset['replay_buffer_size'] <= protocol['training']['requested_frames']+499
            or preset['last_frame'] != 2e6 or preset['device'] != 'cpu'):
        raise ValueError('Original replay must not wrap; retain the original 2M scheduler')


def aggregate(protocol, rows):
    """Same P7k scene/seed statistics, extended roster and new primary node."""
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
    nodes = []; seeds = protocol['training']['run_seeds']
    for n in protocol['training']['checkpoint_frames']:
        selected = [r for r in rows if r['checkpoint_frames'] == n]; arms = {}
        for arm in protocol['arms']:
            per_seed = {str(s): {m: mean(r[m] for r in selected if r['arm'] == arm and r['run_seed'] == s)
                                for m in METRICS} for s in seeds}
            arms[arm] = {'per_run_seed': per_seed,
                'equal_run_seed_mean': {m: mean(v[m] for v in per_seed.values()) for m in METRICS},
                'actual_steps_by_run_seed': {str(s): sorted({r['actual_training_steps'] for r in selected
                    if r['arm'] == arm and r['run_seed'] == s}) for s in seeds}}
        comparisons = {}
        for c in protocol['comparisons']:
            pairs = []
            for s in seeds:
                for v in protocol['evaluation']['simulator_seeds']:
                    left = lookup[f'{c["left"]}_s{s}_n{n}_v{v}']; right = lookup[f'{c["right"]}_s{s}_n{n}_v{v}']
                    pairs.append({'run_seed': s, 'simulator_seed': v, 'right_minus_left': {
                        m: None if left[m] is None or right[m] is None else right[m]-left[m] for m in METRICS}})
            per_seed = {str(s): {m: mean(p['right_minus_left'][m] for p in pairs if p['run_seed'] == s)
                                 for m in METRICS} for s in seeds}
            comparisons[c['id']] = {**c, 'paired_rows': pairs, 'per_run_seed_delta': per_seed,
                'equal_run_seed_mean_delta': {m: mean(v[m] for v in per_seed.values()) for m in METRICS},
                'run_seed_std': {m: statistics.stdev(v[m] for v in per_seed.values() if v[m] is not None)
                    if sum(v[m] is not None for v in per_seed.values()) > 1 else None for m in METRICS}}
        nodes.append({'requested_training_steps': n, 'arms': arms, 'comparisons': comparisons})
    return {'status': 'complete', 'engineering_complete': True, 'method_effect_gate': None,
        'nodes': nodes, 'primary_node': protocol['training']['requested_frames'],
        'checkpoint_selection_by_development_results': False, 'training_replicates': len(seeds),
        'unique_traffic_scenarios': len(roots), 'evaluation_episodes': len(rows),
        'repeated_nodes_are_not_independent_scenes': True, 'author50_results_mixed': False,
        'formal_claim': False, 'test_opened': False,
        'analysis': 'paired_development_descriptive_not_formal_significance_or_convergence_proof'}
