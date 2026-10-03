"""Frozen-policy exploration diagnostic, never a training or method-effect gate."""
from copy import deepcopy
import math
import statistics

import torch

from prediction_rl.data.dataset_contract import digest
from prediction_rl.evaluation.closed_loop_diagnostics import PROTOCOL as P7B

PROTOCOL = {
    'version': 'p07i_ddpg_stability_v1',
    'purpose': 'development_frozen_policy_noise_and_training_artifact_diagnosis',
    'parent_request': P7B['parent_request'],
    'parent_request_hash': P7B['parent_request_hash'],
    'parent_aggregate_sha256': P7B['parent_aggregate_sha256'],
    'recording_request': 'artifacts/p7b/p7b_diag_v1/request.json',
    'recording_request_hash': '98d221e3b8ee24aab965189cc2e2acd5db1f3dd4f7a6b6bf1789dc9fcae91f5f',
    'recording_summary_sha256': '9cd83601e61ac2dd736cc48293d479b9c2dca8f1b72f50b870645ed9c051b040',
    'arms': ['baseline', 'zero', 'ordinary', 'conditional'],
    'run_seeds': [0, 1, 2], 'simulator_seeds': [200, 210, 219],
    'noise_repeats': [0, 1, 2], 'conditions': ['deterministic', 'upstream_noise'],
    'evaluation_protocol': 'original_p7_gym20_not_author50',
    'scene_selection': 'unchanged_p7b_first_midpoint_last',
    'noise': 'original_all_gaussian_then_original_action_bounds_clip',
    'noise_rng': 'isolated_cpu_torch_generator_common_scene_repeat_prefix_v1',
    'execution_contract': 'simulation_blocking_exact_v1',
    'workers': 2, 'torch_threads': 1, 'max_control_steps': 500,
    'low_speed_mps_descriptive': 0.1,
    'training_windows_steps': [0, 5000, 10000, 15000, 20500],
    'training_permitted': False, 'test_opened': False, 'formal_claim': False,
    'noise_is_not_a_deployment_policy': True,
    'deterministic_parity': 'exact_original_p7_actions_rewards_outcomes',
    'events': 'raw_ego_participant_and_route_endpoint_v2',
    'aggregation': 'repeat_mean_then_scene_mean_then_equal_training_seed_mean',
    'checkpoint_selection': 'all_original_final_checkpoints_no_selection',
}
AUDIT = deepcopy(PROTOCOL)
AUDIT.update(version='p07i_ddpg_stability_audit_v1',
             purpose='bounded_eight_episode_engineering_audit',
             run_seeds=[2], simulator_seeds=[200], noise_repeats=[0], workers=1,
             scene_selection='four_arms_seed2_scene200_instrumentation_not_effect_evidence')


def validate_protocol(config):
    if digest(config) not in (digest(PROTOCOL), digest(AUDIT)):
        raise ValueError('Changed stability design; approve a new immutable version')
    return config


def roster(config, parent):
    validate_protocol(config)
    selected = [j for j in parent['evaluation_jobs'] if j['arm'] in config['arms']
                and j['run_seed'] in config['run_seeds'] and j['simulator_seed'] in config['simulator_seeds']]
    if len(selected) != len(config['arms'])*len(config['run_seeds'])*len(config['simulator_seeds']):
        raise ValueError('Missing frozen checkpoint/scene roster')
    if config == PROTOCOL:
        seeds = parent['protocol']['evaluation']['simulator_seeds']
        if config['simulator_seeds'] != [seeds[0], seeds[len(seeds)//2], seeds[-1]]:
            raise ValueError('Changed outcome-independent scene selection')
    return [{**j, 'parent_id': j['id'], 'id': j['id']+('_d' if mode == 'deterministic' else f'_n{repeat}'),
             'condition': mode, 'noise_repeat': repeat}
            for j in selected for mode in config['conditions']
            for repeat in ([None] if mode == 'deterministic' else config['noise_repeats'])]


class FrozenGaussian:
    """Only this generator advances; never call training act() or sample global RNG."""
    def __init__(self, low, high, noise, scene, repeat):
        if (low, high, noise) != (-5., 5., .1) or type(scene) is not int or type(repeat) is not int:
            raise ValueError('Original DDPG exploration contract changed')
        self.low, self.high = low, high
        self.std = noise*(high-low)/2
        self.seed = int(digest({'version': PROTOCOL['noise_rng'], 'scene': scene, 'repeat': repeat})[:16], 16) % (2**63)
        self.generator = torch.Generator(device='cpu').manual_seed(self.seed)
        self.draws = 0

    def apply(self, action):
        if (action.device.type != 'cpu' or action.dtype != torch.float32 or action.shape != (1, 1)
                or not torch.isfinite(action).all() or (action < self.low).any() or (action > self.high).any()):
            raise ValueError('Invalid original deterministic action')
        noise = torch.randn(action.shape, generator=self.generator, dtype=torch.float32)*self.std
        before_clip = action+noise
        # Preserve the installed ALL choose-action min/high then max/low order.
        result = torch.max(torch.min(before_clip, torch.tensor(self.high)), torch.tensor(self.low))
        self.draws += 1
        return result, {'noise': float(noise.item()), 'unclipped_jerk': float(before_clip.item()),
                        'clipped': bool((before_clip != result).any())}


def scalar_summary(rows):
    """Raw ALL scalar windows: q loss is TD MSE; policy loss is negative Q, not accuracy."""
    if not rows:
        return {'count': 0, 'mean': None, 'first': None, 'last': None, 'min': None, 'max': None}
    if any(type(r['step']) is not int or not math.isfinite(r['value']) for r in rows):
        raise ValueError('Invalid scalar training record')
    values = [r['value'] for r in rows]
    return {'count': len(values), 'mean': statistics.mean(values), 'first': values[0],
            'last': values[-1], 'min': min(values), 'max': max(values)}


def training_windows(rows, edges):
    if any(b['step'] < a['step'] for a, b in zip(rows, rows[1:])):
        raise ValueError('Training scalar steps went backwards')
    return {f'{left}_{right}': scalar_summary([r for r in rows if left <= r['step'] < right])
            for left, right in zip(edges, edges[1:])}


def control_summary(trace, discount, low_speed):
    if not trace or len(trace) > 500 or not trace[-1]['done'] or any(r['done'] for r in trace[:-1]):
        raise ValueError('Only a complete bounded trajectory can be analyzed')
    low = [r for r in trace if r['speed_before_mps'] <= low_speed]
    run = longest = 0
    for i, r in enumerate(trace):
        if r['step'] != i or any(not math.isfinite(r[k]) for k in
                ('actor_jerk', 'requested_jerk', 'reward', 'speed_before_mps', 'q_actor', 'q_requested')):
            raise ValueError('Invalid recorded control/critic query')
        run = run+1 if r['speed_before_mps'] <= low_speed else 0
        longest = max(longest, run)
    realized = [0.]*len(trace)
    suffix = 0.
    for i in reversed(range(len(trace))):
        suffix = trace[i]['reward']+discount*suffix
        realized[i] = suffix
    error = [r['q_requested']-g for r, g in zip(trace, realized)]
    return {'steps': len(trace), 'low_speed_samples': len(low),
        'low_speed_fraction': len(low)/len(trace), 'longest_low_speed_sample_grid_s': .2*longest,
        'low_speed_actor_negative_jerk_fraction': None if not low else statistics.mean(r['actor_jerk'] < 0 for r in low),
        'low_speed_requested_negative_jerk_fraction': None if not low else statistics.mean(r['requested_jerk'] < 0 for r in low),
        'action_clip_fraction': statistics.mean(r['clipped'] for r in trace),
        'mean_abs_noise': statistics.mean(abs(r['noise']) for r in trace),
        'mean_abs_actor_to_requested_change': statistics.mean(abs(r['actor_jerk']-r['requested_jerk']) for r in trace),
        'q_actor_mean': statistics.mean(r['q_actor'] for r in trace),
        'q_requested_mean': statistics.mean(r['q_requested'] for r in trace),
        'low_speed_q_requested_minus_actor_mean': None if not low else statistics.mean(r['q_requested']-r['q_actor'] for r in low),
        'discounted_executed_return_from_start': realized[0],
        'q_requested_minus_realized_suffix_mean': statistics.mean(error),
        'q_requested_minus_realized_suffix_mae': statistics.mean(abs(x) for x in error),
        'critic_query_is_not_executed_or_action_selection': True,
        'suffix_error_is_descriptive_not_bellman_loss_or_calibration': True}


METRICS = ('return', 'natural_arrival', 'ego_collision', 'time_limit', 'other_terminal_event', 'low_speed_fraction',
           'longest_low_speed_sample_grid_s', 'low_speed_requested_negative_jerk_fraction',
           'simulation_duration_s', 'mean_abs_measured_jerk')


def mean(values):
    values = [v for v in values if v is not None]
    return statistics.mean(values) if values else None


def aggregate(config, parent, rows):
    expected = roster(config, parent)
    index = {r['id']: r for r in rows}
    if len(index) != len(rows) or set(index) != {j['id'] for j in expected}:
        raise ValueError('Missing/extra/duplicate cells; never discard failed training seeds')
    for j in expected:
        r = index[j['id']]
        if (any(r.get(k) != v for k, v in j.items()) or r['no_policy_updates'] is not True
                or (j['condition'] == 'deterministic' and r['exact_parent_parity'] is not True)):
            raise ValueError('Frozen policy identity/parity changed')
        events = [r[k] for k in ('natural_arrival', 'ego_collision', 'time_limit', 'other_terminal_event')]
        if (any(type(v) is not int or v not in (0, 1) for v in events) or sum(events) != 1 or
                any(r[m] is not None and not math.isfinite(r[m]) for m in METRICS)):
            raise ValueError('Invalid event denominator/finite diagnostic metric')
    for scene in config['simulator_seeds']:
        if len({r['initial_traffic_sha256'] for r in rows if r['simulator_seed'] == scene}) != 1:
            raise ValueError('Diagnostic modes did not start from the same traffic')
    pairs = []
    for d in rows:
        if d['condition'] != 'deterministic':
            continue
        noise = [index[d['parent_id']+f'_n{n}'] for n in config['noise_repeats']]
        avg = {m: mean(r[m] for r in noise) for m in METRICS}
        pairs.append({'arm': d['arm'], 'run_seed': d['run_seed'], 'simulator_seed': d['simulator_seed'],
                      'deterministic': {m: d[m] for m in METRICS}, 'noise_repeat_mean': avg,
                      'noise_minus_deterministic': {m: None if avg[m] is None or d[m] is None else avg[m]-d[m] for m in METRICS}})
    by_arm = {}
    for arm in config['arms']:
        seeds = {}
        for seed in config['run_seeds']:
            cells = [p for p in pairs if p['arm'] == arm and p['run_seed'] == seed]
            seeds[str(seed)] = {mode: {m: mean(c[mode][m] for c in cells) for m in METRICS}
                               for mode in ('deterministic', 'noise_repeat_mean', 'noise_minus_deterministic')}
        by_arm[arm] = {'per_run_seed': seeds, 'equal_run_seed_mean':
            {mode: {m: mean(s[mode][m] for s in seeds.values()) for m in METRICS}
             for mode in ('deterministic', 'noise_repeat_mean', 'noise_minus_deterministic')}}
    return {'status': 'complete', 'engineering_complete': True, 'episodes': len(rows),
            'independent_simulator_scenes': len(config['simulator_seeds']), 'arms': by_arm,
            'paired_cells': pairs, 'rows': rows, 'training_started': False, 'test_opened': False,
            'formal_claim': False, 'method_effect_gate': None,
            'interpretation': 'frozen_policy_noise_sensitivity_not_author50_method_ranking_or_noise_policy_proposal'}
