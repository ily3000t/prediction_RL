"""P7m: bounded frozen-actor sensitivity, NOT counterfactual driving outcomes."""
from copy import deepcopy
import hashlib
import heapq
import itertools
import math

import numpy as np
import torch

from prediction_rl.data.dataset_contract import digest
from prediction_rl.features.probe_features import DIMENSION, METADATA, NAMES, VERSION
from prediction_rl.training.all_ddpg import validate_action

ARMS = ('ordinary', 'conditional')
STRATA = ('low_speed', 'early_ramp', 'short_history', 'omitted_present')
PERTURBATIONS = ('zero_all_prediction', 'zero_values_keep_metadata', 'candidate_mean_keep_metadata')
PROTOCOL = {
    'version': 'p07m_replay_information_v1', 'purpose': 'read_only_information_utilization_diagnostic',
    'parent_request': 'artifacts/p7k/p7k_60k_v1/request.json',
    'parent_request_hash': 'c7e352fb4e760cfc918e7b69fbe81d6ac4cba1b6b93641ad8a12afe236c5c09a',
    'external_aggregate': 'artifacts/p7l/p7l_gym20_v1/aggregate.json',
    'external_aggregate_sha256': '7db4556335191e5617774fd047dd9ed682718c06b18dfd06e110caed84bdf33f',
    'external_request_hash': 'b9edb8adf4e8de45c0c04fdcbedc93c0f5a99df3ec4a39f25049c9fb0435eada',
    'arms': list(ARMS), 'run_seeds': [0, 1, 2], 'checkpoint_frames': 60000,
    'sample': {'uniform_cap': 512, 'stratum_cap': 128, 'scan_prefix': None,
               'selection': 'smallest_sha256_job_transition_id_v1',
               'strata': list(STRATA), 'low_speed_max_mps': .1, 'short_history_frames': 11},
    'perturbations': list(PERTURBATIONS), 'action_change_reference_mps3': .1,
    'batch_size': 64, 'equivalence_atol': 1e-5, 'equivalence_rtol': 1e-5,
    'device': 'cpu', 'torch_threads': 1, 'training_permitted': False,
    'simulation_permitted': False, 'test_opened': False, 'automatic_budget_extension': False,
}
AUDIT = deepcopy(PROTOCOL)
AUDIT.update(version='p07m_replay_information_audit_v1', purpose='bounded_read_only_engineering_smoke')
AUDIT['sample'].update(uniform_cap=8, stratum_cap=4, scan_prefix=128)


def validate_protocol(value):
    if digest(value) not in (digest(PROTOCOL), digest(AUDIT)):
        raise ValueError('Modified diagnostic design; create a reviewed new version')
    return value


def jobs(protocol):
    validate_protocol(protocol)
    return [{'id': f'{arm}_s{seed}', 'arm': arm, 'run_seed': seed,
             'checkpoint_frames': 60000} for arm in ARMS for seed in protocol['run_seeds']]


def validate_vector(row):
    """Check only the input identity/layout, not repeat the reward/bootstrap audit."""
    x = np.asarray(row['state'], dtype=np.float32)
    m = row['physical_before']; meta = m['feature_metadata']
    if (x.shape != (170,) or not np.isfinite(x).all() or type(row['transition_id']) is not int
            or row['transition_id'] < 1 or type(row['step']) is not int or row['step'] < 0
            or type(row['episode']) is not int or row['episode'] < 0 or m['ego'] is None
            or not math.isfinite(m['ego']['speed']) or type(m['history_frames']) is not int
            or not 1 <= m['history_frames'] <= 11 or set(meta) != set(METADATA)
            or digest(x[:169].tolist()) != m['observation_sha256']
            or x[169] != np.float32(row['step'])*np.float32(.001)
            or not np.array_equal(x[160:169], np.asarray([meta[k] for k in METADATA], np.float32))
            or (x[160:169] < 0).any() or (x[160:169] > 1).any()):
        raise ValueError('Saved augmented replay observation identity changed')
    for key in ('front_present', 'rear_present', 'pool_present', 'omitted_present'):
        if meta[key] not in (0., 1.): raise ValueError('Nonbinary presence metadata')
    role = x[20:140].reshape(5, 2, 4, 3)
    for i, key in enumerate(('front_present', 'rear_present')):
        if meta[key] == 0. and np.any(role[:, i] != 0.):
            raise ValueError('Absent role has nonzero forecast')
    if meta['pool_present'] == 0. and np.any(x[140:160] != 0.):
        raise ValueError('Absent pool has nonzero forecast')
    return x


def strata(row):
    m = row['physical_before']
    return {'low_speed': m['ego']['speed'] <= .1, 'early_ramp': m['ego']['lane_id'] == 'ramp_0',
            'short_history': m['history_frames'] < 11,
            'omitted_present': m['feature_metadata']['omitted_present'] == 1.}


def select_states(rows, job, protocol, expected_transitions):
    """Outcome-blind hash-priority reservoirs; bounded memory, no RNG draws.

    Uniform sample is the primary descriptive set. Strata oversample special
    states and MUST NOT be pooled as estimates of population frequencies.
    """
    validate_protocol(protocol); p = protocol['sample']
    caps = {'uniform': p['uniform_cap'], **{k: p['stratum_cap'] for k in STRATA}}
    heaps = {k: [] for k in caps}; counts = {k: 0 for k in caps}; scanned = 0
    stream = rows if p['scan_prefix'] is None else itertools.islice(rows, p['scan_prefix'])
    for row in stream:
        scanned += 1
        if row['transition_id'] != scanned: raise ValueError('Noncontiguous replay transition identity')
        validate_vector(row)
        eligible = {'uniform': True, **strata(row)}
        priority = int(hashlib.sha256(f"p07m-v1:{job['id']}:{scanned}".encode()).hexdigest(), 16)
        compact = {k: row[k] for k in ('transition_id', 'episode', 'step', 'state', 'physical_before')}
        for key, accept in eligible.items():
            if not accept: continue
            counts[key] += 1; item = (-priority, -scanned, compact)
            if len(heaps[key]) < caps[key]: heapq.heappush(heaps[key], item)
            elif item > heaps[key][0]: heapq.heapreplace(heaps[key], item)
    expected = expected_transitions if p['scan_prefix'] is None else min(expected_transitions, p['scan_prefix'])
    if scanned != expected or not scanned: raise ValueError('Incomplete or empty replay scan')
    memberships = {k: sorted(-item[1] for item in h) for k, h in heaps.items()}
    chosen = {item[2]['transition_id']: item[2] for h in heaps.values() for item in h}
    selected = [chosen[k] for k in sorted(chosen)]
    return selected, {'scanned_transitions': scanned, 'eligible_counts': counts,
        'selected_ids': memberships, 'unique_selected_states': len(selected),
        'selected_sha256': digest(selected), 'outcome_blind': True, 'no_rng_draws': True}


def perturb(x, variant):
    if (DIMENSION != 149 or VERSION != 'prediction_probe_features_v1'
            or len(NAMES) != 149 or tuple(NAMES[-9:]) != METADATA):
        raise ValueError('Feature layout changed')
    if x.dtype != torch.float32 or x.ndim != 2 or x.shape[1] != 170 or not torch.isfinite(x).all():
        raise ValueError('Need existing augmented float32 states; do not append another time feature')
    y = x.clone()
    if variant == 'zero_all_prediction': y[:, 20:169] = 0
    elif variant == 'zero_values_keep_metadata': y[:, 20:160] = 0
    elif variant == 'candidate_mean_keep_metadata':
        # Layout is 120 role values THEN 20 pool values, not five contiguous
        # 28-value candidate blocks. B2 also varies with candidate ego geometry.
        roles = x[:, 20:140].reshape(-1, 5, 2, 4, 3)
        pools = x[:, 140:160].reshape(-1, 5, 4)
        y[:, 20:140] = roles.mean(1, keepdim=True).expand_as(roles).reshape(-1, 120)
        y[:, 140:160] = pools.mean(1, keepdim=True).expand_as(pools).reshape(-1, 20)
    else: raise ValueError('Unknown perturbation')
    return y


def describe(values, reference):
    v = np.asarray(values, np.float64)
    if v.ndim != 1 or not np.isfinite(v).all(): raise ValueError('Invalid action deltas')
    if not len(v): return {'count': 0, 'mean': None, 'p50': None, 'p95': None, 'max': None, 'reference_exceedance_rate': None}
    return {'count': len(v), 'mean': float(v.mean()), 'p50': float(np.quantile(v, .5)),
            'p95': float(np.quantile(v, .95)), 'max': float(v.max()),
            'reference_exceedance_rate': float((v >= reference).mean())}


def actor_query(agent, features, batch_size):
    from all.environments import State
    out = []
    for x in features.split(batch_size):
        # Bypass TimeFeature: replay ALREADY contains original time coordinates.
        a = agent.agent.policy.eval(State(x, torch.ones(len(x), dtype=torch.uint8)))
        validate_action(a, len(x), -5., 5.); out.append(a.detach().cpu())
    return torch.cat(out).flatten()


def analyze(agent, selected, selection, protocol):
    validate_protocol(protocol)
    if not agent.evaluation_only or agent.body.timestep is not None:
        raise ValueError('Need strictly loaded, never-executed evaluation-only agent')
    x = torch.tensor([r['state'] for r in selected], dtype=torch.float32)
    rng = torch.get_rng_state().clone()
    before = {n: deepcopy(getattr(agent.agent, n).model.state_dict()) for n in ('policy', 'q')}
    targets = {n: deepcopy(getattr(agent.agent, n)._target._target.state_dict()) for n in ('policy', 'q')}
    guard = {n: (getattr(agent.agent, n)._updates, deepcopy(getattr(agent.agent, n)._scheduler.state_dict()))
             for n in ('policy', 'q')}
    baseline = actor_query(agent, x, protocol['batch_size'])
    single = actor_query(agent, x[:8], 1)
    if not torch.allclose(single, baseline[:8], atol=protocol['equivalence_atol'], rtol=protocol['equivalence_rtol']):
        raise ValueError('Batched/single original-network action disagreement')
    actions = {v: actor_query(agent, perturb(x, v), protocol['batch_size']) for v in PERTURBATIONS}
    if (not torch.equal(rng, torch.get_rng_state()) or agent.body.timestep is not None
            or len(agent.agent.replay_buffer) != 0 or agent.agent._frames_seen != 0
            or agent.agent._state is not None or agent.agent._action is not None):
        raise ValueError('Diagnostic advanced RNG, TimeFeature, replay or agent state')
    for name, saved in before.items():
        app = getattr(agent.agent, name)
        if (any(not torch.equal(v, app.model.state_dict()[k]) for k, v in saved.items())
                or any(not torch.equal(v, app._target._target.state_dict()[k]) for k, v in targets[name].items())
                or app._optimizer.state or (app._updates, app._scheduler.state_dict()) != guard[name]
                or any(p.grad is not None for p in app.model.parameters())):
            raise ValueError('Diagnostic modified weights/optimizer/scheduler/gradients')
    index = {r['transition_id']: i for i, r in enumerate(selected)}
    summary = {group: {v: describe([abs(float(actions[v][index[t]] - baseline[index[t]])) for t in ids],
                                  protocol['action_change_reference_mps3']) for v in PERTURBATIONS}
               for group, ids in selection['selected_ids'].items()}
    rows = [{'transition_id': r['transition_id'], 'episode': r['episode'], 'step': r['step'],
             'original_jerk': float(baseline[i]), 'perturbed_jerk': {v: float(actions[v][i]) for v in PERTURBATIONS}}
            for i, r in enumerate(selected)]
    return {'selection': selection, 'groups': summary, 'action_rows': rows,
            'no_updates_no_time_advance_no_rng_draws': True,
            'batched_single_max_absolute_difference': float((single-baseline[:8]).abs().max())}


LIMITATIONS = [
    'Sensitivity shows feature dependence, not improved return or causal safety benefit.',
    'Zeroing and candidate averaging can create out-of-distribution, inconsistent inputs.',
    'Candidate differences include ego-plan geometry in BOTH ordinary and conditional arms; this does not isolate neighbor action-conditioning.',
    'Each actor is queried on its OWN replay distribution; between-arm sensitivity is not a same-state paired treatment effect.',
    'Replay snapshots omit full neighbor future labels and evaluation input traces; failed episodes cannot be attributed to prediction errors here.',
    'Training replay states are correlated, not independent simulator scenarios or new held-out evidence.',
    'Samples span the entire exploratory training replay, not only the final frozen policy state distribution.',
]


def aggregate(protocol, rows):
    expected = jobs(protocol); index = {r['job']['id']: r for r in rows}
    if len(rows) != len(index) or set(index) != {j['id'] for j in expected}:
        raise ValueError('Missing/duplicate diagnostic models; keep every seed')
    for job in expected:
        r = index[job['id']]
        if r['job'] != job or r['no_updates_no_time_advance_no_rng_draws'] is not True:
            raise ValueError('Invalid model identity or read-only guard')
    return {'status': 'complete', 'engineering_complete': True, 'method_effect_gate': None,
            'models': [index[j['id']] for j in expected], 'limitations': LIMITATIONS,
            'training_permitted': False, 'simulation_episodes': 0, 'test_opened': False,
            'prediction_error_attribution': 'not_identifiable_from_saved_replay_vectors',
            'automatic_budget_extension': False, 'formal_claim': False}
