"""One new B3 arm; frozen P7n arms reused, no best-node or winning-seed selection."""
from copy import deepcopy
import statistics

from prediction_rl.data.dataset_contract import digest
from .fixed_budget_extension import PROTOCOL as REFERENCE, aggregate as reference_aggregate
from .matched_learning_curve import SMOKE as KERNEL_SMOKE, METRICS, mean

ARM = 'conditional_difference'
PROTOCOL = deepcopy(REFERENCE)
PROTOCOL.update(version='candidate_difference_ddpg_120k_v1', arms=[ARM],
    reference_arms=list(REFERENCE['arms']),
    comparisons=[{'id':'difference_objective', 'left':'conditional', 'right':ARM},
                 {'id':'versus_ordinary', 'left':'ordinary', 'right':ARM},
                 {'id':'versus_baseline', 'left':'baseline', 'right':ARM}])
SMOKE = deepcopy(PROTOCOL)
SMOKE.update(version='candidate_difference_ddpg_smoke_v1', purpose='bounded_engineering_not_learning_evidence',
             selection='all_smoke_nodes_engineering_only')
SMOKE['training'] = deepcopy(KERNEL_SMOKE['training'])
SMOKE['evaluation'] = deepcopy(KERNEL_SMOKE['evaluation'])


def validate_protocol(p):
    if digest(p) not in (digest(PROTOCOL), digest(SMOKE)):
        raise ValueError('Difference DDPG design changed; no budget/seed/reward adaptation')
    return p


def jobs(p, stage):
    validate_protocol(p)
    training = [{'id':f'{ARM}_s{s}', 'arm':ARM, 'run_seed':s} for s in p['training']['run_seeds']]
    if stage == 'train': return training
    if stage != 'evaluate': raise ValueError('Unknown stage')
    return [{**j, 'training_id':j['id'], 'checkpoint_frames':n, 'simulator_seed':v,
             'id':f'{j["id"]}_n{n}_v{v}'} for j in training
            for n in p['training']['checkpoint_frames'] for v in p['evaluation']['simulator_seeds']]


def aggregate(p, new_rows, reference_rows):
    if validate_protocol(p) != PROTOCOL: raise ValueError('Smoke is not method evidence')
    expected = jobs(p,'evaluate'); lookup = {r['id']:r for r in new_rows}
    if len(lookup) != len(new_rows) or set(lookup) != {j['id'] for j in expected}:
        raise ValueError('Missing, duplicate or extra difference cells')
    for j in expected:
        if any(lookup[j['id']].get(k) != v for k,v in j.items()):
            raise ValueError('Difference evaluation identity changed')
    old = reference_aggregate(REFERENCE, reference_rows)
    # Reuse the exact frozen validation/statistics kernel, substituting ONLY its
    # conditional cells in a local copy. Stored historical rows are never edited.
    remapped = []
    for row in new_rows:
        r = deepcopy(row); r['arm']='conditional'; r['training_id']=f'conditional_s{r["run_seed"]}'
        r['id']=f'{r["training_id"]}_n{r["checkpoint_frames"]}_v{r["simulator_seed"]}'
        remapped.append(r)
    substituted = reference_aggregate(REFERENCE,
        [deepcopy(r) for r in reference_rows if r['arm'] != 'conditional'] + remapped)
    old_lookup = {r['id']:r for r in reference_rows}; nodes=[]
    for previous, current in zip(old['nodes'], substituted['nodes']):
        n=previous['requested_training_steps']; comparisons={}
        for c in p['comparisons']:
            pairs=[]
            for s in p['training']['run_seeds']:
                for v in p['evaluation']['simulator_seeds']:
                    left=old_lookup[f'{c["left"]}_s{s}_n{n}_v{v}']
                    right=lookup[f'{ARM}_s{s}_n{n}_v{v}']
                    pairs.append({'run_seed':s,'simulator_seed':v,'right_minus_left':{
                        m:None if left[m] is None or right[m] is None else right[m]-left[m] for m in METRICS}})
            per_seed={str(s):{m:mean(r['right_minus_left'][m] for r in pairs if r['run_seed']==s)
                             for m in METRICS} for s in p['training']['run_seeds']}
            comparisons[c['id']]={**c,'paired_rows':pairs,'per_run_seed_delta':per_seed,
                'equal_run_seed_mean_delta':{m:mean(r[m] for r in per_seed.values()) for m in METRICS},
                'run_seed_std':{m:statistics.stdev(r[m] for r in per_seed.values() if r[m] is not None)
                    if sum(r[m] is not None for r in per_seed.values())>1 else None for m in METRICS}}
        nodes.append({'requested_training_steps':n,
            'arms':{**previous['arms'],ARM:current['arms']['conditional']}, 'comparisons':comparisons})
    return {**old,'nodes':nodes,'new_training_instances':3,'new_evaluation_episodes':len(new_rows),
        'reused_evaluation_episodes':len(reference_rows),'evaluation_episodes':len(new_rows)+len(reference_rows),
        'reference_results_are_reused_not_retrained':True,'method_effect_gate':None,
        'difference_effect_established':False}
