"""Frozen matched design; negative effects complete normally, never an engineering gate."""
from copy import deepcopy
from types import FunctionType
import statistics
import math

from prediction_rl.data.dataset_contract import digest
from prediction_rl.evaluation import matched_learning_curve as previous

PROTOCOL = deepcopy(previous.PROTOCOL)
PROTOCOL.update(version='rcci04c_matched_policy_v1', purpose='matched_Raw_RCCI_frozen_predictor_development',
    arms=['ordinary','conditional','rcci_ordinary','rcci_conditional'],
    training_arms=['rcci_ordinary','rcci_conditional'], reuse_raw_weights=True,
    selection='final_120k_primary_earlier_nodes_diagnostic_no_best_selection')
PROTOCOL['training'].update(checkpoint_frames=[20000,40000,60000,120000], requested_frames=120000)
PROTOCOL['comparisons'] = [
    {'id':'b3_interface','left':'conditional','right':'rcci_conditional'},
    {'id':'b2_interface','left':'ordinary','right':'rcci_ordinary'},
    {'id':'raw_conditioning','left':'ordinary','right':'conditional'},
    {'id':'rcci_conditioning','left':'rcci_ordinary','right':'rcci_conditional'}]
SMOKE = deepcopy(PROTOCOL)
SMOKE.update(version='rcci04c_entry_smoke_v1', purpose='bounded_engineering_NOT_method_comparison',
             selection='smoke_weights_not_comparable_to_legacy_120k')
SMOKE['training'].update(run_seeds=[0], checkpoint_frames=[16], requested_frames=16,
                          replay_start_size=8, minibatch_size=8)
SMOKE['evaluation'].update(simulator_seeds=[200], workers=1)


def validate_protocol(p):
    if digest(p) not in (digest(PROTOCOL), digest(SMOKE)):
        raise ValueError('Unknown fields/changed matched design; freeze a new version')
    return p


def jobs(p, stage):
    validate_protocol(p)
    train = [{'id':f'{a}_s{s}', 'arm':a, 'run_seed':s}
             for a in p['training_arms'] for s in p['training']['run_seeds']]
    if stage == 'train': return train
    if stage != 'evaluate': raise ValueError('Unknown stage')
    return [{'id':f'{a}_s{s}_n{n}_v{v}', 'arm':a, 'run_seed':s, 'training_id':f'{a}_s{s}',
             'checkpoint_frames':n, 'simulator_seed':v}
            for a in p['arms'] for s in p['training']['run_seeds']
            for n in ([120000] if p == SMOKE and a not in p['training_arms'] else p['training']['checkpoint_frames'])
            for v in p['evaluation']['simulator_seeds']]


def aggregate(p, rows):
    validate_protocol(p)
    if p == SMOKE:
        if {r['id'] for r in rows} != {j['id'] for j in jobs(p,'evaluate')} or len(rows) != 4:
            raise ValueError('Incomplete entry smoke')
        lookup={r['id']:r for r in rows}
        for j in jobs(p,'evaluate'):
            r=lookup[j['id']]; previous.snapshot_steps(r['actual_training_steps'],j['checkpoint_frames'])
            if (any(r.get(k)!=v for k,v in j.items()) or not r['no_policy_updates'] or not r['native_event_agrees']
                    or sum(r[k] for k in ('natural_arrival','ego_collision','time_limit'))!=1
                    or any(r[m] is not None and not math.isfinite(r[m]) for m in previous.METRICS)):
                raise ValueError('Invalid smoke execution cell')
        if len({r['initial_traffic_sha256'] for r in rows})!=1: raise ValueError('Smoke initial traffic differs')
        return {'status':'complete','engineering_complete':True,'method_effect_gate':None,
                'method_comparison_allowed':False,'evaluation_episodes':4,'unique_traffic_scenarios':1}
    fn = FunctionType(previous.aggregate.__code__, {**previous.aggregate.__globals__,
        'validate_protocol':validate_protocol,'jobs':jobs}, previous.aggregate.__name__)
    result = fn(p, rows)
    lookup = {r['id']:r for r in rows}
    for node in result['nodes']:
        n = node['requested_training_steps']; paired = []
        for s in p['training']['run_seeds']:
            for v in p['evaluation']['simulator_seeds']:
                four = [lookup[f'{a}_s{s}_n{n}_v{v}'] for a in p['arms']]
                paired.append({'run_seed':s,'simulator_seed':v,'difference_of_differences':{
                    m:None if any(r[m] is None for r in four) else (four[3][m]-four[1][m])-(four[2][m]-four[0][m])
                    for m in previous.METRICS}})
        node['interface_conditioning_interaction'] = {'paired_rows':paired,
            'per_run_seed':{str(s):{m:previous.mean(r['difference_of_differences'][m] for r in paired if r['run_seed']==s)
                for m in previous.METRICS} for s in p['training']['run_seeds']}, 'scope':'descriptive_small_seed_sample'}
        interaction=node['interface_conditioning_interaction']; values=list(interaction['per_run_seed'].values())
        interaction['equal_run_seed_mean']={m:previous.mean(v[m] for v in values) for m in previous.METRICS}
        interaction['run_seed_std']={m:statistics.stdev(v[m] for v in values if v[m] is not None)
            if sum(v[m] is not None for v in values)>1 else None for m in previous.METRICS}
        for a in p['arms']:
            selected = [r for r in rows if r['arm']==a and r['checkpoint_frames']==n]
            node['arms'][a]['success_conditioned_duration_by_run_seed'] = {
                str(s):previous.mean(r['simulation_duration_s'] for r in selected if r['run_seed']==s and r['natural_arrival'])
                for s in p['training']['run_seeds']}
    result.update(primary_comparison='b3_interface', automatic_budget_extension=False,
                  prediction_retraining=False, raw_policy_retraining=False)
    return result
