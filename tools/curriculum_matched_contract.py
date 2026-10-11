"""IC04 immutable development design; no simulator, model, or learner imports."""
from copy import deepcopy
import math
import statistics

from prediction_rl.curriculum.interaction_intensity import (
    LEVELS, OfflineEpisodeSampler, proposed_decision, schedule_proposal)
from prediction_rl.data.dataset_contract import digest
from prediction_rl.evaluation.matched_learning_curve import METRICS as BASE_METRICS, mean, snapshot_steps

METRICS = BASE_METRICS + ('minimum_same_path_front_gap_m','minimum_same_path_rear_gap_m')

SCOPE = 'ic04_user_run_interaction_intensity_matched_120k_v1'
ARMS = ['b2_mix', 'b2_curriculum', 'b3_mix', 'b3_curriculum']
THRESHOLDS = {'I_low': .3976121680515756, 'I_high': 2.6839018875244487,
    'repeat_noise_floor': 0., 'complete_source_support': True, 'separated_terciles': True}
PROTOCOL = {
    'schema_version': 1, 'protocol_id': SCOPE,
    'scope': 'matched_development_not_independent_test', 'run_id': 'ic04_train_v1',
    'arms': ARMS, 'task_seeds': list(range(11000,11249,8)),
    'classification': 'frozen_train_only_reference_conditioned_L_M_H',
    'thresholds': THRESHOLDS, 'layer_counts': {'L':11,'M':10,'H':11},
    'target_task_weight': 'uniform_1_over_32',
    'schedule': schedule_proposal({'L':11/32,'M':10/32,'H':11/32}),
    'entry_contract': 'seeded_fresh_gym20_task_v1',
    'execution_contract': 'simulation_blocking_exact_v1', 'reference_prefix_steps': 0,
    'training': {'run_seeds':[0,1,2], 'sampler_seeds':[17003,17004,17005],
        'requested_frames':120000, 'checkpoint_frames':[20000,40000,60000,120000],
        'workers':1, 'torch_threads':1, 'device':'cpu', 'learning_rate':.0002,
        'replay_start_size':5000, 'minibatch_size':100, 'replay_buffer_size':1000000,
        'native_control_cap':500, 'from_scratch':True,
        'snapshot_rule':'first_complete_episode_boundary_at_or_after_requested_node',
        'resume_weights_only':False, 'clear_replay_at_stage_change':False},
    'evaluation': {'simulator_seeds':list(range(200,220)), 'workers':2,
        'torch_threads':1, 'deterministic':True, 'episodes_per_scene':1,
        'driver':'gym20_fresh_process_first_episode', 'native_control_cap':500},
    'main_endpoint':120000, 'selection':'no_best_node_selection',
    'comparisons': [
        {'id':'curriculum_on_B2','left':'b2_mix','right':'b2_curriculum'},
        {'id':'curriculum_on_B3','left':'b3_mix','right':'b3_curriculum'},
        {'id':'conditioning_in_Mix','left':'b2_mix','right':'b3_mix'},
        {'id':'conditioning_in_Curriculum','left':'b2_curriculum','right':'b3_curriculum'}],
    'specificity':'(B3_Curr-B2_Curr)-(B3_Mix-B2_Mix)',
    'auc':'node_trapezoid_20000_to_120000_divided_by_100000_not_continuous_measurement',
    'metadata_in_network_inputs':False, 'predictor_training':False,
    'online_true_counterfactual_branches':False, 'reward_or_action_changes':False,
    'automatic_budget_extension':False, 'resume_incomplete_job':False,
    'test_opened':False, 'author50':False, 'formal_claim':False}


def validate_protocol(value):
    if not isinstance(value,dict) or digest(value) != digest(PROTOCOL):
        raise ValueError('IC04 frozen design changed; no seed, budget, threshold or unknown-field adaptation')
    return value


def jobs(protocol, stage):
    validate_protocol(protocol)
    train = []
    for arm in ARMS:
        model, mode = arm.split('_',1)
        for seed, sampler_seed in zip(protocol['training']['run_seeds'],protocol['training']['sampler_seeds']):
            train.append({'id':f'{arm}_s{seed}','arm':arm,'run_seed':seed,'sampling_seed':sampler_seed,
                'sampling_mode':mode,'predictor_mode':'ordinary' if model=='b2' else 'conditional'})
    if stage == 'train': return train
    if stage != 'evaluate': raise ValueError('Unknown stage')
    return [{**j,'training_id':j['id'],'checkpoint_frames':n,'simulator_seed':v,
        'id':f'{j["id"]}_n{n}_v{v}'} for j in train
        for n in protocol['training']['checkpoint_frames'] for v in protocol['evaluation']['simulator_seeds']]


def freeze_manifest(pack, report, source_sha256):
    """Publish NEW live labels without modifying the historical proposal."""
    from prediction_rl.curriculum.integration_smoke import task_roster
    roster = task_roster(pack)
    if (report['status']!='complete' or report['engineering_gate'] is not True
            or report['summary']['coverage_gate'] is not True
            or pack['request_hash']!=report['request_hash']
            or {k:report['threshold_proposal'][k] for k in THRESHOLDS}!=THRESHOLDS):
        raise ValueError('Need the complete accepted train-only threshold/coverage evidence')
    tasks = []
    for old, choice in zip(pack['tasks'],roster):
        if (proposed_decision(old,THRESHOLDS)['proposed_level']!=choice['level']
                or old['predictor_slot_coverage']['key_actor_missing_slots']
                or not old['distinct_target_front_rear'] or old['repeat_noise_bound']!=0.):
            raise ValueError('Missing response, noise, critical-actor coverage or inconsistent task label')
        tasks.append({**deepcopy(old),'level':choice['level'],'task_level':choice['level'],
            'classification_applied':True,'label_protocol':SCOPE,
            'execution_authorized':False,'training_ready':False})
    return {'version':'ic04_frozen_task_manifest_v1','protocol_id':SCOPE,
        'source_proposal_sha256':source_sha256,'source_request_hash':pack['request_hash'],
        'thresholds':deepcopy(THRESHOLDS),'tasks':tasks,'layer_counts':deepcopy(PROTOCOL['layer_counts']),
        'target':'finite_task32_uniform_not_old_continuing_traffic_distribution',
        'schedule':deepcopy(PROTOCOL['schedule']),
        'D_use':'independent_description_not_classifier_or_filter',
        'future_in_online_observation_reward_action':False,'method_effect_gate':None}


def manifest_roster(manifest):
    if (manifest.get('version')!='ic04_frozen_task_manifest_v1'
            or manifest.get('protocol_id')!=SCOPE or manifest.get('thresholds')!=THRESHOLDS
            or manifest.get('layer_counts')!=PROTOCOL['layer_counts']
            or manifest.get('schedule')!=PROTOCOL['schedule']):
        raise ValueError('Changed frozen task manifest')
    tasks = manifest['tasks']
    if (len(tasks)!=32 or [t['simulator_seed'] for t in tasks]!=PROTOCOL['task_seeds']
            or len({t['task_id'] for t in tasks})!=32
            or any(t['level']!=t['task_level'] or t['classification_applied'] is not True
                   or t['execution_authorized'] is not False or t['training_ready'] is not False
                   or proposed_decision(t,THRESHOLDS)['proposed_level']!=t['level'] for t in tasks)
            or {k:sum(t['level']==k for t in tasks) for k in LEVELS}!=PROTOCOL['layer_counts']):
        raise ValueError('Missing, selected, relabeled or executable-without-request task')
    return [{k:t[k] for k in ('task_id','simulator_seed','level')} for t in tasks]


class TrainingEpisodeSampler:
    """Explicit IC04 binding; old offline and zero-update smoke scopes stay closed."""
    scope = SCOPE
    def __init__(self, manifest, job, request_hash, protocol):
        validate_protocol(protocol)
        if (job not in jobs(protocol,'train') or not isinstance(request_hash,str)
                or len(request_hash)!=64 or any(x not in '0123456789abcdef' for x in request_hash)):
            raise ValueError('Require a frozen matched job and confirmed request digest')
        self.job,self.request_hash = deepcopy(job),request_hash
        self.manifest_hash = digest(manifest)
        self.algorithm = OfflineEpisodeSampler(manifest_roster(manifest),job['sampling_mode'],
            job['sampling_seed'],purpose='offline_algorithm_test_only')
        self.started = self.completed = 0

    def _scope(self, row):
        return {**row,'matched_request_hash':self.request_hash,'task_manifest_hash':self.manifest_hash,
            'matched_development_training_authorized':True,'test_opened':False}

    def begin_episode(self):
        if self.started!=self.completed:
            raise ValueError('Cannot change task inside an episode or retry a begun task')
        result = self.algorithm.begin_episode()  # Refuses a new task at/after 120k.
        self.started+=1
        return self._scope(result)

    def end_episode(self, controls):
        if (self.started!=self.completed+1 or type(controls) is not int or not 1<=controls<=500):
            raise ValueError('Require one complete native episode within the unchanged limit')
        result = self.algorithm.end_episode(controls)
        self.completed+=1
        if result['request_budget_overrun']>499:
            raise ValueError('Matched budget overrun exceeds one unchanged native episode')
        return self._scope(result)


def aggregate(protocol, rows):
    validate_protocol(protocol)
    expected=jobs(protocol,'evaluate'); lookup={r['id']:r for r in rows}
    if len(rows)!=len(lookup) or set(lookup)!={j['id'] for j in expected}:
        raise ValueError('Missing, duplicated or extra evaluations; all seeds and failures required')
    roots={}
    for j in expected:
        r=lookup[j['id']]
        flags=[r[k] for k in ('natural_arrival','ego_collision','time_limit')]
        if (any(r.get(k)!=v for k,v in j.items()) or r.get('no_policy_updates') is not True
                or r.get('native_event_agrees') is not True
                or any(type(v) is not int or v not in (0,1) for v in flags) or sum(flags)!=1
                or type(r['steps']) is not int or not 1<=r['steps']<=500
                or any(r[m] is not None and (isinstance(r[m],bool) or not math.isfinite(r[m])) for m in METRICS)
                or any(not 0<=r[k]<=1 for k in ('low_speed_fraction','action_saturation_fraction'))):
            raise ValueError('Invalid identity, frozen-policy claim or raw/native outcome')
        snapshot_steps(r['actual_training_steps'],j['checkpoint_frames'])
        h=r['initial_traffic_sha256']; v=j['simulator_seed']
        if not isinstance(h,str) or len(h)!=64 or any(x not in '0123456789abcdef' for x in h):
            raise ValueError('Missing initial traffic hash')
        if v in roots and roots[v]!=h: raise ValueError('Same evaluation seed has different initial traffic')
        roots[v]=h
    seeds=protocol['training']['run_seeds']; nodes=[]
    def paired_summary(pairs):
        byseed={str(s):{m:mean(p['delta'][m] for p in pairs if p['run_seed']==s) for m in METRICS} for s in seeds}
        return {'paired_rows':pairs,'per_run_seed_delta':byseed,
            'equal_run_seed_mean_delta':{m:mean(r[m] for r in byseed.values()) for m in METRICS},
            'run_seed_std':{m:statistics.stdev(r[m] for r in byseed.values() if r[m] is not None)
                if sum(r[m] is not None for r in byseed.values())>1 else None for m in METRICS}}
    for n in protocol['training']['checkpoint_frames']:
        arms={}
        for arm in ARMS:
            per_seed={str(s):{m:mean(lookup[f'{arm}_s{s}_n{n}_v{v}'][m]
                for v in protocol['evaluation']['simulator_seeds']) for m in METRICS} for s in seeds}
            arms[arm]={'per_run_seed':per_seed,
                'equal_run_seed_mean':{m:mean(r[m] for r in per_seed.values()) for m in METRICS},
                'metric_available_episodes_by_run_seed':{str(s):{m:sum(lookup[f'{arm}_s{s}_n{n}_v{v}'][m] is not None
                    for v in protocol['evaluation']['simulator_seeds']) for m in METRICS} for s in seeds},
                'run_seed_std':{m:statistics.stdev(r[m] for r in per_seed.values() if r[m] is not None)
                    if sum(r[m] is not None for r in per_seed.values())>1 else None for m in METRICS}}
        comparisons={}
        for c in protocol['comparisons']:
            pairs=[]
            for s in seeds:
                for v in protocol['evaluation']['simulator_seeds']:
                    left=lookup[f'{c["left"]}_s{s}_n{n}_v{v}']; right=lookup[f'{c["right"]}_s{s}_n{n}_v{v}']
                    pairs.append({'run_seed':s,'simulator_seed':v,'delta':{m:None if left[m] is None or right[m] is None
                        else right[m]-left[m] for m in METRICS}})
            comparisons[c['id']]={**c,**paired_summary(pairs)}
        pairs=[]
        for s in seeds:
            for v in protocol['evaluation']['simulator_seeds']:
                r={a:lookup[f'{a}_s{s}_n{n}_v{v}'] for a in ARMS}
                pairs.append({'run_seed':s,'simulator_seed':v,'delta':{m:None if any(x[m] is None for x in r.values())
                    else (r['b3_curriculum'][m]-r['b2_curriculum'][m])-(r['b3_mix'][m]-r['b2_mix'][m]) for m in METRICS}})
        nodes.append({'requested_training_steps':n,'arms':arms,'comparisons':comparisons,'specificity_G':paired_summary(pairs)})
    def auc(values):
        if any(v is None for v in values): return None
        ns=protocol['training']['checkpoint_frames']
        return math.fsum((b-a)*(x+y)/2 for a,b,x,y in zip(ns,ns[1:],values,values[1:]))/(ns[-1]-ns[0])
    auc_rows={a:{str(s):{m:auc([node['arms'][a]['per_run_seed'][str(s)][m] for node in nodes])
        for m in METRICS} for s in seeds} for a in ARMS}
    return {'status':'complete','engineering_complete':True,'method_effect_gate':None,
        'nodes':nodes,'primary_node':120000,'checkpoint_selection_by_results':False,
        'normalized_node_trapezoid_auc':auc_rows,'training_replicates':3,
        'unique_traffic_scenarios':len(roots),'evaluation_episodes':len(rows),
        'repeated_nodes_are_not_independent_scenes':True,'formal_claim':False,'test_opened':False,
        'analysis':'paired_development_descriptive_three_training_seeds_no_formal_significance',
        'nominal_Mix_reset_quota_is_not_actual_transition_exposure_equality':True}
