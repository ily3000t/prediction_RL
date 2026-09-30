"""Frozen exploratory design and seed-paired DESCRIPTIVE summaries, not paper tests."""
from copy import deepcopy
import math
import statistics

from prediction_rl.data.dataset_contract import digest

PROTOCOL = {
    'version':'p07_exploration_v1','purpose':'development_exploration_not_formal_claim',
    'ddpg_audit':'artifacts/p6/p6b_ddpg_01/report.json',
    'ddpg_audit_sha256':'694fe1fc35361a684d2c908becd11d79a3a797963838eaa444fa4b5392c0ea31',
    'arms':['baseline','zero','ordinary','conditional'],
    'training':{'run_seeds':[0,1,2],'requested_frames':20000,'episode_cap':None,
                'workers':1,'device':'cpu','torch_threads':1,'learning_rate':.0002,
                'replay_start_size':5000,'minibatch_size':100,'warm_start':False},
    'selection':'final_checkpoint_only_no_eval_selection',
    'evaluation':{'simulator_seeds':list(range(200,220)),'episodes_per_seed':1,
                  'workers':2,'policy':'deterministic_no_exploration'},
    'execution_contract':'simulation_blocking_exact_v1','test_opened':False,
    'comparisons':[{'id':'input_capacity','left':'baseline','right':'zero'},
                   {'id':'ordinary_information','left':'zero','right':'ordinary'},
                   {'id':'action_conditioning','left':'ordinary','right':'conditional'},
                   {'id':'total_augmentation','left':'baseline','right':'conditional'}],
    'aggregation':'paired_simulator_seeds_within_run_seed_then_equal_run_seed_mean',
    'formal_training':False}
AUDIT = deepcopy(PROTOCOL)
AUDIT.update(version='p07_workflow_smoke_v1',purpose='bounded_pipeline_engineering_audit')
AUDIT['training'].update(run_seeds=[0],requested_frames=64,episode_cap=4,replay_start_size=8,minibatch_size=8)
AUDIT['evaluation'].update(simulator_seeds=[200])
METRICS = ('return','arrival','collision','time_limit','simulation_duration_s',
           'invalid_action_rate','mean_abs_measured_jerk','minimum_observed_front_bumper_distance_m')


def validate_protocol(value):
    if digest(value) not in (digest(PROTOCOL),digest(AUDIT)):
        raise ValueError('Unknown/modified exploration protocol; create and approve a new version')
    return value


def jobs(protocol, stage):
    validate_protocol(protocol)
    training=[{'id':f'{arm}_s{seed}','arm':arm,'run_seed':seed}
              for arm in protocol['arms'] for seed in protocol['training']['run_seeds']]
    if stage=='train':return training
    if stage=='evaluate':return [{**t,'training_id':t['id'],'id':t['id']+f'_v{seed}',
                                 'simulator_seed':seed} for t in training
                                for seed in protocol['evaluation']['simulator_seeds']]
    raise ValueError('Unknown exploration stage')


class EpisodeMetrics:
    def __init__(self):
        self.steps=0;self.total=0.;self.duration=0.;self.invalid=0;self.jerk=[]
        self.nearest=None;self.termination=None;self.initial_hash=None
    def record(self,event,obs,reward,done,info):
        if event=='reset':
            if self.initial_hash is not None:raise ValueError('Metrics object must be new per episode')
            self.initial_hash=digest(info['traffic']);return
        if event!='step' or self.termination is not None or self.initial_hash is None:raise ValueError('Invalid episode event order')
        a=info['execution_audit']
        numbers=[reward,a['simulation_elapsed_s'],a['invalid_action_reward']]
        if a['measured_jerk'] is not None:numbers.append(a['measured_jerk'])
        if any(not math.isfinite(float(x)) for x in numbers) or a['simulation_elapsed_s']<=0:
            raise ValueError('Nonfinite/invalid episode measurements')
        self.total+=float(reward);self.steps+=1;self.duration+=a['simulation_elapsed_s']
        self.invalid+=a['invalid_action_reward']!=0
        if a['measured_jerk'] is not None:self.jerk.append(abs(a['measured_jerk']))
        for frame in (a['before'],a['after']):
            vehicles=frame['vehicles'];ego=vehicles.get('ego')
            if ego is None:continue
            for actor,state in vehicles.items():
                if actor!='ego':
                    distance=math.dist(ego['position'],state['position'])
                    if not math.isfinite(distance):raise ValueError('Nonfinite observed position')
                    self.nearest=distance if self.nearest is None else min(distance,self.nearest)
        if done:
            self.termination=a['termination_reason']
            if self.termination not in ('upstream_arrival','upstream_collision','upstream_time_limit','registered_time_limit'):
                raise ValueError('Unknown terminal reason')
    def result(self):
        if self.termination is None or not self.steps or self.initial_hash is None:raise ValueError('Incomplete episode is not an outcome')
        return {'steps':self.steps,'initial_traffic_sha256':self.initial_hash,
            'termination':self.termination,'return':self.total,
            'arrival':int(self.termination=='upstream_arrival'),
            'collision':int(self.termination=='upstream_collision'),
            'time_limit':int(self.termination in ('upstream_time_limit','registered_time_limit')),
            'simulation_duration_s':self.duration,'invalid_action_rate':self.invalid/self.steps,
            'mean_abs_measured_jerk':statistics.mean(self.jerk) if self.jerk else None,
            'measured_jerk_count':len(self.jerk),'minimum_observed_front_bumper_distance_m':self.nearest}


def mean(values):
    values=[x for x in values if x is not None]
    return statistics.mean(values) if values else None


def zero_channel_query(checked,state,timestep_before):
    """Read-only sensitivity query, with the SAME time as the executed action.

    Never call TimeFeature.eval again: that would advance its counter. Calling
    the deterministic policy directly neither samples exploration nor updates.
    """
    import torch
    from all.environments import State
    from prediction_rl.training.all_ddpg import validate_action
    if state.features.shape!=(1,169):raise ValueError('Query needs expanded input')
    zero=state.features.clone();zero[:,20:]=0
    altered=State(torch.cat([zero,.001*timestep_before[:,None]],1),state.mask,state.info)
    action=checked.agent.policy.eval(altered)
    validate_action(action,1,checked.low,checked.high)
    return action


def aggregate(protocol, rows):
    validate_protocol(protocol);expected=jobs(protocol,'evaluate')
    if len(rows)!=len(expected):raise ValueError('Missing/extra evaluation rows; no seed exclusion')
    lookup={r['id']:r for r in rows}
    if len(lookup)!=len(rows) or set(lookup)!={j['id'] for j in expected}:raise ValueError('Duplicate/wrong evaluation jobs')
    roots={}
    for job in expected:
        r=lookup[job['id']]
        if any(r.get(k)!=v for k,v in job.items()):raise ValueError('Evaluation identity mismatch')
        events=(r['arrival'],r['collision'],r['time_limit'])
        expected_events={'upstream_arrival':(1,0,0),'upstream_collision':(0,1,0),
                         'upstream_time_limit':(0,0,1),'registered_time_limit':(0,0,1)}
        if (type(r['steps']) is not int or not 1<=r['steps']<=500 or events!=expected_events.get(r['termination'])
                or any(type(x)is not int for x in events)
                or not 0<=r['invalid_action_rate']<=1 or r['simulation_duration_s']<=0
                or any(r[m] is not None and not math.isfinite(r[m]) for m in METRICS)):
            raise ValueError('Invalid outcomes; errors cannot become fabricated episodes')
        sensitivity=r.get('mean_abs_zero_channel_action_change')
        if sensitivity is not None and (not math.isfinite(sensitivity) or sensitivity<0):
            raise ValueError('Invalid diagnostic query')
        seed=job['simulator_seed'];h=r['initial_traffic_sha256']
        if not isinstance(h,str) or len(h)!=64:raise ValueError('Missing initial scenario hash')
        if seed in roots and roots[seed]!=h:raise ValueError('Same simulator seed did not start from same traffic')
        roots[seed]=h
    per_arm={};seeds=protocol['training']['run_seeds']
    for arm in protocol['arms']:
        per_seed={str(s):{m:mean([r[m] for r in rows if r['arm']==arm and r['run_seed']==s]) for m in METRICS} for s in seeds}
        diagnostic={str(s):mean([r.get('mean_abs_zero_channel_action_change') for r in rows
                                if r['arm']==arm and r['run_seed']==s]) for s in seeds}
        per_arm[arm]={'per_run_seed':per_seed,'equal_run_seed_mean':{m:mean([v[m] for v in per_seed.values()]) for m in METRICS},
                     'zero_channel_query':{'per_run_seed_mean_abs_action_change':diagnostic,
                                           'mean':mean(diagnostic.values()),'query_was_not_executed':True}}
    comparisons={}
    for c in protocol['comparisons']:
        per_seed={};pairs=[]
        for s in seeds:
            paired=[]
            for v in protocol['evaluation']['simulator_seeds']:
                l=lookup[f'{c["left"]}_s{s}_v{v}'];r=lookup[f'{c["right"]}_s{s}_v{v}']
                delta={m:None if l[m] is None or r[m] is None else r[m]-l[m] for m in METRICS}
                paired.append(delta);pairs.append({'run_seed':s,'simulator_seed':v,'right_minus_left':delta})
            per_seed[str(s)]={m:mean([p[m] for p in paired]) for m in METRICS}
        stats={}
        for m in METRICS:
            values=[d[m] for d in per_seed.values() if d[m] is not None]
            stats[m]={'mean':mean(values),'run_seed_std':statistics.stdev(values) if len(values)>1 else None,
                      'positive_run_seeds':sum(x>0 for x in values),'negative_run_seeds':sum(x<0 for x in values),
                      'paired_cells':sum(p['right_minus_left'][m] is not None for p in pairs)}
        comparisons[c['id']]={**c,'per_run_seed_delta':per_seed,'descriptive_delta':stats,'paired_rows':pairs}
    return {'status':'complete','engineering_complete':True,'method_effect_gate':None,
            'analysis':'exploratory_descriptive_no_significance_claim','training_replicates':len(seeds),
            'simulator_scenarios':len(roots),'evaluated_episodes':len(rows),'arms':per_arm,'comparisons':comparisons,
            'test_opened':False,'formal_claim':False}
