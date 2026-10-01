"""Read-only P7b recording and descriptive ranges; never repairs policy inputs."""
from collections import deque
from copy import deepcopy
import math

import numpy as np

from prediction_rl.data.actor_adapter import build_actor_inputs
from prediction_rl.data.dataset_contract import digest
from prediction_rl.evaluation.exploration import EpisodeMetrics
from prediction_rl.features.probe_features import DIMENSION,NAMES

PROTOCOL={
    'version':'p07b_closed_loop_diagnostics_v1','purpose':'development_recording_not_method_selection',
    'parent_request':'artifacts/p7/p7_explore_v1/request.json',
    'parent_request_hash':'d068c069334874d668bcdbe32c7e62a42b997c40d151feb099794d50fcb7833f',
    'parent_aggregate_sha256':'c6ffb04656ad349442814ddde6b768a361b47bc6edfeeab1c061886281f960bb',
    'arms':['baseline','zero','ordinary','conditional'],'run_seeds':[0,1,2],
    'simulator_seeds':[200,210,219],
    'scene_selection':'first_midpoint_index_n_over_2_last_of_parent_roster',
    'workers':2,'torch_threads':1,'policy':'deterministic_no_exploration',
    'execution_contract':'simulation_blocking_exact_v1',
    'parity':'exact_parent_actions_rewards_outcomes_v1',
    'reference':'observed_eligible_train_roots_no_labels_v1',
    'thresholds_descriptive_only':{'low_speed_mps':.1,'reconstructed_speed_gap_mps':.001},
    'formal_claim':False,'test_opened':False}
AUDIT=deepcopy(PROTOCOL)
AUDIT.update(version='p07b_recording_smoke_v1',purpose='bounded_recording_engineering_audit',
             run_seeds=[2],simulator_seeds=[200],workers=1,
             scene_selection='fixed_failure_coverage_smoke_not_effect_evidence')


def validate_protocol(config):
    if digest(config) not in (digest(PROTOCOL),digest(AUDIT)):
        raise ValueError('Modified diagnostic design: use an approved new version')
    return config


def select_jobs(config,parent):
    validate_protocol(config)
    seeds=parent['protocol']['evaluation']['simulator_seeds']
    if config==PROTOCOL and config['simulator_seeds']!=[seeds[0],seeds[len(seeds)//2],seeds[-1]]:
        raise ValueError('Outcome-independent scene selection changed')
    result=[j for j in parent['evaluation_jobs'] if j['arm']in config['arms']
            and j['run_seed']in config['run_seeds'] and j['simulator_seed']in config['simulator_seeds']]
    if len(result)!=len(config['arms'])*len(config['run_seeds'])*len(config['simulator_seeds']):
        raise ValueError('Incomplete parent roster')
    return result


def feature_mask(values):
    x=np.asarray(values,dtype=np.float64)
    if x.shape!=(DIMENSION,) or not np.isfinite(x).all():raise ValueError('Invalid feature vector')
    if any(x[i]not in (0.,1.) for i in (140,142,144,148)):raise ValueError('Invalid presence channels')
    valid=np.ones(DIMENSION,dtype=bool)
    for c in range(5):
        valid[c*24:c*24+12]=bool(x[140])
        valid[c*24+12:c*24+24]=bool(x[142])
    valid[120:140]=bool(x[144])
    return valid


def cohort(lane,frames):
    return lane+'|'+('full' if frames==11 else 'partial')


def ranges(values):
    x=np.asarray(values,dtype=np.float64)
    if not len(x):return {'count':0,'min':None,'p01':None,'median':None,'p99':None,'max':None}
    if x.ndim!=1 or not np.isfinite(x).all():raise ValueError('Invalid range observations')
    return {'count':len(x),'min':float(x.min()),'p01':float(np.quantile(x,.01)),
            'median':float(np.median(x)),'p99':float(np.quantile(x,.99)),'max':float(x.max())}


def vector_ranges(rows):
    values=np.asarray([r['features'] for r in rows],dtype=np.float64)
    valid=np.asarray([feature_mask(r['features']) for r in rows])
    return {name:ranges(values[valid[:,i],i]) for i,name in enumerate(NAMES)}


def physical_values(inputs):
    result={}
    for kind in ('ego','actor'):
        values=np.asarray(inputs[kind+'_features'],dtype=np.float64)
        masks=np.asarray(inputs[kind+'_history_mask'],dtype=bool)
        if values.shape[:-1]!=masks.shape or not np.isfinite(values).all():raise ValueError('Invalid physical history')
        for i,name in enumerate(inputs['feature_names']):result[kind+'.'+name]=values[...,i][masks].tolist()
    values=np.asarray(inputs['summary_features'],dtype=np.float64)
    masks=np.asarray(inputs['summary_feature_mask'],dtype=bool)
    if values.shape!=masks.shape or not np.isfinite(values).all():raise ValueError('Invalid summary history')
    for i,name in enumerate(inputs['summary_feature_names']):result['summary.'+name]=values[...,i][masks[...,i]].tolist()
    return result


def compare_ranges(values,reference):
    x=np.asarray(values,dtype=np.float64);r=reference
    result={'observed_count':len(x),'observed':ranges(values),'reference_count':r['count']}
    if not r['count'] or not len(x):
        result.update(outside_train_minmax_fraction=None,outside_train_p01_p99_fraction=None)
    else:
        result.update(outside_train_minmax_fraction=float(((x<r['min'])|(x>r['max'])).mean()),
                      outside_train_p01_p99_fraction=float(((x<r['p01'])|(x>r['p99'])).mean()))
    return result


class CapturedProvider:
    """Original provider called ONCE; pure repeated encoding is an audit sidecar."""
    def __init__(self,base):
        self.base,self.model,self.geometry,self.config=base,base.model,base.geometry,base.config
        self.last=None
    def __call__(self,history):
        result=self.base(history)
        inputs=build_actor_inputs(history,self.geometry.geometry,self.config.neighbor_capacity,
                                  self.config.history_steps,self.config.tick_s)
        root=history[-1]
        frame={'simulation_time_s':root['simulation_time_s'],
               'vehicles':{k:{n:deepcopy(v[n]) for n in ('position','speed','acceleration','lane_id',
                                                       'lane_position_m','length_m','width_m')}
                           for k,v in root['vehicles'].items()}}
        self.last={'frame':frame,'inputs':inputs,'history_frames':len(history)}
        return result


class Recorder:
    """Observation i is consumed by action i; step i yields observation i+1."""
    def __init__(self,arm,geometry,reader,provider=None):
        self.arm,self.geometry,self.reader,self.provider=arm,geometry,reader,provider
        self.history=deque(maxlen=11);self.metrics=EpisodeMetrics();self.rows=[]
    def record(self,event,obs,reward,done,info):
        obs=np.asarray(obs)
        expected=20 if self.arm=='baseline' else 169
        if obs.shape!=(expected,) or obs.dtype!=np.float32 or not np.isfinite(obs).all():
            raise ValueError('Invalid policy observation')
        if self.arm=='zero' and np.any(obs[20:]):raise ValueError('B1 must stay zero')
        self.metrics.record(event,obs,reward,done,info)
        capture=None
        if not done:
            if self.provider is None:
                frame=self.reader();self.history.append(frame)
                capture={'frame':frame,'inputs':build_actor_inputs(list(self.history),self.geometry),
                         'history_frames':len(self.history),'inputs_are_shadow_only':True}
            else:
                if self.provider.last is None:raise ValueError('Missing predictor capture')
                capture=deepcopy(self.provider.last);capture['inputs_are_shadow_only']=False
            observed=info['traffic'] if event=='reset' else info['execution_audit']['after']
            core={'simulation_time_s':capture['frame']['simulation_time_s'],
                  'vehicles':{k:{name:v[name] for name in ('position','speed','acceleration','lane_id')}
                              for k,v in capture['frame']['vehicles'].items()}}
            if digest(core)!=digest(observed):raise ValueError('Recorder read changed simulator state')
        elif self.arm!='baseline' and np.any(obs[20:]):raise ValueError('Terminal channels must stay zero')
        self.rows.append({'observation_index':len(self.rows),'event':event,'done':bool(done),
            'observation':obs.tolist(),'prediction_features':deepcopy(info['prediction_features']),
            'execution_audit':deepcopy(info.get('execution_audit')),'capture':capture})

    def mark_policy_input(self,state,timestep_before):
        import torch
        row=self.rows[-1]
        if ('ddpg_input_with_time' in row or state.features.tolist()!=[row['observation']]
                or state.mask.tolist()!=[0 if row['done'] else 1]):
            raise ValueError('Policy input no longer matches recorded observation')
        row['ddpg_input_with_time']=torch.cat([state.features,.001*timestep_before[:,None]],1)[0].tolist()
        row['policy_state_mask']=state.mask.tolist()


def assert_parity(outcome,actions,parent_outcome,parent_actions):
    if digest(outcome)!=digest(parent_outcome) or digest(actions)!=digest(parent_actions):
        raise ValueError('Recording changed P7 actions/rewards/outcomes; diagnostic invalid')
    return True


def summarize_trace(rows,arm,reference,thresholds):
    if not rows or rows[0]['event']!='reset' or not rows[-1]['done']:
        raise ValueError('Only a complete episode can be diagnosed')
    captures=[r['capture'] for r in rows if r['capture']is not None]
    steps=[r['execution_audit'] for r in rows[1:]]
    if any(a is None for a in steps):raise ValueError('Missing execution measurements')
    low=thresholds['low_speed_mps'];speed_gap=thresholds['reconstructed_speed_gap_mps']
    consecutive=maximum=0;slow=0;gaps=[];lanes={};progress=[];role_changes=0;previous=None
    for c in captures:
        ego=c['frame']['vehicles']['ego'];progress.append(float(c['inputs']['ego_features'][-1][4]))
        roles=c['inputs']['mandatory_roles']
        if previous is not None:role_changes+=roles!=previous
        previous=roles
    for a in steps:
        ego=a['before']['vehicles']['ego'];lane=ego['lane_id']
        slow+=ego['speed']<=low;consecutive=consecutive+1 if ego['speed']<=low else 0;maximum=max(maximum,consecutive)
        lanes.setdefault(lane,[]).append(ego['speed'])
        if a['measured_post_step_speed']is not None:gaps.append(a['measured_post_step_speed']-a['commanded_speed_reconstructed'])
    physical={}
    for c in captures:
        for k,v in physical_values(c['inputs']).items():physical.setdefault(k,[]).extend(v)
    result={'observed_nonterminal_states':len(captures),'feature_history_cohorts':{},
        'mandatory_role_change_transitions':role_changes,
        'states_with_omitted_actors':sum(bool(c['inputs']['root_omitted_ids']) for c in captures),
        'physical_input_ranges':{k:compare_ranges(v,reference['physical'][k]) for k,v in physical.items()},
        'control':{'low_speed_before_step_fraction':slow/len(steps),'longest_low_speed_before_samples':maximum,
                   'longest_low_speed_grid_duration_s':maximum*.2,
                   'observed_progress_range_m':None if not progress else [min(progress),max(progress)],
                   'speed_by_lane':{k:ranges(v) for k,v in lanes.items()},
                   'reconstructed_command_gap_count':len(gaps),
                   'mean_abs_reconstructed_command_gap_mps':None if not gaps else float(np.abs(gaps).mean()),
                   'reconstructed_command_gap_exceeds_descriptive_threshold_fraction':None if not gaps else float((np.abs(gaps)>speed_gap).mean())},
        'predictor_features_applicable':arm in ('ordinary','conditional')}
    if arm in ('ordinary','conditional'):
        vectors=[]
        for row in rows:
            c=row['capture']
            if c is None:continue
            key=cohort(c['frame']['vehicles']['ego']['lane_id'],c['history_frames'])
            result['feature_history_cohorts'][key]=result['feature_history_cohorts'].get(key,0)+1
            vectors.append({'cohort':key,'features':row['observation'][20:],
                            'valid':feature_mask(row['observation'][20:])})
        result['feature_ranges_global']={n:compare_ranges([r['features'][i] for r in vectors if r['valid'][i]],
                                                          reference['models'][arm]['global'][n]) for i,n in enumerate(NAMES)}
        result['feature_ranges_matched_cohort']={}
        for key in result['feature_history_cohorts']:
            ref=reference['models'][arm]['cohorts'].get(key)
            result['feature_ranges_matched_cohort'][key]=None if ref is None else {
                n:compare_ranges([r['features'][i] for r in vectors if r['cohort']==key and r['valid'][i]],ref[n])
                for i,n in enumerate(NAMES)}
    return result
