from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest
import torch
from all.environments import State

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
import diagnose_feature_ddpg as runner
from prediction_rl.data.collection_store import write_once,file_hash
from prediction_rl.evaluation.exploration import PROTOCOL as P7,jobs
from prediction_rl.evaluation.closed_loop_diagnostics import (
    PROTOCOL,AUDIT,validate_protocol,select_jobs,feature_mask,ranges,vector_ranges,
    compare_ranges,physical_values,CapturedProvider,Recorder,assert_parity,summarize_trace)
from prediction_rl.features.probe_features import FrozenProbeFeatures,NAMES
from prediction_rl.prediction.interface import PredictorConfig
from prediction_rl.prediction.model import PredictorEnsemble
from test_probe_features import scene


def test_standalone_configs_and_rosters():
    parent={'protocol':P7,'evaluation_jobs':jobs(P7,'evaluate')}
    assert len(select_jobs(PROTOCOL,parent))==36
    assert len(select_jobs(AUDIT,parent))==4
    assert set(j['run_seed'] for j in select_jobs(PROTOCOL,parent))=={0,1,2}
    assert set(j['simulator_seed'] for j in select_jobs(PROTOCOL,parent))=={200,210,219}
    for name,p in [('p07b_closed_loop_diagnostics_v1',PROTOCOL),('p07b_recording_smoke_v1',AUDIT)]:
        assert json.loads((ROOT/f'configs/development/{name}.json').read_text())==p


@pytest.mark.parametrize('change',[
    lambda p:p.update(run_seeds=[0,1]),lambda p:p.update(simulator_seeds=[200,201,202]),
    lambda p:p.update(workers=True),lambda p:p.update(extra=True),
    lambda p:p['thresholds_descriptive_only'].update(low_speed_mps=1)])
def test_protocol_changes_need_new_version(change):
    p=deepcopy(PROTOCOL);change(p)
    with pytest.raises(ValueError):validate_protocol(p)


def test_selection_rejects_missing_parent_roster():
    p={'protocol':P7,'evaluation_jobs':jobs(P7,'evaluate')[:-1]}
    with pytest.raises(ValueError):select_jobs(PROTOCOL,p)


def test_absent_feature_channels_do_not_become_zero_training_evidence():
    x=np.ones(149);x[140]=0;x[144]=0
    m=feature_mask(x)
    for c in range(5):
        assert not m[c*24:c*24+12].any()
        assert m[c*24+12:c*24+24].all()
    assert not m[120:140].any() and m[140:].all()
    r=vector_ranges([{'features':x.tolist()}])
    assert r[NAMES[0]]['count']==0 and r[NAMES[12]]['count']==1
    assert compare_ranges([1],r[NAMES[0]])['outside_train_minmax_fraction']is None


@pytest.mark.parametrize('value',[float('nan'),float('inf'),.5])
def test_invalid_presence_or_feature_values(value):
    x=np.ones(149);x[140]=value
    with pytest.raises(ValueError):feature_mask(x)


def test_descriptive_ranges_are_not_clipping_or_gates():
    r=ranges([0,1,2]);x=[-1,1,3];original=list(x)
    c=compare_ranges(x,r)
    assert c['outside_train_minmax_fraction']==pytest.approx(2/3)
    assert x==original and 'gate'not in c
    assert ranges([])['count']==0
    with pytest.raises(ValueError):ranges([float('nan')])


def test_physical_masks_exclude_padding():
    _,_,x=scene();values=physical_values(x)
    assert len(values['ego.speed_mps'])==1
    assert len(values['actor.speed_mps'])==2
    assert values['summary.count']==[]
    poison=deepcopy(x);poison['actor_features'][0][0][0]=float('nan')
    with pytest.raises(ValueError):physical_values(poison)


@pytest.mark.parametrize('mode',['ordinary','conditional'])
def test_captured_provider_exact_output_once_rng_weights_and_no_future_input(mode):
    g,h,_=scene();cfg=PredictorConfig(hidden_dim=8)
    model=PredictorEnsemble(cfg,mode,[1,2,3]).eval().requires_grad_(False)
    base=FrozenProbeFeatures(g,cfg,model);expected=base(h)
    calls=[]
    class Count:
        model=base.model;geometry=base.geometry;config=base.config
        def __call__(self,history):calls.append(1);return base(history)
    captured=CapturedProvider(Count());before=torch.get_rng_state().clone()
    weights={k:v.clone() for k,v in model.state_dict().items()}
    poisoned=deepcopy(h);poisoned[0]['future_labels']={'bad':float('nan')}
    actual=captured(poisoned)
    np.testing.assert_array_equal(actual,expected)
    assert len(calls)==1 and torch.equal(torch.get_rng_state(),before)
    assert captured.last['history_frames']==1
    assert 'future_labels'not in captured.last['frame']
    assert all(torch.equal(weights[k],v) for k,v in model.state_dict().items())


def core(frame):
    return {'simulation_time_s':frame['simulation_time_s'],'vehicles':{
        k:{n:v[n] for n in ['position','speed','acceleration','lane_id']} for k,v in frame['vehicles'].items()}}


def info(before,after):
    return {'prediction_features':{'history_frames':0,'latency_s':0.},
        'execution_audit':{'before':core(before),'after':core(after),'requested_jerk':0.,
            'commanded_speed_reconstructed':before['vehicles']['ego']['speed'],
            'measured_post_step_speed':after['vehicles']['ego']['speed'],
            'measured_post_step_acceleration':0.,'measured_jerk':0.,
            'invalid_action_reward':0.,'simulation_elapsed_s':.2,'termination_reason':'upstream_arrival'}}


@pytest.mark.parametrize('arm',['baseline','zero'])
def test_shadow_recorder_never_changes_observation_terminal_does_not_read(arm):
    g,h,x=scene();calls=[]
    def reader():calls.append(1);return h[0]
    r=Recorder(arm,g.geometry,reader)
    obs=np.zeros(20 if arm=='baseline' else 169,np.float32)
    r.record('reset',obs,None,False,{'traffic':core(h[0]),'prediction_features':{'latency_s':0.}})
    state=State(torch.from_numpy(obs)[None,:],torch.tensor([1],dtype=torch.uint8),[{}])
    r.mark_policy_input(state,torch.zeros(1))
    r.record('step',obs,-1.,True,info(h[0],h[0]))
    assert len(calls)==1 and r.rows[0]['capture']['inputs_are_shadow_only']is True
    assert r.rows[1]['capture']is None
    np.testing.assert_array_equal(obs,r.rows[0]['observation'])
    assert len(r.rows[0]['ddpg_input_with_time'])==len(obs)+1
    with pytest.raises(ValueError):r.mark_policy_input(state,torch.ones(1))


def test_shadow_reader_cannot_advance_simulation():
    g,h,_=scene();changed=deepcopy(h[0]);changed['simulation_time_s']+=.2
    r=Recorder('baseline',g.geometry,lambda:changed)
    with pytest.raises(ValueError):r.record('reset',np.zeros(20,np.float32),None,False,
                                         {'traffic':core(h[0]),'prediction_features':{}})


def test_summary_reports_control_missing_cohort_and_shadow_semantics():
    g,h,x=scene();r=Recorder('baseline',g.geometry,lambda:h[0]);obs=np.zeros(20,np.float32)
    r.record('reset',obs,None,False,{'traffic':core(h[0]),'prediction_features':{}})
    a=info(h[0],h[0]);a['execution_audit']['measured_post_step_speed']=None
    r.record('step',obs,1.,True,a)
    reference={'physical':{k:ranges(v) for k,v in physical_values(x).items()},'models':{}}
    result=summarize_trace(r.rows,'baseline',reference,PROTOCOL['thresholds_descriptive_only'])
    assert result['control']['reconstructed_command_gap_count']==0
    assert result['control']['mean_abs_reconstructed_command_gap_mps']is None
    assert result['predictor_features_applicable']is False


def test_low_speed_control_is_descriptive_and_command_gap_is_reconstructed():
    g,h,_=scene();h[0]['vehicles']['ego']['speed']=0
    x=runner.build_actor_inputs(h,g.geometry)
    r=Recorder('baseline',g.geometry,lambda:h[0]);obs=np.zeros(20,np.float32)
    r.record('reset',obs,None,False,{'traffic':core(h[0]),'prediction_features':{}})
    a=info(h[0],h[0]);a['execution_audit']['measured_post_step_speed']=.2
    r.record('step',obs,1.,True,a)
    reference={'physical':{k:ranges(v) for k,v in physical_values(x).items()},'models':{}}
    c=summarize_trace(r.rows,'baseline',reference,PROTOCOL['thresholds_descriptive_only'])['control']
    assert c['low_speed_before_step_fraction']==1 and c['longest_low_speed_grid_duration_s']==.2
    assert c['mean_abs_reconstructed_command_gap_mps']==.2
    assert c['reconstructed_command_gap_exceeds_descriptive_threshold_fraction']==1
    assert 'gate'not in c


def test_missing_lane_history_cohort_is_unavailable_not_fake_ood_score():
    g,h,x=scene();cfg=PredictorConfig(hidden_dim=8)
    model=PredictorEnsemble(cfg,'conditional',[1,2,3]).eval().requires_grad_(False)
    provider=CapturedProvider(FrozenProbeFeatures(g,cfg,model));features=provider(h)
    r=Recorder('conditional',g.geometry,lambda:pytest.fail('must reuse original captured frame'),provider)
    obs=np.concatenate([np.zeros(20,np.float32),features])
    r.record('reset',obs,None,False,{'traffic':core(h[0]),'prediction_features':{}})
    r.record('step',np.zeros(169,np.float32),1.,True,info(h[0],h[0]))
    reference={'physical':{k:ranges(v) for k,v in physical_values(x).items()},
               'models':{'conditional':{'global':vector_ranges([{'features':features.tolist()}]),'cohorts':{}}}}
    d=summarize_trace(r.rows,'conditional',reference,PROTOCOL['thresholds_descriptive_only'])
    assert len(d['feature_ranges_matched_cohort'])==1
    assert list(d['feature_ranges_matched_cohort'].values())==[None]


@pytest.mark.parametrize('field',['outcome','actions'])
def test_exact_parent_parity_rejects_any_changed_execution(field):
    row={'return':-2.,'arrival':0};actions=[{'jerk':.2,'reward':-2.}]
    assert assert_parity(row,actions,deepcopy(row),deepcopy(actions))is True
    changed=deepcopy(row);other=deepcopy(actions)
    if field=='outcome':changed['arrival']=1
    else:other[0]['jerk']+=1e-7
    with pytest.raises(ValueError):assert_parity(row,actions,changed,other)


def test_training_reference_reads_observed_history_not_label_paths(tmp_path,monkeypatch):
    monkeypatch.setattr(runner,'ROOT',tmp_path)
    h={'split':'train','root_id':'r','episode_id':'e','history':[{'observed':True}]}
    write_once(tmp_path/'history.json',h)
    review={'test_locked':True,'roots':{'train':[{'eligible_trajectory_root':True,
        'history':'history.json','history_sha256':file_hash(tmp_path/'history.json'),
        'root_id':'r','episode_id':'e','labels':'MUST_NOT_READ.json'}]}}
    write_once(tmp_path/'review.json',review)
    parent={'predictor_training_request':{'dataset_review':'review.json',
        'dataset_review_sha256':file_hash(tmp_path/'review.json')}}
    histories,hashes=runner.observed_train_histories(parent)
    assert histories==[h] and len(hashes)==1 and not (tmp_path/'MUST_NOT_READ.json').exists()


def test_reference_rejects_modified_observed_history(tmp_path,monkeypatch):
    monkeypatch.setattr(runner,'ROOT',tmp_path);write_once(tmp_path/'history.json',{})
    review={'test_locked':True,'roots':{'train':[{'eligible_trajectory_root':True,
        'history':'history.json','history_sha256':'0'*64}]}}
    write_once(tmp_path/'review.json',review)
    parent={'predictor_training_request':{'dataset_review':'review.json','dataset_review_sha256':file_hash(tmp_path/'review.json')}}
    with pytest.raises(ValueError,match='history'):runner.observed_train_histories(parent)
