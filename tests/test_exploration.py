from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import gym
import numpy as np
import pytest
import torch
from all.environments import State
from all.logging import DummyWriter

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
from prediction_rl.evaluation.exploration import (
    PROTOCOL,AUDIT,METRICS,validate_protocol,jobs,EpisodeMetrics,aggregate,zero_channel_query)
from prediction_rl.training.all_ddpg import preset_factory
from prediction_rl.data.collection_store import write_once,seal_episode,read_json
import explore_feature_ddpg as runner


def rows(protocol=PROTOCOL):
    values=[]
    for j in jobs(protocol,'evaluate'):
        arm=protocol['arms'].index(j['arm'])
        values.append({**j,'steps':20,'initial_traffic_sha256':f"{j['simulator_seed']:064x}",
            'termination':'upstream_time_limit','return':-100+arm*(j['run_seed']+1),
            'arrival':0,'collision':0,'time_limit':1,'simulation_duration_s':4.,
            'invalid_action_rate':0.,'mean_abs_measured_jerk':None,
            'minimum_observed_front_bumper_distance_m':10.,
            'mean_abs_zero_channel_action_change':None if arm<2 else .1})
    return values


def test_rosters_and_standalone_configs():
    assert len(jobs(PROTOCOL,'train'))==12
    assert len(jobs(PROTOCOL,'evaluate'))==240
    assert len(jobs(AUDIT,'train'))==len(jobs(AUDIT,'evaluate'))==4
    assert set(PROTOCOL['training']['run_seeds']).isdisjoint(PROTOCOL['evaluation']['simulator_seeds'])
    for name,config in [('p07_exploration_v1',PROTOCOL),('p07_workflow_smoke_v1',AUDIT)]:
        assert json.loads((ROOT/f'configs/development/{name}.json').read_text())==config
    assert PROTOCOL['training']['replay_start_size']==5000
    assert PROTOCOL['training']['minibatch_size']==100
    assert PROTOCOL['selection']=='final_checkpoint_only_no_eval_selection'


@pytest.mark.parametrize('mutation',[
    lambda p:p['training'].update(run_seeds=[0,1,4]),
    lambda p:p['training'].update(requested_frames=20001),
    lambda p:p.update(test_opened=0),
    lambda p:p.update(extra=True),
    lambda p:p['training'].update(workers=4)])
def test_protocol_changes_need_explicit_new_version(mutation):
    p=deepcopy(PROTOCOL);mutation(p)
    with pytest.raises(ValueError):validate_protocol(p)


def test_hierarchical_paired_aggregation_and_negative_outcomes_complete():
    result=aggregate(PROTOCOL,rows())
    assert result['status']=='complete' and result['method_effect_gate'] is None
    assert result['training_replicates']==3 and result['simulator_scenarios']==20
    assert result['evaluated_episodes']==240 and result['formal_claim']is False
    d=result['comparisons']['action_conditioning']
    assert d['per_run_seed_delta']['0']['return']==1
    assert d['per_run_seed_delta']['1']['return']==2
    assert d['per_run_seed_delta']['2']['return']==3
    assert d['descriptive_delta']['return']=={'mean':2,'run_seed_std':1.,
        'positive_run_seeds':3,'negative_run_seeds':0,'paired_cells':60}
    assert d['descriptive_delta']['mean_abs_measured_jerk']['paired_cells']==0
    assert result['arms']['ordinary']['zero_channel_query']['mean']==.1
    assert result['arms']['baseline']['zero_channel_query']['mean']is None


@pytest.mark.parametrize('mutation',[
    lambda r:r.pop(),
    lambda r:r.append(deepcopy(r[0])),
    lambda r:r[1].update(id=r[0]['id']),
    lambda r:r[0].update(run_seed=9),
    lambda r:r[0].update(initial_traffic_sha256='f'*64),
    lambda r:r[0].update(return_=float('nan'),**{'return':float('nan')}),
    lambda r:r[0].update(steps=501),
    lambda r:r[0].update(steps=True),
    lambda r:r[0].update(arrival=1),
    lambda r:r[0].update(termination='upstream_collision'),
    lambda r:r[0].update(invalid_action_rate=2),
    lambda r:r[0].update(mean_abs_zero_channel_action_change=float('inf'))])
def test_invalid_rows_are_not_excluded_or_fabricated(mutation):
    r=rows();mutation(r)
    with pytest.raises(ValueError):aggregate(PROTOCOL,r)


def frame():
    return {'vehicles':{'ego':{'position':[0.,0.]},'rear':{'position':[3.,4.]},
                        'other':{'position':[6.,8.]}}}


def audit(done=True):
    return {'execution_audit':{'before':frame(),'after':frame(),
        'simulation_elapsed_s':.2,'invalid_action_reward':-.5,'measured_jerk':-2.,
        'termination_reason':'upstream_arrival' if done else None}}


def test_episode_metrics_preserve_original_reward_and_valid_jerk():
    m=EpisodeMetrics();m.record('reset',None,None,False,{'traffic':frame()})
    m.record('step',None,-3,False,audit(False))
    final=audit();final['execution_audit']['measured_jerk']=None
    m.record('step',None,5,True,final);r=m.result()
    assert r['return']==2 and r['steps']==2 and r['arrival']==1
    assert r['simulation_duration_s']==.4 and r['invalid_action_rate']==1
    assert r['mean_abs_measured_jerk']==2 and r['measured_jerk_count']==1
    assert r['minimum_observed_front_bumper_distance_m']==5
    with pytest.raises(ValueError):m.record('step',None,0,True,final)


def test_metrics_incomplete_reset_order_and_nonfinite():
    m=EpisodeMetrics()
    with pytest.raises(ValueError):m.result()
    with pytest.raises(ValueError):m.record('step',None,0,False,audit(False))
    m.record('reset',None,None,False,{'traffic':frame()})
    with pytest.raises(ValueError):m.record('reset',None,None,False,{'traffic':frame()})
    with pytest.raises(ValueError):m.record('step',None,float('nan'),True,audit())


def test_zero_query_does_not_advance_time_mutate_inputs_update_or_sample_rng():
    env=SimpleNamespace(feature_arm='conditional',state_space=gym.spaces.Box(-np.inf,np.inf,(169,),np.float32),
                        action_space=gym.spaces.Box(-5,5,(1,),np.float32))
    checked=preset_factory('cpu',2e-4,smoke=True)(env,DummyWriter())
    state=State(torch.ones(1,169),torch.tensor([1],dtype=torch.uint8),[{}])
    checked.eval(state,0.)
    before=checked.body.timestep.clone();original=state.features.clone()
    time=before-1
    torch_rng=torch.get_rng_state().clone();numpy_rng=np.random.get_state()
    query=zero_channel_query(checked,state,time)
    zero=state.features.clone();zero[:,20:]=0
    expected=checked.agent.policy.eval(State(torch.cat([zero,.001*time[:,None]],1),state.mask,state.info))
    assert torch.equal(query,expected) and torch.equal(state.features,original)
    assert torch.equal(checked.body.timestep,before) and torch.equal(torch.get_rng_state(),torch_rng)
    after=np.random.get_state()
    assert all(np.array_equal(a,b) for a,b in zip(numpy_rng,after))
    assert len(checked.agent.replay_buffer)==0
    assert not checked.agent.policy._optimizer.state and not checked.agent.q._optimizer.state


def stub_request(tmp_path,monkeypatch):
    q={'protocol':AUDIT,'train_jobs':jobs(AUDIT,'train'),'evaluation_jobs':jobs(AUDIT,'evaluate')}
    path=tmp_path/'request.json';write_once(path,q)
    monkeypatch.setattr(runner,'load_request',lambda *a:(path,q))
    monkeypatch.setattr(runner.shared,'metadata',lambda:{'git_commit':'a'*40,'working_tree_dirty':False})
    return path,q


def seal_train(path,q,j):
    out=path.parent/'train'/j['id'];write_once(out/'started.json',{})
    return seal_episode(out,runner.binding(q,'train',j),{
        'initial_weight_sha256':{'policy':'a','q':'b'},'weights_sha256':'c'*64})


def test_resume_only_reuses_complete_jobs_and_no_subprocess(tmp_path,monkeypatch):
    path,q=stub_request(tmp_path,monkeypatch)
    for j in q['train_jobs']:seal_train(path,q,j)
    monkeypatch.setattr(runner.subprocess,'run',lambda *a,**k:pytest.fail('should not train again'))
    result=runner.stage_run(path,'ignored','train',resume=True)
    assert read_json(result)['status']=='complete'
    assert {j['action'] for j in read_json(result)['jobs']}=={'reuse'}
    with pytest.raises(ValueError):runner.stage_run(path,'ignored','train')


def test_failed_job_is_preserved_not_retried(tmp_path,monkeypatch):
    path,q=stub_request(tmp_path,monkeypatch)
    write_once(path.parent/'train'/q['train_jobs'][0]['id']/'failure.json',{'error':'retained'})
    monkeypatch.setattr(runner.subprocess,'run',lambda *a,**k:pytest.fail('must not retry'))
    with pytest.raises(ValueError):runner.stage_run(path,'ignored','train',resume=True)
    assert read_json(path.parent/'train'/q['train_jobs'][0]['id']/'failure.json')=={'error':'retained'}


def test_failed_child_prevents_later_batches(tmp_path,monkeypatch):
    path,q=stub_request(tmp_path,monkeypatch);calls=[]
    def fail(command,**kwargs):
        calls.append(command)
        raise runner.subprocess.CalledProcessError(1,command)
    monkeypatch.setattr(runner.subprocess,'run',fail)
    with pytest.raises(runner.subprocess.CalledProcessError):runner.stage_run(path,'ignored','train')
    assert len(calls)==1


def test_aggregate_verifies_policy_lineage_and_is_immutable(tmp_path,monkeypatch):
    path,q=stub_request(tmp_path,monkeypatch)
    for j in q['train_jobs']:seal_train(path,q,j)
    for j,row in zip(q['evaluation_jobs'],rows(AUDIT)):
        directory=path.parent/'evaluate'/j['id'];write_once(directory/'started.json',{})
        row['policy_sha256']='c'*64
        seal_episode(directory,runner.binding(q,'evaluate',j),{'outcome':row})
    result=runner.aggregate_run(path,'ignored')
    assert read_json(result)['status']=='complete'
    with pytest.raises(FileExistsError):runner.aggregate_run(path,'ignored')


def test_aggregate_rejects_wrong_policy_hash(tmp_path,monkeypatch):
    path,q=stub_request(tmp_path,monkeypatch)
    for j in q['train_jobs']:seal_train(path,q,j)
    for j,row in zip(q['evaluation_jobs'],rows(AUDIT)):
        directory=path.parent/'evaluate'/j['id'];write_once(directory/'started.json',{})
        row['policy_sha256']='d'*64
        seal_episode(directory,runner.binding(q,'evaluate',j),{'outcome':row})
    with pytest.raises(ValueError,match='lineage'):runner.aggregate_run(path,'ignored')
