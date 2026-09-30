from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import sys

import gym
import numpy as np
import pytest
import torch
from all.environments import State
from all.logging import DummyWriter
from all.presets.continuous import ddpg
from all.experiments import SingleEnvExperiment

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from prediction_rl.training.all_ddpg import (
    preset_factory, resolved_preset, network_shapes, LegacyFeatureView, make_all_environment,
    checkpoint_contract, save_evaluation_weights, load_evaluation_weights, experiment_type,
)
from prediction_rl.data.collection_store import file_hash


def environment(arm='baseline'):
    return SimpleNamespace(feature_arm=arm,state_space=gym.spaces.Box(-np.inf,np.inf,
        (20 if arm=='baseline' else 169,),np.float32),action_space=gym.spaces.Box(-5,5,(1,),np.float32))


def agent(arm='baseline'):
    return preset_factory('cpu',2e-4,smoke=True)(environment(arm),DummyWriter())


def states(n=32):
    for i in range(n):
        yield State(torch.linspace(-1,1,20).view(1,20)+i*.001,
                    torch.tensor([int(i%10!=9)],dtype=torch.uint8),[{}])


def train_synthetic(policy):
    values=[]
    for i,state in enumerate(states()): values.append(policy.act(state,-.1-(i%3)*.01).clone())
    return values


def test_baseline_actions_replay_updates_exactly_match_unmodified_preset():
    torch.manual_seed(777);np.random.seed(777)
    original=ddpg(device='cpu',lr_q=2e-4,lr_pi=2e-4,replay_start_size=8,minibatch_size=8)(environment())
    original_actions=train_synthetic(original)
    torch.manual_seed(777);np.random.seed(777)
    checked=agent();checked_actions=train_synthetic(checked)
    for a,b in zip(original_actions,checked_actions):assert torch.equal(a,b)
    for component in ('q','policy'):
        a=getattr(original.agent,component);b=getattr(checked.agent,component)
        # ALL 0.5.3 declares _updates but never increments it. Use actual Adam steps.
        a_steps=[float(x['step']) for x in a._optimizer.state.values()]
        b_steps=[float(x['step']) for x in b._optimizer.state.values()]
        assert a_steps==b_steps and min(a_steps)>0
        assert a._scheduler.state_dict()==b._scheduler.state_dict()
        for k,v in a.model.state_dict().items():assert torch.equal(v,b.model.state_dict()[k])
        for k,v in a._target._target.state_dict().items():assert torch.equal(v,b._target._target.state_dict()[k])
    assert len(original.agent.replay_buffer)==len(checked.agent.replay_buffer)
    for a,b in zip(original.agent.replay_buffer,checked.agent.replay_buffer):
        assert torch.equal(a[0].features,b[0].features) and torch.equal(a[3].mask,b[3].mask)


def test_matched_expanded_initialization_architecture():
    models=[]
    for arm in ('zero','ordinary','conditional'):
        torch.manual_seed(123);a=agent(arm)
        assert network_shapes(a)=={'policy':[(170,400),(400,300),(300,1)],'q':[(171,400),(400,300),(300,1)]}
        models.append(a)
    for b in models[1:]:
        for component in ('q','policy'):
            for k,v in getattr(models[0].agent,component).model.state_dict().items():
                assert torch.equal(v,getattr(b.agent,component).model.state_dict()[k])


def test_full_recipe_uses_original_defaults_not_smoke_values():
    p=resolved_preset('cpu',2e-4)
    assert p['replay_start_size']==5000 and p['minibatch_size']==100
    assert p['last_frame']==2e6 and p['discount_factor']==.98 and p['noise']==.1
    assert p['replay_buffer_size']==1e6 and p['polyak_rate']==.005 and p['update_frequency']==1
    assert p['lr_pi']==p['lr_q']==2e-4


class Feature:
    mode='zero'
    def __init__(self):
        self.observation_space=environment('zero').state_space
        self.action_space=environment().action_space
        self.calls=0
    def reset(self):return np.zeros(169,np.float32),{'prediction_features':{'history':1}}
    def step(self,a):
        self.calls+=1
        return np.ones(169,np.float32),-.1,True,{'upstream':True,
            'execution_audit':{'traffic':'large'},'prediction_features':{'x':1}}
    def close(self):pass


def test_all_state_reset_dtype_mask_and_lean_replay_info():
    f=Feature();record=[];env=make_all_environment(f,'cpu',lambda *x:record.append(x))
    initial=env.reset();assert initial.features.shape==(1,169) and not initial.done
    state,reward=env.step(torch.tensor([[.7]],dtype=torch.float32))
    assert state.done and reward==-.1 and state.info==[{'upstream':True}]
    assert f.calls==1 and record[-1][-1]['execution_audit']=={'traffic':'large'}
    assert env.env.last_info==record[-1][-1]


@pytest.mark.parametrize('fault',['shape','nan','mask','action','reward'])
def test_invalid_replay_is_rejected_before_storage(fault):
    a=agent();buffer=a.agent.replay_buffer
    s=next(states());n=deepcopy(s);action=torch.zeros(1,1);reward=-.1
    if fault=='shape':n=State(torch.zeros(1,20),n.mask)  # Buffer expects appended time, 21.
    else:
        s=State(torch.zeros(1,21),s.mask);n=deepcopy(s)
        if fault=='nan':n.features[0,0]=float('nan')
        if fault=='mask':n.mask[0]=2
        if fault=='action':action[0,0]=5.001
        if fault=='reward':reward=float('nan')
    with pytest.raises(ValueError):buffer.store(s,action,reward,n)
    assert len(buffer)==0


def contract(arm='baseline'):
    return checkpoint_contract(environment(arm),resolved_preset('cpu',2e-4,smoke=True),
                               '2'*64 if arm in ('ordinary','conditional') else None,'1'*64)


def test_strict_weight_roundtrip_and_no_resume(tmp_path):
    a=agent();train_synthetic(a);path=tmp_path/'weights.pt';c=contract()
    h=save_evaluation_weights(a,path,c);b=agent();load_evaluation_weights(b,path,c,h)
    state,action,_,next_state=next(iter(a.agent.replay_buffer))
    torch.testing.assert_close(a.agent.policy.eval(state),b.agent.policy.eval(state),rtol=0,atol=0)
    torch.testing.assert_close(a.agent.q.eval(state,action),b.agent.q.eval(state,action),rtol=0,atol=0)
    with pytest.raises(RuntimeError):b.act(next(states()),0.)
    with pytest.raises(ValueError):save_evaluation_weights(b,tmp_path/'copy.pt',c)


@pytest.mark.parametrize('fault',['arm','predictor','dimension','preset','hash','missing','shape','nonfinite','foreign'])
def test_checkpoint_faults_never_partially_mutate(tmp_path,fault):
    a=agent('conditional');path=tmp_path/'weights.pt';c=contract('conditional')
    h=save_evaluation_weights(a,path,c);b=agent('conditional')
    before={k:v.clone() for k,v in b.agent.policy.model.state_dict().items()}
    expected=deepcopy(c)
    if fault=='arm':expected['arm']='ordinary'
    if fault=='predictor':expected['predictor_sha256']='3'*64
    if fault=='dimension':expected['observation_dimension']=20
    if fault=='preset':expected['preset']['lr_pi']=1e-3
    if fault=='hash':h='0'*64
    if fault in ('missing','shape','nonfinite','foreign'):
        payload=torch.load(path,weights_only=True)
        if fault=='foreign':payload={'state_dict':payload['state_dicts']['policy']}
        else:
            q=payload['state_dicts']['q'];key=next(iter(q))
            if fault=='missing':del q[key]
            if fault=='shape':q[key]=torch.zeros(1)
            if fault=='nonfinite':q[key].flatten()[0]=float('nan')
        changed=tmp_path/'changed.pt';torch.save(payload,changed);path=changed;h=file_hash(path)
    with pytest.raises(ValueError):load_evaluation_weights(b,path,expected,h)
    assert not b.evaluation_only
    for k,v in b.agent.policy.model.state_dict().items():assert torch.equal(v,before[k])


def test_experiment_training_loop_is_inherited(tmp_path):
    experiment=experiment_type(tmp_path/'writer')
    for name in ('train','test','_run_training_episode','_done'):
        assert getattr(experiment,name)is getattr(SingleEnvExperiment,name)


def test_terminal_bootstrap_mask_preserves_original_zero_q():
    a=agent()
    terminal=State(torch.ones(1,21),torch.zeros(1,dtype=torch.uint8))
    assert not a.agent.q.target(terminal,a.agent.policy.target(terminal)).any()


def test_smoke_config_exact_and_not_formal():
    import json
    sys.path.insert(0,str(ROOT/'tools'))
    from smoke_feature_ddpg import CONFIG
    assert json.loads((ROOT/'configs/development/p06_ddpg_smoke_v1.json').read_text())==CONFIG
    assert CONFIG['formal_training'] is False and CONFIG['test_opened'] is False
    assert CONFIG['requested_frames']+499==563
