from copy import deepcopy
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from prediction_rl.features.probe_features import (
    DIMENSION, NAMES, summarize, FrozenProbeFeatures,
)
from prediction_rl.envs.prediction_features import PredictionFeatureEnv
from prediction_rl.prediction.interface import PredictorConfig, collate_inputs, probe_plans
from prediction_rl.prediction.kinematic_reference import LaneConstantVelocity
from prediction_rl.prediction.candidate_ranking import rollout_ego
from prediction_rl.prediction.model import PredictorEnsemble
from prediction_rl.data.actor_adapter import build_actor_inputs
from test_actor_adapter import actor, frame


def scene():
    g = LaneConstantVelocity(ROOT/'RL-MPC-LaneMerging-master/merge.net.xml')
    h = [frame(0., {'ego': actor(pos=100.), 'front':actor(pos=120.), 'rear':actor(pos=90.)})]
    return g, h, build_actor_inputs(h,g.geometry)


def test_exact_layout_and_physical_values():
    g,h,x = scene(); cfg = PredictorConfig()
    prediction = g.predict([{'history':h,'inputs':x}],cfg)[0]
    ego = rollout_ego(h[-1],probe_plans(1,cfg)[0],g)
    f = summarize(x,prediction,ego)
    assert f.shape == (149,) and DIMENSION == len(set(NAMES)) == 149
    slot = x['actor_ids'].index('front')
    assert f[:3] == pytest.approx(((prediction[0,slot,0,:3]-ego[0,0,:3])/torch.tensor([100.,100.,30.])).tolist())
    assert f[-9:] == pytest.approx([1,1/11,1,1/11,1,1/11,2/12,0,0])
    np.testing.assert_array_equal(np.concatenate([summarize(x,prediction[c:c+1],ego[c:c+1])[:24]
                                                 for c in range(5)]),f[:120])
    np.testing.assert_array_equal(np.concatenate([summarize(x,prediction[c:c+1],ego[c:c+1])[24:28]
                                                 for c in range(5)]),f[120:140])


def test_empty_roles_have_zero_values_explicit_masks():
    g,h,_ = scene(); h[0]['vehicles'] = {'ego':h[0]['vehicles']['ego']}
    x = build_actor_inputs(h,g.geometry)
    f = summarize(x,torch.zeros(5,12,25,4),rollout_ego(h[0],probe_plans(1,PredictorConfig())[0],g))
    assert not f[:140].any()
    assert f[-9:] == pytest.approx([0,0,0,0,0,1/11,0,0,0])


@pytest.mark.parametrize('mode',['ordinary','conditional'])
def test_frozen_inference_no_future_leak_rng_or_grad(mode):
    g,h,_ = scene(); cfg = PredictorConfig(hidden_dim=8)
    m = PredictorEnsemble(cfg,mode,[1,2,3]).eval().requires_grad_(False)
    f = FrozenProbeFeatures(g,cfg,m)
    before = {k:v.clone() for k,v in m.state_dict().items()}; rng = torch.get_rng_state()
    first = f(h)
    poisoned = deepcopy(h); poisoned[0].update(labels={'future':[float('nan')]},future_ids=['oracle'],terminal_mask=[False])
    np.testing.assert_array_equal(first,f(poisoned))
    assert torch.equal(rng,torch.get_rng_state())
    assert all(torch.equal(before[k],v) for k,v in m.state_dict().items())
    assert all(p.grad is None for p in m.parameters())
    x = build_actor_inputs(h,g.geometry); b = collate_inputs([x],cfg); plans = probe_plans(1,cfg)
    with torch.inference_mode():
        batch = m(b,plans)['mean']
        for c in range(5):
            torch.testing.assert_close(batch[:,c:c+1],m(b,plans[:,c:c+1])['mean'],rtol=1e-6,atol=1e-6)
        if mode == 'ordinary':
            assert torch.equal(batch[:,0],batch[:,4])
            assert not np.array_equal(first[:24],first[96:120])  # Ego plans differ, neighbors do not.


@pytest.mark.parametrize('bad',['nan','shape','identity','role'])
def test_invalid_prediction_fails(bad):
    g,h,x = scene(); p = torch.zeros(5,12,25,4); ego = rollout_ego(h[0],probe_plans(1,PredictorConfig())[0],g)
    if bad == 'nan': p[0,11,0,0] = float('nan')  # Even masked cells are checked.
    if bad == 'shape': p = p[:,:,:24]
    if bad == 'identity': x['actor_ids'][0] = None
    if bad == 'role': x['mandatory_roles']['target_front'] = 'oracle'
    with pytest.raises(ValueError): summarize(x,p,ego)


class Base:
    execution_contract = 'simulation_blocking_exact_v1'
    def __init__(self):
        from gym.spaces import Box
        self.observation_space = Box(-100,100,(3,),np.float32)
        self.action_space = Box(-5,5,(1,),np.float32)
        self.actions=[]; self.resets=0; self.closed=0
    def reset(self):
        self.resets += 1; self.n=0
        return np.array([1,2,3],np.float32), {'original':True}
    def step(self,action):
        self.actions.append(action.copy()); self.n+=1
        return np.full(3,self.n,np.float32),-.125,self.n==2,{'original':True}
    def close(self): self.closed+=1


class Provider:
    model = SimpleNamespace(mode='conditional')
    def __init__(self): self.lengths=[]; self.fail=False
    def __call__(self,history):
        self.lengths.append(len(history))
        return np.full(DIMENSION,float('nan') if self.fail else len(history),np.float32)


@pytest.mark.parametrize('mode',['baseline','zero','conditional'])
def test_wrapper_prefix_action_reward_reset_terminal(mode):
    base=Base(); provider=Provider(); reads=[]
    def reader(): reads.append(1); return {'observed':len(reads)}
    learned=mode=='conditional'
    env=PredictionFeatureEnv(base,mode,provider if learned else None,reader if learned else None)
    obs,info=env.reset(); np.testing.assert_array_equal(obs[:3],[1,2,3])
    assert env.action_space is base.action_space
    if mode=='zero': assert not obs[3:].any()
    if mode=='baseline': assert env.observation_space is base.observation_space
    for i in range(2):
        action=np.array([i+.5],np.float32); obs,reward,done,info=env.step(action)
        assert reward==-.125 and done==(i==1)
        np.testing.assert_array_equal(base.actions[-1],action)
        np.testing.assert_array_equal(obs[:3],np.full(3,i+1,np.float32))
    assert len(base.actions)==2
    if mode!='baseline': assert not obs[3:].any()
    assert len(reads)==(2 if learned else 0)  # No terminal read/prediction.
    env.reset()
    if learned: assert provider.lengths==[1,2,1]
    env.close(); env.close(); assert base.closed==1


def test_feature_error_faults_without_second_step_or_stale():
    base=Base(); p=Provider(); env=PredictionFeatureEnv(base,'conditional',p,lambda:{})
    env.reset(); p.fail=True
    with pytest.raises(ValueError): env.step(np.array([0.],np.float32))
    assert len(base.actions)==1
    with pytest.raises(RuntimeError): env.reset()
    with pytest.raises(RuntimeError): env.step(np.array([0.],np.float32))
    assert len(base.actions)==1
    env.close()


def test_unfrozen_predictor_refused():
    g,_,_=scene(); cfg=PredictorConfig(hidden_dim=8)
    with pytest.raises(ValueError): FrozenProbeFeatures(g,cfg,PredictorEnsemble(cfg,'ordinary',[1,2,3]))


def test_root_omission_history_and_duplicate_id_contract():
    g,h,_=scene()
    h[0]['vehicles'].update({f'n{i}':actor(pos=130+i) for i in range(14)})
    x=build_actor_inputs(h,g.geometry)
    p=g.predict([{'history':h,'inputs':x}],PredictorConfig())[0]
    ego=rollout_ego(h[0],probe_plans(1,PredictorConfig())[0],g)
    f=summarize(x,p,ego)
    assert f[-3:] == pytest.approx([1,4/16,1])
    x['actor_ids'][1]=x['actor_ids'][0]
    with pytest.raises(ValueError):summarize(x,p,ego)


def test_bounded_audit_config_matches_frozen_schema():
    import json
    sys.path.insert(0,str(ROOT/'tools'))
    from audit_prediction_features import CONFIG
    assert json.loads((ROOT/'configs/development/p06_feature_replay_v1.json').read_text())==CONFIG
    assert sum(r['episodes']*r['step_cap'] for r in CONFIG['replays'])*len(CONFIG['arms'])==1440
