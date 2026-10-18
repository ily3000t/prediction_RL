from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from prediction_rl.features.rcci_provider import FrozenRCCIV2Features
from prediction_rl.envs.rcci_features import RCCIV2FeatureEnv
from prediction_rl.features.road_consequences_v2 import VERSION, summarize_road_consequences_v2
from prediction_rl.prediction.interface import PredictorConfig
from prediction_rl.prediction.model import PredictorEnsemble
from prediction_rl.training.all_ddpg import preset_factory
from test_probe_features import Base
from test_road_consequences import scene


def provider(mode='ordinary'):
    cfg = PredictorConfig(hidden_dim=8)
    model = PredictorEnsemble(cfg,mode,[1,2,3]).eval().requires_grad_(False)
    return FrozenRCCIV2Features(ROOT/'RL-MPC-LaneMerging-master/merge.net.xml',cfg,model)


@pytest.mark.parametrize('mode',['ordinary','conditional'])
def test_dedicated_frozen_provider_cached_recompute_exact_no_rng_grad_or_future(mode):
    p = provider(mode); h,_,_,_ = scene(p.road)
    rng = torch.get_rng_state().clone(); before = {k:v.clone() for k,v in p.model.state_dict().items()}
    f = p(h); cache = p.last
    np.testing.assert_array_equal(f,summarize_road_consequences_v2(h,cache['neighbors'],cache['ego'],p.road)['features'])
    poison = deepcopy(h); poison[0].update(future_ids=['oracle'],labels=float('nan'),termination_mask=[True])
    np.testing.assert_array_equal(f,p(poison))
    assert p.calls == 2 and torch.equal(rng,torch.get_rng_state())
    assert all(torch.equal(v,before[k]) for k,v in p.model.state_dict().items())
    assert all(not x.requires_grad and x.grad is None for x in p.model.parameters())


def test_wrapper_reset_terminal_and_raw_preset_refusal():
    from types import SimpleNamespace
    from all.logging import DummyWriter
    p = provider(); h,_,_,_ = scene(p.road); base = Base(); reads = []
    def reader():
        reads.append(1); result = deepcopy(h[0]); result['simulation_time_s'] = (len(reads)-1)*.2
        return result
    env = RCCIV2FeatureEnv(base,'ordinary',p,reader)
    obs,info = env.reset()
    assert env.mode == 'rcci_ordinary' and info['prediction_features']['version'] == VERSION
    assert obs.shape == (152,) and p.calls == 1 and env.action_space is base.action_space
    for i in range(2):
        action = np.array([i+.5],np.float32); obs,reward,done,info = env.step(action)
        np.testing.assert_array_equal(base.actions[-1],action)
        assert reward == -.125 and done == (i == 1)
    assert p.calls == len(reads) == 2 and not obs[3:].any()
    assert info['prediction_features']['terminal_zero_block']
    env.reset(); assert info['prediction_features']['version'] == VERSION
    assert len(env.history) == 1 and p.calls == 3
    with pytest.raises(ValueError,match='Unknown DDPG'):
        preset_factory('cpu',2e-4)(SimpleNamespace(feature_arm=env.mode),DummyWriter())
    env.close(); env.close(); assert base.closed == 1


def test_engineering_fault_cannot_leave_stale_cache_or_step_again(monkeypatch):
    p = provider(); h,_,_,_ = scene(p.road); base = Base()
    reads = []
    def reader():
        reads.append(1); result = deepcopy(h[0]); result['simulation_time_s'] = (len(reads)-1)*.2
        return result
    env = RCCIV2FeatureEnv(base,'ordinary',p,reader); env.reset()
    def fail(*args,**kwargs): raise ValueError('bad model output')
    monkeypatch.setattr(p.model,'forward',fail)
    with pytest.raises(ValueError): env.step(np.array([0.],np.float32))
    assert p.last is None and len(base.actions) == 1 and p.calls == 2
    with pytest.raises(RuntimeError): env.step(np.array([0.],np.float32))
    assert len(base.actions) == 1
    env.close()


def test_unknown_or_unfrozen_provider_refused():
    p = provider(); p.model.train()
    with pytest.raises(ValueError): FrozenRCCIV2Features(ROOT/'RL-MPC-LaneMerging-master/merge.net.xml',p.config,p.model)
    with pytest.raises(ValueError): RCCIV2FeatureEnv(Base(),'ordinary',object(),lambda:{})
