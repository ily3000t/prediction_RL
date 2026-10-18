from copy import deepcopy
import json
import pickle
from pathlib import Path
import subprocess
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
from prediction_rl.training import all_ddpg as raw, rcci_ddpg as rcci
from prediction_rl.features.road_consequences_v2 import contract
from prediction_rl.evaluation.rcci_matched_policy import PROTOCOL,SMOKE,jobs,aggregate,validate_protocol
from prediction_rl.evaluation.matched_learning_curve import METRICS
from prediction_rl.data.collection_store import write_once,seal_episode
import train_rcci_matched_ddpg as cli
from prediction_rl.training.policy_forward_audit import PolicyForwardCounter


def env(arm):
    return SimpleNamespace(feature_arm=arm,state_space=gym.spaces.Box(-np.inf,np.inf,(169,),np.float32),
        action_space=gym.spaces.Box(-5,5,(1,),np.float32),rcci_contract=contract())


def agent(arm='rcci_ordinary'):
    factory=rcci.preset_factory if arm.startswith('rcci_') else raw.preset_factory
    return factory('cpu',.0002,smoke=True)(env(arm),DummyWriter())


def test_config_roster_exact_and_only_new_rcci_training():
    for path,p in ((cli.CONFIG,PROTOCOL),(cli.SMOKE_CONFIG,SMOKE)):
        assert json.loads((ROOT/path).read_text(encoding='utf-8'))==p
    assert len(jobs(PROTOCOL,'train'))==6 and len(jobs(PROTOCOL,'evaluate'))==960
    assert all(j['arm'].startswith('rcci_') for j in jobs(PROTOCOL,'train'))
    assert len(jobs(SMOKE,'train'))==2 and len(jobs(SMOKE,'evaluate'))==4
    assert {j['checkpoint_frames'] for j in jobs(SMOKE,'evaluate') if not j['arm'].startswith('rcci_')}=={120000}
    for key,value in [('unknown',True),('reuse_raw_weights',False)]:
        changed=deepcopy(PROTOCOL); changed[key]=value
        with pytest.raises(ValueError): validate_protocol(changed)


def synthetic(arm):
    torch.manual_seed(871); np.random.seed(871); checked=agent(arm); actions=[]
    for i in range(24):
        state=State(torch.linspace(-1,1,169).view(1,169)+i*.001,
                    torch.tensor([int(i%8!=7)],dtype=torch.uint8),[{}])
        actions.append(checked.act(state,-.1).clone())
    return checked,actions


def test_same_inputs_exact_original_ddpg_actions_rng_replay_targets_and_updates():
    torch.set_num_threads(1)
    a,aa=synthetic('ordinary'); rng=torch.get_rng_state().clone(); nrng=deepcopy(np.random.get_state())
    b,ba=synthetic('rcci_ordinary')
    assert all(torch.equal(x,y) for x,y in zip(aa,ba)) and torch.equal(rng,torch.get_rng_state())
    np.testing.assert_array_equal(nrng[1],np.random.get_state()[1])
    assert len(a.agent.replay_buffer)==len(b.agent.replay_buffer)
    for name in ('policy','q'):
        x,y=getattr(a.agent,name),getattr(b.agent,name)
        for k,v in x.model.state_dict().items(): assert torch.equal(v,y.model.state_dict()[k])
        for k,v in x._target._target.state_dict().items(): assert torch.equal(v,y._target._target.state_dict()[k])
        assert min(float(s['step']) for s in y._optimizer.state.values())>0


def test_mutually_exclusive_identity_strict_roundtrip_and_no_partial_or_resume(tmp_path):
    a,_=synthetic('rcci_ordinary'); c=rcci.checkpoint_contract(env(a.arm),a.preset_parameters,'1'*64,'2'*64)
    path=tmp_path/'policy.pt'; h=rcci.save_evaluation_weights(a,path,c)
    b=agent(); rcci.load_evaluation_weights(b,path,c,h)
    for k,v in a.agent.policy.model.state_dict().items(): assert torch.equal(v,b.agent.policy.model.state_dict()[k])
    with pytest.raises(RuntimeError): b.act(State(torch.zeros(1,169),torch.ones(1,dtype=torch.uint8)),0.)
    with pytest.raises(ValueError): raw.preset_factory('cpu',.0002)(env('rcci_ordinary'),DummyWriter())
    with pytest.raises(ValueError): rcci.preset_factory('cpu',.0002)(env('ordinary'),DummyWriter())
    x=agent('ordinary'); rc=raw.checkpoint_contract(env('ordinary'),x.preset_parameters,'1'*64,'2'*64)
    rp=tmp_path/'raw.pt'; rh=raw.save_evaluation_weights(x,rp,rc); target=agent()
    before={k:v.clone() for k,v in target.agent.policy.model.state_dict().items()}
    with pytest.raises(ValueError): rcci.load_evaluation_weights(target,rp,c,rh)
    for k,v in target.agent.policy.model.state_dict().items(): assert torch.equal(v,before[k])
    wrong=deepcopy(c); wrong['extra']=True
    with pytest.raises(ValueError): rcci.validate(target,wrong)


def rows(p):
    return [{**j,**{m:0. for m in METRICS},'natural_arrival':1,'ego_collision':0,'time_limit':0,
        'steps':3,'no_policy_updates':True,'native_event_agrees':True,'actual_training_steps':j['checkpoint_frames'],
        'initial_traffic_sha256':str(j['simulator_seed']).zfill(64)} for j in jobs(p,'evaluate')]


def test_negative_effects_keep_all_seeds_complete_and_interaction_is_paired():
    r=rows(PROTOCOL)
    rewards={'ordinary':0.,'conditional':1.,'rcci_ordinary':2.,'rcci_conditional':0.}
    for x in r: x['return']=rewards[x['arm']]
    a=aggregate(PROTOCOL,r)
    assert a['engineering_complete'] and a['method_effect_gate'] is None and a['primary_node']==120000
    assert a['nodes'][-1]['comparisons']['b3_interface']['equal_run_seed_mean_delta']['return']==-1.
    assert a['nodes'][-1]['interface_conditioning_interaction']['equal_run_seed_mean']['return']==-3.
    assert a['evaluation_episodes']==960 and a['unique_traffic_scenarios']==20
    for fault in ('missing','duplicate','scene','nan'):
        changed=deepcopy(r)
        if fault=='missing': changed.pop()
        elif fault=='duplicate': changed[-1]=deepcopy(changed[0])
        elif fault=='scene': changed[-1]['initial_traffic_sha256']='f'*64
        else: changed[-1]['return']=float('nan')
        with pytest.raises(ValueError): aggregate(PROTOCOL,changed)
    smoke=aggregate(SMOKE,rows(SMOKE)); assert not smoke['method_comparison_allowed']


def test_fast_receipt_bound_to_complete_index_and_extra_files_rejected(tmp_path):
    q={'x':'frozen'}; model='rcci_ordinary_s0'; d=tmp_path/'train'/model; d.mkdir(parents=True)
    b={'stage':'train'}; m={'result':'complete'}
    write_once(d/'started.json',{}); write_once(d/'report.json',m); sealed=seal_episode(d,b,m)
    write_once(tmp_path/'training_index.json',{'request_hash':cli.digest(q),'models':{
        model:{'receipt_sha256':cli.sha(d/'complete.json'),'metadata':m}}})
    write_once(tmp_path/'training_index_receipt.json',{'request_hash':cli.digest(q),'index_sha256':cli.sha(tmp_path/'training_index.json')})
    assert cli.fast_training_receipt(q,tmp_path,d,b)==sealed
    write_once(d/'extra.json',{})
    with pytest.raises(ValueError): cli.fast_training_receipt(q,tmp_path,d,b)


def test_formal_prepare_requires_entry_acceptance_before_assets(monkeypatch):
    monkeypatch.setattr(cli.shared,'clean',lambda:None)
    monkeypatch.setattr(cli,'read',lambda *a:deepcopy(PROTOCOL))
    monkeypatch.setattr(cli,'parent_inputs',lambda *a:pytest.fail('Opened assets without entry acceptance'))
    with pytest.raises(ValueError): cli.prepare(cli.CONFIG,'unused','formal')


def test_new_provider_reuses_exact_cached_semantics_and_ignores_future_labels():
    from test_rcci_provider import provider
    from test_road_consequences import scene
    old=provider(); new=rcci.MatchedRCCIProvider(ROOT/'RL-MPC-LaneMerging-master/merge.net.xml',old.config,old.model)
    h,_,_,_=scene(old.road); before=torch.get_rng_state().clone()
    np.testing.assert_array_equal(old(h),new(h))
    poison=deepcopy(h); poison[0].update(future_lane_ids=['oracle'],labels=float('nan'),future_reward=999.)
    np.testing.assert_array_equal(new(h),new(poison))
    assert torch.equal(before,torch.get_rng_state()) and all(p.grad is None for p in new.model.parameters())


def test_stale_source_rejected_before_assets(tmp_path,monkeypatch):
    monkeypatch.setattr(cli,'ROOT',tmp_path); monkeypatch.setattr(cli.shared,'clean',lambda:None)
    monkeypatch.setattr(cli.shared,'sources',lambda:{'new':'hash'})
    q={'protocol':deepcopy(SMOKE),'version':'rcci04c_request_v1','source_hashes':{'old':'hash'}}
    monkeypatch.setattr(cli,'read',lambda *a:q)
    monkeypatch.setattr(cli,'sha',lambda *a:pytest.fail('Assets opened for stale source'))
    with pytest.raises(ValueError): cli.load_request(tmp_path/'artifacts/rcci04c/stale/request.json',cli.digest(q))


def test_changed_installed_sources_rejected_before_assets(tmp_path,monkeypatch):
    monkeypatch.setattr(cli,'ROOT',tmp_path); monkeypatch.setattr(cli.shared,'clean',lambda:None)
    monkeypatch.setattr(cli.shared,'sources',lambda:{'same':'hash'})
    monkeypatch.setattr(cli.shared,'environment',lambda:{'env':'same'})
    monkeypatch.setattr(cli,'installed_sources',lambda:{'agents/ddpg.py':'changed'})
    q={'protocol':deepcopy(SMOKE),'version':'rcci04c_request_v1','source_hashes':{'same':'hash'},
       'environment':{'env':'same'},'dependencies':{'agents/ddpg.py':'old'}}
    monkeypatch.setattr(cli,'read',lambda *a:q)
    monkeypatch.setattr(cli,'sha',lambda *a:pytest.fail('Assets opened for changed installed algorithm'))
    with pytest.raises(ValueError): cli.load_request(tmp_path/'artifacts/rcci04c/stale/request.json',cli.digest(q))


def test_forward_counter_pickle_cap_and_no_rng_side_effects():
    counter=PolicyForwardCounter(2); rng=torch.get_rng_state().clone()
    assert counter(None,()) is None
    restored=pickle.loads(pickle.dumps(counter)); assert restored.calls==1 and restored.limit==2
    assert restored(None,()) is None
    with pytest.raises(ValueError,match='forward cap'): restored(None,())
    assert restored.calls==2 and torch.equal(rng,torch.get_rng_state())
    with pytest.raises(ValueError): PolicyForwardCounter(0)


@pytest.mark.parametrize('arm',['rcci_ordinary','rcci_conditional'])
def test_runtime_hooks_cross_original_periodic_save_without_changing_learning(tmp_path,arm):
    """Exercise actual ALL save frequency; the old 16-step SUMO smoke missed it."""
    torch.set_num_threads(1)
    def execute(instrumented):
        directory=tmp_path/('audited' if instrumented else 'plain'); directory.mkdir()
        writer=DummyWriter(); writer.log_dir=str(directory)
        torch.manual_seed(871); np.random.seed(871)
        rt=None
        if instrumented:
            rt=object.__new__(cli.Runtime); rt.job={'arm':arm}; rt.stage='evaluate'
            rt.policy_counter=PolicyForwardCounter(); rt.hooks=[]
            checked=rt.factory('cpu',.0002,smoke=True)(env(arm),writer)
        else:
            checked=rcci.preset_factory('cpu',.0002,smoke=True)(env(arm),writer)
        frequency=checked.agent.policy._checkpointer.frequency
        assert frequency==checked.agent.q._checkpointer.frequency==200
        actions=[]
        for i in range(frequency+40):
            state=State(torch.linspace(-1,1,169).view(1,169)+i*.001,
                torch.tensor([int(i%8!=7)],dtype=torch.uint8),[{}])
            actions.append(checked.act(state,-.1).clone())
        # Both untouched ALL checkpointers really serialized the full model.
        assert all((directory/(n+'.pt')).stat().st_size>1000 for n in ('policy','q'))
        assert checked.agent.policy._checkpointer._updates>frequency
        return checked,actions,torch.get_rng_state().clone(),deepcopy(np.random.get_state()),rt,directory
    plain,pa,pr,pn,_,_=execute(False)
    checked,actions,rng,nrng,rt,directory=execute(True)
    assert all(torch.equal(a,b) for a,b in zip(pa,actions)) and torch.equal(pr,rng)
    assert pn[0]==nrng[0] and pn[2:]==nrng[2:]
    np.testing.assert_array_equal(pn[1],nrng[1])
    assert rt.policy_counter.calls>len(actions)  # Includes policy update and target forwards.
    for name in ('policy','q'):
        a,b=getattr(plain.agent,name),getattr(checked.agent,name)
        for k,v in a.model.state_dict().items(): assert torch.equal(v,b.model.state_dict()[k])
        for k,v in a._target._target.state_dict().items(): assert torch.equal(v,b._target._target.state_dict()[k])
        assert a._optimizer.state_dict()['param_groups']==b._optimizer.state_dict()['param_groups']
        for k,values in a._optimizer.state_dict()['state'].items():
            for n,v in values.items(): assert torch.equal(v,b._optimizer.state_dict()['state'][k][n])
    # Hooks on target models must also serialize; test importability in a fresh interpreter.
    torch.save(checked.agent.policy._target._target,directory/'target.pt')
    child="""import sys,torch
sys.path.insert(0,sys.argv[1])
from prediction_rl.training.policy_forward_audit import PolicyForwardCounter
for path in sys.argv[2:]:
    model=torch.load(path,map_location='cpu',weights_only=False)
    assert len(model._forward_pre_hooks)==1
    hook=next(iter(model._forward_pre_hooks.values()))
    assert isinstance(hook,PolicyForwardCounter) and hook.calls>0 and hook.limit is None
    assert all(torch.isfinite(v).all() for v in model.state_dict().values())
"""
    subprocess.run([sys.executable,'-B','-c',child,str(ROOT/'src'),
        str(directory/'policy.pt'),str(directory/'target.pt')],check=True,timeout=60)
    for handle in rt.hooks: handle.remove()
