from copy import deepcopy
from pathlib import Path
import sys

import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from prediction_rl.data.actor_adapter import build_actor_inputs
from prediction_rl.data.merge_geometry import MergeGeometry
from prediction_rl.prediction.interface import PredictorConfig, probe_plans
from prediction_rl.prediction.model import PredictorEnsemble
from prediction_rl.prediction.supervision import align_supervision, LABEL_FEATURES
from prediction_rl.prediction.losses import trajectory_loss
from prediction_rl.prediction.development_training import DevelopmentTrainer


@pytest.fixture(autouse=True,scope='module')
def threads():
    previous=torch.get_num_threads(); torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def cfg(): return PredictorConfig(hidden_dim=16)


def fixture_data(cfg):
    def actor(x):
        return {'position':[x,0.], 'speed':7., 'acceleration':0., 'lane_id':'highwayrear_0',
                'lane_position_m':x,'length_m':5.,'width_m':1.8}
    traffic={'simulation_time_s':0.,'vehicles':{'ego':actor(80.),'a':actor(100.),'b':actor(120.)}}
    inputs=build_actor_inputs([traffic],MergeGeometry(ROOT/'RL-MPC-LaneMerging-master/merge.net.xml'))
    history={'schema_version':1,'episode_id':'a'*64,'root_id':'b'*64,'split':'development',
             'history':[traffic],'inputs':inputs,'shared_by_candidates':list(range(5))}
    labels={'actor_ids':['ego','a','b'],'feature_names':LABEL_FEATURES,
            'future':[[[i*20.+t*.2,0.,7.,0.] for t in range(25)] for i in range(3)],
            'future_mask':[[True]*25 for _ in range(3)],'transition_mask':[True]*25,
            'observed_steps':25,'termination_reason':None}
    pack={k:history[k] for k in ('schema_version','episode_id','root_id','split')}
    pack['root_traffic']=deepcopy(traffic)
    pack['candidates']=[{'candidate_id':i,'labels':{**deepcopy(labels),'requested_plan':plan}}
                        for i,plan in enumerate(probe_plans(1,cfg)[0].tolist())]
    manifest={k:[] for k in ('development','train','validation','calibration','test')}
    manifest['development']=[history['episode_id']]
    return [history],[pack],manifest


def batch(cfg): return align_supervision(*fixture_data(cfg),cfg,split='development')


def trainer(cfg,b,mode='conditional'):
    return DevelopmentTrainer(PredictorEnsemble(cfg,mode,[71,72,73]),b)


def test_alignment_stable_under_label_and_candidate_permutation(cfg):
    histories,packs,manifest=fixture_data(cfg)
    first=align_supervision(histories,packs,manifest,cfg,split='development')
    for c in packs[0]['candidates']:
        labels=c['labels']
        for key in ('actor_ids','future','future_mask'): labels[key].reverse()
    packs[0]['candidates'].reverse()
    second=align_supervision(histories,packs,manifest,cfg,split='development')
    assert torch.equal(first.targets,second.targets)
    assert first.fingerprint()==second.fingerprint()
    assert first.targets.shape==(1,5,12,25,4)
    assert first.target_mask.sum()==250
    for i,actor in enumerate(histories[0]['inputs']['actor_ids']):
        if actor is not None: assert first.targets[0,0,i,0,0]=={'a':20.,'b':40.}[actor]
    assert 'targets' not in first.observations


def test_terminal_censoring_preserved_and_not_zero_targets(cfg):
    histories,packs,manifest=fixture_data(cfg)
    for c in packs[0]['candidates']:
        labels=c['labels']; labels['observed_steps']=2; labels['termination_reason']='upstream_collision'
        labels['transition_mask']=[True]*2+[False]*23
        labels['future_mask']=[[True]+[False]*24 for _ in range(3)]
    aligned=align_supervision(histories,packs,manifest,cfg,split='development')
    assert aligned.target_mask.sum()==10
    assert torch.count_nonzero(aligned.targets[:,:,: ,1:])==0
    pred=aligned.targets.clone().requires_grad_()
    loss,_=trajectory_loss(pred,aligned.targets,aligned.target_mask)
    loss.backward()
    assert pred.grad[:,:,:,1:].abs().sum()==0


@pytest.mark.parametrize('case',['root','split','duplicate','missing_candidate','plan','actor','terminal','nan','mask','feature','slot'])
def test_bad_alignment_rejected(cfg,case):
    histories,packs,manifest=fixture_data(cfg)
    labels=packs[0]['candidates'][0]['labels']
    if case=='root': packs[0]['root_traffic']['vehicles']['a']['speed']=9.
    elif case=='split': manifest['train']=manifest['development'][:]
    elif case=='duplicate': histories*=2; packs*=2
    elif case=='missing_candidate': packs[0]['candidates'].pop()
    elif case=='plan': labels['requested_plan'][0]=3.
    elif case=='actor': labels['actor_ids'][1]='missing'
    elif case=='terminal': labels['termination_reason']='upstream_arrival'
    elif case=='nan': labels['future'][0][0][0]=float('nan')
    elif case=='mask': labels['future_mask'][0][0]=1
    elif case=='feature': labels['feature_names']=['wrong']*4
    elif case=='slot': histories[0]['inputs']['actor_ids'][0]='ego'
    with pytest.raises(ValueError): align_supervision(histories,packs,manifest,cfg,split='development')


def test_explicit_split_enforced(cfg):
    histories,packs,manifest=fixture_data(cfg)
    with pytest.raises(ValueError): align_supervision(histories,packs,manifest,cfg,split='train')


def test_loss_equal_root_candidate_not_duration_weighted():
    target=torch.zeros(2,2,1,2,4); pred=target.clone(); mask=torch.zeros(2,2,1,2,dtype=torch.bool)
    scale=torch.tensor([10.,3.,5.,2.])
    pred[0,0]=scale; pred[0,1]=3*scale; pred[1,0]=2*scale
    mask[0,0]=True; mask[0,1,0,0]=True; mask[1,0]=True
    loss,counts=trajectory_loss(pred,target,mask)
    assert loss==pytest.approx(((1+9)/2+4)/2)
    assert counts=={'valid_actor_time_cells':5,'supervised_branches':3,'empty_branches':1,'supervised_roots':2,'empty_roots':0}


def test_loss_masked_garbage_no_gradient_and_empty_batch_errors():
    pred=torch.ones(1,1,1,2,4,requires_grad=True); target=torch.zeros_like(pred)
    target[...,1,:]=9999; mask=torch.tensor([[[[True,False]]]])
    loss,_=trajectory_loss(pred,target,mask); loss.backward()
    assert torch.count_nonzero(pred.grad[...,1,:])==0
    with pytest.raises(ValueError,match='No supervised'): trajectory_loss(pred,target,torch.zeros_like(mask))
    target[...,1,0]=float('nan')
    with pytest.raises(ValueError,match='Nonfinite'): trajectory_loss(pred,target,mask)


def compare_tree(left,right):
    if isinstance(left,torch.Tensor): assert torch.equal(left,right)
    elif isinstance(left,dict):
        assert left.keys()==right.keys()
        for k in left: compare_tree(left[k],right[k])
    elif isinstance(left,list):
        assert len(left)==len(right)
        for a,b in zip(left,right): compare_tree(a,b)
    else: assert left==right


@pytest.mark.parametrize('mode',['ordinary','conditional'])
def test_two_step_resume_exact_all_members_and_adam(cfg,tmp_path,mode):
    b=batch(cfg); first=trainer(cfg,b,mode)
    initial=deepcopy(first.model.state_dict())
    first.step(b)
    path=tmp_path/'state.pt'; first.save(path,{'git_commit':'a'*40})
    resumed=DevelopmentTrainer.load(path,b,expected_mode=mode,expected_config=cfg)
    assert first.step(b)==resumed.step(b)
    compare_tree(first.model.state_dict(),resumed.model.state_dict())
    for a,c in zip(first.optimizers,resumed.optimizers): compare_tree(a.state_dict(),c.state_dict())
    assert any(not torch.equal(initial[k],v) for k,v in first.model.state_dict().items())
    with pytest.raises(ValueError,match='budget exhausted'): resumed.step(b)
    with pytest.raises(FileExistsError): resumed.save(path,{'git_commit':'a'*40})


def test_data_cannot_change_or_be_promoted_after_initialization(cfg):
    b=batch(cfg); t=trainer(cfg,b)
    b.targets[0,0,0,0,0]+=1
    with pytest.raises(ValueError,match='Data/split changed'): t.step(b)
    from dataclasses import replace
    with pytest.raises(ValueError,match='development'): trainer(cfg,replace(b,split='test'))


@pytest.mark.parametrize('mutation',['optimizer_missing','optimizer_nan','optimizer_step','optimizer_lr','data','weights','fixed_scale','mode'])
def test_bad_training_checkpoint_refused(cfg,tmp_path,mutation):
    b=batch(cfg); t=trainer(cfg,b); t.step(b)
    path=tmp_path/'state.pt'; t.save(path,{'git_commit':'a'*40})
    p=torch.load(path,weights_only=True)
    state=p['optimizer_states'][0]['state']
    first=next(iter(state))
    if mutation=='optimizer_missing': state.pop(first)
    elif mutation=='optimizer_nan': state[first]['exp_avg'].view(-1)[0]=float('nan')
    elif mutation=='optimizer_step': state[first]['step']+=1
    elif mutation=='optimizer_lr': p['optimizer_states'][0]['param_groups'][0]['lr']=.1
    elif mutation=='data': p['data_hash']='a'*64
    elif mutation=='weights': p['model_state'].pop(next(iter(p['model_state'])))
    elif mutation=='fixed_scale': p['model_state']['members.0.response_scale'][0]+=1
    elif mutation=='mode': p['mode']='ordinary'
    bad=tmp_path/'bad.pt'; torch.save(p,bad)
    with pytest.raises(ValueError): DevelopmentTrainer.load(bad,b,expected_mode='conditional',expected_config=cfg)
