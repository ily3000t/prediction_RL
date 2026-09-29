from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import sys

import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
from test_masked_training import batch, compare_tree
from prediction_rl.prediction.interface import PredictorConfig
from prediction_rl.prediction.epoch_training import MemberTrainer, subset, selection_update, evaluate, initial_member
from prediction_rl.data.frozen_dataset import inside, load_split, verify_historical_inputs
from prediction_rl.data.branching import fingerprint
from prediction_rl.data.dataset_contract import digest
import train_response_predictors as cli


@pytest.fixture(autouse=True,scope='module')
def threads():
    previous=torch.get_num_threads(); torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def setup():
    cfg=PredictorConfig(hidden_dim=16)
    b=subset(batch(cfg),[0,0,0])
    b=replace(b,root_ids=('r0','r1','r2'),episode_ids=('e0','e1','e2'),split='train')
    b.targets[1]+=1.; b.targets[2]+=2.
    val=replace(deepcopy(b),root_ids=('v0','v1','v2'),episode_ids=('v0','v1','v2'),split='validation')
    t=json.loads((ROOT/'configs/formal/p04_response_dataset_v1.json').read_text())['training']
    t.update(max_epochs=2,early_stop_patience=2,batch_roots=2)
    return cfg,t,b,val


def make(s,mode='conditional'):
    c,t,b,v=s
    return MemberTrainer(c,mode,4101,t,b,v)


@pytest.mark.parametrize('mode',['ordinary','conditional'])
def test_epoch_resume_exact_rng_optimizer_best_and_budget(setup,tmp_path,mode):
    a=make(setup,mode); rng=torch.get_rng_state().clone()
    a.run_epoch(); path=tmp_path/'e1.pt'; a.save(path,{'run':'test'})
    c,t,b,v=setup
    resumed=MemberTrainer.load(path,c,mode,4101,t,b,v,{'run':'test'})
    assert a.run_epoch()==resumed.run_epoch()
    compare_tree(a.model.state_dict(),resumed.model.state_dict())
    compare_tree(a.optimizer.state_dict(),resumed.optimizer.state_dict())
    compare_tree(a.best_state,resumed.best_state)
    assert torch.equal(rng,torch.get_rng_state())
    with pytest.raises(ValueError,match='budget exhausted'): resumed.run_epoch()
    with pytest.raises(FileExistsError): resumed.save(path,{'run':'test'})


def test_paired_initialization_sampling_and_ordinary_invariance(setup):
    a,b=make(setup,'ordinary'),make(setup,'conditional')
    compare_tree(a.model.state_dict(),b.model.state_dict())
    assert a.run_epoch()['root_order']==b.run_epoch()['root_order']
    with torch.inference_mode():
        y=a.model(setup[3].observations,setup[3].plans)
        assert torch.equal(y,y[:,:1].expand_as(y))


def test_small_improvement_selects_best_but_not_patience_and_tie_earliest():
    s={'best_loss':None,'best_epoch':None,'patience_anchor':None,'bad_epochs':0}
    assert selection_update(s,1.,1,.0001)
    assert selection_update(s,.99999,2,.0001)
    assert s['best_epoch']==2 and s['bad_epochs']==1
    assert not selection_update(s,.99999,3,.0001)
    assert s['best_epoch']==2 and s['bad_epochs']==2
    with pytest.raises(ValueError): selection_update(s,float('nan'),4,.0001)


def test_validation_root_weighted_with_short_last_batch(setup):
    c,t,b,v=setup
    m=initial_member(c,'ordinary',4101)
    individual=[evaluate(m,subset(v,[i]),1) for i in range(3)]
    assert evaluate(m,v,2)==pytest.approx(sum(individual)/3,rel=1e-6)


def test_empty_roots_preserved_but_excluded_from_updates(setup):
    setup[2].target_mask[1]=False; setup[3].target_mask[2]=False
    a=make(setup); row=a.run_epoch()
    assert set(row['root_order'])=={0,2} and a.steps==1
    assert row['excluded_empty_train_roots']==row['excluded_empty_validation_roots']==1
    setup[2].target_mask[:]=False
    with pytest.raises(ValueError,match='Empty'): make(setup)


@pytest.mark.parametrize('case',['overlap','calibration','test','manifest'])
def test_split_leakage_refused(setup,case):
    c,t,b,v=setup
    if case=='overlap': v=replace(v,episode_ids=b.episode_ids)
    elif case=='manifest': v=replace(v,manifest_hash='other')
    else: v=replace(v,split=case)
    with pytest.raises(ValueError): MemberTrainer(c,'ordinary',4101,t,b,v)


@pytest.mark.parametrize('case',['seed','data','missing_weight','nan_weight','fixed_buffer','optimizer_missing',
    'optimizer_nan','optimizer_step','optimizer_lr','sampler','root_order','selection','best_nan'])
def test_corrupted_epoch_refused(setup,tmp_path,case):
    a=make(setup); a.run_epoch(); path=tmp_path/'e1.pt'; a.save(path,{'run':'test'})
    p=torch.load(path,weights_only=True); opt=p['optimizer']; ident=next(iter(opt['state']))
    if case=='seed': p['seed']+=1
    elif case=='data': p['data_hashes'][0]='bad'
    elif case=='missing_weight': p['model'].pop(next(iter(p['model'])))
    elif case=='nan_weight': p['model']['response_head.0.weight'][0,0]=float('nan')
    elif case=='fixed_buffer': p['model']['response_scale'][0]+=1
    elif case=='optimizer_missing': opt['state'].pop(ident)
    elif case=='optimizer_nan': opt['state'][ident]['exp_avg'].flatten()[0]=float('nan')
    elif case=='optimizer_step': opt['state'][ident]['step']=3
    elif case=='optimizer_lr': opt['param_groups'][0]['lr']=.2
    elif case=='sampler': p['sampler_rng']=torch.Generator().manual_seed(10).get_state()
    elif case=='root_order': p['history'][0]['root_order'].reverse()
    elif case=='selection': p['selection']['best_loss']+=1
    elif case=='best_nan': p['best_model']['response_scale'][0]=float('nan')
    bad=tmp_path/'bad.pt'; torch.save(p,bad)
    c,t,b,v=setup
    with pytest.raises(ValueError): MemberTrainer.load(bad,c,'conditional',4101,t,b,v,{'run':'test'})


def test_historical_sources_are_git_blobs_not_current_imports(tmp_path,monkeypatch):
    import prediction_rl.data.frozen_dataset as frozen
    def read_git(cmd):
        assert cmd[:3]==['git','-C',str(tmp_path)]
        return b'print(1)\r\n' if cmd[-1].endswith('old.py') else b'{"a":1}\r\n'
    monkeypatch.setattr(frozen.subprocess,'check_output',read_git)
    verify_historical_inputs(tmp_path,{'python_lf:old.py':fingerprint('print(1)\n'),
        'json_canonical:old.json':digest({'a':1})},'a'*40)
    with pytest.raises(ValueError): verify_historical_inputs(tmp_path,{'python_lf:old.py':'bad'},'a'*40)
    with pytest.raises(ValueError): inside(tmp_path,'../escape')


@pytest.mark.parametrize('split',['calibration','test','development'])
def test_training_loader_cannot_open_other_splits(tmp_path,split):
    with pytest.raises(ValueError,match='cannot read'): load_split(tmp_path,{},split)


def test_exact_request_confirmation_precedes_training(tmp_path,monkeypatch):
    from argparse import Namespace
    monkeypatch.setattr(cli,'clean',lambda:None)
    monkeypatch.setattr(cli,'load_request',lambda path:(tmp_path,{'key':1},{}))
    monkeypatch.setattr(cli,'load_split',lambda *a:pytest.fail('Must not load training data'))
    with pytest.raises(ValueError,match='Confirm'): cli.run(Namespace(request='unused',confirm_request_hash='wrong'))


def test_member_run_completion_and_verified_reuse(setup,tmp_path,monkeypatch):
    c,t,b,v=setup
    p={'model':c.to_dict(),'training':t,'data_hashes':[b.fingerprint(),v.fingerprint()]}
    job={'id':'conditional_4101','mode':'conditional','member_seed':4101}
    monkeypatch.setattr(cli,'metadata',lambda:{'git_commit':'a'*40})
    out=tmp_path/job['id']; result=cli.train_member(out,p,job,b,v,resume=False)
    assert result['epoch']==2 and result['optimizer_steps']==4
    assert cli.train_member(out,p,job,b,v,resume=True)==result
    with pytest.raises(ValueError): cli.train_member(out,p,job,b,v,resume=False)
    payload=torch.load(out/'selected.pt',weights_only=True)
    assert payload['purpose']=='formal_selected_member' and payload['best_epoch'] in (1,2)
    (out/'history.json').write_text('changed')
    with pytest.raises(ValueError): cli.train_member(out,p,job,b,v,resume=True)


def test_interrupted_epoch_replays_from_last_commit(setup,tmp_path,monkeypatch):
    c,t,b,v=setup
    p={'model':c.to_dict(),'training':t,'data_hashes':[b.fingerprint(),v.fingerprint()]}
    job={'id':'conditional_4101','mode':'conditional','member_seed':4101}
    monkeypatch.setattr(cli,'metadata',lambda:{'git_commit':'a'*40})
    original=MemberTrainer.run_epoch
    def interrupted(self):
        if self.epoch==1: raise KeyboardInterrupt('bounded test interruption')
        return original(self)
    monkeypatch.setattr(MemberTrainer,'run_epoch',interrupted)
    with pytest.raises(KeyboardInterrupt): cli.train_member(tmp_path/'resume',p,job,b,v,resume=False)
    monkeypatch.setattr(MemberTrainer,'run_epoch',original)
    resumed=cli.train_member(tmp_path/'resume',p,job,b,v,resume=True)
    straight=cli.train_member(tmp_path/'straight',p,job,b,v,resume=False)
    assert resumed==straight
    compare_tree(torch.load(tmp_path/'resume/e0002.pt',weights_only=True),
                 torch.load(tmp_path/'straight/e0002.pt',weights_only=True))


def test_three_member_export_is_uncalibrated_and_idempotent(setup,tmp_path,monkeypatch):
    c,t,b,v=setup
    t['member_seeds']=[4101,4102,4103]
    p={'model':c.to_dict(),'training':t,'data_hashes':[b.fingerprint(),v.fingerprint()],
       'jobs':[{'id':f'ordinary_{seed}','mode':'ordinary','member_seed':seed} for seed in t['member_seeds']]}
    monkeypatch.setattr(cli,'ROOT',tmp_path)
    monkeypatch.setattr(cli,'metadata',lambda:{'git_commit':'a'*40})
    for job in p['jobs']: cli.train_member(tmp_path/job['id'],p,job,b,v,resume=False)
    first=cli.export_ensemble(tmp_path,p,'ordinary')
    assert first==cli.export_ensemble(tmp_path,p,'ordinary')
    payload=torch.load(tmp_path/'ordinary_ensemble.pt',weights_only=True)
    assert payload['calibrated'] is False and payload['test_evaluated'] is False
    assert payload['member_seeds']==[4101,4102,4103]
    assert len(payload['selected'])==3


def test_prepare_never_starts_training(setup,tmp_path,monkeypatch):
    from argparse import Namespace
    from prediction_rl.data.collection_store import write_once
    c,t,b,v=setup
    monkeypatch.setattr(cli,'ROOT',tmp_path)
    monkeypatch.setattr(cli,'clean',lambda:None)
    monkeypatch.setattr(cli,'metadata',lambda:{'git_commit':'a'*40})
    review_path=tmp_path/'review.json'; review_path.write_text('{}')
    review={'protocol':{'training':t,'model':c.to_dict()}}
    monkeypatch.setattr(cli,'verify_review',lambda path:(review_path,review))
    monkeypatch.setattr(cli,'load_split',lambda root,r,split: b if split=='train' else v)
    monkeypatch.setattr(MemberTrainer,'run_epoch',lambda self:pytest.fail('Prepare must never train'))
    write_once(tmp_path/'audit.json',{'status':'complete','epoch_training_gate':True,
        'training_purpose':'development_smoke','source_hashes':cli.source_hashes(),'environment':cli.env()})
    request=cli.prepare(Namespace(dataset_review='review.json',training_audit='audit.json',run_id='test'))
    payload=json.loads(request.read_text())
    assert len(payload['jobs'])==6 and payload['test_locked'] is True
    assert sorted(p.name for p in request.parent.iterdir())==['preparation.json','request.json']
