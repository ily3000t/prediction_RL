from argparse import Namespace
from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools'),str(ROOT/'tests')]
from test_candidate_difference import setup_training
from prediction_rl.data.collection_store import write_once, file_hash as sha
from prediction_rl.data.dataset_contract import digest
from prediction_rl.prediction.candidate_difference import difference_training, DIFFERENCE_TRAINING_VERSION
from prediction_rl.prediction.epoch_training import MemberTrainer, VERSION
from prediction_rl.prediction.model import PredictorEnsemble
from prediction_rl.prediction.trained_artifacts import load_trained_ensemble
import train_candidate_difference as cli
import train_response_predictors as shared


@pytest.fixture(autouse=True,scope='module')
def threads():
    previous=torch.get_num_threads(); torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def draft():
    return json.loads((ROOT/'configs/development/p05_candidate_difference_v1.json').read_text())


def baseline(config):
    original=json.loads((ROOT/'configs/formal/p04_response_dataset_v1.json').read_text())
    return {'version':VERSION,'training':original['training'],'model':original['model']}


@pytest.mark.parametrize('case',['epochs','batch','seed','selection','scale','model','boundary','unknown','ordinary'])
def test_single_variable_config_refuses_unmatched_changes(case):
    config=draft(); original=baseline(config)
    if case=='epochs': config['training']['max_epochs']+=1
    elif case=='batch': config['training']['batch_roots']+=1
    elif case=='seed': config['training']['member_seeds'][0]+=1
    elif case=='selection': config['training']['selection_metric']='combined_objective'
    elif case=='scale': config['training']['channel_scales'][0]=1
    elif case=='model': config['model']['hidden_dim']=32
    elif case=='boundary': config['boundaries']['ddpg_training_authorized']=True
    elif case=='unknown': config['unrecognized']=1
    elif case=='ordinary': config['training']['methods']=['ordinary','conditional']
    with pytest.raises(ValueError): cli.checked_config(config,original)


def test_default_config_is_matched_and_keeps_selection():
    config=draft(); original=baseline(config)
    assert cli.checked_config(config,original)==difference_training(original['training'],.2)


def test_artifact_version_cannot_disguise_changed_objective():
    config=draft()
    with pytest.raises(ValueError,match='cannot hide'):
        shared.artifact_identity({'version':VERSION,'training':config['training']})
    original=baseline(config)
    original['version']=DIFFERENCE_TRAINING_VERSION
    with pytest.raises(ValueError,match='explicit loss identity'):
        shared.artifact_identity(original)


def test_prepare_is_read_only_on_reference_and_never_trains(tmp_path,monkeypatch):
    cfg,t,b,v=setup_training(); reference=tmp_path/'reference'; root=tmp_path/'worktree'
    root.mkdir(); reference.mkdir(); config=draft()
    config['model']=cfg.to_dict(); config['training']=difference_training(t,.2)
    original={'version':VERSION,'model':cfg.to_dict(),'training':t,
              'data_hashes':[b.fingerprint(),v.fingerprint()],'dataset_review':'dataset.json'}
    request_path=reference/config['baseline']['training_request']
    write_once(request_path,original)
    write_once(reference/'dataset.json',{'input_hashes':{}})
    write_once(root/'config.json',config)
    monkeypatch.setattr(cli,'ROOT',root); monkeypatch.setattr(shared,'ROOT',root)
    monkeypatch.setattr(shared,'clean',lambda:None)
    monkeypatch.setattr(shared,'metadata',lambda:{'git_commit':'a'*40,'working_tree_dirty':False})
    monkeypatch.setattr(cli,'review_training',lambda *a:{
        'training_commit':'b'*40,'ensembles':{},
        'input_hashes':{str(request_path.relative_to(reference)):sha(request_path)}})
    monkeypatch.setattr(cli,'load_split',lambda repo,r,split:b if split=='train' else v)
    monkeypatch.setattr(MemberTrainer,'run_epoch',lambda self:pytest.fail('Prepare must never train'))
    before={p.relative_to(reference):sha(p) for p in reference.rglob('*') if p.is_file()}
    path=cli.prepare(Namespace(config='config.json',reference_root=str(reference),run_id='delta'))
    request=json.loads(path.read_text())
    assert request['version']==DIFFERENCE_TRAINING_VERSION and len(request['jobs'])==3
    assert request['training']['member_seeds']==t['member_seeds']
    assert cli.load_request(path)[1]==request
    assert {p.relative_to(reference):sha(p) for p in reference.rglob('*') if p.is_file()}==before
    original['training']['learning_rate']=.01
    request_path.write_text(json.dumps(original))
    with pytest.raises(ValueError,match='reference artifact changed'): cli.load_request(path)


def test_wrong_confirmation_cannot_start_training(tmp_path,monkeypatch):
    monkeypatch.setattr(shared,'clean',lambda:None)
    monkeypatch.setattr(cli,'load_request',lambda path:(tmp_path,{'value':1},None,None,None))
    monkeypatch.setattr(shared,'train_member',lambda *a,**k:pytest.fail('Wrong hash must not train'))
    with pytest.raises(ValueError,match='Confirm'):
        cli.run(Namespace(request='unused',confirm_request_hash='wrong',resume=False))


def completed_trial(tmp_path,monkeypatch):
    cfg,t,b,v=setup_training(); t['member_seeds']=[4101,4102,4103]
    p={'version':DIFFERENCE_TRAINING_VERSION,'model':cfg.to_dict(),
       'training':difference_training(t,.2),'data_hashes':[b.fingerprint(),v.fingerprint()]}
    p['baseline_ensembles']={m:{'path':m+'.pt','sha256':'a'*64,'selected':[]} for m in ('ordinary','conditional')}
    p['jobs']=cli.jobs(p['training'])
    monkeypatch.setattr(cli,'ROOT',tmp_path); monkeypatch.setattr(shared,'ROOT',tmp_path)
    monkeypatch.setattr(shared,'metadata',lambda:{'git_commit':'a'*40,'working_tree_dirty':False})
    out=tmp_path/'artifacts/trial'; out.mkdir(parents=True)
    members=[shared.train_member(out/j['id'],p,j,b,v,resume=False) for j in p['jobs']]
    export=shared.export_ensemble(out,p,'conditional')
    report_path=out/'invocations/test/report.json'
    write_once(report_path,{'version':p['version'],'status':'complete','request_hash':digest(p),
        'working_tree_dirty':False,'members':members,'ensembles':{'conditional':export},
        'calibrated':False,'test_evaluated':False,'ddpg_ready':False})
    return out,p,b,v,report_path,export


def test_delta_export_strict_selection_loading_and_legacy_rejection(tmp_path,monkeypatch):
    out,p,b,v,report,export=completed_trial(tmp_path,monkeypatch)
    model,record=cli.completed_ensemble(out,p,report)
    assert record==export and model.mode=='conditional'
    assert not any(x.requires_grad for x in model.parameters())
    assert shared.export_ensemble(out,p,'conditional')==export
    with pytest.raises(ValueError,match='identity/selection'):
        load_trained_ensemble(tmp_path/export['path'],p,'conditional',export['sha256'],export['selected'])
    for job in p['jobs']:
        assert shared.train_member(out/job['id'],p,job,b,v,resume=True)['status']=='complete'
    (out/'writer.lock').write_text('{}')
    with pytest.raises(ValueError,match='complete immutable'): cli.completed_ensemble(out,p,report)


def test_validation_diagnostics_are_paired_no_test_or_policy_gate(tmp_path,monkeypatch):
    out,p,b,v,report,export=completed_trial(tmp_path,monkeypatch)
    cfg,t,_,_=setup_training(); t['member_seeds']=[4101,4102,4103]
    monkeypatch.setattr(shared,'clean',lambda:None)
    monkeypatch.setattr(cli,'confirmed',lambda args:(out,p,tmp_path,{},
        {'manifest':{'validation':list(v.episode_ids)}}))
    requested=[]
    def split(repo,review,name):
        requested.append(name)
        assert name=='validation'
        return v
    monkeypatch.setattr(cli,'load_split',split)
    monkeypatch.setattr(cli,'load_trained_ensemble',lambda path,request,mode,*a:
        PredictorEnsemble(cfg,mode,t['member_seeds']).eval())
    args=Namespace(request='unused',confirm_request_hash=digest(p),training_report=str(report))
    path=cli.diagnose(args); result=json.loads(path.read_text())
    assert result['status']=='complete' and result['negative_results_are_valid'] is True
    assert result['method_effect_established'] is result['ddpg_ready'] is False
    assert set(result['methods'])=={'ordinary_absolute','conditional_absolute','conditional_difference'}
    assert len(requested)==1 and result['paired_delta_minus_absolute']['ade_m']['paired_roots']==3
    with pytest.raises(ValueError,match='already exists'): cli.diagnose(args)
