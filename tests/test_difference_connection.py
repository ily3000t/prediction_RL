from argparse import Namespace
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'tools'), str(ROOT/'tests')]
from test_candidate_difference import setup_training
from prediction_rl.data.collection_store import read_json as read, file_hash as sha
from prediction_rl.prediction.candidate_difference import DIFFERENCE_LOSS_CONTRACT, DIFFERENCE_TRAINING_VERSION, difference_training
from prediction_rl.prediction.difference_artifacts import load_difference_ensemble
from prediction_rl.prediction.trained_artifacts import load_trained_ensemble
import connect_candidate_difference as cli
import train_response_predictors as shared


@pytest.fixture(autouse=True)
def threads():
    previous = torch.get_num_threads(); torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def exported(tmp_path, monkeypatch):
    cfg,t,b,v = setup_training(); t['member_seeds']=[4101,4102,4103]
    p={'version':DIFFERENCE_TRAINING_VERSION,'loss_contract':DIFFERENCE_LOSS_CONTRACT,
       'model':cfg.to_dict(),'training':difference_training(t,.2),
       'data_hashes':[b.fingerprint(),v.fingerprint()]}
    p['jobs']=[{'id':f'conditional_{s}','mode':'conditional','member_seed':s} for s in t['member_seeds']]
    monkeypatch.setattr(shared,'ROOT',tmp_path)
    monkeypatch.setattr(shared,'metadata',lambda:{'git_commit':'a'*40,'working_tree_dirty':False})
    out=tmp_path/'artifacts/trial'; out.mkdir(parents=True)
    for job in p['jobs']: shared.train_member(out/job['id'],p,job,b,v,resume=False)
    record = shared.export_ensemble(out,p,'conditional')
    return out,p,b,v,record


def test_explicit_loader_exact_predictions_and_legacy_rejection(tmp_path, monkeypatch):
    out,p,b,v,record = exported(tmp_path,monkeypatch)
    model = load_difference_ensemble(tmp_path/record['path'],p,record['sha256'],record['selected'])
    assert not model.training and not any(x.requires_grad for x in model.parameters())
    assert torch.isfinite(model(v.observations,v.plans)['mean']).all()
    for member, job in zip(model.members,p['jobs']):
        state = torch.load(out/job['id']/'selected.pt',weights_only=True)['state_dict']
        assert all(torch.equal(value,state[k]) for k,value in member.state_dict().items())
    with pytest.raises(ValueError,match='identity/selection'):
        load_trained_ensemble(tmp_path/record['path'],p,'conditional',record['sha256'],record['selected'])


@pytest.mark.parametrize('fault',['lambda','hash','selection','seed','version','mode'])
def test_loader_refuses_changed_binding(tmp_path,monkeypatch,fault):
    _,p,_,_,record = exported(tmp_path,monkeypatch)
    q = deepcopy(p); selection = deepcopy(record['selected']); h = record['sha256']
    if fault=='lambda': q['training']['difference_lambda']=.3
    elif fault=='hash': h='f'*64
    elif fault=='selection': selection[0]['best_epoch']+=1
    elif fault=='seed': q['training']['member_seeds'][0]+=1
    elif fault=='version': q['version']='legacy'
    else: q['training']['methods']=['ordinary']
    with pytest.raises(ValueError): load_difference_ensemble(tmp_path/record['path'],q,h,selection)


def test_new_calibration_only_uses_calibration_and_does_not_modify_model(tmp_path,monkeypatch):
    _,p,_,v,record = exported(tmp_path,monkeypatch)
    out = tmp_path/'calibration'; out.mkdir()
    b = replace(v,split='calibration')
    q = {'checkpoint':record,'data_hash':b.fingerprint()}
    monkeypatch.setattr(cli,'ROOT',tmp_path)
    monkeypatch.setattr(shared,'clean',lambda:None)
    monkeypatch.setattr(cli,'load_request',lambda *a:(out,q,p,tmp_path,{'manifest':{'calibration':list(dict.fromkeys(b.episode_ids))}}))
    requested=[]
    def split(reference,dataset): requested.append(b.split); return b
    monkeypatch.setattr(cli,'load_calibration_split',split)
    before = sha(tmp_path/record['path'])
    cli.run(Namespace(request='unused',confirm_request_hash='confirmed'))
    r,c = read(out/'report.json'),read(out/'calibration.json')
    assert requested==['calibration'] and r['status']=='complete'
    assert c['checkpoint_sha256']==before and sha(tmp_path/record['path'])==before
    assert c['policy_features_use_bands'] is False and c['test_locked'] is True
    assert r['ddpg_training_started'] is r['test_evaluated'] is False
    with pytest.raises(ValueError,match='already started'): cli.run(Namespace(request='unused',confirm_request_hash='confirmed'))


def test_wrong_confirmation_cannot_fit(monkeypatch,tmp_path):
    monkeypatch.setattr(cli,'ROOT',tmp_path)
    from prediction_rl.data.collection_store import write_once
    write_once(tmp_path/'request.json',{'version':cli.VERSION})
    with pytest.raises(ValueError): cli.load_request('request.json','wrong')
