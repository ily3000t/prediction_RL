from copy import deepcopy
from dataclasses import replace
from argparse import Namespace
import json
from pathlib import Path
import sys

import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
from test_masked_training import batch
from prediction_rl.prediction.interface import PredictorConfig
from prediction_rl.prediction.model import PredictorEnsemble
from prediction_rl.prediction.epoch_training import subset, VERSION
from prediction_rl.prediction.checkpoint import CONTRACT
from prediction_rl.prediction.calibration import (SETTINGS,CHANNEL_SCALES,standardized_residuals,
    episode_scores,fit_episode_quantile,band_diagnostics,load_calibration_split)
from prediction_rl.prediction.trained_artifacts import load_trained_ensemble,checked_history
from prediction_rl.data.collection_store import file_hash,write_once
from prediction_rl.data.dataset_contract import digest
import calibrate_response_predictors as cli


@pytest.fixture(autouse=True,scope='module')
def threads():
    previous=torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


class ZeroEnsemble(torch.nn.Module):
    def forward(self,observations,plans):
        assert 'targets' not in observations
        return {'member_trajectories':torch.zeros(3,plans.shape[0],5,12,25,4)}


def synthetic():
    b=subset(batch(PredictorConfig(hidden_dim=16)),[0,0,0,0])
    b=replace(b,root_ids=('r0','r1','r2','r3'),episode_ids=('a','a','b','c'),split='calibration')
    floor=torch.tensor(CHANNEL_SCALES)*.1
    for i in range(4): b.targets[i]=(i+1)*floor
    b.target_mask[3]=False
    return b


def test_population_disagreement_and_channel_floor():
    y=torch.zeros(1,1,1,1,4); pred=torch.stack([y-2,y,y+2]); mask=torch.ones(y.shape[:-1],dtype=torch.bool)
    scores,scale=standardized_residuals(pred,y+3,mask)
    assert torch.allclose(scale,torch.full_like(scale,(8/3)**.5))
    assert torch.allclose(scores,3/scale)
    _,scale=standardized_residuals(torch.zeros_like(pred),y,mask)
    assert scale.flatten().tolist()==pytest.approx([1.,.3,.5,.2])


@pytest.mark.parametrize('case',['nan','mask','shape','dtype'])
def test_invalid_prediction_inputs_stop_even_when_masked(case):
    y=torch.zeros(1,1,1,1,4); pred=torch.stack([y,y,y]); mask=torch.zeros(y.shape[:-1],dtype=torch.bool)
    if case=='nan':pred.flatten()[0]=float('nan')
    elif case=='mask':mask=mask.float()
    elif case=='shape':pred=pred[:2]
    elif case=='dtype':pred=pred.double()
    with pytest.raises(ValueError): standardized_residuals(pred,y,mask)


def test_episode_not_root_or_candidate_is_calibration_unit():
    b=synthetic(); model=ZeroEnsemble().eval()
    rows,roots=episode_scores(model,b,['a','b','c','missing'],batch_roots=2)
    assert len(rows)==4 and len(roots)==4
    assert rows[0]['score']==pytest.approx(2.) and rows[0]['roots']==2
    assert rows[1]['score']==pytest.approx(3.)
    assert rows[2]['score']is None and rows[2]['empty_roots']==1
    assert rows[3]['score']is None and rows[3]['roots']==0
    fit=fit_episode_quantile(rows,SETTINGS)
    assert fit['eligible_episodes']==2 and fit['excluded_episode_ids']==['c','missing']
    assert fit['unbounded'] and fit['q']is None
    assert (rows,roots)==episode_scores(model,b,['a','b','c','missing'],batch_roots=1)


@pytest.mark.parametrize('split',['train','validation','test','development'])
def test_formal_calibration_cannot_fit_other_splits(split):
    with pytest.raises(ValueError,match='cannot consume'):
        episode_scores(ZeroEnsemble().eval(),replace(synthetic(),split=split),['a','b','c'])


def test_episode_roster_and_training_mode_are_strict():
    for ids in (['a','b'],['a','b','c','c']):
        with pytest.raises(ValueError):episode_scores(ZeroEnsemble().eval(),synthetic(),ids)
    with pytest.raises(ValueError):episode_scores(ZeroEnsemble(),synthetic(),['a','b','c'])


def test_64_episode_order_is_59_no_interpolation():
    rows=[{'episode_id':str(i),'score':float(i)} for i in range(64)]
    fit=fit_episode_quantile(rows,SETTINGS)
    assert fit['order_statistic_one_based']==59 and fit['q']==58.
    assert fit['fitted_sample_coverage']==59/64 and fit['empirical_test_coverage']is None
    assert fit_episode_quantile(list(reversed(rows)),SETTINGS)==fit


def test_ties_small_sample_and_empty_cases():
    rows=[{'episode_id':str(i),'score':2.} for i in range(64)]
    assert fit_episode_quantile(rows,SETTINGS)['fitted_sample_coverage']==1.
    assert fit_episode_quantile(rows[:3],SETTINGS)['unbounded']
    with pytest.raises(ValueError):fit_episode_quantile([],SETTINGS)
    with pytest.raises(ValueError):fit_episode_quantile(rows[:1]*2,SETTINGS)
    with pytest.raises(ValueError):fit_episode_quantile([{'episode_id':'a','score':float('nan')}],SETTINGS)
    with pytest.raises(ValueError):fit_episode_quantile(rows,{**SETTINGS,'nominal_coverage':.95})


def test_fit_diagnostic_is_censor_aware_and_not_test_evidence():
    b=synthetic(); fit={'settings':SETTINGS,'q':2.000001,'unbounded':False}
    d=band_diagnostics(ZeroEnsemble().eval(),b,fit,batch_roots=2)
    assert d['coordinate_coverage']==pytest.approx(2/3)
    assert d['mean_band_width_by_channel']==pytest.approx([4.000002,1.2000006,2.000001,.8000004])
    assert d['scope']=='calibration_fit_sample_only' and d['censored_tail_coverage']=='not_identifiable'


@pytest.fixture
def ensemble(tmp_path):
    cfg=PredictorConfig(hidden_dim=16); seeds=[4101,4102,4103]
    request={'model':cfg.to_dict(),'training':{'member_seeds':seeds},'data_hashes':['train','validation']}
    selected=[{'seed':s,'best_epoch':10,'validation_loss':.1,'member_sha256':'a'*64} for s in seeds]
    m=PredictorEnsemble(cfg,'conditional',seeds)
    p={'version':VERSION,'purpose':'formal_validation_selected_ensemble','contract':CONTRACT,
       'request_hash':digest(request),'model':request['model'],'mode':'conditional','member_seeds':seeds,
       'selected':selected,'data_hashes':request['data_hashes'],'state_dict':m.state_dict(),'calibrated':False,'test_evaluated':False}
    path=tmp_path/'ensemble.pt';torch.save(p,path)
    return path,request,selected,p


def test_trained_loader_freezes_complete_three_member_model(ensemble):
    path,p,s,_=ensemble
    model=load_trained_ensemble(path,p,'conditional',file_hash(path),s)
    assert not model.training and not any(x.requires_grad for x in model.parameters())
    assert len(model.members)==3


@pytest.mark.parametrize('case',['hash','purpose','mode','selection','weights','nan','buffer','shape','dtype','extra'])
def test_trained_loader_rejects_partial_or_wrong_models(ensemble,case):
    path,r,s,p=ensemble
    if case=='purpose':p['purpose']='untrained_interface_smoke'
    elif case=='mode':p['mode']='ordinary'
    elif case=='selection':p['selected']=[]
    elif case=='weights':p['state_dict'].pop(next(iter(p['state_dict'])))
    elif case=='nan':p['state_dict']['members.0.response_scale'][0]=float('nan')
    elif case=='buffer':p['state_dict']['members.0.response_scale'][0]+=1.
    elif case=='shape':p['state_dict']['members.0.response_scale']=torch.ones(3)
    elif case=='dtype':p['state_dict']['members.0.response_scale']=p['state_dict']['members.0.response_scale'].double()
    elif case=='extra':p['unknown']=True
    torch.save(p,path)
    with pytest.raises(ValueError):load_trained_ensemble(path,r,'conditional','bad' if case=='hash' else file_hash(path),s)


def test_complete_training_history_rejects_premature_and_wrong_selection():
    t={'batch_roots':2,'max_epochs':2,'early_stop_patience':2,'early_stop_min_delta':.01}
    history=[{'epoch':1,'optimizer_steps':2,'train_pre_update_loss':1.,'validation_loss':1.,'selected_best':True},
             {'epoch':2,'optimizer_steps':4,'train_pre_update_loss':.5,'validation_loss':.999,'selected_best':True}]
    m={'epoch':2,'optimizer_steps':4,'stop_reason':'epoch_budget','best_epoch':2,'best_validation_loss':.999}
    assert checked_history(history,m,t,3)['best_epoch']==2
    with pytest.raises(ValueError):checked_history(history,{**m,'best_epoch':1},t,3)
    with pytest.raises(ValueError):checked_history(history[:1],{**m,'epoch':1,'optimizer_steps':2},t,3)


def test_calibration_load_does_not_silently_open_other_data(tmp_path):
    with pytest.raises(ValueError):load_calibration_split(tmp_path,{'test_locked':False})


def test_confirmation_checked_before_any_formal_prediction(tmp_path,monkeypatch):
    monkeypatch.setattr(cli.shared,'clean',lambda:None)
    monkeypatch.setattr(cli,'load_request',lambda p:(tmp_path,{},None,None,None))
    monkeypatch.setattr(cli,'load_calibration_split',lambda *a:pytest.fail('Must not read data'))
    with pytest.raises(ValueError,match='Confirm'):cli.run(Namespace(request='none',confirm_request_hash='wrong'))


def test_prepare_only_hashes_data_without_predicting(tmp_path,monkeypatch):
    monkeypatch.setattr(cli,'ROOT',tmp_path);monkeypatch.setattr(cli.shared,'ROOT',tmp_path)
    monkeypatch.setattr(cli.shared,'clean',lambda:None);monkeypatch.setattr(cli.shared,'metadata',lambda:{'git_commit':'a'*40})
    rp=tmp_path/'review.json';rp.write_text('{}')
    p={'training':{'methods':['ordinary','conditional']}}; d={'manifest':{'calibration':['a','b','c']}}
    monkeypatch.setattr(cli,'prerequisites',lambda path:(rp,{},p,d))
    monkeypatch.setattr(cli,'load_calibration_split',lambda *a:synthetic())
    monkeypatch.setattr(cli,'episode_scores',lambda *a,**kw:pytest.fail('Prepare must not fit'))
    write_once(tmp_path/'audit.json',{'status':'complete','calibration_interface_gate':True,'purpose':'development_smoke',
        'source_hashes':cli.shared.source_hashes(),'environment':cli.shared.env(),'training_review_sha256':file_hash(rp)})
    request=cli.prepare(Namespace(training_review='review.json',calibration_audit='audit.json',run_id='test'))
    assert sorted(x.name for x in request.parent.iterdir())==['preparation.json','request.json']
    assert json.loads(request.read_text())['checkpoint_selection_allowed']is False


def test_calibration_run_seals_both_methods_without_claiming_effect(tmp_path,monkeypatch):
    b=synthetic();q={'methods':['ordinary','conditional'],'data_hash':b.fingerprint(),'batch_roots':16,'settings':SETTINGS}
    r={'request_hash':'frozen_training','ensembles':{m:{'path':'unused.pt','sha256':'a'*64,'selected':[]} for m in q['methods']}}
    d={'manifest':{'calibration':['a','b','c']}}
    monkeypatch.setattr(cli,'ROOT',tmp_path)
    monkeypatch.setattr(cli.shared,'clean',lambda:None)
    monkeypatch.setattr(cli.shared,'metadata',lambda:{'git_commit':'a'*40})
    monkeypatch.setattr(cli,'load_request',lambda path:(tmp_path,q,r,{},d))
    monkeypatch.setattr(cli,'load_calibration_split',lambda *a:b)
    monkeypatch.setattr(cli,'load_trained_ensemble',lambda *a:ZeroEnsemble().eval())
    args=Namespace(request='unused',confirm_request_hash=digest(q))
    cli.run(args)
    result=json.loads((tmp_path/'calibration.json').read_text())
    assert result['ddpg_ready']is False and result['test_locked']is True
    assert set(result['methods'])=={'ordinary','conditional'}
    # An unbounded, uninformative band is still a valid reported result.
    assert all(v['fit']['unbounded'] for v in result['methods'].values())
    report=json.loads((tmp_path/'report.json').read_text());assert report['status']=='complete'
    before=file_hash(tmp_path/'calibration.json')
    with pytest.raises(ValueError,match='already started'):cli.run(args)
    assert file_hash(tmp_path/'calibration.json')==before


def test_calibration_failure_is_preserved_without_partial_freeze(tmp_path,monkeypatch):
    q={'data_hash':'changed'}
    monkeypatch.setattr(cli.shared,'clean',lambda:None)
    monkeypatch.setattr(cli.shared,'metadata',lambda:{'git_commit':'a'*40})
    monkeypatch.setattr(cli,'load_request',lambda path:(tmp_path,q,None,None,{}))
    monkeypatch.setattr(cli,'load_calibration_split',lambda *a:synthetic())
    with pytest.raises(ValueError,match='tensors changed'):
        cli.run(Namespace(request='unused',confirm_request_hash=digest(q)))
    assert not (tmp_path/'calibration.json').exists()
    assert json.loads((tmp_path/'report.json').read_text())['status']=='failed'
    assert not (tmp_path/'writer.lock').exists()
