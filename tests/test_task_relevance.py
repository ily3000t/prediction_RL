from argparse import Namespace
from copy import deepcopy
from pathlib import Path
import sys

import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
from test_candidate_ranking import example
from prediction_rl.prediction.kinematic_reference import LaneConstantVelocity
from prediction_rl.prediction.task_relevance import branch_task,coverage,pair_task_agreement,diagnose_root,aggregate
import diagnose_task_relevance as cli


@pytest.fixture(autouse=True,scope='module')
def threads():
    old=torch.get_num_threads();torch.set_num_threads(1);yield;torch.set_num_threads(old)


def branch(reason=None,jerk=2.,invalid=-.1):
    reward=(-10 if reason=='upstream_collision' else 10 if reason=='upstream_arrival' else -.02-.02*jerk**2)+invalid
    return {'trace':[{'done':reason is not None,'reward':reward,'info':{'execution_audit':{
        'termination_reason':reason,'measured_jerk':jerk,'invalid_action_reward':invalid,'simulation_elapsed_s':.2}}}],
        'future_metadata':{'frames':[{}]}}


def label(reason=None):
    return {'observed_steps':1,'termination_reason':reason,'full_time_horizon_observed':False,
        'events':{k:{'value':int(reason==k),'mask':reason==k} for k in ('upstream_collision','upstream_arrival')}}


@pytest.mark.parametrize('reason',[None,'upstream_collision','upstream_arrival','registered_time_limit'])
def test_original_reward_decomposition(reason):
    b=branch(reason);r=branch_task(b,label(reason))
    assert r['observed_return']==b['trace'][0]['reward'] and r['reward_max_abs_error']<1e-12
    assert r['reward_components']['invalid_action']==-.1
    assert r['reward_components']['jerk']==pytest.approx(0 if reason in ('upstream_collision','upstream_arrival') else -.08)


def test_reward_mismatch_is_engineering_error_not_negative_method_result():
    b=branch();b['trace'][0]['reward']+=1
    with pytest.raises(ValueError,match='reward differs'):branch_task(b,label())


def test_timeout_missing_measured_jerk_remains_unverified():
    b=branch('upstream_time_limit');b['trace'][0]['info']['execution_audit']['measured_jerk']=None
    r=branch_task(b,label('upstream_time_limit'))
    assert r['reward_unverified_steps']==1 and r['reward_components']['jerk']is None
    assert r['reward_max_abs_error']is None


def test_terminal_event_survives_zero_trajectory_support():
    r=branch_task(branch('upstream_collision'),label('upstream_collision'))
    assert r['events']['upstream_collision']=={'value':1,'mask':True}
    assert r['events']['upstream_arrival']=={'value':0,'mask':False}


def test_pair_censoring_and_no_geometry_never_become_negative_labels():
    tasks=[branch_task(branch(),label()) for _ in range(5)]
    result=pair_task_agreement([0,1,2,3,4],tasks)
    assert result['upstream_collision']['agreement']is None
    assert result['upstream_collision']['counts']['censored_or_incomparable']==10
    assert result['observed_return']['counts']['censored_or_incomparable']==10
    for i,t in enumerate(tasks):
        t['full_horizon']=True;t['observed_return']=i
        t['events']['upstream_collision']={'value':int(i<2),'mask':True}
    result=pair_task_agreement([0,1,2,3,4],tasks)
    assert result['observed_return']['agreement']==1 and result['upstream_collision']['agreement']==1
    assert pair_task_agreement(None,tasks)['observed_return']['counts']['no_geometry']==10
    assert pair_task_agreement([1]*5,tasks)['observed_return']['agreement']==0


def test_coverage_distinguishes_selected_omitted_new_and_missing():
    g=LaneConstantVelocity(ROOT/'RL-MPC-LaneMerging-master/merge.net.xml');h,_,_,_=example(g)
    frame=deepcopy(h[0]['history'][-1]);ego=frame['vehicles']['ego']
    frame['vehicles']['a']['position']=[ego['position'][0]+1,ego['position'][1]]
    b={'trace':[{'done':False}],'future_metadata':{'frames':[frame]}}
    assert coverage(b,set(),{'ego','a','b'},g.geometry)['nearest_front_bumper']['role']=='omitted_root'
    assert coverage(b,set(),{'ego','b'},g.geometry)['nearest_front_bumper']['role']=='new_actor'
    assert coverage(b,{'a'},{'ego','a','b'},g.geometry)['nearest_front_bumper']['role']=='selected'
    b['trace'][0]['done']=True
    assert coverage(b,{'a'},{'ego','a','b'},g.geometry)['nearest_front_bumper']is None


def test_whole_root_localization_does_not_change_predictor_inputs():
    g=LaneConstantVelocity(ROOT/'RL-MPC-LaneMerging-master/merge.net.xml');h,p,manifest,b=example(g);branches=[]
    for c in p[0]['candidates']:
        labels=c['labels'];labels.update(events=label()['events'],full_time_horizon_observed=True)
        item=branch(None,0.,0.);item['trace']*=25
        item['future_metadata']['frames']=[deepcopy(h[0]['history'][-1]) for _ in range(25)];branches.append(item)
    before=b.fingerprint()
    row=diagnose_root({'ordinary':b.targets.clone(),'conditional':b.targets.clone()},b,h[0],p[0],branches,g)
    assert b.fingerprint()==before and len(row['critical_cells'])==5
    assert all(x['neighbor_position_error_m']['conditional']==0 for x in row['critical_cells'])
    result=aggregate([row],manifest['development'],['ordinary','conditional'])
    assert result['branches']==5 and result['empty_geometry_roots']==0
    assert result['methods']['observed_proxy']['upstream_collision']['eligible_roots']==0


@pytest.mark.parametrize('split',['test','validation','calibration','unknown'])
def test_only_development_and_train_allowed(split):
    with pytest.raises(ValueError,match='cannot open'):cli.data({}, {}, {},split)


def test_confirmation_and_overwrite_protection(tmp_path,monkeypatch):
    monkeypatch.setattr(cli.shared,'clean',lambda:None)
    monkeypatch.setattr(cli,'load_request',lambda _:(tmp_path,{},None,None,None))
    monkeypatch.setattr(cli,'evaluate',lambda *a:pytest.fail('Do not infer'))
    with pytest.raises(ValueError,match='Confirm'):cli.run(Namespace(request='none',confirm_request_hash='wrong'))
    (tmp_path/'started.json').write_text('{}')
    with pytest.raises(ValueError,match='overwrite'):cli.run(Namespace(request='none',confirm_request_hash=cli.digest({})))


def test_config_is_complete_and_frozen():
    assert cli.read(ROOT/'configs/development/p05_task_relevance_v1.json')==cli.CONFIG
    for key in ('model_update','feature_release','ddpg_release','test_release'):assert cli.CONFIG[key]is False


def test_prepare_is_verification_only(tmp_path,monkeypatch):
    monkeypatch.setattr(cli,'ROOT',tmp_path);monkeypatch.setattr(cli.shared,'ROOT',tmp_path)
    monkeypatch.setattr(cli.shared,'clean',lambda:None);monkeypatch.setattr(cli.shared,'metadata',lambda:{'git_commit':'a'*40})
    monkeypatch.setattr(cli,'prerequisites',lambda _:(cli.CONFIG,{}, {},{}, {}))
    monkeypatch.setattr(cli,'data',lambda *a:([{'root_id':'r'}],{'train':['e']},{'receipt':'h'}))
    monkeypatch.setattr(cli,'evaluate',lambda *a:pytest.fail('Prepare cannot infer'))
    cli.write_once(tmp_path/'roots.json',[])
    cli.write_once(tmp_path/'audit.json',{'status':'complete','interface_gate':True,'purpose':'development_smoke',
        'inputs':{},'source_hashes':cli.shared.source_hashes(),'environment':cli.shared.env(),
        'config_hash':cli.digest(cli.CONFIG),'roots_sha256':cli.sha(tmp_path/'roots.json')})
    cli.prepare(Namespace(config='config.json',audit='audit.json',run_id='test'))
    out=tmp_path/'artifacts/task_diagnostics/test'
    assert sorted(p.name for p in out.iterdir())==['preparation.json','request.json']
    assert cli.read(out/'request.json')['roots']==['r']


def test_negative_task_association_completes_without_releasing_features(tmp_path,monkeypatch):
    manifest={'train':['e']};receipts={'receipt':'h'}
    q={'config':cli.CONFIG,'receipts':receipts,'manifest_hash':cli.digest(manifest),'roots':['r']}
    monkeypatch.setattr(cli.shared,'clean',lambda:None);monkeypatch.setattr(cli.shared,'metadata',lambda:{'git_commit':'a'*40})
    monkeypatch.setattr(cli,'load_request',lambda _:(tmp_path,q,{}, {},{}))
    monkeypatch.setattr(cli,'data',lambda *a:([{'root_id':'r'}],manifest,receipts))
    monkeypatch.setattr(cli,'evaluate',lambda *a:([],{'task_association':0}))
    cli.run(Namespace(request='none',confirm_request_hash=cli.digest(q)))
    report=cli.read(tmp_path/'report.json')
    assert report['status']=='complete' and report['ddpg_ready']is False
    assert report['feature_contract_frozen']is False and report['model_updated']is False
