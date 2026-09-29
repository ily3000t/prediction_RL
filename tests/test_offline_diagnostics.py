from copy import deepcopy
from dataclasses import replace
from argparse import Namespace
from pathlib import Path
import sys

import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
from test_masked_training import fixture_data,batch
from prediction_rl.prediction.interface import PredictorConfig
from prediction_rl.prediction.epoch_training import subset
from prediction_rl.prediction.kinematic_reference import LaneConstantVelocity
from prediction_rl.prediction.offline_diagnostics import (root_metrics,summarize,paired_difference,
    residual_attribution,summarize_attribution,FIELDS)
from prediction_rl.prediction.calibration import SETTINGS
import diagnose_response_predictors as cli


@pytest.fixture(autouse=True,scope='module')
def threads():
    n=torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(n)


@pytest.fixture
def reference():return LaneConstantVelocity(ROOT/'RL-MPC-LaneMerging-master/merge.net.xml')


def test_reference_fixed_speed_first_tick_and_same_candidates(reference):
    cfg=PredictorConfig(hidden_dim=16);h,_,_=fixture_data(cfg)
    pred=reference.predict(h,cfg)
    slot=h[0]['inputs']['actor_ids'].index('a')
    assert pred[0,0,slot,0,0]==pytest.approx(21.4,abs=1e-4)
    assert pred[0,0,slot,-1,0]==pytest.approx(55.,abs=1e-3)
    assert torch.equal(pred,pred[:,:1].expand_as(pred))
    assert pred[0,0,slot,0,2]==7 and pred[0,0,slot,0,3]==0


def test_reference_zero_speed_and_missing_slots(reference):
    cfg=PredictorConfig(hidden_dim=16);h,_,_=fixture_data(cfg)
    for a in ('a','b'):h[0]['history'][-1]['vehicles'][a]['speed']=0.
    pred=reference.predict(h,cfg)
    assert torch.equal(pred[:,:,:,0,:],pred[:,:,:,-1,:])
    for i,a in enumerate(h[0]['inputs']['actor_ids']):
        if a is None:assert not pred[:,:,i].any()


def test_reference_lane_transition_and_extrapolation(reference):
    length=reference.geometry.lengths[':mergenode_1_0']
    assert reference.xy(':mergenode_1_0',length)==pytest.approx(reference.xy('highwayahead_0',0),abs=1e-10)
    assert reference.xy(':mergenode_1_0',length+10)==pytest.approx(reference.xy('highwayahead_0',10))
    assert reference.xy('highwayahead_0',108.5)[0]==pytest.approx(110.)
    for lane,pos in [('unknown',0),('ramp_0',-1),('ramp_0',float('nan'))]:
        with pytest.raises(ValueError):reference.xy(lane,pos)


def test_reference_never_reads_label_packs(reference):
    cfg=PredictorConfig(hidden_dim=16);h,p,_=fixture_data(cfg)
    before=reference.predict(h,cfg);p[0]['candidates']=None
    assert torch.equal(before,reference.predict(h,cfg))


def test_exact_predictions_have_zero_error_and_response_delta():
    b=batch(PredictorConfig(hidden_dim=16));rows=root_metrics(b.targets.clone(),b)
    assert all(rows[0][k]==0 for k in FIELDS)
    assert rows[0]['valid_candidate_pairs']==10


def test_last_observed_is_not_five_second_fde():
    b=batch(PredictorConfig(hidden_dim=16));b.target_mask[:,:,:,2:]=False
    pred=b.targets.clone();pred[:,:,:,1,0]+=3
    row=root_metrics(pred,b)[0]
    assert row['last_observed_displacement_m']==3
    assert row['fde_5s_m']is None and row['full_horizon_actor_branches']==0
    assert row['ade_m']==1.5


def test_masked_error_does_not_change_point_metrics_but_nan_rejected():
    b=batch(PredictorConfig(hidden_dim=16));pred=b.targets.clone()
    pred[~b.target_mask]=999
    assert root_metrics(pred,b)[0]['scaled_mse']==0
    pred[~b.target_mask]=float('nan')
    with pytest.raises(ValueError):root_metrics(pred,b)


def test_response_contrast_uses_only_pair_intersection():
    b=batch(PredictorConfig(hidden_dim=16));b.target_mask[:]=False
    b.target_mask[0,0,0,0]=True;b.target_mask[0,1,0,0]=True
    b.targets[:]=0;b.targets[0,1,0,0,0]=10.
    pred=torch.zeros_like(b.targets); row=root_metrics(pred,b)[0]
    assert row['pairwise_response_delta_scaled_mse']==.25 and row['valid_candidate_pairs']==1
    b.target_mask[0,1,0,0]=False;b.target_mask[0,1,0,1]=True
    assert root_metrics(pred,b)[0]['pairwise_response_delta_scaled_mse']is None


def test_root_and_episode_equal_aggregation_differ_without_pseudoreplication():
    b=subset(batch(PredictorConfig(hidden_dim=16)),[0,0,0]);b=replace(b,root_ids=('a','b','c'),episode_ids=('e1','e1','e2'))
    rows=root_metrics(b.targets.clone(),b)
    for r,v in zip(rows,[0,0,3]):r['ade_m']=v
    result=summarize(rows,['e1','e2','empty'])
    assert result['root_equal']['ade_m']==1 and result['episode_equal']['ade_m']==1.5
    assert result['episode_metrics'][-1]['ade_m']is None
    with pytest.raises(ValueError):summarize(rows+rows[:1],['e1','e2'])


def test_paired_rows_are_identity_checked_and_not_independent_candidates():
    b=batch(PredictorConfig(hidden_dim=16));a=root_metrics(b.targets.clone(),b);other=deepcopy(a);other[0]['ade_m']=2.
    r=paired_difference(a,other,list(b.episode_ids))['ade_m']
    assert r['paired_roots']==1 and r['paired_episodes']==1 and r['episode_mean_right_minus_left']==2.
    other[0]['root_id']='changed'
    with pytest.raises(ValueError):paired_difference(a,other,list(b.episode_ids))


def test_attribution_identifies_channel_candidate_actor_and_time():
    b=batch(PredictorConfig(hidden_dim=16));pred=b.targets.repeat(3,1,1,1,1,1)
    b.targets[0,2,0,4,3]+=10
    fit={'settings':SETTINGS,'q':5.,'unbounded':False}
    rows=residual_attribution(pred,b,fit,[['actor']*12]);m=rows[0]['maximum']
    assert (m['channel'],m['candidate_id'],m['actor_slot'],m['horizon_s'])==('acceleration',2,0,1.)
    assert m['standardized_residual']==pytest.approx(50.)
    s=summarize_attribution(rows,list(b.episode_ids),fit,'validation')
    assert s['worst_channel_episode_counts']=={'acceleration':1} and s['episode_coverage']==0
    assert s['floor_active_fraction_by_channel']==[1.,1.,1.,1.]


def test_attribution_zero_error_argmax_avoids_invalid_slots():
    b=batch(PredictorConfig(hidden_dim=16));b.target_mask[:]=False;b.target_mask[0,4,7,3]=True
    fit={'q':0.,'unbounded':False};r=residual_attribution(b.targets.repeat(3,1,1,1,1,1),b,fit,[['actor']*12])[0]
    assert r['maximum']['actor_slot']==7 and r['maximum']['candidate_id']==4
    assert r['maximum']['standardized_residual']==0.


def test_empty_roots_and_episodes_are_not_covered_successes():
    b=batch(PredictorConfig(hidden_dim=16));b.target_mask[:]=False;fit={'q':None,'unbounded':True}
    r=residual_attribution(b.targets.repeat(3,1,1,1,1,1),b,fit,[['actor']*12])
    assert r[0]['maximum']is None
    s=summarize_attribution(r,[*b.episode_ids,'missing'],fit,'calibration')
    assert s['eligible_episodes']==0 and s['empty_episodes']==2 and s['episode_coverage']is None


@pytest.mark.parametrize('split',['train','test','development'])
def test_p5a_data_loader_refuses_unauthorized_splits(split):
    with pytest.raises(ValueError):cli.data({},split)


def test_manual_confirmation_precedes_evaluation(tmp_path,monkeypatch):
    monkeypatch.setattr(cli.shared,'clean',lambda:None)
    monkeypatch.setattr(cli,'load_request',lambda p:(tmp_path,{},None,None,None,None))
    monkeypatch.setattr(cli,'data',lambda *a:pytest.fail('Should not evaluate'))
    with pytest.raises(ValueError,match='Confirm'):cli.run(Namespace(request='none',confirm_request_hash='wrong'))


def test_prepare_hashes_inputs_without_model_evaluation(tmp_path,monkeypatch):
    from prediction_rl.data.collection_store import write_once,read_json
    from prediction_rl.data.dataset_contract import digest
    monkeypatch.setattr(cli,'ROOT',tmp_path);monkeypatch.setattr(cli.shared,'ROOT',tmp_path)
    monkeypatch.setattr(cli.shared,'clean',lambda:None);monkeypatch.setattr(cli.shared,'metadata',lambda:{'git_commit':'a'*40})
    monkeypatch.setattr(cli,'prerequisites',lambda path:(cli.CONFIG,{}, {},{}, {},{}))
    monkeypatch.setattr(cli,'data',lambda d,split:(batch(PredictorConfig(hidden_dim=16)),[]))
    monkeypatch.setattr(cli,'root_metrics',lambda *a:pytest.fail('Prepare cannot evaluate'))
    ap=tmp_path/'audit.json';write_once(ap,{'status':'complete','interface_gate':True,'purpose':'development_smoke',
        'source_hashes':cli.shared.source_hashes(),'environment':cli.shared.env(),'config_hash':digest(cli.CONFIG),'inputs':{}})
    cli.prepare(Namespace(config='config.json',audit=str(ap),run_id='test'))
    out=tmp_path/'artifacts/offline/test'
    assert sorted(x.name for x in out.iterdir())==['preparation.json','request.json']
    q=read_json(out/'request.json');assert q['test_locked']is True and q['ddpg_ready']is False


def test_checked_protocol_file_has_no_hidden_tuning_fields():
    from prediction_rl.data.collection_store import read_json
    assert read_json(ROOT/'configs/development/p05_offline_diagnostics_v1.json')==cli.CONFIG


def test_tracked_json_newlines_do_not_change_parameter_provenance(tmp_path,monkeypatch):
    monkeypatch.setattr(cli,'ROOT',tmp_path)
    cp=tmp_path/'config.json';artifact=tmp_path/'report.json'
    cp.write_bytes(b'{\n  "value": 1\n}\n');artifact.write_bytes(b'{"status":"complete"}')
    before=cli.artifact_hashes(cp,[artifact])
    cp.write_bytes(b'{\r\n  "value": 1\r\n}\r\n')
    assert cli.artifact_hashes(cp,[artifact])==before
    cp.write_bytes(b'{"value":2}')
    assert cli.artifact_hashes(cp,[artifact])!=before
    cp.write_bytes(b'{"value":1}');artifact.write_bytes(b'{"status":"failed"}')
    assert cli.artifact_hashes(cp,[artifact])!=before
