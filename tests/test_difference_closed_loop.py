from copy import deepcopy
from pathlib import Path
import sys

import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools'),str(ROOT/'tests')]
from prediction_rl.data.collection_store import read_json as read
from prediction_rl.evaluation.difference_closed_loop import ARM, PROTOCOL, SMOKE, REFERENCE, jobs, aggregate, validate_protocol
from prediction_rl.evaluation.matched_learning_curve import METRICS
from test_fixed_budget_extension import rows as old_rows
import train_difference_ddpg as cli


def new_rows():
    return [{**j,**{k:0. for k in METRICS},'steps':3,'natural_arrival':1,'ego_collision':0,'time_limit':0,
             'native_event_agrees':True,'no_policy_updates':True,'actual_training_steps':j['checkpoint_frames'],
             'initial_traffic_sha256':str(j['simulator_seed']).zfill(64)} for j in jobs(PROTOCOL,'evaluate')]


def test_only_new_arm_and_matched_budget():
    assert read(ROOT/cli.CONFIG)==PROTOCOL
    p=deepcopy(PROTOCOL); p['version']=REFERENCE['version']; p['arms']=REFERENCE['arms']
    p['comparisons']=REFERENCE['comparisons']; del p['reference_arms']
    assert p==REFERENCE
    assert len(jobs(PROTOCOL,'train'))==3 and len(jobs(PROTOCOL,'evaluate'))==240
    assert 3*PROTOCOL['training']['requested_frames']==360000
    assert len({j['simulator_seed'] for j in jobs(PROTOCOL,'evaluate')})==20
    assert len(jobs(SMOKE,'train'))==1 and len(jobs(SMOKE,'evaluate'))==3


@pytest.mark.parametrize('fault',['seed','budget','lr','warm_start','reward','nodes','workers'])
def test_design_rejects_unapproved_changes(fault):
    p=deepcopy(PROTOCOL)
    if fault=='seed': p['training']['run_seeds']=[0,1,3]
    elif fault=='budget': p['training']['requested_frames']=200000
    elif fault=='lr': p['training']['learning_rate']=.001
    elif fault=='warm_start': p['training']['warm_start']=True
    elif fault=='nodes': p['training']['checkpoint_frames'].append(100000)
    elif fault=='workers': p['training']['workers']=2
    else: p['reward']='new'
    with pytest.raises(ValueError): validate_protocol(p)


def test_negative_new_result_completes_and_old_rows_unchanged():
    old=old_rows(REFERENCE); before=deepcopy(old); new=new_rows()
    for r in new:
        if r['checkpoint_frames']==120000: r.update(natural_arrival=0,ego_collision=1,**{'return':-10.})
    a=aggregate(PROTOCOL,new,old)
    assert old==before and a['engineering_complete'] and a['method_effect_gate'] is None
    assert a['primary_node']==120000 and a['new_evaluation_episodes']==240 and a['reused_evaluation_episodes']==960
    assert a['unique_traffic_scenarios']==20 and a['training_replicates']==3
    assert a['nodes'][-1]['comparisons']['difference_objective']['equal_run_seed_mean_delta']['return']==-10.
    assert set(a['nodes'][-1]['arms'])==set(REFERENCE['arms'])|{ARM}


@pytest.mark.parametrize('fault',['missing','duplicate','scene','nan','event','updates','steps'])
def test_no_silent_exclusion_or_invalid_statistics(fault):
    old=old_rows(REFERENCE); new=new_rows()
    if fault=='missing': new.pop()
    elif fault=='duplicate': new[-1]=deepcopy(new[0])
    elif fault=='scene': new[-1]['initial_traffic_sha256']='f'*64
    elif fault=='nan': new[-1]['return']=float('nan')
    elif fault=='event': new[-1]['ego_collision']=1
    elif fault=='updates': new[-1]['no_policy_updates']=False
    else: new[-1]['actual_training_steps']=120500
    with pytest.raises(ValueError): aggregate(PROTOCOL,new,old)


def test_smoke_not_method_report():
    with pytest.raises(ValueError,match='Smoke'): aggregate(SMOKE,[],[])


def test_context_maps_only_method_label_no_global_mutation(monkeypatch):
    q={'protocol':PROTOCOL,'difference_request':{'frozen':True},'reference_root':'E:/reference',
       'predictor':{'sha256':'a'*64}}
    marker=object(); calls=[]
    monkeypatch.setattr(cli.connection,'provider',lambda p,r,e:marker)
    def env(request,job,provider,recorder): calls.append((request,job,provider,recorder)); return marker
    monkeypatch.setattr(cli.original,'environment',env)
    original=cli.original.provider_for
    context=cli.Execution(q); job={'arm':ARM,'id':'conditional_difference_s0','run_seed':0}
    assert context.provider_for(q,ARM)==(marker,'a'*64)
    assert context.environment(q,job,marker,None) is marker
    assert calls[0][1]['arm']=='conditional' and job['arm']==ARM
    assert cli.original.provider_for is original and context.smoke is False
    with pytest.raises(ValueError): context.provider_for(q,'conditional')


def test_full_prepare_requires_bounded_audit(monkeypatch):
    monkeypatch.setattr(cli.shared,'clean',lambda:None)
    monkeypatch.setattr(cli.connection,'accepted_calibration',lambda *a:({},Path('E:/reference'),{},{}))
    with pytest.raises(ValueError,match='workflow audit'): cli.prepare('unused','new')


def test_kernel_extension_is_explicit_optional():
    import inspect
    assert inspect.signature(cli.kernel.train_job).parameters['execution'].default is None
    assert inspect.signature(cli.kernel.evaluation_job).parameters['execution'].default is None
