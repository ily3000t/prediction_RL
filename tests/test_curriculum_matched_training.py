"""IC04 synthetic/receipt tests; no real SUMO, predictor inference or 120k run."""
from copy import deepcopy
from pathlib import Path
import random
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'src')]
import curriculum_matched_contract as contract
import curriculum_matched_runtime as runtime
import train_curriculum_ddpg as cli
from prediction_rl.curriculum.entry_smoke import rng_fingerprint
from prediction_rl.data.dataset_contract import digest
from prediction_rl.data.collection_store import write_once,read_json,seal_episode,file_hash,verify_episode
from prediction_rl.envs.curriculum_stream import CurriculumTaskStream
from prediction_rl.envs.prediction_features import PredictionFeatureEnv
from prediction_rl.training.all_ddpg import make_all_environment,preset_factory,experiment_type
from test_curriculum_persistent_learner import SyntheticWorker,Provider


def manifest():
    return contract.freeze_manifest(cli.entry.read(ROOT/'artifacts/ic/ic02h_intensity_v1/task_proposals.json'),
        cli.entry.read(ROOT/'artifacts/ic/ic02h_intensity_v1/report.json'),
        cli.accepted.original.SEALED['task_proposals.json'])


@pytest.mark.parametrize('key,value',[('run_id','retry'),('thresholds',{}),('layer_counts',{'L':10,'M':11,'H':11}),
    ('arms',['b3_curriculum']),('main_endpoint',60000),('unknown',True),('test_opened',True)])
def test_unknown_fields_changed_budget_selection_or_strata_rejected(key,value):
    p=deepcopy(contract.PROTOCOL); p[key]=value
    with pytest.raises(ValueError): contract.validate_protocol(p)


def test_independent_resolved_config_budget_and_nominal_quotas():
    p=cli.entry.read(ROOT/cli.CONFIG)
    assert contract.validate_protocol(p)==contract.PROTOCOL
    assert len(contract.jobs(p,'train'))==12 and len(contract.jobs(p,'evaluate'))==960
    assert 12*p['training']['requested_frames']==1440000
    target={'L':11/32,'M':10/32,'H':11/32}
    assert p['schedule']['stages'][-1]['mass']==target
    assert p['schedule']['nominal_Mix_reset_mass']==pytest.approx({'L':.3385416666666667,'M':.27291666666666664,'H':.3885416666666667})
    assert p['schedule']['actual_exposure_equality_claim'] is False


def test_new_manifest_preserves_all_old_tasks_and_the_old_null_labels():
    path=ROOT/'artifacts/ic/ic02h_intensity_v1/task_proposals.json'; before=file_hash(path)
    old=cli.entry.read(path); m=manifest(); tasks=contract.manifest_roster(m)
    assert len(tasks)==32 and [x['simulator_seed'] for x in tasks]==list(range(11000,11249,8))
    assert all(t['task_level'] is None and t['classification_applied'] is False for t in old['tasks'])
    assert all(t['task_level']==t['proposed_level'] and t['classification_applied'] is True for t in m['tasks'])
    assert all(t['difficulty_components']==o['difficulty_components'] for t,o in zip(m['tasks'],old['tasks']))
    assert file_hash(path)==before
    with pytest.raises(ValueError): contract.manifest_roster({**m,'tasks':m['tasks'][:-1]})
    bad=deepcopy(m); bad['tasks'][0]['level']='H'
    with pytest.raises(ValueError): contract.manifest_roster(bad)


def test_sampler_private_rng_full_boundary_no_extension_and_no_mid_episode_change():
    before=rng_fingerprint(); m=manifest(); job=contract.jobs(contract.PROTOCOL,'train')[3]
    a=contract.TrainingEpisodeSampler(m,job,'a'*64,contract.PROTOCOL)
    b=contract.TrainingEpisodeSampler(m,job,'a'*64,contract.PROTOCOL)
    for _ in range(41):
        assert a.begin_episode()==b.begin_episode()
        with pytest.raises(ValueError): a.begin_episode()
        assert a.end_episode(499)==b.end_episode(499)
    assert a.algorithm._step==20459
    choice=a.begin_episode(); assert choice['collection_stage']==1 and choice['stage_boundary_overrun']==459
    assert choice['execution_authorized'] is False and choice['matched_development_training_authorized'] is True
    a.end_episode(1)
    while a.algorithm._step<120000:
        a.begin_episode(); a.end_episode(500)
    assert a.algorithm._step==120460
    with pytest.raises(ValueError): a.begin_episode()
    assert before==rng_fingerprint()


@pytest.mark.parametrize('bad',[0,501,True,1.5])
def test_sampler_control_limit_is_not_synthetic_terminal(bad):
    sampler=contract.TrainingEpisodeSampler(manifest(),contract.jobs(contract.PROTOCOL,'train')[0],'a'*64,contract.PROTOCOL)
    sampler.begin_episode()
    with pytest.raises(ValueError): sampler.end_episode(bad)
    assert sampler.completed==0


def test_old_smoke_and_offline_guards_not_reauthorized():
    sampler=contract.TrainingEpisodeSampler(manifest(),contract.jobs(contract.PROTOCOL,'train')[0],'a'*64,contract.PROTOCOL)
    spaces=SyntheticWorker({'simulator_seed':0},[])
    with pytest.raises(ValueError): CurriculumTaskStream(sampler,lambda _:None,spaces.observation_space,
        spaces.action_space,purpose=contract.SCOPE)
    with pytest.raises(ValueError): runtime.MatchedTaskStream(SimpleNamespace(),lambda _:None,
        spaces.observation_space,spaces.action_space)


class FiniteFixture(SyntheticWorker):
    def __init__(self,selection):
        super().__init__(selection,[],length=250)
        self.seed%=3  # Synthetic normalized input, not a different real task seed.


class PlainTasks:
    """Independent original loop reference; no replay instrumentation."""
    execution_contract='simulation_blocking_exact_v1'
    def __init__(self,sampler):
        self.sampler=sampler; self.worker=None
        spaces=FiniteFixture({'simulator_seed':0})
        self.observation_space,self.action_space=spaces.observation_space,spaces.action_space
    def reset(self):
        if self.worker is not None: self.worker.close()
        self.worker=FiniteFixture(self.sampler.begin_episode()); return self.worker.reset()
    def step(self,action):
        r=self.worker.step(action)
        if r[2]: self.sampler.end_episode(self.worker._steps)
        return r
    def current_frame(self): return self.worker.current_frame()
    def close(self):
        if self.worker is not None: self.worker.close()


def synthetic(path,watched):
    torch.set_num_threads(1); torch.manual_seed(557); np.random.seed(557); random.seed(557)
    job=contract.jobs(contract.PROTOCOL,'train')[9]
    sampler=contract.TrainingEpisodeSampler(manifest(),job,'a'*64,contract.PROTOCOL)
    plain=PlainTasks(sampler); audit=None
    if watched:
        stream=runtime.MatchedTaskStream(sampler,FiniteFixture,plain.observation_space,plain.action_space)
        audit=runtime.MatchedReplayAudit(stream)
    else: stream=plain
    feature=PredictionFeatureEnv(stream,'conditional',Provider('conditional'),stream.current_frame)
    env=make_all_environment(feature,'cpu',None if audit is None else audit.record)
    experiment=experiment_type(path)(preset_factory('cpu',.0002,smoke=False),env,quiet=True)
    if audit is not None: audit.install(experiment._agent)
    experiment.train(frames=5000)  # Test fixture only. Original warmup reached; NO SUMO.
    env.close(); experiment._writer.close()
    return experiment,audit,rng_fingerprint()


def test_new_scope_retains_original_updates_replay_targets_and_rng_exact(tmp_path):
    plain,_,rng0=synthetic(tmp_path/'plain',False)
    watched,audit,rng1=synthetic(tmp_path/'watched',True)
    assert rng0==rng1 and plain.frame==watched.frame==5001
    assert audit.sample_calls>0 and audit.insertions==5000
    a,b=plain._agent.agent,watched._agent.agent
    assert a._frames_seen==b._frames_seen==5020
    for name in ('q','policy'):
        left,right=getattr(a,name),getattr(b,name)
        for model,other in ((left.model,right.model),(left._target._target,right._target._target)):
            assert all(torch.equal(v,other.state_dict()[k]) for k,v in model.state_dict().items())
        assert left._scheduler.state_dict()==right._scheduler.state_dict()
        assert left._optimizer.state and right._optimizer.state
        for l,r in zip(left._optimizer.state.values(),right._optimizer.state.values()):
            assert all(torch.equal(v,r[k]) for k,v in l.items())
    for l,r in zip(a.replay_buffer,b.replay_buffer):
        assert torch.equal(l[0].features,r[0].features) and torch.equal(l[1],r[1]) and l[2]==r[2]
        assert torch.equal(l[3].features,r[3].features) and torch.equal(l[3].mask,r[3].mask)
    s=audit.snapshot()
    assert s['reset_bridge_transitions']==0 and len(s['completed_episodes'])==20
    assert s['original_replay_size']==5000 and s['metadata_in_network_inputs'] is False
    assert s['matched_development_training_authorized'] is True and s['formal_claim'] is False


def rows():
    result=[]
    for j in contract.jobs(contract.PROTOCOL,'evaluate'):
        r={**j,**{m:1. for m in contract.METRICS},'actual_training_steps':j['checkpoint_frames']+1,
            'no_policy_updates':True,'native_event_agrees':True,'steps':1,'initial_traffic_sha256':digest(j['simulator_seed']),
            'natural_arrival':1,'ego_collision':0,'time_limit':0,'low_speed_fraction':.1,'action_saturation_fraction':.2}
        r['return']={'b2_mix':0.,'b2_curriculum':2.,'b3_mix':3.,'b3_curriculum':7.}[j['arm']]
        result.append(r)
    return result


def test_full_roster_negative_results_g_and_nonuniform_node_auc():
    result=contract.aggregate(contract.PROTOCOL,rows())
    assert result['evaluation_episodes']==960 and result['unique_traffic_scenarios']==20
    assert result['primary_node']==120000 and result['method_effect_gate'] is None
    assert all(n['specificity_G']['equal_run_seed_mean_delta']['return']==2 for n in result['nodes'])
    assert result['normalized_node_trapezoid_auc']['b3_curriculum']['0']['return']==7
    negative=rows()
    for r in negative:
        if r['arm']=='b3_curriculum': r['return']=-100
    assert contract.aggregate(contract.PROTOCOL,negative)['status']=='complete'


@pytest.mark.parametrize('change',['drop','duplicate','scene','updates','terminal','overrun','nonfinite'])
def test_invalid_partial_or_fabricated_evaluation_rejected(change):
    r=rows()
    if change=='drop': r.pop()
    elif change=='duplicate': r[-1]=deepcopy(r[0])
    elif change=='scene': r[-1]['initial_traffic_sha256']='a'*64
    elif change=='updates': r[-1]['no_policy_updates']=False
    elif change=='terminal': r[-1]['ego_collision']=1
    elif change=='overrun': r[-1]['actual_training_steps']+=500
    else: r[-1]['return']=float('nan')
    with pytest.raises(ValueError): contract.aggregate(contract.PROTOCOL,r)


def test_native_preconditions_verified_read_only_without_new_inference():
    legacy_before=cli.inventory(cli.accepted.LEGACY_DIR)
    h,models,trained,m,spaces,legacy=cli.prerequisites(ROOT.parent/'Prediction_RL')
    assert len(contract.manifest_roster(m))==32 and models['conditional']['sha256']=='daddeda8569b88c2a3482a0e8c739995785ee0f8216dda6525c86d2ce120e25f'
    assert spaces['observation']['shape']==[20] and len(trained['training']['member_seeds'])==3
    assert legacy['legacy_request_remains_failed'] is True
    assert cli.inventory(cli.accepted.LEGACY_DIR)==legacy_before
    assert 'worktree:tools/train_curriculum_ddpg.py' in h


@pytest.mark.parametrize('arm,expected',[('b2_curriculum','natural_arrival'),('b3_mix','ego_collision')])
def test_episode_recorder_replays_saved_native_evidence_without_simulation(arm,expected):
    from prediction_rl.data.merge_geometry import MergeGeometry
    directory=ROOT/'artifacts/ic/ic03b_remaining_v1'/arm
    class MemoryLog:
        def append(self,row): pass
    settings=cli.entry.read(directory/'e0/runtime.json')['settings']
    job=contract.jobs(contract.PROTOCOL,'train')[3 if arm=='b2_curriculum' else 6]
    selection=cli.entry.read(directory/'e0/started.json')['selection']
    recorder=runtime.EpisodeRecorder(MergeGeometry(ROOT/'RL-MPC-LaneMerging-master/merge.net.xml'),
        settings,runtime.identity(job,selection['simulator_seed']),MemoryLog())
    for row in cli.accepted.jsonl(directory/'native.jsonl.gz'):
        if row['episode_index']!=0: continue
        packet=row['packet']; info={**packet['info'],'prediction_features':row['feature_info']}
        recorder.record(packet['phase'],packet,np.asarray(row['feature_observation'],np.float32),
            packet['reward'],packet['done'],info)
    result=recorder.finish()
    assert result['raw_outcome']['classification']==expected and result['native_event_agrees'] is True
    assert result['reward_max_absolute_error']==0 and result['steps']==result['actual_control_steps']
    assert 0<=result['low_speed_fraction']<=1 and 0<=result['action_saturation_fraction']<=1
    assert result['gap_definition'].startswith('minimum_observed_decision_state_same_path')


def request_fixture(tmp_path,monkeypatch):
    monkeypatch.setattr(cli,'ROOT',tmp_path)
    monkeypatch.setattr(cli,'load',lambda p,h:(Path(p),q))
    q={'protocol':deepcopy(contract.PROTOCOL),'train_jobs':[contract.jobs(contract.PROTOCOL,'train')[0]],'evaluation_jobs':[]}
    return tmp_path/'request.json',q


def test_parent_seals_after_child_stdout_closes(tmp_path,monkeypatch):
    path,q=request_fixture(tmp_path,monkeypatch)
    monkeypatch.setattr(cli,'training_metadata',lambda *_:{})
    def child(command,**kwargs):
        out=path.parent/'train'/q['train_jobs'][0]['id']
        write_once(out/'report.json',{'status':'complete'})
        kwargs['stdout'].write('last child stdout line\n'); kwargs['stdout'].flush()
    monkeypatch.setattr(cli.subprocess,'run',child)
    cli.execute(path,'a'*64,'train')
    directory=path.parent/'train'/q['train_jobs'][0]['id']
    receipt=verify_episode(directory,cli.binding(q,'train',q['train_jobs'][0]))
    assert receipt['files']['worker.log']==file_hash(directory/'worker.log')
    assert (directory/'worker.log').read_text()=='last child stdout line\n'


def test_resume_never_restarts_a_partial_training_with_node_weights(tmp_path,monkeypatch):
    path,q=request_fixture(tmp_path,monkeypatch)
    d=path.parent/'train'/q['train_jobs'][0]['id']; d.mkdir(parents=True)
    write_once(d/'started.json',{}); write_once(d/'n20000.json',{})
    def forbidden(*a,**k): raise AssertionError('Must not start a partial job')
    monkeypatch.setattr(cli.subprocess,'run',forbidden)
    with pytest.raises(ValueError,match='Incomplete/failed'): cli.execute(path,'a'*64,'train',True)
    assert (d/'n20000.json').exists() and not (d/'complete.json').exists()
