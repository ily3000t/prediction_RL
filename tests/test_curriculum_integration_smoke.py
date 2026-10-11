"""IC03b scope/budget tests. Synthetic learner loops, no SUMO or real inference."""
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'tools')]
from prediction_rl.curriculum.integration_smoke import (SCOPE, PROTOCOL, NativePacketAudit,
    SmokeEpisodeSampler, jobs, task_roster, validate_protocol)


def proposals():
    return {'tasks':[{'task_id':f't{s}', 'simulator_seed':s, 'proposed_level':
        'L' if i < 11 else 'M' if i < 21 else 'H', 'task_level':None,
        'classification_applied':False, 'execution_authorized':False}
        for i,s in enumerate(range(11000,11249,8))]}


def test_config_matches_explicit_live_smoke_not_formal_authorization():
    config = json.loads((ROOT/'configs/development/ic03b_persistent_smoke_v1.json').read_text())
    assert validate_protocol(config) == PROTOCOL
    assert config['optimizer_updates'] == 0 and config['formal_training_authorized'] is False
    assert config['sumo_sessions_max'] == 4*2 == 8
    assert config['raw_ticks_max'] == 8*(101+500+1)
    assert config['feature_calls_max'] == 8*(500+1)


@pytest.mark.parametrize('key,value', [('native_control_cap',501),('episodes_per_arm',3),
    ('optimizer_updates',1),('formal_training_authorized',True),('device','cuda'),
    ('optimizer_seed',2),('sampling_seed',17),('original_replay_warmup',8),
    ('original_minibatch_size',8),('test_opened',True),('unknown',False)])
def test_scope_cannot_expand_or_change_original_semantics(key,value):
    config = deepcopy(PROTOCOL); config[key] = value
    with pytest.raises(ValueError): validate_protocol(config)


def test_live_scope_is_distinct_capped_sampler_and_plan_is_reproducible():
    tasks = task_roster(proposals()); plan = jobs(tasks)
    assert len(plan) == 4 and jobs(tasks) == plan
    assert plan[0]['planned_tasks'] == plan[2]['planned_tasks']
    assert plan[1]['planned_tasks'] == plan[3]['planned_tasks']
    for job in plan:
        sampler = SmokeEpisodeSampler(tasks,job,'a'*64,PROTOCOL)
        first = sampler.begin_episode()
        assert first['smoke_scope'] == SCOPE and first['smoke_execution_authorized']
        assert first['formal_training_authorized'] is False
        with pytest.raises(ValueError): sampler.begin_episode()
        sampler.end_episode(500)
        second = sampler.begin_episode()
        assert second['actual_start_step'] == 500 and second['collection_stage'] == 0
        sampler.end_episode(17)
        assert sampler.started == sampler.completed == 2
        with pytest.raises(ValueError): sampler.begin_episode()


@pytest.mark.parametrize('fault',['missing','duplicate','live_label','replaced_seed','layer'])
def test_no_outcome_selection_or_changed_source_pool(fault):
    pack = proposals()
    if fault == 'missing': pack['tasks'].pop()
    elif fault == 'duplicate': pack['tasks'][1]['task_id'] = pack['tasks'][0]['task_id']
    elif fault == 'live_label': pack['tasks'][0]['classification_applied'] = True
    elif fault == 'replaced_seed': pack['tasks'][0]['simulator_seed'] = 200
    else: pack['tasks'][0]['proposed_level'] = 'M'
    with pytest.raises(ValueError): task_roster(pack)


def frame(tick, ego=True, acceleration=0.):
    return {'simulation_time_s':tick*.2, 'vehicles':{'ego':{'position':[0.,0.],
        'speed':7.,'acceleration':acceleration,'lane_id':'ramp_0'}} if ego else {}}


def packet(index, reason=None):
    deadline = reason == 'upstream_time_limit'
    count = 101+index+int(deadline)
    start = 0 if not index else 100+index
    endframe = frame(count, not deadline)
    projection = deepcopy(endframe)
    reward = None if not index else 10. if reason == 'upstream_arrival' else -10. if reason == 'upstream_collision' else -.02
    info = {'traffic':projection} if not index else {'execution_audit':{
        'after':projection,'before':frame(start),'requested_jerk':0.,'measured_jerk':0.,
        'invalid_action_reward':0.,'termination_reason':reason,'execution_contract':'simulation_blocking_exact_v1'}}
    info['prediction_features'] = {'history_frames':min(11,index+1) if reason is None else min(11,index),
                                   'terminal_zero_block':reason is not None}
    value = {'phase':'reset' if not index else 'step','control_step':index,'done':reason is not None,
        'reward':reward,'observation':np.zeros(20,np.float32),'current_frame':endframe,'info':deepcopy(info),
        'raw_step_count':count,'raw_events':{'steps':[{'index':i,
            'before':{'simulation_time_s':i*.2},'after':{'simulation_time_s':(i+1)*.2,
                'ego':{'acceleration':0.} if not deadline or i < count-1 else None}}
            for i in range(start,count)], 'creations':[] if index else [{'vehicle_id':'ego',
            'route_id':'rampRoute','simulation_time_s':20.,'depart_position_m':40.,'arrival_position_m':50.}]}}
    return value,np.zeros(169,np.float32),reward,reason is not None,info


def checker():
    return NativePacketAudit(SimpleNamespace(validate_frame=lambda *a,**k:None),
        {'REWARD_FUNCTION':'Slotted Jerk','TIME_REWARD':-.1,'ALT_J_WEIGHT':.1,
         'SUCCESS_REWARD':10.,'CRASH_REWARD':-10.})


@pytest.mark.parametrize('reason',['upstream_arrival','upstream_collision','upstream_time_limit'])
def test_current_packet_full_native_episode_and_deadline_precleanup_reward(reason):
    audit = checker()
    audit.accept(*packet(0)); audit.accept(*packet(1)); audit.accept(*packet(2,reason))
    result = audit.result()
    assert result['actual_control_steps'] == 2 and result['feature_calls'] == 2
    assert result['raw_ticks'] == 103+int(reason=='upstream_time_limit')
    assert result['reward_max_absolute_error'] < 1e-7 and result['native_terminal_verified']


@pytest.mark.parametrize('fault',['tick','frame','reward','done','history','input','insertion','future','extra_tick'])
def test_bad_native_evidence_does_not_pass_as_a_normal_episode(fault):
    audit = checker()
    row = list(packet(0 if fault == 'insertion' else 1))
    if fault != 'insertion': audit.accept(*packet(0))
    value,obs,reward,done,info = row
    if fault == 'tick': value['raw_events']['steps'][0]['index'] += 1
    elif fault == 'frame': value['current_frame']['vehicles']['ego']['speed'] += 1
    elif fault == 'reward': row[2] += 1
    elif fault == 'done': row[3] = True
    elif fault == 'history': info['prediction_features']['history_frames'] = 11
    elif fault == 'input': obs[0] = 1.
    elif fault == 'insertion': value['raw_events']['creations'][0]['depart_position_m'] = 41.
    elif fault == 'future': value['current_frame']['simulation_time_s'] += .2
    else: value['raw_events']['steps'].append(deepcopy(value['raw_events']['steps'][0]))
    with pytest.raises(ValueError): audit.accept(*row)


def test_missing_native_terminal_cannot_be_replaced_by_engineering_cap():
    audit = checker(); audit.accept(*packet(0)); audit.accept(*packet(1))
    with pytest.raises(ValueError): audit.result()


def test_real_scope_requires_original_preset_not_synthetic_short_warmup(tmp_path):
    import gym
    from all.logging import DummyWriter
    from prediction_rl.envs.curriculum_stream import CurriculumTaskStream
    from prediction_rl.training.curriculum_replay import PersistentTaskReplayAudit
    from prediction_rl.training.all_ddpg import preset_factory
    tasks = task_roster(proposals()); job = jobs(tasks)[0]
    sampler = SmokeEpisodeSampler(tasks,job,'a'*64,PROTOCOL)
    stream = CurriculumTaskStream(sampler,lambda s:None,gym.spaces.Box(-1.,1.,(20,),np.float32),
                                  gym.spaces.Box(-5.,5.,(1,),np.float32),purpose=SCOPE)
    audit = PersistentTaskReplayAudit(stream,purpose=SCOPE)
    env = SimpleNamespace(feature_arm='ordinary', state_space=gym.spaces.Box(-1.,1.,(169,),np.float32),
                          action_space=stream.action_space)
    checked = preset_factory('cpu',.0002,smoke=True)(env,DummyWriter())
    with pytest.raises(ValueError): audit.install(checked)


def test_preparation_source_validator_is_read_only_and_binds_full_sealed_pool():
    import smoke_persistent_curriculum as cli
    p,h,predictors,trained,tasks,spaces = cli.prerequisites(ROOT.parent/'Prediction_RL')
    assert p == PROTOCOL and len(tasks) == 32 and set(predictors) == {'ordinary','conditional'}
    assert spaces['observation']['shape'] == [20]
    assert cli.ADDED_SOURCES.issubset(h)
    assert trained['training']['member_seeds'] and len(trained['training']['member_seeds']) == 3


def test_approved_scope_original_loop_and_initial_target_hash_without_sumo(tmp_path):
    import gym
    import torch
    from test_curriculum_persistent_learner import SyntheticWorker, Provider
    from prediction_rl.envs.curriculum_stream import CurriculumTaskStream
    from prediction_rl.envs.prediction_features import PredictionFeatureEnv
    from prediction_rl.training.curriculum_replay import PersistentTaskReplayAudit
    from prediction_rl.training.all_ddpg import preset_factory, make_all_environment, experiment_type
    from smoke_persistent_curriculum import learner_hashes
    tasks = task_roster(proposals()); job = jobs(tasks)[0]
    sampler = SmokeEpisodeSampler(tasks,job,'a'*64,PROTOCOL)
    space = gym.spaces.Box(-np.inf,np.inf,(20,),np.float32)
    stream = CurriculumTaskStream(sampler,lambda s:SyntheticWorker(s,[]),space,
                                  gym.spaces.Box(-5.,5.,(1,),np.float32),purpose=SCOPE)
    audit = PersistentTaskReplayAudit(stream,purpose=SCOPE)
    features = PredictionFeatureEnv(stream,'ordinary',Provider('ordinary'),stream.current_frame)
    env = make_all_environment(features,'cpu',audit.record)
    torch.set_num_threads(1)
    experiment = experiment_type(tmp_path)(preset_factory('cpu',.0002,smoke=False),env,quiet=True)
    audit.install(experiment._agent)
    before = learner_hashes(experiment._agent)
    for _ in range(2): experiment._run_training_episode()
    assert learner_hashes(experiment._agent) == before
    assert audit.sample_calls == 0 and audit.snapshot()['actual_control_steps'] == 6
    assert experiment._agent.agent._frames_seen == 8
    assert not experiment._agent.agent.q._optimizer.state
    assert not experiment._agent.agent.policy._optimizer.state
    with pytest.raises(ValueError): env.reset()
    experiment._writer.close(); env.close()


def test_parent_seals_after_child_cleanup_and_stdout_close(tmp_path,monkeypatch):
    import smoke_persistent_curriculum as cli
    from prediction_rl.data.collection_store import read_json, verify_episode, write_once
    plan = jobs(task_roster(proposals()))
    request = tmp_path/'request.json'
    q = {'git_commit_sha':'synthetic-source', 'protocol':deepcopy(PROTOCOL), 'runtime':{}, 'jobs':plan}
    write_once(request,q)
    monkeypatch.setattr(cli,'load',lambda path,expected:(request,q))
    handles = []
    original_seal = cli.seal_episode

    def finished_child(command,**kwargs):
        job_id = command[command.index('--job')+1]
        job = next(j for j in plan if j['id'] == job_id)
        directory = tmp_path/job_id
        assert not (directory/'complete.json').exists()
        handles.append(kwargs['stdout'])
        row = {'job':job, 'episodes':[{'actual_control_steps':1}]*2,
               'actual_control_steps':2, 'raw_ticks':204, 'feature_calls':2}
        write_once(directory/'summary.json',row)
        # Model the actual bug: summary exists before final stdout and cleanup.
        kwargs['stdout'].write('smoke_arm_complete='+job_id+'\n')
        kwargs['stdout'].flush()
        write_once(directory/'cleanup_finished.json',{'closed':True})
        return SimpleNamespace(returncode=0)

    def seal_after_exit(directory,binding,metadata):
        assert handles[-1].closed
        assert (directory/'cleanup_finished.json').is_file()
        return original_seal(directory,binding,metadata)

    monkeypatch.setattr(cli.subprocess,'run',finished_child)
    monkeypatch.setattr(cli,'seal_episode',seal_after_exit)
    cli.execute(request,'a'*64)
    report = read_json(tmp_path/'report.json')
    assert report['engineering_gate'] and report['actual_episodes'] == 8
    assert report['actual_control_steps'] == 8
    for job in plan:
        binding = {'request_hash':'a'*64,'job':job}
        receipt = verify_episode(tmp_path/job['id'],binding)
        assert {'worker.log','cleanup_finished.json'}.issubset(receipt['files'])
    # Integrity remains strict: a later log mutation is still rejected.
    with (tmp_path/plan[0]['id']/'worker.log').open('a') as log:
        log.write('unexpected late mutation\n')
    with pytest.raises(ValueError,match='artifacts changed'):
        verify_episode(tmp_path/plan[0]['id'],{'request_hash':'a'*64,'job':plan[0]})


def test_failed_child_cannot_publish_receipt_or_automatically_retry(tmp_path,monkeypatch):
    import smoke_persistent_curriculum as cli
    from prediction_rl.data.collection_store import read_json, write_once
    plan = jobs(task_roster(proposals()))
    request = tmp_path/'request.json'
    q = {'git_commit_sha':'synthetic-source', 'protocol':deepcopy(PROTOCOL), 'runtime':{}, 'jobs':plan}
    write_once(request,q)
    monkeypatch.setattr(cli,'load',lambda path,expected:(request,q))
    calls = []

    def failed_child(command,**kwargs):
        calls.append(command)
        write_once(tmp_path/plan[0]['id']/'summary.json',{'partial':True})
        raise cli.subprocess.CalledProcessError(1,command)

    monkeypatch.setattr(cli.subprocess,'run',failed_child)
    with pytest.raises(cli.subprocess.CalledProcessError): cli.execute(request,'a'*64)
    assert read_json(tmp_path/'failure.json')['engineering_gate'] is False
    assert not (tmp_path/plan[0]['id']/'complete.json').exists()
    assert not (tmp_path/plan[1]['id']).exists()
    with pytest.raises(ValueError,match='No automatic replay'): cli.execute(request,'a'*64)
    assert len(calls) == 1
