"""Remaining-arm authorization/ledger tests; no new native sessions or inference."""
from copy import deepcopy
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'tools'),str(ROOT/'src')]
import smoke_remaining_curriculum as cli
from prediction_rl.data.collection_store import read_json, write_once, file_hash, verify_episode


@pytest.mark.parametrize('key,value',[('arms',['b2_mix','b3_mix','b3_curriculum']),
    ('sumo_sessions_max',8),('control_steps_max',4000),('raw_ticks_max',4816),
    ('feature_calls_max',4008),('episodes_per_arm',3),('optimizer_updates',1),
    ('optimizer_seed',1),('sampling_seed',1),('formal_training_authorized',True),
    ('run_id','unapproved_retry'),('unknown',False)])
def test_cannot_repeat_old_arm_expand_budget_or_change_training_conditions(key,value):
    value_protocol = deepcopy(cli.PROTOCOL); value_protocol[key] = value
    with pytest.raises(ValueError): cli.validate_protocol(value_protocol)


def test_standalone_config_and_readonly_old_failure_remain_bound():
    before = cli.inventory(cli.LEGACY_DIR)
    p,h,models,trained,tasks,spaces,review = cli.prerequisites(ROOT.parent/'Prediction_RL')
    assert p == cli.PROTOCOL and len(tasks) == 32
    assert set(models) == {'ordinary','conditional'} and spaces['action']['shape'] == [1]
    assert len(trained['training']['member_seeds']) == 3
    assert review['legacy_request_remains_failed'] and review['legacy_receipt_remains_invalid']
    assert review['posthoc_tensor_or_rng_reconstruction'] is False
    assert review['consumed'] == cli.CONSUMED
    assert cli.inventory(cli.LEGACY_DIR) == before
    assert sum(r['actual_control_steps'] for r in review['independent_native_reward_and_tick_review']) == 135
    assert sum(r['raw_ticks'] for r in review['independent_native_reward_and_tick_review']) == 337
    plan = cli.remaining_jobs(tasks)
    original_plan = cli.original.jobs(tasks)
    assert plan == original_plan[1:]
    assert [j['id'] for j in plan] == cli.PROTOCOL['arms'] and len(plan) == 3
    assert plan[0]['planned_tasks'] == plan[2]['planned_tasks']
    assert plan[1]['planned_tasks'] == original_plan[0]['planned_tasks']
    assert 'worktree:tools/smoke_remaining_curriculum.py' in h


def test_legacy_inventory_mismatch_is_not_tolerated(monkeypatch):
    _,h,_,_,tasks,_,_ = cli.prerequisites(ROOT.parent/'Prediction_RL')
    original_inventory = cli.inventory
    def corrupted_inventory(path):
        values = original_inventory(path)
        if Path(path) == cli.LEGACY_DIR: values['b2_mix/summary.json'] = '0'*64
        return values
    monkeypatch.setattr(cli,'inventory',corrupted_inventory)
    with pytest.raises(ValueError,match='Legacy failed-run bytes'):
        cli.review_legacy(tasks,h)


def test_legacy_scientific_source_mismatch_is_not_ignored():
    _,h,_,_,tasks,_,_ = cli.prerequisites(ROOT.parent/'Prediction_RL')
    h['worktree:src/prediction_rl/training/all_ddpg.py'] = '0'*64
    with pytest.raises(ValueError,match='Legacy scientific input changed'):
        cli.review_legacy(tasks,h)


def request_fixture(tmp_path,monkeypatch):
    from test_curriculum_integration_smoke import proposals
    tasks = cli.original.task_roster(proposals())
    q = {'version':cli.SCOPE,'git_commit_sha':'synthetic-source','protocol':deepcopy(cli.PROTOCOL),
         'runtime':{},'jobs':cli.remaining_jobs(tasks),'prior_evidence_review':{
             'consumed':deepcopy(cli.CONSUMED),'legacy_request_remains_failed':True,
             'legacy_receipt_remains_invalid':True}}
    request = tmp_path/'request.json'; write_once(request,q)
    monkeypatch.setattr(cli,'load',lambda path,expected:(request,q))
    return request,q


def test_remaining_coordinator_runs_only_three_arms_and_seals_after_exit(tmp_path,monkeypatch):
    request,q = request_fixture(tmp_path,monkeypatch)
    calls = []
    def finished_child(command,**kwargs):
        assert Path(command[2]) == Path(cli.__file__).resolve()
        job_id = command[command.index('--job')+1]; calls.append(job_id)
        assert job_id != 'b2_mix'
        row = {'job':next(j for j in q['jobs'] if j['id'] == job_id),
               'episodes':[{'actual_control_steps':2}]*2,'actual_control_steps':4,
               'raw_ticks':206,'feature_calls':4,'optimizer_updates':0}
        write_once(tmp_path/job_id/'summary.json',row)
        assert not (tmp_path/job_id/'complete.json').exists()
        kwargs['stdout'].write('last child message\n')
        kwargs['stdout'].flush()
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(cli.original.subprocess,'run',finished_child)
    cli.execute(request,'a'*64)
    assert calls == cli.PROTOCOL['arms']
    report = read_json(tmp_path/'report.json')
    review = read_json(tmp_path/'coverage_review.json')
    assert report['actual_episodes'] == 6 and report['engineering_gate']
    assert review['combined_consumption'] == {'sumo_sessions':8,'control_steps':147,
        'raw_ticks':955,'feature_calls':147,'optimizer_updates':0}
    assert review['new_remaining_report_sha256'] == file_hash(tmp_path/'report.json')
    assert review['legacy_request_remains_failed'] and review['legacy_receipt_remains_invalid']
    assert not review['formal_training_authorized'] and not review['training_ready']
    for job in q['jobs']:
        receipt = verify_episode(tmp_path/job['id'],{'request_hash':'a'*64,'job':job})
        assert 'worker.log' in receipt['files']
    with pytest.raises(ValueError,match='No automatic replay'):
        cli.execute(request,'a'*64)
    assert len(calls) == 3


def test_remaining_failure_stops_without_advancing_or_resetting_budget(tmp_path,monkeypatch):
    request,q = request_fixture(tmp_path,monkeypatch)
    calls = []
    def failed_child(command,**kwargs):
        calls.append(command)
        raise cli.original.subprocess.CalledProcessError(1,command)
    monkeypatch.setattr(cli.original.subprocess,'run',failed_child)
    with pytest.raises(cli.original.subprocess.CalledProcessError): cli.execute(request,'a'*64)
    failure = read_json(tmp_path/'failure.json')
    assert not failure['engineering_gate']
    assert failure['prior_evidence_review']['consumed'] == cli.CONSUMED
    assert not (tmp_path/'coverage_review.json').exists()
    assert not (tmp_path/q['jobs'][1]['id']).exists()
    with pytest.raises(ValueError,match='No automatic replay'): cli.execute(request,'a'*64)
    assert len(calls) == 1


def test_remaining_parent_cap_is_not_original_eight_session_cap(tmp_path,monkeypatch):
    request,q = request_fixture(tmp_path,monkeypatch)
    def too_many_episodes(command,**kwargs):
        job_id = command[command.index('--job')+1]
        write_once(tmp_path/job_id/'summary.json',{'job':next(j for j in q['jobs'] if j['id']==job_id),
            'episodes':[{}]*3,'actual_control_steps':6,'raw_ticks':309,'feature_calls':6})
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(cli.original.subprocess,'run',too_many_episodes)
    with pytest.raises(ValueError,match='Smoke budget exceeded'): cli.execute(request,'a'*64)
    assert not read_json(tmp_path/'failure.json')['engineering_gate']
    assert not (tmp_path/'coverage_review.json').exists()


def test_shared_worker_requires_explicit_unchanged_per_job_sampler_contract(tmp_path):
    with pytest.raises(ValueError,match='Changed approved eight-episode smoke'):
        cli.original.worker(tmp_path/'request.json','a'*64,'b3_mix',
            request_loader=lambda path,expected:(Path(path),{'protocol':deepcopy(cli.PROTOCOL)}))
