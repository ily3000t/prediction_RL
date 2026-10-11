from copy import deepcopy
import json
from pathlib import Path
import sys
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'tools'),str(ROOT/'src')]
import collect_curriculum_pilot as pilot
import prediction_rl.curriculum.task_collector as engine


@pytest.fixture
def design():
    return json.loads((ROOT/'configs/development/ic02c_task_design_v1.json').read_text())


def records(design):
    return [{'simulator_seed':s,'proposed_task_id':f't{s}','source_episode_id':f'e{s}',
             'source_job_id':f'j{s}','engineering_stage':s in design['engineering_seeds'],
             'execution_authorized':False,'task_level':None}
            for s in range(11000,11256,8)]


def test_same_design_and_exact_remaining_roster(design):
    config = json.loads((ROOT/pilot.CONFIG).read_text())
    assert pilot.validate_protocol(config,design) == pilot.PROTOCOL
    jobs = pilot.remaining_jobs(records(design),design)
    assert len(jobs) == 29 and set(j['simulator_seed'] for j in jobs).isdisjoint(design['engineering_seeds'])
    changed = {k for k in config if config[k] != pilot.eng.PROTOCOL[k]}
    assert changed == {'version','run_id','authorization','simulator_seeds','tasks_max','root_slots_max',
                       'candidate_branches_max','fresh_sumo_sessions_max','control_steps_max','raw_ticks_max'}


@pytest.mark.parametrize('field,value',[('simulator_seeds',[11000]),('tasks_max',32),('workers',2),
    ('control_steps_max',118263),('automatic_retry',True),('training_ready',True),('unknown',0)])
def test_scope_mutation_rejected(design,field,value):
    config = deepcopy(pilot.PROTOCOL); config[field] = value
    with pytest.raises(ValueError): pilot.validate_protocol(config,design)


@pytest.mark.parametrize('fault',['remove','duplicate','order','level','engineering'])
def test_roster_selection_cannot_be_changed(design,fault):
    rows = records(design)
    if fault == 'remove': rows.pop()
    elif fault == 'duplicate': rows[-1] = deepcopy(rows[-2])
    elif fault == 'order': rows.reverse()
    elif fault == 'level': rows[-1]['task_level'] = 'C1'
    else: rows[-1]['engineering_stage'] = True
    with pytest.raises(ValueError): pilot.remaining_jobs(rows,design)


def test_worker_calls_shared_engine_once_and_refuses_partial_retry(tmp_path,monkeypatch):
    job = {'id':'t11024'}
    q = {'jobs':[job],'design':{},'baseline_root':'readonly','metric_config':{},'normalization_scales':{}}
    path = tmp_path/'request.json'; (tmp_path/'started.json').write_text('{}')
    out = tmp_path/job['id']; out.mkdir()
    (out/'started.json').write_text(json.dumps({'binding':pilot.eng.binding(q,job)}))
    (out/'launch.json').write_text('{}'); (out/'worker.log').write_text('')
    monkeypatch.setattr(pilot,'load',lambda *a:(path,q))
    monkeypatch.setattr(pilot.eng.torch,'set_num_threads',lambda *a:None)
    calls = []; monkeypatch.setattr(engine,'collect_task',lambda *a:calls.append(a))
    pilot.worker(path,'hash',job['id'])
    assert len(calls) == 1 and (out/'worker_started.json').is_file()
    with pytest.raises(ValueError): pilot.worker(path,'hash',job['id'])
    assert len(calls) == 1


@pytest.mark.parametrize('fault',[None,'source','gate','receipt','asset'])
def test_engineering_reuse_requires_frozen_source_and_receipts(tmp_path,monkeypatch,fault):
    out = tmp_path/'artifacts/ic/ic02d_eng_v2'; out.mkdir(parents=True)
    jobs = [{'id':f't{s}'} for s in (11000,11008,11016)]
    q = {'protocol':pilot.eng.PROTOCOL,'git_commit_sha':'3856a78b97a87cf0cac9280161ce576f8d7f8056',
         'jobs':jobs,'input_hashes':{'science.py':'frozen'}}
    report = {'status':'complete','engineering_gate':True,'actual_tasks':3,'training_ready':False,
              'predictors_loaded':False,'tasks':[{'job':j} for j in jobs]}
    if fault == 'source': q['git_commit_sha'] = 'other'
    if fault == 'gate': report['engineering_gate'] = False
    report['request_hash'] = pilot.digest(q)
    for name,value in [('request.json',q),('report.json',report),('postflight.json',{})]:
        (out/name).write_text(json.dumps(value))
    for job in jobs:
        d = out/job['id']; d.mkdir(); (d/'complete.json').write_text('{}')
    monkeypatch.setattr(pilot,'ROOT',tmp_path)
    monkeypatch.setattr(pilot,'ENGINEERING_HASHES',{n:pilot.eng.sha(out/n) for n in pilot.ENGINEERING_HASHES})
    monkeypatch.setattr(pilot,'verify_episode',lambda d,b:{'metadata':{'job':b['job']} if fault!='receipt' else {}})
    hashes = {'science.py':'changed' if fault == 'asset' else 'frozen'}
    if fault is None: assert pilot.engineering_evidence(hashes)['actual_tasks'] == 3
    else:
        with pytest.raises(ValueError): pilot.engineering_evidence(hashes)
