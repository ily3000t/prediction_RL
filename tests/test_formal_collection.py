from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src')); sys.path.insert(0, str(ROOT/'tools'))
from prediction_rl.data.collection_store import write_once, read_json, verify_episode, seal_episode, file_hash
from prediction_rl.data.collection_protocol import collection_plan, selected_jobs, summarize, validate_job_metadata
from prediction_rl.data.dataset_contract import digest, episode_id
from prediction_rl.data.response_collection import base_traffic
import prediction_rl.data.episode_collector as engine
import collect_response_dataset as cli


def config():
    return read_json(ROOT/'configs/formal/p04_response_dataset_v1.json')


def request():
    p = config()
    return {'protocol': p, 'plan': collection_plan(p)}


def test_release_does_not_change_draft_experimental_parameters():
    old = read_json(ROOT/'configs/formal/p04_response_dataset_v1_draft.json')
    new = config(); new.pop('execution')
    new['protocol_id'] = old['protocol_id']; new['review_status'] = old['review_status']
    assert new == old
    r = request()
    jobs = selected_jobs(r, ['train', 'validation', 'calibration'], digest(r))
    assert len(jobs) == 384 and {j['split'] for j in jobs} == {'train','validation','calibration'}
    assert len(r['plan']['jobs']) == 512 and len(r['plan']['split_manifest']['test']) == 128
    assert not r['plan']['formal_training_ready'] and r['plan']['test_locked']
    assert r['plan']['status'] == 'awaiting_manual_start'


@pytest.mark.parametrize('change', ['draft','unknown','test_release','no_confirmation','budget_bool','seeds_overlap'])
def test_unsafe_release_rejected(change):
    p = config()
    if change == 'draft': p['review_status'] = 'draft_not_approved'
    if change == 'unknown': p['execution']['unknown'] = 0
    if change == 'test_release': p['execution']['allowed_splits'].append('test')
    if change == 'no_confirmation': p['execution']['require_hash_confirmation'] = False
    if change == 'budget_bool': p['execution']['worker_wall_limit_s'] = True
    if change == 'seeds_overlap': p['splits']['train']['start'] = 100
    with pytest.raises(ValueError): collection_plan(p)


@pytest.mark.parametrize('splits', [[], ['test'], ['train','test'], ['development'], ['train','train']])
def test_no_test_development_or_ambiguous_execution(splits):
    r = request()
    with pytest.raises(ValueError): selected_jobs(r, splits, digest(r))


def test_changed_request_or_job_manifest_rejected():
    r = request(); old_hash = digest(r)
    with pytest.raises(ValueError): selected_jobs(r, ['train'], 'wrong')
    r['plan']['jobs'][0]['simulator_seed'] = 100
    with pytest.raises(ValueError): selected_jobs(r, ['train'], old_hash)
    with pytest.raises(ValueError): selected_jobs(r, ['train'], digest(r))


def unavailable(job, targets=('approach','internal_mid','downstream')):
    return {'status':'complete','seed':job['simulator_seed'],'episode_id':job['episode_id'],'split':job['split'],
            'roots':[{'target':t,'status':'unavailable','reason':'upstream_collision','candidate_count':0} for t in targets]}


def test_all_unreachable_episodes_remain_complete_data_not_success_gate():
    j = request()['plan']['jobs'][0]; m = unavailable(j)
    out = summarize([(j,m)], ['approach','internal_mid','downstream'])
    assert out['episodes'] == out['episodes_without_reached_roots'] == 1
    assert out['unique_candidates'] == out['reached_roots'] == 0
    assert all(v['unavailable'] == 1 for v in out['locations'].values())
    assert out['coverage_review_required'] and not out['formal_training_ready']


@pytest.mark.parametrize('change', ['seed','split','episode','pending','missing_target','reason'])
def test_receipt_semantics_checked_not_only_hashes(change):
    j = request()['plan']['jobs'][0]; m = unavailable(j)
    if change == 'seed': m['seed'] += 1
    if change == 'split': m['split'] = 'test'
    if change == 'episode': m['episode_id'] = 'x'
    if change == 'pending': m['roots'][0]['status'] = 'pending'
    if change == 'missing_target': m['roots'].pop()
    if change == 'reason': m['roots'][0]['reason'] = ''
    with pytest.raises(ValueError): validate_job_metadata(j,m,['approach','internal_mid','downstream'])


def fake_engine(monkeypatch, tmp_path, *, missing=False, fail_repeat=False):
    p = config(); c = p['collection']; c['root_targets'] = c['root_targets'][:1]
    seed = 11000; ep = episode_id(p['baseline']['episode_group_namespace_sha256'],seed,0)
    job = {'split':'train','simulator_seed':seed,'episode_id':ep}
    def actor(pos):
        return {'position':[pos,0.], 'speed':7.,'acceleration':0.,'lane_id':'highwayrear_0',
                'lane_position_m':pos,'length_m':5.,'width_m':1.8}
    history = [{'simulation_time_s':0.,'vehicles':{'ego':actor(100),'n':actor(120)}}]
    traffic = base_traffic(history[0])
    root = SimpleNamespace(prefix=(),signature='a'*64,prefix_trace_hash='b'*64,traffic=traffic)
    discovery = {'targets':[{'target':c['root_targets'][0], 'prefix_steps':None, 'reason':'upstream_collision'}]}
    if not missing:
        discovery['targets'][0] = {'target':c['root_targets'][0],'prefix_steps':0,
                                  'prefix_actions':[],'selection':{'traffic':traffic}}
    counters = {'discovery':0,'branches':0,'closed':False}
    class Manager:
        backend = 'fresh_sumo_prefix_replay_v1'
        def discover_lane_roots(self, *a, **kw):
            counters['discovery'] += 1; return deepcopy(discovery)
        def capture_with_history(self, *a): return root,deepcopy(history)
        def close(self): counters['closed'] = True
    @contextmanager
    def session(*args):
        yield SimpleNamespace(MINIMUM_NEGATIVE_JERK=-5,MAXIMUM_POSITIVE_JERK=5,TICK_LENGTH=.2,
                              CUDA=False,export_settings=lambda:{'reward':'unchanged'})
    def branch(manager, root, actions, geometry):
        counters['branches'] += 1
        after = {'simulation_time_s':.2,'vehicles':{}}
        step = {'done':True,'reward':-10.,'observation':[0.], 'info':{'execution_audit':{
                'before':traffic,'after':after,'requested_jerk':actions[0],'simulation_elapsed_s':.2,
                'termination_reason':'upstream_collision','execution_contract':'simulation_blocking_exact_v1',
                'commanded_speed_reconstructed':7.,'upstream_reward_projected_jerk':actions[0],
                'invalid_action_reward':0.,'measured_post_step_speed':None}}}
        if fail_repeat and counters['branches'] > 5: step['reward'] -= 1
        return [step], {'contract':'lane_position_length_width_v1','frames':[after],'trajectory_usable':[False]}
    monkeypatch.setattr(engine,'ReplayBrancher',Manager)
    monkeypatch.setattr(engine,'upstream_session',session)
    monkeypatch.setattr(engine,'AuthorDDPGReference',lambda *a: object())
    monkeypatch.setattr(engine.MergeGeometry,'verify_runtime',lambda self: None)
    monkeypatch.setattr(engine,'branch_with_geometry',branch)
    path = tmp_path/'episode'
    args = (tmp_path,p['baseline'],c,ROOT/'RL-MPC-LaneMerging-master/merge.net.xml',job,path,{'job':job})
    return args,counters,path


def test_engine_generates_new_labels_without_historical_artifacts(monkeypatch,tmp_path):
    args,count,path = fake_engine(monkeypatch,tmp_path)
    receipt = engine.collect_episode(*args)
    assert count == {'discovery':2,'branches':10,'closed':True}
    pack = read_json(path/'r0/labels.json'); hist = read_json(path/'r0/history.json')
    assert pack['split'] == hist['split'] == 'train' and len(pack['candidates']) == 5
    assert all(c['labels']['events']['upstream_collision']['value'] == 1 for c in pack['candidates'])
    assert receipt['metadata']['roots'][0]['terminal_branches'] == 5
    assert hist['label_pack_sha256'] == file_hash(path/'r0/labels.json')
    assert verify_episode(path,args[-1]) == receipt
    summary = summarize([(args[4], receipt['metadata'])], ['approach'])
    assert summary['unique_candidates'] == 5 and summary['termination_counts']['upstream_collision'] == 5
    assert summary['valid_selected_actor_time_cells'] == 0 and summary['empty_trajectory_candidates'] == 5
    assert not summary['formal_training_ready']


def test_engine_accounts_missing_root_without_collecting_replacement(monkeypatch,tmp_path):
    args,count,path = fake_engine(monkeypatch,tmp_path,missing=True)
    r = engine.collect_episode(*args)
    assert count['branches'] == 0 and r['metadata']['roots'][0]['status'] == 'unavailable'
    assert read_json(path/'r0/accounting.json')['candidate_count'] == 0
    assert not (path/'r0/labels.json').exists()


def test_engine_repeat_failure_preserves_partial_artifacts(monkeypatch,tmp_path):
    args,count,path = fake_engine(monkeypatch,tmp_path,fail_repeat=True)
    with pytest.raises(AssertionError,match='Reverse-order'): engine.collect_episode(*args)
    assert count['closed'] and (path/'failure.json').exists() and (path/'r0/c0.json').exists()
    assert not (path/'complete.json').exists()


def test_engine_rejects_test_before_starting_worker(monkeypatch,tmp_path):
    args,count,path = fake_engine(monkeypatch,tmp_path)
    args[4]['split'] = 'test'
    with pytest.raises(ValueError,match='locked'): engine.collect_episode(*args)
    assert not path.exists() and count['discovery'] == 0


def test_response_metric_does_not_compare_terminal_or_mismatched_times():
    def f(t,x,done=False):
        return {'done':done,'info':{'execution_audit':{'after':{
            'simulation_time_s':t,'vehicles':{'n':{'position':[x,0.],'speed':x}}}}}}
    result = engine.response_magnitude({'0':[f(.2,1),f(.6,100,True)],'1':[f(.2,2),f(.4,3)]},{'ego','n'})
    assert result['paired_nonterminal_neighbor_cells'] == 1
    assert result['max_neighbor_position_delta_m'] == result['max_neighbor_speed_delta_mps'] == 1
    assert engine.response_magnitude({'0':[f(.2,1,True)],'1':[f(.4,3)]},{'n'})['max_neighbor_speed_delta_mps'] is None


def mock_parent(monkeypatch,tmp_path,fail=False):
    r = request(); jobs = r['plan']['jobs'][:2]
    output = tmp_path/'data'; output.mkdir()
    monkeypatch.setattr(cli,'clean_tree',lambda: None)
    monkeypatch.setattr(cli,'load_request',lambda path:(output,r))
    monkeypatch.setattr(cli,'selected_jobs',lambda *a:jobs)
    monkeypatch.setattr(cli,'provenance',lambda:{'git_commit':'test'})
    monkeypatch.setattr(cli.audit,'verify_inputs',lambda hashes:None)
    r['input_hashes'] = {}
    calls = []
    def run(command,**kw):
        job = next(j for j in jobs if j['job_id'] == command[command.index('--job-id')+1])
        calls.append(job['job_id']); directory = output/job['job_id']
        write_once(directory/'started.json',{})
        if fail: raise RuntimeError('bounded injected failure')
        seal_episode(directory,cli.binding(r,job),unavailable(job))
    monkeypatch.setattr(cli.subprocess,'run',run)
    args = SimpleNamespace(request='unused',splits=['train'],confirm_request_hash=digest(r),resume=False)
    return args,output,calls


def test_manual_executor_resume_does_not_restart_completed_workers(monkeypatch,tmp_path):
    args,out,calls = mock_parent(monkeypatch,tmp_path)
    first = cli.execute(args); h = file_hash(first)
    args.resume = True
    second = cli.execute(args)
    assert calls == ['e0000','e0001'] and first != second and file_hash(first) == h
    report = read_json(second)
    assert report['executed_jobs'] == [] and report['reused_jobs'] == calls
    assert report['summaries']['train']['episodes_without_reached_roots'] == 2
    assert report['status'] == 'complete' and not report['formal_training_ready']


def test_manual_executor_failure_stops_and_cannot_silently_retry(monkeypatch,tmp_path):
    args,out,calls = mock_parent(monkeypatch,tmp_path,fail=True)
    with pytest.raises(RuntimeError): cli.execute(args)
    args.resume = True
    with pytest.raises(ValueError,match='automatic retry'): cli.execute(args)
    assert calls == ['e0000'] and not (out/'e0001').exists()
    reports = list((out/'invocations').glob('*/report.json'))
    assert len(reports) == 2 and all(read_json(p)['status']=='failed' for p in reports)


def test_prepare_is_plan_only(monkeypatch,tmp_path):
    p = config(); config_path = tmp_path/'config.json'; write_once(config_path,p)
    reference = {'checkpoint':p['baseline']['reference_checkpoint']}
    reports = {'mechanism':{'plan':{'upstream_config':p['baseline']['upstream_config'],
               'reference_policy':reference,'root_targets':p['collection']['root_targets']},
               'policy_checkpoint_hash':p['baseline']['reference_sha256']},
               'label':{'environment_identity_sha256':p['baseline']['episode_group_namespace_sha256']}}
    monkeypatch.setattr(cli,'ROOT',tmp_path); monkeypatch.setattr(cli,'clean_tree',lambda:None)
    monkeypatch.setattr(cli,'verify_acceptance',lambda path:(reports,{}))
    monkeypatch.setattr(cli,'sha',lambda path: next(p['sources'][k+'_sha256'] for k in
                         ('mechanism','training_interface') if Path(path).as_posix().endswith(p['sources'][k+'_report'])))
    monkeypatch.setattr(cli,'environment',lambda:{'test':True}); monkeypatch.setattr(cli,'provenance',lambda:{})
    monkeypatch.setattr(cli,'collect_episode',lambda *a,**k:pytest.fail('Prepare must not simulate'))
    monkeypatch.setattr(cli.subprocess,'run',lambda *a,**k:pytest.fail('Prepare must not launch worker'))
    args = SimpleNamespace(config=str(config_path),collector_report=str(tmp_path/'audit.json'),run_id='planned')
    output = cli.prepare(args)
    report = read_json(output.parent/'preparation.json')
    assert report['executable_episode_budget'] == 384 and report['locked_test_episode_budget'] == 128
    assert not report['collection_started'] and not list(output.parent.glob('e[0-9]*'))


@pytest.mark.parametrize('case', ['test_job','wrong_token','outside_authorization'])
def test_worker_cannot_bypass_parent_or_unlock_test(monkeypatch,tmp_path,case):
    r = request(); output = tmp_path/'data'; output.mkdir()
    token = 'owner'; write_once(output/'writer.lock',{'token':token})
    jobs = selected_jobs(r,['train'],digest(r))
    authorization = {'splits':['train'],'request_hash':digest(r),
                     'job_ids':[j['job_id'] for j in jobs],'lock_token':token}
    if case == 'wrong_token': authorization['lock_token'] = 'stale'
    path = output/'invocations/i/authorization.json'
    if case == 'outside_authorization': path = tmp_path/'authorization.json'
    write_once(path,authorization)
    monkeypatch.setattr(cli,'load_request',lambda path:(output,r))
    monkeypatch.setattr(cli,'collect_episode',lambda *a,**kw:pytest.fail('Unauthorized worker reached engine'))
    job = 'e0384' if case == 'test_job' else 'e0000'
    with pytest.raises(ValueError):
        cli.worker(SimpleNamespace(request='unused',authorization=str(path),job_id=job))
