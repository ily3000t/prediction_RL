"""IC02e: approved remaining 29 train tasks; shared collector, no training."""
import argparse
from pathlib import Path
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'tools'), str(ROOT/'src')]
import collect_curriculum_tasks as eng
from prediction_rl.curriculum.task_design import replay_budget, validate_design
from prediction_rl.data.collection_store import run_lock, write_once, verify_episode
from prediction_rl.data.dataset_contract import digest

PROTOCOL = {**eng.PROTOCOL, 'version':'ic02e-remaining-task-pilot-v1', 'run_id':'ic02e_pilot_v1',
    'authorization':'provenance/ic02e_pilot_authorization.json',
    'simulator_seeds':list(range(11024,11249,8)), 'tasks_max':29, 'root_slots_max':87,
    'candidate_branches_max':870, 'fresh_sumo_sessions_max':1102,
    'control_steps_max':118262, 'raw_ticks_max':230666}
CONFIG = 'configs/development/ic02e_remaining_task_pilot_v1.json'
ENGINEERING_HASHES = {
    'request.json':'306f65b90d8562757f462bce9ccfab07ce2c1d9da8cf064ee543ff6ac885003a',
    'report.json':'d4a57833d741869a8351062ec56a0906caf7b19e075d64569654e289ab56009c',
    'postflight.json':'2cdb2d1d2553ae02ada7f2a5f4e6dd4aded632468b625bf2f0a6be3b0fbc70cd'}


def validate_protocol(value, design):
    validate_design(design)
    if not isinstance(value,dict) or digest(value) != digest(PROTOCOL):
        raise ValueError('Changed approved remaining-task scope')
    expected_seeds = [s for s in range(design['source_train_seed_range'][0],
        design['source_train_seed_range'][1]+1,design['pilot_seed_stride']) if s not in design['engineering_seeds']]
    budget = replay_budget(29,design)
    if (value['simulator_seeds'] != expected_seeds or len(expected_seeds) != 29
            or any(value[k] != budget[b] for k,b in {
                'tasks_max':'unique_training_tasks','root_slots_max':'reference_checkpoint_slots_max',
                'candidate_branches_max':'candidate_branch_executions_max',
                'fresh_sumo_sessions_max':'fresh_sumo_sessions_max',
                'control_steps_max':'total_control_steps_max','raw_ticks_max':'total_raw_ticks_max'}.items())):
        raise ValueError('Remaining roster or complete replay budget differs from design')
    return value


def remaining_jobs(records, design):
    all_seeds = list(range(design['source_train_seed_range'][0],design['source_train_seed_range'][1]+1,
                           design['pilot_seed_stride']))
    if ([r['simulator_seed'] for r in records] != all_seeds
            or any(r['engineering_stage'] != (r['simulator_seed'] in design['engineering_seeds'])
                   or r['execution_authorized'] is not False or r['task_level'] is not None for r in records)):
        raise ValueError('Need the unmodified full proposal, not selected outcomes')
    jobs = [{'id':r['proposed_task_id'],'simulator_seed':r['simulator_seed'],
             'source_episode_id':r['source_episode_id'],'source_job_id':r['source_job_id'],'split':'train'}
            for r in records if not r['engineering_stage']]
    if [j['simulator_seed'] for j in jobs] != PROTOCOL['simulator_seeds'] or len({j['id'] for j in jobs}) != 29:
        raise ValueError('Duplicate, replaced or previously completed task')
    return jobs


def engineering_evidence(hashes, deep=False):
    directory = ROOT/'artifacts/ic/ic02d_eng_v2'
    if {name:eng.sha(directory/name) for name in ENGINEERING_HASHES} != ENGINEERING_HASHES:
        raise ValueError('Completed engineering evidence bytes changed')
    q, report = eng.read(directory/'request.json'), eng.read(directory/'report.json')
    if (q['protocol'] != eng.PROTOCOL or q['git_commit_sha'] != '3856a78b97a87cf0cac9280161ce576f8d7f8056'
            or report['request_hash'] != digest(q) or report['status'] != 'complete'
            or report['engineering_gate'] is not True or report['actual_tasks'] != 3
            or report['training_ready'] is not False or report['predictors_loaded'] is not False):
        raise ValueError('Need the original complete three-task engineering request')
    # Adding this orchestration tool must not silently change scientific source/assets.
    for name,value in q['input_hashes'].items():
        if name in hashes and hashes[name] != value:
            raise ValueError('Shared engineering source/asset changed: '+name)
    for name,value in ENGINEERING_HASHES.items():
        hashes['worktree:artifacts/ic/ic02d_eng_v2/'+name] = value
    rows = []
    for job in q['jobs']:
        receipt = verify_episode(directory/job['id'],eng.binding(q,job))
        row = eng.checked_task(directory,q,job) if deep else receipt['metadata']
        rows.append(row)
        hashes['worktree:artifacts/ic/ic02d_eng_v2/'+job['id']+'/complete.json'] = eng.sha(directory/job['id']/'complete.json')
    if rows != report['tasks']:
        raise ValueError('Three-task receipts differ from engineering report')
    return report


def prerequisites(baseline, deep=False):
    _,design,_,hashes,metric,scales = eng.prerequisites(baseline)
    evidence = engineering_evidence(hashes,deep)
    if design != eng.read(ROOT/'artifacts/ic/ic02d_eng_v2/request.json')['design']:
        raise ValueError('Pilot scientific design differs from engineering')
    protocol = validate_protocol(eng.read(ROOT/CONFIG),design)
    jobs = remaining_jobs(eng.read((ROOT/PROTOCOL['design_report']).parent/'task_proposals.json')['records'],design)
    auth = eng.read(ROOT/PROTOCOL['authorization'])
    expected = {'protocol_id':protocol['version'],'run_id':protocol['run_id'],
        'simulator_seeds':protocol['simulator_seeds'],'maximum_tasks':29,
        'maximum_fresh_sumo_sessions':1102,'maximum_control_steps':118262,
        'maximum_raw_ticks':230666,'maximum_candidate_branches':870,
        'reuse_completed_engineering_tasks_without_rerun':True,
        'remaining_29_pilot_tasks_authorized':True,'predictor_inference_authorized':False,
        'predictor_training_authorized':False,'ddpg_training_authorized':False,
        'task_layer_assignment_authorized':False,'merge_main_authorized':False,
        'push_authorized':False,'old_baseline_access':'read_only','engineering_hashes':ENGINEERING_HASHES}
    if any(auth.get(k) != v for k,v in expected.items()):
        raise ValueError('Human pilot authorization scope changed')
    for name in (CONFIG,PROTOCOL['authorization'],'tools/collect_curriculum_pilot.py'):
        hashes['worktree:'+name] = eng.sha(ROOT/name)
    return protocol,design,jobs,hashes,metric,scales,evidence


def prepare(baseline):
    eng.require_clean(); baseline = Path(baseline).resolve()
    protocol,design,jobs,hashes,metric,scales,evidence = prerequisites(baseline,True)
    q = {'version':'ic02e-pilot-request-v1','git_commit_sha':eng.git(ROOT,'rev-parse','HEAD'),
        'working_tree_dirty':False,'launch_command':subprocess.list2cmdline(sys.orig_argv),'prepared_at':eng.now(),
        'baseline_root':str(baseline),'protocol':protocol,'design':design,'jobs':jobs,'input_hashes':hashes,
        'metric_config':metric,'normalization_scales':scales,'runtime':eng.runtime(),
        'engineering_request_hash':evidence['request_hash'],'human_approved_scope_only':True}
    output = ROOT/'artifacts/ic'/PROTOCOL['run_id']; output.mkdir(parents=True,exist_ok=False)
    write_once(output/'request.json',q)
    write_once(output/'preparation.json',{'status':'prepared_not_executed','request_hash':digest(q)})
    print('pilot_request='+str(output/'request.json')); print('confirm_request_hash='+digest(q))


def load(path, expected):
    eng.require_clean(); path = Path(path).resolve()
    if path != ROOT/'artifacts/ic'/PROTOCOL['run_id']/'request.json':
        raise ValueError('Only the approved named remaining-task request can execute')
    q = eng.read(path)
    if (q['version'] != 'ic02e-pilot-request-v1' or digest(q) != expected
            or eng.read(path.parent/'preparation.json')['request_hash'] != expected
            or q['git_commit_sha'] != eng.git(ROOT,'rev-parse','HEAD') or q['working_tree_dirty'] is not False
            or q['human_approved_scope_only'] is not True):
        raise ValueError('Request/hash/source mismatch')
    p,d,j,h,m,s,e = prerequisites(q['baseline_root'])
    if (q['protocol'] != p or q['design'] != d or q['jobs'] != j or q['input_hashes'] != h
            or q['metric_config'] != m or q['normalization_scales'] != s or q['runtime'] != eng.runtime()
            or q['engineering_request_hash'] != e['request_hash']):
        raise ValueError('Frozen pilot inputs changed; no automatic adaptation')
    return path,q


def worker(path, expected, job_id):
    path,q = load(path,expected)
    if (not (path.parent/'started.json').is_file() or (path.parent/'report.json').exists()
            or (path.parent/'failure.json').exists()):
        raise ValueError('Worker requires approved unfinished parent invocation')
    job = next(j for j in q['jobs'] if j['id'] == job_id)
    out = path.parent/job_id; eng.require_fresh_worker_directory(out,eng.binding(q,job))
    write_once(out/'worker_started.json',{'binding':eng.binding(q,job),'started_at':eng.now()})
    eng.torch.set_num_threads(PROTOCOL['torch_threads'])
    from prediction_rl.curriculum.task_collector import collect_task
    collect_task(ROOT,q['baseline_root'],q['design'],job,out,eng.binding(q,job),
                 q['metric_config'],q['normalization_scales'],lambda:load(path,expected))


def execute(path, expected, resume=False):
    path,q = load(path,expected); out = path.parent
    if (out/'report.json').exists() or (out/'failure.json').exists():
        raise ValueError('Immutable complete/failed report cannot be retried')
    report = {'status':'running','request_hash':expected,'git_commit_sha':q['git_commit_sha'],
        'started_at':eng.now(),'launch_command':subprocess.list2cmdline(sys.orig_argv),'tasks':[]}
    with run_lock(out):
        if not (out/'started.json').exists(): write_once(out/'started.json',report)
        elif not resume: raise ValueError('Interrupted invocation needs explicit COMPLETE-only reuse')
        error = None; before = eng.rng_fingerprint()
        try:
            for index,job in enumerate(q['jobs']):
                d = out/job['id']
                if d.exists():
                    if not resume: raise ValueError('Existing task cannot be overwritten')
                    verify_episode(d,eng.binding(q,job))
                else:
                    d.mkdir(); write_once(d/'started.json',{'binding':eng.binding(q,job),'started_at':eng.now()})
                    command = [sys.executable,'-B',str(Path(__file__).resolve()),'worker','--request',str(path),
                               '--confirm-request-hash',expected,'--job',job['id']]
                    write_once(d/'launch.json',{'command':command})
                    with (d/'worker.log').open('x',encoding='utf-8') as log:
                        subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
                row = eng.checked_task(out,q,job); report['tasks'].append(row)
                print(f"[ic02e] {index+1}/29 task={job['id']} controls={row['budget']['completed_control_steps']} complete",flush=True)
        except BaseException as caught:
            error = caught; report['traceback'] = traceback.format_exc()
        finally:
            post = {'parent_rng_unchanged':eng.rng_fingerprint()==before,'finished_at':eng.now(),
                'task_attempt_budgets':{j['id']:eng.read(out/j['id']/'postflight.json')['budget']
                    for j in q['jobs'] if (out/j['id']/'postflight.json').is_file()}}
            try:
                load(path,expected); engineering_evidence(dict(q['input_hashes']),True)
                post['immutable_input_postflight_passed'] = True
            except BaseException as caught:
                post['immutable_input_postflight_passed'] = False; post['failure_reason'] = str(caught)
                if error is None: error = caught
            write_once(out/'postflight.json',post)
        if error is None and not all(post[k] for k in ('parent_rng_unchanged','immutable_input_postflight_passed')):
            error = AssertionError('Parent RNG/inputs changed')
        if error is not None:
            report.update(status='failed',engineering_gate=False,failure_reason=str(error),finished_at=eng.now())
            write_once(out/'failure.json',report); raise error
        roots = [r for t in report['tasks'] for r in t['roots']]
        totals = {'actual_tasks':len(report['tasks']),
            'fresh_sumo_sessions':sum(t['budget']['fresh_sumo_sessions'] for t in report['tasks']),
            'actual_control_steps':sum(t['budget']['completed_control_steps'] for t in report['tasks']),
            'actual_raw_ticks':sum(t['budget']['completed_raw_ticks'] for t in report['tasks']),
            'candidate_branch_executions':sum(r.get('branch_executions',0) for r in roots)}
        if any(totals[k] > PROTOCOL[p] for k,p in {'actual_tasks':'tasks_max',
                'fresh_sumo_sessions':'fresh_sumo_sessions_max','actual_control_steps':'control_steps_max',
                'actual_raw_ticks':'raw_ticks_max','candidate_branch_executions':'candidate_branches_max'}.items()):
            report.update(status='failed',engineering_gate=False,failure_reason='Approved budget exceeded',finished_at=eng.now())
            write_once(out/'failure.json',report); raise ValueError('Approved pilot budget exceeded')
        report.update(totals,status='complete',engineering_gate=True,finished_at=eng.now(),parent_postflight=post,
            reached_roots=sum(r['status']=='complete' for r in roots),unavailable_roots=sum(r['status']=='unavailable' for r in roots),
            reused_engineering_tasks=3,reused_engineering_request_hash=q['engineering_request_hash'],
            complete_pilot_tasks=32,predictors_loaded=False,predictor_training_steps=0,ddpg_training_steps=0,
            learner_instantiated=False,test_opened=False,task_classifier=None,target_distribution_pi=None,
            training_ready=False,remaining_gate='review_full_task_coverage_then_freeze_task_rules_and_target_distribution')
        write_once(out/'report.json',report); print('pilot_report='+str(out/'report.json'))


def main():
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest='command',required=True)
    p = sub.add_parser('prepare'); p.add_argument('--baseline-root',required=True)
    for name in ('run','worker'):
        p = sub.add_parser(name); p.add_argument('--request',required=True); p.add_argument('--confirm-request-hash',required=True)
        if name == 'worker': p.add_argument('--job',required=True)
        else: p.add_argument('--resume',action='store_true')
    args = parser.parse_args()
    if args.command == 'prepare': prepare(args.baseline_root)
    elif args.command == 'worker': worker(args.request,args.confirm_request_hash,args.job)
    else: execute(args.request,args.confirm_request_hash,args.resume)


if __name__ == '__main__': main()
