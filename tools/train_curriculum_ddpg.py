"""IC04 user-run matched 120k training and frozen Gym20 development evaluation."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback
import uuid

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
import numpy as np
import torch
import smoke_curriculum_entry as entry
import smoke_remaining_curriculum as accepted
from curriculum_matched_contract import (PROTOCOL,SCOPE,validate_protocol,jobs,freeze_manifest,
    manifest_roster,aggregate,snapshot_steps)
from prediction_rl.curriculum.entry_smoke import json_safe,rng_fingerprint
from prediction_rl.curriculum.train_reader import inside
from prediction_rl.data.collection_store import (write_once,file_hash as sha,verify_episode,
    seal_episode,run_lock,inventory)
from prediction_rl.data.dataset_contract import digest
from prediction_rl.training.all_ddpg import resolved_preset

CONFIG='configs/development/ic04_matched_120k_v1.json'
EVIDENCE_PINS={
    'request.json':'97dd8cb5c216020be6403a8a7c0f73e5ccd98c8399f6ebd2405d8cbd2da6c146',
    'report.json':'6074c3b8bf3d6ab795d86638d7d9c3e31bdadcc209fd9e5d896e6a60b0806f7b',
    'coverage_review.json':'35c9edf1968e253adc0b1bb3514bbc6cf2035705e07ccf857cdbb85d4669b755'}


def source_inventory():
    return {'worktree:'+p.relative_to(ROOT).as_posix():sha(p)
        for directory in ('src/prediction_rl','tools') for p in sorted((ROOT/directory).rglob('*.py'))}


def prerequisites(baseline):
    """Old gates run unchanged; new code is ONLY versioned orchestration in tools/."""
    _,h,models,trained,tasks,spaces,legacy=accepted.prerequisites(baseline)
    root=ROOT/'artifacts/ic/ic03b_remaining_v1'
    if {n:sha(root/n) for n in EVIDENCE_PINS}!=EVIDENCE_PINS:
        raise ValueError('Accepted IC03b evidence bytes changed')
    q=entry.read(root/'request.json'); report=entry.read(root/'report.json'); coverage=entry.read(root/'coverage_review.json')
    if (digest(q)!='3f5f910adbda5ed1242e19d7630fe85d03a3d2e1d507f242f076bb95e9d98ec4'
            or q['runtime']!=entry.runtime()
            or report['status']!='complete' or report['engineering_gate'] is not True
            or coverage['all_four_paths_have_native_integration_evidence'] is not True
            or coverage['legacy_request_remains_failed'] is not True
            or coverage['legacy_receipt_remains_invalid'] is not True
            or coverage['combined_consumption']!={'sumo_sessions':8,'control_steps':1282,
                'raw_ticks':2090,'feature_calls':1282,'optimizer_updates':0}):
        raise ValueError('Four-path native smoke coverage is incomplete')
    for key,value in q['input_hashes'].items():
        location,relative=key.split(':',1)
        base=ROOT if location=='worktree' else Path(baseline) if location=='baseline' else None
        if base is None or sha(inside(base,relative))!=value:
            raise ValueError('Accepted smoke source/artifact changed: '+key)
        h[key]=value
    for job,row in zip(q['jobs'],report['jobs']):
        receipt=verify_episode(root/job['id'],{'request_hash':digest(q),'job':job})
        if receipt['metadata']!=row: raise ValueError('Native smoke complete receipt changed')
        h['worktree:'+str((root/job['id']/'complete.json').relative_to(ROOT)).replace('\\','/')]=sha(root/job['id']/'complete.json')
        for name,value in inventory(root/job['id']).items():
            h['worktree:'+str((root/job['id']/name).relative_to(ROOT)).replace('\\','/')]=value
    for name,value in EVIDENCE_PINS.items(): h['worktree:artifacts/ic/ic03b_remaining_v1/'+name]=value
    pack=entry.read(ROOT/'artifacts/ic/ic02h_intensity_v1/task_proposals.json')
    oldreport=entry.read(ROOT/'artifacts/ic/ic02h_intensity_v1/report.json')
    manifest=freeze_manifest(pack,oldreport,accepted.original.SEALED['task_proposals.json'])
    if manifest_roster(manifest)!=tasks: raise ValueError('New live manifest changed approved task roster')
    h.update(source_inventory()); h['worktree:'+CONFIG]=sha(ROOT/CONFIG)
    return h,models,trained,manifest,spaces,legacy


def prepare(baseline):
    entry.require_clean(); torch.set_num_threads(1)
    p=validate_protocol(entry.read(ROOT/CONFIG))
    before=rng_fingerprint()
    h,models,trained,manifest,spaces,legacy=prerequisites(baseline)
    if before!=rng_fingerprint(): raise ValueError('Preparation consumed training RNG')
    out=ROOT/'artifacts/ic'/p['run_id']; out.mkdir(parents=True,exist_ok=False)
    write_once(out/'task_manifest.json',manifest)
    q={'version':SCOPE,'protocol':deepcopy(p),'baseline_root':str(Path(baseline).resolve()),
        'git_commit_sha':entry.git(ROOT,'rev-parse','HEAD'),'working_tree_dirty':False,
        'prepared_at':entry.now(),'launch_command':subprocess.list2cmdline(sys.orig_argv),
        'input_hashes':h,'source_inventory':source_inventory(),'runtime':entry.runtime(),
        'predictors':models,'predictor_training_request':trained,'task_manifest':manifest,
        'task_manifest_hash':digest(manifest),'task_manifest_file_sha256':sha(out/'task_manifest.json'),
        'native_spaces':spaces,'resolved_ddpg_preset':resolved_preset('cpu',.0002,smoke=False),
        'upstream_config':entry.PROTOCOL['upstream_config'],
        'upstream_config_sha256':sha(ROOT/entry.PROTOCOL['upstream_config']),
        'train_jobs':jobs(p,'train'),'evaluation_jobs':jobs(p,'evaluate'),
        'prior_failed_smoke_preserved':legacy['legacy_request_remains_failed'],
        'prior_policies_loaded_for_training':False,'full_training_started':False,
        'execution_authorization':'human_invokes_run_with_exact_request_digest_not_agent_auto_execution'}
    write_once(out/'request.json',q)
    write_once(out/'preparation.json',{'request_hash':digest(q),'status':'prepared_not_trained',
        'training_instances':12,'requested_total_control_steps':1440000,'evaluation_episodes':960,
        'unique_evaluation_traffic_scenarios':20,'full_training_started':False,
        'automatic_budget_extension':False,'independent_test':False})
    print('curriculum_request='+str(out/'request.json'),flush=True)
    print('confirm_request_hash='+digest(q),flush=True)
    return out/'request.json',digest(q)


def load(path,expected):
    entry.require_clean(); torch.set_num_threads(1)
    path=Path(path).resolve()
    if path!=ROOT/'artifacts/ic'/PROTOCOL['run_id']/'request.json':
        raise ValueError('One frozen request only; no renamed retry or output escape')
    q=entry.read(path); p=validate_protocol(q['protocol'])
    if (q['version']!=SCOPE or digest(q)!=expected
            or entry.read(path.parent/'preparation.json')['request_hash']!=expected
            or q['git_commit_sha']!=entry.git(ROOT,'rev-parse','HEAD') or q['working_tree_dirty'] is not False
            or q['source_inventory']!=source_inventory() or q['runtime']!=entry.runtime()
            or q['resolved_ddpg_preset']!=resolved_preset('cpu',.0002,smoke=False)
            or q['train_jobs']!=jobs(p,'train') or q['evaluation_jobs']!=jobs(p,'evaluate')
            or entry.read(ROOT/CONFIG)!=p or q['prior_policies_loaded_for_training'] is not False
            or digest(q['task_manifest'])!=q['task_manifest_hash']
            or sha(path.parent/'task_manifest.json')!=q['task_manifest_file_sha256']
            or entry.read(path.parent/'task_manifest.json')!=q['task_manifest']):
        raise ValueError('Frozen source, clean commit, runtime, config, budget or manifest changed')
    manifest_roster(q['task_manifest'])
    baseline=Path(q['baseline_root']).resolve()
    if (entry.git(baseline,'rev-parse','HEAD')!='75051acc06f6ea2ab6dda9be678f191b63d04569'
            or entry.git(baseline,'status','--porcelain')):
        raise ValueError('Read-only validated baseline changed')
    for key,value in q['input_hashes'].items():
        location,relative=key.split(':',1)
        base=ROOT if location=='worktree' else baseline if location=='baseline' else None
        if base is None or sha(inside(base,relative))!=value:
            raise ValueError('Immutable run input changed: '+key)
    return path,q


def binding(q,stage,job): return {'request_hash':digest(q),'stage':stage,'job':job}


def training_metadata(q,root):
    trained={j['id']:verify_episode(root/'train'/j['id'],binding(q,'train',j))['metadata'] for j in q['train_jobs']}
    for j in q['train_jobs']:
        r=trained[j['id']]
        if (r['job']!=j or r['status']!='complete' or r['from_scratch'] is not True
                or r['predictor_frozen'] is not True or r['actual_steps']<120000 or r['actual_steps']>120499
                or [s['requested_steps'] for s in r['snapshots']]!=[20000,40000,60000,120000]
                or r['replay_audit']['original_replay_size']!=r['actual_steps']
                or r['replay_audit']['actual_control_steps']!=r['actual_steps']
                or any(n!=r['original_minibatches'] for n in r['optimizer_loss_counts'].values())):
            raise ValueError('Incomplete matched training or original replay/update audit')
        for s in r['snapshots']:
            snapshot_steps(s['actual_steps'],s['requested_steps']); d=root/'train'/j['id']
            if (s['weights_file']!=f'n{s["requested_steps"]}.pt' or sha(d/s['weights_file'])!=s['weights_sha256']
                    or entry.read(d/f'n{s["requested_steps"]}.json')!=s):
                raise ValueError('Frozen snapshot changed')
    for seed in [0,1,2]:
        initial=[trained[f'{a}_s{seed}']['initial_weight_sha256'] for a in PROTOCOL['arms']]
        if any(x!=initial[0] for x in initial[1:]): raise ValueError('Four-arm matched initialization changed')
    return trained


def worker(args):
    path,q=load(args.request,args.confirm_request_hash)
    plan=q['train_jobs' if args.stage=='train' else 'evaluation_jobs']
    job=next(j for j in plan if j['id']==args.job)
    out=path.parent/args.stage/job['id']
    if not out.is_dir() or {p.name for p in out.iterdir()}!={'started.json','launch.json','worker.log'}:
        raise ValueError('Worker needs its parent-owned unstarted job; no direct retry')
    if entry.read(out/'started.json')['binding']!=binding(q,args.stage,job):
        raise ValueError('Worker parent binding differs')
    write_once(out/'worker_claim.json',{'binding':binding(q,args.stage,job),'started_at':entry.now()})
    random.seed(job['run_seed']); np.random.seed(job['run_seed']); torch.manual_seed(job['run_seed'])
    begin=time.perf_counter()
    try:
        import curriculum_matched_runtime as runtime
        live={**q,'request_hash':digest(q)}
        if args.stage=='train': result=runtime.train(live,job,out,ROOT/'RL-MPC-LaneMerging-master')
        else:
            tj=next(j for j in q['train_jobs'] if j['id']==job['training_id']); td=path.parent/'train'/tj['id']
            trained=verify_episode(td,binding(q,'train',tj))['metadata']
            snapshot=next(s for s in trained['snapshots'] if s['requested_steps']==job['checkpoint_frames'])
            result=runtime.evaluate(live,job,out,ROOT/'RL-MPC-LaneMerging-master',snapshot,td/snapshot['weights_file'])
        load(path,args.confirm_request_hash)
        result.update(elapsed_s=time.perf_counter()-begin,finished_at=entry.now(),
            request_hash=digest(q),git_commit_sha=q['git_commit_sha'],working_tree_dirty=False,
            source_postflight_passed=True,launch_command=subprocess.list2cmdline(sys.orig_argv))
        write_once(out/'report.json',json_safe(result))
        print('curriculum_job_complete='+job['id'],flush=True)
        # Parent seals only AFTER process exit, finally cleanup and log closure.
    except BaseException:
        write_once(out/'failure.json',{'status':'failed','binding':binding(q,args.stage,job),
            'failure_reason':traceback.format_exc(),'finished_at':entry.now()}); raise


def execute(path,expected,stage,resume=False):
    path,q=load(path,expected); root=path.parent
    if stage not in ('train','evaluate'): raise ValueError('Unknown stage')
    with run_lock(root):
        invocation=root/'invocations'/uuid.uuid4().hex[:10]; invocation.mkdir(parents=True,exist_ok=False)
        result={'request_hash':digest(q),'stage':stage,'started_at':entry.now(),'jobs':[],
            'launch_command':subprocess.list2cmdline(sys.orig_argv)}
        try:
            if stage=='evaluate': training_metadata(q,root)
            pending=[]
            for job in q['train_jobs' if stage=='train' else 'evaluation_jobs']:
                directory=root/stage/job['id']
                if directory.exists():
                    if not resume: raise ValueError('Use --resume only for sealed COMPLETE whole jobs')
                    verify_episode(directory,binding(q,stage,job))
                    result['jobs'].append({'id':job['id'],'action':'reuse'})
                else: pending.append(job)
            def child(job):
                directory=root/stage/job['id']; directory.mkdir(parents=True,exist_ok=False)
                write_once(directory/'started.json',{'binding':binding(q,stage,job),'started_at':entry.now()})
                command=[sys.executable,'-B',str(Path(__file__).resolve()),'worker','--request',str(path),
                    '--confirm-request-hash',digest(q),'--stage',stage,'--job',job['id']]
                write_once(directory/'launch.json',{'command':command})
                print('[ic04] '+stage+' '+job['id'],flush=True)
                with (directory/'worker.log').open('x',encoding='utf-8') as log:
                    subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
                report=entry.read(directory/'report.json')
                seal_episode(directory,binding(q,stage,job),report)
                verify_episode(directory,binding(q,stage,job))
                return {'id':job['id'],'action':'run'}
            width=q['protocol']['training' if stage=='train' else 'evaluation']['workers']
            with ThreadPoolExecutor(max_workers=width) as pool:
                for i in range(0,len(pending),width):
                    futures=[pool.submit(child,j) for j in pending[i:i+width]]
                    result['jobs'].extend(f.result() for f in futures)
            if stage=='train': training_metadata(q,root)
            load(path,expected)
            result.update(status='complete',finished_at=entry.now())
        except BaseException:
            result.update(status='failed',failure_reason=traceback.format_exc(),finished_at=entry.now()); raise
        finally: write_once(invocation/'report.json',result)
    print('stage_report='+str(invocation/'report.json'),flush=True)
    if stage=='evaluate': aggregate_run(path,expected)


def aggregate_run(path,expected):
    path,q=load(path,expected); root=path.parent
    with run_lock(root):
        trained=training_metadata(q,root); rows=[]
        for job in q['evaluation_jobs']:
            d=root/'evaluate'/job['id']; result=verify_episode(d,binding(q,'evaluate',job))['metadata']
            r=entry.read(d/'episode.json')
            snapshot=next(s for s in trained[job['training_id']]['snapshots'] if s['requested_steps']==job['checkpoint_frames'])
            if r!=result['outcome'] or r['policy_sha256']!=snapshot['weights_sha256']:
                raise ValueError('Evaluation/frozen policy lineage mismatch')
            rows.append(r)
        result={**aggregate(q['protocol'],rows),'request_hash':digest(q),'training':trained,
            'git_commit_sha':q['git_commit_sha'],'finished_at':entry.now()}
        write_once(root/'development_curves.json',{'label':'frozen_Gym20_development_NOT_test_NOT_training_returns',
            'unique_scenarios':20,'nodes':result['nodes']})
        write_once(root/'aggregate.json',result)
        write_once(root/'aggregate_receipt.json',{'request_hash':digest(q),'aggregate_sha256':sha(root/'aggregate.json'),
            'development_curves_sha256':sha(root/'development_curves.json')})
    print('curriculum_aggregate='+str(root/'aggregate.json'),flush=True)


def status(path,expected):
    path,q=load(path,expected)
    for stage in ('train','evaluate'):
        counts={'complete':0,'unstarted':0,'incomplete_or_invalid':0}
        for job in q['train_jobs' if stage=='train' else 'evaluation_jobs']:
            d=path.parent/stage/job['id']
            if not d.exists(): counts['unstarted']+=1
            else:
                try: verify_episode(d,binding(q,stage,job)); counts['complete']+=1
                except ValueError: counts['incomplete_or_invalid']+=1
        print(stage+'='+str(counts),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__); commands=parser.add_subparsers(dest='command',required=True)
    p=commands.add_parser('prepare'); p.add_argument('--baseline-root',required=True)
    for name in ('run','worker','aggregate','status'):
        p=commands.add_parser(name); p.add_argument('--request',required=True); p.add_argument('--confirm-request-hash',required=True)
        if name in ('run','worker'): p.add_argument('--stage',choices=['train','evaluate'],required=True)
        if name=='run': p.add_argument('--resume',action='store_true')
        if name=='worker': p.add_argument('--job',required=True)
    args=parser.parse_args()
    if args.command=='prepare': prepare(args.baseline_root)
    elif args.command=='worker': worker(args)
    elif args.command=='run': execute(args.request,args.confirm_request_hash,args.stage,args.resume)
    elif args.command=='aggregate': aggregate_run(args.request,args.confirm_request_hash)
    else: status(args.request,args.confirm_request_hash)


if __name__=='__main__': main()
