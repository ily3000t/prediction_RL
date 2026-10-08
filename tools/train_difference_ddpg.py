"""User-run new difference B3 only; reuse sealed P7n B0--B3 evidence read-only."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
import subprocess
import sys
import time
import traceback
import uuid

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'tools')]
import train_response_predictors as shared
import connect_candidate_difference as connection
import train_matched_ddpg as kernel
import train_fixed_budget_ddpg as fixed
import explore_feature_ddpg as original
import smoke_feature_ddpg as smoke
from prediction_rl.data.collection_store import read_json as read, file_hash as sha, write_once, run_lock, seal_episode, verify_episode
from prediction_rl.data.dataset_contract import digest
from prediction_rl.data.frozen_dataset import inside, verify_historical_inputs, verify_historical_text_checkout
from prediction_rl.evaluation.difference_closed_loop import ARM, PROTOCOL, SMOKE, REFERENCE, validate_protocol, jobs, aggregate
from prediction_rl.evaluation.fixed_budget_extension import jobs as reference_jobs
from prediction_rl.evaluation.matched_learning_curve import snapshot_steps
from prediction_rl.training.all_ddpg import resolved_preset

REFERENCE_REQUEST = 'artifacts/p7n/p7n_120k_v1/request.json'
REFERENCE_HASH = '31301c64cf15606f8fc05a0280b5f7f75fd1c6e33b5cd877087284e59f31befa'
REFERENCE_AGGREGATE = 'e2dd769d3943df09db5b2d46991c739cb92604cbbc5277de749e1383b7318c79'
CONFIG = 'configs/development/p07_candidate_difference_120k_v1.json'
VERSION = 'candidate_difference_ddpg_request_v1'


def checked_upstream_checkout(reference, q):
    """Accept only Git LF/CRLF spelling, never a reward/configuration change."""
    relative = original.UPSTREAM.relative_to(ROOT).as_posix()
    for repo in (reference, ROOT):
        verify_historical_text_checkout(repo, relative, q['upstream_config_sha256'], q['git_commit'])
    return {'contract':'historical_git_lf_crlf_only_v1', 'path':relative,
            'historical_commit':q['git_commit'], 'reference_byte_sha256':q['upstream_config_sha256'],
            'current_byte_sha256':sha(original.UPSTREAM)}


def checked_reference(reference, *, full):
    path = inside(reference, REFERENCE_REQUEST); root=path.parent
    q,a,receipt = read(path),read(root/'aggregate.json'),read(root/'aggregate_receipt.json')
    if (digest(q) != REFERENCE_HASH or q['protocol'] != REFERENCE
            or sha(root/'aggregate.json') != REFERENCE_AGGREGATE
            or receipt['aggregate_sha256'] != REFERENCE_AGGREGATE or receipt['request_hash'] != REFERENCE_HASH
            or a['request_hash'] != REFERENCE_HASH or a['status'] != 'complete'
            or a['engineering_complete'] is not True or a['primary_node'] != 120000
            or a['method_effect_gate'] is not None or a['checkpoint_selection_by_development_results'] is not False
            or (root/'writer.lock').exists() or q['environment'] != shared.env()
            or q['dependencies'] != smoke.dependencies() or q['runtime_identity'] != original.runtime_identity()
            or q['upstream_hashes'] != original.upstream_hashes()
            or read(root/'preparation.json')['request_hash'] != REFERENCE_HASH):
        raise ValueError('P7n frozen reference/runtime changed; no fallback or relabeling')
    checked_upstream_checkout(reference, q)
    if full:
        verify_historical_inputs(reference, {'python_lf:'+k:v for k,v in q['source_hashes'].items()}, q['git_commit'])
    files={str((root/n).relative_to(reference)):sha(root/n) for n in
           ('request.json','preparation.json','aggregate.json','aggregate_receipt.json')}
    initial={str(s):a['training'][f'conditional_s{s}']['initial_weight_sha256'] for s in REFERENCE['training']['run_seeds']}
    if not full: return q,a,files,initial,None
    # Verify only evidence actually consumed: source lineage, selected policies,
    # sealed metadata and episode outcomes. Bulky unused replay/loss files are not
    # rescanned in every new worker, and raw historical results are never copied.
    trained=a['training']
    for j in reference_jobs(REFERENCE,'train'):
        d=root/'train'/j['id']; r=read(d/'complete.json'); m=trained[j['id']]
        if (r['status'] != 'complete' or r['binding'] != kernel.binding(q,'train',j)
                or r['metadata'] != m or m['job'] != j or m['from_scratch'] is not True
                or m['predictor_frozen'] is not True or read(d/'report.json') != m):
            raise ValueError('Reused policy receipt/lineage changed')
        paths=['report.json','resolved_config.json']
        for snap in m['snapshots']:
            n=snap['requested_steps']; snapshot_steps(snap['actual_steps'],n)
            if (snap['weights_file'] != f'n{n}.pt' or sha(d/f'n{n}.pt') != snap['weights_sha256']
                    or read(d/f'n{n}.json') != snap): raise ValueError('Reused selected policy changed')
            paths.extend([f'n{n}.pt',f'n{n}.json'])
        for name in paths:
            if r['files'].get(name) != sha(d/name): raise ValueError('Reused policy inventory changed')
    for s in REFERENCE['training']['run_seeds']:
        if any(trained[f'{arm}_s{s}']['initial_weight_sha256'] != initial[str(s)] for arm in ('zero','ordinary')):
            raise ValueError('Frozen expanded initialization mismatch')
    rows=[]
    for j in reference_jobs(REFERENCE,'evaluate'):
        d=root/'evaluate'/j['id']; r=read(d/'complete.json'); outcome=read(d/'episode.json')
        if (r['status'] != 'complete' or r['binding'] != kernel.binding(q,'evaluate',j)
                or r['metadata']['job'] != j or r['metadata']['outcome'] != outcome
                or r['metadata']['no_policy_updates'] is not True or read(d/'report.json') != r['metadata']
                or any(r['files'].get(n) != sha(d/n) for n in ('episode.json','report.json'))):
            raise ValueError('Reused evaluation receipt/outcome changed')
        snap=next(x for x in trained[j['training_id']]['snapshots'] if x['requested_steps']==j['checkpoint_frames'])
        if outcome['policy_sha256'] != snap['weights_sha256']: raise ValueError('Reused episode checkpoint mismatch')
        rows.append(outcome)
    return q,a,files,initial,rows


class Execution:
    """No global monkey-patching; only predictor binding and method label adapt."""
    def __init__(self,q):
        self.q=q; self.source=original.SOURCE; self.upstream=original.UPSTREAM
        self.smoke=q['protocol']==SMOKE
    def provider_for(self,q,arm):
        if q is not self.q or arm != ARM: raise ValueError('Difference execution context mismatch')
        return connection.provider(q['difference_request'],Path(q['reference_root']),q['predictor']),q['predictor']['sha256']
    def environment(self,q,job,provider,recorder):
        if q is not self.q or job['arm'] != ARM: raise ValueError('Difference environment identity mismatch')
        # Same conditional observation contract; checkpoint's predictor hash makes
        # old and new policy weights non-interchangeable despite equal dimensions.
        return original.environment(q,{**job,'arm':'conditional'},provider,recorder)


def prepare(calibration_request,run_id,*,audit=False,engineering_audit=None):
    shared.clean(); torch.set_num_threads(1)
    cp=inside(ROOT,CONFIG)
    if read(cp) != PROTOCOL: raise ValueError('Only the frozen standalone 120k config is supported')
    p,reference,record,inputs=connection.accepted_calibration(calibration_request)
    if not audit:
        if engineering_audit is None: raise ValueError('Completed bounded workflow audit required before full training preparation')
        ap=inside(ROOT,engineering_audit); ar=read(ap); aq=read(ap.parent/'request.json')
        _,checked=load_request(ap.parent/'request.json',digest(aq))
        if (checked['protocol'] != SMOKE or ar['status'] != 'complete' or ar['engineering_gate'] is not True
                or ar['request_hash'] != digest(aq) or ar['formal_training_started'] is not False
                or aq['predictor'] != record or aq['difference_request'] != p):
            raise ValueError('Workflow audit does not bind this difference predictor/runtime')
        training_metadata(aq,ap.parent)
        for j in aq['evaluation_jobs']: verify_episode(ap.parent/'evaluate'/j['id'],kernel.binding(aq,'evaluate',j))
        for v in (ap,ap.parent/'request.json',ap.parent/'preparation.json'):
            inputs[str(v.relative_to(ROOT))]=sha(v)
    previous,a,reference_files,initial,rows=checked_reference(reference,full=True)
    if (previous['predictors'] != p['baseline_ensembles']
            or digest(previous['predictor_training_request']) != p['baseline_request_hash']):
        raise ValueError('Predictor baseline does not match reused P7n policies')
    protocol=deepcopy(SMOKE if audit else PROTOCOL)
    preset=resolved_preset('cpu',protocol['training']['learning_rate'],smoke=audit)
    if not audit and preset != previous['preset']: raise ValueError('Matched DDPG preset changed')
    if preset['replay_buffer_size'] <= protocol['training']['requested_frames']+499 or preset['last_frame'] != 2e6:
        raise ValueError('Replay must not wrap; original schedule remains 2M')
    q={**shared.metadata(),'version':VERSION,'protocol':protocol,'config_path':CONFIG,'config_sha256':sha(cp),
       'calibration_request':str(inside(ROOT,calibration_request).relative_to(ROOT)),
       'input_hashes':inputs,'reference_root':str(reference),'reference_files':reference_files,
       'reference_rows_hash':digest(rows),'reference_initial_weights':initial,
       'difference_request':p,'predictor':record,'source_hashes':shared.source_hashes(),
       'dependencies':smoke.dependencies(),'environment':shared.env(),
       'runtime_identity':original.runtime_identity(),'upstream_hashes':original.upstream_hashes(),
       'upstream_config_sha256':sha(original.UPSTREAM),
       'upstream_config_provenance':checked_upstream_checkout(reference,previous),'preset':preset,
       'train_jobs':jobs(protocol,'train'),'evaluation_jobs':jobs(protocol,'evaluate'),
       'previous_policy_weights_used_for_training':False,'policy_features_use_bands':False,
       'author50_results_mixed':False,'test_opened':False}
    out=shared.output_dir('difference_ddpg',run_id); write_once(out/'request.json',q)
    write_once(out/'preparation.json',{'request_hash':digest(q),'status':'prepared_not_trained',
        'new_training_instances':len(q['train_jobs']),
        'new_requested_training_steps':len(q['train_jobs'])*protocol['training']['requested_frames'],
        'new_evaluation_episodes':len(q['evaluation_jobs']),'reused_evaluation_episodes':len(rows),
        'unique_development_scenarios':len(protocol['evaluation']['simulator_seeds']),
        'full_training_started':False,'smoke_only':audit,'automatic_budget_extension':False})
    print('difference_ddpg_request='+str(out/'request.json'),flush=True)
    print('confirm_request_hash='+digest(q),flush=True)
    return out/'request.json',digest(q)


def load_request(path,expected_hash):
    shared.clean(); torch.set_num_threads(1); path=inside(ROOT,path); q=read(path)
    p=validate_protocol(q['protocol']); preset=resolved_preset('cpu',p['training']['learning_rate'],smoke=p==SMOKE)
    if (q['version'] != VERSION or digest(q) != expected_hash
            or read(path.parent/'preparation.json')['request_hash'] != expected_hash
            or q['source_hashes'] != shared.source_hashes() or q['environment'] != shared.env()
            or q['dependencies'] != smoke.dependencies() or q['runtime_identity'] != original.runtime_identity()
            or q['upstream_hashes'] != original.upstream_hashes() or q['upstream_config_sha256'] != sha(original.UPSTREAM)
            or q['preset'] != preset or read(inside(ROOT,q['config_path'])) != PROTOCOL
            or sha(inside(ROOT,q['config_path'])) != q['config_sha256']
            or q['train_jobs'] != jobs(p,'train') or q['evaluation_jobs'] != jobs(p,'evaluate')
            or q['previous_policy_weights_used_for_training'] is not False or q['policy_features_use_bands'] is not False
            or q['author50_results_mixed'] is not False or q['test_opened'] is not False):
        raise ValueError('Frozen difference DDPG source/config/runtime/roster changed')
    for name,h in q['input_hashes'].items():
        if sha(inside(ROOT,name)) != h: raise ValueError('Frozen difference asset changed')
    reference=Path(q['reference_root']).resolve()
    for name,h in q['reference_files'].items():
        if sha(inside(reference,name)) != h: raise ValueError('Frozen reference evidence changed')
    if q['upstream_config_provenance'] != checked_upstream_checkout(reference,read(inside(reference,REFERENCE_REQUEST))):
        raise ValueError('Historical upstream checkout provenance changed')
    cp=read(inside(ROOT,q['calibration_request']))
    if q['predictor'] != cp['checkpoint'] or q['difference_request'] != read(inside(ROOT,cp['training_request'])):
        raise ValueError('Predictor/calibration request mismatch')
    return path,q


def training_metadata(q,root):
    result={}
    for j in q['train_jobs']:
        d=root/'train'/j['id']; m=verify_episode(d,kernel.binding(q,'train',j))['metadata']
        if (m['job'] != j or m['from_scratch'] is not True or m['predictor_frozen'] is not True
                or m['initial_weight_sha256'] != q['reference_initial_weights'][str(j['run_seed'])]
                or [r['requested_steps'] for r in m['snapshots']] != q['protocol']['training']['checkpoint_frames']):
            raise ValueError('Matched from-scratch initialization/training changed')
        for r in m['snapshots']:
            snapshot_steps(r['actual_steps'],r['requested_steps'])
            if (r['weights_contract']['predictor_sha256'] != q['predictor']['sha256']
                    or sha(d/r['weights_file']) != r['weights_sha256']):
                raise ValueError('Difference policy checkpoint binding changed')
        result[j['id']]=m
    return result


def worker(args):
    path,q=load_request(args.request,args.confirm_request_hash)
    plan=q['train_jobs' if args.stage=='train' else 'evaluation_jobs']; j=next(j for j in plan if j['id']==args.job)
    out=path.parent/args.stage/j['id']; out.mkdir(parents=True,exist_ok=False)
    start={**shared.metadata(),'binding':kernel.binding(q,args.stage,j)}; write_once(out/'started.json',start)
    begin=time.perf_counter()
    try:
        execution=Execution(q)
        r=(kernel.train_job(q,j,out,execution=execution) if args.stage=='train'
           else kernel.evaluation_job(q,j,out,path.parent,execution=execution))
        if args.stage=='train' and r['initial_weight_sha256'] != q['reference_initial_weights'][str(j['run_seed'])]:
            raise ValueError('New expanded policy initialization differs from frozen P7n')
        r.update(provenance=start,elapsed_s=time.perf_counter()-begin,finished_at=shared.now())
        write_once(out/'report.json',r); seal_episode(out,kernel.binding(q,args.stage,j),r)
    except BaseException:
        write_once(out/'failure.json',{'status':'failed','failure_reason':traceback.format_exc()}); raise


def run(path,expected_hash,stage,resume=False):
    path,q=load_request(path,expected_hash); root=path.parent
    with run_lock(root):
        invocation=root/'invocations'/uuid.uuid4().hex[:10]; invocation.mkdir(parents=True)
        report={**shared.metadata(),'stage':stage,'request_hash':digest(q),'jobs':[]}
        try:
            if stage=='evaluate': training_metadata(q,root)
            pending=[]
            for j in q['train_jobs' if stage=='train' else 'evaluation_jobs']:
                if (root/stage/j['id']).exists():
                    if not resume: raise ValueError('Existing job; --resume only reuses sealed COMPLETE jobs')
                    verify_episode(root/stage/j['id'],kernel.binding(q,stage,j))
                    report['jobs'].append({'id':j['id'],'action':'reuse'})
                else: pending.append(j)
            def child(j):
                command=[sys.executable,'-B',str(Path(__file__).resolve()),'worker','--request',str(path),
                         '--confirm-request-hash',digest(q),'--stage',stage,'--job',j['id']]
                print(f'[difference_ddpg] {stage} {j["id"]}',flush=True)
                with (invocation/(j['id']+'.log')).open('x',encoding='utf-8') as log:
                    subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
                verify_episode(root/stage/j['id'],kernel.binding(q,stage,j))
                return {'id':j['id'],'action':'run'}
            width=q['protocol']['training' if stage=='train' else 'evaluation']['workers']
            with ThreadPoolExecutor(max_workers=width) as pool:
                for i in range(0,len(pending),width): report['jobs'].extend(f.result() for f in [pool.submit(child,j) for j in pending[i:i+width]])
            if stage=='train': training_metadata(q,root)
            report.update(status='complete',finished_at=shared.now())
        except BaseException:
            report.update(status='failed',failure_reason=traceback.format_exc()); raise
        finally: write_once(invocation/'report.json',report)
    print('difference_ddpg_stage_report='+str(invocation/'report.json'),flush=True)


def aggregate_run(path,expected_hash):
    path,q=load_request(path,expected_hash); root=path.parent
    if q['protocol']==SMOKE: raise ValueError('Smoke is not a learning-effect report')
    with run_lock(root):
        trained=training_metadata(q,root); rows=[]
        for j in q['evaluation_jobs']:
            d=root/'evaluate'/j['id']; m=verify_episode(d,kernel.binding(q,'evaluate',j))['metadata']; r=read(d/'episode.json')
            snap=next(s for s in trained[j['training_id']]['snapshots'] if s['requested_steps']==j['checkpoint_frames'])
            if r != m['outcome'] or r['policy_sha256'] != snap['weights_sha256']: raise ValueError('Evaluation policy changed')
            rows.append(r)
        _,_,_,_,reference_rows=checked_reference(Path(q['reference_root']),full=True)
        if digest(reference_rows) != q['reference_rows_hash']: raise ValueError('Reused episode rows changed')
        result={**aggregate(q['protocol'],rows,reference_rows),'request_hash':digest(q),'provenance':shared.metadata(),
            'training':trained,'predictor_sha256':q['predictor']['sha256'],'finished_at':shared.now()}
        write_once(root/'training_curves.json',fixed.curve_index(root,trained,q))
        write_once(root/'development_curves.json',{'label':'frozen_policy_Gym20_development_NOT_test',
            'unique_scenarios':20,'nodes':result['nodes']})
        write_once(root/'aggregate.json',result)
        write_once(root/'aggregate_receipt.json',{'request_hash':digest(q),'aggregate_sha256':sha(root/'aggregate.json'),
            'training_curves_sha256':sha(root/'training_curves.json'),'development_curves_sha256':sha(root/'development_curves.json')})
    print('difference_ddpg_aggregate='+str(root/'aggregate.json'),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__); sub=parser.add_subparsers(dest='command',required=True)
    for name in ('prepare','audit'):
        p=sub.add_parser(name); p.add_argument('--calibration-request',required=True); p.add_argument('--run-id',required=True)
        if name=='prepare': p.add_argument('--engineering-audit',required=True)
    for name in ('run','worker','aggregate'):
        p=sub.add_parser(name); p.add_argument('--request',required=True); p.add_argument('--confirm-request-hash',required=True)
        if name!='aggregate': p.add_argument('--stage',choices=['train','evaluate'],required=True)
        if name=='run': p.add_argument('--resume',action='store_true')
        if name=='worker': p.add_argument('--job',required=True)
    a=parser.parse_args()
    if a.command=='prepare': prepare(a.calibration_request,a.run_id,engineering_audit=a.engineering_audit)
    elif a.command=='worker': worker(a)
    elif a.command=='run': run(a.request,a.confirm_request_hash,a.stage,a.resume)
    elif a.command=='aggregate': aggregate_run(a.request,a.confirm_request_hash)
    else:
        path,h=prepare(a.calibration_request,a.run_id,audit=True)
        run(path,h,'train'); run(path,h,'evaluate')
        _,q=load_request(path,h); training_metadata(q,path.parent)
        write_once(path.parent/'audit.json',{**shared.metadata(),'status':'complete','engineering_gate':True,
            'request_hash':h,'bounded_training_jobs':1,'bounded_evaluation_episodes':3,
            'method_effect_established':False,'formal_training_started':False})


if __name__=='__main__': main()
