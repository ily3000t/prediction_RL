"""Review frozen data, prepare a training request, then USER-run six CPU members."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import platform
import re
import subprocess
import sys
import uuid

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from prediction_rl.data.collection_store import read_json as read, write_once, file_hash as sha, run_lock, seal_episode, verify_episode
from prediction_rl.data.dataset_contract import digest
from prediction_rl.data.branching import fingerprint
from prediction_rl.data.frozen_dataset import review_dataset, load_split, inside
from prediction_rl.data.collection_protocol import collection_plan
from prediction_rl.prediction.interface import PredictorConfig
from prediction_rl.prediction.model import PredictorEnsemble
from prediction_rl.prediction.checkpoint import CONTRACT
from prediction_rl.prediction.epoch_training import MemberTrainer, atomic_torch, VERSION


def now(): return datetime.now(timezone.utc).isoformat()
def git(*a): return subprocess.check_output(['git','-C',str(ROOT),*a],text=True,encoding='utf-8').strip()
def clean():
    if git('status','--porcelain'): raise ValueError('Clean committed source/config required')
def metadata():
    return {'git_commit':git('rev-parse','HEAD'),'working_tree_dirty':False,'started_at':now(),
            'command':subprocess.list2cmdline([sys.executable,*sys.argv])}
def env():
    return {'python':sys.version,'executable':sys.executable,'os':platform.platform(),'torch':str(torch.__version__),
            'device':'cpu','dtype':'float32','torch_threads':1}
def source_hashes():
    return {str(p.relative_to(ROOT)):fingerprint(p.read_text(encoding='utf-8'))
            for folder in ('src','tools') for p in sorted((ROOT/folder).rglob('*.py'))}
def output_dir(kind,run_id):
    if not re.fullmatch('[A-Za-z0-9_-]{1,24}',run_id): raise ValueError('Use a short new run ID')
    out=ROOT/'artifacts'/kind/run_id; out.mkdir(parents=True,exist_ok=False); return out


def review(args):
    clean(); out=output_dir('data_reviews',args.run_id)
    started=metadata(); write_once(out/'started.json',started)
    try:
        result=review_dataset(ROOT,Path(args.request),Path(args.collection_report))
        result.update(provenance=started,finished_at=now(),environment=env())
        write_once(out/'review.json',result)
    except BaseException as error:
        write_once(out/'failure.json',{**started,'status':'failed','error':f'{type(error).__name__}: {error}'})
        raise
    print('dataset_review='+str(out/'review.json')); return out/'review.json'


def verify_review(path):
    path=inside(ROOT,path); r=read(path)
    if (r['status']!='complete' or not r['eligible_for_initial_training_trial'] or r['test_locked']is not True
            or r['prediction_effect_established']is not False): raise ValueError('Unsupported dataset review')
    collection_plan(r['protocol'])
    for name,h in r['input_hashes'].items():
        if sha(inside(ROOT,name))!=h: raise ValueError('Frozen dataset receipt/report changed')
    return path,r


def prepare(args):
    clean(); path,r=verify_review(args.dataset_review)
    acceptance_path=inside(ROOT,args.training_audit); a=read(acceptance_path)
    if (a['status']!='complete' or not a['epoch_training_gate'] or a['training_purpose']!='development_smoke'
            or a['source_hashes']!=source_hashes() or a['environment']!=env()):
        raise ValueError('Need accepted current CPU epoch/resume development audit')
    # Data preparation never computes gradients; calibration/test targets are not loaded.
    train,val=load_split(ROOT,r,'train'),load_split(ROOT,r,'validation')
    training=r['protocol']['training']; cfg=r['protocol']['model']
    jobs=[{'id':f'{mode}_{seed}','mode':mode,'member_seed':seed}
          for mode in training['methods'] for seed in training['member_seeds']]
    request={'version':VERSION,'dataset_review':str(path.relative_to(ROOT)),'dataset_review_sha256':sha(path),
             'data_hashes':[train.fingerprint(),val.fingerprint()],'training':training,'model':cfg,'jobs':jobs,
             'source_hashes':source_hashes(),'environment':env(),'training_audit':str(acceptance_path.relative_to(ROOT)),
             'training_audit_sha256':sha(acceptance_path),'calibration_used_for_selection':False,'test_locked':True}
    out=output_dir('training',args.run_id); write_once(out/'request.json',request)
    write_once(out/'preparation.json',{**metadata(),'status':'prepared_not_trained','request_hash':digest(request),
        'job_count':len(jobs),'train_roots':len(train.root_ids),'validation_roots':len(val.root_ids),'finished_at':now()})
    print('training_request='+str(out/'request.json')); print('confirm_request_hash='+digest(request))
    return out/'request.json'


def load_request(path):
    path=inside(ROOT,path); p=read(path)
    if (p['version']!=VERSION or p['test_locked']is not True or p['calibration_used_for_selection']is not False
            or p['source_hashes']!=source_hashes() or p['environment']!=env()
            or read(path.parent/'preparation.json')['request_hash']!=digest(p)):
        raise ValueError('Training request/source/environment changed')
    rp,r=verify_review(p['dataset_review'])
    if sha(rp)!=p['dataset_review_sha256'] or sha(inside(ROOT,p['training_audit']))!=p['training_audit_sha256']:
        raise ValueError('Training prerequisite changed')
    t=r['protocol']['training']
    expected=[{'id':f'{mode}_{seed}','mode':mode,'member_seed':seed} for mode in t['methods'] for seed in t['member_seeds']]
    if p['training']!=t or p['model']!=r['protocol']['model'] or p['jobs']!=expected:
        raise ValueError('Unreviewed training settings/member selection')
    return path.parent,p,r


def train_member(directory,p,job,train,val,*,resume):
    cfg=PredictorConfig.from_dict(p['model']); binding={'request_hash':digest(p),'job':job}
    if (directory/'complete.json').exists():
        if not resume: raise ValueError('Use --resume for a completed member')
        return verify_episode(directory,binding)['metadata']
    if directory.exists() and not resume: raise ValueError('Existing training member; use reviewed resume')
    if not directory.exists():
        directory.mkdir(parents=True); write_once(directory/'started.json',{'binding':binding,**metadata()})
    if read(directory/'started.json')['binding']!=binding: raise ValueError('Member run identity changed')
    # Epoch commits are immutable. An interrupted unsaved epoch is deterministically
    # replayed from the last committed epoch, not treated as a new training seed.
    epochs=sorted(directory.glob('e[0-9][0-9][0-9][0-9].pt'))
    if [f.name for f in epochs]!=[f'e{i:04d}.pt' for i in range(1,len(epochs)+1)]:
        raise ValueError('Noncontiguous training checkpoints')
    if epochs:
        for f in epochs:
            receipt=read(f.with_suffix('.json'))
            if receipt!={'sha256':sha(f),'binding':binding}: raise ValueError('Checkpoint hash/identity changed')
        trainer=MemberTrainer.load(epochs[-1],cfg,job['mode'],job['member_seed'],p['training'],train,val,binding)
    else:
        trainer=MemberTrainer(cfg,job['mode'],job['member_seed'],p['training'],train,val)
    while not trainer.finished:
        row=trainer.run_epoch(); path=directory/f'e{trainer.epoch:04d}.pt'
        trainer.save(path,binding); write_once(path.with_suffix('.json'),{'sha256':sha(path),'binding':binding})
        print(f"[predictor] {job['id']} epoch={trainer.epoch}/{p['training']['max_epochs']} train={row['train_pre_update_loss']:.6f} validation={row['validation_loss']:.6f} best={row['selected_best']}",flush=True)
    result={**binding,'status':'complete','epoch':trainer.epoch,'best_epoch':trainer.selection['best_epoch'],
            'best_validation_loss':trainer.selection['best_loss'],'optimizer_steps':trainer.steps,
            'stop_reason':'patience' if trainer.selection['bad_epochs']>=p['training']['early_stop_patience'] else 'epoch_budget'}
    # If interruption happened during final publication, refuse overwrite for diagnosis.
    atomic_torch(directory/'selected.pt',{'version':VERSION,'purpose':'formal_selected_member','binding':binding,
        'model':p['model'],'contract':CONTRACT,'best_epoch':result['best_epoch'],
        'validation_loss':result['best_validation_loss'],'data_hashes':p['data_hashes'],'state_dict':trainer.best_state})
    write_once(directory/'history.json',trainer.history)
    seal_episode(directory,binding,result)
    return result


def export_ensemble(out,p,mode):
    cfg=PredictorConfig.from_dict(p['model']); seeds=p['training']['member_seeds']
    model=PredictorEnsemble(cfg,mode,seeds); selected=[]
    for i,seed in enumerate(seeds):
        job=next(j for j in p['jobs'] if j['mode']==mode and j['member_seed']==seed)
        directory=out/job['id']; verify_episode(directory,{'request_hash':digest(p),'job':job})
        payload=torch.load(directory/'selected.pt',map_location='cpu',weights_only=True)
        if (payload['version']!=VERSION or payload['model']!=p['model']
                or payload['purpose']!='formal_selected_member' or payload['contract']!=CONTRACT
                or payload['binding']!={'request_hash':digest(p),'job':job} or payload['data_hashes']!=p['data_hashes']):
            raise ValueError('Selected member lineage mismatch')
        template=model.members[i].state_dict(); state=payload['state_dict']
        if set(state)!=set(template) or any(not isinstance(state[k],torch.Tensor)
                or state[k].shape!=v.shape or state[k].dtype!=v.dtype or not torch.isfinite(state[k]).all()
                for k,v in template.items()): raise ValueError('Invalid selected member weights')
        if any(not torch.equal(state[k],v) for k,v in model.members[i].named_buffers()):
            raise ValueError('Selected member fixed buffers changed')
        model.members[i].load_state_dict(payload['state_dict'],strict=True)
        selected.append({'seed':seed,'best_epoch':payload['best_epoch'],'validation_loss':payload['validation_loss'],
                         'member_sha256':sha(directory/'selected.pt')})
    path=out/(mode+'_ensemble.pt')
    envelope={'version':VERSION,'purpose':'formal_validation_selected_ensemble','contract':CONTRACT,
              'request_hash':digest(p),'model':p['model'],'mode':mode,'member_seeds':seeds,'selected':selected,
              'data_hashes':p['data_hashes'],'state_dict':model.state_dict(),'calibrated':False,'test_evaluated':False}
    if path.exists():
        old=torch.load(path,map_location='cpu',weights_only=True)
        if old.keys()!=envelope.keys() or set(old['state_dict'])!=set(envelope['state_dict']) or any(old[k]!=v for k,v in envelope.items() if k!='state_dict') or any(
                not torch.equal(old['state_dict'][k],v) for k,v in envelope['state_dict'].items()):
            raise ValueError('Existing ensemble export differs')
    else: atomic_torch(path,envelope)
    return {'path':str(path.relative_to(ROOT)),'sha256':sha(path),'selected':selected}


def run(args):
    clean(); out,p,r=load_request(args.request)
    if args.confirm_request_hash!=digest(p): raise ValueError('Confirm the exact prepared training request')
    previous=torch.get_num_threads(); torch.set_num_threads(1)
    try:
        with run_lock(out):
            invocation=out/'invocations'/uuid.uuid4().hex[:12]; invocation.mkdir(parents=True)
            report={**metadata(),'request_hash':digest(p),'status':'running','members':[]}
            write_once(invocation/'started.json',report)
            try:
                train,val=load_split(ROOT,r,'train'),load_split(ROOT,r,'validation')
                if [train.fingerprint(),val.fingerprint()]!=p['data_hashes']: raise ValueError('Training tensors changed')
                for job in p['jobs']:
                    report['members'].append(train_member(out/job['id'],p,job,train,val,resume=args.resume))
                report.update(status='complete',ensembles={m:export_ensemble(out,p,m) for m in p['training']['methods']},
                              calibrated=False,test_evaluated=False,next_stage='calibration_and_P5_offline_acceptance')
            except BaseException as error:
                report.update(status='failed',failure_reason=f'{type(error).__name__}: {error}'); raise
            finally:
                report['finished_at']=now(); write_once(invocation/'report.json',report)
                print('predictor_training_report='+str(invocation/'report.json'),flush=True)
    finally: torch.set_num_threads(previous)


def main():
    parser=argparse.ArgumentParser(description=__doc__); sub=parser.add_subparsers(dest='command',required=True)
    q=sub.add_parser('review'); q.add_argument('--request',required=True); q.add_argument('--collection-report',required=True); q.add_argument('--run-id',required=True)
    q=sub.add_parser('prepare'); q.add_argument('--dataset-review',required=True); q.add_argument('--training-audit',required=True); q.add_argument('--run-id',required=True)
    q=sub.add_parser('run'); q.add_argument('--request',required=True); q.add_argument('--confirm-request-hash',required=True); q.add_argument('--resume',action='store_true')
    args=parser.parse_args(); {'review':review,'prepare':prepare,'run':run}[args.command](args)


if __name__=='__main__': main()
