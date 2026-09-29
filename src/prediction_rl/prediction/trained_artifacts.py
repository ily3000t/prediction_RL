"""Read-only acceptance of completed training and strict selected ensembles."""
import math
from pathlib import Path

import torch

from prediction_rl.data.collection_store import read_json as read, file_hash as sha, verify_episode
from prediction_rl.data.dataset_contract import digest
from prediction_rl.data.frozen_dataset import inside, verify_historical_inputs
from prediction_rl.data.collection_protocol import collection_plan
from .checkpoint import CONTRACT
from .epoch_training import VERSION, selection_update
from .interface import PredictorConfig
from .model import PredictorEnsemble


def checked_state(model, state):
    template=model.state_dict()
    if not isinstance(state,dict) or set(state)!=set(template) or any(
            not isinstance(state[k],torch.Tensor) or state[k].shape!=v.shape or state[k].dtype!=v.dtype
            or not torch.isfinite(state[k]).all() for k,v in template.items()):
        raise ValueError('Incomplete/incompatible/nonfinite trained weights')
    if any(not torch.equal(state[k],v) for k,v in model.named_buffers()):
        raise ValueError('Trained fixed buffers changed')


def load_trained_ensemble(path, request, mode, expected_sha256, selected):
    if sha(path)!=expected_sha256: raise ValueError('Ensemble artifact hash changed')
    p=torch.load(path,map_location='cpu',weights_only=True)
    expected={'version':VERSION,'purpose':'formal_validation_selected_ensemble','contract':CONTRACT,
        'request_hash':digest(request),'model':request['model'],'mode':mode,
        'member_seeds':request['training']['member_seeds'],'selected':selected,
        'data_hashes':request['data_hashes'],'calibrated':False,'test_evaluated':False}
    if not isinstance(p,dict) or set(p)!=set(expected)|{'state_dict'} or any(p[k]!=v for k,v in expected.items()):
        raise ValueError('Trained ensemble identity/selection mismatch')
    model=PredictorEnsemble(PredictorConfig.from_dict(request['model']),mode,expected['member_seeds'])
    checked_state(model,p['state_dict']); model.load_state_dict(p['state_dict'],strict=True)
    model.eval(); model.requires_grad_(False)
    return model


def checked_history(history, metadata, training, eligible_roots):
    n=metadata['epoch']; batches=math.ceil(eligible_roots/training['batch_roots'])
    if (type(n)is not int or not 1<=n<=training['max_epochs'] or len(history)!=n
            or metadata['optimizer_steps']!=n*batches): raise ValueError('Invalid completed training budget')
    selection={'best_loss':None,'best_epoch':None,'patience_anchor':None,'bad_epochs':0}
    for epoch,row in enumerate(history,1):
        if selection['bad_epochs']>=training['early_stop_patience']: raise ValueError('Training continued after early stop')
        if (row['epoch']!=epoch or row['optimizer_steps']!=epoch*batches
                or not math.isfinite(row['train_pre_update_loss'])): raise ValueError('Invalid training history')
        best=selection_update(selection,row['validation_loss'],epoch,training['early_stop_min_delta'])
        if best!=row['selected_best']: raise ValueError('Best checkpoint selection changed')
    stopped=selection['bad_epochs']>=training['early_stop_patience']
    if not stopped and n!=training['max_epochs']: raise ValueError('Premature complete training')
    if (metadata['stop_reason']!=('patience' if stopped else 'epoch_budget')
            or metadata['best_epoch']!=selection['best_epoch']
            or metadata['best_validation_loss']!=selection['best_loss']): raise ValueError('Training summary differs')
    return selection


def review_training(repo, request_path, report_path):
    repo=Path(repo).resolve(); request_path=inside(repo,request_path); report_path=inside(repo,report_path)
    p,r=read(request_path),read(report_path); out=request_path.parent
    if (not report_path.is_relative_to(out/'invocations') or (out/'writer.lock').exists()
            or p['version']!=VERSION or r['status']!='complete' or r['working_tree_dirty']is not False
            or r['request_hash']!=digest(p) or r['calibrated']is not False or r['test_evaluated']is not False
            or p['calibration_used_for_selection']is not False or p['test_locked']is not True
            or read(out/'preparation.json')['request_hash']!=digest(p)):
        raise ValueError('Need immutable complete uncalibrated training with locked test')
    verify_historical_inputs(repo,{'python_lf:'+k:v for k,v in p['source_hashes'].items()},r['git_commit'])
    dp=inside(repo,p['dataset_review']); dataset=read(dp)
    if (sha(dp)!=p['dataset_review_sha256'] or dataset['status']!='complete' or dataset['test_locked']is not True
            or sha(inside(repo,p['training_audit']))!=p['training_audit_sha256']):
        raise ValueError('Training prerequisites changed')
    protocol=dataset['protocol']; plan=collection_plan(protocol)
    jobs=[{'id':f'{mode}_{seed}','mode':mode,'member_seed':seed}
          for mode in protocol['training']['methods'] for seed in protocol['training']['member_seeds']]
    if (p['training']!=protocol['training'] or p['model']!=protocol['model'] or p['jobs']!=jobs
            or [x['job'] for x in r['members']]!=jobs or set(r['ensembles'])!=set(protocol['training']['methods'])):
        raise ValueError('Frozen training job/config set changed')
    # Collection receipts bind the underlying data; formal test remains unopened.
    for name,h in dataset['input_hashes'].items():
        if sha(inside(repo,name))!=h: raise ValueError('Dataset receipt/report changed')
        if name.replace('\\','/').endswith('/request.json'):
            collection=inside(repo,name).parent
            if any((collection/j['job_id']).exists() for j in plan['jobs'] if j['split']=='test'):
                raise ValueError('Test collection opened before calibration freeze')
    paths=[request_path,report_path,out/'preparation.json',dp,inside(repo,p['training_audit'])]
    members=[]; states={}
    for job,reported in zip(jobs,r['members']):
        d=out/job['id']; binding={'request_hash':digest(p),'job':job}
        receipt=verify_episode(d,binding); m=receipt['metadata']; history=read(d/'history.json')
        if m!=reported or m['status']!='complete': raise ValueError('Member/report mismatch')
        checked_history(history,m,p['training'],dataset['summaries']['train']['eligible_roots'])
        selected=torch.load(d/'selected.pt',map_location='cpu',weights_only=True)
        expected={'version':VERSION,'purpose':'formal_selected_member','binding':binding,'model':p['model'],
            'contract':CONTRACT,'best_epoch':m['best_epoch'],'validation_loss':m['best_validation_loss'],'data_hashes':p['data_hashes']}
        if set(selected)!=set(expected)|{'state_dict'} or any(selected[k]!=v for k,v in expected.items()):
            raise ValueError('Selected member metadata changed')
        epoch=d/f"e{m['best_epoch']:04d}.pt"
        if read(epoch.with_suffix('.json'))!={'sha256':sha(epoch),'binding':binding}:
            raise ValueError('Best epoch receipt changed')
        checkpoint=torch.load(epoch,map_location='cpu',weights_only=True)
        if (checkpoint['binding']!=binding or checkpoint['epoch']!=m['best_epoch']
                or checkpoint['history']!=history[:m['best_epoch']]): raise ValueError('Best epoch lineage changed')
        state=selected['state_dict']; reference=checkpoint['model']
        if set(state)!=set(reference) or any(not torch.equal(state[k],reference[k]) for k in state):
            raise ValueError('Selected weights differ from best epoch')
        states[job['id']]=state
        paths.extend([d/'complete.json',d/'selected.pt',d/'history.json'])
        members.append({**m,'first_validation_loss':history[0]['validation_loss'],
            'last_validation_loss':history[-1]['validation_loss'],'last_train_pre_update_loss':history[-1]['train_pre_update_loss'],
            'milestones':[x for x in history if x['epoch'] in (1,20,50,80,100)]})
    for mode,e in r['ensembles'].items():
        ep=inside(repo,e['path'])
        expected_selected=[{'seed':j['member_seed'],'best_epoch':m['best_epoch'],
            'validation_loss':m['best_validation_loss'],'member_sha256':sha(out/j['id']/'selected.pt')}
            for j,m in zip(jobs,r['members']) if j['mode']==mode]
        if e['selected']!=expected_selected: raise ValueError('Ensemble member selection changed')
        model=load_trained_ensemble(ep,p,mode,e['sha256'],expected_selected)
        for member,j in zip(model.members,[j for j in jobs if j['mode']==mode]):
            if any(not torch.equal(v,states[j['id']][k]) for k,v in member.state_dict().items()):
                raise ValueError('Ensemble does not contain the selected member')
        paths.append(ep)
    return {'version':'completed_predictor_review_v1','status':'complete','training_commit':r['git_commit'],
        'request':str(request_path.relative_to(repo)),'request_hash':digest(p),'training_report':str(report_path.relative_to(repo)),
        'dataset_review':str(dp.relative_to(repo)),'input_hashes':{str(q.relative_to(repo)):sha(q) for q in paths},
        'members':members,'ensembles':r['ensembles'],'calibrated':False,'test_locked':True,
        'eligible_for_calibration':True,'prediction_effect_established':False,'ddpg_ready':False}
