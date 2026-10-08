"""USER-run opt-in predictor objective trial; no SUMO, calibration or DDPG run.

Frozen baseline/data are read in place from a separate reference worktree.
Every new output belongs to THIS worktree. Legacy requests are not rewritten.
"""
import argparse
from pathlib import Path
import sys
import uuid

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
import train_response_predictors as shared
from prediction_rl.data.collection_store import read_json as read, file_hash as sha, write_once, run_lock, verify_episode
from prediction_rl.data.dataset_contract import digest
from prediction_rl.data.frozen_dataset import inside, load_split
from prediction_rl.prediction.candidate_difference import (
    DIFFERENCE_LOSS_CONTRACT, DIFFERENCE_TRAINING_VERSION, difference_training,
    response_difference_loss,
)
from prediction_rl.prediction.checkpoint import CONTRACT
from prediction_rl.prediction.epoch_training import subset, VERSION
from prediction_rl.prediction.interface import PredictorConfig
from prediction_rl.prediction.model import PredictorEnsemble
from prediction_rl.prediction.offline_diagnostics import root_metrics, summarize, paired_difference
from prediction_rl.prediction.trained_artifacts import review_training, load_trained_ensemble, checked_state


BOUNDARIES = {
    'execution_contract':'simulation_blocking_exact_v1',
    'online_feature_layout':'unchanged',
    'reward_action_policy_network':'unchanged',
    'data_splits':'reuse_frozen_train_validation_only',
    'selection_uses_calibration_or_test':False,
    'ddpg_training_authorized':False,
    'ordinary_and_absolute_conditional':'reuse_frozen_reference_ensembles',
}


def checked_config(config, baseline):
    if (set(config)!={'schema_version','experiment_id','review_status','baseline','model','training','boundaries'}
            or config['schema_version']!=1 or config['experiment_id']!='candidate_response_difference_v1'
            or config['review_status']!='predictor_training_manual_start_required'
            or config['boundaries']!=BOUNDARIES
            or set(config['baseline'])!={'training_request','training_report'}
            or baseline['version']!=VERSION):
        raise ValueError('Unsupported difference trial config/boundaries/baseline')
    PredictorConfig.from_dict(config['model'])
    expected = difference_training(baseline['training'], config['training'].get('difference_lambda'))
    if config['model']!=baseline['model'] or config['training']!=expected:
        raise ValueError('Only loss/weight and conditional job scheduling may change from the frozen baseline')
    return expected


def jobs(training):
    seeds=training['member_seeds']
    if len(seeds)!=3 or len(set(seeds))!=3:
        raise ValueError('Keep all three frozen predictor member seeds')
    return [{'id':f'conditional_{seed}','mode':'conditional','member_seed':seed} for seed in seeds]


def verify_reference(reference, bindings):
    for name, expected in bindings.items():
        if sha(inside(reference,name))!=expected:
            raise ValueError('Frozen reference artifact changed: '+name)


def prepare(args):
    shared.clean()
    config_path=inside(ROOT,args.config); config=read(config_path)
    reference=Path(args.reference_root).resolve()
    if not reference.is_dir() or reference==ROOT.resolve():
        raise ValueError('Use a separate existing baseline/data worktree as read-only reference')
    baseline_path=inside(reference,config['baseline']['training_request'])
    baseline=read(baseline_path); training=checked_config(config,baseline)
    # Strict read-only acceptance of original members, receipts and historical code.
    reviewed=review_training(reference,baseline_path,config['baseline']['training_report'])
    dataset=read(inside(reference,baseline['dataset_review']))
    train,val=load_split(reference,dataset,'train'),load_split(reference,dataset,'validation')
    if [train.fingerprint(),val.fingerprint()]!=baseline['data_hashes']:
        raise ValueError('Do not change or recollect the matched training tensors')
    bindings={**reviewed['input_hashes'],**dataset['input_hashes']}
    verify_reference(reference,bindings)
    request={
        'version':DIFFERENCE_TRAINING_VERSION,'loss_contract':DIFFERENCE_LOSS_CONTRACT,
        'configuration_path':str(config_path.relative_to(ROOT)), 'configuration_sha256':sha(config_path),
        'resolved_config':config,'reference_root':str(reference),
        'baseline_request_hash':digest(baseline),'baseline_training_commit':reviewed['training_commit'],
        'baseline_input_hashes':bindings,'baseline_ensembles':reviewed['ensembles'],
        'dataset_review':baseline['dataset_review'],'data_hashes':baseline['data_hashes'],
        'model':config['model'],'training':training,'jobs':jobs(training),
        'source_hashes':shared.source_hashes(),'environment':shared.env(),
        'calibration_used_for_selection':False,'test_locked':True,'ddpg_ready':False,
    }
    out=shared.output_dir('prediction_difference',args.run_id)
    write_once(out/'request.json',request)
    # Coverage arithmetic only; no model forward, gradient or calibration fitting.
    coverage={split:response_difference_loss(b.targets,b.targets,b.target_mask)[1]
              for split,b in (('train',train),('validation',val))}
    write_once(out/'preparation.json',{
        **shared.metadata(),'status':'prepared_not_trained','request_hash':digest(request),
        'job_count':len(request['jobs']),'reference_artifacts_read_only':True,
        'train_roots':len(train.root_ids),'validation_roots':len(val.root_ids),
        'pair_coverage':coverage,'finished_at':shared.now(),
    })
    print('difference_training_request='+str(out/'request.json'))
    print('confirm_request_hash='+digest(request))
    return out/'request.json'


def load_request(path):
    path=inside(ROOT,path); p=read(path)
    if (p['version']!=DIFFERENCE_TRAINING_VERSION or p['loss_contract']!=DIFFERENCE_LOSS_CONTRACT
            or p['source_hashes']!=shared.source_hashes() or p['environment']!=shared.env()
            or p['test_locked']is not True or p['calibration_used_for_selection']is not False
            or p['ddpg_ready']is not False
            or read(path.parent/'preparation.json')['request_hash']!=digest(p)):
        raise ValueError('Difference request/source/environment/boundaries changed')
    config_path=inside(ROOT,p['configuration_path']); config=read(config_path)
    if sha(config_path)!=p['configuration_sha256'] or config!=p['resolved_config']:
        raise ValueError('Frozen resolved configuration changed')
    reference=Path(p['reference_root']).resolve()
    if reference==ROOT.resolve(): raise ValueError('Do not write into the reference worktree')
    verify_reference(reference,p['baseline_input_hashes'])
    baseline=read(inside(reference,config['baseline']['training_request']))
    if digest(baseline)!=p['baseline_request_hash']:
        raise ValueError('Frozen baseline request changed')
    training=checked_config(config,baseline)
    if p['training']!=training or p['model']!=config['model'] or p['jobs']!=jobs(training):
        raise ValueError('Unreviewed loss/member/budget selection')
    dataset=read(inside(reference,p['dataset_review']))
    if p['dataset_review']!=baseline['dataset_review'] or p['data_hashes']!=baseline['data_hashes']:
        raise ValueError('Unreviewed dataset lineage')
    return path.parent,p,reference,baseline,dataset


def confirmed(args):
    loaded=load_request(args.request)
    if args.confirm_request_hash!=digest(loaded[1]):
        raise ValueError('Confirm the exact prepared difference request hash')
    return loaded


def run(args):
    shared.clean(); out,p,reference,baseline,dataset=confirmed(args)
    previous=torch.get_num_threads(); torch.set_num_threads(1)
    try:
        with run_lock(out):
            invocation=out/'invocations'/uuid.uuid4().hex[:12]; invocation.mkdir(parents=True)
            report={**shared.metadata(),'version':DIFFERENCE_TRAINING_VERSION,
                'request_hash':digest(p),'status':'running','members':[],
                'reference_artifacts_read_only':True,'ddpg_ready':False}
            write_once(invocation/'started.json',report)
            try:
                train,val=load_split(reference,dataset,'train'),load_split(reference,dataset,'validation')
                if [train.fingerprint(),val.fingerprint()]!=p['data_hashes']:
                    raise ValueError('Frozen training tensors changed')
                for job in p['jobs']:
                    report['members'].append(shared.train_member(out/job['id'],p,job,train,val,resume=args.resume))
                report.update(status='complete',ensembles={'conditional':shared.export_ensemble(out,p,'conditional')},
                    calibrated=False,test_evaluated=False,next_stage='offline_validation_then_manual_review')
            except BaseException as error:
                report.update(status='failed',failure_reason=f'{type(error).__name__}: {error}'); raise
            finally:
                report['finished_at']=shared.now(); write_once(invocation/'report.json',report)
                print('difference_training_report='+str(invocation/'report.json'),flush=True)
    finally: torch.set_num_threads(previous)


def completed_ensemble(out,p,report_path):
    report_path=inside(ROOT,report_path); report=read(report_path)
    if ((out/'writer.lock').exists() or not report_path.is_relative_to(out/'invocations') or report['status']!='complete'
            or report['version']!=p['version'] or report['request_hash']!=digest(p)
            or report['working_tree_dirty']is not False or report['ddpg_ready']is not False
            or report['calibrated']is not False or report['test_evaluated']is not False
            or [r['job'] for r in report['members']]!=p['jobs']
            or set(report['ensembles'])!={'conditional'}):
        raise ValueError('Require a complete immutable difference training report')
    selected=[]
    for job,metadata in zip(p['jobs'],report['members']):
        d=out/job['id']; binding={'request_hash':digest(p),'job':job}
        if verify_episode(d,binding)['metadata']!=metadata:
            raise ValueError('Difference member report/receipt mismatch')
        selected.append({'seed':job['member_seed'],'best_epoch':metadata['best_epoch'],
            'validation_loss':metadata['best_validation_loss'],'member_sha256':sha(d/'selected.pt')})
    record=report['ensembles']['conditional']; path=inside(ROOT,record['path'])
    if record['selected']!=selected or sha(path)!=record['sha256']:
        raise ValueError('Difference ensemble selection/hash changed')
    payload=torch.load(path,map_location='cpu',weights_only=True)
    expected={**shared.artifact_identity(p),'purpose':'formal_validation_selected_ensemble','contract':CONTRACT,
        'request_hash':digest(p),'model':p['model'],'mode':'conditional',
        'member_seeds':p['training']['member_seeds'],'selected':selected,
        'data_hashes':p['data_hashes'],'calibrated':False,'test_evaluated':False}
    if set(payload)!=set(expected)|{'state_dict'} or any(payload[k]!=v for k,v in expected.items()):
        raise ValueError('Difference ensemble envelope/contract changed')
    model=PredictorEnsemble(PredictorConfig.from_dict(p['model']),'conditional',p['training']['member_seeds'])
    checked_state(model,payload['state_dict']); model.load_state_dict(payload['state_dict'],strict=True)
    for member,job in zip(model.members,p['jobs']):
        state=torch.load(out/job['id']/'selected.pt',map_location='cpu',weights_only=True)['state_dict']
        if set(state)!=set(member.state_dict()) or any(not torch.equal(v,state[k]) for k,v in member.state_dict().items()):
            raise ValueError('Ensemble differs from selected members')
    model.eval(); model.requires_grad_(False)
    return model,record


def diagnose(args):
    shared.clean(); out,p,reference,baseline,dataset=confirmed(args)
    if (out/'offline_validation.json').exists():
        raise ValueError('Offline validation already exists; preserve the immutable result')
    delta,record=completed_ensemble(out,p,args.training_report)
    models={'conditional_difference':delta}
    for mode in ('ordinary','conditional'):
        original=p['baseline_ensembles'][mode]
        models[mode+'_absolute']=load_trained_ensemble(inside(reference,original['path']),
            baseline,mode,original['sha256'],original['selected'])
    # Only validation; no calibration fit, test collection, policy training or SUMO.
    batch=load_split(reference,dataset,'validation')
    if batch.fingerprint()!=p['data_hashes'][1]: raise ValueError('Validation tensors changed')
    previous=torch.get_num_threads(); torch.set_num_threads(1)
    try:
        rows={name:[] for name in models}
        with torch.inference_mode():
            for offset in range(0,len(batch.root_ids),p['training']['batch_roots']):
                b=subset(batch,list(range(offset,min(offset+p['training']['batch_roots'],len(batch.root_ids)))))
                for name,model in models.items(): rows[name].extend(root_metrics(model(b.observations,b.plans)['mean'],b))
        roster=dataset['manifest']['validation']
        result={**shared.metadata(),'version':'candidate_difference_validation_diagnostics_v1',
            'status':'complete','request_hash':digest(p),'scope':'frozen_validation_not_independent_test',
            'training_report_sha256':sha(inside(ROOT,args.training_report)),
            'checkpoint_sha256':record['sha256'],'methods':{name:summarize(r,roster) for name,r in rows.items()},
            'paired_delta_minus_absolute':paired_difference(rows['conditional_absolute'],rows['conditional_difference'],roster),
            'paired_delta_minus_ordinary':paired_difference(rows['ordinary_absolute'],rows['conditional_difference'],roster),
            'root_metrics':rows,'method_effect_established':False,'ddpg_ready':False,
            'negative_results_are_valid':True,'finished_at':shared.now()}
        path=out/'offline_validation.json'; write_once(path,result)
        print('difference_validation_report='+str(path)); return path
    finally: torch.set_num_threads(previous)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    q=sub.add_parser('prepare'); q.add_argument('--config',required=True)
    q.add_argument('--reference-root',required=True); q.add_argument('--run-id',required=True)
    for command in ('run','diagnose'):
        q=sub.add_parser(command); q.add_argument('--request',required=True)
        q.add_argument('--confirm-request-hash',required=True)
        if command=='run': q.add_argument('--resume',action='store_true')
        else: q.add_argument('--training-report',required=True)
    args=parser.parse_args(); {'prepare':prepare,'run':run,'diagnose':diagnose}[args.command](args)


if __name__=='__main__': main()
