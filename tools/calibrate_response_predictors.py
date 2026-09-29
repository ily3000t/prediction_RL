"""Review frozen training; prepare and USER-run calibration only (test stays sealed)."""
import argparse
from pathlib import Path
import sys

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import train_response_predictors as shared
from prediction_rl.data.collection_store import read_json as read, file_hash as sha, write_once, run_lock
from prediction_rl.data.dataset_contract import digest
from prediction_rl.data.frozen_dataset import inside
from prediction_rl.prediction.trained_artifacts import review_training, load_trained_ensemble
from prediction_rl.prediction.calibration import SETTINGS, load_calibration_split, episode_scores, fit_episode_quantile, band_diagnostics

VERSION='frozen_response_calibration_v1'


def review(args):
    shared.clean(); out=shared.output_dir('training_reviews',args.run_id); start=shared.metadata()
    write_once(out/'started.json',start)
    try:
        result=review_training(ROOT,args.training_request,args.training_report)
        result.update(provenance=start,environment=shared.env(),finished_at=shared.now())
        write_once(out/'review.json',result)
    except BaseException as error:
        write_once(out/'failure.json',{**start,'status':'failed','error':f'{type(error).__name__}: {error}'})
        raise
    print('training_review='+str(out/'review.json')); return out/'review.json'


def prerequisites(path):
    path=inside(ROOT,path); r=read(path)
    if (r['version']!='completed_predictor_review_v1' or r['status']!='complete'
            or r['test_locked']is not True or r['calibrated']is not False
            or r['ddpg_ready']is not False or r['eligible_for_calibration']is not True):
        raise ValueError('Unaccepted training review')
    for name,h in r['input_hashes'].items():
        if sha(inside(ROOT,name))!=h: raise ValueError('Reviewed training artifact changed')
    p=read(inside(ROOT,r['request'])); d=read(inside(ROOT,r['dataset_review']))
    if digest(p)!=r['request_hash'] or p['environment']!=shared.env():
        raise ValueError('Training request/environment changed')
    for name,h in d['input_hashes'].items():
        if sha(inside(ROOT,name))!=h: raise ValueError('Dataset receipt/report changed')
    if d['protocol']['calibration']!=SETTINGS: raise ValueError('Unreviewed calibration settings')
    return path,r,p,d


def prepare(args):
    shared.clean(); rp,r,p,d=prerequisites(args.training_review)
    audit_path=inside(ROOT,args.calibration_audit); a=read(audit_path)
    if (a['status']!='complete' or a['calibration_interface_gate']is not True
            or a['purpose']!='development_smoke' or a['source_hashes']!=shared.source_hashes()
            or a['environment']!=shared.env() or a['training_review_sha256']!=sha(rp)):
        raise ValueError('Current trained-model development calibration audit required')
    # Only deserialize/alignment/hash data here: no predictor forward or quantile fit.
    batch=load_calibration_split(ROOT,d)
    request={'version':VERSION,'training_review':str(rp.relative_to(ROOT)),'training_review_sha256':sha(rp),
        'calibration_audit':str(audit_path.relative_to(ROOT)),'calibration_audit_sha256':sha(audit_path),
        'settings':SETTINGS,'data_hash':batch.fingerprint(),'source_hashes':shared.source_hashes(),
        'environment':shared.env(),'methods':p['training']['methods'],'batch_roots':16,'test_locked':True,
        'checkpoint_selection_allowed':False,'ddpg_ready':False}
    out=shared.output_dir('calibration',args.run_id); write_once(out/'request.json',request)
    write_once(out/'preparation.json',{**shared.metadata(),'status':'prepared_not_fitted','request_hash':digest(request),
        'calibration_roots':len(batch.root_ids),'calibration_episodes':len(d['manifest']['calibration'])})
    print('calibration_request='+str(out/'request.json'));print('confirm_request_hash='+digest(request))
    return out/'request.json'


def load_request(path):
    path=inside(ROOT,path); q=read(path)
    if (q['version']!=VERSION or q['settings']!=SETTINGS or q['methods']!=['ordinary','conditional']
            or q['batch_roots']!=16 or q['test_locked']is not True or q['checkpoint_selection_allowed']is not False
            or q['ddpg_ready']is not False or q['source_hashes']!=shared.source_hashes() or q['environment']!=shared.env()
            or read(path.parent/'preparation.json')['request_hash']!=digest(q)):
        raise ValueError('Calibration request/source/environment changed')
    rp,r,p,d=prerequisites(q['training_review'])
    if sha(rp)!=q['training_review_sha256'] or sha(inside(ROOT,q['calibration_audit']))!=q['calibration_audit_sha256']:
        raise ValueError('Calibration prerequisites changed')
    return path.parent,q,r,p,d


def run(args):
    shared.clean(); out,q,r,p,d=load_request(args.request)
    if args.confirm_request_hash!=digest(q): raise ValueError('Confirm exact calibration request hash')
    if any((out/name).exists() for name in ('started.json','report.json','calibration.json')):
        raise ValueError('Calibration already started; immutable output requires review, not overwrite')
    previous=torch.get_num_threads(); torch.set_num_threads(1)
    try:
        with run_lock(out):
            report={**shared.metadata(),'status':'running','request_hash':digest(q),'methods':{}}
            write_once(out/'started.json',report)
            try:
                batch=load_calibration_split(ROOT,d)
                if batch.fingerprint()!=q['data_hash']: raise ValueError('Calibration tensors changed')
                for mode in q['methods']:
                    source=r['ensembles'][mode]
                    model=load_trained_ensemble(inside(ROOT,source['path']),p,mode,source['sha256'],source['selected'])
                    rows,roots=episode_scores(model,batch,d['manifest']['calibration'],batch_roots=q['batch_roots'])
                    fit=fit_episode_quantile(rows,q['settings'])
                    report['methods'][mode]={'checkpoint_sha256':source['sha256'],'fit':fit,
                        'episode_scores':rows,'root_scores':roots,'diagnostics':band_diagnostics(model,batch,fit)}
                    print(f"[calibration] {mode} eligible={fit['eligible_episodes']} rank={fit['order_statistic_one_based']} q={fit['q']}",flush=True)
                frozen={'version':VERSION,'purpose':'frozen_observed_trajectory_bands','request_hash':digest(q),
                    'training_request_hash':r['request_hash'],'calibration_data_hash':q['data_hash'],'test_locked':True,
                    'method_effect_established':False,'ddpg_ready':False,
                    'methods':{m:{'checkpoint_sha256':v['checkpoint_sha256'],'fit':v['fit']} for m,v in report['methods'].items()}}
                write_once(out/'calibration.json',frozen)
                report.update(status='complete',calibration_sha256=sha(out/'calibration.json'),test_evaluated=False,
                    next_stage='P5_reference_and_candidate_ranking_protocol_review',ddpg_ready=False)
            except BaseException as error:
                report.update(status='failed',failure_reason=f'{type(error).__name__}: {error}'); raise
            finally:
                report['finished_at']=shared.now();write_once(out/'report.json',report)
                print('calibration_report='+str(out/'report.json'),flush=True)
    finally: torch.set_num_threads(previous)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); sub=parser.add_subparsers(dest='command',required=True)
    s=sub.add_parser('review');s.add_argument('--training-request',required=True);s.add_argument('--training-report',required=True);s.add_argument('--run-id',required=True)
    s=sub.add_parser('prepare');s.add_argument('--training-review',required=True);s.add_argument('--calibration-audit',required=True);s.add_argument('--run-id',required=True)
    s=sub.add_parser('run');s.add_argument('--request',required=True);s.add_argument('--confirm-request-hash',required=True)
    args=parser.parse_args();{'review':review,'prepare':prepare,'run':run}[args.command](args)
