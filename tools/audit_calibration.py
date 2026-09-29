"""Bounded nine-development-root inference audit; never fit formal calibration."""
import argparse
from pathlib import Path
import sys

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import train_response_predictors as shared
import calibrate_response_predictors as cli
from prediction_rl.data.collection_store import read_json as read, file_hash as sha, write_once
from prediction_rl.prediction.trained_artifacts import load_trained_ensemble
from prediction_rl.prediction.interface import PredictorConfig
from prediction_rl.prediction.supervision import align_supervision
from prediction_rl.prediction.calibration import episode_scores, fit_episode_quantile, SETTINGS


def run(args):
    shared.clean(); rp,r,p,d=cli.prerequisites(args.training_review)
    source=ROOT/'artifacts/p4/p4_training_v1_01/report.json'
    if sha(source)!='174146a8d78962929e18bcd426f9c136db26f0f94deda5bfc449c488459dc3b5': raise ValueError('Development source changed')
    evidence=read(source); histories=[];packs=[]
    for record in evidence['history_inputs']:
        hp=ROOT/record['path']
        if sha(hp)!=record['sha256']: raise ValueError('Development history changed')
        h=read(hp);lp=ROOT/h['label_pack']
        if sha(lp)!=h['label_pack_sha256']: raise ValueError('Development labels changed')
        histories.append(h);packs.append(read(lp))
    manifest=read((ROOT/evidence['label_inputs'][0]['path']).parent/'split_manifest.json')
    batch=align_supervision(histories,packs,manifest,PredictorConfig(),split='development')
    if batch.fingerprint()!=evidence['data_hash']: raise ValueError('Development batch changed')
    out=shared.output_dir('p4',args.run_id)
    report={**shared.metadata(),'status':'running','purpose':'development_smoke','calibration_interface_gate':False,
        'source_hashes':shared.source_hashes(),'environment':shared.env(),'training_review_sha256':sha(rp),
        'formal_calibration_fitted':False,'test_opened':False,'methods':{}}
    previous=torch.get_num_threads();torch.set_num_threads(1)
    try:
        for mode,e in r['ensembles'].items():
            model=load_trained_ensemble(ROOT/e['path'],p,mode,e['sha256'],e['selected'])
            roster=manifest['development']
            a,ar=episode_scores(model,batch,roster,batch_roots=3,purpose='development_smoke')
            b,br=episode_scores(model,batch,roster,batch_roots=1,purpose='development_smoke')
            delta=max(abs(x['score']-y['score']) for x,y in zip(ar,br) if x['score']is not None)
            if any((x['score']is None)!=(y['score']is None) or x['valid_actor_time_cells']!=y['valid_actor_time_cells'] for x,y in zip(ar,br)):
                raise AssertionError('Batched eligibility changed')
            if not all(x['score']is None or abs(x['score']-y['score'])<=1e-5+1e-5*abs(y['score']) for x,y in zip(ar,br)):
                raise AssertionError('Batched residual mismatch')
            fit=fit_episode_quantile(a,SETTINGS)
            if fit['q']is not None or fit['order_statistic_one_based']!=4: raise AssertionError('Small-sample infinity mishandled')
            report['methods'][mode]={'checkpoint_sha256':e['sha256'],'roots':len(ar),'episodes':len(a),
                'maximum_batched_score_difference':delta,'three_episode_quantile_correctly_unbounded':True}
        report.update(status='complete',calibration_interface_gate=True)
    except BaseException as error:
        report.update(status='failed',failure_reason=f'{type(error).__name__}: {error}');raise
    finally:
        torch.set_num_threads(previous);report['finished_at']=shared.now();write_once(out/'report.json',report)
    print('calibration_audit='+str(out/'report.json'))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--training-review',required=True);parser.add_argument('--run-id',required=True)
    run(parser.parse_args())
