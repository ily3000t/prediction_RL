"""Manual P5a validation/reference and frozen-band diagnosis; never opens test."""
import argparse
from pathlib import Path
import sys

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import train_response_predictors as shared
import calibrate_response_predictors as previous
from prediction_rl.data.collection_store import read_json as read,file_hash as sha,write_once,run_lock
from prediction_rl.data.dataset_contract import digest
from prediction_rl.data.frozen_dataset import inside,verify_historical_inputs,load_split
from prediction_rl.data.collection_protocol import collection_plan
from prediction_rl.prediction.calibration import load_calibration_split,fit_episode_quantile,SETTINGS
from prediction_rl.prediction.epoch_training import subset
from prediction_rl.prediction.interface import PredictorConfig
from prediction_rl.prediction.supervision import align_supervision
from prediction_rl.prediction.trained_artifacts import load_trained_ensemble
from prediction_rl.prediction.kinematic_reference import LaneConstantVelocity
from prediction_rl.prediction.offline_diagnostics import root_metrics,summarize,paired_difference,residual_attribution,summarize_attribution

CONFIG={'version':'p05_offline_diagnostics_v1','analysis_status':'development_validation_diagnostic_not_test',
    'calibration_request':'artifacts/calibration/predictor_v1/request.json','network':'RL-MPC-LaneMerging-master/merge.net.xml',
    'methods':['lane_chain_constant_velocity','ordinary','conditional'],'point_error_split':'validation',
    'band_attribution_splits':['validation','calibration'],'batch_roots':16,
    'point_metrics':'candidate_then_root_and_episode_equal_v1',
    'response_metric':'all_ten_candidate_pair_delta_scaled_mse_common_masks_v1',
    'reference':'root_anchored_lane_chain_constant_speed_zero_acceleration_v1',
    'unobserved_tail':'exclude_and_count_never_impute','hypothesis_test':False,'test_release':False,'ddpg_release':False}


def artifact_hashes(config_path,paths):
    # Tracked JSON can be checked out as LF or CRLF. Bind parsed parameters,
    # while retaining byte hashes for immutable run artifacts and the map.
    return {'json_canonical:'+str(config_path.relative_to(ROOT)):digest(read(config_path)),
            **{str(x.relative_to(ROOT)):sha(x) for x in paths}}


def prerequisites(config_path):
    cp=inside(ROOT,config_path); config=read(cp)
    if config!=CONFIG: raise ValueError('Unsupported P5a diagnostic protocol; do not tune on inspected outcomes')
    qp=inside(ROOT,config['calibration_request']);q=read(qp);out=qp.parent
    report=read(out/'report.json'); frozen=read(out/'calibration.json')
    if ((out/'writer.lock').exists() or report['status']!='complete' or report['working_tree_dirty']is not False
            or report['request_hash']!=digest(q) or frozen['request_hash']!=digest(q)
            or read(out/'preparation.json')['request_hash']!=digest(q) or report['test_evaluated']is not False
            or frozen['test_locked']is not True or frozen['ddpg_ready']is not False
            or sha(out/'calibration.json')!=report['calibration_sha256']): raise ValueError('Incomplete/changed calibration')
    verify_historical_inputs(ROOT,{'python_lf:'+k:v for k,v in q['source_hashes'].items()},report['git_commit'])
    rp,r,p,d=previous.prerequisites(q['training_review'])
    if (sha(rp)!=q['training_review_sha256'] or sha(inside(ROOT,q['calibration_audit']))!=q['calibration_audit_sha256']
            or frozen['training_request_hash']!=r['request_hash'] or frozen['calibration_data_hash']!=q['data_hash']
            or q['settings']!=SETTINGS or q['environment']!=shared.env()
            or set(report['methods'])!={'ordinary','conditional'} or set(frozen['methods'])!={'ordinary','conditional'}):
        raise ValueError('Calibration provenance changed')
    for mode,m in report['methods'].items():
        if {x['episode_id'] for x in m['episode_scores']}!=set(d['manifest']['calibration']): raise ValueError('Wrong calibration roster')
        fit=fit_episode_quantile(m['episode_scores'],SETTINGS)
        expected={'checkpoint_sha256':r['ensembles'][mode]['sha256'],'fit':fit}
        if m['fit']!=fit or m['checkpoint_sha256']!=expected['checkpoint_sha256'] or frozen['methods'][mode]!=expected:
            raise ValueError('Frozen fit differs from completed scores/model')
    # Remain an inspected validation diagnostic even though calibration is now frozen.
    plan=collection_plan(d['protocol'])
    for name in d['input_hashes']:
        if name.replace('\\','/').endswith('/request.json'):
            collection=inside(ROOT,name).parent
            if any((collection/j['job_id']).exists() for j in plan['jobs'] if j['split']=='test'):
                raise ValueError('P5a requires still-sealed test; review release separately')
    paths=[qp,out/'preparation.json',out/'report.json',out/'calibration.json',rp,inside(ROOT,config['network'])]
    return config,r,p,d,frozen,artifact_hashes(cp,paths)


def data(d,split):
    if split not in ('validation','calibration'): raise ValueError('No train/test evaluation in P5a')
    b=load_split(ROOT,d,split) if split=='validation' else load_calibration_split(ROOT,d)
    histories=[]
    for row in d['roots'][split]:
        path=inside(ROOT,row['history'])
        if sha(path)!=row['history_sha256']:raise ValueError('Observed history changed')
        histories.append(read(path))
    if [h['root_id'] for h in histories]!=list(b.root_ids): raise ValueError('History/batch order differs')
    return b,histories


def audit(args):
    shared.clean();config,r,p,d,frozen,hashes=prerequisites(args.config)
    evidence_path=ROOT/'artifacts/p4/p4_training_v1_01/report.json'
    if sha(evidence_path)!='174146a8d78962929e18bcd426f9c136db26f0f94deda5bfc449c488459dc3b5':raise ValueError('Development provenance changed')
    evidence=read(evidence_path); histories=[];packs=[]
    for row in evidence['history_inputs']:
        hp=ROOT/row['path']
        if sha(hp)!=row['sha256']:raise ValueError('Development history changed')
        h=read(hp);lp=ROOT/h['label_pack']
        if sha(lp)!=h['label_pack_sha256']:raise ValueError('Development labels changed')
        histories.append(h);packs.append(read(lp))
    manifest=read((ROOT/evidence['label_inputs'][0]['path']).parent/'split_manifest.json')
    cfg=PredictorConfig.from_dict(p['model']);b=align_supervision(histories,packs,manifest,cfg,split='development')
    if b.fingerprint()!=evidence['data_hash']:raise ValueError('Development tensors changed')
    out=shared.output_dir('p5',args.run_id);old=torch.get_num_threads();torch.set_num_threads(1)
    report={**shared.metadata(),'status':'running','purpose':'development_smoke','interface_gate':False,
        'source_hashes':shared.source_hashes(),'environment':shared.env(),'config_hash':digest(config),'inputs':hashes}
    try:
        reference=LaneConstantVelocity(ROOT/config['network']).predict(histories,cfg)
        root_metrics(reference,b);report['roots']=len(b.root_ids);report['methods']={}
        for mode,e in r['ensembles'].items():
            model=load_trained_ensemble(ROOT/e['path'],p,mode,e['sha256'],e['selected'])
            with torch.inference_mode():pred=model(b.observations,b.plans)['member_trajectories']
            rows=root_metrics(pred.mean(0),b);summary=summarize(rows,manifest['development'])
            attr=residual_attribution(pred,b,frozen['methods'][mode]['fit'],[h['inputs']['actor_ids'] for h in histories])
            summarize_attribution(attr,manifest['development'],frozen['methods'][mode]['fit'],'development')
            if mode=='ordinary' and summary['root_equal']['pairwise_response_delta_scaled_mse']!=summarize(root_metrics(reference,b),manifest['development'])['root_equal']['pairwise_response_delta_scaled_mse']:
                raise AssertionError('Plan-invariant references must have identical zero-response contrast error')
            report['methods'][mode]={'metrics_finite':True,'attributed_roots':len(attr)}
        report.update(status='complete',interface_gate=True,formal_evaluation_started=False,test_opened=False)
    except BaseException as error:
        report.update(status='failed',failure_reason=f'{type(error).__name__}: {error}');raise
    finally:
        torch.set_num_threads(old);report['finished_at']=shared.now();write_once(out/'report.json',report)
    print('p5a_audit='+str(out/'report.json'))


def prepare(args):
    shared.clean();config,r,p,d,frozen,hashes=prerequisites(args.config)
    ap=inside(ROOT,args.audit);a=read(ap)
    if (a['status']!='complete' or a['interface_gate']is not True or a['purpose']!='development_smoke'
            or a['source_hashes']!=shared.source_hashes() or a['environment']!=shared.env()
            or a['config_hash']!=digest(config) or a['inputs']!=hashes):raise ValueError('Need current bounded P5a audit')
    fingerprints={s:data(d,s)[0].fingerprint() for s in config['band_attribution_splits']}
    q={'config':config,'config_path':str(inside(ROOT,args.config).relative_to(ROOT)),'input_hashes':hashes,
        'audit':str(ap.relative_to(ROOT)),'audit_sha256':sha(ap),'data_hashes':fingerprints,
        'source_hashes':shared.source_hashes(),'environment':shared.env(),'test_locked':True,'ddpg_ready':False}
    out=shared.output_dir('offline',args.run_id);write_once(out/'request.json',q)
    write_once(out/'preparation.json',{**shared.metadata(),'status':'prepared_not_evaluated','request_hash':digest(q)})
    print('offline_request='+str(out/'request.json'));print('confirm_request_hash='+digest(q))


def load_request(path):
    path=inside(ROOT,path);q=read(path)
    if (q['source_hashes']!=shared.source_hashes() or q['environment']!=shared.env() or q['test_locked']is not True
            or q['ddpg_ready']is not False or read(path.parent/'preparation.json')['request_hash']!=digest(q)
            or sha(inside(ROOT,q['audit']))!=q['audit_sha256']):raise ValueError('Offline request/source changed')
    config,r,p,d,frozen,hashes=prerequisites(q['config_path'])
    if config!=q['config'] or hashes!=q['input_hashes']:raise ValueError('Offline inputs changed')
    return path.parent,q,r,p,d,frozen


def run(args):
    shared.clean();out,q,r,p,d,frozen=load_request(args.request)
    if args.confirm_request_hash!=digest(q):raise ValueError('Confirm exact P5a diagnostic request')
    if (out/'started.json').exists() or (out/'report.json').exists():raise ValueError('Never overwrite an existing diagnostic')
    old=torch.get_num_threads();torch.set_num_threads(1)
    try:
        with run_lock(out):
            report={**shared.metadata(),'status':'running','request_hash':digest(q),'splits':{},'test_evaluated':False,'ddpg_ready':False}
            write_once(out/'started.json',report)
            try:
                cfg=PredictorConfig.from_dict(p['model']);reference=LaneConstantVelocity(ROOT/q['config']['network'])
                models={mode:load_trained_ensemble(ROOT/e['path'],p,mode,e['sha256'],e['selected']) for mode,e in r['ensembles'].items()}
                for split in q['config']['band_attribution_splits']:
                    b,histories=data(d,split)
                    if b.fingerprint()!=q['data_hashes'][split]:raise ValueError('Offline tensors changed')
                    output={'point_metrics':{},'frozen_band_attribution':{}};report['splits'][split]=output;point_rows={}
                    if split=='validation':point_rows['lane_chain_constant_velocity']=root_metrics(reference.predict(histories,cfg),b)
                    for mode,model in models.items():
                        metrics=[];attribution=[]
                        with torch.inference_mode():
                            for start in range(0,len(b.root_ids),q['config']['batch_roots']):
                                stop=min(start+q['config']['batch_roots'],len(b.root_ids));sub=subset(b,list(range(start,stop)))
                                pred=model(sub.observations,sub.plans)['member_trajectories']
                                if split=='validation':metrics.extend(root_metrics(pred.mean(0),sub))
                                attribution.extend(residual_attribution(pred,sub,frozen['methods'][mode]['fit'],[h['inputs']['actor_ids'] for h in histories[start:stop]]))
                        if split=='validation':point_rows[mode]=metrics
                        output['frozen_band_attribution'][mode]=summarize_attribution(attribution,d['manifest'][split],frozen['methods'][mode]['fit'],split)
                        write_once(out/f'{split}_{mode}_attribution.json',attribution)
                    for mode,rows in point_rows.items():
                        output['point_metrics'][mode]=summarize(rows,d['manifest'][split]);write_once(out/f'{split}_{mode}_roots.json',rows)
                    if split=='validation':
                        output['paired_differences']={left+'__to__conditional':paired_difference(point_rows[left],point_rows['conditional'],d['manifest'][split]) for left in ('ordinary','lane_chain_constant_velocity')}
                    print('[p5a] completed split='+split,flush=True)
                report.update(status='complete',action_ranking_evaluated=False,method_effect_established=False,
                    next_stage='review_diagnostics_then_define_shared_ego_candidate_ranking')
            except BaseException as error:
                report.update(status='failed',failure_reason=f'{type(error).__name__}: {error}');raise
            finally:
                report['finished_at']=shared.now();write_once(out/'report.json',report);print('p5a_report='+str(out/'report.json'))
    finally:torch.set_num_threads(old)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    for name in ('audit','prepare'):
        s=sub.add_parser(name);s.add_argument('--config',required=True);s.add_argument('--run-id',required=True)
        if name=='prepare':s.add_argument('--audit',required=True)
    s=sub.add_parser('run');s.add_argument('--request',required=True);s.add_argument('--confirm-request-hash',required=True)
    args=parser.parse_args();{'audit':audit,'prepare':prepare,'run':run}[args.command](args)
