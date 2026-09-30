"""P5c development audit and USER-run training-data diagnostic, no retraining."""
import argparse
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'tools')]
import diagnose_candidate_ranking as ranking
import train_response_predictors as shared
from prediction_rl.data.collection_store import read_json as read, file_hash as sha, write_once, run_lock, verify_episode
from prediction_rl.data.dataset_contract import digest, build_branch_labels
from prediction_rl.data.frozen_dataset import inside, verify_historical_inputs
from prediction_rl.data.collection_protocol import collection_plan
from prediction_rl.prediction.interface import PredictorConfig
from prediction_rl.prediction.supervision import align_supervision
from prediction_rl.prediction.epoch_training import subset
from prediction_rl.prediction.trained_artifacts import load_trained_ensemble
from prediction_rl.prediction.kinematic_reference import LaneConstantVelocity
from prediction_rl.prediction.candidate_ranking import summarize_ranking
from prediction_rl.prediction.task_relevance import diagnose_root, aggregate

CONFIG = {'version': 'p05_task_relevance_v1', 'purpose': 'in_sample_task_diagnosis_not_generalization',
    'ranking_request': 'artifacts/ranking/predictor_v1_rank1/request.json',
    'training_collection_request': 'artifacts/data/response_v1/request.json',
    'development_request': 'artifacts/p4/p4_collect_v2_01/request.json',
    'development_report': 'artifacts/p4/p4_collect_v2_01/invocations/398fb383670c/report.json',
    'methods': ['lane_chain_constant_velocity', 'ordinary', 'conditional'], 'analysis_split': 'train',
    'batch_roots': 16, 'return_definition': 'undiscounted_recorded_prefix_not_policy_Q',
    'return_comparability': 'both_terminal_or_both_complete_5s',
    'events': 'preserve_original_labels_and_censoring',
    'coverage': 'all_observed_nonterminal_vehicles_nearest_bumper_and_same_path_gap',
    'localization': 'selected_common_support_true_critical_cell',
    'model_update': False, 'feature_release': False, 'ddpg_release': False, 'test_release': False}


def prerequisites(path):
    cp = inside(ROOT, path); config = read(cp)
    if config != CONFIG: raise ValueError('Unknown or modified P5c diagnostic contract')
    qp = inside(ROOT, config['ranking_request']); q = read(qp); out = qp.parent
    report = read(out/'report.json'); rows = read(out/'roots.json')
    if ((out/'writer.lock').exists() or report['status'] != 'complete' or report['working_tree_dirty'] is not False
            or report['request_hash'] != digest(q) or report['test_evaluated'] is not False or report['ddpg_ready'] is not False
            or sha(out/'roots.json') != report['roots_sha256'] or read(out/'preparation.json')['request_hash'] != digest(q)
            or sha(inside(ROOT, q['audit'])) != q['audit_sha256'] or q['environment'] != shared.env()):
        raise ValueError('Ranking evidence changed/incomplete')
    verify_historical_inputs(ROOT, {'python_lf:'+k: v for k,v in q['source_hashes'].items()}, report['git_commit'])
    c,r,p,d,hashes = ranking.prerequisites(q['config_path'])
    settings=read(ROOT/c['upstream_config'])
    expected={'REWARD_FUNCTION':'Slotted Jerk','CRASH_REWARD':-10,'SUCCESS_REWARD':10,'TIME_REWARD':-.1,'ALT_J_WEIGHT':.1}
    if any(settings.get(k)!=v for k,v in expected.items()):raise ValueError('Unsupported upstream reward settings')
    if c != q['config'] or hashes != q['input_hashes'] or summarize_ranking(rows,d['manifest']['validation'],c['methods']) != report['summary']:
        raise ValueError('Prior ranking source/summary differs')
    # Negative ranking/parity results are retained, not required to turn green.
    dp = inside(ROOT,config['development_request']); drp = inside(ROOT,config['development_report'])
    dq, dr = read(dp), read(drp)
    if dr['status'] != 'complete' or dr['collector_gate'] is not True or dr['request_hash'] != digest(dq) or dr['working_tree_dirty'] is not False:
        raise ValueError('Development collection not accepted')
    verify_historical_inputs(ROOT,dq['input_hashes'],dr['git_commit'])
    tp = inside(ROOT,config['training_collection_request'])
    if sha(tp) != d['input_hashes'].get(str(tp.relative_to(ROOT))): raise ValueError('Wrong training collection')
    paths = [qp,out/'report.json',out/'roots.json',out/'preparation.json',inside(ROOT,q['audit']),dp,drp,tp]
    hashes.update(ranking.previous.artifact_hashes(cp,paths))
    for name in ('dqn.py','merge_gym.py'):
        file = ROOT/'RL-MPC-LaneMerging-master'/name
        hashes['python_lf:'+str(file.relative_to(ROOT))] = shared.fingerprint(file.read_text(encoding='utf-8'))
    return config,r,p,d,hashes


def load_root(directory, expected, manifest, cfg, split):
    h, pack = read(directory/'history.json'), read(directory/'labels.json')
    if h['root_id'] != expected or inside(ROOT,h['label_pack']) != directory/'labels.json' or sha(directory/'labels.json') != h['label_pack_sha256']:
        raise ValueError('Wrong root/history-label join')
    align_supervision([h],[pack],manifest,cfg,split=split)
    branches = []
    for candidate in sorted(pack['candidates'],key=lambda c:c['candidate_id']):
        cid = candidate['candidate_id']; branch = read(directory/f'c{cid}.json')
        if read(directory/f'c{cid}_repeat.json') != branch: raise ValueError('Branch repeat differs')
        if build_branch_labels(pack['root_traffic'],branch['trace'],candidate['labels']['requested_plan'],.2) != candidate['labels']:
            raise ValueError('Raw trace and labels differ')
        fm = branch['future_metadata']
        if len(fm['frames']) != len(branch['trace']) or fm['trajectory_usable'] != [not t['done'] for t in branch['trace']]:
            raise ValueError('Future geometry mask/length differs')
        for frame,step in zip(fm['frames'],branch['trace']):
            after = step['info']['execution_audit']['after']
            if frame['simulation_time_s'] != after['simulation_time_s'] or set(frame['vehicles']) != set(after['vehicles']):
                raise ValueError('Future/base frame differs')
            for actor,state in after['vehicles'].items():
                if any(frame['vehicles'][actor].get(k) != v for k,v in state.items()): raise ValueError('Future/base actor differs')
        branches.append(branch)
    return h,pack,branches


def data(config,p,d,split):
    if split not in ('development','train'): raise ValueError('P5c cannot open validation/calibration/test data for new analysis')
    cfg = PredictorConfig.from_dict(p['model']); result = []; receipts = {}
    if split == 'development':
        qp = inside(ROOT,config['development_request']); q = read(qp); manifest = q['split_manifest']
        jobs = [(qp.parent/f'e{i:03d}',{'request_hash':digest(q),'seed':seed}) for i,seed in enumerate(q['config']['simulator_seeds'])]
    else:
        qp = inside(ROOT,config['training_collection_request']); q = read(qp); manifest = d['manifest']
        jobs = [(qp.parent/job['job_id'],{'request_hash':digest(q),'job':job}) for job in collection_plan(q['protocol'])['jobs'] if job['split']=='train']
    for directory,binding in jobs:
        receipt_path = directory/'complete.json'
        if split == 'train' and sha(receipt_path) != d['input_hashes'].get(str(receipt_path.relative_to(ROOT))):
            raise ValueError('Training receipt changed')
        receipt = verify_episode(directory,binding); receipts[str(receipt_path.relative_to(ROOT))] = sha(receipt_path)
        for i,root in enumerate(receipt['metadata']['roots']):
            if root['status'] == 'complete':
                h,pack,branches=load_root(directory/f'r{i}',root['root_id'],manifest,cfg,split)
                result.append({'directory':str((directory/f'r{i}').relative_to(ROOT)),
                    'root_id':h['root_id'],'episode_id':h['episode_id'],
                    'files':{key.split('/',1)[1]:value for key,value in receipt['files'].items() if key.startswith(f'r{i}/')}})
                del h,pack,branches
    if split == 'train' and [(h['root_id'],h['episode_id']) for h in result] != [(x['root_id'],x['episode_id']) for x in d['roots']['train']]:
        raise ValueError('Training roster changed')
    return result,manifest,receipts


def evaluate(config,r,p,dataset,manifest,split):
    cfg = PredictorConfig.from_dict(p['model']); geometry = LaneConstantVelocity(ROOT/ranking.CONFIG['network'])
    models = {m:load_trained_ensemble(ROOT/e['path'],p,m,e['sha256'],e['selected']) for m,e in r['ensembles'].items()}
    rows = []
    with torch.inference_mode():
        for start in range(0,len(dataset),config['batch_roots']):
            part=[]
            for record in dataset[start:start+config['batch_roots']]:
                directory=inside(ROOT,record['directory'])
                if any(sha(directory/name)!=value for name,value in record['files'].items()):raise ValueError('Branch changed during diagnosis')
                part.append(load_root(directory,record['root_id'],manifest,cfg,split))
            hs=[x[0] for x in part]; ps=[x[1] for x in part]
            b=align_supervision(hs,ps,manifest,cfg,split=split)
            predictions={'lane_chain_constant_velocity':geometry.predict(hs,cfg)}
            predictions.update({m:model(b.observations,b.plans)['member_trajectories'].mean(0) for m,model in models.items()})
            for i,(h,pack,branches) in enumerate(part):
                rows.append(diagnose_root({m:v[i:i+1] for m,v in predictions.items()},subset(b,[i]),h,pack,branches,geometry))
    return rows,aggregate(rows,manifest[split],config['methods'])


def audit(args):
    shared.clean(); config,r,p,d,hashes=prerequisites(args.config)
    dataset,manifest,receipts=data(config,p,d,'development'); out=shared.output_dir('p5',args.run_id)
    report={**shared.metadata(),'status':'running','purpose':'development_smoke','interface_gate':False,
        'source_hashes':shared.source_hashes(),'environment':shared.env(),'inputs':hashes,'config_hash':digest(config),'receipts':receipts}
    old=torch.get_num_threads();torch.set_num_threads(1)
    try:
        rows,summary=evaluate(config,r,p,dataset,manifest,'development');write_once(out/'roots.json',rows)
        report.update(status='complete',interface_gate=True,summary=summary,roots_sha256=sha(out/'roots.json'),
                      test_opened=False,training_data_evaluated=False,ddpg_ready=False)
    except BaseException as error:
        report.update(status='failed',failure_reason=f'{type(error).__name__}: {error}');raise
    finally:
        torch.set_num_threads(old);report['finished_at']=shared.now();write_once(out/'report.json',report)
    print('task_audit='+str(out/'report.json'))


def prepare(args):
    shared.clean();config,r,p,d,hashes=prerequisites(args.config);ap=inside(ROOT,args.audit);a=read(ap)
    if (a['status']!='complete' or a['interface_gate']is not True or a['purpose']!='development_smoke'
            or a['inputs']!=hashes or a['source_hashes']!=shared.source_hashes() or a['environment']!=shared.env()
            or a['config_hash']!=digest(config) or sha(ap.parent/'roots.json')!=a['roots_sha256']):raise ValueError('Need current bounded P5c audit')
    dataset,manifest,receipts=data(config,p,d,'train')  # Verify only. No predictor execution.
    q={'config':config,'config_path':str(inside(ROOT,args.config).relative_to(ROOT)),'input_hashes':hashes,
        'receipts':receipts,'roots':[h['root_id'] for h in dataset],'manifest_hash':digest(manifest),
        'audit':str(ap.relative_to(ROOT)),'audit_sha256':sha(ap),'source_hashes':shared.source_hashes(),
        'environment':shared.env(),'test_locked':True,'ddpg_ready':False}
    out=shared.output_dir('task_diagnostics',args.run_id);write_once(out/'request.json',q)
    write_once(out/'preparation.json',{**shared.metadata(),'status':'prepared_not_evaluated','request_hash':digest(q)})
    print('task_request='+str(out/'request.json'));print('confirm_request_hash='+digest(q))


def load_request(path):
    path=inside(ROOT,path);q=read(path)
    if (q['source_hashes']!=shared.source_hashes() or q['environment']!=shared.env() or q['test_locked']is not True
            or q['ddpg_ready']is not False or read(path.parent/'preparation.json')['request_hash']!=digest(q)
            or sha(inside(ROOT,q['audit']))!=q['audit_sha256']):raise ValueError('P5c request/source changed')
    config,r,p,d,hashes=prerequisites(q['config_path'])
    if config!=q['config'] or hashes!=q['input_hashes']:raise ValueError('P5c prerequisites changed')
    return path.parent,q,r,p,d


def run(args):
    shared.clean();out,q,r,p,d=load_request(args.request)
    if args.confirm_request_hash!=digest(q):raise ValueError('Confirm exact P5c request')
    if (out/'started.json').exists() or (out/'report.json').exists():raise ValueError('Never overwrite diagnostics')
    old=torch.get_num_threads();torch.set_num_threads(1)
    try:
        with run_lock(out):
            report={**shared.metadata(),'status':'running','request_hash':digest(q),'test_evaluated':False,
                'split':'train','in_sample':True,'ddpg_ready':False,'feature_contract_frozen':False,'model_updated':False}
            write_once(out/'started.json',report)
            try:
                dataset,manifest,receipts=data(q['config'],p,d,'train')
                if receipts!=q['receipts'] or digest(manifest)!=q['manifest_hash'] or [h['root_id'] for h in dataset]!=q['roots']:
                    raise ValueError('Training data changed')
                rows,summary=evaluate(q['config'],r,p,dataset,manifest,'train');write_once(out/'roots.json',rows)
                report.update(status='complete',summary=summary,roots_sha256=sha(out/'roots.json'),next_stage='review_task_relevance_before_feature_design')
            except BaseException as error:
                report.update(status='failed',failure_reason=f'{type(error).__name__}: {error}');raise
            finally:
                report['finished_at']=shared.now();write_once(out/'report.json',report);print('task_report='+str(out/'report.json'))
    finally:torch.set_num_threads(old)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    for name in ('audit','prepare'):
        s=sub.add_parser(name);s.add_argument('--config',required=True);s.add_argument('--run-id',required=True)
        if name=='prepare':s.add_argument('--audit',required=True)
    s=sub.add_parser('run');s.add_argument('--request',required=True);s.add_argument('--confirm-request-hash',required=True)
    args=parser.parse_args();{'audit':audit,'prepare':prepare,'run':run}[args.command](args)
