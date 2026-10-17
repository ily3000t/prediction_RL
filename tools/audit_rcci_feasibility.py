"""RCCI01: user-run offline audit of frozen train/validation metadata only."""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import re
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from prediction_rl.data.collection_store import read_json as read, file_hash as sha, write_once, run_lock
from prediction_rl.data.dataset_contract import digest, build_branch_labels
from prediction_rl.data.frozen_dataset import inside, verify_historical_text_checkout
from prediction_rl.prediction.interface import PredictorConfig
from prediction_rl.prediction.supervision import align_supervision
from prediction_rl.evaluation.rcci_feasibility import VERSION, CONTRACT, AuditMap, AuditAccumulator, audit_branch, summarize

BASE = '75051acc06f6ea2ab6dda9be678f191b63d04569'
PREDICTORS = {'ordinary':'939242f358315bd00ce89978422e4b55b8059f41837463d2e80f661c63817e8d',
              'conditional':'daddeda8569b88c2a3482a0e8c739995785ee0f8216dda6525c86d2ce120e25f'}
CONFIG = CONTRACT


def git(root,*args): return subprocess.check_output(['git','-C',str(root),*args],text=True,encoding='utf-8').strip()
def clean():
    if git(ROOT,'status','--porcelain'): raise ValueError('Commit clean RCCI source before preparing/running')
def sources():
    return {p.relative_to(ROOT).as_posix():digest(p.read_text(encoding='utf-8').replace('\r\n','\n'))
            for directory in ('src','tools') for p in sorted((ROOT/directory).rglob('*.py'))}
def now(): return datetime.now(timezone.utc).isoformat()
def environment():
    versions = {}
    for name in ('numpy','torch','autonomous-learning-library','traci','sumolib'):
        try: versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: versions[name] = None
    return {'python':sys.version, 'executable':sys.executable, 'os':platform.platform(), 'packages':versions}
def metadata():
    return {'git_commit':git(ROOT,'rev-parse','HEAD'),'working_tree_dirty':False,
            'command':subprocess.list2cmdline([sys.executable,*sys.argv]),'timestamp':now()}


def select_roots(review, scope):
    if scope not in ('bounded','full'): raise ValueError('Unknown scope')
    rows = []; episodes = []
    for split in ('train','validation'):
        cohort = review['roots'][split]
        if scope == 'bounded': cohort = cohort[:3]
        if not cohort: raise ValueError('Missing split')
        rows.extend([{**r,'split':split} for r in cohort])
        episodes.append(set(review['manifest'][split]))
        if any(r['episode_id'] not in episodes[-1] for r in cohort): raise ValueError('Wrong split episode')
    if episodes[0]&episodes[1] or len({r['root_id'] for r in rows}) != len(rows):
        raise ValueError('Cross-split leakage/duplicate root')
    return rows


def verify_map_identity(baseline, relative):
    # Bind BOTH checkouts to the exact base Git blob. Permit Git LF/CRLF only,
    # never XML reserialization, changed whitespace/attributes or changed maps.
    recorded=sha(inside(baseline,relative))
    verify_historical_text_checkout(baseline,relative,recorded,BASE)
    verify_historical_text_checkout(ROOT,relative,recorded,BASE)
    return sha(ROOT/relative)


def inputs(baseline, config, scope):
    baseline = Path(baseline).resolve()
    if baseline == ROOT.resolve() or baseline.is_relative_to(ROOT.resolve()) or ROOT.resolve().is_relative_to(baseline):
        raise ValueError('Read-only baseline must be a separate project')
    if git(baseline,'rev-parse','HEAD') != BASE or git(baseline,'status','--porcelain'):
        raise ValueError('Need unchanged clean 75051acc baseline; never alter its guards')
    if config != CONFIG or config['version'] != VERSION or config['splits'] != ['train','validation']:
        raise ValueError('Unknown audit config; no calibration/test data access')
    review_path = inside(baseline,config['dataset_review'])
    if sha(review_path) != config['dataset_review_sha256']: raise ValueError('Frozen data review changed')
    review = read(review_path); collection = read(inside(baseline,config['collection_request']))
    training = read(inside(baseline,config['predictor_request']))
    if (review['status'] != 'complete' or not review['test_locked']
            or digest(collection) != review['collection_request_hash']
            or training['dataset_review_sha256'] != sha(review_path) or not training['test_locked']):
        raise ValueError('Dataset/predictor lineage mismatch')
    if any((inside(baseline,config[k]).parent/'writer.lock').exists()
           for k in ('dataset_review','collection_request','predictor_request')):
        raise ValueError('Source collection/review/training is active')
    selected = select_roots(review,scope); hashes = {}
    def record(relative):
        p = inside(baseline,relative); hashes[p.relative_to(baseline).as_posix()] = sha(p)
    for key in ('dataset_review','collection_request','predictor_request','network'): record(config[key])
    verify_map_identity(baseline,config['network'])
    for name,expected in PREDICTORS.items():
        asset=f'artifacts/training/predictor_v1/{name}_ensemble.pt'; record(asset)
        if hashes[asset] != expected: raise ValueError('Frozen predictor checksum changed')
    receipts = {}
    for r in selected:
        directory = inside(baseline,r['history']).parent
        ep = directory.parent; receipt_path=ep/'complete.json'; rel=receipt_path.relative_to(baseline)
        expected = review['input_hashes'].get(str(rel))
        if expected is None: expected = review['input_hashes'].get(rel.as_posix())
        if expected != sha(receipt_path): raise ValueError('Untrusted/changed collection receipt')
        receipt=read(receipt_path)
        if receipt['status']!='complete' or receipt['binding']['request_hash']!=digest(collection):
            raise ValueError('Collection episode lineage changed')
        source_job=receipt['binding']['job']
        if source_job['episode_id']!=r['episode_id'] or source_job['split']!=r['split']:
            raise ValueError('Source episode/split identity changed')
        r['collection_job']=source_job
        receipts[rel.as_posix()] = expected; record(rel)
        if (ep/'writer.lock').exists(): raise ValueError('Active source episode')
        names=['history.json','labels.json']+[f'c{i}.json' for i in range(5)]
        for name in names:
            p=directory/name; relative=p.relative_to(ep).as_posix()
            if receipt['files'].get(relative)!=sha(p): raise ValueError('Selected source file changed: '+relative)
            record(p.relative_to(baseline))
        if (r['history_sha256']!=sha(directory/'history.json') or r['labels_sha256']!=sha(directory/'labels.json')):
            raise ValueError('Review/root file mismatch')
    return selected,review,hashes


def prepare(args):
    clean(); cp=inside(ROOT,args.config); config=read(cp)
    selected,review,hashes=inputs(args.baseline_root,config,args.scope)
    if not re.fullmatch('[A-Za-z0-9_-]{1,24}',args.run_id): raise ValueError('Use short run ID')
    out=ROOT/'artifacts/rcci01'/args.run_id; out.mkdir(parents=True,exist_ok=False)
    q={**metadata(),'version':VERSION,'config':config,'config_path':str(cp.relative_to(ROOT)),
       'config_sha256':sha(cp),'baseline_root':str(Path(args.baseline_root).resolve()),'baseline_commit':BASE,
       'scope':args.scope,'roots':selected,'input_hashes':hashes,'source_hashes':sources(),
       'environment':environment(),'manifest_hash':digest(review['manifest']),
       'worktree_map_sha256':sha(ROOT/config['network']),
       'execution_contract':'offline_readonly_metadata_audit_v1','optimizer_seed':None,
       'future_lane_ids':'offline_labels_only_not_online_projection_inputs'}
    write_once(out/'request.json',q)
    write_once(out/'preparation.json',{'status':'prepared_not_executed','request_hash':digest(q),'roots':len(selected),
        'simulation_permitted':False,'predictor_inference_permitted':False,'training_permitted':False})
    print('rcci01_request='+str(out/'request.json'));print('confirm_request_hash='+digest(q))
    return out/'request.json',digest(q)


def run(path, expected):
    clean(); path=inside(ROOT,path); q=read(path); root=path.parent
    if (path.parent.parent!=ROOT/'artifacts/rcci01' or q['version']!=VERSION or digest(q)!=expected
            or read(root/'preparation.json')['request_hash']!=expected or q['source_hashes']!=sources()
            or q['environment']!=environment() or sha(inside(ROOT,q['config_path']))!=q['config_sha256']
            or q['worktree_map_sha256']!=sha(ROOT/q['config']['network'])):
        raise ValueError('Request/source/config/environment changed')
    selected,review,hashes=inputs(q['baseline_root'],q['config'],q['scope'])
    if selected!=q['roots'] or hashes!=q['input_hashes'] or digest(review['manifest'])!=q['manifest_hash']:
        raise ValueError('Frozen cohort/input identities changed')
    if (root/'started.json').exists() or (root/'report.json').exists(): raise ValueError('Never overwrite or retry failed audit')
    with run_lock(root):
        report={**metadata(),'status':'running','request_hash':expected,'scope':q['scope'],
            'test_opened':False,'calibration_opened':False,'training_performed':False,
            'predictor_inference_performed':False,'simulation_episodes':0,
            'model_reuse_verified':'checksum_only_not_runtime_load',
            'online_adapter_released':False,'method_effect_established':False}
        write_once(root/'started.json',report)
        try:
            baseline=Path(q['baseline_root']); geometry=AuditMap(inside(baseline,q['config']['network'])); rows=[]
            cfg=PredictorConfig.from_dict(review['protocol']['model'])
            accumulators={s:AuditAccumulator() for s in ('train','validation')}
            for i,r in enumerate(selected):
                hp=inside(baseline,r['history']); directory=hp.parent
                h=read(hp); pack=read(directory/'labels.json')
                if (h['root_id']!=r['root_id'] or h['episode_id']!=r['episode_id']
                        or inside(baseline,h['label_pack'])!=directory/'labels.json'
                        or h['label_pack_sha256']!=sha(directory/'labels.json')):
                    raise ValueError('Root/history/label join changed')
                align_supervision([h],[pack],review['manifest'],cfg,split=r['split'])
                branches=[]
                for candidate in sorted(pack['candidates'],key=lambda c:c['candidate_id']):
                    branch=read(directory/f'c{candidate["candidate_id"]}.json')
                    if build_branch_labels(pack['root_traffic'],branch['trace'],candidate['labels']['requested_plan'],.2)!=candidate['labels']:
                        raise ValueError('Labels/raw branch mismatch')
                    branches.append(audit_branch(branch,h['history'][-1],geometry))
                row={'root_id':r['root_id'],'episode_id':r['episode_id'],'split':r['split'],
                    'omitted_actors':len(h['inputs']['root_omitted_ids']),
                    'ego_history_fraction':sum(h['inputs']['ego_history_mask'])/11,
                    'root_states':geometry.audit_frame(h['history'][-1]),'branches':branches}
                accumulators[r['split']].add(row)
                rows.append({k:row[k] for k in ('root_id','episode_id','split','omitted_actors','ego_history_fraction')}
                    | {'statistics':summarize([row],q['config']['diagnostic_distance_bins_m']),
                       'branches':[{k:v for k,v in b.items() if k!='states'} for b in branches]})
                if (i+1)%20==0: print(f'[rcci01] roots={i+1}/{len(selected)}',flush=True)
            write_once(root/'roots.json',rows)
            report.update(status='complete',roots_sha256=sha(root/'roots.json'),
                summaries={s:accumulators[s].report(q['config']['diagnostic_distance_bins_m'])
                           for s in ('train','validation')},
                next_stage='review_RCCI01_before_freezing_RCCI02_contract')
        except BaseException:
            report.update(status='failed',failure_reason=traceback.format_exc());raise
        finally:
            report['finished_at']=now();write_once(root/'report.json',report)
    write_once(root/'receipt.json',{'request_hash':expected,'report_sha256':sha(root/'report.json'),
                                   'roots_sha256':sha(root/'roots.json')})
    print('rcci01_report='+str(root/'report.json'))


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('prepare');a.add_argument('--baseline-root',required=True)
    a.add_argument('--config',default='configs/development/rcci01_feasibility_v1.json')
    a.add_argument('--scope',choices=['bounded','full'],default='bounded');a.add_argument('--run-id',required=True)
    a=sub.add_parser('run');a.add_argument('--request',required=True);a.add_argument('--confirm-request-hash',required=True)
    a=sub.add_parser('audit');a.add_argument('--baseline-root',required=True);a.add_argument('--run-id',required=True)
    args=p.parse_args()
    if args.command=='prepare':prepare(args)
    elif args.command=='run':run(args.request,args.confirm_request_hash)
    else:
        args.config='configs/development/rcci01_feasibility_v1.json';args.scope='bounded'
        path,h=prepare(args);run(path,h)


if __name__=='__main__': main()
