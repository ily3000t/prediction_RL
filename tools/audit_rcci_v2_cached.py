"""Immutable v2 replay from saved means: no torch.load or model forward."""
import argparse
from pathlib import Path
import re
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np

import audit_rcci_feasibility as shared
from prediction_rl.data.collection_store import read_json as read, file_hash as sha, write_once, run_lock
from prediction_rl.data.dataset_contract import digest
from prediction_rl.data.frozen_dataset import inside
from prediction_rl.data.actor_adapter import build_actor_inputs
from prediction_rl.features.road_consequence_geometry import RoadConsequenceMap
from prediction_rl.features.road_consequences import contract as v1_contract, summarize_road_consequences as v1_summary
from prediction_rl.features.road_consequences_v2 import contract, summarize_road_consequences_v2
from prediction_rl.evaluation.rcci_reversal_diagnostics import SETTINGS as PARENT_SETTINGS
from prediction_rl.evaluation.rcci_v2_cached_audit import VERSION, SETTINGS, MODES, compare, aggregate

CONFIG_PATH = 'configs/development/rcci02_v2_cached_audit_v1.json'


def parent_inputs(config):
    if config != SETTINGS: raise ValueError('Only exact cached six-root settings accepted')
    path = inside(ROOT,config['parent_request']); parent = read(path)
    out = path.parent; report = read(out/'report.json'); receipt = read(out/'receipt.json')
    if (digest(parent) != config['parent_request_hash'] or receipt['request_hash'] != digest(parent)
            or report['request_hash'] != digest(parent) or report['status'] != 'complete'
            or not report['parent_result_exactly_reproduced'] or parent['config'] != PARENT_SETTINGS
            or sha(out/'report.json') != config['parent_report_sha256']
            or sha(out/'forecasts.json') != config['parent_forecasts_sha256']
            or sha(out/'diagnostics.json') != config['parent_diagnostics_sha256']
            or receipt['report_sha256'] != config['parent_report_sha256']
            or receipt['forecasts_sha256'] != config['parent_forecasts_sha256']
            or receipt['diagnostics_sha256'] != config['parent_diagnostics_sha256']
            or report['forecasts_sha256'] != config['parent_forecasts_sha256']
            or report['diagnostics_sha256'] != config['parent_diagnostics_sha256']
            or parent['adapter_contract'] != v1_contract() or parent['environment'] != shared.environment()
            or sha(inside(ROOT,parent['config_path'])) != parent['config_sha256']):
        raise ValueError('Immutable cached parent/evidence changed')
    current = shared.sources()
    if any(current.get(name) != h for name,h in parent['source_hashes'].items()):
        raise ValueError('Parent v1/Raw/inference/diagnostic source must remain unchanged')
    baseline = Path(parent['baseline_root']).resolve()
    if (baseline == ROOT.resolve() or shared.git(baseline,'rev-parse','HEAD') != shared.BASE
            or shared.git(baseline,'status','--porcelain')):
        raise ValueError('Need unchanged separate clean baseline')
    shared.verify_map_identity(baseline,shared.CONFIG['network'])
    bound = {p.relative_to(ROOT).as_posix():sha(p) for p in
             (path,out/'report.json',out/'receipt.json',out/'forecasts.json',out/'diagnostics.json',inside(ROOT,parent['config_path']))}
    for name,h in parent['parent_artifact_hashes'].items():
        if sha(inside(ROOT,name)) != h: raise ValueError('Earlier parent evidence changed')
        bound[name] = h
    # Hash existing accepted assets; never deserialize weights or targets.
    for name,h in parent['input_hashes'].items():
        if sha(inside(baseline,name)) != h: raise ValueError('Frozen source file changed')
    if [r['split'] for r in parent['roots']] != ['train']*3+['validation']*3:
        raise ValueError('Do not expand/change cached cohort')
    saved = read(out/'forecasts.json'); diagnostics = read(out/'diagnostics.json')
    ids = [r['root_id'] for r in parent['roots']]
    if ([r['root_id'] for r in saved] != ids or [r['root_id'] for r in diagnostics] != ids
            or len(set(ids)) != 6):
        raise ValueError('Cached root order/identity changed')
    return parent,bound,saved,diagnostics


def prepare(args):
    shared.clean(); cp = inside(ROOT,args.config); config = read(cp)
    parent,bound,saved,diagnostics = parent_inputs(config)
    if not re.fullmatch('[A-Za-z0-9_-]{1,24}',args.run_id): raise ValueError('Use fresh short run ID')
    out = ROOT/'artifacts/rcci02_v2'/args.run_id; out.mkdir(parents=True,exist_ok=False)
    q = {**shared.metadata(),'version':VERSION,'config':config,'config_path':cp.relative_to(ROOT).as_posix(),
        'config_sha256':sha(cp),'design_contract_sha256':sha(inside(ROOT,config['design_contract'])),
        'source_hashes':shared.sources(),'environment':shared.environment(),'adapter_contract':contract(),
        'worktree_map_sha256':sha(ROOT/shared.CONFIG['network']),
        'parent_artifact_hashes':bound,'baseline_root':parent['baseline_root'],
        'input_hashes':parent['input_hashes'],'roots':parent['roots'],
        'optimizer_seed':None,'execution_contract':'offline_saved_forecast_prefix_replay_v1'}
    write_once(out/'request.json',q)
    write_once(out/'preparation.json',{'status':'prepared_not_executed','request_hash':digest(q),
        'roots':6,'model_weight_loading_permitted':False,'model_forward_calls_permitted':0,
        'simulation_permitted':False,'training_permitted':False})
    print('rcci02_v2_request='+str(out/'request.json')); print('confirm_request_hash='+digest(q))
    return out/'request.json',digest(q)


def load_request(path,expected):
    shared.clean(); path = inside(ROOT,path); q = read(path)
    if (path.parent.parent != ROOT/'artifacts/rcci02_v2' or q['version'] != VERSION or digest(q) != expected
            or read(path.parent/'preparation.json')['request_hash'] != expected
            or q['config'] != SETTINGS or q['source_hashes'] != shared.sources()
            or q['environment'] != shared.environment() or q['adapter_contract'] != contract()
            or sha(ROOT/shared.CONFIG['network']) != q['worktree_map_sha256']
            or sha(inside(ROOT,q['config_path'])) != q['config_sha256']
            or sha(inside(ROOT,q['config']['design_contract'])) != q['design_contract_sha256']):
        raise ValueError('Cached replay request/source/contract/environment changed')
    parent,bound,saved,diagnostics = parent_inputs(q['config'])
    if (bound != q['parent_artifact_hashes'] or parent['roots'] != q['roots']
            or parent['input_hashes'] != q['input_hashes'] or parent['baseline_root'] != q['baseline_root']):
        raise ValueError('Cached cohort/input identity changed')
    if any((path.parent/name).exists() for name in ('started.json','report.json','roots.json')):
        raise ValueError('Never overwrite/retry a started cached replay')
    return path.parent,q,parent,saved,diagnostics


def run(path,expected):
    out,q,parent,saved,diagnostics = load_request(path,expected)
    with run_lock(out):
        report = {**shared.metadata(),'version':VERSION,'status':'running','request_hash':expected,
            'model_weights_loaded':False,'model_forward_calls':0,'simulation_episodes':0,'training_performed':False,
            'calibration_target_opened':False,'test_target_opened':False,'native_targets_used_for_features':False,
            'online_released':False,'prediction_accuracy_gain_established':False,'method_effect_established':False}
        write_once(out/'started.json',report)
        try:
            baseline = Path(q['baseline_root']); road = RoadConsequenceMap(ROOT/shared.CONFIG['network']); rows = []
            for index,(source,cache,reference) in enumerate(zip(q['roots'],saved,diagnostics)):
                h = read(inside(baseline,source['history']))
                if (h['root_id'] != source['root_id'] or h['episode_id'] != source['episode_id']
                        or build_actor_inputs(h['history'],road.geometry) != h['inputs']
                        or cache['mean_dtype'] != 'float32' or cache['ego_dtype'] != 'float64'
                        or cache['frame'] != 'fixed_root_ego_origin_XY' or set(cache['mean_predictions']) != set(MODES)):
                    raise ValueError('Cached observation/frame/dtype identity differs')
                methods = {}; ego = np.asarray(cache['ego_rollout'],dtype=np.float64)
                for mode in MODES:
                    neighbors = np.asarray(cache['mean_predictions'][mode],dtype=np.float32)
                    old = v1_summary(h['history'],neighbors,ego,road)
                    if old['features'].tolist() != reference['methods'][mode]['feature_vector']:
                        raise ValueError('Saved forecasts do not exactly reproduce v1')
                    new = summarize_road_consequences_v2(h['history'],neighbors,ego,road)
                    methods[mode] = {**compare(old,new),'v1_exactly_reproduced':True}
                rows.append({'root_id':source['root_id'],'episode_id':source['episode_id'],'split':source['split'],'methods':methods})
                print(f'[rcci02_v2_cached] root={index+1}/6 full_5s_unchanged=True',flush=True)
            shared.clean()
            if shared.sources() != q['source_hashes']: raise ValueError('Source changed during cached replay')
            for name,h in q['input_hashes'].items():
                if sha(inside(baseline,name)) != h: raise ValueError('Source asset changed during replay')
            for name,h in q['parent_artifact_hashes'].items():
                if sha(inside(ROOT,name)) != h: raise ValueError('Parent/cache changed during replay')
            if (shared.git(baseline,'rev-parse','HEAD') != shared.BASE or shared.git(baseline,'status','--porcelain')
                    or sha(inside(ROOT,q['config_path'])) != q['config_sha256']
                    or sha(ROOT/shared.CONFIG['network']) != q['worktree_map_sha256']
                    or sha(inside(ROOT,q['config']['design_contract'])) != q['design_contract_sha256']):
                raise ValueError('Baseline/config/contract changed during replay')
            write_once(out/'roots.json',rows)
            report.update(status='complete',v1_exactly_reproduced=True,full_5s_support_unchanged=True,
                roots_sha256=sha(out/'roots.json'),summaries=aggregate(rows),
                scope='same_six_design_roots_two_episodes_not_independent_effect_validation',
                next_stage='review_definition_and_freeze_broader_offline_validation_before_native_or_training')
        except BaseException:
            report.update(status='failed',failure_reason=traceback.format_exc()); raise
        finally:
            report['finished_at'] = shared.now(); write_once(out/'report.json',report)
    write_once(out/'receipt.json',{'request_hash':expected,'report_sha256':sha(out/'report.json'),
                                  'roots_sha256':sha(out/'roots.json')})
    print('rcci02_v2_report='+str(out/'report.json'))


def main():
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest='command',required=True)
    p = sub.add_parser('prepare'); p.add_argument('--config',default=CONFIG_PATH); p.add_argument('--run-id',required=True)
    p = sub.add_parser('run'); p.add_argument('--request',required=True); p.add_argument('--confirm-request-hash',required=True)
    args = parser.parse_args()
    if args.command == 'prepare': prepare(args)
    else: run(args.request,args.confirm_request_hash)


if __name__ == '__main__': main()
