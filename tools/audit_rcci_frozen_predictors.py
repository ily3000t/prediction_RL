"""Immutable six-root RCCI02 audit: frozen CPU forecasts, no SUMO/training."""
import argparse
from pathlib import Path
import re
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import torch

import audit_rcci_feasibility as shared
from prediction_rl.data.collection_store import read_json as read, file_hash as sha, write_once, run_lock
from prediction_rl.data.dataset_contract import digest, build_branch_labels
from prediction_rl.data.frozen_dataset import inside, verify_historical_text_checkout
from prediction_rl.data.actor_adapter import build_actor_inputs
from prediction_rl.prediction.trained_artifacts import review_training, load_trained_ensemble
from prediction_rl.prediction.interface import PredictorConfig, collate_inputs, probe_plans
from prediction_rl.prediction.supervision import align_supervision
from prediction_rl.prediction.kinematic_reference import LaneConstantVelocity
from prediction_rl.prediction.candidate_ranking import rollout_ego
from prediction_rl.features.road_consequence_geometry import RoadConsequenceMap
from prediction_rl.features.road_consequences import contract
from prediction_rl.evaluation.rcci_feasibility import AuditMap, audit_branch
from prediction_rl.evaluation.rcci_frozen_audit import VERSION, SETTINGS, MODES, compare_branch, sensitivity, aggregate

CONFIG_PATH = 'configs/development/rcci02_frozen_audit_v1.json'
INFERENCE_FILES = ('data/actor_adapter.py','data/merge_geometry.py','prediction/model.py',
    'prediction/interface.py','prediction/trained_artifacts.py','prediction/kinematic_reference.py',
    'prediction/candidate_ranking.py','prediction/supervision.py','prediction/checkpoint.py')


def prerequisites(baseline, config):
    if config != SETTINGS:
        raise ValueError('Only exact bounded settings accepted; no unknown fields/test access')
    selected,review,hashes = shared.inputs(baseline,shared.CONFIG,'bounded')
    if len(selected) != 6 or any(sum(r['split'] == s for r in selected) != 3 for s in SETTINGS['splits']):
        raise ValueError('Need exact frozen six-root prefix')
    baseline = Path(baseline).resolve()
    for short in INFERENCE_FILES:
        name = 'src/prediction_rl/'+short
        verify_historical_text_checkout(ROOT,name,sha(inside(baseline,name)),shared.BASE)
    prior = inside(ROOT,config['prior_report'])
    if sha(prior) != config['prior_report_sha256'] or read(prior)['status'] != 'complete':
        raise ValueError('RCCI01 accepted evidence changed/missing')
    receipt = read(prior.parent/'receipt.json')
    if (receipt['report_sha256'] != sha(prior) or receipt['roots_sha256'] != sha(prior.parent/'roots.json')
            or read(prior)['roots_sha256'] != receipt['roots_sha256']):
        raise ValueError('RCCI01 output receipt changed')
    training = read(inside(baseline,shared.CONFIG['predictor_request']))
    acceptance = review_training(baseline,shared.CONFIG['predictor_request'],config['training_report'])
    if acceptance['request_hash'] != digest(training) or set(acceptance['ensembles']) != set(MODES):
        raise ValueError('Frozen training acceptance changed')
    for name,h in acceptance['input_hashes'].items():
        path = inside(baseline,name)
        if sha(path) != h: raise ValueError('Accepted training input changed')
        hashes[path.relative_to(baseline).as_posix()] = h
    for mode,entry in acceptance['ensembles'].items():
        if entry['sha256'] != shared.PREDICTORS[mode]:
            raise ValueError('Do not substitute/reselect frozen model')
    return selected,review,hashes,training,acceptance


def prepare(args):
    shared.clean(); cp = inside(ROOT,args.config); config = read(cp)
    selected,review,hashes,training,accepted = prerequisites(args.baseline_root,config)
    if not re.fullmatch('[A-Za-z0-9_-]{1,24}',args.run_id):
        raise ValueError('Use short fresh run ID')
    out = ROOT/'artifacts/rcci02'/args.run_id
    out.mkdir(parents=True,exist_ok=False)
    request = {**shared.metadata(),'version':VERSION,'config':config,
        'config_path':cp.relative_to(ROOT).as_posix(),'config_sha256':sha(cp),
        'baseline_root':str(Path(args.baseline_root).resolve()),'baseline_commit':shared.BASE,
        'source_hashes':shared.sources(),'environment':shared.environment(),
        'adapter_contract':contract(),'roots':selected,'input_hashes':hashes,
        'training_acceptance':accepted,'manifest_hash':digest(review['manifest']),
        'worktree_map_sha256':sha(ROOT/shared.CONFIG['network']),
        'prior_report_sha256':sha(ROOT/config['prior_report']),
        'execution_contract':'offline_blocking_frozen_mean_predictor_audit_v1',
        'optimizer_seed':None,'device':'cpu','dtype':'float32','max_model_forward_calls':12,
        'native_future':'offline_comparison_only_never_adapter_or_model_inputs'}
    write_once(out/'request.json',request)
    write_once(out/'preparation.json',{'status':'prepared_not_executed','request_hash':digest(request),
        'roots':6,'predictor_inference_permitted':True,'simulation_permitted':False,'training_permitted':False})
    print('rcci02_request='+str(out/'request.json')); print('confirm_request_hash='+digest(request))
    return out/'request.json',digest(request)


def load_request(path, expected):
    shared.clean(); path = inside(ROOT,path); q = read(path)
    if (path.parent.parent != ROOT/'artifacts/rcci02' or q['version'] != VERSION or digest(q) != expected
            or read(path.parent/'preparation.json')['request_hash'] != expected
            or q['source_hashes'] != shared.sources() or q['environment'] != shared.environment()
            or q['adapter_contract'] != contract() or q['config'] != SETTINGS
            or sha(inside(ROOT,q['config_path'])) != q['config_sha256']
            or q['worktree_map_sha256'] != sha(ROOT/shared.CONFIG['network'])
            or q['prior_report_sha256'] != sha(ROOT/q['config']['prior_report'])):
        raise ValueError('Request/source/config/environment/evidence changed')
    selected,review,hashes,training,accepted = prerequisites(q['baseline_root'],q['config'])
    if (selected != q['roots'] or hashes != q['input_hashes'] or accepted != q['training_acceptance']
            or digest(review['manifest']) != q['manifest_hash']):
        raise ValueError('Frozen cohort/asset lineage changed')
    if any((path.parent/name).exists() for name in ('started.json','report.json','roots.json')):
        raise ValueError('Never overwrite/retry a started audit; preserve failure and use new version/run ID')
    return path.parent,q,review,training


def run(path, expected):
    out,q,review,training = load_request(path,expected)
    previous = torch.get_num_threads(); torch.set_num_threads(1)
    try:
        with run_lock(out):
            report = {**shared.metadata(),'version':VERSION,'status':'running','request_hash':expected,
                'simulation_episodes':0,'training_performed':False,'predictor_inference_performed':False,
                'test_opened':False,'calibration_opened':False,'online_adapter_released':False,
                'method_effect_established':False,'model_forward_calls':0,'scope':'six_roots_two_episodes_not_representative'}
            write_once(out/'started.json',report)
            try:
                baseline = Path(q['baseline_root']); network = ROOT/shared.CONFIG['network']
                road = RoadConsequenceMap(network); native_audit = AuditMap(network)
                ego_geometry = LaneConstantVelocity(network)
                cfg = PredictorConfig.from_dict(training['model'])
                models = {mode:load_trained_ensemble(inside(baseline,entry['path']),training,mode,
                          entry['sha256'],entry['selected']) for mode,entry in q['training_acceptance']['ensembles'].items()}
                rows = []
                for r in q['roots']:
                    hp = inside(baseline,r['history']); h = read(hp); pack = read(hp.parent/'labels.json')
                    if (h['root_id'] != r['root_id'] or h['episode_id'] != r['episode_id']
                            or inside(baseline,h['label_pack']) != hp.parent/'labels.json'
                            or h['label_pack_sha256'] != sha(hp.parent/'labels.json')):
                        raise ValueError('Root/history/label join changed')
                    aligned = align_supervision([h],[pack],review['manifest'],cfg,split=r['split'])
                    observed = h['history']
                    inputs = build_actor_inputs(observed,road.geometry)
                    if inputs != h['inputs']: raise ValueError('Rebuilt root actor inputs differ from frozen source')
                    # Forecast only from rebuilt observation tensors and frozen probes.
                    # Neither aligned.targets/masks nor native branch data are passed.
                    batch = collate_inputs([inputs],cfg); plans = probe_plans(1,cfg)
                    with torch.inference_mode():
                        predictions = {mode:model(batch,plans)['mean'][0].cpu().numpy() for mode,model in models.items()}
                        ego = rollout_ego(observed[-1],plans[0],ego_geometry).numpy()
                    report['predictor_inference_performed'] = True; report['model_forward_calls'] += 2
                    branches = []
                    for candidate in sorted(pack['candidates'],key=lambda c:c['candidate_id']):
                        cid = candidate['candidate_id']; branch = read(hp.parent/f'c{cid}.json')
                        if build_branch_labels(pack['root_traffic'],branch['trace'],candidate['labels']['requested_plan'],.2) != candidate['labels']:
                            raise ValueError('Native branch/label mismatch')
                        audit_branch(branch,observed[-1],native_audit)
                        branches.append(compare_branch(observed,branch,cid,predictions,ego,road))
                    rows.append({'root_id':r['root_id'],'episode_id':r['episode_id'],'split':r['split'],
                        'source_tensor_fingerprint':aligned.fingerprint(),
                        'root_omitted_actors':len(inputs['root_omitted_ids']),
                        'sensitivity':sensitivity(observed,predictions,ego,road),'branches':branches})
                    print(f'[rcci02] root={len(rows)}/6 split={r["split"]}',flush=True)
                if report['model_forward_calls'] != q['max_model_forward_calls']:
                    raise ValueError('Bounded forward budget changed')
                # Recheck exact selected files/assets before sealing results.
                shared.clean()
                if shared.sources() != q['source_hashes']: raise ValueError('Source changed during audit')
                for name,h in q['input_hashes'].items():
                    if sha(inside(baseline,name)) != h: raise ValueError('Input changed during audit')
                if (shared.git(baseline,'rev-parse','HEAD') != shared.BASE or shared.git(baseline,'status','--porcelain')
                        or sha(inside(ROOT,q['config_path'])) != q['config_sha256']
                        or sha(network) != q['worktree_map_sha256']
                        or sha(ROOT/q['config']['prior_report']) != q['prior_report_sha256']):
                    raise ValueError('Baseline/config/map/evidence changed during audit')
                write_once(out/'roots.json',rows)
                report.update(status='complete',roots_sha256=sha(out/'roots.json'),summaries=aggregate(rows),
                    next_stage='review_coverage_and_errors_before_any_native_smoke_or_training',
                    native_reference='known_lane_progress_with_road_aligned_control_volume_not_collision_truth',
                    error_support='joint_B2_B3_observed_prefix_projection_support_no_tail_imputation')
            except BaseException:
                report.update(status='failed',failure_reason=traceback.format_exc()); raise
            finally:
                report['finished_at'] = shared.now(); write_once(out/'report.json',report)
        write_once(out/'receipt.json',{'request_hash':expected,'report_sha256':sha(out/'report.json'),
                                      'roots_sha256':sha(out/'roots.json')})
        print('rcci02_report='+str(out/'report.json'))
    finally:
        torch.set_num_threads(previous)


def main():
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest='command',required=True)
    p = sub.add_parser('prepare'); p.add_argument('--baseline-root',required=True)
    p.add_argument('--config',default=CONFIG_PATH); p.add_argument('--run-id',required=True)
    p = sub.add_parser('run'); p.add_argument('--request',required=True); p.add_argument('--confirm-request-hash',required=True)
    p = sub.add_parser('audit'); p.add_argument('--baseline-root',required=True); p.add_argument('--run-id',required=True)
    args = parser.parse_args()
    if args.command == 'prepare': prepare(args)
    elif args.command == 'run': run(args.request,args.confirm_request_hash)
    else:
        args.config = CONFIG_PATH; path,h = prepare(args); run(path,h)


if __name__ == '__main__': main()
