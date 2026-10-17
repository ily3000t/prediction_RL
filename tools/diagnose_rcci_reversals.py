"""New immutable diagnostic request; never rerun/relax the parent request."""
import argparse
from pathlib import Path
import re
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import torch

import audit_rcci_frozen_predictors as frozen
from prediction_rl.data.collection_store import read_json as read, file_hash as sha, write_once, run_lock
from prediction_rl.data.dataset_contract import digest, build_branch_labels
from prediction_rl.data.frozen_dataset import inside
from prediction_rl.data.actor_adapter import build_actor_inputs
from prediction_rl.prediction.interface import PredictorConfig, collate_inputs, probe_plans
from prediction_rl.prediction.supervision import align_supervision
from prediction_rl.prediction.trained_artifacts import load_trained_ensemble
from prediction_rl.prediction.kinematic_reference import LaneConstantVelocity
from prediction_rl.prediction.candidate_ranking import rollout_ego
from prediction_rl.features.road_consequence_geometry import RoadConsequenceMap
from prediction_rl.features.road_consequences import contract
from prediction_rl.evaluation.rcci_feasibility import AuditMap, audit_branch
from prediction_rl.evaluation.rcci_reversal_diagnostics import VERSION, SETTINGS, diagnose_root, aggregate

CONFIG_PATH = 'configs/development/rcci02_reversal_diagnostics_v1.json'
shared = frozen.shared


def parent_evidence(config):
    if config != SETTINGS:
        raise ValueError('Only unchanged six-root diagnostic settings accepted')
    path = inside(ROOT,config['parent_request']); q = read(path)
    report = read(path.parent/'report.json'); rows = read(path.parent/'roots.json')
    receipt = read(path.parent/'receipt.json')
    if (digest(q) != config['parent_request_hash'] or report['status'] != 'complete'
            or sha(path.parent/'report.json') != config['parent_report_sha256']
            or sha(path.parent/'roots.json') != config['parent_roots_sha256']
            or receipt['request_hash'] != digest(q) or report['request_hash'] != digest(q)
            or receipt['report_sha256'] != config['parent_report_sha256']
            or receipt['roots_sha256'] != config['parent_roots_sha256']
            or report['roots_sha256'] != config['parent_roots_sha256']
            or q['config'] != frozen.SETTINGS or q['adapter_contract'] != contract()
            or q['environment'] != shared.environment()
            or sha(inside(ROOT,q['config_path'])) != q['config_sha256']
            or sha(ROOT/shared.CONFIG['network']) != q['worktree_map_sha256']):
        raise ValueError('Parent evidence/contract/environment changed')
    # A new request may ADD diagnostics, but all parent implementation files
    # must remain byte-for-byte LF-normalized identical. The old loader/guard
    # is untouched and is never used to execute its already-complete request.
    current = shared.sources()
    if any(current.get(name) != h for name,h in q['source_hashes'].items()):
        raise ValueError('Do not alter parent adapter/inference/audit implementation')
    if frozen.aggregate(rows) != report['summaries']:
        raise ValueError('Parent per-root aggregation differs')
    if [r['root_id'] for r in rows] != [r['root_id'] for r in q['roots']]:
        raise ValueError('Parent root order changed')
    bound = {p.relative_to(ROOT).as_posix():sha(p) for p in
             (path,path.parent/'report.json',path.parent/'roots.json',path.parent/'receipt.json',inside(ROOT,q['config_path']))}
    return q,report,rows,bound


def prerequisites(config):
    parent,report,rows,bound = parent_evidence(config)
    selected,review,hashes,training,accepted = frozen.prerequisites(parent['baseline_root'],parent['config'])
    if (selected != parent['roots'] or hashes != parent['input_hashes']
            or accepted != parent['training_acceptance'] or digest(review['manifest']) != parent['manifest_hash']):
        raise ValueError('Parent frozen source identities changed')
    return parent,report,rows,bound,review,training


def prepare(args):
    shared.clean(); cp = inside(ROOT,args.config); config = read(cp)
    parent,report,rows,bound,review,training = prerequisites(config)
    if not re.fullmatch('[A-Za-z0-9_-]{1,24}',args.run_id): raise ValueError('Use fresh short run ID')
    out = ROOT/'artifacts/rcci02_diag'/args.run_id; out.mkdir(parents=True,exist_ok=False)
    q = {**shared.metadata(),'version':VERSION,'config':config,'config_path':cp.relative_to(ROOT).as_posix(),
        'config_sha256':sha(cp),'source_hashes':shared.sources(),'environment':shared.environment(),
        'adapter_contract':contract(),'parent_artifact_hashes':bound,
        'baseline_root':parent['baseline_root'],'input_hashes':parent['input_hashes'],'roots':parent['roots'],
        'device':'cpu','dtype':'float32','optimizer_seed':None,
        'execution_contract':'offline_frozen_prediction_explanation_no_repairs_v1'}
    write_once(out/'request.json',q)
    write_once(out/'preparation.json',{'status':'prepared_not_executed','request_hash':digest(q),
        'max_ensemble_forward_calls':12,'roots':6,'simulation_permitted':False,'training_permitted':False})
    print('rcci02_diagnostic_request='+str(out/'request.json')); print('confirm_request_hash='+digest(q))
    return out/'request.json',digest(q)


def load_request(path,expected):
    shared.clean(); path = inside(ROOT,path); q = read(path)
    if (path.parent.parent != ROOT/'artifacts/rcci02_diag' or q['version'] != VERSION or digest(q) != expected
            or read(path.parent/'preparation.json')['request_hash'] != expected
            or q['config'] != SETTINGS or q['source_hashes'] != shared.sources()
            or q['environment'] != shared.environment() or q['adapter_contract'] != contract()
            or sha(inside(ROOT,q['config_path'])) != q['config_sha256']):
        raise ValueError('Diagnostic request/source/config/environment changed')
    parent,report,rows,bound,review,training = prerequisites(q['config'])
    if (bound != q['parent_artifact_hashes'] or parent['input_hashes'] != q['input_hashes']
            or parent['roots'] != q['roots'] or parent['baseline_root'] != q['baseline_root']):
        raise ValueError('Frozen parent cohort/assets changed')
    if any((path.parent/name).exists() for name in ('started.json','report.json','diagnostics.json','forecasts.json')):
        raise ValueError('Started diagnostics cannot be overwritten/retried')
    return path.parent,q,parent,report,rows,review,training


def run(path,expected):
    out,q,parent,prior_report,prior_rows,review,training = load_request(path,expected)
    previous = torch.get_num_threads(); torch.set_num_threads(1)
    try:
        with run_lock(out):
            report = {**shared.metadata(),'version':VERSION,'status':'running','request_hash':expected,
                'model_forward_calls':0,'simulation_episodes':0,'training_performed':False,
                'calibration_opened':False,'test_opened':False,'adapter_changed':False,
                'thresholds_changed':False,'model_changed':False,'online_released':False}
            write_once(out/'started.json',report)
            try:
                baseline = Path(q['baseline_root']); road = RoadConsequenceMap(ROOT/shared.CONFIG['network'])
                ego_geometry = LaneConstantVelocity(ROOT/shared.CONFIG['network'])
                native_audit = AuditMap(ROOT/shared.CONFIG['network'])
                cfg = PredictorConfig.from_dict(training['model'])
                models = {m:load_trained_ensemble(inside(baseline,e['path']),training,m,e['sha256'],e['selected'])
                          for m,e in parent['training_acceptance']['ensembles'].items()}
                diagnostics = []; captured = []; reproduced = []
                for index,r in enumerate(q['roots']):
                    hp = inside(baseline,r['history']); h = read(hp); pack = read(hp.parent/'labels.json')
                    if (h['root_id'] != r['root_id'] or h['episode_id'] != r['episode_id']
                            or inside(baseline,h['label_pack']) != hp.parent/'labels.json'
                            or h['label_pack_sha256'] != sha(hp.parent/'labels.json')):
                        raise ValueError('Observed root/labels join differs')
                    aligned = align_supervision([h],[pack],review['manifest'],cfg,split=r['split'])
                    observed = h['history']; inputs = build_actor_inputs(observed,road.geometry)
                    if inputs != h['inputs']: raise ValueError('Observed actor contract differs')
                    with torch.inference_mode():
                        batch = collate_inputs([inputs],cfg); plans = probe_plans(1,cfg)
                        predictions = {m:model(batch,plans)['mean'][0].cpu().numpy() for m,model in models.items()}
                        ego = rollout_ego(observed[-1],plans[0],ego_geometry).numpy()
                    report['model_forward_calls'] += 2
                    raw_branches = []; old_branches = []
                    for candidate in sorted(pack['candidates'],key=lambda c:c['candidate_id']):
                        cid = candidate['candidate_id']; branch = read(hp.parent/f'c{cid}.json')
                        if build_branch_labels(pack['root_traffic'],branch['trace'],candidate['labels']['requested_plan'],.2) != candidate['labels']:
                            raise ValueError('Native branch/labels differ')
                        audit_branch(branch,observed[-1],native_audit); raw_branches.append(branch)
                        old_branches.append(frozen.compare_branch(observed,branch,cid,predictions,ego,road))
                    old_row = {'root_id':r['root_id'],'episode_id':r['episode_id'],'split':r['split'],
                        'source_tensor_fingerprint':aligned.fingerprint(),'root_omitted_actors':len(inputs['root_omitted_ids']),
                        'sensitivity':frozen.sensitivity(observed,predictions,ego,road),'branches':old_branches}
                    if old_row != prior_rows[index]: raise ValueError('Parent result not exactly reproduced; do not reinterpret changed outputs')
                    reproduced.append(old_row)
                    row = diagnose_root(observed,predictions,ego,raw_branches,road)
                    row.update(root_id=r['root_id'],episode_id=r['episode_id'],split=r['split'])
                    diagnostics.append(row)
                    captured.append({'root_id':r['root_id'],'mean_dtype':'float32','ego_dtype':'float64','frame':'fixed_root_ego_origin_XY',
                        'mean_predictions':{m:p.tolist() for m,p in predictions.items()},'ego_rollout':ego.tolist()})
                    print(f'[rcci02_diagnostic] root={index+1}/6 parent_exact=True',flush=True)
                if (report['model_forward_calls'] != SETTINGS['max_model_forward_calls']
                        or frozen.aggregate(reproduced) != prior_report['summaries']):
                    raise ValueError('Bounded budget/parent reproduction differs')
                shared.clean()
                if shared.sources() != q['source_hashes']: raise ValueError('Source changed during diagnosis')
                for name,h in q['input_hashes'].items():
                    if sha(inside(baseline,name)) != h: raise ValueError('Frozen input changed during diagnosis')
                for name,h in q['parent_artifact_hashes'].items():
                    if sha(inside(ROOT,name)) != h: raise ValueError('Parent artifact changed during diagnosis')
                if (shared.git(baseline,'rev-parse','HEAD') != shared.BASE or shared.git(baseline,'status','--porcelain')
                        or sha(inside(ROOT,q['config_path'])) != q['config_sha256']
                        or sha(ROOT/shared.CONFIG['network']) != parent['worktree_map_sha256']):
                    raise ValueError('Baseline/config/map changed during diagnosis')
                write_once(out/'forecasts.json',captured); write_once(out/'diagnostics.json',diagnostics)
                report.update(status='complete',parent_result_exactly_reproduced=True,
                    parent_summary_digest=digest(prior_report['summaries']),summaries=aggregate(diagnostics),
                    forecasts_sha256=sha(out/'forecasts.json'),diagnostics_sha256=sha(out/'diagnostics.json'),
                    next_stage='review_physical_consistency_and_field_support_no_automatic_repairs')
            except BaseException:
                report.update(status='failed',failure_reason=traceback.format_exc()); raise
            finally:
                report['finished_at'] = shared.now(); write_once(out/'report.json',report)
        write_once(out/'receipt.json',{'request_hash':expected,'report_sha256':sha(out/'report.json'),
            'forecasts_sha256':sha(out/'forecasts.json'),'diagnostics_sha256':sha(out/'diagnostics.json')})
        print('rcci02_diagnostic_report='+str(out/'report.json'))
    finally:
        torch.set_num_threads(previous)


def main():
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest='command',required=True)
    p = sub.add_parser('prepare'); p.add_argument('--config',default=CONFIG_PATH); p.add_argument('--run-id',required=True)
    p = sub.add_parser('run'); p.add_argument('--request',required=True); p.add_argument('--confirm-request-hash',required=True)
    args = parser.parse_args()
    if args.command == 'prepare': prepare(args)
    else: run(args.request,args.confirm_request_hash)


if __name__ == '__main__': main()
