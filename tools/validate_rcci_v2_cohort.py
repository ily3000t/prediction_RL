"""User-run full existing train/validation RCCI v2 audit; no SUMO/training."""
import argparse
from collections import Counter
from pathlib import Path
import re
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import torch

import audit_rcci_feasibility as shared
import audit_rcci_v2_cached as cached
from prediction_rl.data.collection_store import read_json as read, file_hash as sha, write_once, run_lock
from prediction_rl.data.dataset_contract import digest, build_branch_labels
from prediction_rl.data.frozen_dataset import inside
from prediction_rl.data.actor_adapter import build_actor_inputs
from prediction_rl.prediction.interface import PredictorConfig, collate_inputs, probe_plans
from prediction_rl.prediction.trained_artifacts import load_trained_ensemble
from prediction_rl.prediction.candidate_ranking import rollout_ego
from prediction_rl.prediction.kinematic_reference import LaneConstantVelocity
from prediction_rl.features.road_consequence_geometry import RoadConsequenceMap
from prediction_rl.features.road_consequences import summarize_road_consequences as v1_summary
from prediction_rl.features.road_consequences_v2 import contract, summarize_road_consequences_v2
from prediction_rl.evaluation.rcci_v2_cached_audit import SETTINGS as CACHED_SETTINGS, compare
from prediction_rl.evaluation.rcci_v2_cohort import (
    VERSION, SETTINGS, MODES, project_predictions, compare_native, mechanism, aggregate,
)

CONFIG_PATH = 'configs/development/rcci02_v2_cohort_validation_v1.json'


def frozen_review_parent(ancestor):
    # The reversal request intentionally did not duplicate the original model
    # review. Follow its already hash-bound frozen-audit parent, not a guessed
    # field on the intermediate request, and do not rerun the old loader.
    path = inside(ROOT,ancestor['config']['parent_request'])
    if ancestor['parent_artifact_hashes'].get(path.relative_to(ROOT).as_posix()) != sha(path):
        raise ValueError('Frozen model-review parent is not hash-bound')
    frozen = read(path)
    if (digest(frozen) != ancestor['config']['parent_request_hash']
            or frozen['input_hashes'] != ancestor['input_hashes']
            or frozen['baseline_root'] != ancestor['baseline_root']):
        raise ValueError('Frozen model-review parent identity differs')
    return frozen


def inherited_training_acceptance(ancestor):
    return frozen_review_parent(ancestor)['training_acceptance']


def accepted_full_cohort(ancestor):
    frozen = frozen_review_parent(ancestor)
    path = inside(ROOT,frozen['config']['prior_report']); directory = path.parent
    report = read(path); receipt = read(directory/'receipt.json'); request = read(directory/'request.json')
    if (sha(path) != frozen['config']['prior_report_sha256'] or report['status'] != 'complete'
            or request['scope'] != 'full' or request['baseline_root'] != ancestor['baseline_root']
            or receipt['request_hash'] != digest(request) or report['request_hash'] != digest(request)
            or receipt['report_sha256'] != sha(path) or receipt['roots_sha256'] != sha(directory/'roots.json')
            or report['roots_sha256'] != sha(directory/'roots.json')):
        raise ValueError('Accepted full cohort/report/receipt changed')
    return read(directory/'roots.json'),{p.relative_to(ROOT).as_posix():sha(p) for p in
        (path,directory/'request.json',directory/'receipt.json',directory/'roots.json')}


def prerequisites(config, baseline):
    if config != SETTINGS: raise ValueError('Only exact frozen train/validation cohort accepted')
    pp = inside(ROOT,config['parent_request']); parent = read(pp); out = pp.parent
    receipt = read(out/'receipt.json'); report = read(out/'report.json')
    if (digest(parent) != config['parent_request_hash'] or parent['config'] != CACHED_SETTINGS
            or receipt['request_hash'] != digest(parent) or report['request_hash'] != digest(parent)
            or sha(out/'report.json') != config['parent_report_sha256']
            or sha(out/'roots.json') != config['parent_roots_sha256']
            or receipt['report_sha256'] != config['parent_report_sha256']
            or receipt['roots_sha256'] != config['parent_roots_sha256']
            or report['roots_sha256'] != config['parent_roots_sha256']
            or report['status'] != 'complete' or not report['v1_exactly_reproduced']
            or not report['full_5s_support_unchanged'] or parent['adapter_contract'] != contract()
            or parent['source_hashes'].items() - shared.sources().items()
            or parent['environment'] != shared.environment()
            or sha(inside(ROOT,parent['config_path'])) != parent['config_sha256']
            or sha(inside(ROOT,parent['config']['design_contract'])) != parent['design_contract_sha256']):
        raise ValueError('Accepted v2 cache lineage changed')
    ancestor,bound,_,_ = cached.parent_inputs(CACHED_SETTINGS)
    baseline = Path(baseline).resolve()
    if baseline != Path(ancestor['baseline_root']).resolve(): raise ValueError('Do not substitute baseline')
    bound.update({p.relative_to(ROOT).as_posix():sha(p) for p in
        (pp,out/'report.json',out/'receipt.json',out/'roots.json',inside(ROOT,parent['config_path']),
         inside(ROOT,parent['config']['design_contract']))})
    selected,review,hashes = shared.inputs(baseline,shared.CONFIG,'full')
    counts = Counter(r['split'] for r in selected)
    if (dict(counts) != config['expected_roots'] or len({r['episode_id'] for r in selected}) != config['expected_episodes']
            or len(selected)*len(MODES) != config['max_ensemble_forward_calls']
            or config['max_member_forward_calls'] != config['max_ensemble_forward_calls']*3):
        raise ValueError('Frozen cohort/budget changed')
    old_full,full_bound = accepted_full_cohort(ancestor); bound.update(full_bound)
    if [(r['root_id'],r['episode_id'],r['split']) for r in selected] != [
            (r['root_id'],r['episode_id'],r['split']) for r in old_full]:
        raise ValueError('Full cohort order differs from accepted RCCI01')
    acceptance = inherited_training_acceptance(ancestor)
    if set(acceptance['ensembles']) != set(MODES): raise ValueError('Selected model set changed')
    for name,h in acceptance['input_hashes'].items():
        p = inside(baseline,name)
        if sha(p) != h: raise ValueError('Accepted training receipt changed')
        hashes[p.relative_to(baseline).as_posix()] = h
    for mode,entry in acceptance['ensembles'].items():
        if entry['sha256'] != shared.PREDICTORS[mode]: raise ValueError('Model replacement forbidden')
    training = read(inside(baseline,shared.CONFIG['predictor_request']))
    if digest(training) != acceptance['request_hash']: raise ValueError('Training identity changed')
    design_episodes = sorted({r['episode_id'] for r in parent['roots']})
    if len(design_episodes) != 2: raise ValueError('Design episode markers changed')
    return selected,review,hashes,bound,acceptance,training,design_episodes


def prepare(args):
    shared.clean(); cp = inside(ROOT,args.config); config = read(cp)
    roots,review,hashes,bound,accepted,_,design_episodes = prerequisites(config,args.baseline_root)
    if not re.fullmatch('[A-Za-z0-9_-]{1,24}',args.run_id): raise ValueError('Use fresh short run ID')
    out = ROOT/'artifacts/rcci02_v2_cohort'/args.run_id; out.mkdir(parents=True,exist_ok=False)
    q = {**shared.metadata(),'version':VERSION,'config':config,'config_path':cp.relative_to(ROOT).as_posix(),
        'config_sha256':sha(cp),'design_contract_sha256':sha(inside(ROOT,config['design_contract'])),
        'baseline_root':str(Path(args.baseline_root).resolve()),'baseline_commit':shared.BASE,
        'source_hashes':shared.sources(),'environment':shared.environment(),'adapter_contract':contract(),
        'input_hashes':hashes,'parent_artifact_hashes':bound,'training_acceptance':accepted,
        'manifest_hash':digest(review['manifest']),'roots':roots,'design_episodes':design_episodes,
        'worktree_map_sha256':sha(ROOT/shared.CONFIG['network']), 'optimizer_seed':None,
        'execution_contract':'offline_blocking_full_existing_cohort_v2_validation_v1'}
    write_once(out/'request.json',q)
    write_once(out/'preparation.json',{'status':'prepared_not_executed','request_hash':digest(q),
        'roots':len(roots),'ensemble_forward_calls_permitted':config['max_ensemble_forward_calls'],
        'simulation_permitted':False,'training_permitted':False,'calibration_test_access':False})
    print('rcci02_v2_cohort_request='+str(out/'request.json')); print('confirm_request_hash='+digest(q))
    return out/'request.json',digest(q)


def load_request(path,expected):
    shared.clean(); path = inside(ROOT,path); q = read(path)
    if (path.parent.parent != ROOT/'artifacts/rcci02_v2_cohort' or q['version'] != VERSION or digest(q) != expected
            or read(path.parent/'preparation.json')['request_hash'] != expected
            or q['config'] != SETTINGS or q['source_hashes'] != shared.sources()
            or q['environment'] != shared.environment() or q['adapter_contract'] != contract()
            or sha(inside(ROOT,q['config_path'])) != q['config_sha256']
            or sha(inside(ROOT,q['config']['design_contract'])) != q['design_contract_sha256']
            or sha(ROOT/shared.CONFIG['network']) != q['worktree_map_sha256']):
        raise ValueError('Cohort request/source/environment/contract changed')
    if any((path.parent/name).exists() for name in ('started.json','report.json','receipt.json','roots')):
        raise ValueError('Never overwrite or retry a started cohort audit')
    roots,review,hashes,bound,accepted,training,episodes = prerequisites(q['config'],q['baseline_root'])
    if (roots != q['roots'] or hashes != q['input_hashes'] or bound != q['parent_artifact_hashes']
            or accepted != q['training_acceptance'] or episodes != q['design_episodes']
            or digest(review['manifest']) != q['manifest_hash']):
        raise ValueError('Frozen cohort/model/input lineage changed')
    return path.parent,q,training


def infer(history, road, ego_geometry, cfg, models):
    """Observation-only whitelist, no branch objects/labels in this signature."""
    inputs = build_actor_inputs(history,road.geometry)
    batch = collate_inputs([inputs],cfg); plans = probe_plans(1,cfg)
    with torch.inference_mode():
        predictions = {m:models[m](batch,plans)['mean'][0].cpu().numpy() for m in MODES}
        ego = rollout_ego(history[-1],plans[0],ego_geometry).numpy()
    return inputs,predictions,ego


def evaluate_features(history, predictions, ego, road):
    """Complete all feature and substitution work before opening native labels."""
    coverage = {}; failures = {}
    projected = project_predictions(history,predictions,ego,road)
    for mode in MODES:
        old = v1_summary(history,predictions[mode],ego,road)
        new = summarize_road_consequences_v2(history,predictions[mode],ego,road)
        coverage[mode] = compare(old,new)
        reasons = Counter()
        for candidate in new['candidates']:
            for prefix in [candidate['ego_prefix'],*candidate['actor_prefixes'].values()]:
                if not prefix['full_5s']: reasons[prefix['first_unavailable_reason']] += 1
        failures[mode] = dict(reasons)
    return projected,coverage,failures,mechanism(history,projected,road)


def run(path,expected):
    out,q,training = load_request(path,expected)
    previous = torch.get_num_threads(); torch.set_num_threads(1)
    try:
        with run_lock(out):
            report = {**shared.metadata(),'version':VERSION,'status':'running','request_hash':expected,
                'ensemble_forward_calls':0,'member_forward_calls':0,'simulation_episodes':0,'training_performed':False,
                'test_opened':False,'calibration_opened':False,'online_released':False,
                'prediction_accuracy_gain_established':False,'method_effect_established':False,
                'native_targets_used_for_features':False,'independent_confirmation':False}
            write_once(out/'started.json',report)
            try:
                baseline = Path(q['baseline_root']); network = ROOT/shared.CONFIG['network']
                road = RoadConsequenceMap(network); ego_geometry = LaneConstantVelocity(network)
                cfg = PredictorConfig.from_dict(training['model'])
                models = {m:load_trained_ensemble(inside(baseline,e['path']),training,m,e['sha256'],e['selected'])
                          for m,e in q['training_acceptance']['ensembles'].items()}
                rows = []; root_files = {}
                for index,r in enumerate(q['roots']):
                    hp = inside(baseline,r['history']); h = read(hp)
                    if h['root_id'] != r['root_id'] or h['episode_id'] != r['episode_id']:
                        raise ValueError('Root observation identity changed')
                    inputs,predictions,ego = infer(h['history'],road,ego_geometry,cfg,models)
                    report['ensemble_forward_calls'] += len(MODES); report['member_forward_calls'] += len(MODES)*3
                    if inputs != h['inputs']: raise ValueError('Rebuilt actor input differs from frozen source')
                    projected,coverage,failures,mechanisms = evaluate_features(h['history'],predictions,ego,road)
                    # Native branch files are opened ONLY after prediction features are complete.
                    pack = read(hp.parent/'labels.json')
                    if (inside(baseline,h['label_pack']) != hp.parent/'labels.json'
                            or sha(hp.parent/'labels.json') != h['label_pack_sha256']
                            or sorted(c['candidate_id'] for c in pack['candidates']) != list(range(5))):
                        raise ValueError('Root/label pack join changed')
                    branches = []
                    for candidate in sorted(pack['candidates'],key=lambda c:c['candidate_id']):
                        cid = candidate['candidate_id']; branch = read(hp.parent/f'c{cid}.json')
                        if build_branch_labels(pack['root_traffic'],branch['trace'],candidate['labels']['requested_plan'],.2) != candidate['labels']:
                            raise ValueError('Frozen native branch/labels differ')
                        branches.append(compare_native(h['history'],branch,cid,projected,road))
                    row = {'root_id':r['root_id'],'episode_id':r['episode_id'],'split':r['split'],
                        'design_episode_reuse':r['episode_id'] in q['design_episodes'],
                        'omitted_actor_count':len(inputs['root_omitted_ids']),
                        'coverage':coverage,'prefix_failures':failures,'mechanism':mechanisms,'branches':branches}
                    file = out/'roots'/f'r{index:04d}.json'
                    write_once(file,{'diagnostics':row,'forecasts':{'frame':'fixed_root_ego_origin_XY',
                        'mean_dtype':'float32','ego_dtype':'float64','mean_predictions':{m:p.tolist() for m,p in predictions.items()},
                        'ego_rollout':ego.tolist()}})
                    root_files[file.relative_to(out).as_posix()] = sha(file); rows.append(row)
                    if (index+1)%10 == 0: print(f'[rcci02_v2_cohort] roots={index+1}/{len(q["roots"])}',flush=True)
                if (report['ensemble_forward_calls'] != q['config']['max_ensemble_forward_calls']
                        or report['member_forward_calls'] != q['config']['max_member_forward_calls']):
                    raise ValueError('Frozen inference budget differs')
                shared.clean()
                if shared.sources() != q['source_hashes'] or shared.environment() != q['environment']:
                    raise ValueError('Source/environment changed during audit')
                for names,base in ((q['input_hashes'],baseline),(q['parent_artifact_hashes'],ROOT)):
                    for name,hsh in names.items():
                        if sha(inside(base,name)) != hsh: raise ValueError('Source/evidence changed during audit')
                if (shared.git(baseline,'rev-parse','HEAD') != shared.BASE or shared.git(baseline,'status','--porcelain')
                        or sha(inside(ROOT,q['config_path'])) != q['config_sha256']
                        or sha(inside(ROOT,q['config']['design_contract'])) != q['design_contract_sha256']
                        or sha(network) != q['worktree_map_sha256']):
                    raise ValueError('Baseline/config/contract/map changed during audit')
                for name,hsh in root_files.items():
                    if sha(inside(out,name)) != hsh: raise ValueError('Written root artifact changed')
                write_once(out/'index.json',{'root_files':root_files,'root_ids':[r['root_id'] for r in rows]})
                report.update(status='complete',roots=len(rows),episodes=len({r['episode_id'] for r in rows}),
                    index_sha256=sha(out/'index.json'),summaries=aggregate(rows),
                    next_stage='manual_review_no_automatic_training_or_online_release')
            except BaseException:
                report.update(status='failed',failure_reason=traceback.format_exc()); raise
            finally:
                report['finished_at'] = shared.now(); write_once(out/'report.json',report)
        write_once(out/'receipt.json',{'request_hash':expected,'report_sha256':sha(out/'report.json'),
            'index_sha256':sha(out/'index.json')})
        print('rcci02_v2_cohort_report='+str(out/'report.json'))
    finally:
        torch.set_num_threads(previous)


def main():
    p = argparse.ArgumentParser(description=__doc__); sub = p.add_subparsers(dest='command',required=True)
    a = sub.add_parser('prepare'); a.add_argument('--baseline-root',required=True)
    a.add_argument('--config',default=CONFIG_PATH); a.add_argument('--run-id',required=True)
    a = sub.add_parser('run'); a.add_argument('--request',required=True); a.add_argument('--confirm-request-hash',required=True)
    args = p.parse_args()
    if args.command == 'prepare': prepare(args)
    else: run(args.request,args.confirm_request_hash)


if __name__ == '__main__': main()
