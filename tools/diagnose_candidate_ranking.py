"""User-run P5b validation proximity ranking; no SUMO, training or test release."""
import argparse
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'tools')]
import train_response_predictors as shared
import diagnose_response_predictors as previous
from prediction_rl.data.collection_store import read_json as read, file_hash as sha, write_once, run_lock
from prediction_rl.data.dataset_contract import digest
from prediction_rl.data.frozen_dataset import inside, verify_historical_inputs
from prediction_rl.data.protocol_plan import resolve_upstream_settings
from prediction_rl.prediction.interface import PredictorConfig
from prediction_rl.prediction.supervision import align_supervision
from prediction_rl.prediction.epoch_training import subset
from prediction_rl.prediction.trained_artifacts import load_trained_ensemble
from prediction_rl.prediction.kinematic_reference import LaneConstantVelocity
from prediction_rl.prediction.offline_diagnostics import summarize, paired_difference, summarize_attribution
from prediction_rl.prediction.candidate_ranking import CONTROL, PARITY, TIE_EPS_M, ranking_rows, summarize_ranking

CONFIG = {'version': 'p05_candidate_ranking_v1', 'analysis_status': 'exploratory_validation_not_test',
    'p5a_request': 'artifacts/offline/predictor_v1_diag2/request.json',
    'network': 'RL-MPC-LaneMerging-master/merge.net.xml',
    'upstream_config': 'RL-MPC-LaneMerging-master/configs/train_default_1.json',
    'methods': ['lane_chain_constant_velocity', 'ordinary', 'conditional'], 'split': 'validation', 'batch_roots': 16,
    'ego_rollout': 'root_only_saturated_jerk_lane_chain_semi_implicit_v1', 'ego_parity_tolerances': PARITY,
    'score': 'maximize_min_front_bumper_euclidean_distance_selected_actors_v1',
    'support': 'same_actor_time_intersection_all_five_candidates_including_observed_ego',
    'views': ['all_common_observed_support', 'complete_5s_all_selected_actors'],
    'numeric_tie_epsilon_m': TIE_EPS_M, 'tie_break': 'lowest_candidate_id',
    'calibrated_bands_used': False, 'hypothesis_test': False, 'test_release': False, 'ddpg_release': False}
DEV = 'artifacts/p4/p4_training_v1_01/report.json'
DEV_SHA = '174146a8d78962929e18bcd426f9c136db26f0f94deda5bfc449c488459dc3b5'


def prerequisites(config_path):
    cp = inside(ROOT, config_path); config = read(cp)
    if config != CONFIG:
        raise ValueError('Unsupported ranking protocol; do not tune on inspected outcomes')
    qp = inside(ROOT, config['p5a_request']); q = read(qp); out = qp.parent; report = read(out/'report.json')
    if ((out/'writer.lock').exists() or report['status'] != 'complete' or report['working_tree_dirty'] is not False
            or report['request_hash'] != digest(q) or read(out/'preparation.json')['request_hash'] != digest(q)
            or report['test_evaluated'] is not False or report['ddpg_ready'] is not False
            or report['action_ranking_evaluated'] is not False or q['environment'] != shared.env()
            or q['test_locked'] is not True or q['ddpg_ready'] is not False
            or sha(inside(ROOT, q['audit'])) != q['audit_sha256']):
        raise ValueError('Need complete unchanged P5a with test still sealed')
    verify_historical_inputs(ROOT, {'python_lf:'+k: v for k, v in q['source_hashes'].items()}, report['git_commit'])
    cfg, r, p, d, frozen, hashes = previous.prerequisites(q['config_path'])
    if cfg != q['config'] or hashes != q['input_hashes']:
        raise ValueError('P5a inputs changed')
    paths = [qp, out/'report.json', out/'preparation.json', inside(ROOT, q['audit']), ROOT/config['network']]
    point_rows = {}
    for mode in config['methods']:
        path = out/f'validation_{mode}_roots.json'; rows = read(path); paths.append(path); point_rows[mode] = rows
        if [(x['root_id'], x['episode_id']) for x in rows] != [(x['root_id'], x['episode_id']) for x in d['roots']['validation']]:
            raise ValueError('P5a root roster differs')
        if summarize(rows, d['manifest']['validation']) != report['splits']['validation']['point_metrics'][mode]:
            raise ValueError('P5a point rows/summary differ')
    expected_pairs = {mode+'__to__conditional': paired_difference(point_rows[mode], point_rows['conditional'], d['manifest']['validation'])
                      for mode in config['methods'] if mode != 'conditional'}
    if expected_pairs != report['splits']['validation']['paired_differences']:
        raise ValueError('P5a paired summary differs')
    for split in ('validation', 'calibration'):
        for mode in ('ordinary', 'conditional'):
            path = out/f'{split}_{mode}_attribution.json'; rows = read(path); paths.append(path)
            if summarize_attribution(rows, d['manifest'][split], frozen['methods'][mode]['fit'], split) != report['splits'][split]['frozen_band_attribution'][mode]:
                raise ValueError('P5a attribution rows/summary differ')
    settings_path = ROOT/'RL-MPC-LaneMerging-master/config.py'
    control_path = ROOT/'RL-MPC-LaneMerging-master/control.py'
    upstream_config = ROOT/config['upstream_config']
    settings = resolve_upstream_settings(settings_path.read_text(encoding='utf-8'), read(upstream_config), cuda_available=False, system='Windows')
    if {k: settings[k] for k in CONTROL} != CONTROL or d['protocol']['baseline']['upstream_config'] != config['upstream_config']:
        raise ValueError('Different upstream execution settings')
    all_hashes = {**hashes, **previous.artifact_hashes(cp, paths),
        'json_canonical:'+config['upstream_config']: digest(read(upstream_config)),
        **{'python_lf:'+str(path.relative_to(ROOT)): shared.fingerprint(path.read_text(encoding='utf-8')) for path in (settings_path, control_path)}}
    return config, r, p, d, all_hashes


def development_data(p):
    ep = ROOT/DEV
    if sha(ep) != DEV_SHA:
        raise ValueError('Development evidence changed')
    evidence = read(ep); histories = []; packs = []
    for row in evidence['history_inputs']:
        hp = inside(ROOT, row['path'])
        if sha(hp) != row['sha256']:
            raise ValueError('Development history changed')
        h = read(hp); lp = inside(ROOT, h['label_pack'])
        if sha(lp) != h['label_pack_sha256']:
            raise ValueError('Development labels changed')
        histories.append(h); packs.append(read(lp))
    manifest = read((ROOT/evidence['label_inputs'][0]['path']).parent/'split_manifest.json')
    b = align_supervision(histories, packs, manifest, PredictorConfig.from_dict(p['model']), split='development')
    if b.fingerprint() != evidence['data_hash']:
        raise ValueError('Development tensors changed')
    return b, histories, packs, manifest['development']


def validation_data(d):
    b, histories = previous.data(d, 'validation'); packs = []
    for h, row in zip(histories, d['roots']['validation']):
        lp = inside(ROOT, row['labels'])
        if inside(ROOT, h['label_pack']) != lp or sha(lp) != row['labels_sha256'] or sha(lp) != h['label_pack_sha256']:
            raise ValueError('Validation labels changed')
        packs.append(read(lp))
    return b, histories, packs, d['manifest']['validation']


def evaluate(r, p, config, data):
    b, histories, packs, roster = data; cfg = PredictorConfig.from_dict(p['model'])
    geometry = LaneConstantVelocity(ROOT/config['network'])
    models = {mode: load_trained_ensemble(ROOT/e['path'], p, mode, e['sha256'], e['selected']) for mode, e in r['ensembles'].items()}
    rows = []
    with torch.inference_mode():
        for start in range(0, len(b.root_ids), config['batch_roots']):
            stop = min(start+config['batch_roots'], len(b.root_ids)); sub = subset(b, list(range(start, stop)))
            predictions = {'lane_chain_constant_velocity': geometry.predict(histories[start:stop], cfg)}
            predictions.update({mode: model(sub.observations, sub.plans)['member_trajectories'].mean(0) for mode, model in models.items()})
            rows.extend(ranking_rows(predictions, sub, histories[start:stop], packs[start:stop], geometry))
    return rows, summarize_ranking(rows, roster, config['methods'])


def audit(args):
    shared.clean(); config, r, p, d, hashes = prerequisites(args.config)
    data = development_data(p); out = shared.output_dir('p5', args.run_id)
    report = {**shared.metadata(), 'status': 'running', 'purpose': 'development_smoke', 'interface_gate': False,
        'source_hashes': shared.source_hashes(), 'environment': shared.env(), 'inputs': hashes,
        'config_hash': digest(config), 'development_evidence_sha256': DEV_SHA, 'development_data_hash': data[0].fingerprint()}
    old = torch.get_num_threads(); torch.set_num_threads(1)
    try:
        rows, summary = evaluate(r, p, config, data)
        write_once(out/'roots.json', rows)
        report.update(status='complete', summary=summary, interface_gate=summary['ego_parity_gate'],
                      roots_sha256=sha(out/'roots.json'), formal_evaluation_started=False, test_opened=False, ddpg_ready=False)
    except BaseException as error:
        report.update(status='failed', failure_reason=f'{type(error).__name__}: {error}'); raise
    finally:
        torch.set_num_threads(old); report['finished_at'] = shared.now(); write_once(out/'report.json', report)
    print('ranking_audit='+str(out/'report.json'))


def prepare(args):
    shared.clean(); config, r, p, d, hashes = prerequisites(args.config)
    ap = inside(ROOT, args.audit); a = read(ap)
    if (a['status'] != 'complete' or a['purpose'] != 'development_smoke' or a['interface_gate'] is not True
            or a['source_hashes'] != shared.source_hashes() or a['environment'] != shared.env()
            or a['inputs'] != hashes or a['config_hash'] != digest(config)
            or sha(ap.parent/'roots.json') != a['roots_sha256'] or a['development_evidence_sha256'] != DEV_SHA):
        raise ValueError('Need accepted current bounded ranking audit')
    b, _, _, _ = validation_data(d)  # Hash only, never evaluate during preparation.
    q = {'config': config, 'config_path': str(inside(ROOT, args.config).relative_to(ROOT)), 'input_hashes': hashes,
        'audit': str(ap.relative_to(ROOT)), 'audit_sha256': sha(ap), 'data_hash': b.fingerprint(),
        'source_hashes': shared.source_hashes(), 'environment': shared.env(), 'test_locked': True, 'ddpg_ready': False}
    out = shared.output_dir('ranking', args.run_id); write_once(out/'request.json', q)
    write_once(out/'preparation.json', {**shared.metadata(), 'status': 'prepared_not_evaluated', 'request_hash': digest(q)})
    print('ranking_request='+str(out/'request.json')); print('confirm_request_hash='+digest(q))


def load_request(path):
    path = inside(ROOT, path); q = read(path)
    if (q['source_hashes'] != shared.source_hashes() or q['environment'] != shared.env()
            or q['test_locked'] is not True or q['ddpg_ready'] is not False
            or read(path.parent/'preparation.json')['request_hash'] != digest(q)
            or sha(inside(ROOT, q['audit'])) != q['audit_sha256']):
        raise ValueError('Ranking request/source changed')
    config, r, p, d, hashes = prerequisites(q['config_path'])
    if config != q['config'] or hashes != q['input_hashes']:
        raise ValueError('Ranking prerequisites changed')
    return path.parent, q, r, p, d


def run(args):
    shared.clean(); out, q, r, p, d = load_request(args.request)
    if args.confirm_request_hash != digest(q):
        raise ValueError('Confirm exact P5b diagnostic request')
    if (out/'started.json').exists() or (out/'report.json').exists():
        raise ValueError('Never overwrite a ranking diagnostic')
    old = torch.get_num_threads(); torch.set_num_threads(1)
    try:
        with run_lock(out):
            report = {**shared.metadata(), 'status': 'running', 'request_hash': digest(q),
                      'test_evaluated': False, 'ddpg_ready': False, 'policy_effect_established': False,
                      'scope': 'exploratory_selected_actor_proximity_proxy_not_safety_or_task_ranking'}
            write_once(out/'started.json', report)
            try:
                data = validation_data(d)
                if data[0].fingerprint() != q['data_hash']:
                    raise ValueError('Ranking data changed')
                rows, summary = evaluate(r, p, q['config'], data); write_once(out/'roots.json', rows)
                # Negative predictor results complete normally; parity failure is
                # separately visible and cannot be mistaken for policy readiness.
                report.update(status='complete', summary=summary, roots_sha256=sha(out/'roots.json'),
                              next_stage='human_review_ranking_and_ego_parity_before_any_P6_release')
            except BaseException as error:
                report.update(status='failed', failure_reason=f'{type(error).__name__}: {error}'); raise
            finally:
                report['finished_at'] = shared.now(); write_once(out/'report.json', report)
                print('ranking_report='+str(out/'report.json'))
    finally:
        torch.set_num_threads(old)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest='command', required=True)
    for name in ('audit', 'prepare'):
        s = sub.add_parser(name); s.add_argument('--config', required=True); s.add_argument('--run-id', required=True)
        if name == 'prepare': s.add_argument('--audit', required=True)
    s = sub.add_parser('run'); s.add_argument('--request', required=True); s.add_argument('--confirm-request-hash', required=True)
    args = parser.parse_args(); {'audit': audit, 'prepare': prepare, 'run': run}[args.command](args)
