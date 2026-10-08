"""Frozen difference predictor: separate calibration and unchanged online features.

Completed training is accepted by historical Git identity, not rewritten to the
current revision. Training/resume CLIs retain their stricter current-source gate.
"""
import argparse
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'tools')]
import train_response_predictors as shared
import train_candidate_difference as delta
from prediction_rl.data.collection_store import read_json as read, file_hash as sha, write_once, run_lock
from prediction_rl.data.dataset_contract import digest
from prediction_rl.data.frozen_dataset import inside, verify_historical_inputs, verify_historical_text_checkout
from prediction_rl.prediction.trained_artifacts import checked_history
from prediction_rl.prediction.difference_artifacts import load_difference_ensemble
from prediction_rl.prediction.calibration import (
    SETTINGS, load_calibration_split, episode_scores, fit_episode_quantile, band_diagnostics)
from prediction_rl.features.probe_features import FrozenProbeFeatures
from prediction_rl.prediction.interface import PredictorConfig
from prediction_rl.prediction.kinematic_reference import LaneConstantVelocity

VERSION = 'candidate_difference_calibration_v1'
# The later DDPG bridge adds explicit dependency injection to this kernel only.
# Predictor/data/feature consumers must still match the frozen training source.
ALLOWED_EXECUTION_EXTENSION = {'tools/train_matched_ddpg.py'}


def accepted_trial(request_path, report_path, diagnostic_path):
    rp, tp, dp = [inside(ROOT, v) for v in (request_path, report_path, diagnostic_path)]
    p, report, diagnostic = read(rp), read(tp), read(dp)
    if p['environment'] != shared.env() or read(rp.parent/'preparation.json')['request_hash'] != digest(p):
        raise ValueError('Training environment/request receipt changed')
    verify_historical_inputs(ROOT, {'python_lf:'+k:v for k,v in p['source_hashes'].items()}, report['git_commit'])
    current = {k.replace('\\', '/'):v for k,v in shared.source_hashes().items()}
    if any(current.get(k.replace('\\', '/')) != v for k,v in p['source_hashes'].items()
           if k.replace('\\', '/') not in ALLOWED_EXECUTION_EXTENSION):
        raise ValueError('Frozen predictor/data/feature implementation changed')
    cp = inside(ROOT, p['configuration_path'])
    verify_historical_text_checkout(ROOT, p['configuration_path'], p['configuration_sha256'], report['git_commit'])
    if sha(cp) != p['configuration_sha256'] or read(cp) != p['resolved_config']:
        raise ValueError('Frozen difference configuration changed')
    reference = Path(p['reference_root']).resolve()
    if reference == ROOT.resolve(): raise ValueError('Reference project must remain separate and read-only')
    delta.verify_reference(reference, p['baseline_input_hashes'])
    baseline = read(inside(reference, p['resolved_config']['baseline']['training_request']))
    if (digest(baseline) != p['baseline_request_hash']
            or delta.checked_config(p['resolved_config'], baseline) != p['training']
            or p['jobs'] != delta.jobs(p['training']) or p['test_locked'] is not True
            or p['calibration_used_for_selection'] is not False or p['ddpg_ready'] is not False):
        raise ValueError('Difference trial boundaries/baseline changed')
    dataset = read(inside(reference, p['dataset_review']))
    if dataset['test_locked'] is not True or dataset['protocol']['calibration'] != SETTINGS:
        raise ValueError('Unsealed test or changed calibration rule')
    _, record = delta.completed_ensemble(rp.parent, p, tp)
    for job, metadata in zip(p['jobs'], report['members']):
        checked_history(read(rp.parent/job['id']/'history.json'), metadata, p['training'],
                        dataset['summaries']['train']['eligible_roots'])
    if (diagnostic['status'] != 'complete' or diagnostic['request_hash'] != digest(p)
            or diagnostic['training_report_sha256'] != sha(tp)
            or diagnostic['checkpoint_sha256'] != record['sha256']
            or diagnostic['scope'] != 'frozen_validation_not_independent_test'
            or diagnostic['method_effect_established'] is not False or diagnostic['ddpg_ready'] is not False
            or diagnostic['working_tree_dirty'] is not False):
        raise ValueError('Completed paired validation lineage required; no positive-effect gate')
    paths = [rp, tp, dp, rp.parent/'preparation.json', inside(ROOT, record['path'])]
    paths += [rp.parent/j['id']/name for j in p['jobs'] for name in ('complete.json', 'selected.pt', 'history.json')]
    inputs = {str(v.relative_to(ROOT)):sha(v) for v in paths}
    return p, reference, dataset, record, inputs


def prepare(args):
    shared.clean()
    p, reference, dataset, record, inputs = accepted_trial(args.training_request, args.training_report, args.diagnostic)
    batch = load_calibration_split(reference, dataset)
    q = {'version':VERSION, 'training_request':str(inside(ROOT, args.training_request).relative_to(ROOT)),
         'training_report':str(inside(ROOT, args.training_report).relative_to(ROOT)),
         'diagnostic':str(inside(ROOT, args.diagnostic).relative_to(ROOT)), 'input_hashes':inputs,
         'settings':SETTINGS, 'data_hash':batch.fingerprint(), 'checkpoint':record,
         'source_hashes':shared.source_hashes(), 'environment':shared.env(),
         'batch_roots':16, 'test_locked':True, 'checkpoint_selection_allowed':False,
         'policy_features_use_bands':False, 'ddpg_training_authorized':False}
    out = shared.output_dir('difference_calibration', args.run_id)
    write_once(out/'request.json', q)
    write_once(out/'preparation.json', {**shared.metadata(), 'status':'prepared_not_fitted',
        'request_hash':digest(q), 'calibration_roots':len(batch.root_ids),
        'calibration_episodes':len(dataset['manifest']['calibration'])})
    print('difference_calibration_request='+str(out/'request.json'))
    print('confirm_request_hash='+digest(q))
    return out/'request.json'


def load_request(path, expected_hash):
    path = inside(ROOT, path); q = read(path)
    if (digest(q) != expected_hash or q['version'] != VERSION or q['settings'] != SETTINGS
            or read(path.parent/'preparation.json')['request_hash'] != expected_hash
            or q['source_hashes'] != shared.source_hashes() or q['environment'] != shared.env()
            or q['batch_roots'] != 16 or q['test_locked'] is not True
            or q['checkpoint_selection_allowed'] is not False or q['policy_features_use_bands'] is not False
            or q['ddpg_training_authorized'] is not False):
        raise ValueError('Calibration request/source/environment/boundaries changed')
    p, reference, dataset, record, inputs = accepted_trial(q['training_request'], q['training_report'], q['diagnostic'])
    if q['input_hashes'] != inputs or q['checkpoint'] != record:
        raise ValueError('Calibration input/model binding changed')
    return path.parent, q, p, reference, dataset


def run(args):
    shared.clean(); out, q, p, reference, dataset = load_request(args.request, args.confirm_request_hash)
    if any((out/name).exists() for name in ('started.json', 'report.json', 'calibration.json')):
        raise ValueError('Calibration already started; preserve immutable outputs')
    previous = torch.get_num_threads(); torch.set_num_threads(1)
    try:
        with run_lock(out):
            report = {**shared.metadata(), 'status':'running', 'request_hash':digest(q)}
            write_once(out/'started.json', report)
            try:
                record = q['checkpoint']
                model = load_difference_ensemble(inside(ROOT, record['path']), p, record['sha256'], record['selected'])
                batch = load_calibration_split(reference, dataset)
                if batch.fingerprint() != q['data_hash']: raise ValueError('Calibration tensors changed')
                rows, roots = episode_scores(model, batch, dataset['manifest']['calibration'], batch_roots=16)
                fit = fit_episode_quantile(rows, SETTINGS)
                frozen = {'version':VERSION, 'purpose':'frozen_observed_trajectory_bands',
                    'request_hash':digest(q), 'training_request_hash':digest(p),
                    'checkpoint_sha256':record['sha256'], 'calibration_data_hash':q['data_hash'],
                    'fit':fit, 'test_locked':True, 'policy_features_use_bands':False,
                    'method_effect_established':False}
                write_once(out/'calibration.json', frozen)
                report.update(status='complete', calibration_sha256=sha(out/'calibration.json'),
                    fit=fit, episode_scores=rows, root_scores=roots,
                    diagnostics=band_diagnostics(model, batch, fit), test_evaluated=False,
                    ddpg_training_started=False)
            except BaseException as error:
                report.update(status='failed', failure_reason=f'{type(error).__name__}: {error}'); raise
            finally:
                report['finished_at'] = shared.now(); write_once(out/'report.json', report)
                print('difference_calibration_report='+str(out/'report.json'), flush=True)
    finally: torch.set_num_threads(previous)


def accepted_calibration(request_path):
    """Read a completed fit after code evolves, without refitting/relabeling it."""
    rp = inside(ROOT, request_path); q = read(rp); out = rp.parent
    r, c = read(out/'report.json'), read(out/'calibration.json')
    if (q['version'] != VERSION or q['settings'] != SETTINGS or q['environment'] != shared.env()
            or (out/'writer.lock').exists() or r['status'] != 'complete' or r['working_tree_dirty'] is not False
            or read(out/'preparation.json')['request_hash'] != digest(q) or r['request_hash'] != digest(q)
            or r['calibration_sha256'] != sha(out/'calibration.json')
            or c['request_hash'] != digest(q) or c['fit'] != r['fit']
            or c['policy_features_use_bands'] is not False or c['test_locked'] is not True
            or r['test_evaluated'] is not False or r['ddpg_training_started'] is not False):
        raise ValueError('Incomplete or modified difference calibration')
    verify_historical_inputs(ROOT, {'python_lf:'+k:v for k,v in q['source_hashes'].items()}, r['git_commit'])
    p, reference, dataset, record, inputs = accepted_trial(q['training_request'], q['training_report'], q['diagnostic'])
    if (q['input_hashes'] != inputs or q['checkpoint'] != record or c['checkpoint_sha256'] != record['sha256']
            or c['training_request_hash'] != digest(p) or c['calibration_data_hash'] != q['data_hash']):
        raise ValueError('Calibration is not bound to the new difference ensemble')
    if fit_episode_quantile(r['episode_scores'], SETTINGS) != r['fit']:
        raise ValueError('Frozen episode quantile changed')
    for name in ('request.json', 'preparation.json', 'report.json', 'calibration.json'):
        inputs[str((out/name).relative_to(ROOT))] = sha(out/name)
    return p, reference, record, inputs


def provider(p, reference, record):
    model = load_difference_ensemble(inside(ROOT, record['path']), p, record['sha256'], record['selected'])
    return FrozenProbeFeatures(LaneConstantVelocity(reference/'RL-MPC-LaneMerging-master/merge.net.xml'),
                               PredictorConfig.from_dict(p['model']), model)


def main():
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest='command', required=True)
    q = sub.add_parser('prepare'); q.add_argument('--training-request', required=True)
    q.add_argument('--training-report', required=True); q.add_argument('--diagnostic', required=True)
    q.add_argument('--run-id', required=True)
    q = sub.add_parser('run'); q.add_argument('--request', required=True); q.add_argument('--confirm-request-hash', required=True)
    args = parser.parse_args(); {'prepare':prepare, 'run':run}[args.command](args)


if __name__ == '__main__': main()
