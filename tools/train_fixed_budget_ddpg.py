"""P7n user-run 120k matched experiment; immutable P7k execution kernels reused."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import gzip
import json
import math
from pathlib import Path
import subprocess
import sys
import time
import traceback
import uuid

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'tools')]
import train_response_predictors as shared
import train_matched_ddpg as kernel
import explore_feature_ddpg as original
import smoke_feature_ddpg as smoke
import diagnose_replay_information as info
from prediction_rl.data.collection_store import (
    read_json as read, write_once, file_hash as sha, run_lock, seal_episode, verify_episode)
from prediction_rl.data.dataset_contract import digest
from prediction_rl.data.frozen_dataset import inside, verify_historical_inputs, verify_historical_text_checkout
from prediction_rl.evaluation.fixed_budget_extension import (
    PROTOCOL, SMOKE, DECISION, jobs, aggregate, validate_protocol, validate_replay_capacity)
from prediction_rl.evaluation.replay_information import PROTOCOL as INFO_PROTOCOL, jobs as info_jobs
from prediction_rl.training.all_ddpg import resolved_preset


def decision_inputs(*, verify_files=False):
    """Completed diagnostics are lineage, NOT a positive-effect gate.

    At prepare, verify the complete six diagnostic inventories and historical
    training precheck. Workers check the small frozen receipts/files they use;
    they do not rescan 360k old replay states for every new evaluation episode.
    """
    path = inside(ROOT, DECISION['request']); root = path.parent
    q = read(path); a = read(root/'aggregate.json'); current = shared.source_hashes()
    if (digest(q) != DECISION['request_hash'] or q['protocol'] != INFO_PROTOCOL
            or sha(root/'aggregate.json') != DECISION['aggregate_sha256']
            or a['request_hash'] != digest(q) or a['status'] != 'complete' or not a['engineering_complete']
            or a['method_effect_gate'] is not None or a['simulation_episodes'] != 0
            or read(root/'aggregate_receipt.json') != {'request_hash': digest(q), 'aggregate_sha256': sha(root/'aggregate.json')}
            or q['runtime']['dependencies'] != smoke.dependencies() or q['runtime']['environment'] != shared.env()
            or any(current.get(k) != v for k, v in q['runtime']['source_hashes'].items())
            or (root/'writer.lock').exists()):
        raise ValueError('P7m completed decision evidence/source changed or active')
    inputs = {}
    def record(p): inputs[str(p.relative_to(ROOT))] = sha(p)
    for name in ('request.json', 'preparation.json', 'aggregate.json', 'aggregate_receipt.json'): record(root/name)
    if read(root/'preparation.json')['request_hash'] != digest(q): raise ValueError('P7m request receipt changed')
    if verify_files:
        verify_historical_inputs(ROOT, {'python_lf:'+k: v for k, v in q['runtime']['source_hashes'].items()}, q['git_commit'])
        verify_historical_text_checkout(ROOT, q['config_path'], q['config_sha256'], q['git_commit'])
    expected = info_jobs(INFO_PROTOCOL)
    if [{k: m[k] for k in ('id', 'arm', 'run_seed', 'checkpoint_frames')} for m in q['models']] != expected:
        raise ValueError('Missing diagnostic arms or training seeds')
    for model, row in zip(q['models'], a['models']):
        directory = root/'models'/model['id']
        receipt = verify_episode(directory, info.binding(q, model)) if verify_files else read(directory/'complete.json')
        if (receipt['binding'] != info.binding(q, model) or receipt['status'] != 'complete'
                or receipt['metadata'] != row or read(directory/'report.json') != row
                or row['no_updates_no_time_advance_no_rng_draws'] is not True):
            raise ValueError('P7m complete model receipt changed')
        for name in ('complete.json', 'report.json'): record(directory/name)
    kp = inside(ROOT, INFO_PROTOCOL['parent_request']); kroot = kp.parent; kq = read(kp); ka = read(kroot/'aggregate.json')
    if (digest(kq) != INFO_PROTOCOL['parent_request_hash'] or kq['protocol'] != kernel.PROTOCOL
            or sha(kroot/'aggregate.json') != info.external.PROTOCOL['parent_aggregate_sha256']
            or ka['request_hash'] != digest(kq) or ka['status'] != 'complete' or ka['primary_node'] != 60000
            or kq['dependencies'] != smoke.dependencies() or kq['runtime_identity'] != original.runtime_identity()
            or kq['upstream_hashes'] != original.upstream_hashes()
            or any(current.get(k) != v for k, v in kq['source_hashes'].items()) or (kroot/'writer.lock').exists()):
        raise ValueError('P7k lineage/runtime changed; do not change old execution semantics')
    for name in ('request.json', 'aggregate.json', 'aggregate_receipt.json'): record(kroot/name)
    if verify_files:
        parent, prechecks = kernel.prerequisites()
        if (prechecks != kq['request_inputs'] or parent['predictors'] != kq['predictors']
                or parent['predictor_training_request'] != kq['predictor_training_request']):
            raise ValueError('Audited predictor/precheck identity changed')
    for p in (kernel.PRECHECK, kernel.PRECHECK.parent/'request.json', kernel.PRECHECK.parent/'aggregate_receipt.json'):
        record(p)
    for asset in kq['predictors'].values():
        p = inside(ROOT, asset['path'])
        if sha(p) != asset['sha256']: raise ValueError('Frozen predictor checkpoint changed')
        record(p)
    return inputs, kq


def prepare(config_path, run_id):
    shared.clean(); torch.set_num_threads(1); cp = inside(ROOT, config_path)
    p = validate_protocol(read(cp)); inputs, previous = decision_inputs(verify_files=True)
    t = p['training']; preset = resolved_preset(t['device'], t['learning_rate'], smoke=p == SMOKE)
    validate_replay_capacity(p, preset)
    if p != SMOKE and preset != previous['preset']: raise ValueError('Training preset changed beyond approved budget')
    out = shared.output_dir('p7n', run_id)
    q = {**shared.metadata(), 'version': 'p07n_request_v1', 'protocol': deepcopy(p),
        'config_path': str(cp.relative_to(ROOT)), 'config_sha256': sha(cp), 'decision_inputs': inputs,
        'source_hashes': shared.source_hashes(), 'dependencies': smoke.dependencies(), 'environment': shared.env(),
        'runtime_identity': original.runtime_identity(), 'upstream_hashes': original.upstream_hashes(),
        'upstream_config_sha256': sha(original.UPSTREAM), 'preset': preset,
        'predictors': previous['predictors'], 'predictor_training_request': previous['predictor_training_request'],
        'train_jobs': jobs(p, 'train'), 'evaluation_jobs': jobs(p, 'evaluate'),
        'previous_policy_weights_used_for_training': False,
        'execution_kernel': 'immutable_P7k_training_and_Gym20_evaluation',
        'author50_results_mixed': False}
    write_once(out/'request.json', q)
    write_once(out/'preparation.json', {'request_hash': digest(q), 'status': 'prepared_not_trained',
        'requested_training_instances': len(q['train_jobs']),
        'requested_total_training_steps': len(q['train_jobs'])*t['requested_frames'],
        'evaluation_episodes': len(q['evaluation_jobs']),
        'unique_traffic_scenarios': len(p['evaluation']['simulator_seeds']),
        'full_training_started': False, 'automatic_budget_extension': False})
    print('fixed_budget_request='+str(out/'request.json'), flush=True)
    print('confirm_request_hash='+digest(q), flush=True)
    return out/'request.json', digest(q)


def load_request(path, expected_hash):
    shared.clean(); torch.set_num_threads(1); path = inside(ROOT, path); q = read(path)
    p = validate_protocol(q['protocol']); t = p['training']
    preset = resolved_preset(t['device'], t['learning_rate'], smoke=p == SMOKE)
    validate_replay_capacity(p, preset)
    if (path.parent.parent != ROOT/'artifacts/p7n' or q['version'] != 'p07n_request_v1'
            or digest(q) != expected_hash or read(path.parent/'preparation.json')['request_hash'] != expected_hash
            or q['source_hashes'] != shared.source_hashes() or q['dependencies'] != smoke.dependencies()
            or q['environment'] != shared.env() or q['runtime_identity'] != original.runtime_identity()
            or q['upstream_hashes'] != original.upstream_hashes() or q['upstream_config_sha256'] != sha(original.UPSTREAM)
            or q['preset'] != preset or sha(inside(ROOT, q['config_path'])) != q['config_sha256']
            or read(inside(ROOT, q['config_path'])) != p or q['train_jobs'] != jobs(p, 'train')
            or q['evaluation_jobs'] != jobs(p, 'evaluate') or q['previous_policy_weights_used_for_training'] is not False):
        raise ValueError('Frozen source/config/runtime/roster changed; no silent adaptation')
    inputs, previous = decision_inputs()
    if (q['decision_inputs'] != inputs or q['predictors'] != previous['predictors']
            or q['predictor_training_request'] != previous['predictor_training_request']):
        raise ValueError('Decision/precheck/predictor identity changed')
    return path, q


def train_job(q, job, out):
    validate_replay_capacity(q['protocol'], q['preset'])
    # No checkpoint loading, continuation, global monkey-patching or new DDPG.
    return kernel.train_job(q, job, out)


def worker(args):
    path, q = load_request(args.request, args.confirm_request_hash)
    plan = q['train_jobs' if args.stage == 'train' else 'evaluation_jobs']
    job = next(j for j in plan if j['id'] == args.job)
    out = path.parent/args.stage/job['id']; out.mkdir(parents=True, exist_ok=False)
    started = {**shared.metadata(), 'binding': kernel.binding(q, args.stage, job)}
    write_once(out/'started.json', started); begin = time.perf_counter()
    try:
        r = train_job(q, job, out) if args.stage == 'train' else kernel.evaluation_job(q, job, out, path.parent)
        r.update(provenance=started, elapsed_s=time.perf_counter()-begin, finished_at=shared.now())
        write_once(out/'report.json', r); seal_episode(out, kernel.binding(q, args.stage, job), r)
    except BaseException:
        write_once(out/'failure.json', {'status': 'failed', 'failure_reason': traceback.format_exc()}); raise


def run(path, expected_hash, stage, resume=False):
    if stage not in ('train', 'evaluate'): raise ValueError('Unknown stage')
    path, q = load_request(path, expected_hash); root = path.parent
    with run_lock(root):
        invocation = root/'invocations'/uuid.uuid4().hex[:10]; invocation.mkdir(parents=True, exist_ok=False)
        report = {**shared.metadata(), 'stage': stage, 'request_hash': digest(q), 'jobs': []}
        try:
            if stage == 'evaluate': kernel.training_metadata(q, root)
            plan = q['train_jobs' if stage == 'train' else 'evaluation_jobs']; pending = []
            for job in plan:
                if (root/stage/job['id']).exists():
                    if not resume: raise ValueError('Use --resume only to reuse sealed COMPLETE whole jobs')
                    verify_episode(root/stage/job['id'], kernel.binding(q, stage, job))
                    report['jobs'].append({'id': job['id'], 'action': 'reuse'})
                else: pending.append(job)
            def child(job):
                command = [sys.executable, '-B', str(Path(__file__).resolve()), 'worker', '--request', str(path),
                           '--confirm-request-hash', digest(q), '--stage', stage, '--job', job['id']]
                print(f'[p7n] {stage} {job["id"]}', flush=True)
                with (invocation/(job['id']+'.log')).open('x', encoding='utf-8') as log:
                    subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
                verify_episode(root/stage/job['id'], kernel.binding(q, stage, job))
                return {'id': job['id'], 'action': 'run'}
            width = q['protocol']['training' if stage == 'train' else 'evaluation']['workers']
            with ThreadPoolExecutor(max_workers=width) as pool:
                for i in range(0, len(pending), width):
                    futures = [pool.submit(child, j) for j in pending[i:i+width]]
                    report['jobs'].extend(f.result() for f in futures)
            if stage == 'train': kernel.training_metadata(q, root)
            report.update(status='complete', finished_at=shared.now())
        except BaseException:
            report.update(status='failed', failure_reason=traceback.format_exc()); raise
        finally: write_once(invocation/'report.json', report)
    print('stage_report='+str(invocation/'report.json'), flush=True)


def curve_index(root, trained, q):
    """Keep original gzip losses; index them instead of duplicating ~0.5GB JSON."""
    result = {}
    for job in q['train_jobs']:
        d = root/'train'/job['id']; counts = {'q': 0, 'policy': 0}
        with gzip.open(d/'losses.jsonl.gz', 'rt', encoding='utf-8') as handle:
            for line in handle:
                r = json.loads(line)
                if r['component'] not in counts or not math.isfinite(r['value']): raise ValueError('Invalid saved optimizer loss')
                counts[r['component']] += 1
        if any(n != trained[job['id']]['coverage']['sample_calls'] for n in counts.values()):
            raise ValueError('Optimizer loss/sample counts differ')
        result[job['id']] = {'training_episodes': str((d/'episodes.json').relative_to(root)),
            'training_episodes_sha256': sha(d/'episodes.json'),
            'optimizer_losses_jsonl_gz': str((d/'losses.jsonl.gz').relative_to(root)),
            'optimizer_losses_sha256': sha(d/'losses.jsonl.gz'), 'optimizer_loss_counts': counts}
    return {'version': 'p07n_training_curves_index_v1',
            'label': 'noisy_training_returns_and_original_losses_NOT_validation', 'jobs': result}


def aggregate_run(path, expected_hash):
    path, q = load_request(path, expected_hash); root = path.parent
    with run_lock(root):
        trained = kernel.training_metadata(q, root); rows = []
        for job in q['evaluation_jobs']:
            out = root/'evaluate'/job['id']; m = verify_episode(out, kernel.binding(q, 'evaluate', job))['metadata']
            r = read(out/'episode.json')
            snap = next(s for s in trained[job['training_id']]['snapshots'] if s['requested_steps'] == job['checkpoint_frames'])
            if r != m['outcome'] or r['policy_sha256'] != snap['weights_sha256']:
                raise ValueError('Evaluation/checkpoint lineage changed')
            rows.append(r)
        result = {**aggregate(q['protocol'], rows), 'request_hash': digest(q), 'provenance': shared.metadata(),
                  'training': trained, 'finished_at': shared.now(), 'previous_policy_weights_used_for_training': False,
                  'automatic_budget_extension': False}
        write_once(root/'training_curves.json', curve_index(root, trained, q))
        write_once(root/'development_curves.json', {'label': 'independent_frozen_policy_Gym20_development_NOT_test',
            'unique_scenarios': len(q['protocol']['evaluation']['simulator_seeds']), 'nodes': result['nodes']})
        write_once(root/'aggregate.json', result)
        write_once(root/'aggregate_receipt.json', {'request_hash': digest(q), 'aggregate_sha256': sha(root/'aggregate.json'),
            'training_curves_sha256': sha(root/'training_curves.json'), 'development_curves_sha256': sha(root/'development_curves.json')})
    print('fixed_budget_aggregate='+str(root/'aggregate.json'), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__); s = p.add_subparsers(dest='command', required=True)
    a = s.add_parser('prepare'); a.add_argument('--config', default='configs/development/p07n_matched_120k_v1.json')
    a.add_argument('--run-id', required=True)
    for name in ('run', 'worker', 'aggregate'):
        a = s.add_parser(name); a.add_argument('--request', required=True); a.add_argument('--confirm-request-hash', required=True)
        if name != 'aggregate': a.add_argument('--stage', choices=['train', 'evaluate'], required=True)
        if name == 'run': a.add_argument('--resume', action='store_true')
        if name == 'worker': a.add_argument('--job', required=True)
    a = s.add_parser('audit'); a.add_argument('--run-id', required=True)
    args = p.parse_args()
    if args.command == 'prepare': prepare(args.config, args.run_id)
    elif args.command == 'worker': worker(args)
    elif args.command == 'run': run(args.request, args.confirm_request_hash, args.stage, args.resume)
    elif args.command == 'aggregate': aggregate_run(args.request, args.confirm_request_hash)
    else:
        path, h = prepare('configs/development/p07k_workflow_smoke_v1.json', args.run_id)
        run(path, h, 'train'); run(path, h, 'evaluate'); aggregate_run(path, h)


if __name__ == '__main__': main()
