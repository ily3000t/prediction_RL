"""Read-only RCCI04a audit of six historical Raw policies; no policy execution."""
import argparse
import hashlib
from pathlib import Path
import re
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'tools')]
import torch

import audit_rcci_feasibility as shared
import explore_feature_ddpg as original
import smoke_feature_ddpg as smoke
from prediction_rl.data.branching import fingerprint
from prediction_rl.data.collection_store import read_json as read, write_once, run_lock
from prediction_rl.data.dataset_contract import digest
from prediction_rl.data.frozen_dataset import inside, verify_historical_inputs, verify_historical_text_checkout
from prediction_rl.evaluation.fixed_budget_extension import PROTOCOL, jobs
from prediction_rl.features.probe_features import NAMES, VERSION
from prediction_rl.training.all_ddpg import resolved_preset

SETTINGS = {'version': 'rcci04a_raw_reuse_readonly_v1',
    'design': 'docs/design/rcci04a_cached_geometry_and_raw_reuse_v1.md',
    'request': 'artifacts/p7n/p7n_120k_v1/request.json',
    'request_hash': '31301c64cf15606f8fc05a0280b5f7f75fd1c6e33b5cd877087284e59f31befa',
    'aggregate_sha256': 'e2dd769d3943df09db5b2d46991c739cb92604cbbc5277de749e1383b7318c79',
    'arms': ['ordinary', 'conditional'], 'run_seeds': [0, 1, 2], 'primary_node': 120000,
    'policy_loading': False, 'simulation_episodes': 0, 'training_performed': False}


def require(ok, message):
    if not ok: raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b''): h.update(chunk)
    return h.hexdigest()


def sealed(directory, binding):
    """Same complete-inventory checks as verify_episode, bounded hash memory."""
    receipt = read(directory/'complete.json')
    require(set(receipt) == {'schema_version', 'status', 'binding', 'files', 'metadata'}
        and receipt['schema_version'] == 1 and receipt['status'] == 'complete'
        and receipt['binding'] == binding, 'Historical receipt binding differs')
    files = {}
    for p in sorted(directory.rglob('*')):
        require(not p.is_symlink() and p.resolve().is_relative_to(directory.resolve()), 'Receipt path escape')
        if p.is_file() and p != directory/'complete.json': files[p.relative_to(directory).as_posix()] = sha(p)
    require(bool(files) and 'started.json' in files and 'failure.json' not in files
        and files == receipt['files'], 'Historical inventory changed')
    require(read(directory/'report.json') == receipt['metadata'], 'Historical report/receipt mismatch')
    return receipt


def check_weights(payload, expected_contract):
    require(isinstance(payload, dict) and set(payload) == {'contract', 'state_dicts'}
        and payload['contract'] == expected_contract and set(payload['state_dicts']) == {'policy', 'q'},
        'Raw checkpoint contract changed; no partial adaptation')
    for name, input_size in (('policy', 170), ('q', 171)):
        shapes = {'model.0.weight': (400, input_size), 'model.0.bias': (400,),
            'model.2.weight': (300, 400), 'model.2.bias': (300,),
            'model.4.weight': (1, 300), 'model.4.bias': (1,)}
        state = payload['state_dicts'][name]
        require(isinstance(state, dict) and set(state) == set(shapes), 'Raw parameter roster differs')
        require(all(isinstance(state[k], torch.Tensor) and tuple(state[k].shape) == shape
            and state[k].dtype == torch.float32 and torch.isfinite(state[k]).all()
            for k, shape in shapes.items()), 'Raw parameter shape/dtype/finiteness differs')


def audit(baseline, run_id):
    shared.clean(); torch.set_num_threads(1)
    require(bool(re.fullmatch('[A-Za-z0-9_-]{1,24}', run_id)), 'Use a fresh short run ID')
    baseline = Path(baseline).resolve()
    require(baseline == Path('E:/Prediction_RL').resolve() and
        shared.git(baseline, 'rev-parse', 'HEAD') == shared.BASE and not shared.git(baseline, 'status', '--porcelain'),
        'Require clean unchanged read-only baseline')
    path = inside(baseline, SETTINGS['request']); root = path.parent; q = read(path)
    a = read(root/'aggregate.json'); receipt = read(root/'aggregate_receipt.json')
    require(digest(q) == SETTINGS['request_hash'] and q['protocol'] == PROTOCOL
        and q['train_jobs'] == jobs(PROTOCOL, 'train') and q['evaluation_jobs'] == jobs(PROTOCOL, 'evaluate')
        and a['request_hash'] == digest(q) and a['status'] == 'complete' and a['engineering_complete']
        and a['primary_node'] == 120000 and a['evaluation_episodes'] == 960 and not q['working_tree_dirty']
        and sha(root/'aggregate.json') == SETTINGS['aggregate_sha256']
        and receipt['request_hash'] == digest(q) and receipt['aggregate_sha256'] == SETTINGS['aggregate_sha256']
        and read(root/'preparation.json')['request_hash'] == digest(q)
        and not (root/'writer.lock').exists(), 'Frozen 120k parent changed/active')
    bound = {n: sha(root/n) for n in ('request.json', 'preparation.json', 'aggregate.json', 'aggregate_receipt.json')}
    for name in ('training_curves', 'development_curves'):
        require(sha(root/(name+'.json')) == receipt[name+'_sha256'], 'Historical curve index changed')
        bound[name+'.json'] = sha(root/(name+'.json'))
    # Current and historical sources are compared without running old loaders.
    verify_historical_inputs(baseline, {'python_lf:'+k: h for k, h in q['source_hashes'].items()}, q['git_commit'])
    source_equal = all(fingerprint(inside(ROOT, n).read_text(encoding='utf-8')) == h for n, h in q['source_hashes'].items())
    require(source_equal, 'Historical Raw execution source changed')
    for name, h in q['upstream_hashes'].items():
        require(fingerprint(inside(ROOT, name).read_text(encoding='utf-8')) == h, 'Upstream text semantics changed')
    upstream_config = 'RL-MPC-LaneMerging-master/configs/train_default_1.json'
    verify_historical_text_checkout(ROOT, upstream_config, q['upstream_config_sha256'], q['git_commit'])
    require(q['dependencies'] == smoke.dependencies(), 'Installed ALL source changed')
    require(q['preset'] == resolved_preset('cpu', .0002), 'Matched DDPG preset changed')
    for mode in SETTINGS['arms']:
        require(q['predictors'][mode]['sha256'] == shared.PREDICTORS[mode]
            and sha(inside(baseline, q['predictors'][mode]['path'])) == shared.PREDICTORS[mode], 'Frozen predictor differs')
    runtime = original.runtime_identity()
    historical_runtime = q['runtime_identity']
    runtime_nonbinary_equal = {k: v for k, v in runtime.items() if k != 'upstream_compiled_modules'} == {
        k: v for k, v in historical_runtime.items() if k != 'upstream_compiled_modules'}
    binary_equal = {k.replace('\\', '/'): v for k, v in runtime['upstream_compiled_modules'].items()} == {
        k.replace('\\', '/'): v for k, v in historical_runtime['upstream_compiled_modules'].items()}
    directory = ROOT/'artifacts/rcci04a_raw'/run_id; directory.mkdir(parents=True, exist_ok=False)
    request = {**shared.metadata(), 'config': SETTINGS, 'baseline_root': str(baseline),
        'source_hashes': shared.sources(), 'environment': shared.environment(), 'parent_files': bound,
        'design_sha256': sha(ROOT/SETTINGS['design'])}
    write_once(directory/'request.json', request)
    with run_lock(directory):
        report = {**shared.metadata(), 'status': 'running', 'request_hash': digest(request),
            'policy_loaded': False, 'weights_deserialized_for_inspection_only': True,
            'simulation_episodes': 0, 'policy_forward_calls': 0, 'training_performed': False}
        write_once(directory/'started.json', report)
        try:
            trained = {}; inventory = {}; evaluations = []
            for job in q['train_jobs']:
                if job['arm'] not in SETTINGS['arms']: continue
                d = root/'train'/job['id']; r = sealed(d, {'request_hash': digest(q), 'stage': 'train', 'job': job})
                m = r['metadata']
                require(m == a['training'][job['id']] and m['from_scratch'] and m['predictor_frozen']
                    and m['checkpoint_purpose'] == 'evaluation_only_not_training_resume'
                    and m['initial_weight_sha256'] == a['training']['zero_s'+str(job['run_seed'])]['initial_weight_sha256'],
                    'Historical matched initialization/training differs')
                require([s['requested_steps'] for s in m['snapshots']] == [20000, 40000, 60000, 120000], 'Snapshot roster differs')
                for snap in m['snapshots']:
                    require(snap['requested_steps'] <= snap['actual_steps'] <= snap['requested_steps']+499
                        and snap['weights_file'] == f'n{snap["requested_steps"]}.pt'
                        and sha(d/snap['weights_file']) == snap['weights_sha256']
                        and read(d/f'n{snap["requested_steps"]}.json') == snap, 'Historical snapshot differs')
                final = m['snapshots'][-1]
                expected = {'version': 'ddpg_feature_weights_v1', 'purpose': 'evaluation_only_not_training_resume',
                    'arm': job['arm'], 'observation_dimension': 169, 'time_feature_scale': .001,
                    'all_version': '0.5.3', 'preset': q['preset'], 'feature_version': VERSION,
                    'feature_names_sha256': digest(list(NAMES)), 'predictor_sha256': q['predictors'][job['arm']]['sha256'],
                    'upstream_config_sha256': q['upstream_config_sha256'],
                    'execution_contract': 'simulation_blocking_exact_v1', 'action_bounds': [-5., 5.]}
                require(final['weights_contract'] == expected, 'Historical final contract differs')
                check_weights(torch.load(d/final['weights_file'], map_location='cpu', weights_only=True), expected)
                trained[job['id']] = {'job': job, 'actual_steps': final['actual_steps'],
                    'weights_sha256': final['weights_sha256'], 'weights_contract': expected, 'snapshots': m['snapshots']}
                inventory['train/'+job['id']] = {'receipt_sha256': sha(d/'complete.json'), 'files': r['files']}
                print('[raw_reuse] verified '+job['id'], flush=True)
            for job in q['evaluation_jobs']:
                if job['arm'] not in SETTINGS['arms']: continue
                d = root/'evaluate'/job['id']; r = sealed(d, {'request_hash': digest(q), 'stage': 'evaluate', 'job': job})
                m = r['metadata']; outcome = read(d/'episode.json')
                snap = next(s for s in trained[job['training_id']]['snapshots'] if s['requested_steps'] == job['checkpoint_frames'])
                require(m['job'] == job and m['no_policy_updates'] and outcome == m['outcome']
                    and outcome['policy_sha256'] == snap['weights_sha256'] and outcome['actual_training_steps'] == snap['actual_steps']
                    and outcome['native_event_agrees'] and all(outcome.get(k) == v for k, v in job.items()),
                    'Historical evaluation/checkpoint join differs')
                evaluations.append({'job': job, 'policy_sha256': outcome['policy_sha256']})
                inventory['evaluate/'+job['id']] = {'receipt_sha256': sha(d/'complete.json'), 'files': r['files']}
            require(len(trained) == 6 and len(evaluations) == 480, 'Raw reuse roster incomplete')
            write_once(directory/'verified_inventory.json', inventory)
            report.update(status='complete', historical_evidence_complete=True, models=trained,
                verified_evaluation_episodes=480, primary_node_evaluation_episodes=120, unique_simulator_scenarios=20,
                historical_execution_source_equal=True, installed_all_source_equal=True,
                runtime_nonbinary_equal=runtime_nonbinary_equal, native_binary_identity_equal=binary_equal,
                original_runtime=historical_runtime, current_runtime=runtime,
                upstream_config_bytes_equal=sha(ROOT/upstream_config) == q['upstream_config_sha256'],
                upstream_config_only_git_line_endings_differ=sha(ROOT/upstream_config) != q['upstream_config_sha256'],
                upstream_config_original_sha256=q['upstream_config_sha256'], upstream_config_current_sha256=sha(ROOT/upstream_config),
                strict_runtime_identity_equal=runtime_nonbinary_equal and binary_equal,
                reuse_decision='historical_Raw_controls_valid_new_matched_reuse_requires_explicit_runtime_bridge',
                semantic_equivalence_of_recompiled_native_module_established=False,
                rcci_policy_initialization_allowed=False, automatic_retraining_required=False,
                verified_inventory_sha256=sha(directory/'verified_inventory.json'))
            require(shared.sources() == request['source_hashes'] and shared.git(ROOT, 'rev-parse', 'HEAD') == request['git_commit']
                and not shared.git(baseline, 'status', '--porcelain'), 'Source/baseline changed during audit')
            for n, h in bound.items(): require(sha(root/n) == h, 'Historical parent changed during audit')
            shared.clean()
        except BaseException:
            report.update(status='failed', failure_reason=traceback.format_exc()); raise
        finally:
            report['finished_at'] = shared.now(); write_once(directory/'report.json', report)
    write_once(directory/'receipt.json', {'request_hash': digest(request), 'report_sha256': sha(directory/'report.json')})
    print('raw_reuse_report='+str(directory/'report.json'), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline-root', required=True); p.add_argument('--run-id', required=True)
    args = p.parse_args(); audit(args.baseline_root, args.run_id)
