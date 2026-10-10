"""IC02b sealed-artifact review only. No simulation, inference or learner calls."""
import argparse
import ast
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'tools')]

from prediction_rl.curriculum.entry_smoke import PROTOCOL, compare_traces, jobs, validate_protocol, validate_trace
from prediction_rl.curriculum.train_reader import BASELINE_COMMIT, git, inside
from prediction_rl.data.collection_store import file_hash as sha, write_once
from prediction_rl.data.dataset_contract import digest
from prediction_rl.data.merge_geometry import MergeGeometry
from audit_curriculum_interaction import output_directory
from smoke_curriculum_entry import inputs, read, read_sealed_trace, require_clean

ORIGINAL_SOURCE_COMMIT = '5bef30bac6e08249e77431fbfd29096c01ca1fc7'
RUNNER = 'tools/smoke_curriculum_entry.py'
REVIEW_CONTRACT = {
    'version': 'ic02b_sealed_artifact_review_v1',
    'original_source_commit': ORIGINAL_SOURCE_COMMIT,
    'only_allowed_changed_input': 'worktree:'+RUNNER,
    'comparison': 'unchanged_exact_native_and_feature_parity',
    'new_episodes': 0, 'new_control_steps': 0, 'predictor_inference_run': False,
    'sumo_started': False, 'ddpg_started': False, 'training_ready': False,
    'rewrite_original_report': False,
}


def now():
    return datetime.now(timezone.utc).isoformat()


def runtime_files(expected):
    """Verify executable/dependency bytes without invoking SUMO --version."""
    binary = shutil.which('sumo.exe') or shutil.which('sumo')
    extensions = list((ROOT/'RL-MPC-LaneMerging-master').glob('st_cy*.pyd'))
    if binary is None or len(extensions) != 1:
        raise ValueError('Original runtime executable/extension is unavailable')
    binary = Path(binary).resolve()
    actual = {'python_executable': sys.executable, 'python': sys.version, 'os': platform.platform(),
        'sumo_executable': str(binary), 'sumo_sha256': sha(binary),
        'SUMO_HOME': os.environ.get('SUMO_HOME'), 'extension_path': str(extensions[0]),
        'extension_sha256': sha(extensions[0]),
        'packages': {name: importlib.metadata.version(name) for name in expected['packages']}}
    if any(value != expected.get(key) for key, value in actual.items()):
        raise ValueError('Original runtime files/dependencies changed')
    return {**actual, 'sumo_version_recorded_in_original_request': expected['sumo_version'],
            'sumo_version_command_reexecuted': False}


def verify_historical_inputs(q, current):
    previous = q['input_hashes']
    key = REVIEW_CONTRACT['only_allowed_changed_input']
    if set(previous) != set(current) or key not in previous:
        raise ValueError('Original/current input inventory differs')
    changed = {name for name in previous if previous[name] != current[name]}
    if not changed <= {key}:
        raise ValueError('Inputs other than the aggregation runner changed: '+str(sorted(changed)))
    original = subprocess.check_output(['git', '-C', str(ROOT), 'show', f'{ORIGINAL_SOURCE_COMMIT}:{RUNNER}'])
    if hashlib.sha256(original).hexdigest() != previous[key]:
        raise ValueError('Historical runner blob differs from the original recorded bytes')
    return {'original_runner_sha256': previous[key], 'current_runner_sha256': current[key],
            'changed_input_names': sorted(changed), 'verified_input_files': len(previous),
            'historical_runner_verified_from_git': True,
            'comparator_and_runtime_sources_unchanged': True}


def validate_original_failure(failure, q, expected_hash):
    if (failure['status'] != 'failed' or failure['engineering_gate'] is not False
            or failure['request_hash'] != expected_hash or failure['git_commit_sha'] != q['git_commit_sha']
            or failure['working_tree_dirty'] is not False or failure['protocol'] != q['protocol']
            or failure['failure_type'] != 'ValueError'
            or failure['failure_reason'] != 'Native/feature trajectory mismatch; exact parity cannot be relaxed'
            or failure['runtime_parity_verified'] is not False or failure['training_ready'] is not False
            or failure['ddpg_started'] is not False or failure['test_opened'] is not False
            or failure['optimizer_updates'] != 0 or failure['new_counterfactual_branches'] != 0
            or [row['job'] for row in failure['rows']] != q['jobs']):
        raise ValueError('Not the completed IC02b aggregation-only failure under review')
    if any(name in failure for name in ('predictor_tensor_hashes', 'predictors_frozen', 'parent_rng_unchanged')):
        raise ValueError('Unexpected original postflight evidence; review scope must be reconsidered')


def representation_diagnostic(traces):
    """Reconstruct the verified source literal, not a missing process snapshot."""
    tree = ast.parse((ROOT/'RL-MPC-LaneMerging-master/config.py').read_text(encoding='utf-8'))
    assignments = [node for node in ast.walk(tree) if isinstance(node, ast.Assign)
                   and any(isinstance(target, ast.Name) and target.id == 'JERK_VALUES_DQN'
                           for target in node.targets)]
    if len(assignments) != 1:
        raise ValueError('Need the unchanged upstream settings literal')
    native_keys = ast.literal_eval(assignments[0].value)
    if not isinstance(native_keys, dict) or not all(type(key) is int for key in native_keys):
        raise ValueError('Original numeric settings-key definition changed')
    left = traces['d11000r0']
    saved_worker = traces['w11000r0']
    json_keys = {str(key): value for key, value in native_keys.items()}
    if (left['runtime']['settings']['JERK_VALUES_DQN'] != json_keys
            or saved_worker['runtime']['settings']['JERK_VALUES_DQN'] != json_keys):
        raise ValueError('Source settings values do not match the sealed evidence')
    reconstructed = deepcopy(saved_worker)
    reconstructed['runtime']['settings']['JERK_VALUES_DQN'] = native_keys
    try:
        compare_traces(left, reconstructed)
    except ValueError as error:
        if str(error) != 'Native/feature trajectory mismatch; exact parity cannot be relaxed':
            raise
    else:
        raise ValueError('The original representation mismatch was not reproduced')
    compare_traces(left, json.loads(json.dumps(reconstructed, allow_nan=False)))
    return {'field': 'runtime.settings.JERK_VALUES_DQN', 'source': 'RL-MPC-LaneMerging-master/config.py',
        'in_memory_literal_key_type': 'int', 'sealed_json_key_type': 'str',
        'source_based_reconstruction_not_original_memory_capture': True,
        'mismatch_reproduced_with_unchanged_comparator': True,
        'json_roundtrip_passes_unchanged_comparator': True,
        'numeric_values_unchanged': True}


def load_original(path, expected_hash):
    require_clean()
    path = Path(path).resolve()
    if not path.is_relative_to(ROOT/'artifacts/ic') or path.name != 'request.json':
        raise ValueError('Review request must be a local immutable IC artifact')
    q = read(path)
    preparation = read(path.parent/'preparation.json')
    if (q['version'] != 'ic02b_bounded_entry_request_v1' or digest(q) != expected_hash
            or preparation != {'request_hash': expected_hash, 'status': 'prepared_not_executed'}
            or q['git_commit_sha'] != ORIGINAL_SOURCE_COMMIT or q['working_tree_dirty'] is not False
            or q['approval'] != {'source': 'human_continue_after_explicit_ic02b_12_episode_144_step_question',
                                 'date': '2026-10-10', 'training_authorized': False}):
        raise ValueError('Original frozen request/preparation/approval changed')
    config = validate_protocol(q['protocol'])
    current, ensembles, training_request = inputs(q['baseline_root'], config)
    if (q['jobs'] != jobs(config) or q['predictors'] != ensembles
            or q['predictor_training_request'] != training_request
            or git(q['baseline_root'], 'branch', '--show-current') != 'main'):
        raise ValueError('Frozen jobs/predictors/baseline branch changed')
    source_verification = verify_historical_inputs(q, current)
    environment = runtime_files(q['runtime'])
    if (path.parent/'report.json').exists() or (path.parent/'writer.lock').exists():
        raise ValueError('Original run is not an inactive failed run')
    failure = read(path.parent/'failure.json')
    validate_original_failure(failure, q, expected_hash)
    started = read(path.parent/'started.json')
    if (started['request_hash'] != expected_hash or started['git_commit_sha'] != ORIGINAL_SOURCE_COMMIT
            or started['status'] != 'running' or started['rows'] != [] or started['comparisons'] != []
            or started['started_at'] != failure['started_at']):
        raise ValueError('Original start/failure receipt binding changed')
    hashes = {name: sha(path.parent/name) for name in ('request.json', 'preparation.json', 'started.json', 'failure.json')}
    return path, q, failure, source_verification, environment, hashes


def review_jobs(path, q, failure):
    geometry = MergeGeometry(ROOT/'RL-MPC-LaneMerging-master/merge.net.xml')
    traces, rows, hashes = {}, [], {}
    for job, original_row in zip(q['jobs'], failure['rows']):
        directory = inside(path.parent, job['id'])
        trace, row = read_sealed_trace(directory, {'request_hash': digest(q), 'job': job})
        checked = validate_trace(trace, job, geometry)
        if (row != original_row or row['job'] != job or row['parent_rng_unchanged'] is not True
                or any(row.get(key) != value for key, value in checked.items())):
            raise ValueError('Original episode metadata/RNG witness differs from sealed trace')
        started = read(directory/'started.json')
        if (started['request_hash'] != digest(q) or started['job'] != job
                or started['git_commit_sha'] != ORIGINAL_SOURCE_COMMIT or started['status'] != 'running'):
            raise ValueError('Episode start receipt changed')
        receipt = read(directory/'complete.json')
        hashes[job['id']] = {'complete.json': sha(directory/'complete.json'), **receipt['files']}
        traces[job['id']] = trace
        rows.append(row)
    comparisons = []
    for seed in PROTOCOL['simulator_seeds']:
        for left, right in ((f'd{seed}r0', f'd{seed}r1'), (f'w{seed}r0', f'w{seed}r1'),
                            (f'd{seed}r0', f'w{seed}r0'), (f'd{seed}r1', f'w{seed}r1')):
            comparisons.append({'left': left, 'right': right, **compare_traces(traces[left], traces[right])})
    if failure['comparisons'] != comparisons[:len(failure['comparisons'])] or len(failure['comparisons']) != 2:
        raise ValueError('Original completed comparison prefix differs')
    counts = {'actual_episodes': len(rows), 'actual_control_steps': sum(r['actual_control_steps'] for r in rows),
        'waiting_insertion_ticks': sum(r['warmup_and_insertion_ticks'] for r in rows),
        'raw_ticks': sum(r['raw_ticks'] for r in rows),
        'feature_calls_per_mode': sum(r['feature_calls_per_mode'] for r in rows),
        'native_terminal_episodes': sum(r['native_done'] for r in rows),
        'unique_simulator_scenes': len({r['job']['simulator_seed'] for r in rows})}
    if (counts['actual_episodes'] != PROTOCOL['episodes_max']
            or counts['actual_control_steps'] > PROTOCOL['control_steps_max']
            or counts['waiting_insertion_ticks'] > PROTOCOL['warmup_and_insertion_ticks_max']
            or counts['feature_calls_per_mode'] > PROTOCOL['feature_calls_per_mode_max']):
        raise ValueError('Recorded engineering budget exceeded or episode grid incomplete')
    return {'rows': rows, 'comparisons': comparisons, 'counts': counts, 'episode_file_hashes': hashes,
            'representation_diagnostic': representation_diagnostic(traces)}


def execute(path, expected_hash, run_id):
    path, q, failure, source_checks, environment, original_hashes = load_original(path, expected_hash)
    out = output_directory(ROOT, Path(q['baseline_root']).resolve(), run_id)
    report = {'run_id': run_id, 'experiment_id': REVIEW_CONTRACT['version'], 'review_contract': REVIEW_CONTRACT,
        'resolved_config': q['protocol'], 'resolved_config_hash': digest(q['protocol']),
        'git_commit_sha': git(ROOT, 'rev-parse', 'HEAD'), 'working_tree_dirty': False,
        'command': subprocess.list2cmdline(sys.orig_argv), 'started_at': now(), 'status': 'running',
        'original_request': str(path), 'original_request_hash': expected_hash,
        'original_run_commit': ORIGINAL_SOURCE_COMMIT, 'baseline_commit': BASELINE_COMMIT,
        'original_file_hashes': original_hashes, 'source_verification': source_checks,
        'environment': environment, 'original_status': 'failed', 'original_engineering_gate': False,
        'reviewer_sha256': sha(Path(__file__)), 'training_ready': False, 'test_opened': False,
        'sumo_started': False, 'predictor_inference_run': False, 'ddpg_started': False,
        'new_episodes': 0, 'new_control_steps': 0, 'optimizer_updates': 0,
        'post_run_predictor_tensor_snapshot_available': False,
        'original_global_rng_postflight_available': False,
        'full_original_engineering_gate_recovered': False,
        'limitations': ['original process failed before final in-memory tensor/global-RNG checks',
                        'per-episode RNG witnesses do not replace the missing final global postflight',
                        'short live traces contain no native terminal; terminal transport uses offline fixtures',
                        'task-layer coverage, target pi and persistent learner/replay integration remain unresolved']}
    out.mkdir(parents=True, exist_ok=False)
    write_once(out/'started.json', report)
    try:
        results = review_jobs(path, q, failure)
        # Recheck all immutable inputs/receipts, without opening a simulator or model.
        _, _, _, after_sources, _, after_hashes = load_original(path, expected_hash)
        repeated = review_jobs(path, q, failure)
        if (after_sources != source_checks or after_hashes != original_hashes or repeated != results
                or git(ROOT, 'rev-parse', 'HEAD') != report['git_commit_sha']):
            raise ValueError('Review input/source bytes changed during offline aggregation')
        report.update(results, status='complete', finished_at=now(), artifact_parity_gate=True,
            runtime_parity_verified=True, per_episode_parent_rng_unchanged=True,
            predictor_file_hashes_unchanged=True, original_failure_preserved=True,
            task_classifier=None, target_distribution_pi=None,
            root_cause='in_memory_integer_settings_keys_vs_persisted_json_string_keys',
            remaining_gate='task_layer_coverage_and_target_distribution_not_approved')
        write_once(out/'report.json', report)
    except BaseException as error:
        report.update(status='failed', finished_at=now(), artifact_parity_gate=False,
            failure_type=type(error).__name__, failure_reason=str(error), traceback=traceback.format_exc())
        write_once(out/'failure.json', report)
        raise
    print('ic02b_offline_review='+str(out/'report.json'), flush=True)
    print('artifact_parity_gate=True training_ready=False new_episodes=0', flush=True)
    return out/'report.json'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', required=True)
    parser.add_argument('--confirm-request-hash', required=True)
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    execute(args.request, args.confirm_request_hash, args.run_id)


if __name__ == '__main__':
    main()
