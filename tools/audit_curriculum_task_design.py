"""IC02c pure offline design audit. No collection, model load or training CLI."""
import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'tools')]
from prediction_rl.curriculum.task_design import (validate_design, proposal_jobs, replay_budget,
    existing_root_coverage, initial_description)
from prediction_rl.curriculum.train_reader import BASELINE_COMMIT, git, inside, strict_json, validate_config
from prediction_rl.data.collection_store import file_hash as sha, verify_episode, write_once
from prediction_rl.data.dataset_contract import digest
from prediction_rl.data.merge_geometry import MergeGeometry
from audit_curriculum_interaction import output_directory

CONFIG = 'configs/development/ic02c_task_design_v1.json'


def now():
    return datetime.now(timezone.utc).isoformat()


def run(args):
    started_at = now()
    if git(ROOT, 'status', '--porcelain') or git(ROOT, 'branch', '--show-current') != 'codex/interaction-curriculum-v1':
        raise ValueError('Commit reviewed offline design on the isolated research branch')
    hashes = {}
    def checked(root, name, expected=None):
        path = inside(root, name)
        raw = path.read_bytes()
        actual = hashlib.sha256(raw).hexdigest()
        if expected is not None and actual != expected:
            raise ValueError('Immutable input bytes changed: '+name)
        hashes[('worktree:' if Path(root).resolve() == ROOT else 'baseline:')+name] = actual
        return strict_json(raw)
    config = validate_design(checked(ROOT, CONFIG))
    baseline = Path(args.baseline_root).resolve()
    refs = checked(ROOT, config['baseline_references'])
    if (baseline != Path(refs['baseline_repository']).resolve() or ROOT != Path(refs['worktree']).resolve()
            or refs['baseline_commit'] != BASELINE_COMMIT or baseline == ROOT
            or git(baseline, 'rev-parse', 'HEAD') != BASELINE_COMMIT
            or git(baseline, 'branch', '--show-current') != 'main' or git(baseline, 'status', '--porcelain')):
        raise ValueError('Read-only baseline identity/clean state changed')
    for ref in refs['references']:
        path = inside(baseline, ref['relative_path'])
        if sha(path) != ref['sha256']:
            raise ValueError('Frozen baseline asset changed')
        hashes['baseline:'+ref['relative_path']] = ref['sha256']
    review_ref = next(r for r in refs['references'] if r['role'] == 'predictor_dataset_review')
    data_review = checked(baseline, review_ref['relative_path'], review_ref['sha256'])
    request_name = 'artifacts/data/response_v1/request.json'
    normalized = {key.replace('\\','/'): value for key,value in data_review['input_hashes'].items()}
    old_request = checked(baseline, request_name, normalized[request_name])
    if (data_review['test_locked'] is not True or old_request['test_locked'] is not True
            or old_request['protocol']['baseline']['reference_checkpoint'] != config['reference_checkpoint']
            or old_request['protocol']['baseline']['reference_sha256'] != config['reference_sha256']):
        raise ValueError('Need the unchanged training-side reference policy, no test access')
    ref_path = inside(baseline, config['reference_checkpoint'])
    if sha(ref_path) != config['reference_sha256']:
        raise ValueError('Frozen author reference checkpoint changed')
    hashes['baseline:'+config['reference_checkpoint']] = config['reference_sha256']
    for name in ('RL-MPC-LaneMerging-master/merge.net.xml', config['upstream_config'],
                 'src/prediction_rl/data/actor_adapter.py', 'src/prediction_rl/data/merge_geometry.py',
                 'src/prediction_rl/data/reference_policy.py', 'src/prediction_rl/data/branching.py',
                 'src/prediction_rl/data/episode_collector.py'):
        path, original = inside(ROOT, name), inside(baseline, name)
        if path.read_bytes().replace(b'\r\n',b'\n') != original.read_bytes().replace(b'\r\n',b'\n'):
            raise ValueError('Frozen geometry/adapter/collector source changed')
        hashes['worktree:'+name], hashes['baseline:'+name] = sha(path), sha(original)
    old_report = checked(ROOT, config['ic01_report'], config['ic01_report_sha256'])
    roots = checked(ROOT, str(Path(config['ic01_report']).parent/'roots.json').replace('\\','/'), old_report['roots_sha256'])
    mapping = checked(ROOT, config['entry_mapping'], config['entry_mapping_sha256'])
    entry_review = checked(ROOT, config['entry_review'], config['entry_review_sha256'])
    if (entry_review['status'] != 'complete' or entry_review['artifact_parity_gate'] is not True
            or entry_review['training_ready'] is not False or entry_review['new_episodes'] != 0
            or entry_review['full_original_engineering_gate_recovered'] is not False):
        raise ValueError('Need the limited sealed-entry review, not an invented full engineering gate')
    native_request_name = str(Path(entry_review['original_request']).relative_to(ROOT)).replace('\\','/')
    native_request = checked(ROOT, native_request_name, entry_review['original_file_hashes']['request.json'])
    native_root = inside(ROOT, native_request_name).parent
    geometry = MergeGeometry(ROOT/'RL-MPC-LaneMerging-master/merge.net.xml')
    root_config = validate_config(checked(ROOT, 'configs/development/ic01_train_interaction_v1.json'))
    if root_config != old_report['resolved_config']:
        raise ValueError('IC01 roles/metrics configuration changed')
    initial = []
    for job in native_request['jobs']:
        if job['adapter'] != 'original_direct' or job['repeat'] != 0:
            continue
        directory = inside(native_root, job['id'])
        expected = entry_review['episode_file_hashes'][job['id']]
        receipt = verify_episode(directory, {'request_hash': digest(native_request), 'job': job})
        if sha(directory/'complete.json') != expected['complete.json'] or receipt['files'] != {k:v for k,v in expected.items() if k != 'complete.json'}:
            raise ValueError('Sealed native evidence inventory differs from offline review')
        name = (directory/'trace.json').relative_to(ROOT).as_posix()
        trace = checked(ROOT, name, expected['trace.json'])
        initial.append(initial_description(trace, geometry, root_config))
    proposed = proposal_jobs(mapping, config)
    out = output_directory(ROOT, baseline, args.run_id)
    report = {'run_id': args.run_id, 'experiment_id': config['protocol_id'], 'status': 'complete',
        'git_commit_sha': git(ROOT, 'rev-parse', 'HEAD'), 'working_tree_dirty': False,
        'command': subprocess.list2cmdline(sys.orig_argv), 'started_at': started_at,
        'resolved_config': config, 'resolved_config_hash': digest(config), 'input_hashes': hashes,
        'source_hashes': {name: sha(ROOT/name) for name in (CONFIG, 'tools/audit_curriculum_task_design.py',
            'src/prediction_rl/curriculum/task_design.py', 'src/prediction_rl/curriculum/interaction.py',
            'src/prediction_rl/data/actor_adapter.py', 'src/prediction_rl/data/merge_geometry.py',
            'src/prediction_rl/data/branching.py', 'src/prediction_rl/data/episode_collector.py')},
        'environment': {'python': sys.version, 'os': platform.platform(), 'numpy': version('numpy'),
            'torch_installed_not_loaded': version('torch'), 'sumo_started': False},
        'old_root_coverage': existing_root_coverage(roots, old_report), 'initial_entry_descriptions': initial,
        'engineering_stage_budget_proposal': replay_budget(len(config['engineering_seeds']), config),
        'pilot_total_budget_proposal': replay_budget(len(proposed), config),
        'pilot_includes_engineering_tasks_without_duplicate_executions': True,
        'legacy_collector_supports_repeated_same_lane_checkpoints': False,
        'dedicated_checkpoint_observer_required': True, 'runtime_collector_implemented': False,
        'runtime_authorized': False, 'sumo_started': False, 'predictor_inference_run': False,
        'ddpg_started': False, 'test_opened': False, 'new_counterfactual_branches': 0,
        'task_classifier': None, 'target_distribution_pi': None, 'training_ready': False,
        'remaining_gate': 'approve_finite_collector_scope_then_audit_task_coverage_and_freeze_task_rules'}
    if (len(initial) != 3 or git(ROOT, 'status', '--porcelain') or git(baseline, 'status', '--porcelain')
            or git(baseline, 'rev-parse', 'HEAD') != BASELINE_COMMIT):
        raise ValueError('Offline inputs/worktree changed during audit')
    for key, expected in hashes.items():
        label, relative = key.split(':', 1)
        if sha(inside(ROOT if label == 'worktree' else baseline, relative)) != expected:
            raise ValueError('Input changed during offline audit')
    out.mkdir(parents=True, exist_ok=False)
    write_once(out/'task_proposals.json', {'status': 'not_executable_not_classified', 'records': proposed})
    report.update(finished_at=now(), engineering_gate=True,
                  task_proposals_sha256=sha(out/'task_proposals.json'))
    write_once(out/'report.json', report)
    print('ic02c_design_report='+str(out/'report.json'))
    print('runtime_authorized=False training_ready=False')
    return out/'report.json'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-root', required=True)
    parser.add_argument('--run-id', required=True)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
