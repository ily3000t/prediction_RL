"""IC02a: read-only source and root-task entry audit, NOT an entry implementation."""
import argparse
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'tools')]
from prediction_rl.curriculum.entry_design import (
    SOURCE_PATHS, inspect_source, mapping_summary, smoke_budget, validate_design)
from prediction_rl.curriculum.train_reader import BASELINE_COMMIT, git, inside, strict_json
from prediction_rl.data.collection_store import write_once
from prediction_rl.data.dataset_contract import digest
from audit_curriculum_interaction import output_directory


def run(args):
    started_at = datetime.now(timezone.utc).isoformat()
    if git(ROOT, 'status', '--porcelain'):
        raise ValueError('Commit tested offline design before audit')
    config_path = inside(ROOT, args.config); config_raw = config_path.read_bytes()
    config = validate_design(strict_json(config_raw))
    baseline = Path(args.baseline_root).resolve()
    refs_path = inside(ROOT, config['baseline_references']); refs = strict_json(refs_path.read_bytes())
    if (baseline != Path(refs['baseline_repository']).resolve()
            or ROOT.resolve() != Path(refs['worktree']).resolve()
            or refs['baseline_commit'] != BASELINE_COMMIT
            or git(baseline, 'rev-parse', 'HEAD') != BASELINE_COMMIT or git(baseline, 'status', '--porcelain')):
        raise ValueError('Isolated worktree/pinned baseline identity changed')
    hashes = {}
    def read_bytes(root, relative, expected=None):
        raw = inside(root, relative).read_bytes(); value = hashlib.sha256(raw).hexdigest()
        if expected is not None and value != expected:
            raise ValueError('Frozen input changed: '+relative)
        hashes[('baseline:' if root == baseline else 'worktree:')+relative] = value
        return raw
    for ref in refs['references']:
        read_bytes(baseline, ref['relative_path'], ref['sha256'])
    report_path = config['ic01_report']
    report = strict_json(read_bytes(ROOT, report_path, config['ic01_report_sha256']))
    if (report['status'] != 'complete' or report['engineering_gate'] is not True
            or report['test_opened'] is not False or report['scope'] != 'train_only'
            or report['training_ready'] is not False or report['baseline_commit'] != BASELINE_COMMIT):
        raise ValueError('Need the complete immutable IC01 train audit')
    roots_path = (Path(report_path).parent/'roots.json').as_posix()
    rows = strict_json(read_bytes(ROOT, roots_path, report['roots_sha256']))
    sources = {}
    for relative in SOURCE_PATHS:
        raw = read_bytes(ROOT, relative); original = read_bytes(baseline, relative)
        if raw.replace(b'\r\n', b'\n') != original.replace(b'\r\n', b'\n'):
            raise ValueError('Original execution source differs from baseline: '+relative)
        if relative.endswith('.py'):
            sources[relative] = raw.decode('utf-8-sig')
        else:
            upstream_config = strict_json(raw)
    source_contract = inspect_source(sources, upstream_config)
    mapping = mapping_summary(rows, report)
    for relative in ('src/prediction_rl/curriculum/entry_design.py', 'tools/audit_curriculum_entry.py',
                     config['baseline_references'], args.config):
        read_bytes(ROOT, relative)
    if git(baseline, 'rev-parse', 'HEAD') != BASELINE_COMMIT or git(baseline, 'status', '--porcelain'):
        raise ValueError('Baseline changed during offline inspection')
    out = output_directory(ROOT, baseline, args.run_id)
    out.mkdir(parents=True, exist_ok=False)
    result = {'status': 'complete', 'experiment_id': config['protocol_id'], 'run_id': args.run_id,
        'git_commit_sha': git(ROOT, 'rev-parse', 'HEAD'), 'working_tree_dirty': False,
        'baseline_commit': BASELINE_COMMIT, 'baseline_root': str(baseline),
        'launch_command': subprocess.list2cmdline(sys.orig_argv),
        'started_at': started_at,
        'finished_at': datetime.now(timezone.utc).isoformat(),
        'environment': {'python': sys.version, 'os': platform.platform()},
        'optimizer_seed': None, 'simulator_seed': None, 'execution_contract': 'offline_no_simulator',
        'resolved_design': config, 'resolved_design_hash': digest(config),
        'source_contract': source_contract, 'mapping': {k:v for k,v in mapping.items() if k != 'records'},
        'proposed_smoke_budget': smoke_budget(config['smoke_proposal']),
        'input_hashes': hashes, 'static_inspection_gate': True, 'runtime_parity_verified': False,
        'new_entry_implemented': False, 'new_entry_approved': False, 'sumo_started': False,
        'predictor_inference_run': False, 'ddpg_started': False, 'test_opened': False,
        'training_ready': False, 'required_decision': 'approve_named_new_entry_and_bounded_smoke_before_implementation'}
    write_once(out/'episode_mapping.json', mapping)
    result['episode_mapping_sha256'] = hashlib.sha256((out/'episode_mapping.json').read_bytes()).hexdigest()
    write_once(out/'report.json', result)
    print('ic02a_report='+str(out/'report.json'))
    print('runtime_parity_verified=False; training_ready=False')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/development/ic02a_entry_design_v1.json')
    parser.add_argument('--baseline-root', required=True)
    parser.add_argument('--run-id', required=True)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
