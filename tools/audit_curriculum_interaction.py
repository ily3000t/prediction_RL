"""IC01 only: verified offline train analysis. Never starts SUMO or DDPG."""
import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
from pathlib import Path
import platform
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from prediction_rl.curriculum.interaction import analyze_root, stratify, summarize
from prediction_rl.curriculum.train_reader import TrainReader, strict_json, inside, git, validate_config
from prediction_rl.data.collection_store import write_once
from prediction_rl.data.dataset_contract import digest


def now():
    return datetime.now(timezone.utc).isoformat()


def output_directory(worktree, baseline, run_id):
    if not re.fullmatch(r'[a-z][a-z0-9_]{0,23}', run_id):
        raise ValueError('Use a short lowercase run ID')
    out = inside(worktree, 'artifacts/ic/'+run_id)
    if out.is_relative_to(Path(baseline).resolve()) or out.exists():
        raise ValueError('Refuse baseline writes or output overwrite')
    return out


def run(args):
    if git(ROOT, 'status', '--porcelain'):
        raise ValueError('Commit tested analysis/config before this immutable offline audit')
    path = inside(ROOT, args.config)
    config = validate_config(strict_json(path.read_bytes()))
    reader = TrainReader(ROOT, args.baseline_root, config)
    out = output_directory(ROOT, reader.baseline, args.run_id)
    source_paths = [Path(__file__), path, inside(ROOT, config['baseline_references']),
        *sorted((ROOT/'src/prediction_rl/curriculum').glob('*.py')),
        *[ROOT/'src/prediction_rl/data'/n for n in ('actor_adapter.py', 'merge_geometry.py',
                                                  'dataset_contract.py', 'collection_store.py')]]
    source_hashes = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in source_paths}
    provenance = {'run_id': args.run_id, 'experiment_id': config['protocol_id'],
        'git_commit_sha': git(ROOT, 'rev-parse', 'HEAD'), 'working_tree_dirty': False,
        'command': subprocess.list2cmdline([sys.executable, *sys.argv]), 'started_at': now(),
        'baseline_commit': reader.references['baseline_commit'], 'baseline_root': str(reader.baseline),
        'resolved_config': config, 'resolved_config_hash': digest(config), 'source_hashes': source_hashes,
        'environment': {'python': sys.version, 'os': platform.platform(), 'numpy': version('numpy'),
                        'torch_installed_not_loaded': version('torch'), 'sumo': 'not_started'},
        'optimizer_seed': None, 'scope': 'train_only', 'test_opened': False,
        'predictor_inference_run': False, 'sumo_started': False, 'ddpg_started': False,
        'training_ready': False, 'thresholds_status': 'provisional_not_training_release'}
    out.mkdir(parents=True, exist_ok=False)
    write_once(out/'started.json', provenance)
    try:
        rows = [analyze_root(h, p, b, r, reader.geometry, config, binding)
                for h, p, b, r, binding in reader.root_families()]
        classification = stratify(rows, config)
        summary = summarize(rows, reader.episodes, reader.unavailable, classification)
        tasks = [{'task_id': r['job_id']+'_'+r['target'], 'source_episode_id': r['episode_id'],
                  'traffic_seed': r['simulator_seed'], 'reference_root_id': r['root_id'],
                  'source_root_directory': r['source_root_directory'], 'provisional_root_level': r['level'],
                  'mapping': r['task_mapping'], 'predictor_input_coverage': r['root_context']['coverage'],
                  'execution_authorized': False} for r in rows]
        reader.verify_references()
        if git(reader.baseline, 'rev-parse', 'HEAD') != reader.references['baseline_commit'] or git(reader.baseline, 'status', '--porcelain'):
            raise ValueError('Baseline changed during audit')
        write_once(out/'roots.json', rows)
        write_once(out/'task_candidates.json', {'status': 'not_executable_task_pool', 'records': tasks,
            'unavailable_roots': reader.unavailable, 'original_target_distribution_pi': None})
        write_once(out/'input_inventory.json', reader.hashes)
        report = {**provenance, 'status': 'complete', 'finished_at': now(), 'engineering_gate': True,
            'summary': summary, 'read_file_operations': reader.files_read,
            'verified_unique_input_files': len(reader.hashes), 'future_split_reads': ['train'],
            'repeat_noise_method': 'verified_actual_byte_exact_pairs_then_identity_grid_difference',
            'roots_sha256': hashlib.sha256((out/'roots.json').read_bytes()).hexdigest(),
            'input_inventory_hash': digest(reader.hashes), 'pause_before_training': True,
            'remaining_gate': 'root_to_executable_task_entry_and_target_distribution_unresolved'}
        write_once(out/'report.json', report)
    except BaseException as error:
        write_once(out/'failure.json', {**provenance, 'status': 'failed', 'engineering_gate': False,
            'finished_at': now(), 'failure_reason': f'{type(error).__name__}: {error}'})
        raise
    print('ic01_report='+str(out/'report.json'))
    print('ic01_levels='+str(classification['levels']))
    print('training_ready=False')
    return out/'report.json'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/development/ic01_train_interaction_v1.json')
    parser.add_argument('--baseline-root', required=True)
    parser.add_argument('--run-id', required=True)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
