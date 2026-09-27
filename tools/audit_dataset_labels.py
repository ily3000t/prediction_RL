"""Read-only P3 source conversion into development-only masked label fixtures."""
import argparse
from datetime import datetime, timezone
import json
import platform
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from audit_environment import sha, write
from prediction_rl.data.dataset_contract import build_branch_labels, digest, episode_id, validate_rows, SPLITS


def load_config(path):
    config = json.loads(Path(path).read_text(encoding='utf-8'))
    if set(config) != {'schema_version', 'source_report', 'source_report_sha256', 'horizon_steps',
                       'tick_s', 'split', 'training_ready'}:
        raise ValueError('Unknown/missing label audit fields')
    if (type(config['schema_version']) is not int or config['schema_version'] != 1
            or type(config['horizon_steps']) is not int or config['horizon_steps'] != 25
            or type(config['tick_s']) not in (int, float) or config['tick_s'] != .2
            or config['split'] != 'development' or config['training_ready'] is not False):
        raise ValueError('This audit is a fixed development-only label fixture')
    source = Path(config['source_report'])
    if source.is_absolute() or '..' in source.parts or not (ROOT / source).resolve().is_relative_to(ROOT / 'artifacts/p3'):
        raise ValueError('Source must be a project-relative P3 artifact')
    if not isinstance(config['source_report_sha256'], str) or not re.fullmatch('[0-9a-f]{64}', config['source_report_sha256']):
        raise ValueError('Pin the source report SHA256')
    return config


def run(config_path, run_id):
    if not re.fullmatch('[A-Za-z0-9_-]{1,40}', run_id):
        raise ValueError('Use a short unique run ID')
    config = load_config(config_path)
    source_path = ROOT / config['source_report']
    if sha(source_path) != config['source_report_sha256']:
        raise ValueError('Source report hash mismatch')
    source = json.loads(source_path.read_text(encoding='utf-8'))
    if source['status'] != 'complete' or source['engineering_gate'] is not True or source['plan']['schema_version'] != 3:
        raise ValueError('Require complete audited v3 reference-prefix evidence')
    if source['plan']['horizon_steps'] != config['horizon_steps']:
        raise ValueError('Source horizon mismatch')
    output = ROOT / 'artifacts/p4' / run_id
    output.mkdir(parents=True, exist_ok=False)
    git = lambda *args: subprocess.check_output(['git', '-C', str(ROOT), *args], text=True, encoding='utf-8')
    report = {'status': 'running', 'git_commit': git('rev-parse', 'HEAD').strip(),
              'working_tree_dirty': bool(git('status', '--porcelain').strip()),
              'command': subprocess.list2cmdline([sys.executable, *sys.argv]),
              'python_executable': sys.executable, 'python_version': sys.version,
              'os': platform.platform(),
              'started_at': datetime.now(timezone.utc).isoformat(), 'config_sha256': sha(Path(config_path)),
              'source_report_sha256': sha(source_path), 'source_git_commit': source['git_commit'],
              'source_environment_versions': source['environment_versions'], 'training_ready': False,
              'training_blockers': ['continuous_history_not_recorded', 'actor_adapter_not_implemented',
                                    'formal_split_manifest_not_approved'],
              'execution_contract': source['execution_contract'], 'source_file_hashes': {}}
    try:
        if report['working_tree_dirty']:
            raise ValueError('Run the audit from a clean commit to pin all conversion code')
        write(output / 'resolved_config.json', config)
        environment_hash = digest({k: v for k, v in source['source_sha256'].items()
                                   if k.startswith('RL-MPC-LaneMerging-master')})
        manifest = {split: [] for split in SPLITS}
        index, events, branch_count, roots_count = [], {}, 0, 0
        for child in source['children']:
            if child['status'] != 'complete' or child['intervention_s'] != config['tick_s']:
                raise ValueError('Incomplete or inconsistent child evidence')
            episode = episode_id(environment_hash, child['seed'], 0)
            manifest['development'].append(episode)
            for root in child['roots']:
                if root['status'] != 'complete':
                    raise ValueError('Missing diagnostic root; not silently discarded')
                path = source_path.parent / f"seed_{child['seed']}" / f"root_{root['prefix_steps']}_branches.json"
                report['source_file_hashes'][str(path.relative_to(ROOT))] = sha(path)
                data = json.loads(path.read_text(encoding='utf-8'))
                if data['root'] != root or data['branches'] != data['reverse_order_repeats']:
                    raise ValueError('Root/branch evidence differs from source report or repeat')
                if set(data['branches']) != {'0', '1', '2', '3', '4'}:
                    raise ValueError('Incomplete candidate family')
                root_id = digest([episode, root['signature']])
                examples = []
                for candidate in range(5):
                    plan = [child['probe_jerks'][candidate]] + [source['plan']['continuation_jerk']] * (config['horizon_steps']-1)
                    labels = build_branch_labels(root['root_traffic'], data['branches'][str(candidate)], plan, config['tick_s'])
                    examples.append({'candidate_id': candidate, 'labels': labels})
                    index.append({'episode_id': episode, 'root_id': root_id, 'candidate_id': candidate, 'split': 'development'})
                    reason = labels['termination_reason'] or 'horizon_nonterminal'
                    events[reason] = events.get(reason, 0) + 1
                    branch_count += 1
                write(output / f'r{roots_count:02d}.json', {'schema_version': 1, 'episode_id': episode,
                      'root_id': root_id, 'simulator_seed': child['seed'], 'episode_index': 0,
                      'split': 'development', 'root_traffic': root['root_traffic'],
                      'prefix_actions': root['prefix_actions'], 'history_available': False,
                      'history': None, 'training_ready': False, 'candidates': examples})
                roots_count += 1
        validate_rows(index, manifest)
        write(output / 'split_manifest.json', manifest)
        write(output / 'index.json', index)
        report.update(status='complete', label_contract_gate=True, episode_count=len(manifest['development']),
                      root_count=roots_count, candidate_count=branch_count, terminal_counts=events,
                      environment_identity_sha256=environment_hash,
                      output_sha256={p.name: sha(p) for p in output.glob('*.json')})
    except Exception as error:
        report.update(status='failed', label_contract_gate=False, failure_reason=f'{type(error).__name__}: {error}')
        raise
    finally:
        report['finished_at'] = datetime.now(timezone.utc).isoformat()
        write(output / 'report.json', report)
    print('p4_label_report=' + str(output / 'report.json'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    run(args.config, args.run_id)
