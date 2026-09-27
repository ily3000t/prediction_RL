"""Offline, untrained P4c acceptance using nine pinned P4b input roots."""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import re
import subprocess
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from audit_environment import sha, write
from prediction_rl.data.branching import fingerprint
from prediction_rl.prediction.interface import PredictorConfig, collate_inputs, probe_plans
from prediction_rl.prediction.model import PredictorEnsemble
from prediction_rl.prediction.checkpoint import save_checkpoint, load_checkpoint


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def load_config(path):
    config = read(path)
    keys = {'schema_version', 'source_history_report', 'source_history_sha256',
            'member_initialization_seeds', 'model', 'device', 'torch_threads',
            'absolute_tolerance', 'relative_tolerance'}
    if not isinstance(config, dict) or set(config) != keys:
        raise ValueError('Unknown/missing interface audit fields')
    if (type(config['schema_version']) is not int or config['schema_version'] != 1 or
            config['device'] != 'cpu' or type(config['torch_threads']) is not int or config['torch_threads'] != 1 or
            config['member_initialization_seeds'] != [3101, 3102, 3103] or
            any(type(s) is not int for s in config['member_initialization_seeds']) or
            config['absolute_tolerance'] != 1e-5 or config['relative_tolerance'] != 1e-5):
        raise ValueError('Bounded v1 audit settings are frozen')
    if PredictorConfig.from_dict(config['model']) != PredictorConfig():
        raise ValueError('Bounded v1 audit uses the declared 64-wide model')
    source = Path(config['source_history_report'])
    if source.is_absolute() or '..' in source.parts or not (ROOT / source).resolve().is_relative_to(ROOT):
        raise ValueError('Require project-relative source')
    if config['source_history_sha256'] != '89d4f73676a1fa6816421731cc30ee4d6f379d3e6d971879ccfdda20cfbb6c8c':
        raise ValueError('Use the accepted P4b development report')
    if sha(ROOT / source) != config['source_history_sha256']:
        raise ValueError('History evidence changed')
    return config


def load_roots(config):
    source_path = ROOT / config['source_history_report']
    source = read(source_path)
    if (source['status'] != 'complete' or not source['history_actor_gate'] or
            source['formal_training_ready'] or source['root_count'] != 9):
        raise ValueError('Require complete development history gate')
    if [child['seed'] for child in source['children']] != [0, 1, 100]:
        raise ValueError('Do not replace the accepted simulator seeds')
    rows, records = [], []
    for child in source['children']:
        if child['status'] != 'complete' or len(child['roots']) != 3:
            raise ValueError('Incomplete source roots')
        for index, root in enumerate(child['roots']):
            path = source_path.parent / f"seed_{child['seed']}" / f'r{index}.json'
            if sha(path) != root['artifact_sha256']:
                raise ValueError('Pinned history input changed')
            payload = read(path)
            if (payload['split'] != 'development' or payload['formal_training_ready'] or
                    payload['shared_by_candidates'] != list(range(5)) or
                    payload['root_id'] != root['root_id'] or
                    fingerprint(payload['inputs']) != root['input_hash']):
                raise ValueError('Root input provenance mismatch')
            rows.append(payload['inputs'])
            records.append({'root_id': root['root_id'], 'simulator_seed': child['seed'],
                            'path': str(path.relative_to(ROOT)), 'sha256': sha(path)})
    if len({r['root_id'] for r in records}) != 9:
        raise ValueError('Duplicate root family')
    return rows, records


def run(args):
    config = load_config(args.config)
    if not re.fullmatch('[A-Za-z0-9_-]{1,40}', args.run_id):
        raise ValueError('Use a short unique run ID')
    git = lambda *a: subprocess.check_output(['git', '-C', str(ROOT), *a], text=True, encoding='utf-8').strip()
    if git('status', '--porcelain'):
        raise ValueError('Acceptance requires a clean committed source tree')
    rows, records = load_roots(config)
    output = ROOT / 'artifacts/p4' / args.run_id
    output.mkdir(parents=True, exist_ok=False)
    report = {'status': 'running', 'git_commit': git('rev-parse', 'HEAD'), 'working_tree_dirty': False,
              'command': subprocess.list2cmdline([sys.executable, *sys.argv]),
              'started_at': datetime.now(timezone.utc).isoformat(), 'config_sha256': sha(Path(args.config)),
              'source_history_sha256': config['source_history_sha256'], 'roots': records,
              'training_status': 'untrained_interface_smoke', 'formal_training_ready': False,
              'execution_contract': 'offline_interface_only_no_simulation', 'optimizer_seed': None,
              'member_initialization_seeds': config['member_initialization_seeds'],
              'models': {}, 'interface_gate': False,
              'environment_versions': {'python': sys.version, 'python_executable': sys.executable,
                   'torch': importlib.metadata.version('torch'), 'numpy': importlib.metadata.version('numpy'),
                   'os': platform.platform(), 'processor': platform.processor(), 'device': config['device']}}
    write(output / 'resolved_config.json', config)
    report['resolved_config_sha256'] = sha(output / 'resolved_config.json')
    started, previous_threads = time.perf_counter(), torch.get_num_threads()
    try:
        torch.set_num_threads(config['torch_threads'])
        cfg = PredictorConfig.from_dict(config['model'])
        batch = collate_inputs(rows, cfg)
        plans = probe_plans(len(rows), cfg)
        report['candidate_plans_sha256'] = fingerprint(plans.tolist())
        provenance = {'training_status': 'untrained_interface_smoke', 'git_commit': report['git_commit'],
                      'source_history_sha256': config['source_history_sha256'], 'config_sha256': report['config_sha256']}
        with torch.inference_mode():
            for mode in ('ordinary', 'conditional'):
                model = PredictorEnsemble(cfg, mode, config['member_initialization_seeds']).eval()
                calls = []
                hooks = [m.scene_encoder.register_forward_hook(lambda *unused: calls.append(1)) for m in model.members]
                full = model(batch, plans)
                for hook in hooks: hook.remove()
                if len(calls) != 3: raise AssertionError('Scene must encode once per root batch per member')
                values = full['member_trajectories']
                if tuple(values.shape) != (3, 9, 5, 12, 25, 4) or not torch.isfinite(values).all():
                    raise AssertionError('Invalid full response tensor')
                serial = torch.cat([model(batch, plans[:, i:i+1])['member_trajectories'] for i in range(5)], 2)
                root_serial = torch.cat([model({k:v[i:i+1] for k,v in batch.items()}, plans[i:i+1])['member_trajectories'] for i in range(9)], 1)
                for other in (serial, root_serial):
                    torch.testing.assert_close(values, other, atol=config['absolute_tolerance'], rtol=config['relative_tolerance'])
                spread = (values[:, :, 0] - values[:, :, -1]).abs().max().item()
                if mode == 'ordinary' and spread != 0: raise AssertionError('Ordinary mode leaked candidate values')
                if mode == 'conditional' and spread == 0: raise AssertionError('Conditional path disconnected')
                checkpoint = output / (mode + '.pt')
                save_checkpoint(checkpoint, model, provenance)
                restored, restored_provenance = load_checkpoint(checkpoint, expected_mode=mode, expected_config=cfg)
                if restored_provenance != provenance or not torch.equal(values, restored(batch, plans)['member_trajectories']):
                    raise AssertionError('Checkpoint roundtrip changed output/provenance')
                report['models'][mode] = {
                    'parameter_count_per_member': sum(p.numel() for p in model.members[0].parameters()),
                    'member_count': 3, 'response_shape': list(values.shape), 'finite': True,
                    'scene_encoding_calls': len(calls), 'candidate_serial_max_abs_error': (values-serial).abs().max().item(),
                    'root_serial_max_abs_error': (values-root_serial).abs().max().item(),
                    'candidate_information_path_verified': True, 'checkpoint_roundtrip_exact': True,
                    'checkpoint': str(checkpoint.relative_to(ROOT)), 'checkpoint_sha256': sha(checkpoint)}
                print(f'[p4c] {mode}: batched/serial and checkpoint checks passed', flush=True)
        if any(sha(ROOT/r['path']) != r['sha256'] for r in records) or sha(ROOT/config['source_history_report']) != config['source_history_sha256']:
            raise AssertionError('Source history evidence changed during read-only audit')
        report.update(status='complete', interface_gate=True, root_count=9, candidate_count=45,
                      source_evidence_unchanged=True, next_stage='P4d_label_alignment_and_masked_training_interfaces')
    except Exception as error:
        report.update(status='failed', failure_reason=f'{type(error).__name__}: {error}')
        raise
    finally:
        torch.set_num_threads(previous_threads)
        report.update(finished_at=datetime.now(timezone.utc).isoformat(), elapsed_s=time.perf_counter()-started)
        write(output / 'report.json', report)
    return output / 'report.json'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--run-id', required=True)
    print(run(parser.parse_args()))


if __name__ == '__main__':
    main()
