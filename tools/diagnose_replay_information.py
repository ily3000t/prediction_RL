"""Manual-start P7m; read saved replay, never start SUMO or train DDPG."""
import argparse
from copy import deepcopy
import gzip
import json
from pathlib import Path
import sys
import traceback

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'tools')]
import train_response_predictors as shared
import compare_gym20_external as external
import smoke_feature_ddpg as smoke
from prediction_rl.data.collection_store import (
    read_json as read, write_once, file_hash as sha, run_lock, seal_episode, verify_episode)
from prediction_rl.data.dataset_contract import digest
from prediction_rl.data.frozen_dataset import inside, verify_historical_inputs
from prediction_rl.evaluation.replay_information import (
    PROTOCOL, AUDIT, jobs, validate_protocol, select_states, analyze, aggregate)
from prediction_rl.training.all_ddpg import preset_factory, load_evaluation_weights


def runtime():
    return {'source_hashes': shared.source_hashes(), 'dependencies': smoke.dependencies(), 'environment': shared.env()}


def evidence(*, verify_files):
    """Historical hashes remain frozen; new source files do not relax old checks."""
    inputs, project = external.parent_evidence(verify_files=verify_files)
    ap = inside(ROOT, PROTOCOL['external_aggregate']); root = ap.parent
    q = read(root/'request.json'); a = read(ap); current_sources = shared.source_hashes()
    if (sha(ap) != PROTOCOL['external_aggregate_sha256'] or digest(q) != PROTOCOL['external_request_hash']
            or q['protocol'] != external.PROTOCOL or q['jobs'] != external.jobs(external.PROTOCOL)
            or q['parent_inputs'] != inputs or a['status'] != 'complete' or not a['engineering_complete']
            or a['request_hash'] != digest(q) or a['new_author_episodes'] != 80
            or read(root/'aggregate_receipt.json') != {'request_hash': digest(q), 'aggregate_sha256': sha(ap)}
            or (root/'writer.lock').exists()
            or any(current_sources.get(k) != v for k, v in q['runtime']['source_hashes'].items())):
        raise ValueError('P7l external comparison evidence changed or remains active')
    if verify_files:
        verify_historical_inputs(ROOT, {'python_lf:'+k: v for k, v in q['runtime']['source_hashes'].items()}, q['git_commit'])
    def record(path): inputs[str(path.relative_to(ROOT))] = sha(path)
    for name in ('request.json', 'aggregate.json', 'aggregate_receipt.json'): record(root/name)
    for job in q['jobs']:
        directory = root/'evaluate'/job['id']
        receipt = verify_episode(directory, external.binding(q, job)) if verify_files else read(directory/'complete.json')
        if (receipt['status'] != 'complete' or receipt['binding'] != external.binding(q, job)
                or receipt['metadata']['outcome'] != read(directory/'episode.json')):
            raise ValueError('P7l author episode/receipt changed')
        for name in ('complete.json', 'episode.json'): record(directory/name)
    parent = inside(ROOT, PROTOCOL['parent_request']).parent
    models = []
    for job in jobs(PROTOCOL):
        d = parent/'train'/job['id']; receipt = read(d/'complete.json'); snap = read(d/'n60000.json')
        if (d/'writer.lock').exists(): raise ValueError('Active training input')
        if (snap['actual_steps'] != receipt['metadata']['actual_steps']
                or snap['weights_contract']['arm'] != job['arm']
                or snap['weights_contract']['preset']['device'] != 'cpu'):
            raise ValueError('P7k final replay/checkpoint mismatch')
        # Verify ONLY files used by this diagnostic against the original sealed
        # inventory. No need to reopen old losses/raw events or bootstrap audit.
        for name in ('replay.jsonl.gz', 'n60000.json', 'n60000.pt'):
            if sha(d/name) != receipt['files'][name]: raise ValueError('Frozen training input changed: '+name)
            record(d/name)
        models.append({**job, 'directory': str(d.relative_to(ROOT)), 'actual_steps': snap['actual_steps'],
                       'weights_contract': snap['weights_contract'], 'weights_sha256': snap['weights_sha256']})
    failures = [{k: r[k] for k in ('arm', 'run_seed', 'simulator_seed', 'natural_arrival', 'ego_collision', 'time_limit', 'return')}
                for r in project if r['arm'] in PROTOCOL['arms'] and not r['natural_arrival']]
    return inputs, models, failures


def prepare(config, run_id, audit=False):
    shared.clean(); torch.set_num_threads(1)
    if validate_protocol(read(inside(ROOT, config))) != PROTOCOL:
        raise ValueError('Use the full frozen configuration; --audit selects the bounded engineering prefix')
    inputs, models, failures = evidence(verify_files=True)
    p = deepcopy(AUDIT if audit else PROTOCOL)
    out = shared.output_dir('p7m', run_id)
    q = {**shared.metadata(), 'version': 'p07m_request_v1', 'protocol': p,
         'config_path': config, 'config_sha256': sha(inside(ROOT, config)),
         'runtime': runtime(), 'input_hashes': inputs, 'models': models, 'evaluation_failure_cells': failures}
    write_once(out/'request.json', q)
    write_once(out/'preparation.json', {'request_hash': digest(q), 'status': 'prepared_not_diagnosed',
        'models': len(models), 'maximum_queried_states_per_model': p['sample']['uniform_cap']+4*p['sample']['stratum_cap'],
        'simulation_episodes': 0, 'training_permitted': False})
    print('replay_information_request='+str(out/'request.json'), flush=True)
    print('confirm_request_hash='+digest(q), flush=True)
    return out/'request.json', digest(q)


def load(path, expected):
    shared.clean(); torch.set_num_threads(1); path = inside(ROOT, path); q = read(path)
    validate_protocol(q['protocol'])
    if (path.parent.parent != ROOT/'artifacts/p7m' or q['version'] != 'p07m_request_v1'
            or digest(q) != expected or read(path.parent/'preparation.json')['request_hash'] != expected
            or q['runtime'] != runtime() or sha(inside(ROOT, q['config_path'])) != q['config_sha256']
            or read(inside(ROOT, q['config_path'])) != PROTOCOL):
        raise ValueError('Frozen source/configuration/runtime/request changed')
    inputs, models, failures = evidence(verify_files=False)
    if q['input_hashes'] != inputs or q['models'] != models or q['evaluation_failure_cells'] != failures:
        raise ValueError('Frozen parent/result/replay/checkpoint inputs changed')
    return path.parent, q


class SpaceOnlyEnv:
    """Original preset shape construction, with NO reset/step/SUMO interface."""
    def __init__(self, arm):
        import gym
        self.feature_arm = arm
        self.state_space = gym.spaces.Box(-np.inf, np.inf, (169,), np.float32)
        self.action_space = gym.spaces.Box(-5., 5., (1,), np.float32)


def loaded_agent(model):
    from all.logging import DummyWriter
    contract = model['weights_contract']
    agent = preset_factory('cpu', contract['preset']['lr_pi'])(SpaceOnlyEnv(model['arm']), DummyWriter())
    load_evaluation_weights(agent, inside(ROOT, model['directory'])/'n60000.pt', contract, model['weights_sha256'])
    return agent


def binding(q, model):
    return {'request_hash': digest(q), 'stage': 'read_only_replay_information', 'model': model}


def model_run(directory, q, model, resume):
    b = binding(q, model)
    if directory.exists():
        if not resume: raise ValueError('Use --resume only for completely sealed model diagnostics')
        return verify_episode(directory, b)['metadata']
    directory.mkdir(parents=True)
    write_once(directory/'started.json', {**shared.metadata(), 'binding': b})
    try:
        with gzip.open(inside(ROOT, model['directory'])/'replay.jsonl.gz', 'rt', encoding='utf-8') as handle:
            selected, selection = select_states((json.loads(line) for line in handle), model, q['protocol'], model['actual_steps'])
        result = analyze(loaded_agent(model), selected, selection, q['protocol'])
        write_once(directory/'selected_states.json', selected)
        write_once(directory/'action_rows.json', result.pop('action_rows'))
        row = {**result, 'job': {k: model[k] for k in ('id', 'arm', 'run_seed', 'checkpoint_frames')},
               'policy_sha256': model['weights_sha256']}
        write_once(directory/'report.json', row); seal_episode(directory, b, row)
        print(f"[replay_information] {model['id']} scanned={selection['scanned_transitions']} selected={len(selected)} complete", flush=True)
        return row
    except BaseException as error:
        write_once(directory/'failure.json', {'status': 'failed', 'error': f'{type(error).__name__}: {error}',
                                            'traceback': traceback.format_exc()})
        raise


def execute(path, expected, resume=False):
    out, q = load(path, expected)
    with run_lock(out):
        rows = [model_run(out/'models'/m['id'], q, m, resume) for m in q['models']]
        result = {**aggregate(q['protocol'], rows), 'request_hash': digest(q), 'provenance': shared.metadata(),
                  'evaluation_failure_cells': q['evaluation_failure_cells'],
                  'evaluation_failure_cells_are_context_not_replay_attribution': True, 'finished_at': shared.now()}
        path = out/'aggregate.json'
        if path.exists():
            previous = read(path)
            if (not resume or read(out/'aggregate_receipt.json') != {'request_hash': digest(q), 'aggregate_sha256': sha(path)}
                    or any(previous[k] != v for k, v in result.items() if k not in ('provenance', 'finished_at'))):
                raise ValueError('Completed immutable aggregate changed; no overwrite')
        else:
            write_once(path, result)
            write_once(out/'aggregate_receipt.json', {'request_hash': digest(q), 'aggregate_sha256': sha(path)})
    print('replay_information_aggregate='+str(path), flush=True)
    return path


def main():
    p = argparse.ArgumentParser(description=__doc__); s = p.add_subparsers(dest='command', required=True)
    a = s.add_parser('prepare'); a.add_argument('--config', default='configs/development/p07m_replay_information_v1.json')
    a.add_argument('--run-id', required=True); a.add_argument('--audit', action='store_true')
    a = s.add_parser('run'); a.add_argument('--request', required=True)
    a.add_argument('--confirm-request-hash', required=True); a.add_argument('--resume', action='store_true')
    args = p.parse_args()
    if args.command == 'prepare': prepare(args.config, args.run_id, args.audit)
    else: execute(args.request, args.confirm_request_hash, args.resume)


if __name__ == '__main__': main()
