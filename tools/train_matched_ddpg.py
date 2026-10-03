"""P7k user-run from-scratch matched learning curves. No weights-only continuation."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import gzip
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback
import uuid

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'tools')]
import train_response_predictors as shared
import explore_feature_ddpg as original
import diagnose_feature_ddpg as recording
import smoke_feature_ddpg as smoke
import audit_training_contract as precheck
from prediction_rl.data.collection_store import (read_json as read, write_once, file_hash as sha,
                                                  run_lock, seal_episode, verify_episode)
from prediction_rl.data.dataset_contract import digest
from prediction_rl.data.frozen_dataset import inside, verify_historical_inputs
from prediction_rl.envs.upstream import upstream_session
from prediction_rl.evaluation.closed_loop_diagnostics import PROTOCOL as P7B
from prediction_rl.evaluation.exploration import EpisodeMetrics
from prediction_rl.evaluation.matched_learning_curve import (PROTOCOL, SMOKE, jobs, aggregate,
                                                             validate_protocol, snapshot_steps)
from prediction_rl.evaluation.terminal_events import StepObserver
from prediction_rl.evaluation.terminal_events_v2 import classify
from prediction_rl.evaluation.training_contract import verify_reward, state_measurement
from prediction_rl.training.all_ddpg import (resolved_preset, preset_factory, checkpoint_contract,
                                             experiment_type, save_evaluation_weights, load_evaluation_weights)
from prediction_rl.training.matched_learning_curve import StreamingReplay, Jsonl, train_nodes, Coverage

PRECHECK = ROOT/'artifacts/p7j/p7j_audit_v1/aggregate.json'
PRECHECK_SHA = '8ec67ea67d9be2f29238accb6563488636322ae4426c89292a9cbc12b38de7a9'


def prerequisites():
    pp, parent = recording.load_parent(P7B)
    a = read(PRECHECK); q = read(PRECHECK.parent/'request.json')
    if (sha(PRECHECK) != PRECHECK_SHA or a['status'] != 'complete'
            or a['training_transition_semantics_gate'] is not True
            or a['initial_gym20_parity_gate'] is not True or a['test_opened'] is not False
            or a['request_hash'] != digest(q) or q['config'] != precheck.CONFIG
            or q['jobs'] != precheck.JOBS or q['dependencies'] != smoke.dependencies()
            or q['runtime'] != original.runtime_identity()
            or read(PRECHECK.parent/'preparation.json')['request_hash'] != digest(q)
            or read(PRECHECK.parent/'aggregate_receipt.json') != {
                'request_hash': digest(q), 'aggregate_sha256': sha(PRECHECK)}
            or q['request_inputs'] != {'parent_request_hash': digest(parent),
                                     'parent_aggregate_sha256': sha(pp.parent/'aggregate.json')}):
        raise ValueError('Accepted P7j prechecks changed or absent')
    if (PRECHECK.parent/'writer.lock').exists(): raise ValueError('Unresolved P7j writer lock')
    verify_historical_inputs(ROOT, {'python_lf:'+k: v for k, v in q['source_hashes'].items()}, q['git_commit'])
    if any(shared.source_hashes().get(k) != v for k, v in q['source_hashes'].items()):
        raise ValueError('Audited training implementation changed')
    actual = [verify_episode(PRECHECK.parent/j['id'], precheck.binding(q, j))['metadata'] for j in precheck.JOBS]
    if actual != a['jobs']: raise ValueError('P7j receipts differ from aggregate')
    return parent, {'parent_request_hash': digest(parent), 'precheck_sha256': sha(PRECHECK),
                    'precheck_request_hash': digest(q)}


def binding(q, stage, job): return {'request_hash': digest(q), 'stage': stage, 'job': job}


def prepare(config_path, run_id):
    shared.clean(); config_path = inside(ROOT, config_path)
    config = validate_protocol(read(config_path)); parent, inputs = prerequisites()
    torch.set_num_threads(1)
    out = shared.output_dir('p7k', run_id)
    t = config['training']
    q = {**shared.metadata(), 'version': 'p07k_request_v1', 'protocol': deepcopy(config),
         'config_path': str(config_path.relative_to(ROOT)), 'config_sha256': sha(config_path),
         'request_inputs': inputs, 'source_hashes': shared.source_hashes(),
         'upstream_hashes': original.upstream_hashes(), 'dependencies': smoke.dependencies(),
         'environment': shared.env(), 'runtime_identity': original.runtime_identity(),
         'upstream_config_sha256': sha(original.UPSTREAM),
         'preset': resolved_preset(t['device'], t['learning_rate'], smoke=config == SMOKE),
         'predictors': parent['predictors'], 'predictor_training_request': parent['predictor_training_request'],
         'train_jobs': jobs(config, 'train'), 'evaluation_jobs': jobs(config, 'evaluate')}
    write_once(out/'request.json', q); write_once(out/'preparation.json', {'request_hash': digest(q),
        'status': 'prepared_not_trained', 'requested_training_instances': len(q['train_jobs']),
        'requested_total_training_steps': len(q['train_jobs'])*t['requested_frames'],
        'evaluation_episodes': len(q['evaluation_jobs']), 'unique_traffic_scenarios': len(config['evaluation']['simulator_seeds'])})
    print('matched_request='+str(out/'request.json'), flush=True)
    print('confirm_request_hash='+digest(q), flush=True)
    return out/'request.json', digest(q)


def load_request(path, expected_hash):
    shared.clean(); torch.set_num_threads(1); path = inside(ROOT, path); q = read(path)
    p = validate_protocol(q['protocol']); t = p['training']
    if (path.parent.parent != ROOT/'artifacts'/'p7k' or q['version'] != 'p07k_request_v1'
            or digest(q) != expected_hash or read(path.parent/'preparation.json')['request_hash'] != expected_hash
            or q['source_hashes'] != shared.source_hashes() or q['dependencies'] != smoke.dependencies()
            or q['environment'] != shared.env() or q['runtime_identity'] != original.runtime_identity()
            or q['upstream_hashes'] != original.upstream_hashes() or q['upstream_config_sha256'] != sha(original.UPSTREAM)
            or q['preset'] != resolved_preset(t['device'], t['learning_rate'], smoke=p == SMOKE)
            or q['config_sha256'] != sha(inside(ROOT, q['config_path'])) or read(inside(ROOT, q['config_path'])) != p
            or q['train_jobs'] != jobs(p, 'train') or q['evaluation_jobs'] != jobs(p, 'evaluate')):
        raise ValueError('Frozen source/config/runtime/request changed; no silent adaptation')
    parent, inputs = prerequisites()
    if (q['request_inputs'] != inputs or q['predictors'] != parent['predictors']
            or q['predictor_training_request'] != parent['predictor_training_request']):
        raise ValueError('Prechecks or frozen predictor identities changed')
    return path, q


class TrainingRecorder:
    def __init__(self, observer, settings, out, arm):
        self.observer, self.settings, self.arm = observer, settings, arm
        self.replay = None; self.current = None; self.start = 0
        self.episodes = []; self.raw = Jsonl(out/'raw_training_events.jsonl.gz')
        self.action_count = 0; self.saturated = 0; self.max_reward_error = 0.
        self.current_raw = None
    def begin(self):
        # Original continuing SUMO/RNG is NOT restarted. Only audit buffers clear.
        self.observer.events.clear(); self.observer.removals.clear(); self.observer.creations.clear()
        self.current_raw = None
    def record(self, event, obs, reward, done, info):
        if self.arm == 'zero' and np.any(obs[20:]): raise ValueError('Zero channels changed')
        if event == 'reset':
            self.current = EpisodeMetrics(); self.start = len(self.observer.events)
            self.initial = state_measurement(obs, info)
        self.current.record(event, obs, reward, done, info)
        self.replay.record(event, obs, reward, done, info)
    def verify_row(self, row):
        self.action_count += 1; self.saturated += int(abs(row['requested_jerk']) >= 4.99)
        if row['done']:
            if len(self.observer.creations) != 1: raise ValueError('Episode creation mismatch')
            self.current_raw = precheck.raw_outcome(self.observer, self.start, [row], self.observer.creations[0])
        reward = verify_reward([row], self.settings)
        self.max_reward_error = max(self.max_reward_error, reward['max_absolute_reward_error'])
    def end(self, actual):
        if self.current_raw is None or self.replay.last_row is None or not self.replay.last_row['done']:
            raise ValueError('Episode not fully stored and raw-event checked')
        r = {**self.current.result(), 'finished_training_step': actual, 'raw_outcome': self.current_raw,
             'initial_state': self.initial}
        self.episodes.append(r)
        if sum(e['steps'] for e in self.episodes) != actual: raise ValueError('Episode/replay step misalignment')
        self.raw.append({'episode': len(self.episodes)-1, 'finished_training_step': actual,
                         'events': self.observer.export(), 'outcome': self.current_raw})
        self.raw.flush(); self.replay.flush()
    def summary(self):
        return {'completed_episodes': len(self.episodes), 'training_action_count': self.action_count,
                'action_saturation_fraction': self.saturated/self.action_count if self.action_count else None,
                'max_absolute_reward_error': self.max_reward_error,
                'training_returns_are_exploration_not_validation': True}


def train_job(q, job, out):
    p = q['protocol']; t = p['training']; provider, predictor_sha = original.provider_for(q, job['arm'])
    frozen = None if provider is None else smoke.tensor_hash(provider.model.state_dict())
    with upstream_session(original.SOURCE, original.UPSTREAM, job['run_seed']) as settings:
        import control, traci
        if settings.LEARNING_RATE != t['learning_rate']: raise ValueError('Upstream learning rate changed')
        with StepObserver(control, traci, {'id': job['id'], 'condition_id': 'gym20'}) as observer:
            recorder = TrainingRecorder(observer, settings, out, job['arm'])
            env = original.environment(q, job, provider, recorder.record); experiment = None; wrapper = None
            snapshots = []
            try:
                base = experiment_type(out/'training_logs')
                class RecordedExperiment(base):
                    def _run_training_episode(self):
                        recorder.begin()
                        super()._run_training_episode()
                        recorder.end(self.frame-1)
                experiment = RecordedExperiment(preset_factory('cpu', settings.LEARNING_RATE, smoke=p == SMOKE), env, quiet=False)
                checked = experiment._agent; core = checked.agent
                wrapper = StreamingReplay(checked, out, recorder); recorder.replay = wrapper
                initial = {n: smoke.tensor_hash(a.model.state_dict()) for n, a in [('policy', core.policy), ('q', core.q)]}
                contract = checkpoint_contract(env, q['preset'], predictor_sha, q['upstream_config_sha256'])
                write_once(out/'resolved_config.json', {'upstream': settings.export_settings(), 'protocol': p,
                    'preset': q['preset'], 'job': job, 'predictor_sha256': predictor_sha, 'weights_contract': contract})
                def snapshot(n, actual):
                    if actual != wrapper.insertions or not env.done: raise ValueError('Snapshot not at stored episode boundary')
                    path = out/f'n{n}.pt'
                    weights_sha = save_evaluation_weights(checked, path, contract)
                    item = {'requested_steps': n, 'actual_steps': actual, 'weights_sha256': weights_sha,
                        'weights_file': path.name, 'weights_contract': contract,
                        'optimizer_updates': {name: smoke.optimizer_steps(a) for name, a in [('policy', core.policy), ('q', core.q)]},
                        'all_frames_seen_including_reset_calls': core._frames_seen,
                        'coverage': wrapper.summary(), 'training': recorder.summary(),
                        'optimizer_replay_sumo_reinitialized': False}
                    write_once(out/f'n{n}.json', item); snapshots.append(item)
                    print(f'[p7k snapshot] {job["id"]} requested={n} actual={actual}', flush=True)
                train_nodes(experiment, t['checkpoint_frames'], snapshot)
                if provider is not None and (smoke.tensor_hash(provider.model.state_dict()) != frozen
                        or any(x.grad is not None or x.requires_grad for x in provider.model.parameters())):
                    raise ValueError('Predictor changed during policy training')
                write_once(out/'episodes.json', recorder.episodes)
                return {'status': 'complete', 'job': job, 'from_scratch': True,
                        'initial_weight_sha256': initial, 'snapshots': snapshots,
                        'actual_steps': experiment.frame-1, 'coverage': wrapper.summary(),
                        'training': recorder.summary(), 'predictor_frozen': True,
                        'checkpoint_purpose': 'evaluation_only_not_training_resume',
                        'no_automatic_budget_extension': True}
            finally:
                if wrapper is not None: wrapper.close()
                recorder.raw.close()
                try: env.close()
                finally:
                    if experiment is not None: experiment._writer.close()


def training_metadata(q, root):
    trained = {j['id']: verify_episode(root/'train'/j['id'], binding(q, 'train', j))['metadata'] for j in q['train_jobs']}
    for s in q['protocol']['training']['run_seeds']:
        initial = [trained[f'{a}_s{s}']['initial_weight_sha256'] for a in ('zero', 'ordinary', 'conditional')]
        if any(i != initial[0] for i in initial[1:]): raise ValueError('Matched expanded initialization changed')
    for j in q['train_jobs']:
        m = trained[j['id']]
        if (m['job'] != j or m['from_scratch'] is not True or m['predictor_frozen'] is not True
                or [s['requested_steps'] for s in m['snapshots']] != q['protocol']['training']['checkpoint_frames']):
            raise ValueError('Training lineage/node roster changed')
        for s in m['snapshots']:
            snapshot_steps(s['actual_steps'], s['requested_steps'])
            d = root/'train'/j['id']
            if (s['weights_file'] != f'n{s["requested_steps"]}.pt' or sha(d/s['weights_file']) != s['weights_sha256']
                    or read(d/f'n{s["requested_steps"]}.json') != s): raise ValueError('Snapshot changed')
    return trained


def evaluation_job(q, job, out, root):
    tj = next(j for j in q['train_jobs'] if j['id'] == job['training_id'])
    td = root/'train'/tj['id']; m = verify_episode(td, binding(q, 'train', tj))['metadata']
    n = job['checkpoint_frames']; snap = next(s for s in m['snapshots'] if s['requested_steps'] == n)
    if read(td/f'n{n}.json') != snap: raise ValueError('Checkpoint receipt mismatch')
    provider, predictor_sha = original.provider_for(q, job['arm'])
    frozen = None if provider is None else smoke.tensor_hash(provider.model.state_dict())
    metrics = EpisodeMetrics(); trace = []; decisions = []; measurements = []
    def record(event, obs, reward, done, info):
        metrics.record(event, obs, reward, done, info)
        if job['arm'] == 'zero' and np.any(obs[20:]): raise ValueError('Zero channels changed')
        measurements.append(state_measurement(obs, info))
        if event == 'step': decisions.append({'frame': info['execution_audit']['before']})
    identity = {'id': job['id'], 'model_id': 'p7k_'+job['arm'], 'condition_id': 'gym20',
                'run_seed': job['run_seed'], 'simulator_seed': job['simulator_seed'], 'training_id': job['training_id']}
    with upstream_session(original.SOURCE, original.UPSTREAM, job['simulator_seed']) as settings:
        import control, traci
        with StepObserver(control, traci, identity) as observer:
            env = original.environment(q, job, provider, record)
            try:
                from all.logging import DummyWriter
                checked = preset_factory('cpu', settings.LEARNING_RATE, smoke=q['protocol'] == SMOKE)(env, DummyWriter())
                contract = checkpoint_contract(env, q['preset'], predictor_sha, q['upstream_config_sha256'])
                if contract != snap['weights_contract']: raise ValueError('Training/evaluation observation/action mismatch')
                load_evaluation_weights(checked, td/snap['weights_file'], contract, snap['weights_sha256'])
                initial = {k: smoke.tensor_hash(a.model.state_dict()) for k, a in [('policy', checked.agent.policy), ('q', checked.agent.q)]}
                write_once(out/'resolved_config.json', {'upstream': settings.export_settings(), 'protocol': q['protocol'],
                    'preset': q['preset'], 'job': job, 'snapshot': snap, 'predictor_sha256': predictor_sha})
                state = env.reset(); reward = 0.
                for step in range(500):
                    action = checked.eval(state, reward)
                    state, reward = env.step(action)
                    trace.append({'step': step, 'jerk': float(action.item()), 'reward': float(reward), 'done': bool(env.done)})
                    if env.done:
                        checked.eval(state, reward); break
                else: raise ValueError('Original 500-step limit changed')
                native = metrics.result()
                raw = classify(observer.export(), {**identity, **native, 'reported_collision': native['collision']}, decisions)
                if raw['classification'] not in ('natural_arrival', 'ego_collision', 'time_limit') or not raw['native_event_agrees']:
                    raise ValueError('Native and raw terminal semantics disagree; invalid run, not relabeled reward')
                if (len(checked.agent.replay_buffer) or any(a._optimizer.state or smoke.tensor_hash(a.model.state_dict()) != initial[k]
                        for k, a in [('policy', checked.agent.policy), ('q', checked.agent.q)])):
                    raise ValueError('Frozen evaluation trained or populated replay')
                if provider is not None and (smoke.tensor_hash(provider.model.state_dict()) != frozen
                        or any(p.grad is not None or p.requires_grad for p in provider.model.parameters())):
                    raise ValueError('Evaluation changed frozen predictor')
                cov = Coverage(); cov.add([Coverage.vector(r) for r in measurements[:-1]])
                row = {**job, **native, 'actual_training_steps': snap['actual_steps'],
                    'natural_arrival': int(raw['classification'] == 'natural_arrival'),
                    'ego_collision': int(raw['classification'] == 'ego_collision'),
                    'native_event_agrees': raw['native_event_agrees'], 'raw_outcome': raw,
                    'policy_sha256': snap['weights_sha256'], 'no_policy_updates': True,
                    'low_speed_fraction': cov.result('evaluation_decision_states')['moments']['low_speed']['mean'],
                    'action_saturation_fraction': sum(abs(r['jerk']) >= 4.99 for r in trace)/len(trace),
                    'state_coverage': cov.result('evaluation_decision_states')}
                write_once(out/'episode.json', row); write_once(out/'actions.json', trace)
                write_once(out/'states.json', measurements); write_once(out/'raw_events.json', observer.export())
                return {'status': 'complete', 'job': job, 'outcome': row, 'no_policy_updates': True}
            finally: env.close()


def worker(args):
    path, q = load_request(args.request, args.confirm_request_hash)
    plan = q['train_jobs' if args.stage == 'train' else 'evaluation_jobs']
    job = next(j for j in plan if j['id'] == args.job)
    out = path.parent/args.stage/job['id']; out.mkdir(parents=True, exist_ok=False)
    started = {**shared.metadata(), 'binding': binding(q, args.stage, job)}
    write_once(out/'started.json', started); begin = time.perf_counter()
    try:
        result = train_job(q, job, out) if args.stage == 'train' else evaluation_job(q, job, out, path.parent)
        result.update(provenance=started, elapsed_s=time.perf_counter()-begin, finished_at=shared.now())
        write_once(out/'report.json', result); seal_episode(out, binding(q, args.stage, job), result)
    except BaseException:
        write_once(out/'failure.json', {'status': 'failed', 'failure_reason': traceback.format_exc()}); raise


def run(path, expected_hash, stage, resume=False):
    path, q = load_request(path, expected_hash); root = path.parent
    invocation = root/'invocations'/uuid.uuid4().hex[:10]; invocation.mkdir(parents=True, exist_ok=False)
    report = {**shared.metadata(), 'stage': stage, 'request_hash': digest(q), 'jobs': []}
    with run_lock(root):
        try:
            if stage == 'evaluate': training_metadata(q, root)
            plan = q['train_jobs' if stage == 'train' else 'evaluation_jobs']; pending = []
            for j in plan:
                if (root/stage/j['id']).exists():
                    if not resume: raise ValueError('Existing output: --resume reuses COMPLETE WHOLE jobs only')
                    verify_episode(root/stage/j['id'], binding(q, stage, j))
                    report['jobs'].append({'id': j['id'], 'action': 'reuse'})
                else: pending.append(j)
            def child(j):
                command = [sys.executable, '-B', str(Path(__file__).resolve()), 'worker', '--request', str(path),
                           '--confirm-request-hash', digest(q), '--stage', stage, '--job', j['id']]
                print(f'[p7k] {stage} {j["id"]}', flush=True)
                with (invocation/(j['id']+'.log')).open('x', encoding='utf-8') as log:
                    subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
                verify_episode(root/stage/j['id'], binding(q, stage, j))
                return {'id': j['id'], 'action': 'run'}
            width = q['protocol']['training' if stage == 'train' else 'evaluation']['workers']
            with ThreadPoolExecutor(max_workers=width) as pool:
                for i in range(0, len(pending), width):
                    futures = [pool.submit(child, j) for j in pending[i:i+width]]
                    report['jobs'].extend(f.result() for f in futures)
            if stage == 'train': training_metadata(q, root)
            report.update(status='complete', finished_at=shared.now())
        except BaseException:
            report.update(status='failed', failure_reason=traceback.format_exc()); raise
        finally: write_once(invocation/'report.json', report)
    print('stage_report='+str(invocation/'report.json'), flush=True)


def aggregate_run(path, expected_hash):
    path, q = load_request(path, expected_hash); root = path.parent
    with run_lock(root):
        trained = training_metadata(q, root); rows = []
        for j in q['evaluation_jobs']:
            out = root/'evaluate'/j['id']; m = verify_episode(out, binding(q, 'evaluate', j))['metadata']
            r = read(out/'episode.json')
            snap = next(s for s in trained[j['training_id']]['snapshots'] if s['requested_steps'] == j['checkpoint_frames'])
            if r != m['outcome'] or r['policy_sha256'] != snap['weights_sha256']:
                raise ValueError('Evaluation checkpoint lineage changed')
            rows.append(r)
        result = {**aggregate(q['protocol'], rows), 'request_hash': digest(q), 'provenance': shared.metadata(),
                  'training': trained, 'finished_at': shared.now()}
        # Preserve genuine original optimizer losses, not reconstructed Q estimates.
        curves = {}
        for j in q['train_jobs']:
            d = root/'train'/j['id']; losses = {'q': [], 'policy': []}
            with gzip.open(d/'losses.jsonl.gz', 'rt', encoding='utf-8') as handle:
                for line in handle:
                    row = json.loads(line); losses[row['component']].append(row)
            if any(len(v) != trained[j['id']]['coverage']['sample_calls'] for v in losses.values()):
                raise ValueError('Optimizer loss/sample counts differ')
            curves[j['id']] = {'training_episodes': read(d/'episodes.json'), 'optimizer_losses': losses}
        write_once(root/'training_curves.json', {'label': 'noisy_training_returns_and_original_losses_NOT_validation', 'jobs': curves})
        write_once(root/'development_curves.json', {'label': 'independent_frozen_policy_Gym20_development_NOT_test',
            'unique_scenarios': len(q['protocol']['evaluation']['simulator_seeds']), 'nodes': result['nodes']})
        write_once(root/'aggregate.json', result)
        write_once(root/'aggregate_receipt.json', {'request_hash': digest(q),
            'aggregate_sha256': sha(root/'aggregate.json'), 'training_curves_sha256': sha(root/'training_curves.json'),
            'development_curves_sha256': sha(root/'development_curves.json')})
    print('matched_aggregate='+str(root/'aggregate.json'), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare'); p.add_argument('--config', default='configs/development/p07k_matched_60k_v1.json'); p.add_argument('--run-id', required=True)
    for name in ('run', 'worker', 'aggregate'):
        p = sub.add_parser(name); p.add_argument('--request', required=True); p.add_argument('--confirm-request-hash', required=True)
        if name != 'aggregate': p.add_argument('--stage', choices=['train', 'evaluate'], required=True)
        if name == 'worker': p.add_argument('--job', required=True)
        if name == 'run': p.add_argument('--resume', action='store_true')
    p = sub.add_parser('audit'); p.add_argument('--run-id', required=True)
    a = parser.parse_args()
    if a.command == 'prepare': prepare(a.config, a.run_id)
    elif a.command == 'worker': worker(a)
    elif a.command == 'run': run(a.request, a.confirm_request_hash, a.stage, a.resume)
    elif a.command == 'aggregate': aggregate_run(a.request, a.confirm_request_hash)
    else:
        path, h = prepare('configs/development/p07k_workflow_smoke_v1.json', a.run_id)
        run(path, h, 'train'); run(path, h, 'evaluate'); aggregate_run(path, h)


if __name__ == '__main__': main()
