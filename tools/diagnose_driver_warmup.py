"""P7f manual-start 2x2 driver/warmup diagnosis; frozen models, no training."""
import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from functools import lru_cache
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
import crosscheck_model_protocols as cross
import diagnose_feature_ddpg as recording
import explore_feature_ddpg as p7
import train_response_predictors as shared
import smoke_feature_ddpg as smoke
from prediction_rl.data.collection_store import (read_json as read, write_once, file_hash as sha,
    verify_episode, seal_episode, run_lock)
from prediction_rl.data.dataset_contract import digest
from prediction_rl.data.frozen_dataset import inside, verify_historical_inputs
from prediction_rl.data.merge_geometry import read_extended_traffic
from prediction_rl.envs.upstream import upstream_session, traffic_snapshot
from prediction_rl.evaluation.driver_warmup import (PROTOCOL, AUDIT, AUTHOR, B0, B3, CONDITIONS,
    jobs, roster, is_new, validate_protocol, canonical_frame, row_from_native, aggregate, validate_decisions)
from prediction_rl.evaluation.exploration import EpisodeMetrics
from prediction_rl.training.all_ddpg import (preset_factory, checkpoint_contract,
    load_evaluation_weights, validate_state, validate_action)


def historical_source(q):
    verify_historical_inputs(ROOT, {'python_lf:'+k: v for k, v in q['source_hashes'].items()}, q['git_commit'])
    current = shared.source_hashes()
    if any(current.get(k) != v for k, v in q['source_hashes'].items()) or q['environment'] != shared.env():
        raise ValueError('Historical executable source/environment changed')


def finished_crosscheck(path, pp, p, ep, e, expected_protocol, expected_hash, expected_sha):
    path = inside(ROOT, path); q = read(path); out = path.parent
    if ((out/'writer.lock').exists() or q['version'] != 'p07e_request_v1'
            or q['protocol'] != expected_protocol or digest(q) != expected_hash
            or sha(out/'aggregate.json') != expected_sha
            or q['jobs'] != cross.jobs(expected_protocol)
            or read(out/'preparation.json')['request_hash'] != expected_hash
            or q['parent_identity'] != cross.parent_identity(pp, p, ep, e)
            or q['runtime_identity'] != p7.runtime_identity() or q['author_assets'] != cross.external.assets()):
        raise ValueError('Completed P7e identity/report changed or active')
    historical_source(q)
    rows = cross.historical_rows(expected_protocol, pp, p, ep, e)
    if q['historical_rows'] != rows: raise ValueError('P7e historical rows changed')
    for j in q['jobs']:
        directory = out/'evaluate'/j['id']; m = verify_episode(directory, cross.binding(q, j))['metadata']
        if m['no_policy_updates'] is not True or m['interface_audit_pass'] is not True or read(directory/'episode.json') != m['outcome']:
            raise ValueError('Completed P7e interface/episode invalid')
        rows.append(cross.native_row(j, m['outcome'], cross.artifact_evidence(directory, q, 'new')))
    a = read(out/'aggregate.json')
    if (a['request_hash'] != expected_hash or any(a[k] != v for k, v in cross.aggregate(expected_protocol, rows).items())
            or read(out/'aggregate_receipt.json') != {'request_hash': expected_hash, 'sha256': expected_sha}):
        raise ValueError('Completed P7e aggregate not reproducible')
    return path, q


@lru_cache(maxsize=1)
def parents():
    pp, p, ep, e = cross.parents()
    cp = inside(ROOT, PROTOCOL['crosscheck_request']); cq = read(cp)
    ev = cq['engineering_audit']
    _, aq = finished_crosscheck(Path(ev['path']).parent/'request.json', pp, p, ep, e,
        cross.AUDIT, ev['request_hash'], ev['sha256'])
    if aq['engineering_audit'] is not None: raise ValueError('Self-dependent P7e audit')
    cp, cq = finished_crosscheck(cp, pp, p, ep, e, cross.PROTOCOL,
        PROTOCOL['crosscheck_request_hash'], PROTOCOL['crosscheck_aggregate_sha256'])
    bp = inside(ROOT, PROTOCOL['recording_request']); bq = read(bp); out = bp.parent
    if ((out/'writer.lock').exists() or bq['version'] != 'p07b_request_v1' or bq['protocol'] != recording.PROTOCOL
            or digest(bq) != PROTOCOL['recording_request_hash']
            or sha(out/'summary.json') != PROTOCOL['recording_summary_sha256']
            or read(out/'preparation.json')['request_hash'] != digest(bq)
            or read(out/'summary_receipt.json') != {'request_hash': digest(bq), 'summary_sha256': sha(out/'summary.json')}
            or bq['jobs'] != recording.select_jobs(bq['protocol'], p) or bq['parent_request_hash'] != digest(p)
            or sha(out/'reference.json') != bq['reference_sha256']
            or sha(out/'reference_rows.json') != bq['reference_rows_sha256']):
        raise ValueError('Completed P7b recording changed or active')
    historical_source(bq); reports = []
    for j in bq['jobs']:
        d = out/'recording'/j['id']; m = verify_episode(d, recording.binding(bq, j))['metadata']
        original = pp.parent/'evaluate'/j['id']
        if m['exact_parent_parity'] is not True or m['no_policy_updates'] is not True or read(d/'report.json') != m:
            raise ValueError('Historical P7b interface/parity invalid')
        recording.assert_parity(read(d/'episode.json'), read(d/'actions.json'),
            read(original/'episode.json'), read(original/'actions.json'))
        reports.append(m)
    s = read(out/'summary.json')
    if s['request_hash'] != digest(bq) or s['all_replays_exact'] is not True or s['jobs'] != reports:
        raise ValueError('Historical P7b summary/receipts mismatch')
    return pp, p, ep, e, cp, cq, bp, bq


def evidence(kind, q, directory, filenames):
    return {'kind': kind, 'request_hash': digest(q),
            'files': {str((directory/name).relative_to(ROOT)): sha(directory/name) for name in filenames}}


def decision(frame, jerk, speed, obs, index, *, time_feature=None, time_source='reconstructed_original_single_query_sequence',
             speed_source='original_controller_return'):
    return {'frame': canonical_frame(frame), 'requested_jerk': float(jerk), 'commanded_speed': float(speed),
            'observation': None if obs is None else list(obs),
            'observation_source': 'not_recorded_in_historical_native_trace' if obs is None else 'recorded_policy_input',
            'time_feature': float(np.float32(.001)*np.float32(index)) if time_feature is None else float(time_feature),
            'time_feature_source': time_source, 'commanded_speed_source': speed_source}


def gym_trace(captures, queries, actual_time=False):
    if len(captures) != len(queries)+1: raise ValueError('Gym capture/query count changed')
    return [decision(captures[i]['info']['traffic'] if i == 0 else captures[i]['info']['execution_audit']['after'],
        r['requested_jerk'], captures[i+1]['info']['execution_audit']['commanded_speed_reconstructed'],
        r['observation'], i, time_feature=r['time_feature'] if actual_time else None,
        time_source='recorded_policy_query' if actual_time else 'reconstructed_original_single_query_sequence',
        speed_source='upstream_helper_reconstruction')
        for i, r in enumerate(queries)]


def historical(job):
    pp, p, ep, e, cp, cq, bp, bq = parents()
    old_driver = cross.GYM if job['condition_id'] == 'gym20' else cross.AUTHOR_LOOP
    original_job = next(j for j in cross.roster(cross.PROTOCOL) if j['model_id'] == job['model_id']
        and j['run_seed'] == job['run_seed'] and j['simulator_seed'] == job['simulator_seed'] and j['driver_id'] == old_driver)
    if cross.is_new(original_job):
        d = cp.parent/'evaluate'/original_job['id']; trace = read(d/'trace.json'); outcome = read(d/'episode.json')
        if job['condition_id'] == 'gym20': records = gym_trace(trace['captures'], trace['actual_query_records'])
        else:
            records = [decision(f, r['requested_jerk'], trace['histories']['control_history'][i], r['observation'], i,
                        time_source='recorded_policy_query')
                for i, (f, r) in enumerate(zip(trace['actual_observed_frames'], trace['actual_query_records']))]
        ev = evidence('historical', cq, d, ('episode.json', 'trace.json', 'complete.json'))
    elif job['condition_id'] == 'gym20':
        old = next(j for j in bq['jobs'] if j['training_id'] == job['training_id'] and j['simulator_seed'] == job['simulator_seed'])
        d = bp.parent/'recording'/old['id']; obs = read(d/'observations.json'); actions = read(d/'actions.json'); outcome = read(d/'episode.json')
        records = [decision(obs[i]['capture']['frame'], a['jerk'], obs[i+1]['execution_audit']['commanded_speed_reconstructed'],
            obs[i]['observation'], i, time_feature=obs[i]['ddpg_input_with_time'][-1], time_source='recorded_policy_query',
            speed_source='upstream_helper_reconstruction') for i, a in enumerate(actions)]
        # P7b preserves parent outcomes exactly; predictor hash is in the training contract.
        train = next(j for j in p['train_jobs'] if j['id'] == job['training_id'])
        tm = verify_episode(pp.parent/'train'/train['id'], p7.binding(p, 'train', train))['metadata']
        outcome = {**outcome, 'predictor_sha256': tm['weights_contract']['predictor_sha256']}
        ev = evidence('historical', bq, d, ('episode.json', 'observations.json', 'actions.json', 'complete.json'))
    else:
        arm = 'author_ddpg' if job['model_id'] == AUTHOR else 'conditional'
        old = next(j for j in e['jobs'] if j['arm'] == arm and j['run_seed'] == job['run_seed'] and j['simulator_seed'] == job['simulator_seed'])
        d = ep.parent/'evaluate'/old['id']; trace = read(d/'trace.json'); outcome = read(d/'episode.json')
        jerks = trace['requested_conditional_jerks'] if job['model_id'] == B3 else [a[0] for a in trace['decision_policy_queries']]
        if job['model_id'] == AUTHOR and any(len(a) != 1 for a in trace['decision_policy_queries']):
            raise ValueError('Only original pure-DDPG single-query histories apply')
        records = [decision(f, jerks[i], trace['histories']['control_history'][i], None, i)
                   for i, f in enumerate(trace['actual_observed_frames'])]
        ev = evidence('historical', e, d, ('episode.json', 'trace.json', 'complete.json'))
    row = row_from_native(job, outcome, records, ev)
    return row, records


def historical_rows(config):
    return [historical(j)[0] for j in roster(config) if not is_new(j)]


def binding(q, j):
    return {'request_hash': digest(q), 'stage': 'driver_warmup_diagnosis', 'job': j}


class FrozenQuery:
    """Exact network/time parity for both 20- and 169-channel frozen policies."""
    def __init__(self, body, network, dimension):
        if dimension not in (20, 169) or body.scale != .001 or body.timestep is not None:
            raise ValueError('Need fresh frozen policy/time contract')
        self.body, self.network, self.dimension = body, network, dimension
        self.records = []; self.terminal_resets = 0

    def __call__(self, state, reward=0.):
        from all.environments import State
        validate_state(state, self.dimension, 1)
        if not np.isfinite(reward): raise ValueError('Nonfinite native reward')
        t = torch.zeros(1) if self.body.timestep is None else self.body.timestep.clone()
        if float(t.item()) != len(self.records): raise ValueError('TimeFeature advanced outside actual decisions')
        augmented = State(torch.cat([state.features, .001*t[:, None]], 1), state.mask, state.info)
        with torch.no_grad(): reference = self.network(augmented)
        action = self.body.eval(state, reward); validate_action(action, 1, -5., 5.)
        if not torch.equal(action, reference) or not torch.equal(self.body.timestep, state.mask.float()*(t+1)):
            raise ValueError('Frozen network/TimeFeature parity failed')
        if int(state.mask.item()) == 0: self.terminal_resets += 1
        else: self.records.append({'observation': state.features[0].tolist(),
                                  'time_feature': float(augmented.features[0, -1].item()), 'requested_jerk': float(action.item())})
        return action

    def complete(self, steps):
        if len(self.records) != steps or self.terminal_resets != 1:
            raise ValueError('Query count/terminal reset changed')


class FrozenModel:
    def __init__(self, pp, p, job, settings, recorder=None):
        self.arm = 'conditional' if job['model_id'] == B3 else 'baseline'
        self.provider, self.predictor_sha = p7.provider_for(p, self.arm)
        self.env = p7.environment(p, {'arm': self.arm}, self.provider, recorder)
        self.env._lazy_init()  # Codec only: SUMO started by the raw constructor, reset not called.
        self.checked = None
        if job['model_id'] == AUTHOR:
            body, self.policy, self.critic = cross.author_policy(self.env)
            self.weights_sha = cross.external.ASSETS['policy.pt']
        else:
            from all.logging import DummyWriter
            train = next(j for j in p['train_jobs'] if j['id'] == job['training_id']); td = pp.parent/'train'/train['id']
            m = verify_episode(td, p7.binding(p, 'train', train))['metadata']
            self.checked = preset_factory('cpu', settings.LEARNING_RATE)(self.env, DummyWriter())
            contract = checkpoint_contract(self.env, p['preset'], self.predictor_sha, p['upstream_config_sha256'])
            if contract != m['weights_contract']: raise ValueError('Frozen weight/input contract changed')
            load_evaluation_weights(self.checked, td/'final.pt', contract, m['weights_sha256'])
            body, self.policy, self.critic = self.checked.body, self.checked.agent.policy.model, self.checked.agent.q.model
            self.weights_sha = m['weights_sha256']
        self.before = self.fingerprint(); self.query = FrozenQuery(body, self.policy, 169 if self.provider else 20)

    def fingerprint(self):
        return {'policy': smoke.tensor_hash(self.policy.state_dict()), 'q': smoke.tensor_hash(self.critic.state_dict()),
                'predictor': None if self.provider is None else smoke.tensor_hash(self.provider.model.state_dict())}

    def verify(self, steps):
        self.query.complete(steps)
        if self.before != self.fingerprint(): raise ValueError('Evaluation changed weights')
        if self.checked is not None and (len(self.checked.agent.replay_buffer)
                or any(a._optimizer.state for a in (self.checked.agent.policy, self.checked.agent.q))):
            raise ValueError('Evaluation updated replay/optimizer')
        if self.provider is not None and any(p.grad is not None or p.requires_grad for p in self.provider.model.parameters()):
            raise ValueError('Predictor not frozen')


def configure_warmup(env, warmup):
    raw = env.env.feature_env.base.raw
    if raw.wait_before_start != 20 or warmup not in (20, 50) or raw.current_episode_ticks != 0:
        raise ValueError('Warmup override must precede first reset, with original defaults')
    raw.wait_before_start = warmup  # Instance-local sole experimental variable; no upstream source/settings edit.


def gym_episode(pp, p, job):
    metrics = EpisodeMetrics(); captures = []
    def record(event, obs, reward, done, info):
        metrics.record(event, obs, reward, done, info)
        captures.append({'event': event, 'observation': obs.tolist(), 'info': info})
    with upstream_session(p7.SOURCE, p7.UPSTREAM, job['simulator_seed']) as settings:
        settings.CUDA = False; model = FrozenModel(pp, p, job, settings, record)
        try:
            configure_warmup(model.env, CONDITIONS[job['condition_id']]['warmup_s'])
            state = model.env.reset(); reward = 0.
            for _ in range(500):
                import dqn, prediction
                base = np.asarray(dqn.get_state_vector_from_base_state(prediction.HighwayState.from_sumo()), dtype=np.float32)
                if not np.array_equal(state.features[0, :20].numpy(), base): raise ValueError('Base observation codec changed')
                action = model.query(state, reward); state, reward = model.env.step(action)
                if model.env.done:
                    model.query(state, reward); break
            else: raise ValueError('Original Gym 500-step limit not honored')
            model.verify(len(model.query.records)); records = gym_trace(captures, model.query.records, actual_time=True)
            outcome = {**job, **metrics.result(), 'policy_sha256': model.weights_sha, 'predictor_sha256': model.predictor_sha}
            return outcome, {'decisions': records, 'terminal_frame': canonical_frame(traffic_snapshot()),
                'native_terminal_reason': outcome['termination']}, settings.export_settings()
        finally: model.env.close()


def author_episode(pp, p, job):
    with upstream_session(p7.SOURCE, p7.UPSTREAM, job['simulator_seed']) as settings:
        import control, prediction, dqn, rl
        settings.CUDA = False; model = FrozenModel(pp, p, job, settings)
        history = deque(maxlen=11); records = []; latencies = []; last_obs = None
        def controller(state):
            nonlocal last_obs
            started = time.perf_counter()
            frame = read_extended_traffic(); history.append(frame)
            base = np.asarray(dqn.get_state_vector_from_base_state(state), dtype=np.float32)
            obs = base if model.provider is None else np.concatenate([base, model.provider(list(history))])
            last_obs = obs
            action = model.query(model.env._make_state(obs, False)); jerk = float(action.item())
            speed = control.set_ego_jerk(jerk)
            latencies.append(time.perf_counter()-started)
            records.append(decision(frame, jerk, speed, obs.tolist(), len(records),
                time_feature=model.query.records[-1]['time_feature'], time_source='recorded_policy_query'))
            return speed
        def end(state):
            if model.provider is None: obs = np.asarray(dqn.get_state_vector_from_base_state(state), dtype=np.float32)
            else: obs = np.concatenate([last_obs[:20], np.zeros(149, np.float32)])
            model.query(model.env._make_state(obs, True))  # No prediction or command on terminal query.
        episode = control.run_episode(controller, prediction.HighwayState.from_sumo,
            max_episode_length=100, wait_before_start=CONDITIONS[job['condition_id']]['warmup_s'], end_episode_callback=end)
        model.verify(len(episode['control_history']))
        score = rl.get_rl_custom_episode_stats(episode, dqn.get_reward_function())['total_reward']
        native = cross.external.outcome(episode, digest(records[0]['frame']), None, latencies, score)
        outcome = {**job, **native, 'policy_sha256': model.weights_sha, 'predictor_sha256': model.predictor_sha}
        histories = {k: episode[k] for k in ('crashed', 'merged', 'control_history', 'position_history',
            'speed_history', 'acceleration_history', 'jerk_history', 'simulation_time_taken')}
        return outcome, {'decisions': records, 'terminal_frame': canonical_frame(traffic_snapshot()),
            'native_histories': histories, 'native_terminal_reason': 'collision' if episode['crashed'] else 'arrival' if episode['merged'] else 'time_limit'}, settings.export_settings()


def audit_evidence(path):
    path = inside(ROOT, path); q = read(path.parent/'request.json')
    load(path.parent/'request.json', digest(q))
    a = collect(q, path.parent)
    old = read(path)
    if (q['protocol'] != AUDIT or any(old[k] != v for k, v in a.items())
            or read(path.parent/'aggregate_receipt.json') != {'request_hash': digest(q), 'sha256': sha(path)}):
        raise ValueError('Need current complete driver/warmup interface audit')
    return {'path': str(path.relative_to(ROOT)), 'sha256': sha(path), 'request_hash': digest(q)}


def prepare(config, run_id, audit_path=None):
    shared.clean(); validate_protocol(config); parents()
    accepted = None if config == AUDIT else audit_evidence(audit_path or 'artifacts/p7f/p7f_audit_v1/aggregate.json')
    q = {**shared.metadata(), 'version': 'p07f_request_v1', 'protocol': deepcopy(config), 'jobs': jobs(config),
         'historical_rows': historical_rows(config), 'source_hashes': shared.source_hashes(),
         'environment': shared.env(), 'runtime_identity': p7.runtime_identity(),
         'author_assets': cross.external.assets(), 'engineering_audit': accepted}
    out = shared.output_dir('p7f', run_id); write_once(out/'request.json', q)
    write_once(out/'preparation.json', {'request_hash': digest(q), 'new_episodes': len(q['jobs']),
        'verified_historical_episodes': len(q['historical_rows']), 'status': 'prepared_not_simulated'})
    print('driver_warmup_request='+str(out/'request.json'), flush=True)
    print('confirm_request_hash='+digest(q), flush=True); return out/'request.json', digest(q)


def load(path, expected):
    shared.clean(); path = inside(ROOT, path); q = read(path); validate_protocol(q['protocol'])
    if (q['version'] != 'p07f_request_v1' or digest(q) != expected
            or read(path.parent/'preparation.json')['request_hash'] != expected
            or q['jobs'] != jobs(q['protocol']) or q['source_hashes'] != shared.source_hashes()
            or q['environment'] != shared.env() or q['runtime_identity'] != p7.runtime_identity()
            or q['author_assets'] != cross.external.assets()):
        raise ValueError('Driver/warmup request/source/runtime changed')
    parents()
    if q['historical_rows'] != historical_rows(q['protocol']): raise ValueError('Historical trace evidence changed')
    if q['protocol'] == PROTOCOL:
        if audit_evidence(q['engineering_audit']['path']) != q['engineering_audit']: raise ValueError('Audit changed')
    elif q['engineering_audit'] is not None: raise ValueError('Self-dependent audit')
    return path, q


def worker(args):
    path, q = load(args.request, args.confirm_request_hash); pp, p, *_ = parents()
    job = next(j for j in q['jobs'] if j['id'] == args.job)
    out = path.parent/'evaluate'/job['id']; out.mkdir(parents=True, exist_ok=False)
    started = shared.metadata(); write_once(out/'started.json', started)
    torch.set_num_threads(q['protocol']['torch_threads'])
    try:
        outcome, trace, settings = (gym_episode if CONDITIONS[job['condition_id']]['driver'] == 'gym' else author_episode)(pp, p, job)
        write_once(out/'episode.json', outcome); write_once(out/'trace.json', trace)
        write_once(out/'resolved_config.json', {'upstream': settings, 'protocol': q['protocol'], 'job': job,
            'actual_warmup_s': CONDITIONS[job['condition_id']]['warmup_s']})
        row_from_native(job, outcome, trace['decisions'], evidence('new', q, out, ('episode.json', 'trace.json')))
        seal_episode(out, binding(q, job), {**started, 'status': 'complete', 'outcome': outcome,
            'interface_audit_pass': True, 'no_policy_updates': True, 'finished_at': shared.now()})
    except BaseException:
        write_once(out/'failure.json', {**started, 'failure_reason': traceback.format_exc()}); raise


def collect(q, out):
    rows = []; traces = {}
    for j in roster(q['protocol']):
        if is_new(j):
            d = out/'evaluate'/j['id']; m = verify_episode(d, binding(q, j))['metadata']; outcome = read(d/'episode.json')
            if m['no_policy_updates'] is not True or m['interface_audit_pass'] is not True or m['outcome'] != outcome:
                raise ValueError('New episode interface/receipt mismatch')
            trace = read(d/'trace.json')['decisions']
            row = row_from_native(j, outcome, trace, evidence('new', q, d, ('episode.json', 'trace.json', 'complete.json')))
        else: row, trace = historical(j)
        rows.append(row); traces[j['id']] = trace
    return {**aggregate(q['protocol'], rows, traces), 'request_hash': digest(q)}


def aggregate_request(path, expected):
    path, q = load(path, expected); out = path.parent
    with run_lock(out):
        result = collect(q, out)
        if (out/'aggregate.json').exists():
            old = read(out/'aggregate.json')
            if (any(old[k] != v for k, v in result.items())
                    or read(out/'aggregate_receipt.json') != {'request_hash': expected, 'sha256': sha(out/'aggregate.json')}):
                raise ValueError('Immutable driver/warmup aggregate changed')
        else:
            write_once(out/'aggregate.json', {**shared.metadata(), **result, 'finished_at': shared.now()})
            write_once(out/'aggregate_receipt.json', {'request_hash': expected, 'sha256': sha(out/'aggregate.json')})
    print('driver_warmup_aggregate='+str(out/'aggregate.json'), flush=True); return out/'aggregate.json'


def execute(path, expected, resume=False):
    path, q = load(path, expected); out = path.parent
    invocation = out/'invocations'/uuid.uuid4().hex[:12]; invocation.mkdir(parents=True, exist_ok=False)
    report = {**shared.metadata(), 'request_hash': expected, 'jobs': [], 'status': 'running'}
    with run_lock(out):
        try:
            pending = []
            for j in q['jobs']:
                d = out/'evaluate'/j['id']
                if d.exists():
                    if not resume: raise ValueError('Use --resume for verified COMPLETE jobs only')
                    m = verify_episode(d, binding(q, j))['metadata']
                    if m['no_policy_updates'] is not True or m['interface_audit_pass'] is not True:
                        raise ValueError('Incomplete interface audit')
                    report['jobs'].append({'id': j['id'], 'action': 'reuse'})
                else: pending.append(j)
            def child(j):
                command = [sys.executable, str(Path(__file__).resolve()), 'worker', '--request', str(path),
                           '--confirm-request-hash', expected, '--job', j['id']]
                print('[p7f] evaluating '+j['id'], flush=True)
                with (invocation/(j['id']+'.log')).open('w', encoding='utf-8') as log:
                    subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
                verify_episode(out/'evaluate'/j['id'], binding(q, j))
                print('[p7f] complete '+j['id'], flush=True); return {'id': j['id'], 'action': 'run'}
            width = q['protocol']['workers']
            with ThreadPoolExecutor(max_workers=width) as pool:
                for i in range(0, len(pending), width):
                    report['jobs'].extend(f.result() for f in [pool.submit(child, j) for j in pending[i:i+width]])
            report.update(status='complete', finished_at=shared.now())
        except BaseException:
            report.update(status='failed', failure_reason=traceback.format_exc()); raise
        finally: write_once(invocation/'report.json', report)
    return aggregate_request(path, expected)


def main():
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest='command', required=True)
    c = sub.add_parser('audit'); c.add_argument('--run-id', default='p7f_audit_v1')
    c = sub.add_parser('prepare'); c.add_argument('--config', default='configs/development/p07f_driver_warmup_v1.json')
    c.add_argument('--run-id', default='p7f_diag_v1'); c.add_argument('--engineering-audit')
    for name in ('run', 'aggregate', 'worker'):
        c = sub.add_parser(name); c.add_argument('--request', required=True); c.add_argument('--confirm-request-hash', required=True)
        if name == 'run': c.add_argument('--resume', action='store_true')
        if name == 'worker': c.add_argument('--job', required=True)
    args = parser.parse_args()
    if args.command == 'audit':
        path, expected = prepare(AUDIT, args.run_id); execute(path, expected)
    elif args.command == 'prepare': prepare(read(inside(ROOT, args.config)), args.run_id, args.engineering_audit)
    elif args.command == 'run': execute(args.request, args.confirm_request_hash, args.resume)
    elif args.command == 'aggregate': aggregate_request(args.request, args.confirm_request_hash)
    else: worker(args)


if __name__ == '__main__': main()
