"""P7e: add two missing frozen-model/protocol cells; never train or rewrite P7/P7d."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
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
import compare_external_baselines as external
import diagnose_feature_ddpg as diagnostic
import explore_feature_ddpg as p7
import train_response_predictors as shared
import smoke_feature_ddpg as smoke
from prediction_rl.data.collection_store import (read_json as read, write_once, file_hash as sha,
    verify_episode, seal_episode, run_lock)
from prediction_rl.data.dataset_contract import digest
from prediction_rl.data.frozen_dataset import inside, verify_historical_inputs
from prediction_rl.data.merge_geometry import read_extended_traffic
from prediction_rl.envs.upstream import upstream_session
from prediction_rl.evaluation.closed_loop_diagnostics import PROTOCOL as P7_PARENT
from prediction_rl.evaluation.exploration import EpisodeMetrics
from prediction_rl.evaluation.model_protocol_crosscheck import (
    PROTOCOL, AUDIT, AUTHOR, B0, B3, GYM, AUTHOR_LOOP, jobs, roster, is_new,
    validate_protocol, native_row, aggregate)
from prediction_rl.training.all_ddpg import (preset_factory, checkpoint_contract,
    load_evaluation_weights, validate_state, validate_action)


def unchanged_historical_source(q):
    """Permit added audit files, NEVER changed historical executable files."""
    verify_historical_inputs(ROOT, {'python_lf:'+k: v for k, v in q['source_hashes'].items()}, q['git_commit'])
    current = shared.source_hashes()
    if (any(current.get(k) != v for k, v in q['source_hashes'].items())
            or q['environment'] != shared.env() or q['runtime_identity'] != p7.runtime_identity()
            or q['author_assets'] != external.assets()
            or q['upstream_config_hashes'] != external.configurations()):
        raise ValueError('Historical source/runtime/author asset changed')


def external_evidence(path, expected_protocol, expected_hash=None, expected_aggregate=None):
    path = inside(ROOT, path); q = read(path); out = path.parent
    if ((out/'writer.lock').exists() or q['version'] != 'p07d_request_v1'
            or q['protocol'] != expected_protocol
            or (expected_hash is not None and digest(q) != expected_hash)
            or (expected_aggregate is not None and sha(out/'aggregate.json') != expected_aggregate)
            or read(out/'preparation.json')['request_hash'] != digest(q)
            or q['jobs'] != external.jobs(expected_protocol)):
        raise ValueError('Historical external request/report changed or active')
    unchanged_historical_source(q)
    rows = []
    for j in q['jobs']:
        d = out/'evaluate'/j['id']; m = verify_episode(d, external.binding(q, j))['metadata']
        row = read(d/'episode.json')
        if m['no_policy_updates'] is not True or m['outcome'] != row:
            raise ValueError('Historical external episode/receipt mismatch')
        if expected_protocol == external.AUDIT and j['arm'] != 'conditional' and m['exact_raw_controller_parity'] is not True:
            raise ValueError('Historical author controller audit failed')
        rows.append(row)
    a = read(out/'aggregate.json'); recomputed = external.aggregate(expected_protocol, rows)
    if (a['request_hash'] != digest(q) or any(a[k] != v for k, v in recomputed.items())
            or read(out/'aggregate_receipt.json') != {'request_hash': digest(q), 'sha256': sha(out/'aggregate.json')}):
        raise ValueError('Historical external aggregate not reproducible')
    return path, q


def parents():
    pp, p = diagnostic.load_parent(P7_PARENT)
    ep, e = external_evidence(PROTOCOL['external_request'], external.PROTOCOL,
        PROTOCOL['external_request_hash'], PROTOCOL['external_aggregate_sha256'])
    ev = e['engineering_audit']
    ap, aq = external_evidence(Path(ev['path']).parent/'request.json', external.AUDIT, ev['request_hash'], ev['sha256'])
    if e['parent_request_hash'] != digest(p) or aq['parent_request_hash'] != digest(p):
        raise ValueError('P7/P7d do not bind the same trained models')
    trained = {j['id']: verify_episode(pp.parent/'train'/j['id'], p7.binding(p, 'train', j))['metadata'] for j in p['train_jobs']}
    for j in e['jobs']:
        row = read(ep.parent/'evaluate'/j['id']/'episode.json')
        if j['arm'] == 'conditional':
            expected = trained[j['training_id']]
            if (row['policy_sha256'] != expected['weights_sha256']
                    or row['predictor_sha256'] != expected['weights_contract']['predictor_sha256']):
                raise ValueError('External conditional checkpoint binding changed')
        elif j['arm'] != 'author_st' and row['policy_sha256'] != external.ASSETS['policy.pt']:
            raise ValueError('External author checkpoint binding changed')
    return pp, p, ep, e


def artifact_evidence(directory, request, kind):
    return {'kind': kind, 'request_hash': digest(request),
            'episode_path': str((directory/'episode.json').relative_to(ROOT)),
            'episode_sha256': sha(directory/'episode.json'),
            'receipt_path': str((directory/'complete.json').relative_to(ROOT)),
            'receipt_sha256': sha(directory/'complete.json')}


def historical_rows(config, pp, p, ep, e):
    rows = []
    for j in roster(config):
        if is_new(j): continue
        if j['driver_id'] == GYM:
            original = next(x for x in p['evaluation_jobs'] if x['training_id'] == j['training_id']
                            and x['simulator_seed'] == j['simulator_seed'])
            directory = pp.parent/'evaluate'/original['id']
            m = verify_episode(directory, p7.binding(p, 'evaluate', original))['metadata']
            row = read(directory/'episode.json')
            train = next(x for x in p['train_jobs'] if x['id'] == j['training_id'])
            tm = verify_episode(pp.parent/'train'/train['id'], p7.binding(p, 'train', train))['metadata']
            row = {**row, 'predictor_sha256': tm['weights_contract']['predictor_sha256']}
            request = p
        else:
            arm = 'author_ddpg' if j['model_id'] == AUTHOR else 'conditional'
            original = next(x for x in e['jobs'] if x['arm'] == arm and x['run_seed'] == j['run_seed']
                            and x['simulator_seed'] == j['simulator_seed'])
            directory = ep.parent/'evaluate'/original['id']
            m = verify_episode(directory, external.binding(e, original))['metadata']
            row = read(directory/'episode.json'); request = e
        if m['no_policy_updates'] is not True:
            raise ValueError('Historical policy updates detected')
        rows.append(native_row(j, row, artifact_evidence(directory, request, 'historical')))
    return rows


def parent_identity(pp, p, ep, e):
    return {'p7_request_hash': digest(p), 'p7d_request_hash': digest(e),
            'p7_aggregate_sha256': sha(pp.parent/'aggregate.json'),
            'p7d_aggregate_sha256': sha(ep.parent/'aggregate.json')}


def binding(q, j):
    return {'request_hash': digest(q), 'stage': 'model_protocol_crosscheck', 'job': j}


def audit_evidence(path):
    path = inside(ROOT, path); q = read(path.parent/'request.json'); a = read(path)
    if (q['protocol'] != AUDIT or q['source_hashes'] != shared.source_hashes()
            or q['jobs'] != jobs(AUDIT) or q['author_assets'] != external.assets()
            or q['environment'] != shared.env() or q['runtime_identity'] != p7.runtime_identity()
            or a['request_hash'] != digest(q) or a['engineering_complete'] is not True
            or read(path.parent/'preparation.json')['request_hash'] != digest(q)
            or read(path.parent/'aggregate_receipt.json') != {'request_hash': digest(q), 'sha256': sha(path)}):
        raise ValueError('Need current accepted crosscheck interface audit')
    pp, p, ep, e = parents()
    if q['parent_identity'] != parent_identity(pp, p, ep, e): raise ValueError('Audit parents changed')
    rows = historical_rows(AUDIT, pp, p, ep, e)
    if q['historical_rows'] != rows: raise ValueError('Audit historical rows changed')
    for j in jobs(AUDIT):
        d = path.parent/'evaluate'/j['id']; m = verify_episode(d, binding(q, j))['metadata']
        if m['no_policy_updates'] is not True or m['interface_audit_pass'] is not True or read(d/'episode.json') != m['outcome']:
            raise ValueError('Crosscheck interface audit incomplete')
        rows.append(native_row(j, m['outcome'], artifact_evidence(d, q, 'new')))
    if any(a[k] != v for k, v in aggregate(AUDIT, rows).items()): raise ValueError('Audit aggregate changed')
    return {'path': str(path.relative_to(ROOT)), 'sha256': sha(path), 'request_hash': digest(q)}


def prepare(config, run_id, audit_path=None):
    shared.clean(); validate_protocol(config); pp, p, ep, e = parents()
    evidence = None if config == AUDIT else audit_evidence(audit_path or 'artifacts/p7e/p7e_audit_v1/aggregate.json')
    q = {**shared.metadata(), 'version': 'p07e_request_v1', 'protocol': deepcopy(config),
         'jobs': jobs(config), 'historical_rows': historical_rows(config, pp, p, ep, e),
         'parent_identity': parent_identity(pp, p, ep, e), 'source_hashes': shared.source_hashes(),
         'environment': shared.env(), 'runtime_identity': p7.runtime_identity(),
         'author_assets': external.assets(), 'engineering_audit': evidence}
    out = shared.output_dir('p7e', run_id)
    write_once(out/'request.json', q)
    write_once(out/'preparation.json', {'request_hash': digest(q), 'new_episodes': len(q['jobs']),
        'verified_historical_episodes': len(q['historical_rows']), 'status': 'prepared_not_simulated'})
    print('crosscheck_request='+str(out/'request.json'), flush=True)
    print('confirm_request_hash='+digest(q), flush=True)
    return out/'request.json', digest(q)


def load(path, expected):
    shared.clean(); path = inside(ROOT, path); q = read(path); validate_protocol(q['protocol'])
    if (q['version'] != 'p07e_request_v1' or digest(q) != expected
            or read(path.parent/'preparation.json')['request_hash'] != expected
            or q['jobs'] != jobs(q['protocol']) or q['source_hashes'] != shared.source_hashes()
            or q['environment'] != shared.env() or q['runtime_identity'] != p7.runtime_identity()
            or q['author_assets'] != external.assets()):
        raise ValueError('Crosscheck request/source/runtime/roster changed')
    pp, p, ep, e = parents()
    if (q['parent_identity'] != parent_identity(pp, p, ep, e)
            or q['historical_rows'] != historical_rows(q['protocol'], pp, p, ep, e)):
        raise ValueError('Frozen historical evidence changed')
    if q['protocol'] == PROTOCOL:
        if audit_evidence(q['engineering_audit']['path']) != q['engineering_audit']:
            raise ValueError('Crosscheck engineering evidence changed')
    elif q['engineering_audit'] is not None: raise ValueError('Self-dependent audit')
    return path, q, pp, p


class FrozenQuery:
    """One TimeFeature query, exact read-only network parity, no action execution."""
    def __init__(self, body, network):
        self.body, self.network, self.records = body, network, []
        self.terminal_resets = 0
        if body.scale != .001 or body.timestep is not None:
            raise ValueError('Need fresh original TimeFeature')

    def __call__(self, state, reward):
        from all.environments import State
        validate_state(state, 20, 1)
        if not np.isfinite(reward): raise ValueError('Nonfinite native reward')
        t = torch.zeros(1, device=state.features.device) if self.body.timestep is None else self.body.timestep.clone()
        expected_t = len(self.records)
        if float(t.item()) != expected_t: raise ValueError('TimeFeature advanced outside actual decisions')
        augmented = State(torch.cat([state.features, .001*t[:, None]], 1), state.mask, state.info)
        with torch.no_grad(): reference = self.network(augmented)
        action = self.body.eval(state, reward)
        validate_action(action, 1, -5., 5.)
        if not torch.equal(action, reference) or not torch.equal(self.body.timestep, state.mask.float()*(t+1)):
            raise ValueError('Original policy forward/TimeFeature parity failed')
        if int(state.mask.item()) == 0:
            self.terminal_resets += 1
            if float(self.body.timestep.item()) != 0: raise ValueError('Terminal time reset failed')
        else:
            self.records.append({'observation': state.features[0].tolist(),
                                 'time_feature': float(.001*t.item()), 'requested_jerk': float(action.item())})
        return action

    def complete(self, steps):
        if len(self.records) != steps or self.terminal_resets != 1:
            raise ValueError('Actual query count/terminal contract changed')
        return True


def state_vector_parity(state):
    import dqn, prediction
    original = np.asarray(dqn.get_state_vector_from_base_state(prediction.HighwayState.from_sumo()), dtype=np.float32)
    if original.shape != (20,) or not np.array_equal(state.features[0].cpu().numpy(), original):
        raise ValueError('Gym observation differs from original vehicle ordering/normalization')


def author_policy(env):
    """Use trusted pinned ORIGINAL modules; do not transplant into project weights."""
    from all.experiments.watch import GreedyAgent
    from all.bodies.time import TimeFeature
    from all.policies.deterministic import DeterministicPolicyNetwork
    from all.approximation.q_continuous import QContinuousModule
    external.assets()  # Validate complete file set/hashes BEFORE unpickling known author modules.
    greedy = GreedyAgent.load(str(external.MODEL), env)
    critic = torch.load(external.MODEL/'q.pt', map_location='cpu', weights_only=False)
    if type(greedy.policy) is not DeterministicPolicyNetwork or type(critic) is not QContinuousModule:
        raise ValueError('Unknown author policy/critic class')
    for name, network, size in (('policy', greedy.policy, 21), ('q', critic, 22)):
        shapes = [(m.in_features, m.out_features) for m in network.modules() if isinstance(m, torch.nn.Linear)]
        if (shapes != [(size, 400), (400, 300), (300, 1)]
                or any(isinstance(m, (torch.nn.modules.dropout._DropoutNd, torch.nn.modules.batchnorm._BatchNorm)) for m in network.modules())
                or any(not torch.isfinite(v).all() for v in network.state_dict().values())):
            raise ValueError('Incompatible/nonfinite/stateful author '+name)
    if (not torch.equal(greedy.policy._tanh_scale, torch.tensor([5.]))
            or not torch.equal(greedy.policy._tanh_mean, torch.tensor([0.]))):
        raise ValueError('Author continuous action transform changed')
    return TimeFeature(greedy), greedy.policy, critic


def gym_episode(p, job, q):
    metrics = EpisodeMetrics(); captures = []; actions = []
    def record(event, obs, reward, done, info):
        metrics.record(event, obs, reward, done, info)
        captures.append({'event': event, 'observation': obs.tolist(),
                         'reward': None if reward is None else float(reward), 'done': bool(done), 'info': info})
    with upstream_session(p7.SOURCE, p7.UPSTREAM, job['simulator_seed']) as settings:
        env = p7.environment(p, {'arm': 'baseline'}, None, record)
        try:
            body, network, critic = author_policy(env)
            before = {'policy': smoke.tensor_hash(network.state_dict()), 'q': smoke.tensor_hash(critic.state_dict())}
            query = FrozenQuery(body, network); state = env.reset(); reward = 0.
            if env.env.feature_env.base.raw.wait_before_start != 20: raise ValueError('P7 warmup changed')
            for _ in range(500):
                state_vector_parity(state)
                action = query(state, reward)  # NO do_control/set_ego_jerk here.
                state, reward = env.step(action)  # Sole owner of original action execution/SUMO advancement.
                actions.append(float(action.item()))
                if env.done:
                    query(state, reward); break  # Original P7 terminal mask/time reset.
            else: raise ValueError('Original P7 500-step limit not honored')
            query.complete(len(actions))
            if before != {'policy': smoke.tensor_hash(network.state_dict()), 'q': smoke.tensor_hash(critic.state_dict())}:
                raise ValueError('Author evaluation changed weights')
            row = {**job, **metrics.result(), 'policy_sha256': external.ASSETS['policy.pt'], 'predictor_sha256': None}
            return {'outcome': row, 'actual_query_records': query.records, 'captures': captures,
                    'actions': actions, 'interface_audit_pass': True, 'no_policy_updates': True,
                    'resolved_config': {'upstream': settings.export_settings(), 'crosscheck': q['protocol'],
                        'native_protocol': p['protocol'], 'job': job, 'author_assets': external.assets()}}
        finally: env.close()


class ProjectBaselineController:
    """Original P7 B0 checkpoint, on original author loop, no Gym reset/step."""
    def __init__(self, pp, p, job, settings):
        from all.logging import DummyWriter
        train = next(j for j in p['train_jobs'] if j['id'] == job['training_id'])
        td = pp.parent/'train'/train['id']; m = verify_episode(td, p7.binding(p, 'train', train))['metadata']
        self.env = p7.environment(p, {'arm': 'baseline'}, None, None)
        self.env._lazy_init()  # Codec initialization ONLY. SUMO starts in raw constructor.
        self.checked = preset_factory('cpu', settings.LEARNING_RATE)(self.env, DummyWriter())
        contract = checkpoint_contract(self.env, p['preset'], None, p['upstream_config_sha256'])
        if contract != m['weights_contract']: raise ValueError('B0 frozen input/weight contract changed')
        load_evaluation_weights(self.checked, td/'final.pt', contract, m['weights_sha256'])
        self.weights_sha = m['weights_sha256']; self.before = self.fingerprint()
        self.query = FrozenQuery(self.checked.body, self.checked.agent.policy.model)

    def fingerprint(self):
        return {n: smoke.tensor_hash(a.model.state_dict()) for n, a in (('policy', self.checked.agent.policy), ('q', self.checked.agent.q))}

    def __call__(self, state):
        import dqn, control
        obs = np.asarray(dqn.get_state_vector_from_base_state(state), dtype=np.float32)
        if obs.shape != (20,) or not np.isfinite(obs).all(): raise ValueError('B0 observation changed')
        encoded = self.env._make_state(obs, False)
        action = self.query(encoded, 0.)
        return control.set_ego_jerk(float(action.item()))

    def end(self, state):
        import dqn
        obs = np.asarray(dqn.get_state_vector_from_base_state(state), dtype=np.float32)
        self.query(self.env._make_state(obs, True), 0.)

    def verify_frozen(self, steps):
        if (self.before != self.fingerprint() or len(self.checked.agent.replay_buffer)
                or any(a._optimizer.state for a in (self.checked.agent.policy, self.checked.agent.q))):
            raise ValueError('B0 evaluation changed model/optimizer/replay')
        return self.query.complete(steps)


def author_loop_episode(pp, p, job, q):
    with upstream_session(p7.SOURCE, p7.UPSTREAM, job['simulator_seed']) as settings:
        import control, prediction, rl, dqn
        settings.CUDA = False
        current = ProjectBaselineController(pp, p, job, settings)
        frames = []; latencies = []
        def recorded(state):
            frames.append(read_extended_traffic()); start = time.perf_counter()
            speed = current(state); latencies.append(time.perf_counter()-start); return speed
        e = control.run_episode(recorded, prediction.HighwayState.from_sumo,
            max_episode_length=100, wait_before_start=50, end_episode_callback=current.end)
        current.verify_frozen(len(e['control_history']))
        score = rl.get_rl_custom_episode_stats(e, dqn.get_reward_function())['total_reward']
        row = {**job, **external.outcome(e, digest(frames[0]), None, latencies, score),
               'policy_sha256': current.weights_sha, 'predictor_sha256': None}
        histories = {k: e[k] for k in ('crashed', 'merged', 'control_history', 'position_history', 'speed_history',
            'acceleration_history', 'jerk_history', 'closest_vehicle_history', 'disruption_history', 'simulation_time_taken')}
        return {'outcome': row, 'histories': histories, 'actual_observed_frames': frames,
                'actual_query_records': current.query.records, 'interface_audit_pass': True, 'no_policy_updates': True,
                'resolved_config': {'upstream': settings.export_settings(), 'crosscheck': q['protocol'],
                    'native_protocol': external.PROTOCOL, 'job': job, 'weights_sha256': current.weights_sha}}


def worker(args):
    path, q, pp, p = load(args.request, args.confirm_request_hash)
    job = next(j for j in q['jobs'] if j['id'] == args.job)
    out = path.parent/'evaluate'/job['id']; out.mkdir(parents=True, exist_ok=False)
    started = shared.metadata(); write_once(out/'started.json', started)
    torch.set_num_threads(q['protocol']['torch_threads'])
    try:
        result = gym_episode(p, job, q) if job['driver_id'] == GYM else author_loop_episode(pp, p, job, q)
        write_once(out/'episode.json', result['outcome']); write_once(out/'trace.json', result)
        write_once(out/'resolved_config.json', result['resolved_config'])
        seal_episode(out, binding(q, job), {**started, 'status': 'complete', 'outcome': result['outcome'],
            'interface_audit_pass': result['interface_audit_pass'], 'no_policy_updates': True, 'finished_at': shared.now()})
    except BaseException:
        write_once(out/'failure.json', {**started, 'failure_reason': traceback.format_exc()}); raise


def aggregate_request(path, expected):
    path, q, _, _ = load(path, expected); out = path.parent
    with run_lock(out):
        rows = deepcopy(q['historical_rows'])
        for j in q['jobs']:
            d = out/'evaluate'/j['id']; m = verify_episode(d, binding(q, j))['metadata']
            if m['no_policy_updates'] is not True or m['interface_audit_pass'] is not True or read(d/'episode.json') != m['outcome']:
                raise ValueError('Incomplete crosscheck interface/receipt')
            rows.append(native_row(j, m['outcome'], artifact_evidence(d, q, 'new')))
        a = {**shared.metadata(), **aggregate(q['protocol'], rows), 'request_hash': digest(q), 'finished_at': shared.now()}
        if (out/'aggregate.json').exists():
            old = read(out/'aggregate.json')
            if (any(old[k] != a[k] for k in a if k not in ('git_commit', 'command', 'started_at', 'finished_at'))
                    or read(out/'aggregate_receipt.json') != {'request_hash': digest(q), 'sha256': sha(out/'aggregate.json')}):
                raise ValueError('Immutable crosscheck aggregate changed')
        else:
            write_once(out/'aggregate.json', a)
            write_once(out/'aggregate_receipt.json', {'request_hash': digest(q), 'sha256': sha(out/'aggregate.json')})
    print('crosscheck_aggregate='+str(out/'aggregate.json'), flush=True); return out/'aggregate.json'


def execute(path, expected, resume=False):
    path, q, _, _ = load(path, expected); out = path.parent
    invocation = out/'invocations'/uuid.uuid4().hex[:12]; invocation.mkdir(parents=True, exist_ok=False)
    report = {**shared.metadata(), 'request_hash': digest(q), 'jobs': [], 'status': 'running'}
    with run_lock(out):
        try:
            pending = []
            for j in q['jobs']:
                d = out/'evaluate'/j['id']
                if d.exists():
                    if not resume: raise ValueError('Use --resume for verified COMPLETE jobs only')
                    verify_episode(d, binding(q, j)); report['jobs'].append({'id': j['id'], 'action': 'reuse'})
                else: pending.append(j)
            def child(j):
                command = [sys.executable, str(Path(__file__).resolve()), 'worker', '--request', str(path),
                           '--confirm-request-hash', expected, '--job', j['id']]
                print('[p7e] evaluating '+j['id'], flush=True)
                with (invocation/(j['id']+'.log')).open('w', encoding='utf-8') as log:
                    subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
                verify_episode(out/'evaluate'/j['id'], binding(q, j))
                print('[p7e] complete '+j['id'], flush=True); return {'id': j['id'], 'action': 'run'}
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
    c = sub.add_parser('audit'); c.add_argument('--run-id', default='p7e_audit_v1')
    c = sub.add_parser('prepare'); c.add_argument('--config', default='configs/development/p07e_model_protocol_crosscheck_v1.json')
    c.add_argument('--run-id', default='p7e_cross_v1'); c.add_argument('--engineering-audit')
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
