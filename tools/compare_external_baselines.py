"""Manual-start P7d external author RL/ST/RL+MPC comparison; no training."""
import argparse
from collections import deque
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
import diagnose_feature_ddpg as diagnostic
import explore_feature_ddpg as parent_tools
import train_response_predictors as shared
import smoke_feature_ddpg as smoke
from prediction_rl.data.collection_store import read_json as read, write_once, file_hash as sha, run_lock, seal_episode, verify_episode
from prediction_rl.data.dataset_contract import digest
from prediction_rl.data.frozen_dataset import inside
from prediction_rl.data.merge_geometry import read_extended_traffic
from prediction_rl.envs.upstream import upstream_session
from prediction_rl.evaluation.closed_loop_diagnostics import PROTOCOL as P7B
from prediction_rl.evaluation.external_baselines import (
    PROTOCOL, AUDIT, ARMS, CONFIGS, ASSETS, validate_protocol, jobs, common_settings, outcome, aggregate, parity)
from prediction_rl.training.all_ddpg import preset_factory, checkpoint_contract, load_evaluation_weights

SOURCE = parent_tools.SOURCE
MODEL = SOURCE/'pretrained_models/ddpg_default1_extended'


def assets():
    if {p.name for p in MODEL.iterdir()} != set(ASSETS):
        raise ValueError('Unexpected author model asset; refuse unverified pickle loading')
    actual = {k: sha(MODEL/k) for k in ASSETS}
    if actual != ASSETS: raise ValueError('Pinned author checkpoint changed')
    return actual


def configurations():
    return {n: sha(SOURCE/'configs'/n) for n in sorted(set(CONFIGS.values()))}


def parent(config):
    validate_protocol(config)
    if any(config[k] != P7B[k] for k in ('parent_request', 'parent_request_hash', 'parent_aggregate_sha256')):
        raise ValueError('Unknown parent')
    return diagnostic.load_parent(P7B)


def audit_evidence(path):
    path = inside(ROOT, path); q = read(path.parent/'request.json'); a = read(path)
    if (q['protocol'] != AUDIT or q['source_hashes'] != shared.source_hashes()
            or q['author_assets'] != assets() or q['upstream_config_hashes'] != configurations()
            or q['environment'] != shared.env() or q['runtime_identity'] != parent_tools.runtime_identity()
            or a['request_hash'] != digest(q) or a['engineering_complete'] is not True
            or read(path.parent/'preparation.json')['request_hash'] != digest(q)
            or read(path.parent/'aggregate_receipt.json') != {'request_hash': digest(q), 'sha256': sha(path)}):
        raise ValueError('Need accepted current external controller engineering audit')
    reports = [verify_episode(path.parent/'evaluate'/j['id'], binding(q, j))['metadata'] for j in jobs(AUDIT)]
    if any(r['no_policy_updates'] is not True or (r['outcome']['arm'] != 'conditional'
            and r['exact_raw_controller_parity'] is not True) for r in reports):
        raise ValueError('Invalid controller parity/frozen model audit')
    recomputed = aggregate(AUDIT, [r['outcome'] for r in reports])
    if any(a[k] != v for k, v in recomputed.items()): raise ValueError('Audit aggregate changed')
    return {'path': str(path.relative_to(ROOT)), 'sha256': sha(path), 'request_hash': digest(q)}


def prepare(config, run_id, audit_path=None):
    shared.clean(); pp, p = parent(config)
    evidence = None if config == AUDIT else audit_evidence(audit_path or 'artifacts/p7d/p7d_audit_v1/aggregate.json')
    if PROTOCOL['simulator_seeds'] != p['protocol']['evaluation']['simulator_seeds']:
        raise ValueError('Use existing complete development scene roster')
    resolved = {}
    for arm in ARMS:
        with upstream_session(SOURCE, SOURCE/'configs'/CONFIGS[arm], 200) as settings:
            settings.CUDA = False  # Declared CPU inference-only override, not reward/control.
            resolved[arm] = settings.export_settings()
    common = common_settings(resolved[ARMS[0]])
    if any(common_settings(v) != common for v in resolved.values()):
        raise ValueError('External configs change shared environment/reward/planner')
    if common['REWARD_FUNCTION'] != 'Slotted Jerk' or common['TICK_LENGTH'] != .2:
        raise ValueError('Original reward/period changed')
    out = shared.output_dir('p7d', run_id)
    q = {**shared.metadata(), 'version': 'p07d_request_v1', 'protocol': deepcopy(config), 'jobs': jobs(config),
         'source_hashes': shared.source_hashes(), 'environment': shared.env(),
         'runtime_identity': parent_tools.runtime_identity(), 'author_assets': assets(),
         'upstream_config_hashes': configurations(), 'common_settings_sha256': digest(common),
         'parent_request_hash': digest(p), 'resolved_settings_at_reference_seed': resolved,
         'engineering_audit': evidence}
    write_once(out/'request.json', q)
    write_once(out/'preparation.json', {'request_hash': digest(q), 'status': 'prepared_not_simulated', 'evaluation_episodes': len(q['jobs'])})
    print('external_request='+str(out/'request.json'), flush=True); print('confirm_request_hash='+digest(q), flush=True)
    return out/'request.json', digest(q)


def load(path, expected):
    shared.clean(); path = inside(ROOT, path); q = read(path); validate_protocol(q['protocol'])
    if (q['version'] != 'p07d_request_v1' or digest(q) != expected
            or read(path.parent/'preparation.json')['request_hash'] != expected
            or q['source_hashes'] != shared.source_hashes() or q['environment'] != shared.env()
            or q['runtime_identity'] != parent_tools.runtime_identity() or q['author_assets'] != assets()
            or q['upstream_config_hashes'] != configurations() or q['jobs'] != jobs(q['protocol'])):
        raise ValueError('External request/source/environment/config changed')
    pp, p = parent(q['protocol'])
    if digest(p) != q['parent_request_hash']: raise ValueError('Parent changed')
    if q['protocol'] == PROTOCOL:
        if audit_evidence(q['engineering_audit']['path']) != q['engineering_audit']: raise ValueError('Engineering audit changed')
    elif q['engineering_audit'] is not None: raise ValueError('Self-dependent audit')
    return path, q, pp, p


class ConditionalController:
    """Frozen P7 features on ACTUAL observed states, continuous jerk unchanged."""
    def __init__(self, p, pp, job, settings):
        train = next(j for j in p['train_jobs'] if j['id'] == job['training_id']); td = pp.parent/'train'/train['id']
        m = verify_episode(td, parent_tools.binding(p, 'train', train))['metadata']
        self.provider, predictor_sha = parent_tools.provider_for(p, 'conditional')
        self.env = parent_tools.environment(p, {'arm': 'conditional'}, self.provider, None)
        from all.logging import DummyWriter
        self.env._lazy_init()  # Observation codec ONLY; no Gym reset/step.
        self.checked = preset_factory('cpu', settings.LEARNING_RATE)(self.env, DummyWriter())
        contract = checkpoint_contract(self.env, p['preset'], predictor_sha, p['upstream_config_sha256'])
        if contract != m['weights_contract']: raise ValueError('P7 frozen feature/weight contract changed')
        load_evaluation_weights(self.checked, td/'final.pt', contract, m['weights_sha256'])
        self.weights_sha, self.predictor_sha = m['weights_sha256'], predictor_sha
        self.history = deque(maxlen=11); self.last_obs = None; self.actions = []; self.before = self.fingerprint()

    def fingerprint(self):
        return {'policy': smoke.tensor_hash(self.checked.agent.policy.model.state_dict()),
                'q': smoke.tensor_hash(self.checked.agent.q.model.state_dict()), 'predictor': smoke.tensor_hash(self.provider.model.state_dict())}

    def __call__(self, state):
        import dqn, control
        self.history.append(read_extended_traffic())
        base = np.asarray(dqn.get_state_vector_from_base_state(state), dtype=np.float32)
        features = self.provider(list(self.history))
        if base.shape != (20,) or features.shape != (149,) or features.dtype != np.float32:
            raise ValueError('Original/expanded observation layout changed')
        obs = np.concatenate([base, features])
        if obs.shape != (169,) or not np.isfinite(obs).all(): raise ValueError('Invalid observed features')
        self.last_obs = obs
        jerk = float(self.checked.eval(self.env._make_state(obs, False), 0).item())
        self.actions.append(jerk)
        return control.set_ego_jerk(jerk)

    def end(self, state):
        # Original terminal mask resets TimeFeature; no prediction or action executed.
        obs = np.concatenate([self.last_obs[:20], np.zeros(149, np.float32)])
        self.checked.eval(self.env._make_state(obs, True), 0)

    def verify_frozen(self):
        if (self.before != self.fingerprint() or len(self.checked.agent.replay_buffer)
                or any(a._optimizer.state for a in (self.checked.agent.policy, self.checked.agent.q))
                or any(p.grad is not None or p.requires_grad for p in self.provider.model.parameters())):
            raise ValueError('Evaluation changed frozen model/optimizer/replay')


def episode(pp, p, q, job, instrument=True):
    from all.policies.deterministic import DeterministicPolicyNetwork
    policy = None; current = None
    with upstream_session(SOURCE, SOURCE/'configs'/job['config'], job['simulator_seed']) as settings:
        import ddpg, dqn, control, prediction, st, sumo, rl, merge_gym
        settings.CUDA = False; resolved = settings.export_settings()
        if digest(common_settings(resolved)) != q['common_settings_sha256']: raise ValueError('Shared runtime semantics changed')
        queries = []; frames = []; latencies = []; decision_queries = []
        if job['arm'] == 'conditional':
            current = ConditionalController(p, pp, job, settings); controller, callback = current, current.end
        elif job['arm'] == 'author_st':
            sumo.start_sumo(); controller, callback = st.do_st_control, None
        else:
            assets(); merge_gym.register_environments()
            policy = ddpg.DDPGAgent.load(str(MODEL)); network = policy.agent.agent.policy
            if type(network) is not DeterministicPolicyNetwork or any(isinstance(m, (
                    torch.nn.modules.dropout._DropoutNd, torch.nn.modules.batchnorm._BatchNorm)) for m in network.modules()):
                raise ValueError('Unknown/stateful author policy architecture')
            frozen = smoke.tensor_hash(network.state_dict())
            if instrument:
                original_query = policy.get_control
                def query(state):
                    jerk = float(original_query(state))
                    if not np.isfinite(jerk) or not -5 <= jerk <= 5: raise ValueError('Invalid original policy output')
                    queries.append(jerk); return jerk
                policy.get_control = query  # Local-object recording, same original query count/order.
            controller = policy.do_control if job['arm'] == 'author_ddpg' else policy.do_combined_control
            callback = policy.end_episode_callback
        def recorded(state):
            frames.append(read_extended_traffic()); start = time.perf_counter(); n = len(queries)
            speed = controller(state); latencies.append(time.perf_counter()-start); decision_queries.append(queries[n:])
            if not np.isfinite(float(speed)): raise ValueError('Nonfinite commanded speed')
            return speed
        e = control.run_episode(recorded, prediction.HighwayState.from_sumo, max_episode_length=q['protocol']['max_episode_s'],
            wait_before_start=q['protocol']['warmup_s'], end_episode_callback=callback)
        takeovers = list(policy.takeover_history) if policy is not None and 'mpc' in job['arm'] else None
        if policy is not None and frozen != smoke.tensor_hash(network.state_dict()): raise ValueError('Author policy weights changed')
        if current is not None: current.verify_frozen()
        score = rl.get_rl_custom_episode_stats(e, dqn.get_reward_function())['total_reward']
        initial = digest(frames[0]) if frames else None
        if initial is None: raise ValueError('Episode ended without any decision')
        row = {**job, **outcome(e, initial, takeovers, latencies, score),
            'policy_sha256': current.weights_sha if current else ASSETS['policy.pt'] if policy else None,
            'predictor_sha256': current.predictor_sha if current else None,
            'policy_queries': len(current.actions) if current else len(queries) if instrument and policy else None}
        histories = {k: e[k] for k in ('crashed', 'merged', 'control_history', 'position_history', 'speed_history',
            'acceleration_history', 'jerk_history', 'closest_vehicle_history', 'disruption_history', 'simulation_time_taken')}
        return {'outcome': row, 'histories': histories, 'takeovers': takeovers, 'actual_observed_frames': frames,
                'decision_policy_queries': decision_queries, 'requested_conditional_jerks': current.actions if current else None,
                'resolved_config': {'upstream': resolved, 'protocol': q['protocol'], 'job': job,
                    'model_path_mapping': None if policy is None else {'original': resolved['MODEL_NAME'], 'loaded': str(MODEL.relative_to(ROOT))}},
                'no_policy_updates': True}


def binding(q, j): return {'request_hash': digest(q), 'stage': 'external_evaluation', 'job': j}


def worker(args):
    path, q, pp, p = load(args.request, args.confirm_request_hash); job = next(j for j in q['jobs'] if j['id'] == args.job)
    out = path.parent/'evaluate'/job['id']; out.mkdir(parents=True, exist_ok=False)
    started = shared.metadata(); write_once(out/'started.json', started); torch.set_num_threads(q['protocol']['torch_threads'])
    try:
        result = episode(pp, p, q, job)
        if q['protocol'] == AUDIT and job['arm'] != 'conditional':
            raw = episode(pp, p, q, job, instrument=False)
            result['exact_raw_controller_parity'] = parity(raw['histories'], result['histories'])
            if raw['takeovers'] != result['takeovers'] or raw['outcome']['initial_traffic_sha256'] != result['outcome']['initial_traffic_sha256']:
                raise ValueError('Recording changed supervisor/initial scene')
            write_once(out/'raw_author_replay.json', raw)
        write_once(out/'resolved_config.json', result['resolved_config'])
        write_once(out/'episode.json', result['outcome']); write_once(out/'trace.json', result)
        meta = {**started, 'status': 'complete', 'outcome': result['outcome'], 'no_policy_updates': True,
                'exact_raw_controller_parity': result.get('exact_raw_controller_parity'), 'finished_at': shared.now()}
        seal_episode(out, binding(q, job), meta)
    except BaseException:
        write_once(out/'failure.json', {**started, 'failure_reason': traceback.format_exc()}); raise


def aggregate_request(path, expected):
    path, q, _, _ = load(path, expected); out = path.parent
    with run_lock(out):
        reports = [verify_episode(out/'evaluate'/j['id'], binding(q, j))['metadata'] for j in q['jobs']]
        if any(r['no_policy_updates'] is not True for r in reports): raise ValueError('Model changed')
        for j, r in zip(q['jobs'], reports):
            if read(out/'evaluate'/j['id']/'episode.json') != r['outcome']: raise ValueError('Outcome receipt mismatch')
        if q['protocol'] == AUDIT and any(r['exact_raw_controller_parity'] is not True for r in reports if r['outcome']['arm'] != 'conditional'):
            raise ValueError('Author controller parity incomplete')
        a = {**shared.metadata(), **aggregate(q['protocol'], [r['outcome'] for r in reports]), 'request_hash': digest(q), 'finished_at': shared.now()}
        if (out/'aggregate.json').exists():
            old = read(out/'aggregate.json')
            if any(old[k] != a[k] for k in a if k not in ('git_commit', 'command', 'started_at', 'finished_at')):
                raise ValueError('Prior immutable aggregate not reproducible')
            if read(out/'aggregate_receipt.json') != {'request_hash': digest(q), 'sha256': sha(out/'aggregate.json')}: raise ValueError('Aggregate receipt changed')
        else:
            write_once(out/'aggregate.json', a); write_once(out/'aggregate_receipt.json', {'request_hash': digest(q), 'sha256': sha(out/'aggregate.json')})
    print('external_aggregate='+str(out/'aggregate.json'), flush=True); return out/'aggregate.json'


def execute(path, expected, resume=False):
    path, q, _, _ = load(path, expected); out = path.parent
    invocation = out/'invocations'/uuid.uuid4().hex[:12]; invocation.mkdir(parents=True, exist_ok=False)
    report = {**shared.metadata(), 'request_hash': digest(q), 'jobs': [], 'status': 'running'}
    with run_lock(out):
        try:
            pending = []
            for j in q['jobs']:
                directory = out/'evaluate'/j['id']
                if directory.exists():
                    if not resume: raise ValueError('Use --resume only for verified COMPLETE episodes')
                    verify_episode(directory, binding(q, j)); report['jobs'].append({'id': j['id'], 'action': 'reuse'})
                else: pending.append(j)
            def child(j):
                command = [sys.executable, str(Path(__file__).resolve()), 'worker', '--request', str(path), '--confirm-request-hash', expected, '--job', j['id']]
                print('[p7d] evaluating '+j['id'], flush=True)
                with (invocation/(j['id']+'.log')).open('w', encoding='utf-8') as log:
                    subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
                verify_episode(out/'evaluate'/j['id'], binding(q, j)); print('[p7d] complete '+j['id'], flush=True)
                return {'id': j['id'], 'action': 'run'}
            with ThreadPoolExecutor(max_workers=q['protocol']['workers']) as pool:
                width = q['protocol']['workers']
                for i in range(0, len(pending), width): report['jobs'].extend(f.result() for f in [pool.submit(child, j) for j in pending[i:i+width]])
            report.update(status='complete', finished_at=shared.now())
        except BaseException:
            report.update(status='failed', failure_reason=traceback.format_exc()); raise
        finally: write_once(invocation/'report.json', report)
    return aggregate_request(path, expected)


def main():
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest='command', required=True)
    c = sub.add_parser('prepare'); c.add_argument('--config', default='configs/development/p07d_external_baselines_v1.json'); c.add_argument('--run-id', required=True); c.add_argument('--engineering-audit')
    for name in ('run', 'aggregate', 'worker'):
        c = sub.add_parser(name); c.add_argument('--request', required=True); c.add_argument('--confirm-request-hash', required=True)
        if name == 'run': c.add_argument('--resume', action='store_true')
        if name == 'worker': c.add_argument('--job', required=True)
    c = sub.add_parser('audit'); c.add_argument('--run-id', required=True)
    args = parser.parse_args()
    if args.command == 'prepare': prepare(read(inside(ROOT, args.config)), args.run_id, args.engineering_audit)
    elif args.command == 'worker': worker(args)
    elif args.command == 'run': execute(args.request, args.confirm_request_hash, args.resume)
    elif args.command == 'aggregate': aggregate_request(args.request, args.confirm_request_hash)
    else:
        path, h = prepare(AUDIT, args.run_id); execute(path, h)


if __name__ == '__main__': main()
