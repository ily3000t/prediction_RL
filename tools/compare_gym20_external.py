"""P7l manual-start author Gym20 comparison; reuse immutable P7k 60k episodes."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
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
import train_response_predictors as shared
import train_matched_ddpg as matched
import explore_feature_ddpg as original
import compare_external_baselines as external
import crosscheck_model_protocols as cross
import smoke_feature_ddpg as smoke
from prediction_rl.data.collection_store import (
    read_json as read, write_once, file_hash as sha, run_lock, seal_episode, verify_episode)
from prediction_rl.data.dataset_contract import digest
from prediction_rl.data.frozen_dataset import inside, verify_historical_inputs, verify_historical_text_checkout
from prediction_rl.envs.upstream import upstream_session
from prediction_rl.envs.gym20_speed_controller import Gym20SpeedControllerEnv
from prediction_rl.envs.prediction_features import PredictionFeatureEnv
from prediction_rl.training.all_ddpg import make_all_environment
from prediction_rl.evaluation.exploration import EpisodeMetrics
from prediction_rl.evaluation.external_baselines import ASSETS, CONFIGS, common_settings
from prediction_rl.evaluation.gym20_external import (
    PROTOCOL, AUDIT, AUTHORS, PROJECT, jobs, project_job, validate_protocol, aggregate, summarize)
from prediction_rl.evaluation.terminal_events import StepObserver
from prediction_rl.evaluation.terminal_events_v2 import classify


def parent_evidence(*, verify_files=False):
    path = inside(ROOT, PROTOCOL['parent_request']); q = read(path); root = path.parent
    ap = root/'aggregate.json'; a = read(ap)
    current_sources = shared.source_hashes()
    if (digest(q) != PROTOCOL['parent_request_hash'] or sha(ap) != PROTOCOL['parent_aggregate_sha256']
            or q['protocol'] != matched.PROTOCOL or a['request_hash'] != digest(q)
            or a['status'] != 'complete' or a['primary_node'] != 60000
            or a['engineering_complete'] is not True or a['test_opened'] is not False
            or read(root/'preparation.json')['request_hash'] != digest(q)
            or q['upstream_hashes'] != original.upstream_hashes()
            or q['runtime_identity'] != original.runtime_identity() or q['dependencies'] != smoke.dependencies()
            or any(current_sources.get(k) != v for k, v in q['source_hashes'].items())):
        raise ValueError('Completed P7k source/runtime/results changed; no historical loader relaxation')
    if (root/'writer.lock').exists():
        raise ValueError('P7k writer lock exists; do not use active or unresolved output')
    if verify_files:
        verify_historical_inputs(ROOT, {'python_lf:'+k: v for k, v in q['source_hashes'].items()}, q['git_commit'])
        verify_historical_text_checkout(ROOT, q['config_path'], q['config_sha256'], q['git_commit'])
    inputs = {str(p.relative_to(ROOT)): sha(p) for p in (path, ap, root/'aggregate_receipt.json')}
    primary = next(n for n in a['nodes'] if n['requested_training_steps'] == 60000)
    rows = []
    for tj in q['train_jobs']:
        directory = root/'train'/tj['id']; receipt = read(directory/'complete.json')
        metadata = receipt['metadata']
        if (receipt['status'] != 'complete' or receipt['binding'] != matched.binding(q, 'train', tj)
                or metadata != a['training'][tj['id']] or metadata['predictor_frozen'] is not True):
            raise ValueError('P7k training receipt lineage changed')
        snap = next(s for s in metadata['snapshots'] if s['requested_steps'] == 60000)
        matched.snapshot_steps(snap['actual_steps'], 60000)
        if read(directory/'n60000.json') != snap or sha(directory/'n60000.pt') != snap['weights_sha256']:
            raise ValueError('P7k primary checkpoint changed')
        for name in ('complete.json', 'n60000.json', 'n60000.pt'):
            p = directory/name; inputs[str(p.relative_to(ROOT))] = sha(p)
    for j in q['evaluation_jobs']:
        if j['checkpoint_frames'] != 60000:
            continue
        d = root/'evaluate'/j['id']; cp = d/'complete.json'; ep = d/'episode.json'
        receipt = verify_episode(d, matched.binding(q, 'evaluate', j)) if verify_files else read(cp)
        r = read(ep)
        if (receipt['status'] != 'complete' or receipt['binding'] != matched.binding(q, 'evaluate', j)
                or receipt['metadata']['outcome'] != r or receipt['metadata']['no_policy_updates'] is not True
                or any(r.get(k) != v for k, v in j.items()) or r['native_event_agrees'] is not True
                or r['policy_sha256'] != next(s for s in a['training'][j['training_id']]['snapshots']
                    if s['requested_steps'] == 60000)['weights_sha256']):
            raise ValueError('P7k final episode/weights/event evidence changed')
        for p in (cp, ep):
            inputs[str(p.relative_to(ROOT))] = sha(p)
        rows.append({**r, **project_job(j['arm'], j['run_seed'], j['simulator_seed']),
                     'source_job_id': j['id'], 'adapter_parity': True})
    for arm in PROJECT:
        for seed in PROTOCOL['project_run_seeds']:
            selected = [r for r in rows if r['arm'] == arm and r['run_seed'] == seed]
            if len(selected) != 20 or any(summarize(selected)[m] != expected for m, expected in
                    primary['arms'][arm]['per_run_seed'][str(seed)].items()):
                raise ValueError('Reused final episode rows differ from accepted aggregate')
    return inputs, rows


def runtime():
    return {'source_hashes': shared.source_hashes(), 'dependencies': smoke.dependencies(),
        'environment': shared.env(), 'runtime_identity': original.runtime_identity(),
        'upstream_hashes': original.upstream_hashes(), 'author_assets': external.assets(),
        'author_config_hashes': external.configurations()}


def audit_evidence(path):
    path = inside(ROOT, path); q = read(path.parent/'request.json'); a = read(path)
    if (q['protocol'] != AUDIT or q['runtime'] != runtime() or q['jobs'] != jobs(AUDIT)
            or a['request_hash'] != digest(q) or a['status'] != 'complete'
            or a['engineering_complete'] is not True or a['new_author_episodes'] != 4
            or read(path.parent/'aggregate_receipt.json') != {'request_hash': digest(q), 'aggregate_sha256': sha(path)}):
        raise ValueError('Need completed current four-controller Gym20 adapter audit')
    for j in q['jobs']:
        m = verify_episode(path.parent/'evaluate'/j['id'], binding(q, j))['metadata']
        if m['adapter_parity'] is not True or m['outcome']['adapter_parity'] is not True:
            raise ValueError('Adapter reference parity failed')
    return {'path': str(path.relative_to(ROOT)), 'sha256': sha(path), 'request_hash': digest(q)}


def binding(q, job):
    return {'request_hash': digest(q), 'stage': 'gym20_external_evaluation', 'job': job}


def controller_environment(base):
    env = make_all_environment(PredictionFeatureEnv(base, 'baseline'), 'cpu')
    # Unlike DDPG's env.reset(), speed-controller reset is owned by the base.
    # Initialize ALL's uint8 masks without resetting SUMO or querying policy.
    env._lazy_init()
    return env


def prepare(config_path, run_id, adapter_audit=None):
    shared.clean(); torch.set_num_threads(1)
    p = AUDIT if config_path is None else validate_protocol(read(inside(ROOT, config_path)))
    inputs, project = parent_evidence(verify_files=True)
    evidence = None if p == AUDIT else audit_evidence(adapter_audit or 'artifacts/p7l/p7l_audit_v2/aggregate.json')
    settings_hashes = []
    for arm in AUTHORS:
        with upstream_session(original.SOURCE, original.SOURCE/'configs'/CONFIGS[arm], 200) as settings:
            settings.CUDA = False
            if settings.REWARD_FUNCTION != 'Slotted Jerk' or settings.INVALID_ACTION_PENALTY != 0:
                raise ValueError('Unsupported planner reward contract')
            settings_hashes.append(digest(common_settings(settings.export_settings())))
    if len(set(settings_hashes)) != 1:
        raise ValueError('Selected author configs differ in shared environment/reward')
    out = shared.output_dir('p7l', run_id)
    q = {**shared.metadata(), 'version': 'p07l_request_v1', 'protocol': deepcopy(p),
         'config_path': config_path, 'config_sha256': None if config_path is None else sha(inside(ROOT, config_path)),
         'runtime': runtime(), 'parent_inputs': inputs, 'jobs': jobs(p), 'adapter_audit': evidence,
         'common_settings_sha256': settings_hashes[0]}
    write_once(out/'request.json', q)
    write_once(out/'preparation.json', {'request_hash': digest(q), 'status': 'prepared_not_evaluated',
        'new_author_episodes': len(q['jobs']), 'reused_project_episodes':
        sum(r['simulator_seed'] in p['simulator_seeds'] for r in project),
        'training_permitted': False, 'audit_reference_episodes': 4 if p == AUDIT else 0})
    print('gym20_external_request='+str(out/'request.json'), flush=True)
    print('confirm_request_hash='+digest(q), flush=True)
    return out/'request.json', digest(q)


def load(path, expected):
    shared.clean(); torch.set_num_threads(1); path = inside(ROOT, path); q = read(path)
    validate_protocol(q['protocol'])
    if (path.parent.parent != ROOT/'artifacts/p7l' or q['version'] != 'p07l_request_v1'
            or digest(q) != expected or read(path.parent/'preparation.json')['request_hash'] != expected
            or q['runtime'] != runtime() or q['jobs'] != jobs(q['protocol'])):
        raise ValueError('Frozen request/source/runtime changed')
    if q['config_path'] is not None and (sha(inside(ROOT, q['config_path'])) != q['config_sha256']
            or read(inside(ROOT, q['config_path'])) != q['protocol']):
        raise ValueError('Frozen configuration changed')
    if any(sha(inside(ROOT, p)) != h for p, h in q['parent_inputs'].items()):
        raise ValueError('Immutable parent/checkpoint/episode evidence changed')
    if q['protocol'] != AUDIT and q['adapter_audit'] != audit_evidence(q['adapter_audit']['path']):
        raise ValueError('Accepted adapter audit changed')
    return path, q


def episode(q, job, *, observed=True):
    metrics = EpisodeMetrics(); trace = []; decisions = []; low = []
    with upstream_session(original.SOURCE, original.SOURCE/'configs'/job['config'], job['simulator_seed']) as settings:
        import control, traci, prediction, dqn, ddpg, st
        settings.CUDA = False
        if digest(common_settings(settings.export_settings())) != q['common_settings_sha256']:
            raise ValueError('Shared environment/reward changed')
        identity = {'id': job['id'], 'model_id': job['arm'], 'condition_id': 'gym20',
                    'run_seed': None, 'simulator_seed': job['simulator_seed'], 'training_id': None}
        context = StepObserver(control, traci, identity) if observed else nullcontext(None)
        with context as observer:
            speed_base = None; body = policy = critic = agent = query = None
            def record(event, obs, reward, done, info):
                metrics.record(event, obs, reward, done, info)
                if event == 'step':
                    decisions.append({'frame': info['execution_audit']['before']})
                    low.append(info['execution_audit']['before']['vehicles']['ego']['speed'] <= .1)
                    trace.append({'reward': float(reward), 'done': bool(done),
                                  'execution_audit': info['execution_audit']})
            if job['arm'] == 'author_ddpg':
                env = original.environment({'protocol': {'training': {'device': 'cpu'}}},
                    {'arm': 'baseline'}, None, record)
            else:
                speed_base = Gym20SpeedControllerEnv(settings)
                env = controller_environment(speed_base)
            try:
                before = None; virtual_queries = []
                if job['arm'] != 'author_st':
                    body, policy, critic = cross.author_policy(env)
                    before = {'policy': smoke.tensor_hash(policy.state_dict()), 'q': smoke.tensor_hash(critic.state_dict())}
                    if job['arm'] == 'author_ddpg':
                        query = cross.FrozenQuery(body, policy)
                    else:
                        # Use original DDPGAgent.get_control/do_combined_control,
                        # but bind the already-owned Gym environment and original
                        # pinned networks; its constructor would start a second SUMO.
                        agent = ddpg.DDPGAgent.__new__(ddpg.DDPGAgent)
                        dqn.RLAgent.__init__(agent)
                        agent.env, agent.agent, agent.device = env, body, 'cpu'
                        original_query = agent.get_control
                        def checked_query(highway):
                            features = dqn.get_state_vector_from_base_state(highway)
                            state = env._make_state(features, False)
                            t = 0 if body.timestep is None else int(body.timestep.item())
                            # Independent original-network check without a second
                            # TimeFeature call or any mutation of controller state.
                            from all.environments import State
                            augmented = State(torch.cat([state.features,
                                torch.tensor([[t*.001]], dtype=torch.float32)], 1), state.mask, state.info)
                            # Match TimeFeature's float32 multiplication exactly.
                            augmented.features[0, -1] = torch.tensor(float(t))*.001
                            with torch.no_grad(): reference = policy(augmented)
                            value = float(original_query(highway))
                            if (not np.isfinite(value) or not -5 <= value <= 5
                                    or value != float(reference.item()) or int(body.timestep.item()) != t+1):
                                raise ValueError('Native virtual policy query/TimeFeature parity changed')
                            virtual_queries.append({'time_index': t, 'jerk': value, 'observation': state.features[0].tolist()})
                            return value
                        if observed:
                            agent.get_control = checked_query
                        speed_base.bind(agent.do_combined_control)
                else:
                    speed_base.bind(st.do_st_control)
                if speed_base is None:
                    state = env.reset(); reward = 0.
                else:
                    obs, info = speed_base.reset(); record('reset', obs, None, False, info)
                query_counts = []
                for step in range(500):
                    start = 0 if body is None or body.timestep is None else int(body.timestep.item())
                    if speed_base is None:
                        cross.state_vector_parity(state)
                        action = query(state, reward)
                        state, reward = env.step(action); done = bool(env.done)
                    else:
                        obs, reward, done, info = speed_base.step()
                        record('step', obs, reward, done, info)
                    end = 0 if body is None else int(body.timestep.item())
                    query_counts.append(end-start)
                    if done:
                        terminal = state if speed_base is None else env._make_state(obs, True)
                        if query is not None:
                            query(terminal, reward); query.complete(step+1)
                        elif body is not None:
                            body.eval(terminal, reward)
                            if int(body.timestep.item()) != 0:
                                raise ValueError('Gym terminal query failed to reset TimeFeature')
                        break
                else:
                    raise ValueError('Gym20 failed to end within 500 original control steps')
                if before is not None and before != {
                        'policy': smoke.tensor_hash(policy.state_dict()), 'q': smoke.tensor_hash(critic.state_dict())}:
                    raise ValueError('Frozen author weights changed')
                native = metrics.result()
                raw = None if observer is None else classify(observer.export(),
                    {**identity, **native, 'reported_collision': native['collision']}, decisions)
                if raw is not None and (not raw['native_event_agrees'] or raw['classification'] not in
                        ('natural_arrival', 'ego_collision', 'time_limit')):
                    raise ValueError('Invalid Gym outcome: do not turn an event error into normal results')
                takeovers = None if agent is None else list(agent.takeover_history)
                if takeovers is not None and (len(takeovers) != native['steps'] or any(type(x) is not bool for x in takeovers)):
                    raise ValueError('Original supervisor decision count changed')
                row = {**job, **native, 'natural_arrival': native['arrival'], 'ego_collision': native['collision'],
                    'native_event_agrees': None if raw is None else raw['native_event_agrees'],
                    'raw_outcome': raw, 'low_speed_fraction': sum(low)/len(low),
                    'action_saturation_fraction': (sum(abs(r['execution_audit']['requested_jerk']) >= 4.99 for r in trace)/len(trace)
                        if speed_base is None else None),
                    'st_takeover_rate': None if takeovers is None else sum(takeovers)/len(takeovers),
                    'policy_sha256': None if job['arm'] == 'author_st' else ASSETS['policy.pt'],
                    'no_policy_updates': True, 'adapter_parity': True}
                return {'outcome': row, 'actions': trace, 'query_counts': query_counts,
                    'virtual_queries': virtual_queries, 'takeovers': takeovers,
                    'raw_events': None if observer is None else observer.export(),
                    'resolved_config': {'upstream': settings.export_settings(), 'protocol': q['protocol'],
                        'job': job, 'original_policy_loaded_without_partial_transplant': True}}
            finally:
                env.close()


def assert_adapter_parity(reference, recorded):
    # Raw observers may differ, never the actual native commands/transitions.
    for field in ('actions', 'query_counts', 'takeovers'):
        if reference[field] != recorded[field]:
            raise ValueError('Recording/controller-query wrapper changed native trajectory: '+field)
    for field in ('return', 'steps', 'arrival', 'collision', 'time_limit', 'initial_traffic_sha256'):
        if reference['outcome'][field] != recorded['outcome'][field]:
            raise ValueError('Original native Gym result changed: '+field)
    return True


def worker(args):
    path, q = load(args.request, args.confirm_request_hash)
    job = next(j for j in q['jobs'] if j['id'] == args.job)
    out = path.parent/'evaluate'/job['id']; out.mkdir(parents=True, exist_ok=False)
    write_once(out/'started.json', {**shared.metadata(), 'binding': binding(q, job)})
    try:
        reference = episode(q, job, observed=False) if q['protocol'] == AUDIT else None
        result = episode(q, job)
        if reference is not None:
            assert_adapter_parity(reference, result)
            write_once(out/'unrecorded_reference.json', reference)
        row = result['outcome']
        write_once(out/'episode.json', row); write_once(out/'trace.json', result)
        metadata = {'status': 'complete', 'outcome': row, 'adapter_parity': True,
                    'no_policy_updates': True, 'finished_at': shared.now()}
        seal_episode(out, binding(q, job), metadata)
    except BaseException:
        write_once(out/'failure.json', {'status': 'failed', 'failure_reason': traceback.format_exc()})
        raise


def aggregate_run(path, expected):
    path, q = load(path, expected)
    with run_lock(path.parent):
        _, project = parent_evidence(verify_files=True)
        rows = []
        for j in q['jobs']:
            d = path.parent/'evaluate'/j['id']; m = verify_episode(d, binding(q, j))['metadata']
            r = read(d/'episode.json')
            if r != m['outcome'] or m['adapter_parity'] is not True:
                raise ValueError('External receipt/outcome changed')
            rows.append(r)
        project = [r for r in project if r['simulator_seed'] in q['protocol']['simulator_seeds']]
        result = {**aggregate(q['protocol'], rows, project), 'request_hash': digest(q),
                  'provenance': shared.metadata(), 'finished_at': shared.now()}
        write_once(path.parent/'aggregate.json', result)
        write_once(path.parent/'aggregate_receipt.json', {'request_hash': digest(q), 'aggregate_sha256': sha(path.parent/'aggregate.json')})
    print('gym20_external_aggregate='+str(path.parent/'aggregate.json'), flush=True)


def run(path, expected, resume=False):
    path, q = load(path, expected); root = path.parent
    invocation = root/'invocations'/uuid.uuid4().hex[:10]; invocation.mkdir(parents=True, exist_ok=False)
    report = {**shared.metadata(), 'request_hash': digest(q), 'jobs': []}
    with run_lock(root):
        try:
            pending = []
            for j in q['jobs']:
                d = root/'evaluate'/j['id']
                if d.exists():
                    if not resume:
                        raise ValueError('Existing output: --resume reuses verified complete whole episodes only')
                    verify_episode(d, binding(q, j)); report['jobs'].append({'id': j['id'], 'action': 'reuse'})
                else:
                    pending.append(j)
            def child(j):
                command = [sys.executable, '-B', str(Path(__file__).resolve()), 'worker', '--request', str(path),
                           '--confirm-request-hash', expected, '--job', j['id']]
                print('[p7l] '+j['id'], flush=True)
                log_path = invocation/(j['id']+'.log')
                with log_path.open('x', encoding='utf-8') as log:
                    result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                if result.returncode:
                    raise RuntimeError(f'Gym20 child failed: {j["id"]}; inspect {log_path}')
                verify_episode(root/'evaluate'/j['id'], binding(q, j))
                return {'id': j['id'], 'action': 'run'}
            width = q['protocol']['workers']
            with ThreadPoolExecutor(max_workers=width) as pool:
                for i in range(0, len(pending), width):
                    futures = [pool.submit(child, j) for j in pending[i:i+width]]
                    report['jobs'].extend(f.result() for f in futures)
            report.update(status='complete', finished_at=shared.now())
        except BaseException:
            report.update(status='failed', failure_reason=traceback.format_exc()); raise
        finally:
            write_once(invocation/'report.json', report)
    aggregate_run(path, expected)


def main():
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare'); p.add_argument('--config', default='configs/development/p07l_gym20_external_v1.json')
    p.add_argument('--run-id', required=True); p.add_argument('--adapter-audit', default='artifacts/p7l/p7l_audit_v2/aggregate.json')
    for name in ('run', 'worker', 'aggregate'):
        p = sub.add_parser(name); p.add_argument('--request', required=True); p.add_argument('--confirm-request-hash', required=True)
        if name == 'worker': p.add_argument('--job', required=True)
        if name == 'run': p.add_argument('--resume', action='store_true')
    p = sub.add_parser('audit'); p.add_argument('--run-id', required=True)
    a = parser.parse_args()
    if a.command == 'prepare': prepare(a.config, a.run_id, a.adapter_audit)
    elif a.command == 'worker': worker(a)
    elif a.command == 'run': run(a.request, a.confirm_request_hash, a.resume)
    elif a.command == 'aggregate': aggregate_run(a.request, a.confirm_request_hash)
    else:
        path, h = prepare(None, a.run_id); run(path, h)


if __name__ == '__main__':
    main()
