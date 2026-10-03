"""P7i: offline training audit plus USER-run frozen-policy exploration diagnosis."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import importlib.metadata
import inspect
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
import diagnose_feature_ddpg as parent_recording
import explore_feature_ddpg as original
import smoke_feature_ddpg as smoke
import train_response_predictors as shared
from prediction_rl.data.collection_store import (read_json as read, write_once, file_hash as sha,
    verify_episode, seal_episode, run_lock)
from prediction_rl.data.dataset_contract import digest
from prediction_rl.data.frozen_dataset import inside, verify_historical_inputs
from prediction_rl.evaluation.closed_loop_diagnostics import PROTOCOL as P7B
from prediction_rl.evaluation.ddpg_stability import (PROTOCOL, AUDIT, validate_protocol, roster,
    FrozenGaussian, scalar_summary, training_windows, control_summary, aggregate)
from prediction_rl.evaluation.exploration import EpisodeMetrics, zero_channel_query
from prediction_rl.evaluation.terminal_events import StepObserver
from prediction_rl.evaluation.terminal_events_v2 import classify
from prediction_rl.envs.upstream import upstream_session
from prediction_rl.training.all_ddpg import (preset_factory, checkpoint_contract, load_evaluation_weights,
    validate_action)


def noise_runtime():
    """Bind actual installed ALL semantics, not a reimplementation guessed from docs."""
    from all.agents.ddpg import DDPG
    from all.bodies import TimeFeature
    return {'all_version': importlib.metadata.version('autonomous-learning-library'),
            'tensorboard_version': importlib.metadata.version('tensorboard'),
            'implementation_hashes': {name: digest(inspect.getsource(fn)) for name, fn in
                [('init', DDPG.__init__), ('choose', DDPG._choose_action), ('train', DDPG._train),
                 ('eval', DDPG.eval), ('time', TimeFeature._append_time_feature)]}}


def load_parents(config):
    validate_protocol(config)
    pp, parent = parent_recording.load_parent(P7B)
    rp = inside(ROOT, config['recording_request']); recording = read(rp)
    if (digest(recording) != config['recording_request_hash'] or
            recording['protocol'] != P7B or recording['parent_request_hash'] != digest(parent) or
            read(rp.parent/'preparation.json')['request_hash'] != digest(recording) or
            sha(rp.parent/'summary.json') != config['recording_summary_sha256'] or
            (rp.parent/'writer.lock').exists()):
        raise ValueError('Frozen P7b recording changed or is active')
    verify_historical_inputs(ROOT, {'python_lf:'+k: v for k, v in recording['source_hashes'].items()}, recording['git_commit'])
    current = shared.source_hashes()
    if any(current.get(k) != v for k, v in recording['source_hashes'].items()):
        raise ValueError('An old executable changed; new sidecars do not permit rewriting it')
    reports = [verify_episode(rp.parent/'recording'/j['id'], parent_recording.binding(recording, j))['metadata']
               for j in recording['jobs']]
    summary = read(rp.parent/'summary.json')
    if (summary['jobs'] != reports or summary['all_replays_exact'] is not True or
            summary['request_hash'] != digest(recording) or
            read(rp.parent/'summary_receipt.json') !=
            {'request_hash': digest(recording), 'summary_sha256': sha(rp.parent/'summary.json')}):
        raise ValueError('P7b exact-replay evidence/receipt changed')
    return pp, parent


def training_artifacts(pp, parent, config):
    """Read all 12 frozen training receipts and logged scalars, no inference or labels."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    result = []; curves = {}
    prefix = 'sumo-jerk-continuous-v0/'
    tags = {'critic_td_mse': prefix+'loss/q', 'actor_negative_q': prefix+'loss/policy',
            'noisy_training_return': prefix+'evaluation/returns/frame'}
    for job in parent['train_jobs']:
        directory = pp.parent/'train'/job['id']
        receipt = verify_episode(directory, original.binding(parent, 'train', job)); m = receipt['metadata']
        episode_rows = read(directory/'episodes.json')
        event_files = sorted((directory/'training_logs').glob('events.out.tfevents.*'))
        if len(event_files) != 1:
            raise ValueError('Need exactly one frozen original training event file')
        events = EventAccumulator(str(event_files[0]), size_guidance={'scalars': 0}); events.Reload()
        if not set(tags.values()) <= set(events.Tags()['scalars']):
            raise ValueError('Missing actor/critic/return training scalars')
        values = {name: [{'step': int(r.step), 'value': float(r.value)} for r in events.Scalars(tag)]
                  for name, tag in tags.items()}
        if (len(values['critic_td_mse']) != m['optimizer_updates']['q'] or
                len(values['actor_negative_q']) != m['optimizer_updates']['policy'] or
                len(values['noisy_training_return']) != len(episode_rows) or
                any(a['step'] != b['finished_training_step']+1 or a['value'] != float(np.float32(b['return']))
                    for a, b in zip(values['noisy_training_return'], episode_rows))):
            raise ValueError('Training log no longer matches original episodes/optimizer counters')
        weights = torch.load(directory/'final.pt', map_location='cpu', weights_only=True)
        if weights['contract'] != m['weights_contract']:
            raise ValueError('Evaluation-only policy/critic contract changed')
        norms = {name: float(torch.sqrt(sum((v.double().square().sum() for v in state.values()))).item())
                 for name, state in weights['state_dicts'].items()}
        curves[job['id']] = values
        result.append({**job, 'actual_steps': m['actual_steps'], 'completed_episodes': len(episode_rows),
            'optimizer_updates': m['optimizer_updates'], 'checkpoint_sha256': m['weights_sha256'],
            'predictor_sha256': m['weights_contract']['predictor_sha256'], 'final_weight_l2': norms,
            'event_file_sha256': sha(event_files[0]),
            'scalars': {name: {'overall': scalar_summary(rows),
                              'windows': training_windows(rows, config['training_windows_steps'])}
                        for name, rows in values.items()},
            'episode_windows': {f'{lo}_{hi}': {
                'completed_episodes': len(v := [r for r in episode_rows if lo <= r['finished_training_step'] < hi]),
                'native_arrival_rate': None if not v else sum(r['arrival'] for r in v)/len(v),
                'native_collision_rate': None if not v else sum(r['collision'] for r in v)/len(v),
                'native_time_limit_rate': None if not v else sum(r['time_limit'] for r in v)/len(v)}
                for lo, hi in zip(config['training_windows_steps'], config['training_windows_steps'][1:])}})
    return {'status': 'complete', 'policies': result, 'training_started': False, 'inference_started': False,
            'test_opened': False, 'convergence_gate': None,
            'training_return_tags_are_not_deterministic_validation': True,
            'actor_loss_is_negative_critic_value_not_prediction_error': True,
            'critic_loss_decrease_does_not_prove_true_value_or_policy_convergence': True,
            'native_training_events_not_relabelled_without_raw_evidence': True}, curves


def prepare(config, run_id):
    shared.clean(); validate_protocol(config)
    pp, parent = load_parents(config)
    out = shared.output_dir('p7i', run_id); started = shared.metadata(); write_once(out/'started.json', started)
    try:
        audit, curves = training_artifacts(pp, parent, config)
        write_once(out/'training_audit.json', audit); write_once(out/'training_curves.json', curves)
        q = {**started, 'version': 'p07i_request_v1', 'protocol': deepcopy(config), 'jobs': roster(config, parent),
             'source_hashes': shared.source_hashes(), 'environment': shared.env(), 'noise_runtime': noise_runtime(),
             'training_audit_sha256': sha(out/'training_audit.json'), 'training_curves_sha256': sha(out/'training_curves.json')}
        write_once(out/'request.json', q)
        write_once(out/'preparation.json', {'request_hash': digest(q), 'status': 'prepared_no_rollouts_no_training'})
    except BaseException:
        write_once(out/'failure.json', {**started, 'failure_reason': traceback.format_exc()}); raise
    print('stability_request='+str(out/'request.json'), flush=True)
    print('confirm_request_hash='+digest(q), flush=True)
    return out/'request.json', digest(q)


def load(path, expected):
    shared.clean(); path = inside(ROOT, path); q = read(path); validate_protocol(q['protocol'])
    if (q['version'] != 'p07i_request_v1' or digest(q) != expected or
            read(path.parent/'preparation.json')['request_hash'] != expected or
            q['source_hashes'] != shared.source_hashes() or q['environment'] != shared.env() or
            q['noise_runtime'] != noise_runtime() or
            sha(path.parent/'training_audit.json') != q['training_audit_sha256'] or
            sha(path.parent/'training_curves.json') != q['training_curves_sha256']):
        raise ValueError('Stability request/source/runtime/audit changed')
    pp, parent = load_parents(q['protocol'])
    if q['jobs'] != roster(q['protocol'], parent):
        raise ValueError('Modified checkpoint/scene/noise roster')
    return path, q, pp, parent


def binding(q, job):
    return {'request_hash': digest(q), 'stage': 'frozen_stability', 'job': job}


def raw_identity(job):
    return {'id': job['id'], 'model_id': job['arm'], 'condition_id': 'gym20', 'run_seed': job['run_seed'],
            'simulator_seed': job['simulator_seed'], 'training_id': job['training_id']}


def run_episode(pp, parent, q, job, out):
    train_dir = pp.parent/'train'/job['training_id']
    train_job = next(j for j in parent['train_jobs'] if j['id'] == job['training_id'])
    meta = verify_episode(train_dir, original.binding(parent, 'train', train_job))['metadata']
    provider, predictor_sha = original.provider_for(parent, job['arm'])
    predictor_before = None if provider is None else smoke.tensor_hash(provider.model.state_dict())
    metrics = EpisodeMetrics()
    with upstream_session(original.SOURCE, original.UPSTREAM, job['simulator_seed']) as settings:
        import control, traci
        with StepObserver(control, traci, raw_identity(job)) as observer:
            env = original.environment(parent, job, provider, metrics.record)
            try:
                from all.logging import DummyWriter
                from all.environments import State
                checked = preset_factory('cpu', settings.LEARNING_RATE)(env, DummyWriter())
                contract = checkpoint_contract(env, parent['preset'], predictor_sha, parent['upstream_config_sha256'])
                if contract != meta['weights_contract']:
                    raise ValueError('Frozen input/action/time contract mismatch')
                load_evaluation_weights(checked, train_dir/'final.pt', contract, meta['weights_sha256'])
                initial = {n: smoke.tensor_hash(a.model.state_dict()) for n, a in
                           [('policy', checked.agent.policy), ('q', checked.agent.q)]}
                # Validate the installed agent's ACTUAL distribution and clipping bounds.
                if (checked.agent._noise.loc.item() != 0 or checked.agent._noise.scale.item() != .5 or
                        checked.agent._low.item() != -5 or checked.agent._high.item() != 5):
                    raise ValueError('Original exploration distribution changed')
                noise = None if job['condition'] == 'deterministic' else FrozenGaussian(
                    checked.low, checked.high, parent['preset']['noise'], job['simulator_seed'], job['noise_repeat'])
                write_once(out/'resolved_config.json', {'upstream': settings.export_settings(), 'protocol': q['protocol'],
                    'job': job, 'parent_preset': parent['preset'], 'policy_sha256': meta['weights_sha256'],
                    'predictor_sha256': predictor_sha, 'noise_seed': None if noise is None else noise.seed,
                    'noise_std_mps3': 0. if noise is None else noise.std})
                state = env.reset(); reward = 0.; trace = []; original_actions = []; sensitivity = []; decisions = []
                for step in range(q['protocol']['max_control_steps']):
                    before_time = torch.zeros(1) if checked.body.timestep is None else checked.body.timestep.clone()
                    critic_state = State(torch.cat([state.features, .001*before_time[:, None]], 1), state.mask, state.info)
                    actor = checked.eval(state, reward)  # Exactly ONE TimeFeature advance.
                    requested, fields = (actor, {'noise': 0., 'unclipped_jerk': float(actor.item()), 'clipped': False}) if noise is None else noise.apply(actor)
                    validate_action(requested, 1, checked.low, checked.high)
                    zero = None if provider is None else zero_channel_query(checked, state, before_time)
                    if zero is not None:
                        sensitivity.append(float((actor-zero).abs().item()))
                    q_actor = float(checked.agent.q.eval(critic_state, actor).item())
                    q_requested = float(checked.agent.q.eval(critic_state, requested).item())
                    if not np.isfinite([q_actor, q_requested]).all():
                        raise ValueError('Nonfinite frozen critic query')
                    previous = env._env.last_info
                    frame = previous['traffic'] if step == 0 else previous['execution_audit']['after']
                    decisions.append({'frame': deepcopy(frame)})
                    ego = frame['vehicles']['ego']
                    policy_input = critic_state.features[0].tolist()
                    state, reward = env.step(requested)
                    done = bool(env.done)
                    original_actions.append({'step': step, 'jerk': float(actor.item()), 'reward': float(reward), 'done': done,
                                            'zero_channel_query_jerk': None if zero is None else float(zero.item())})
                    trace.append({'step': step, 'actor_jerk': float(actor.item()), 'requested_jerk': float(requested.item()),
                        **fields, 'q_actor': q_actor, 'q_requested': q_requested, 'reward': float(reward), 'done': done,
                        'ddpg_input_with_time': policy_input, 'policy_state_mask': critic_state.mask.tolist(),
                        'speed_before_mps': ego['speed'], 'acceleration_before_mps2': ego['acceleration'],
                        'lane_before': ego['lane_id'], 'position_before': ego['position'],
                        'execution_audit': deepcopy(env._env.last_info['execution_audit'])})
                    if done:
                        checked.eval(state, reward)  # Original terminal time-counter handling.
                        break
                else:
                    raise ValueError('Original 500-step task limit not honored')
                if (len(checked.agent.replay_buffer) != 0 or
                        any(a._optimizer.state for a in (checked.agent.policy, checked.agent.q)) or
                        any(smoke.tensor_hash(a.model.state_dict()) != initial[n] for n, a in
                            [('policy', checked.agent.policy), ('q', checked.agent.q)])):
                    raise ValueError('Frozen diagnosis changed weights, optimizer or replay')
                if provider is not None and (smoke.tensor_hash(provider.model.state_dict()) != predictor_before or
                        any(p.grad is not None or p.requires_grad for p in provider.model.parameters())):
                    raise ValueError('Frozen prediction model changed')
                base = metrics.result()
                old = pp.parent/'evaluate'/job['parent_id']
                if base['initial_traffic_sha256'] != read(old/'episode.json')['initial_traffic_sha256']:
                    raise ValueError('Changed starting traffic')
                parity = None
                if noise is None:
                    old_job = next(j for j in parent['evaluation_jobs'] if j['id'] == job['parent_id'])
                    row = {**old_job, **base,
                        'mean_abs_zero_channel_action_change': None if not sensitivity else float(np.mean(sensitivity)),
                        'policy_sha256': meta['weights_sha256'], 'query_is_not_executed': True}
                    parity = parent_recording.assert_parity(row, original_actions, read(old/'episode.json'), read(old/'actions.json'))
                sidecar = observer.export()
                native = {**raw_identity(job), **base, 'reported_collision': base['collision']}
                events = classify(sidecar, native, decisions)
                summary = control_summary(trace, parent['preset']['discount_factor'], q['protocol']['low_speed_mps_descriptive'])
                result = {**job, **base, 'native_outcome': base, 'independent_events': events,
                    'natural_arrival': int(events['classification'] == 'natural_arrival'),
                    'ego_collision': int(events['classification'] == 'ego_collision'),
                    'time_limit': int(events['classification'] == 'time_limit'),
                    'other_terminal_event': int(events['classification'] not in ('natural_arrival', 'ego_collision', 'time_limit')),
                    **summary, 'exact_parent_parity': parity, 'no_policy_updates': True,
                    'policy_sha256': meta['weights_sha256'], 'predictor_sha256': predictor_sha,
                    'noise_draws': 0 if noise is None else noise.draws,
                    'noise_seed': None if noise is None else noise.seed}
                write_once(out/'trace.json', trace); write_once(out/'decisions.json', decisions)
                write_once(out/'raw_events.json', sidecar); write_once(out/'episode.json', result)
                return result
            finally:
                env.close()


def verify_result(directory, q, job):
    m = verify_episode(directory, binding(q, job))['metadata']
    row = read(directory/'episode.json'); trace = read(directory/'trace.json')
    if row != m['outcome'] or any(row.get(k) != v for k, v in job.items()) or row['no_policy_updates'] is not True:
        raise ValueError('Wrong frozen episode/receipt')
    events = classify(read(directory/'raw_events.json'),
                      {**raw_identity(job), **row['native_outcome'], 'reported_collision': row['native_outcome']['collision']},
                      read(directory/'decisions.json'))
    if row['independent_events'] != events:
        raise ValueError('Raw independent event analysis changed')
    derived = control_summary(trace, .98, q['protocol']['low_speed_mps_descriptive'])
    if any(row[k] != v for k, v in derived.items()):
        raise ValueError('Control/critic trace summary not reproducible')
    noise = None if job['condition'] == 'deterministic' else FrozenGaussian(-5., 5., .1, job['simulator_seed'], job['noise_repeat'])
    for r in trace:
        if noise is None:
            expected, fields = r['actor_jerk'], {'noise': 0., 'unclipped_jerk': r['actor_jerk'], 'clipped': False}
        else:
            action, fields = noise.apply(torch.tensor([[r['actor_jerk']]], dtype=torch.float32))
            expected = float(action.item())
        if r['requested_jerk'] != expected or any(r[k] != v for k, v in fields.items()):
            raise ValueError('Isolated noise trace changed')
    if row['noise_draws'] != (0 if noise is None else len(trace)) or row['noise_seed'] != (None if noise is None else noise.seed):
        raise ValueError('Noise draw/seed identity changed')
    return row


def worker(args):
    path, q, pp, parent = load(args.request, args.confirm_request_hash)
    job = next(j for j in q['jobs'] if j['id'] == args.job)
    out = path.parent/'episodes'/job['id']; out.mkdir(parents=True, exist_ok=False)
    started = {**shared.metadata(), 'binding': binding(q, job)}; write_once(out/'started.json', started)
    torch.set_num_threads(q['protocol']['torch_threads']); start = time.perf_counter()
    try:
        outcome = run_episode(pp, parent, q, job, out)
        result = {'status': 'complete', 'job': job, 'outcome': outcome, 'provenance': started,
                  'elapsed_s': time.perf_counter()-start, 'finished_at': shared.now()}
        write_once(out/'report.json', result); seal_episode(out, binding(q, job), result)
    except BaseException:
        write_once(out/'failure.json', {**started, 'failure_reason': traceback.format_exc()}); raise


def collect(path, q, parent):
    rows = [verify_result(path.parent/'episodes'/j['id'], q, j) for j in q['jobs']]
    return aggregate(q['protocol'], parent, rows)


def execute(path, expected, resume=False):
    path, q, _, parent = load(path, expected); out = path.parent
    invocation = out/'invocations'/uuid.uuid4().hex[:12]; invocation.mkdir(parents=True, exist_ok=False)
    report = {**shared.metadata(), 'status': 'running', 'request_hash': expected, 'jobs': []}
    with run_lock(out):
        try:
            pending = []
            for j in q['jobs']:
                directory = out/'episodes'/j['id']
                if directory.exists():
                    if not resume:
                        raise ValueError('Existing result: only --resume may reuse verified COMPLETE episodes')
                    verify_result(directory, q, j); report['jobs'].append({'id': j['id'], 'action': 'reuse'})
                else:
                    pending.append(j)
            def child(j):
                command = [sys.executable, '-B', str(Path(__file__).resolve()), 'worker', '--request', str(path),
                           '--confirm-request-hash', expected, '--job', j['id']]
                print('[p7i] '+j['id'], flush=True)
                with (invocation/(j['id']+'.log')).open('w', encoding='utf-8') as log:
                    subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
                verify_result(out/'episodes'/j['id'], q, j)
                return {'id': j['id'], 'action': 'run'}
            width = q['protocol']['workers']
            with ThreadPoolExecutor(max_workers=width) as pool:
                for i in range(0, len(pending), width):
                    report['jobs'].extend(f.result() for f in [pool.submit(child, j) for j in pending[i:i+width]])
            result = collect(path, q, parent)
            if (out/'aggregate.json').exists():
                saved = read(out/'aggregate.json')
                if any(saved.get(k) != v for k, v in result.items()) or saved['request_hash'] != expected or read(out/'aggregate_receipt.json') != {'request_hash': expected, 'sha256': sha(out/'aggregate.json')}:
                    raise ValueError('Prior aggregate changed; do not overwrite')
            else:
                write_once(out/'aggregate.json', {**shared.metadata(), **result, 'request_hash': expected,
                    'training_audit_sha256': q['training_audit_sha256'], 'finished_at': shared.now()})
                write_once(out/'aggregate_receipt.json', {'request_hash': expected, 'sha256': sha(out/'aggregate.json')})
            report.update(status='complete', finished_at=shared.now())
        except BaseException:
            report.update(status='failed', failure_reason=traceback.format_exc(), finished_at=shared.now()); raise
        finally:
            write_once(invocation/'report.json', report)
    print('stability_aggregate='+str(out/'aggregate.json'), flush=True)
    return out/'aggregate.json'


def main():
    p = argparse.ArgumentParser(description=__doc__); sub = p.add_subparsers(dest='command', required=True)
    c = sub.add_parser('prepare'); c.add_argument('--config', default='configs/development/p07i_ddpg_stability_v1.json'); c.add_argument('--run-id', required=True)
    for name in ('run', 'worker'):
        c = sub.add_parser(name); c.add_argument('--request', required=True); c.add_argument('--confirm-request-hash', required=True)
        if name == 'run': c.add_argument('--resume', action='store_true')
        else: c.add_argument('--job', required=True)
    c = sub.add_parser('audit'); c.add_argument('--run-id', required=True)
    args = p.parse_args()
    if args.command == 'prepare': prepare(read(inside(ROOT, args.config)), args.run_id)
    elif args.command == 'worker': worker(args)
    elif args.command == 'run': execute(args.request, args.confirm_request_hash, args.resume)
    else:
        path, h = prepare(AUDIT, args.run_id); execute(path, h)


if __name__ == '__main__': main()
