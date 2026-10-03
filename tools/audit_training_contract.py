"""P7j bounded training-target and reset/coverage audit; NEVER starts 60k training."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'tools')]
import train_response_predictors as shared
import explore_feature_ddpg as original
import diagnose_feature_ddpg as recording
import smoke_feature_ddpg as smoke
from prediction_rl.data.collection_store import read_json as read, write_once, file_hash as sha, seal_episode, verify_episode, run_lock
from prediction_rl.data.dataset_contract import digest
from prediction_rl.envs.upstream import upstream_session
from prediction_rl.evaluation.closed_loop_diagnostics import PROTOCOL as P7B
from prediction_rl.evaluation.terminal_events import StepObserver
from prediction_rl.evaluation.training_contract import VERSION, ARMS, TransitionWitness, install_witness, coverage, verify_reward, state_measurement
from prediction_rl.training.all_ddpg import preset_factory, checkpoint_contract, load_evaluation_weights, experiment_type

CONFIG = {'version': VERSION, 'purpose': 'bounded_pre_60k_semantics_and_distribution_audit',
    'training_arms': list(ARMS), 'training_seed': 2, 'requested_frames': 64, 'episode_cap': 2,
    'smoke_overrides': {'replay_start_size': 8, 'minibatch_size': 8},
    'frozen_arms': ['zero', 'conditional'], 'frozen_training_seed': 2,
    'continuing_simulator_seed': 200, 'continuing_episodes': 3,
    'initial_census_seeds': list(range(200, 220)), 'workers': 2, 'torch_threads': 1,
    'max_training_transitions_per_job': 563, 'max_frozen_control_steps': 3000,
    'execution_contract': 'simulation_blocking_exact_v1', 'training_60k_authorized': False,
    'formal_claim': False, 'test_opened': False}
JOBS = ([{'id': 'train_'+a, 'kind': 'train', 'arm': a} for a in ARMS] +
        [{'id': 'continuing_'+a, 'kind': 'continuing', 'arm': a} for a in CONFIG['frozen_arms']] +
        [{'id': 'initial_census', 'kind': 'census', 'arm': 'conditional'}])
CONFIG_PATH = ROOT/'configs/development/p07j_training_contract_v1.json'


class BoundedObserver(StepObserver):
    def __init__(self, control, traci, job):
        super().__init__(control, traci, job)
        self.cap = 1900  # <= 3*(100 warmup+1 insertion+500 controls+1 cleanup)


def parent(): return recording.load_parent(P7B)


def binding(q, job): return {'request_hash': digest(q), 'stage': VERSION, 'job': job}


def legacy_audit(pp, p):
    results = []
    for j in p['train_jobs']:
        d = pp.parent/'train'/j['id']; receipt = verify_episode(d, original.binding(p, 'train', j))['metadata']
        resolved = read(d/'resolved_config.json'); episodes = read(d/'episodes.json')
        if resolved['job'] != j or resolved['preset'] != p['preset'] or resolved['predictor_sha256'] != receipt['weights_contract']['predictor_sha256']:
            raise ValueError('Historical training configuration mismatch')
        results.append({'job': j, 'actual_steps': receipt['actual_steps'], 'completed_episodes': len(episodes),
            'checkpoint_sha256': receipt['weights_sha256'], 'reset_contract': 'upstream_continuing_traffic_v1',
            'native_arrival_rate': sum(r['arrival'] for r in episodes)/len(episodes),
            'native_event_labels_independently_verified': False,
            'historical_replay_states_saved': False, 'historical_minibatch_coverage_recoverable': False})
    return {'policies': results, 'historical_training_return_is_not_validation': True,
        'missing_old_replay_is_not_filled_with_new_smoke': True,
        'evaluation_protocol': 'gym20_fresh_process_first_episode',
        'old_checkpoint_purpose': 'evaluation_only_not_training_resume'}


def raw_outcome(observer, start, rows, creation):
    events = observer.events[start:]
    collision = [e for e in events if 'ego' in e['colliding_vehicle_ids'] or
                 any('ego' in (c['collider'], c['victim']) for c in e['collisions'])]
    disappearance = [e for e in events if e['before']['ego'] is not None and e['after']['ego'] is None]
    if collision: label = 'ego_collision'
    elif disappearance:
        event = disappearance[0]; ego = event['before']['ego']
        endpoint = (ego['lane_id'] == 'highwayahead_0' and ego['route_index'] == 1 and
                    ego['lane_position_m'] >= creation['arrival_position_m']-7.)
        label = 'natural_arrival' if 'ego' in event['arrived_vehicle_ids'] and endpoint else 'unexplained_disappearance'
    elif rows[-1]['reason'] == 'upstream_time_limit' and any(
        r['before']['simulation_time_s'] >= events[0]['before']['simulation_time_s'] and
        r['before']['ego'] is not None and r['after']['ego'] is None for r in observer.removals): label = 'time_limit'
    else: label = 'unexplained_terminal'
    native = {'upstream_collision': 'ego_collision', 'upstream_arrival': 'natural_arrival',
              'upstream_time_limit': 'time_limit', 'registered_time_limit': 'time_limit'}[rows[-1]['reason']]
    if label != native: raise ValueError('Training native reward/terminal event disagrees with independent raw events')
    if rows[-1]['reason'] == 'upstream_time_limit':
        before = rows[-1]['execution_audit']['before']['vehicles']['ego']
        if len(events) < 2 or events[-2]['after']['ego'] is None:
            raise ValueError('Missing actual deadline pre-cleanup frame')
        rows[-1]['reward_jerk_before_cleanup'] = (events[-2]['after']['ego']['acceleration']-before['acceleration'])/.2
    return {'classification': label, 'native_event_agrees': True, 'raw_steps': len(events)}


def train_worker(p, job, out):
    provider, _ = original.provider_for(p, job['arm'])
    predictor_before = None if provider is None else smoke.tensor_hash(provider.model.state_dict())
    witness = TransitionWitness(); episode_rows = []; boundaries = []; observer = None
    def record(event, obs, reward, done, info):
        witness.record(event, obs, reward, done, info)
        if event == 'reset': boundaries.append(len(observer.events))
        elif done:
            # Store is called immediately AFTER this environment recorder.
            episode_rows.append((len(witness.rows), len(observer.events)))
    with upstream_session(original.SOURCE, original.UPSTREAM, CONFIG['training_seed']) as settings:
        import control, traci
        identity = {'id': job['id'], 'condition_id': 'gym20'}
        with BoundedObserver(control, traci, identity) as observer:
            env = original.environment(p, job, provider, record); experiment = None
            try:
                experiment = experiment_type(out/'logs')(preset_factory('cpu', settings.LEARNING_RATE, smoke=True), env, quiet=True)
                wrapper = install_witness(experiment._agent, witness)
                experiment.train(frames=CONFIG['requested_frames'], episodes=CONFIG['episode_cap'])
                alignment = witness.finish()
                if alignment['transitions'] != experiment.frame-1 or alignment['transitions'] > 563:
                    raise ValueError('Original episode-boundary budget changed')
                if len(wrapper.losses) != len(wrapper.samples) or not wrapper.losses:
                    raise ValueError('No actual critic updates audited')
                start = 0; outcomes = []
                for e, (last, boundary) in enumerate(episode_rows):
                    rows = witness.rows[start:last+1]; saved = observer.events
                    observer.events = saved[:boundary]
                    try: outcomes.append(raw_outcome(observer, boundaries[e], rows, observer.creations[e]))
                    finally: observer.events = saved
                    start = last+1
                reward_check = verify_reward(witness.rows, settings)
                if provider is not None and predictor_before != smoke.tensor_hash(provider.model.state_dict()):
                    raise ValueError('Predictor changed')
                write_once(out/'transitions.json', witness.rows); write_once(out/'raw_events.json', observer.export())
                write_once(out/'states.json', witness.states)
                return {'alignment': alignment, 'reward': reward_check, 'raw_outcomes': outcomes,
                    'optimizer_updates': len(wrapper.losses), 'minibatches': wrapper.samples,
                    'reset_states': witness.resets, 'state_coverage': coverage(witness.states),
                    'actual_steps': alignment['transitions'], 'preset': experiment._agent.preset_parameters,
                    'full_5000_replay_warmup_used': False, 'tiny_training_is_not_learning_curve': True}
            finally:
                env.close()
                if experiment is not None: experiment._writer.close()


def continuing_worker(pp, p, job, out):
    provider, predictor_sha = original.provider_for(p, job['arm'])
    predictor_before = None if provider is None else smoke.tensor_hash(provider.model.state_dict())
    training_id = job['arm']+'_s2'; tj = next(j for j in p['train_jobs'] if j['id'] == training_id)
    td = pp.parent/'train'/training_id; m = verify_episode(td, original.binding(p, 'train', tj))['metadata']
    episodes = []; states = []; current = None
    def record(event, obs, reward, done, info):
        nonlocal current
        measurement = {**state_measurement(obs, info), 'event':event, 'episode':len(episodes)}; states.append(measurement)
        if event == 'reset': current = {'initial': measurement, 'rows': []}
        else:
            a = info['execution_audit']
            current['rows'].append({'requested_jerk': a['requested_jerk'], 'reward': float(reward), 'done': bool(done),
                'reason': a['termination_reason'], 'execution_audit': a})
    with upstream_session(original.SOURCE, original.UPSTREAM, CONFIG['continuing_simulator_seed']) as settings:
        import control, traci
        with BoundedObserver(control, traci, {'id': job['id'], 'condition_id': 'gym20'}) as observer:
            env = original.environment(p, job, provider, record)
            try:
                from all.logging import DummyWriter
                checked = preset_factory('cpu', settings.LEARNING_RATE)(env, DummyWriter())
                c = checkpoint_contract(env, p['preset'], predictor_sha, p['upstream_config_sha256'])
                load_evaluation_weights(checked, td/'final.pt', c, m['weights_sha256'])
                before = {n: smoke.tensor_hash(a.model.state_dict()) for n, a in [('policy', checked.agent.policy), ('q', checked.agent.q)]}
                for index in range(CONFIG['continuing_episodes']):
                    state = env.reset(); reward = 0.; start = len(observer.events); begin = len(states)
                    for _ in range(500):
                        action = checked.eval(state, reward)
                        state, reward = env.step(action)
                        if env.done:
                            checked.eval(state, reward); break
                    else: raise ValueError('Gym20 control cap violated')
                    outcome = raw_outcome(observer, start, current['rows'], observer.creations[index])
                    current.update(index=index, raw_outcome=outcome, state_coverage=coverage(states[begin-1:]),
                                   return_value=sum(r['reward'] for r in current['rows']))
                    verify_reward(current['rows'], settings); episodes.append(current)
                # First episode must retain exact old Gym20 behavior; later ones are
                # diagnostic, never substitute them for the approved evaluation.
                old = read(pp.parent/'evaluate'/f'{training_id}_v200'/'actions.json')
                new = episodes[0]['rows']
                if len(new) != len(old) or any(a['requested_jerk'] != b['jerk'] or a['reward'] != b['reward'] or a['done'] != b['done'] for a,b in zip(new, old)):
                    raise ValueError('Fresh-process deterministic replay changed')
                if (len(checked.agent.replay_buffer) or any(smoke.tensor_hash(a.model.state_dict()) != before[n] or a._optimizer.state
                    for n,a in [('policy',checked.agent.policy),('q',checked.agent.q)])):
                    raise ValueError('Frozen policy updated')
                if provider is not None and predictor_before != smoke.tensor_hash(provider.model.state_dict()): raise ValueError('Predictor changed')
                write_once(out/'episodes.json', episodes); write_once(out/'raw_events.json', observer.export()); write_once(out/'states.json', states)
                return {'episodes': [{k:v for k,v in e.items() if k != 'rows'} for e in episodes],
                    'first_episode_exact_parity': True, 'policy_sha256': m['weights_sha256'], 'no_policy_updates': True,
                    'only_one_continuing_traffic_stream_not_three_independent_scenarios': True}
            finally: env.close()


def census_worker(p, job, out):
    provider, _ = original.provider_for(p, 'conditional'); values = []
    for seed in CONFIG['initial_census_seeds']:
        capture = []
        with upstream_session(original.SOURCE, original.UPSTREAM, seed):
            env = original.environment(p, job, provider, lambda event,obs,reward,done,info: capture.append(state_measurement(obs,info)))
            try: env.reset()
            finally: env.close()
        if len(capture) != 1 or capture[0]['history_frames'] != 1: raise ValueError('Fresh-reset history changed')
        original_row = p['evaluation_jobs']
        j = next(j for j in original_row if j['arm'] == 'conditional' and j['run_seed'] == 2 and j['simulator_seed'] == seed)
        # P7's immutable initial traffic hash is arm/optimizer independent.
        old_row = read((ROOT/P7B['parent_request']).parent/'evaluate'/j['id']/'episode.json')
        if capture[0]['traffic_sha256'] != old_row['initial_traffic_sha256']: raise ValueError('Initial traffic differs from original Gym20')
        values.append({'simulator_seed': seed, **capture[0]})
    write_once(out/'initial_states.json', values)
    return {'initial_states': values, 'coverage': coverage(values), 'all_20_initial_traffic_hashes_match': True,
            'policy_rollouts': 0, 'census_does_not_test_all_20_outcomes': True}


def worker(args):
    shared.clean(); path = Path(args.request).resolve(); q = read(path)
    if path.parent.parent != ROOT/'artifacts'/'p7j': raise ValueError('Request outside P7j outputs')
    if (read(CONFIG_PATH) != CONFIG or q['config_sha256'] != sha(CONFIG_PATH) or q['config'] != CONFIG or q['jobs'] != JOBS or q['source_hashes'] != shared.source_hashes() or
        q['runtime'] != original.runtime_identity() or q['dependencies'] != smoke.dependencies() or
        q['request_inputs'] != read(path.parent/'preparation.json')['request_inputs'] or
        digest(q) != read(path.parent/'preparation.json')['request_hash']): raise ValueError('Request/source/runtime changed')
    pp,p = parent(); job = next(j for j in JOBS if j['id'] == args.job)
    if digest(p) != q['request_inputs']['parent_request_hash']: raise ValueError('Parent changed')
    out = path.parent/job['id']; out.mkdir(exist_ok=False); started = shared.metadata(); write_once(out/'started.json', started)
    torch.set_num_threads(1); start = time.perf_counter()
    try:
        result = train_worker(p,job,out) if job['kind']=='train' else continuing_worker(pp,p,job,out) if job['kind']=='continuing' else census_worker(p,job,out)
        result.update(status='complete', job=job, provenance=started, elapsed_s=time.perf_counter()-start)
        write_once(out/'report.json', result); seal_episode(out,binding(q,job),result)
    except BaseException:
        write_once(out/'failure.json',{'status':'failed','failure_reason':traceback.format_exc()}); raise


def audit(run_id):
    shared.clean()
    if read(CONFIG_PATH) != CONFIG: raise ValueError('Changed frozen bounded protocol')
    pp,p = parent(); out = shared.output_dir('p7j', run_id)
    inputs = {'parent_request_hash': digest(p), 'parent_aggregate_sha256': sha(pp.parent/'aggregate.json')}
    q = {**shared.metadata(), 'config': deepcopy(CONFIG), 'config_sha256':sha(CONFIG_PATH), 'source_hashes':shared.source_hashes(),
         'dependencies':smoke.dependencies(), 'runtime':original.runtime_identity(), 'request_inputs': inputs, 'jobs': JOBS}
    write_once(out/'request.json',q); write_once(out/'preparation.json',{'request_hash':digest(q),'request_inputs':inputs})
    write_once(out/'historical_limits.json',legacy_audit(pp,p))
    children = []
    with run_lock(out):
        def child(j):
            command = [sys.executable,'-B',str(Path(__file__).resolve()),'worker','--request',str(out/'request.json'),'--job',j['id']]
            print('[p7j] '+j['id'],flush=True)
            with (out/(j['id']+'.log')).open('w',encoding='utf-8') as log:
                subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=600)
            return verify_episode(out/j['id'],binding(q,j))['metadata']
        try:
            with ThreadPoolExecutor(max_workers=2) as pool:
                for first in range(0,len(JOBS),2): children.extend(f.result() for f in [pool.submit(child,j) for j in JOBS[first:first+2]])
            result = {**shared.metadata(),'status':'complete','request_hash':digest(q),'jobs':children,
                'training_transition_semantics_gate':True, 'initial_gym20_parity_gate':True,
                'historical_20k_replay_coverage_gate':None, 'full_distribution_equivalence_claim':False,
                'training_60k_started':False, 'training_60k_authorized':False, 'formal_claim':False,
                'test_opened':False, 'finished_at':shared.now()}
            write_once(out/'aggregate.json',result); write_once(out/'aggregate_receipt.json',{'request_hash':digest(q),'aggregate_sha256':sha(out/'aggregate.json')})
        except BaseException:
            write_once(out/'failure.json',{'status':'failed','failure_reason':traceback.format_exc()}); raise
    print('training_contract_aggregate='+str(out/'aggregate.json'),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__); sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('audit'); p.add_argument('--run-id',required=True)
    p=sub.add_parser('worker'); p.add_argument('--request',required=True); p.add_argument('--job',required=True)
    a=parser.parse_args(); audit(a.run_id) if a.command=='audit' else worker(a)


if __name__=='__main__': main()
