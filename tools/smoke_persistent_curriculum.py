"""IC03b: eight native episodes, original learner warmup, zero optimization."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import random
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'tools')]
import numpy as np
import torch

import smoke_curriculum_entry as entry
from prediction_rl.curriculum.entry_smoke import json_safe, rng_fingerprint
from prediction_rl.curriculum.integration_smoke import (SCOPE, PROTOCOL, NativePacketAudit,
    SmokeEpisodeSampler, jobs, task_roster, validate_protocol)
from prediction_rl.curriculum.train_reader import inside
from prediction_rl.data.collection_store import file_hash as sha, run_lock, seal_episode, verify_episode, write_once
from prediction_rl.data.dataset_contract import digest
from prediction_rl.data.merge_geometry import MergeGeometry
from prediction_rl.envs.curriculum_stream import CurriculumTaskStream
from prediction_rl.envs.fresh_task import FreshGym20TaskEnv
from prediction_rl.envs.prediction_features import PredictionFeatureEnv
from prediction_rl.training.all_ddpg import make_all_environment, preset_factory, experiment_type, resolved_preset
from prediction_rl.training.curriculum_replay import PersistentTaskReplayAudit
from prediction_rl.training.matched_learning_curve import Jsonl

CONFIG = 'configs/development/ic03b_persistent_smoke_v1.json'
AUTH = 'provenance/ic03b_smoke_authorization.json'
SEALED = {
    'request.json':'da4c0f0b6b75ee45cc1314233494975dbb6b6f7e13e9e2163edc88e970a17924',
    'report.json':'1a8694de0f3fb3f76687e08c0ce56289c809eff4d9ef4645666fb4eff7bafffb',
    'task_proposals.json':'a535496ad79220ca679f7cf1661403f2e16fdf1e4151a85aa575d8b151b68e17',
    'postflight.json':'efda7a248c43b7a0fd75142d25220517300c81f36b52f8c45ddfec898f92d5ab'}
SPACE_TRACE = 'artifacts/ic/ic02b_v1/d11000r0/native_trace.json'
SPACE_SHA = '84201764a7b24c9e130e2e04be2ec6d4dfc5dda24887123b09d8a85a980215b2'
ADDED_SOURCES = {'worktree:src/prediction_rl/envs/curriculum_stream.py',
    'worktree:src/prediction_rl/training/curriculum_replay.py',
    'worktree:src/prediction_rl/curriculum/integration_smoke.py'}


def now(): return datetime.now(timezone.utc).isoformat()


def prerequisites(baseline):
    protocol = validate_protocol(entry.read(ROOT/CONFIG))
    hashes, predictors, training_request = entry.inputs(baseline, entry.PROTOCOL)
    sealed = ROOT/'artifacts/ic/ic02h_intensity_v1'
    if {name:sha(sealed/name) for name in SEALED} != SEALED:
        raise ValueError('Sealed IC02h bytes changed')
    q, report, pack, post = [entry.read(sealed/name) for name in SEALED]
    if (q['git_commit_sha'] != '5b6a2d17fd9117303ac4a357473570dcdbdd8618'
            or report['request_hash'] != digest(q) or pack['request_hash'] != digest(q)
            or report['status'] != 'complete' or report['engineering_gate'] is not True
            or report['summary']['coverage_gate'] is not True or report['training_ready'] is not False
            or report['classification_applied'] is not False or report['parent_postflight'] != post
            or not all(post[k] for k in ('parent_rng_unchanged','immutable_input_postflight_passed'))):
        raise ValueError('Need the complete unmodified IC02h proposal, not a training release')
    # Validate EVERY sealed input. Only the three explicitly reviewed IC03/03b
    # modules may be new. Old gates/tools/configuration are not monkey-patched.
    for key, expected in q['input_hashes'].items():
        location, relative = key.split(':', 1)
        base = ROOT if location == 'worktree' else Path(baseline) if location == 'baseline' else None
        if base is None or sha(inside(base, relative)) != expected:
            raise ValueError('Changed sealed scientific input: '+key)
        hashes[key] = expected
    current_sources = {'worktree:'+p.relative_to(ROOT).as_posix()
                       for p in (ROOT/'src/prediction_rl').rglob('*.py')}
    previous_sources = {k for k in q['input_hashes'] if k.startswith('worktree:src/prediction_rl/')}
    if current_sources-previous_sources != ADDED_SOURCES or previous_sources-current_sources:
        raise ValueError('Unreviewed project source added or removed')
    for name, value in SEALED.items(): hashes['worktree:artifacts/ic/ic02h_intensity_v1/'+name] = value
    if sha(ROOT/SPACE_TRACE) != SPACE_SHA:
        raise ValueError('Original transported space evidence changed')
    spaces = entry.read(ROOT/SPACE_TRACE)['spaces']
    if spaces != {'action':{'dtype':'float32','shape':[1],'low':[-5.],'high':[5.]},
                  'observation':{'dtype':'float32','shape':[20],'low':[-1.]*20,'high':[1.]*20}}:
        raise ValueError('Original raw Box space changed')
    auth = entry.read(ROOT/AUTH)
    expected = {'protocol_id':SCOPE, 'run_id':PROTOCOL['run_id'], 'maximum_sumo_sessions':8,
        'maximum_control_steps':4000, 'maximum_raw_ticks':4816, 'maximum_feature_calls':4008,
        'episodes_per_arm':2, 'optimizer_updates_authorized':False, 'formal_training_authorized':False,
        'predictor_training_authorized':False, 'new_counterfactual_branches':0,
        'automatic_retry_authorized':False, 'old_baseline_access':'read_only',
        'merge_main_authorized':False, 'push_authorized':False}
    if any(auth.get(k) != v for k,v in expected.items()):
        raise ValueError('Human smoke scope differs from the frozen maximum budget')
    for name in (CONFIG, AUTH, SPACE_TRACE, 'tools/smoke_persistent_curriculum.py'):
        hashes['worktree:'+name] = sha(ROOT/name)
    tasks = task_roster(pack)
    return protocol, hashes, predictors, training_request, tasks, spaces


def prepare(baseline):
    entry.require_clean()
    p, h, predictors, trained, tasks, spaces = prerequisites(baseline)
    q = {'version':SCOPE, 'protocol':p, 'baseline_root':str(Path(baseline).resolve()),
        'git_commit_sha':entry.git(ROOT, 'rev-parse', 'HEAD'), 'working_tree_dirty':False,
        'prepared_at':now(), 'launch_command':subprocess.list2cmdline(sys.orig_argv),
        'input_hashes':h, 'runtime':entry.runtime(), 'predictors':predictors,
        'predictor_training_request':trained, 'smoke_task_roster':tasks, 'native_spaces':spaces,
        'jobs':jobs(tasks), 'formal_training_authorized':False, 'human_approved_scope_only':True,
        'resolved_ddpg_preset':resolved_preset('cpu', .0002, smoke=False)}
    out = ROOT/'artifacts/ic'/p['run_id']
    out.mkdir(parents=True, exist_ok=False)
    write_once(out/'request.json', q)
    write_once(out/'preparation.json', {'request_hash':digest(q), 'status':'prepared_not_executed'})
    print('smoke_request='+str(out/'request.json'), flush=True)
    print('confirm_request_hash='+digest(q), flush=True)


def load(path, expected):
    entry.require_clean()
    path = Path(path).resolve()
    if path != ROOT/'artifacts/ic'/PROTOCOL['run_id']/'request.json':
        raise ValueError('One fixed approved smoke only; renaming cannot reset the budget')
    q = entry.read(path)
    if (q['version'] != SCOPE or digest(q) != expected
            or entry.read(path.parent/'preparation.json')['request_hash'] != expected
            or q['git_commit_sha'] != entry.git(ROOT, 'rev-parse', 'HEAD')
            or q['working_tree_dirty'] is not False or q['formal_training_authorized'] is not False):
        raise ValueError('Clean source, approval or request hash changed')
    p, h, predictors, trained, tasks, spaces = prerequisites(q['baseline_root'])
    if (q['protocol'] != p or q['input_hashes'] != h or q['runtime'] != entry.runtime()
            or q['predictors'] != predictors or q['predictor_training_request'] != trained
            or q['smoke_task_roster'] != tasks or q['native_spaces'] != spaces or q['jobs'] != jobs(tasks)
            or q['resolved_ddpg_preset'] != resolved_preset('cpu', .0002, smoke=False)):
        raise ValueError('Frozen source/runtime/roster/preset changed; no automatic adaptation')
    return path, q


class RngNeutralWorker(FreshGym20TaskEnv):
    def reset(self):
        before = rng_fingerprint(); result = super().reset()
        if before != rng_fingerprint(): raise ValueError('Native reset consumed persistent learner RNG')
        return result
    def step(self, action):
        before = rng_fingerprint(); result = super().step(action)
        if before != rng_fingerprint(): raise ValueError('Native step consumed persistent learner RNG')
        return result


class RngNeutralFeatures:
    def __init__(self, provider): self.provider, self.model, self.calls = provider, provider.model, 0
    def __call__(self, history):
        before = rng_fingerprint(); result = self.provider(history)
        if before != rng_fingerprint(): raise ValueError('Frozen predictor consumed learner RNG')
        self.calls += 1
        return result


def learner_hashes(checked):
    result = {}
    for name in ('policy','q'):
        approximation = getattr(checked.agent, name)
        for suffix, model in (('model',approximation.model), ('target',approximation._target._target)):
            value = hashlib.sha256()
            for key, tensor in sorted(model.state_dict().items()):
                value.update(key.encode()); value.update(str((tensor.dtype, tuple(tensor.shape))).encode())
                value.update(tensor.detach().cpu().numpy().tobytes())
            result[name+'_'+suffix] = value.hexdigest()
    return result


def worker(path, expected, job_id, *, request_loader=None, sampler_protocol=None):
    # Explicit dependency injection for the separately approved remaining-arm
    # coordinator; no global loader patch or relaxation of the original request.
    request_loader = load if request_loader is None else request_loader
    path, q = request_loader(path, expected)
    sampler_protocol = q['protocol'] if sampler_protocol is None else sampler_protocol
    validate_protocol(sampler_protocol)
    job = next(j for j in q['jobs'] if j['id'] == job_id)
    if (not (path.parent/'started.json').is_file() or (path.parent/'report.json').exists()
            or (path.parent/'failure.json').exists()):
        raise ValueError('Worker requires its unfinished approved parent')
    out = path.parent/job_id
    if {p.name for p in out.iterdir()} != {'started.json','launch.json','worker.log'}:
        raise ValueError('No retry of a started/incomplete learner')
    write_once(out/'worker_claim.json', {'request_hash':expected,'job':job,'started_at':now()})
    torch.set_num_threads(1)
    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    before = rng_fingerprint()
    chosen = {**q, 'predictors':{job['predictor_mode']:q['predictors'][job['predictor_mode']]}}
    frozen = entry.providers(chosen)[job['predictor_mode']]
    if before != rng_fingerprint(): raise ValueError('Frozen loading consumed learner RNG')
    predictor_before = entry.tensor_hash(frozen)
    provider = RngNeutralFeatures(frozen)
    geometry = MergeGeometry(ROOT/'RL-MPC-LaneMerging-master/merge.net.xml')
    episode_audits, transports = [], []
    streams, experiment, feature = [], None, None
    episode_rows = []
    try:
        native_log = Jsonl(out/'native.jsonl.gz'); streams.append(native_log)
        replay_log = Jsonl(out/'replay.jsonl.gz'); streams.append(replay_log)
        def factory(selection):
            index = selection['episode_index']
            if index >= 2 or len(transports) != index: raise ValueError('Exceeded unique task session budget')
            directory = out/f'e{index}'
            directory.mkdir(exist_ok=False)
            write_once(directory/'started.json', {'selection':selection,'request_hash':expected,'started_at':now()})
            witness = rng_fingerprint()
            base = RngNeutralWorker(ROOT/'RL-MPC-LaneMerging-master', ROOT/entry.PROTOCOL['upstream_config'],
                selection['simulator_seed'], directory/'environment.log', control_cap=500)
            if witness != rng_fingerprint(): raise ValueError('Task creation reseeded or consumed learner RNG')
            transports.append(base)
            episode_audits.append(NativePacketAudit(geometry, base.runtime['settings']))
            write_once(directory/'runtime.json', json_safe(base.runtime))
            return base
        from gym.spaces import Box
        spaces = {name:Box(np.asarray(spec['low'],np.float32),np.asarray(spec['high'],np.float32),dtype=np.float32)
                  for name,spec in q['native_spaces'].items()}
        sampler = SmokeEpisodeSampler(q['smoke_task_roster'],job,expected,sampler_protocol)
        task_stream = CurriculumTaskStream(sampler,factory,spaces['observation'],spaces['action'],purpose=SCOPE)
        def emit(kind, value):
            replay_log.append(json_safe({'kind':kind,'value':value}))
            if kind == 'episode':
                index = value['episode_index']
                row = {**value, **episode_audits[index].result()}
                episode_rows.append(row)
                write_once(out/f'e{index}'/'summary.json', json_safe(row))
        audit = PersistentTaskReplayAudit(task_stream,purpose=SCOPE,on_event=emit)
        feature = PredictionFeatureEnv(task_stream,job['predictor_mode'],provider,task_stream.current_frame)
        def record(event, obs, reward, done, info):
            packet = task_stream._worker.last_packet
            episode_audits[-1].accept(packet,obs,reward,done,info)
            native_log.append(json_safe({'episode_index':task_stream.selection['episode_index'],
                'packet':packet, 'feature_observation':obs, 'feature_info':info['prediction_features']}))
            audit.record(event,obs,reward,done,info)
        env = make_all_environment(feature,'cpu',record)
        experiment = experiment_type(out/'tensorboard')(preset_factory('cpu',.0002,smoke=False),env,quiet=True)
        audit.install(experiment._agent)
        initial = learner_hashes(experiment._agent)
        def forbidden_update(*args, **kwargs): raise ValueError('Optimizer update forbidden in IC03b')
        for name in ('q','policy'): getattr(experiment._agent.agent,name).reinforce = forbidden_update
        original_identity = (id(experiment._agent),id(experiment._agent.agent.q._optimizer),
                             id(experiment._agent.agent.policy._optimizer),id(audit.base))
        for _ in range(2):
            experiment._run_training_episode()  # Unchanged upstream ALL loop.
            if original_identity != (id(experiment._agent),id(experiment._agent.agent.q._optimizer),
                    id(experiment._agent.agent.policy._optimizer),id(audit.base)):
                raise ValueError('Task boundary replaced learner/optimizer/replay')
        snapshot = audit.snapshot()
        if (len(episode_rows) != 2 or audit.sample_calls or learner_hashes(experiment._agent) != initial
                or any(getattr(experiment._agent.agent,k)._optimizer.state for k in ('q','policy'))
                or entry.tensor_hash(frozen) != predictor_before
                or any(p.grad is not None or p.requires_grad for p in frozen.model.parameters())
                or snapshot['actual_control_steps'] != experiment.frame-1
                or provider.calls != sum(row['feature_calls'] for row in episode_rows)):
            raise ValueError('Persistent learner/frozen predictor/control clock postflight failed')
        feature.close()
        experiment._writer.close()
        for log in streams: log.close()
        streams.clear()
        if any(name in sys.modules for name in ('config','control','merge_gym','sumo','prediction','dqn','st','st_cy')):
            raise ValueError('Upstream module escaped into persistent learner process')
        request_loader(path, expected)
        row = {'job':job,'execution_request_version':q['version'],
            'episodes':episode_rows,'actual_control_steps':snapshot['actual_control_steps'],
            'raw_ticks':sum(r['raw_ticks'] for r in episode_rows),'feature_calls':provider.calls,
            'optimizer_updates':0,'original_minibatches':audit.sample_calls,
            'original_agent_calls':experiment._agent.agent._frames_seen,
            'learner_initial_and_final_tensor_hashes':initial, 'predictor_tensor_hash':predictor_before,
            'persistent_learner_verified':True,'environment_and_features_rng_neutral':True,
            'learner_rng_expected_to_change_with_original_exploration':True,
            'predictor_frozen':True,'source_postflight_passed':True,'replay_audit':snapshot,
            'formal_training_authorized':False,'finished_at':now()}
        write_once(out/'summary.json',json_safe(row))
        # The coordinator owns worker.log until this process exits. Even this
        # final message, stdout flushing and finally cleanup can change files,
        # so the child must not publish an immutable directory receipt here.
        print('smoke_arm_complete='+job_id+' steps='+str(row['actual_control_steps']),flush=True)
    except BaseException as error:
        write_once(out/'failure.json', {'request_hash':expected,'job':job,'finished_at':now(),
            'failure_type':type(error).__name__,'failure_reason':str(error),'traceback':traceback.format_exc(),
            'sessions_attempted':len(transports),'completed_episodes':json_safe(episode_rows)})
        raise
    finally:
        if feature is not None: feature.close()
        if experiment is not None: experiment._writer.close()
        for log in streams: log.close()


def execute(path, expected, *, request_loader=None, worker_script=None):
    request_loader = load if request_loader is None else request_loader
    worker_script = Path(__file__).resolve() if worker_script is None else Path(worker_script).resolve()
    path, q = request_loader(path, expected)
    out = path.parent
    if any((out/name).exists() for name in ('started.json','report.json','failure.json')):
        raise ValueError('No automatic replay of completed, failed or interrupted smoke')
    report = {'request_hash':expected,'git_commit_sha':q['git_commit_sha'],'working_tree_dirty':False,
        'started_at':now(),'launch_command':subprocess.list2cmdline(sys.orig_argv),
        'protocol':q['protocol'],'runtime':q['runtime'],'jobs':[], 'formal_training_authorized':False,
        'training_ready':False,'test_opened':False,'new_counterfactual_branches':0}
    if 'prior_evidence_review' in q:
        report['prior_evidence_review'] = deepcopy(q['prior_evidence_review'])
    before = rng_fingerprint()
    with run_lock(out):
        write_once(out/'started.json',report)
        try:
            for job in q['jobs']:
                directory = out/job['id']; directory.mkdir(exist_ok=False)
                write_once(directory/'started.json',{'request_hash':expected,'job':job,'started_at':now()})
                command = [sys.executable,'-B',str(worker_script),'worker','--request',str(path),
                           '--confirm-request-hash',expected,'--job',job['id']]
                write_once(directory/'launch.json',{'command':command})
                print('smoke_arm_start='+job['id'],flush=True)
                with (directory/'worker.log').open('x',encoding='utf-8') as log:
                    subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
                # Successful process exit includes its finally cleanup. Close
                # the parent's stdout handle too, then seal ALL final files;
                # never exclude worker.log or repair an existing bad receipt.
                row = entry.read(directory/'summary.json')
                binding = {'request_hash':expected,'job':job}
                seal_episode(directory,binding,row)
                row = verify_episode(directory,binding)['metadata']
                report['jobs'].append(row)
                print('smoke_arm_complete='+job['id']+' steps='+str(row['actual_control_steps']),flush=True)
            count = sum(len(r['episodes']) for r in report['jobs'])
            controls = sum(r['actual_control_steps'] for r in report['jobs'])
            ticks = sum(r['raw_ticks'] for r in report['jobs'])
            calls = sum(r['feature_calls'] for r in report['jobs'])
            p = q['protocol']
            if (count != len(q['jobs'])*p['episodes_per_arm'] or count > p['sumo_sessions_max']
                    or controls > p['control_steps_max'] or ticks > p['raw_ticks_max']
                    or calls > p['feature_calls_max'] or before != rng_fingerprint()):
                raise ValueError('Smoke budget exceeded or coordinator RNG changed')
            request_loader(path,expected)
            report.update(status='complete',engineering_gate=True,finished_at=now(),actual_episodes=count,
                actual_control_steps=controls,raw_ticks=ticks,feature_calls=calls,optimizer_updates=0,
                parent_rng_unchanged=True,source_postflight_passed=True,
                remaining_gate='freeze_IC04_formal_request_then_human_approval')
            write_once(out/'report.json',report)
            print('smoke_report='+str(out/'report.json'),flush=True)
            return report
        except BaseException as error:
            report.update(status='failed',engineering_gate=False,finished_at=now(),failure_type=type(error).__name__,
                          failure_reason=str(error),traceback=traceback.format_exc())
            write_once(out/'failure.json',report)
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command',required=True)
    p = commands.add_parser('prepare'); p.add_argument('--baseline-root',required=True)
    for name in ('run','worker'):
        p = commands.add_parser(name); p.add_argument('--request',required=True)
        p.add_argument('--confirm-request-hash',required=True)
        if name == 'worker': p.add_argument('--job',required=True)
    args = parser.parse_args()
    if args.command == 'prepare': prepare(args.baseline_root)
    elif args.command == 'run': execute(args.request,args.confirm_request_hash)
    else: worker(args.request,args.confirm_request_hash,args.job)


if __name__ == '__main__': main()
