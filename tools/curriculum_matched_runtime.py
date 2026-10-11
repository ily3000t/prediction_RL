"""Explicit IC04 execution adapter; unchanged IC03 classes are NOT reauthorized.

Only constructor authorization differs. Native lifecycle/store/sample methods
are inherited from the tested core; the original ALL learner loop is retained.
"""
from collections import Counter
from copy import deepcopy
import math

import numpy as np
import torch

import smoke_curriculum_entry as entry
from smoke_persistent_curriculum import RngNeutralWorker, RngNeutralFeatures, learner_hashes
from curriculum_matched_contract import SCOPE, TrainingEpisodeSampler
from prediction_rl.curriculum.entry_smoke import json_safe, rng_fingerprint
from prediction_rl.curriculum.integration_smoke import NativePacketAudit
from prediction_rl.data.collection_store import write_once
from prediction_rl.data.merge_geometry import MergeGeometry
from prediction_rl.envs.curriculum_stream import CurriculumTaskStream
from prediction_rl.envs.prediction_features import PredictionFeatureEnv
from prediction_rl.evaluation.exploration import EpisodeMetrics
from prediction_rl.evaluation.terminal_events import PROTOCOL as EVENT_PROTOCOL, classify
from prediction_rl.evaluation.training_contract import TransitionWitness, state_measurement
from prediction_rl.training.all_ddpg import (make_all_environment, preset_factory, experiment_type,
    resolved_preset, checkpoint_contract, save_evaluation_weights, load_evaluation_weights)
from prediction_rl.training.curriculum_replay import PersistentTaskReplayAudit
from prediction_rl.training.matched_learning_curve import Jsonl, Coverage, train_nodes


class MatchedTaskStream(CurriculumTaskStream):
    """New typed scope, same terminal-store-before-reset transport lifecycle."""
    def __init__(self,sampler,factory,observation_space,action_space):
        if type(sampler) is not TrainingEpisodeSampler or not callable(factory):
            raise ValueError('Only the confirmed IC04 training sampler is permitted')
        if (observation_space.shape!=(20,) or observation_space.dtype!=np.float32
                or action_space.shape!=(1,) or action_space.dtype!=np.float32
                or not np.array_equal(action_space.low,[-5.]) or not np.array_equal(action_space.high,[5.])):
            raise ValueError('Require unchanged native spaces')
        # Same fields as the tested core; no offline-purpose rename or old gate patch.
        self.purpose=SCOPE
        self.observation_space,self.action_space=deepcopy(observation_space),deepcopy(action_space)
        self._sampler,self._factory=sampler,factory
        self._worker=self._owner=self._selection=None
        self._terminal=self._acknowledged=self._faulted=self._closed=False
        self.actual_control_steps=self.episode_control_steps=0
        self.last_completion=None


class MatchedReplayAudit(PersistentTaskReplayAudit):
    def __init__(self,stream,*,on_event=None,recent_window=1024):
        if type(stream) is not MatchedTaskStream or stream.purpose!=SCOPE or (on_event is not None and not callable(on_event)):
            raise ValueError('Require the IC04 typed task stream')
        self.purpose=SCOPE
        self.stream,self.on_event,self.recent_window=stream,on_event,recent_window
        self.witness=TransitionWitness()
        self.inner=self.base=self.audit=None
        self._by_state={}
        self.insertions=self.sample_calls=0
        self._faulted=self._detached=False
        stream.bind_replay_owner(self)

    def install(self,checked):
        if checked.preset_parameters!=resolved_preset('cpu',.0002,smoke=False):
            raise ValueError('Matched training must retain original warmup, batch, uniform replay and optimizers')
        return super().install(checked)

    def snapshot(self):
        return {**super().snapshot(),'matched_development_training_authorized':True,
            'formal_claim':False,'training_resume_supported':False}


class EpisodeRecorder:
    """Observe raw/native task outcomes and current-state exposure, not a teacher."""
    def __init__(self,geometry,settings,identity,native_log):
        self.native=NativePacketAudit(geometry,settings)
        self.geometry,self.identity,self.log=geometry,identity,native_log
        self.metrics=EpisodeMetrics(); self.coverage=Coverage()
        self.decisions=[]; self.actions=[]
        self.sidecar={'version':'raw_terminal_events_v1','job':identity,
            'capture_boundary':EVENT_PROTOCOL['capture_boundary'],
            'arrival_position_source':EVENT_PROTOCOL['arrival_position_source'],
            'creations':[],'explicit_removals':[],'steps':[]}
        self.previous_frame=None
        self.minimum_gaps={'front':None,'rear':None}

    def record(self,event,packet,obs,reward,done,info):
        self.native.accept(packet,obs,reward,done,info)
        self.metrics.record(event,obs,reward,done,info)
        for key in ('creations','explicit_removals','steps'):
            self.sidecar[key].extend(deepcopy(packet['raw_events'][key]))
        if event=='step':
            self.decisions.append({'frame':info['execution_audit']['before']})
            self.actions.append(float(info['execution_audit']['requested_jerk']))
            self.gaps(self.previous_frame)  # Current decision state, no true future features.
        if not done:
            self.coverage.add([Coverage.vector(state_measurement(obs,info))])
        self.previous_frame=deepcopy(packet['current_frame'])
        self.log.append(json_safe({'identity':self.identity,'packet':packet,
            'feature_observation':obs,'feature_info':info['prediction_features']}))

    def gaps(self,frame):
        ego=frame['vehicles']['ego']
        for actor,other in frame['vehicles'].items():
            if actor=='ego': continue
            pair=self.geometry.pair(ego,other)
            # Across-path projected gaps are not an instantaneous safety bound.
            if pair['same_path_gap_m'] is None: continue
            key='front' if pair['progress_delta_m']>=0 else 'rear'
            value=pair['same_path_gap_m']
            old=self.minimum_gaps[key]
            self.minimum_gaps[key]=value if old is None else min(old,value)

    def finish(self):
        native=self.metrics.result()
        raw=classify(self.sidecar,{**self.identity,**native,'reported_collision':native['collision']},self.decisions)
        if raw['classification'] not in ('natural_arrival','ego_collision','time_limit') or raw['native_event_agrees'] is not True:
            raise ValueError('Raw/native terminal disagreement invalidates the run')
        cov=self.coverage.result('decision_states_NOT_future_labels')
        return {**native,**self.native.result(),'natural_arrival':int(raw['classification']=='natural_arrival'),
            'ego_collision':int(raw['classification']=='ego_collision'),'native_event_agrees':True,
            'raw_outcome':raw,'low_speed_fraction':cov['moments']['low_speed']['mean'],
            'action_saturation_fraction':sum(abs(x)>=4.99 for x in self.actions)/len(self.actions),
            'state_coverage':cov,'minimum_same_path_front_gap_m':self.minimum_gaps['front'],
            'minimum_same_path_rear_gap_m':self.minimum_gaps['rear'],
            'gap_definition':'minimum_observed_decision_state_same_path_bumper_gap_not_cross_path_risk',
            'legacy_distance_definition':'minimum_observed_front_bumper_distance_m_is_Euclidean_reference_point_distance'}


def identity(job,simulator_seed):
    return {'id':job['id'],'model_id':'ic04_'+job['arm'],'condition_id':'gym20',
        'run_seed':job['run_seed'],'simulator_seed':simulator_seed,
        'training_id':job.get('training_id',job['id'])}


def frozen_provider(q,job):
    before=rng_fingerprint()
    provider=entry.providers({**q,'predictors':{job['predictor_mode']:q['predictors'][job['predictor_mode']]}})[job['predictor_mode']]
    if before!=rng_fingerprint(): raise ValueError('Frozen provider loading consumed learner RNG')
    return provider,RngNeutralFeatures(provider),entry.tensor_hash(provider)


def assert_frozen(provider,expected):
    if entry.tensor_hash(provider)!=expected or any(p.grad is not None or p.requires_grad for p in provider.model.parameters()):
        raise ValueError('Frozen predictor changed or became trainable')


def spaces(q):
    from gym.spaces import Box
    return {name:Box(np.asarray(s['low'],np.float32),np.asarray(s['high'],np.float32),dtype=np.float32)
        for name,s in q['native_spaces'].items()}


def train(q,job,out,source):
    """One original SingleEnvExperiment across all frozen tasks and four nodes."""
    provider,features,frozen=frozen_provider(q,job)
    geometry=MergeGeometry(source/'merge.net.xml')
    native_log=Jsonl(out/'native.jsonl.gz'); replay_log=Jsonl(out/'replay.jsonl.gz')
    minibatch_log=Jsonl(out/'minibatches.jsonl.gz'); losses=Jsonl(out/'losses.jsonl.gz')
    logs=[native_log,replay_log,minibatch_log,losses]
    recordings=[]; episodes=[]; snapshots=[]; workers=[]
    feature=experiment=None
    try:
        def factory(selection):
            index=selection['episode_index']
            if index!=len(recordings) or selection['actual_start_step']>=120000:
                raise ValueError('Unexpected extra task or automatic budget extension')
            directory=out/f'e{index}'; directory.mkdir(exist_ok=False)
            write_once(directory/'selection.json',selection)
            before=rng_fingerprint()
            worker=RngNeutralWorker(source,source/'configs/train_default_1.json',selection['simulator_seed'],
                directory/'environment.log',control_cap=500)
            if before!=rng_fingerprint(): raise ValueError('Creating a task consumed learner RNG')
            if worker.runtime['settings']['LEARNING_RATE']!=.0002:
                raise ValueError('Original configured learning rate changed')
            workers.append(worker)
            recordings.append(EpisodeRecorder(geometry,worker.runtime['settings'],
                identity(job,selection['simulator_seed']),native_log))
            write_once(directory/'runtime.json',json_safe(worker.runtime))
            return worker
        def emit(kind,value):
            if kind=='minibatch':
                # Preserve real sampled indices; do not duplicate 100 large audit rows per update.
                counts=Counter(r['level'] for r in value['transition_metadata'])
                minibatch_log.append({k:v for k,v in value.items() if k!='transition_metadata'}|{
                    'level_counts':{k:counts[k] for k in ('L','M','H')}})
            else:
                replay_log.append(json_safe({'kind':kind,'value':value}))
                if kind=='episode':
                    result={**value,**recordings[-1].finish()}
                    if result['actual_control_steps']!=value['actual_control_steps']:
                        raise ValueError('Episode recorder and real stored transitions differ')
                    episodes.append(result)
                    write_once(out/f'e{value["episode_index"]}'/'summary.json',json_safe(result))
                    recordings[-1]=None  # Full native packets are on disk, not retained across tasks.
        native_spaces=spaces(q)
        sampler=TrainingEpisodeSampler(q['task_manifest'],job,q['request_hash'],q['protocol'])
        stream=MatchedTaskStream(sampler,factory,native_spaces['observation'],native_spaces['action'])
        audit=MatchedReplayAudit(stream,on_event=emit)
        feature=PredictionFeatureEnv(stream,job['predictor_mode'],features,stream.current_frame)
        def record(event,obs,reward,done,info):
            recordings[-1].record(event,stream._worker.last_packet,obs,reward,done,info)
            audit.record(event,obs,reward,done,info)
        env=make_all_environment(feature,'cpu',record)
        experiment=experiment_type(out/'tensorboard')(preset_factory('cpu',.0002,smoke=False),env,quiet=True)
        checked=experiment._agent; audit.install(checked)
        initial=learner_hashes(checked)
        persistent=(id(checked),id(checked.agent.q._optimizer),id(checked.agent.policy._optimizer),id(audit.base))
        loss_counts={'q':0,'policy':0}
        for name in loss_counts:
            approximation=getattr(checked.agent,name); original=approximation.reinforce
            def observed(loss,*,original=original,name=name):
                value=float(loss.detach())
                if not math.isfinite(value): raise ValueError('Nonfinite original optimizer loss')
                loss_counts[name]+=1
                losses.append({'component':name,'value':value,'training_steps':audit.insertions,'sample_call':audit.sample_calls})
                return original(loss)
            approximation.reinforce=observed
        contract=checkpoint_contract(env,q['resolved_ddpg_preset'],q['predictors'][job['predictor_mode']]['sha256'],
            q['upstream_config_sha256'])
        write_once(out/'resolved_config.json',{'protocol':q['protocol'],'job':job,
            'weights_contract':contract,'task_manifest_hash':q['task_manifest_hash'],
            'preset':q['resolved_ddpg_preset'],'entry_contract':stream.reset_contract})
        def snapshot(node,actual):
            if (actual!=audit.insertions or actual!=stream.actual_control_steps or not env.done
                    or persistent!=(id(checked),id(checked.agent.q._optimizer),id(checked.agent.policy._optimizer),id(audit.base))):
                raise ValueError('Snapshot is not a persistent learner/replay stored terminal boundary')
            assert_frozen(provider,frozen)
            path=out/f'n{node}.pt'; h=save_evaluation_weights(checked,path,contract)
            item={'requested_steps':node,'actual_steps':actual,'weights_file':path.name,'weights_sha256':h,
                'weights_contract':contract,'replay_audit':audit.snapshot(),
                'all_frames_seen_including_reset_calls':checked.agent._frames_seen,
                'optimizer_loss_counts':deepcopy(loss_counts),'optimizer_replay_reinitialized':False}
            write_once(out/f'n{node}.json',json_safe(item)); snapshots.append(item)
            for log in logs: log.flush()
            print(f'[ic04 snapshot] {job["id"]} requested={node} actual={actual}',flush=True)
        train_nodes(experiment,q['protocol']['training']['checkpoint_frames'],snapshot)
        final=audit.snapshot()
        if (sampler.started!=sampler.completed or sum(r['steps'] for r in episodes)!=audit.insertions
                or not 120000<=audit.insertions<=120499
                or [s['requested_steps'] for s in snapshots]!=[20000,40000,60000,120000]
                or len(audit)!=audit.insertions or any(n!=audit.sample_calls for n in loss_counts.values())
                or features.calls!=sum(r['feature_calls'] for r in episodes)):
            raise ValueError('Training, minibatch, optimizer or feature audit count mismatch')
        assert_frozen(provider,frozen)
        write_once(out/'episodes.json',json_safe(episodes))
        return {'status':'complete','job':job,'from_scratch':True,'snapshots':snapshots,
            'actual_steps':experiment.frame-1,'initial_weight_sha256':initial,
            'predictor_frozen':True,'predictor_tensor_hash':frozen,'replay_audit':final,
            'optimizer_loss_counts':loss_counts,'original_minibatches':audit.sample_calls,
            'original_agent_calls':checked.agent._frames_seen,'feature_calls':features.calls,
            'raw_ticks':sum(r['raw_ticks'] for r in episodes),'complete_native_episodes':len(episodes),
            'checkpoint_purpose':'evaluation_only_not_training_resume','no_automatic_budget_extension':True,
            'training_returns_are_exploration_not_validation':True}
    finally:
        if feature is not None: feature.close()
        for worker in workers: worker.close()
        if experiment is not None: experiment._writer.close()
        for log in logs: log.close()


def evaluate(q,job,out,source,snapshot,weights_path):
    provider,features,frozen=frozen_provider(q,job)
    log=Jsonl(out/'native.jsonl.gz'); worker=feature=None
    try:
        worker=RngNeutralWorker(source,source/'configs/train_default_1.json',job['simulator_seed'],out/'environment.log',control_cap=500)
        recorder=EpisodeRecorder(MergeGeometry(source/'merge.net.xml'),worker.runtime['settings'],identity(job,job['simulator_seed']),log)
        feature=PredictionFeatureEnv(worker,job['predictor_mode'],features,worker.current_frame)
        def record(event,obs,reward,done,info): recorder.record(event,worker.last_packet,obs,reward,done,info)
        env=make_all_environment(feature,'cpu',record)
        from all.logging import DummyWriter
        checked=preset_factory('cpu',.0002,smoke=False)(env,DummyWriter())
        contract=checkpoint_contract(env,q['resolved_ddpg_preset'],q['predictors'][job['predictor_mode']]['sha256'],q['upstream_config_sha256'])
        if contract!=snapshot['weights_contract']: raise ValueError('Training/evaluation complete observation contract mismatch')
        load_evaluation_weights(checked,weights_path,contract,snapshot['weights_sha256'])
        initial=learner_hashes(checked)
        write_once(out/'resolved_config.json',{'protocol':q['protocol'],'job':job,'snapshot':snapshot,'upstream':worker.runtime['settings']})
        state=env.reset(); reward=0.
        for _ in range(500):
            action=checked.eval(state,reward)
            state,reward=env.step(action)
            if env.done:
                checked.eval(state,reward); break
        else: raise ValueError('No original native terminal within 500 controls')
        result=recorder.finish()
        if (len(checked.agent.replay_buffer) or learner_hashes(checked)!=initial
                or any(getattr(checked.agent,n)._optimizer.state for n in ('q','policy'))):
            raise ValueError('Frozen evaluation updated the policy or replay')
        assert_frozen(provider,frozen)
        row={**job,**result,'actual_training_steps':snapshot['actual_steps'],
            'policy_sha256':snapshot['weights_sha256'],'no_policy_updates':True}
        write_once(out/'episode.json',json_safe(row)); write_once(out/'actions.json',recorder.actions)
        return {'status':'complete','job':job,'outcome':json_safe(row),'no_policy_updates':True}
    finally:
        if feature is not None: feature.close()
        if worker is not None: worker.close()
        log.close()
