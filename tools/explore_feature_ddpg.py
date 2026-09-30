"""User-run P7 exploration: immutable prepare/train/evaluate/aggregate, no resume of weights."""
import argparse
import importlib.metadata
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
import subprocess
import shutil
import sys
import time
import traceback
import uuid

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
import train_response_predictors as shared
import smoke_feature_ddpg as smoke
from prediction_rl.data.collection_store import (read_json as read, write_once, file_hash as sha,
    run_lock, seal_episode, verify_episode)
from prediction_rl.data.dataset_contract import digest
from prediction_rl.data.frozen_dataset import inside, verify_historical_inputs, verify_historical_text_checkout
from prediction_rl.data.merge_geometry import read_extended_traffic
from prediction_rl.envs.upstream import upstream_session, AuditedJerkEnv
from prediction_rl.envs.prediction_features import PredictionFeatureEnv
from prediction_rl.features.probe_features import FrozenProbeFeatures
from prediction_rl.prediction.interface import PredictorConfig
from prediction_rl.prediction.kinematic_reference import LaneConstantVelocity
from prediction_rl.prediction.trained_artifacts import load_trained_ensemble
from prediction_rl.training.all_ddpg import (make_all_environment, preset_factory, resolved_preset,
    experiment_type, checkpoint_contract, save_evaluation_weights, load_evaluation_weights)
from prediction_rl.evaluation.exploration import PROTOCOL,AUDIT,validate_protocol,jobs,EpisodeMetrics,aggregate,zero_channel_query

SOURCE=ROOT/'RL-MPC-LaneMerging-master'
UPSTREAM=SOURCE/'configs/train_default_1.json'


def upstream_hashes():
    paths=[p for pattern in ('*.py','*.pyx','*.xml','*.sumocfg') for p in SOURCE.glob(pattern)]
    paths.append(UPSTREAM)
    return {str(p.relative_to(ROOT)):shared.fingerprint(p.read_text(encoding='utf-8')) for p in paths}


def runtime_identity():
    executable=shutil.which('sumo.exe') or shutil.which('sumo')
    if executable is None:raise FileNotFoundError('SUMO executable not found')
    binary=Path(executable).resolve()
    return {'sumo_executable':str(binary),'sumo_sha256':sha(binary),
        'sumo_version':subprocess.check_output([str(binary),'--version'],text=True).splitlines()[0],
        'packages':{k:importlib.metadata.version(k) for k in
                    ('torch','numpy','gym','autonomous-learning-library','traci','tensorboardX')},
        'upstream_compiled_modules':{str(p.relative_to(ROOT)):sha(p) for p in SOURCE.glob('*.pyd')}}


def prerequisites(config):
    validate_protocol(config);rp=inside(ROOT,config['ddpg_audit']);r=read(rp);q=read(rp.parent/'request.json')
    if (sha(rp)!=config['ddpg_audit_sha256'] or r['status']!='complete' or r['engineering_gate']is not True
            or r['matched_expanded_initialization']is not True or r['test_opened']is not False
            or r['working_tree_dirty']is not False or q['source_hashes']!=r['source_hashes']):
        raise ValueError('Need accepted immutable P6b optimizer integration')
    verify_historical_inputs(ROOT,{'python_lf:'+k:v for k,v in q['source_hashes'].items()},r['git_commit'])
    verify_historical_text_checkout(ROOT,q['config_path'],q['config_sha256'],r['git_commit'])
    if q['dependencies']!=smoke.dependencies():raise ValueError('Installed ALL dependency sources changed')
    return smoke.prerequisites(q['config_path'])


def prepare(config,run_id):
    shared.clean();r,p=prerequisites(config);out=shared.output_dir('p7',run_id)
    t=config['training'];parameters=resolved_preset(t['device'],t['learning_rate'],smoke=config==AUDIT)
    q={**shared.metadata(),'version':'p07_request_v1','protocol':deepcopy(config),
       'source_hashes':shared.source_hashes(),'upstream_hashes':upstream_hashes(),
       'dependencies':smoke.dependencies(),'environment':shared.env(),'runtime_identity':runtime_identity(),'preset':parameters,
       'predictors':r['ensembles'],'predictor_training_request':p,'upstream_config_sha256':sha(UPSTREAM),
       'train_jobs':jobs(config,'train'),'evaluation_jobs':jobs(config,'evaluate')}
    write_once(out/'request.json',q);write_once(out/'preparation.json',{'request_hash':digest(q)})
    print('exploration_request='+str(out/'request.json'),flush=True)
    print('confirm_request_hash='+digest(q),flush=True)
    return out/'request.json',digest(q)


def load_request(path,expected_hash):
    shared.clean();path=inside(ROOT,path);q=read(path);validate_protocol(q['protocol'])
    if (digest(q)!=expected_hash or read(path.parent/'preparation.json')['request_hash']!=expected_hash
            or q['source_hashes']!=shared.source_hashes() or q['upstream_hashes']!=upstream_hashes()
            or q['dependencies']!=smoke.dependencies() or q['environment']!=shared.env()
            or q['runtime_identity']!=runtime_identity()
            or q['upstream_config_sha256']!=sha(UPSTREAM)
            or q['train_jobs']!=jobs(q['protocol'],'train') or q['evaluation_jobs']!=jobs(q['protocol'],'evaluate')):
        raise ValueError('Request/source/config/environment changed; no silent adaptation')
    r,p=prerequisites(q['protocol'])
    if q['predictors']!=r['ensembles'] or q['predictor_training_request']!=p:
        raise ValueError('Predictor identities changed')
    t=q['protocol']['training']
    if q['preset']!=resolved_preset(t['device'],t['learning_rate'],smoke=q['protocol']==AUDIT):
        raise ValueError('DDPG budget/defaults changed')
    return path,q


def binding(q,stage,job):return {'request_hash':digest(q),'stage':stage,'job':job}


def provider_for(q,arm):
    if arm not in ('ordinary','conditional'):return None,None
    e=q['predictors'][arm];p=q['predictor_training_request']
    model=load_trained_ensemble(inside(ROOT,e['path']),p,arm,e['sha256'],e['selected'])
    return FrozenProbeFeatures(LaneConstantVelocity(SOURCE/'merge.net.xml'),PredictorConfig.from_dict(p['model']),model),e['sha256']


def environment(q,job,provider,recorder):
    raw=PredictionFeatureEnv(AuditedJerkEnv(),job['arm'],provider,read_extended_traffic if provider else None)
    return make_all_environment(raw,q['protocol']['training']['device'],recorder)


def train_job(q,job,out):
    t=q['protocol']['training'];provider,predictor_sha=provider_for(q,job['arm'])
    frozen=None if provider is None else smoke.tensor_hash(provider.model.state_dict())
    episodes=[];current=None;finished=0
    def record(event,obs,reward,done,info):
        nonlocal current,finished
        if job['arm']=='zero' and np.any(obs[20:]):raise ValueError('B1 channels changed')
        if event=='reset':current=EpisodeMetrics()
        current.record(event,obs,reward,done,info)
        if done:
            row=current.result();finished+=row['steps'];row['finished_training_step']=finished;episodes.append(row)
    with upstream_session(SOURCE,UPSTREAM,job['run_seed']) as settings:
        if settings.LEARNING_RATE!=t['learning_rate']:raise ValueError('Original learning rate changed')
        env=environment(q,job,provider,record);experiment=None
        try:
            write_once(out/'resolved_config.json',{'upstream':settings.export_settings(),'protocol':q['protocol'],
                'preset':q['preset'],'job':job,'predictor_sha256':predictor_sha})
            experiment=experiment_type(out/'training_logs')(
                preset_factory(t['device'],settings.LEARNING_RATE,smoke=q['protocol']==AUDIT),env,quiet=False)
            checked=experiment._agent;core=checked.agent
            initial={n:smoke.tensor_hash(a.model.state_dict()) for n,a in (('policy',core.policy),('q',core.q))}
            experiment.train(frames=t['requested_frames'],episodes=np.inf if t['episode_cap'] is None else t['episode_cap'])
            if finished!=experiment.frame-1 or finished>t['requested_frames']+499:
                raise ValueError('Original episode-boundary training budget violated')
            if q['protocol']!=AUDIT and finished<t['requested_frames']:raise ValueError('Exploration training stopped early')
            if provider is not None and (smoke.tensor_hash(provider.model.state_dict())!=frozen
                    or any(p.grad is not None or p.requires_grad for p in provider.model.parameters())):
                raise ValueError('Frozen predictor changed')
            contract=checkpoint_contract(env,q['preset'],predictor_sha,q['upstream_config_sha256'])
            weights_sha=save_evaluation_weights(checked,out/'final.pt',contract)
            updates={n:smoke.optimizer_steps(a) for n,a in (('policy',core.policy),('q',core.q))}
            write_once(out/'episodes.json',episodes)
            return {'job':job,'status':'complete','actual_steps':finished,'completed_episodes':len(episodes),
                'optimizer_updates':updates,'initial_weight_sha256':initial,'weights_sha256':weights_sha,
                'weights_contract':contract,'predictor_frozen':True,'selection':q['protocol']['selection'],
                'episodes_sha256':sha(out/'episodes.json')}
        finally:
            try:env.close()
            finally:
                if experiment is not None:experiment._writer.close()


def evaluation_job(q,job,out,root):
    train=next(j for j in q['train_jobs'] if j['id']==job['training_id'])
    parent=root/'train'/train['id'];receipt=verify_episode(parent,binding(q,'train',train));metadata=receipt['metadata']
    if sha(parent/'final.pt')!=metadata['weights_sha256']:raise ValueError('Policy weights changed')
    provider,predictor_sha=provider_for(q,job['arm']);metrics=EpisodeMetrics();trace=[]
    frozen=None if provider is None else smoke.tensor_hash(provider.model.state_dict())
    with upstream_session(SOURCE,UPSTREAM,job['simulator_seed']) as settings:
        env=environment(q,job,provider,metrics.record)
        try:
            from all.logging import DummyWriter
            t=q['protocol']['training'];checked=preset_factory(t['device'],settings.LEARNING_RATE,smoke=q['protocol']==AUDIT)(env,DummyWriter())
            contract=checkpoint_contract(env,q['preset'],predictor_sha,q['upstream_config_sha256'])
            if contract!=metadata['weights_contract']:raise ValueError('Evaluation uses different inputs/control contract')
            load_evaluation_weights(checked,parent/'final.pt',contract,metadata['weights_sha256'])
            initial={n:smoke.tensor_hash(a.model.state_dict()) for n,a in (('policy',checked.agent.policy),('q',checked.agent.q))}
            write_once(out/'resolved_config.json',{'upstream':settings.export_settings(),'protocol':q['protocol'],
                'preset':q['preset'],'job':job,'policy_sha256':metadata['weights_sha256'],'predictor_sha256':predictor_sha})
            state=env.reset();reward=0.;sensitivity=[]
            for step in range(500):
                # Preserve original TimeFeature exactly; the zero-channel query
                # is a separate deterministic forward, NOT a second body.eval().
                time_value=torch.zeros(1,device=state.features.device) if checked.body.timestep is None else checked.body.timestep.clone()
                action=checked.eval(state,reward)
                zero_action=None
                if provider is not None:
                    zero_action=zero_channel_query(checked,state,time_value)
                    sensitivity.append(float((action-zero_action).abs().item()))
                state,reward=env.step(action)
                trace.append({'step':step,'jerk':float(action.item()),'reward':float(reward),'done':bool(env.done),
                    'zero_channel_query_jerk':None if zero_action is None else float(zero_action.item())})
                if env.done:
                    checked.eval(state,reward)  # Original test-loop terminal time reset.
                    break
            else:raise ValueError('Original 500-step limit not honored')
            if (len(checked.agent.replay_buffer)!=0 or any(a._optimizer.state for a in (checked.agent.policy,checked.agent.q))
                    or any(smoke.tensor_hash(a.model.state_dict())!=initial[n]
                           for n,a in (('policy',checked.agent.policy),('q',checked.agent.q)))):
                raise ValueError('Evaluation updated policy/critic or replay')
            if provider is not None and (smoke.tensor_hash(provider.model.state_dict())!=frozen
                    or any(p.grad is not None or p.requires_grad for p in provider.model.parameters())):
                raise ValueError('Evaluation changed predictor')
            row={**job,**metrics.result(),'mean_abs_zero_channel_action_change':None if not sensitivity else float(np.mean(sensitivity)),
                 'policy_sha256':metadata['weights_sha256'],'query_is_not_executed':True}
            write_once(out/'episode.json',row);write_once(out/'actions.json',trace)
            return {'job':job,'status':'complete','outcome':row,'episode_sha256':sha(out/'episode.json'),
                    'policy_sha256':metadata['weights_sha256'],'no_policy_updates':True}
        finally:env.close()


def worker(args):
    path,q=load_request(args.request,args.confirm_request_hash)
    stage=args.stage;job=next(j for j in q['train_jobs' if stage=='train' else 'evaluation_jobs'] if j['id']==args.job)
    out=path.parent/stage/job['id'];out.mkdir(parents=True,exist_ok=False)
    started={**shared.metadata(),'binding':binding(q,stage,job)};write_once(out/'started.json',started)
    torch.set_num_threads(q['protocol']['training']['torch_threads']);start=time.perf_counter()
    try:
        result=train_job(q,job,out) if stage=='train' else evaluation_job(q,job,out,path.parent)
        result.update(provenance=started,elapsed_s=time.perf_counter()-start,finished_at=shared.now())
        write_once(out/'report.json',result);seal_episode(out,binding(q,stage,job),result)
    except BaseException:
        write_once(out/'failure.json',{**started,'status':'failed','failure_reason':traceback.format_exc()});raise


def stage_run(path,expected_hash,stage,*,resume=False):
    path,q=load_request(path,expected_hash);out=path.parent
    invocation=out/'invocations'/uuid.uuid4().hex[:12];invocation.mkdir(parents=True,exist_ok=False)
    report={**shared.metadata(),'status':'running','stage':stage,'request_hash':digest(q),'jobs':[]}
    with run_lock(out):
        try:
            if stage=='evaluate':
                for j in q['train_jobs']:verify_episode(out/'train'/j['id'],binding(q,'train',j))
            plan=q['train_jobs' if stage=='train' else 'evaluation_jobs'];pending=[]
            for j in plan:
                directory=out/stage/j['id']
                if directory.exists():
                    if not resume:raise ValueError('Existing job: --resume skips verified COMPLETE jobs only')
                    verify_episode(directory,binding(q,stage,j));report['jobs'].append({'id':j['id'],'action':'reuse'})
                else:pending.append(j)
            def run_child(j):
                command=[sys.executable,str(Path(__file__).resolve()),'worker','--request',str(path),
                    '--confirm-request-hash',digest(q),'--stage',stage,'--job',j['id']]
                print(f'[p7] {stage} {j["id"]}',flush=True)
                with (invocation/(j['id']+'.log')).open('w',encoding='utf-8') as log:
                    # No deadline-based observation substitutions and no automatic retry.
                    subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
                verify_episode(out/stage/j['id'],binding(q,stage,j));return {'id':j['id'],'action':'run'}
            workers=q['protocol']['training']['workers'] if stage=='train' else q['protocol']['evaluation']['workers']
            with ThreadPoolExecutor(max_workers=workers) as pool:
                # Never enqueue the entire long suite after an invalid job fails.
                for first in range(0,len(pending),workers):
                    futures=[pool.submit(run_child,j) for j in pending[first:first+workers]]
                    report['jobs'].extend(f.result() for f in futures)
            if stage=='train':
                for s in q['protocol']['training']['run_seeds']:
                    initial=[verify_episode(out/'train'/f'{a}_s{s}',binding(q,'train',next(j for j in plan if j['id']==f'{a}_s{s}')))['metadata']['initial_weight_sha256']
                             for a in ('zero','ordinary','conditional')]
                    if any(x!=initial[0] for x in initial[1:]):raise ValueError('Expanded policy initializations differ')
            report.update(status='complete',finished_at=shared.now())
        except BaseException:
            report.update(status='failed',failure_reason=traceback.format_exc(),finished_at=shared.now());raise
        finally:
            write_once(invocation/'report.json',report);print('stage_report='+str(invocation/'report.json'),flush=True)
    return invocation/'report.json'


def aggregate_run(path,expected_hash):
    path,q=load_request(path,expected_hash);out=path.parent
    with run_lock(out):
        trained={j['id']:verify_episode(out/'train'/j['id'],binding(q,'train',j))['metadata'] for j in q['train_jobs']}
        for seed in q['protocol']['training']['run_seeds']:
            initial=[trained[f'{arm}_s{seed}']['initial_weight_sha256'] for arm in ('zero','ordinary','conditional')]
            if any(x!=initial[0] for x in initial[1:]):raise ValueError('Expanded policy initializations differ')
        rows=[verify_episode(out/'evaluate'/j['id'],binding(q,'evaluate',j))['metadata']['outcome'] for j in q['evaluation_jobs']]
        if any(r['policy_sha256']!=trained[r['training_id']]['weights_sha256'] for r in rows):
            raise ValueError('Evaluation policy lineage changed')
        report={**aggregate(q['protocol'],rows),'request_hash':digest(q),'provenance':shared.metadata(),
                'purpose':q['protocol']['purpose'],'finished_at':shared.now()}
        write_once(out/'aggregate.json',report)
    print('exploration_aggregate='+str(out/'aggregate.json'),flush=True);return out/'aggregate.json'


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare');p.add_argument('--config',default='configs/development/p07_exploration_v1.json');p.add_argument('--run-id',required=True)
    for name in ('run','aggregate','worker'):
        p=sub.add_parser(name);p.add_argument('--request',required=True);p.add_argument('--confirm-request-hash',required=True)
        if name!='aggregate':p.add_argument('--stage',choices=['train','evaluate'],required=True)
        if name=='worker':p.add_argument('--job',required=True)
        if name=='run':p.add_argument('--resume',action='store_true')
    p=sub.add_parser('audit');p.add_argument('--run-id',required=True)
    args=parser.parse_args()
    if args.command=='prepare':prepare(read(inside(ROOT,args.config)),args.run_id)
    elif args.command=='worker':worker(args)
    elif args.command=='run':stage_run(args.request,args.confirm_request_hash,args.stage,resume=args.resume)
    elif args.command=='aggregate':aggregate_run(args.request,args.confirm_request_hash)
    else:
        path,h=prepare(AUDIT,args.run_id);stage_run(path,h,'train');stage_run(path,h,'evaluate');aggregate_run(path,h)


if __name__=='__main__':main()
