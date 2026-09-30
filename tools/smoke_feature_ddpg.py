"""P6b bounded four-arm DDPG optimizer smoke; no formal training/evaluation CLI."""
import argparse
import hashlib
import importlib.metadata
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
import train_response_predictors as shared
import audit_prediction_features as features
from prediction_rl.data.collection_store import read_json as read, write_once, file_hash as sha
from prediction_rl.data.frozen_dataset import inside, verify_historical_inputs
from prediction_rl.data.dataset_contract import digest
from prediction_rl.data.merge_geometry import read_extended_traffic
from prediction_rl.envs.upstream import upstream_session, AuditedJerkEnv
from prediction_rl.envs.prediction_features import PredictionFeatureEnv
from prediction_rl.features.probe_features import FrozenProbeFeatures, NAMES
from prediction_rl.prediction.interface import PredictorConfig
from prediction_rl.prediction.kinematic_reference import LaneConstantVelocity
from prediction_rl.prediction.trained_artifacts import load_trained_ensemble
from prediction_rl.training.all_ddpg import (make_all_environment, preset_factory, resolved_preset,
    experiment_type, network_shapes, checkpoint_contract, save_evaluation_weights, load_evaluation_weights)

CONFIG={'version':'p06_ddpg_feature_smoke_v1','purpose':'bounded_optimizer_integration_not_method_evaluation',
    'feature_audit':'artifacts/p6/p6a_features_01/report.json',
    'feature_audit_sha256':'18abb76be0d4c7a665ac7ff4f08e4c53ee1622dfa334c90aa296605b9373e06b',
    'arms':['baseline','zero','ordinary','conditional'],'run_seed':0,'requested_frames':64,'episode_cap':4,
    'device':'cpu','torch_threads':1,'diagnostic_overrides':{'replay_start_size':8,'minibatch_size':8},
    'execution_contract':'simulation_blocking_exact_v1','formal_training':False,'test_opened':False}


def dependencies():
    import all
    base=Path(all.__file__).resolve().parent
    files=['presets/continuous/ddpg.py','presets/continuous/models/__init__.py',
        'agents/ddpg.py','environments/gym.py','bodies/time.py','memory/replay_buffer.py',
        'approximation/approximation.py','approximation/q_continuous.py',
        'policies/deterministic.py','experiments/single_env_experiment.py']
    return {f:sha(base/f) for f in files}


def prerequisites(path):
    config=read(inside(ROOT,path))
    if config!=CONFIG:raise ValueError('Modified/unbounded DDPG smoke configuration')
    rp=inside(ROOT,config['feature_audit']);report=read(rp);request=read(rp.parent/'request.json')
    if (sha(rp)!=config['feature_audit_sha256'] or report['status']!='complete'
            or report['engineering_gate']is not True or report['working_tree_dirty']is not False
            or not all(report['checks'].values()) or request['source_hashes']!=report['source_hashes']
            or report['test_opened']is not False or report['ddpg_connected']is not False):
        raise ValueError('Need immutable accepted P6a replay')
    verify_historical_inputs(ROOT,{'python_lf:'+k:v for k,v in request['source_hashes'].items()},report['git_commit'])
    if sha(inside(ROOT,request['config_path']))!=request['config_sha256']:
        raise ValueError('P6a frozen configuration changed')
    r,p=features.prerequisites(request['config_path'])
    if r['ensembles']!=report['ensembles']:raise ValueError('Predictor selection changed')
    return r,p


def tensor_hash(state):
    h=hashlib.sha256()
    for k,v in sorted(state.items()):
        h.update(k.encode());h.update(str(v.dtype).encode());h.update(str(tuple(v.shape)).encode())
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def optimizer_steps(approximation):
    values=[int(float(x['step'])) for x in approximation._optimizer.state.values()]
    if not values or len(set(values))!=1:raise ValueError('Missing/inconsistent Adam updates')
    return values[0]


def worker(args):
    shared.clean();qp=inside(ROOT,args.request);q=read(qp)
    if (q['git_commit']!=shared.git('rev-parse','HEAD') or q['source_hashes']!=shared.source_hashes()
            or q['dependencies']!=dependencies() or q['config']!=CONFIG
            or sha(inside(ROOT,q['config_path']))!=q['config_sha256']):
        raise ValueError('DDPG worker source/config/dependencies changed')
    r,p=prerequisites(q['config_path']);out=qp.parent/args.arm;out.mkdir(exist_ok=False)
    report={**shared.metadata(),'status':'running','arm':args.arm,'run_seed':CONFIG['run_seed'],
            'optimizer_seed':CONFIG['run_seed'],'simulator_seed':CONFIG['run_seed'],
            'seed_contract':'upstream_single_run_seed_v1'}
    start=time.perf_counter();torch.set_num_threads(CONFIG['torch_threads'])
    try:
        provider=None;predictor_sha=None;predictor_before=None
        if args.arm in ('ordinary','conditional'):
            e=r['ensembles'][args.arm];predictor_sha=e['sha256']
            model=load_trained_ensemble(inside(ROOT,e['path']),p,args.arm,e['sha256'],e['selected'])
            provider=FrozenProbeFeatures(LaneConstantVelocity(ROOT/'RL-MPC-LaneMerging-master/merge.net.xml'),
                                         PredictorConfig.from_dict(p['model']),model)
            predictor_before=tensor_hash(model.state_dict())
        episodes=[];current=None
        def record(event,obs,reward,done,info):
            nonlocal current
            if args.arm=='zero' and np.any(obs[20:]):raise AssertionError('B1 is not zero')
            if event=='reset':
                current={'initial_observation':obs.tolist(),'reset_history_frames':info['prediction_features']['history_frames'],
                         'steps':[]};episodes.append(current)
            else:
                audit=info['execution_audit']
                current['steps'].append({'requested_jerk':audit['requested_jerk'],
                    'commanded_speed':audit['commanded_speed_reconstructed'],'reward':float(reward),
                    'done':bool(done),'termination':audit['termination_reason'],
                    'observation':obs.tolist(),'feature_latency_s':info['prediction_features']['latency_s']})
                if done and args.arm!='baseline' and np.any(obs[20:]):raise AssertionError('Terminal prediction block not zero')
        source=ROOT/'RL-MPC-LaneMerging-master';upstream=source/'configs/train_default_1.json'
        with upstream_session(source,upstream,CONFIG['run_seed']) as settings:
            params=resolved_preset(CONFIG['device'],settings.LEARNING_RATE,smoke=True)
            raw=PredictionFeatureEnv(AuditedJerkEnv(),args.arm,provider,read_extended_traffic if provider else None)
            env=make_all_environment(raw,CONFIG['device'],record)
            resolved={'upstream':settings.export_settings(),'ddpg_preset':params,'smoke_config':CONFIG,
                'feature_names':[] if args.arm=='baseline' else list(NAMES),'arm':args.arm,
                'upstream_config_sha256':sha(upstream),'predictor_sha256':predictor_sha}
            write_once(out/'resolved_config.json',resolved)
            experiment=None
            try:
                experiment=experiment_type(out/'training_logs')(
                    preset_factory(CONFIG['device'],settings.LEARNING_RATE,smoke=True),env,quiet=True)
                checked=experiment._agent;core=checked.agent
                before={name:{k:v.clone() for k,v in approximation.model.state_dict().items()}
                    for name,approximation in (('policy',core.policy),('q',core.q))}
                init={k:tensor_hash(v) for k,v in before.items()}
                experiment.train(frames=CONFIG['requested_frames'],episodes=CONFIG['episode_cap'])
                deltas={name:sum(float((before[name][k]-v).abs().sum()) for k,v in approximation.model.state_dict().items())
                        for name,approximation in (('policy',core.policy),('q',core.q))}
                updates={name:optimizer_steps(approximation) for name,approximation in (('policy',core.policy),('q',core.q))}
                if any(x<=0 or not np.isfinite(x) for x in deltas.values()) or min(updates.values())<=0:
                    raise AssertionError('Actor/critic optimizer did not update finite parameters')
                if any(not torch.isfinite(v).all() for a in (core.policy,core.q) for v in a.model.state_dict().values()):
                    raise AssertionError('Invalid trained weights')
                transitions=sum(len(e['steps']) for e in episodes)
                if (transitions!=experiment.frame-1 or transitions>CONFIG['requested_frames']+499
                        or len(episodes)>CONFIG['episode_cap']):raise AssertionError('Original bounded training loop differs')
                if provider is not None:
                    if (tensor_hash(provider.model.state_dict())!=predictor_before
                            or any(v.grad is not None or v.requires_grad for v in provider.model.parameters())):
                        raise AssertionError('Predictor was changed by DDPG training')
                states,actions,rewards,next_states,weights=core.replay_buffer.sample(8)
                if any('execution_audit' in (x or {}) or 'prediction_features' in (x or {}) for x in states.info):
                    raise AssertionError('Detailed traffic audits leaked into replay')
                with torch.no_grad():
                    expected_policy=core.policy.eval(states);expected_q=core.q.eval(states,actions)
                contract=checkpoint_contract(env,params,predictor_sha,sha(upstream))
                weights_path=out/'evaluation_weights.pt';weights_sha=save_evaluation_weights(checked,weights_path,contract)
                from all.logging import DummyWriter
                restored=preset_factory(CONFIG['device'],settings.LEARNING_RATE,smoke=True)(env,DummyWriter())
                load_evaluation_weights(restored,weights_path,contract,weights_sha)
                torch.testing.assert_close(expected_policy,restored.agent.policy.eval(states),rtol=0,atol=0)
                torch.testing.assert_close(expected_q,restored.agent.q.eval(states,actions),rtol=0,atol=0)
                write_once(out/'episodes.json',episodes)
                report.update(status='complete',initial_weight_sha256=init,network_shapes=network_shapes(checked),
                    actual_steps=transitions,completed_episodes=len(episodes),replay_transitions=len(core.replay_buffer),
                    optimizer_updates=updates,parameter_l1_changes=deltas,episode_returns=[sum(s['reward'] for s in e['steps']) for e in episodes],
                    predictor_unchanged=True,checkpoint_roundtrip_exact=True,evaluation_weights_sha256=weights_sha,
                    checkpoint_contract=contract,episodes_sha256=sha(out/'episodes.json'),resolved_config_sha256=sha(out/'resolved_config.json'),
                    policy_input_dimension=env.state_space.shape[0]+1,formal_training=False,method_effect_established=False)
            finally:
                env.close()
                if experiment is not None:experiment._writer.close()
        if not list((out/'training_logs').rglob('events.out.tfevents.*')):raise AssertionError('TensorBoard logs absent')
    except BaseException:
        report.update(status='failed',failure_reason=traceback.format_exc());raise
    finally:
        report.update(finished_at=shared.now(),elapsed_s=time.perf_counter()-start)
        write_once(out/'report.json',report)


def audit(args):
    shared.clean();cp=inside(ROOT,args.config);prerequisites(cp)
    out=shared.output_dir('p6',args.run_id)
    q={**shared.metadata(),'config_path':str(cp.relative_to(ROOT)),'config':read(cp),
        'config_sha256':sha(cp),'source_hashes':shared.source_hashes(),'dependencies':dependencies(),'environment':shared.env()}
    write_once(out/'request.json',q);report={**q,'status':'running'};start=time.perf_counter()
    try:
        children={}
        for arm in CONFIG['arms']:
            print(f'[p6b] {arm}: requested_frames=64 episode_cap=4 step_cap<=563',flush=True)
            command=[sys.executable,str(Path(__file__).resolve()),'--request',str(out/'request.json'),'--arm',arm]
            with (out/(arm+'.log')).open('w',encoding='utf-8') as log:
                subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=180)
            children[arm]=read(out/arm/'report.json')
        matching=all(children[arm]['initial_weight_sha256']==children['zero']['initial_weight_sha256']
                     for arm in ('ordinary','conditional'))
        if not matching:raise AssertionError('Expanded arms did not share matched initialization')
        report.update(status='complete',engineering_gate=True,children=children,
            matched_expanded_initialization=True,ddpg_connected=True,formal_training=False,
            method_effect_established=False,test_opened=False,
            sumo_version=subprocess.check_output(['sumo','--version'],text=True).splitlines()[0],
            versions={k:importlib.metadata.version(k) for k in ('torch','numpy','gym','autonomous-learning-library','traci')})
    except BaseException:
        report.update(status='failed',engineering_gate=False,failure_reason=traceback.format_exc());raise
    finally:
        report.update(finished_at=shared.now(),elapsed_s=time.perf_counter()-start)
        write_once(out/'report.json',report);print('p6b_report='+str(out/'report.json'),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/development/p06_ddpg_smoke_v1.json')
    parser.add_argument('--run-id');parser.add_argument('--request');parser.add_argument('--arm',choices=CONFIG['arms'])
    args=parser.parse_args()
    if args.request:
        if not args.arm:parser.error('Worker requires arm')
        worker(args)
    else:
        if not args.run_id:parser.error('Smoke needs short new run ID')
        audit(args)
