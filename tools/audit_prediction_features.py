"""P6a bounded isolated SUMO replay with frozen predictors; NEVER trains DDPG."""
import argparse
import importlib.metadata
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'),str(ROOT/'tools')]
import train_response_predictors as shared
import diagnose_task_relevance as task
from prediction_rl.data.collection_store import read_json as read, write_once, file_hash as sha
from prediction_rl.data.dataset_contract import digest
from prediction_rl.data.frozen_dataset import inside, verify_historical_inputs
from prediction_rl.data.merge_geometry import read_extended_traffic
from prediction_rl.prediction.interface import PredictorConfig
from prediction_rl.prediction.kinematic_reference import LaneConstantVelocity
from prediction_rl.prediction.trained_artifacts import load_trained_ensemble
from prediction_rl.features.probe_features import FrozenProbeFeatures, VERSION, NAMES, DIMENSION
from prediction_rl.envs.upstream import AuditedJerkEnv, upstream_session, traffic_snapshot
from prediction_rl.envs.prediction_features import PredictionFeatureEnv

CONFIG = {
    'version':VERSION, 'purpose':'bounded_engineering_replay_not_policy_evaluation',
    'task_report':'artifacts/task_diagnostics/predictor_v1_task1/report.json',
    'task_report_sha256':'0c2606bd1266d4ddbf9d6d8db5ab6f7f96fcab457ffdb7a11f8f15a29edd4938',
    'arms':['baseline','zero','ordinary','conditional'],
    'replays':[{'id':'stream','simulator_seed':0,'jerks':[5.],'episodes':2,'step_cap':160,'require_terminal':True},
               {'id':'mixed','simulator_seed':100,'jerks':[-5.,-2.5,0.,2.5,5.],'episodes':1,'step_cap':40,'require_terminal':False}],
    'execution_contract':'simulation_blocking_exact_v1', 'formal_training':False,'test_opened':False}


def prerequisites(path):
    config = read(inside(ROOT,path))
    if config != CONFIG: raise ValueError('Modified/unbounded replay configuration')
    rp = inside(ROOT,config['task_report']); report = read(rp); q = read(rp.parent/'request.json')
    if (sha(rp)!=config['task_report_sha256'] or report['status']!='complete'
            or report['working_tree_dirty'] is not False or report['request_hash']!=digest(q)
            or sha(rp.parent/'roots.json')!=report['roots_sha256']):
        raise ValueError('P5c evidence changed')
    verify_historical_inputs(ROOT,{'python_lf:'+k:v for k,v in q['source_hashes'].items()},report['git_commit'])
    c,r,p,d,hashes = task.prerequisites(q['config_path'])
    if c != q['config'] or hashes != q['input_hashes']:
        raise ValueError('P5c prerequisite identities changed')
    return r,p


def rng_hash():
    state = np.random.get_state()
    return digest({'python':repr(random.getstate()),'numpy':[state[0],state[1].tolist(),*state[2:]],
                   'torch':torch.get_rng_state().tolist()})


def worker(args):
    shared.clean(); request_path=inside(ROOT,args.request); request=read(request_path)
    if request['git_commit']!=shared.git('rev-parse','HEAD') or request['source_hashes']!=shared.source_hashes():
        raise ValueError('Worker source changed')
    r,p=prerequisites(request['config_path']); config=request['config']
    if config!=CONFIG: raise ValueError('Worker configuration changed')
    replay=next(x for x in config['replays'] if x['id']==args.replay)
    output=request_path.parent/f'{args.replay}_{args.arm}';output.mkdir(exist_ok=False)
    started=shared.metadata(); start=time.perf_counter(); report={**started,'status':'running','arm':args.arm,'replay':replay}
    torch.set_num_threads(1)
    try:
        provider=None
        if args.arm in ('ordinary','conditional'):
            e=r['ensembles'][args.arm]
            model=load_trained_ensemble(inside(ROOT,e['path']),p,args.arm,e['sha256'],e['selected'])
            provider=FrozenProbeFeatures(LaneConstantVelocity(ROOT/'RL-MPC-LaneMerging-master/merge.net.xml'),
                                         PredictorConfig.from_dict(p['model']),model)
        source=ROOT/'RL-MPC-LaneMerging-master'
        with upstream_session(source,source/'configs/train_default_1.json',replay['simulator_seed']) as settings:
            import control
            write_once(output/'resolved_config.json',settings.export_settings())
            env=PredictionFeatureEnv(AuditedJerkEnv(),args.arm,provider,read_extended_traffic if provider else None)
            traces=[];features=[];latencies=[]
            try:
                for episode in range(replay['episodes']):
                    obs,info=env.reset(); latencies.append(info.pop('prediction_features')['latency_s'])
                    trace={'reset_observation':obs[:env.base_dimension].tolist(),'reset_info':info,
                           'reset_rng':rng_hash(),'reset_delay':float(control.delay),'steps':[]}
                    feature=[obs[env.base_dimension:].tolist()]
                    done=False
                    for step in range(replay['step_cap']):
                        action=np.array([replay['jerks'][step%len(replay['jerks'])]],np.float32)
                        obs,reward,done,info=env.step(action)
                        latencies.append(info.pop('prediction_features')['latency_s'])
                        feature.append(obs[env.base_dimension:].tolist())
                        trace['steps'].append({'action':action.tolist(),'observation':obs[:env.base_dimension].tolist(),
                            'reward':float(reward),'done':bool(done),'info':info,'traffic':traffic_snapshot(),
                            'delay':float(control.delay),'rng':rng_hash()})
                        if done: break
                    if replay['require_terminal'] and not done: raise ValueError('Replay cap reached before required termination')
                    if args.arm!='baseline' and done and np.any(obs[env.base_dimension:]): raise AssertionError('Terminal features not zero')
                    if args.arm=='zero' and np.any(feature): raise AssertionError('B1 channels not zero')
                    traces.append(trace); features.append(feature)
            finally: env.close()
        write_once(output/'traces.json',traces);write_once(output/'features.json',features)
        report.update(status='complete',steps=[len(x['steps']) for x in traces],
            traces_sha256=sha(output/'traces.json'),features_sha256=sha(output/'features.json'),
            resolved_config_sha256=sha(output/'resolved_config.json'),
            expanded_dimension=env.observation_space.shape[0],
            feature_nonzero_cells=sum(int(np.count_nonzero(x)) for x in features),
            feature_latency_s={'count':len(latencies),'mean':float(np.mean(latencies)),
                'p95':float(np.quantile(latencies,.95)),'max':max(latencies)},
            predictor_sha256=None if provider is None else r['ensembles'][args.arm]['sha256'])
    except BaseException:
        report.update(status='failed',failure_reason=traceback.format_exc());raise
    finally:
        report.update(finished_at=shared.now(),elapsed_s=time.perf_counter()-start)
        write_once(output/'report.json',report)


def audit(args):
    shared.clean(); cp=inside(ROOT,args.config);r,p=prerequisites(cp)
    out=shared.output_dir('p6',args.run_id);request={**shared.metadata(),
        'config_path':str(cp.relative_to(ROOT)),'config':read(cp),'config_sha256':sha(cp),
        'source_hashes':shared.source_hashes(),'environment':shared.env(),
        'feature_names':list(NAMES),'feature_dimension':DIMENSION,'ensembles':r['ensembles']}
    write_once(out/'request.json',request);start=time.perf_counter();report={**request,'status':'running'}
    try:
        children={};checks={}
        for replay in CONFIG['replays']:
            reference=None
            for arm in CONFIG['arms']:
                name=replay['id']+'_'+arm; print(f'[p6a] {name} episodes<={replay["episodes"]} steps/episode<={replay["step_cap"]}',flush=True)
                command=[sys.executable,str(Path(__file__).resolve()),'--request',str(out/'request.json'),
                         '--arm',arm,'--replay',replay['id']]
                with (out/(name+'.log')).open('w',encoding='utf-8') as log:
                    subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=120)
                child=read(out/name/'report.json');trace=read(out/name/'traces.json');children[name]=child
                if arm=='baseline': reference=trace
                else: checks[name+'_exact_base_reward_action_traffic_rng_parity']=trace==reference
        if not all(checks.values()):raise AssertionError('Original execution replay parity failed')
        report.update(status='complete',checks=checks,children=children,engineering_gate=True,
            formal_training=False,ddpg_connected=False,method_effect_established=False,test_opened=False,
            versions={k:importlib.metadata.version(k) for k in ('torch','numpy','gym','traci')},
            sumo_version=subprocess.check_output(['sumo','--version'],text=True).splitlines()[0])
    except BaseException:
        report.update(status='failed',engineering_gate=False,failure_reason=traceback.format_exc());raise
    finally:
        report.update(finished_at=shared.now(),elapsed_s=time.perf_counter()-start)
        write_once(out/'report.json',report);print('p6a_report='+str(out/'report.json'),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/development/p06_feature_replay_v1.json')
    parser.add_argument('--run-id');parser.add_argument('--request')
    parser.add_argument('--arm',choices=CONFIG['arms']);parser.add_argument('--replay',choices=['stream','mixed'])
    args=parser.parse_args()
    if args.request:
        if not args.arm or not args.replay:parser.error('Worker needs arm and replay')
        worker(args)
    else:
        if not args.run_id:parser.error('Audit needs short new run ID')
        audit(args)
