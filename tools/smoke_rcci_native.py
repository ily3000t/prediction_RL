"""RCCI03 bounded native observation replay; no agent, optimizer or training."""
import argparse
from copy import deepcopy
import hashlib
from pathlib import Path
import random
import re
import shutil
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'),str(ROOT/'tools')]
import numpy as np
import torch

import audit_rcci_feasibility as shared
import validate_rcci_v2_cohort as cohort
from prediction_rl.data.collection_store import read_json as read, file_hash as sha, write_once, run_lock
from prediction_rl.data.dataset_contract import digest
from prediction_rl.data.frozen_dataset import inside, verify_historical_text_checkout
from prediction_rl.data.actor_adapter import build_actor_inputs
from prediction_rl.data.merge_geometry import read_extended_traffic
from prediction_rl.envs.upstream import upstream_session, AuditedJerkEnv, traffic_snapshot
from prediction_rl.envs.prediction_features import PredictionFeatureEnv
from prediction_rl.envs.rcci_features import RCCIV2FeatureEnv
from prediction_rl.features.probe_features import FrozenProbeFeatures, summarize as raw_summary
from prediction_rl.features.rcci_provider import FrozenRCCIV2Features
from prediction_rl.features.road_consequences_v2 import contract, summarize_road_consequences_v2
from prediction_rl.prediction.interface import PredictorConfig, probe_plans
from prediction_rl.prediction.candidate_ranking import rollout_ego
from prediction_rl.prediction.kinematic_reference import LaneConstantVelocity
from prediction_rl.prediction.trained_artifacts import load_trained_ensemble

VERSION = 'rcci03_bounded_native_smoke_v1'
CONFIG_PATH = 'configs/development/rcci03_native_smoke_v1.json'
SETTINGS = {'version':VERSION,'design_contract':'docs/design/rcci03_native_smoke_v1.md',
    'parent_request':'artifacts/rcci02_v2_cohort/rcci02_v2_full_v1/request.json',
    'parent_request_hash':'68b9f264739f07fd77df81cb2315eea7f058b542d184cf4d5db798a45e47ec59',
    'parent_report_sha256':'efe1c5eede5ad69552adacca3002f066ab93cdec18287161f7a201b88ab821f7',
    'parent_index_sha256':'4c1552e280ae66f724302241516dd958420d01717a94a63dc6d0034c9e336c92',
    'native_acceptance':'artifacts/rcci03_build/native_audit_v1/report.json',
    'native_acceptance_sha256':'afd5bd0fec517d368180a99ac254847d4ff4664ea4535232db8ac00410ee771a',
    'upstream_config':'RL-MPC-LaneMerging-master/configs/train_default_1.json',
    'arms':['baseline','raw_ordinary','raw_conditional','rcci_ordinary','rcci_conditional'],
    'replays':[{'id':'stream','simulator_seed':0,'jerks':[5.],'episodes':2,'step_cap':160,'require_terminal':True},
               {'id':'mixed','simulator_seed':100,'jerks':[-5.,-2.5,0.,2.5,5.],'episodes':1,'step_cap':40,'require_terminal':False}],
    'max_sessions':10,'max_episodes':15,'max_controlled_steps':1800,'max_simulation_steps':3315,
    'max_ensemble_forward_calls':1444,'max_member_forward_calls':4332,'worker_watchdog_s':600,
    'cache_observation_indices':[0,1,10], 'device':'cpu','torch_threads':1,
    'execution_contract':'simulation_blocking_exact_v1','training':False,'policy_loading':False,
    'model_updates':False,'formal_evaluation':False,'calibration_access':False,'test_access':False}


def native_identity(config):
    path = inside(ROOT,config['native_acceptance']); r = read(path)
    if (sha(path) != config['native_acceptance_sha256'] or not r['import_path_exact']
            or r['status'] != 'complete_with_build_stdout_capture_gap'
            or sha(inside(ROOT,r['extension'])) != r['extension_sha256']
            or sha(ROOT/'RL-MPC-LaneMerging-master/st_cy.c') != r['generated_c_sha256']):
        raise ValueError('Local native extension/acceptance changed')
    files = {path.relative_to(ROOT).as_posix():sha(path),r['extension']:r['extension_sha256']}
    for name,h in r['inputs'].items():
        relative = 'RL-MPC-LaneMerging-master/'+name
        if sha(inside(ROOT,relative)) != h: raise ValueError('Native build source changed')
        files[relative] = h
    old = ROOT/'artifacts/rcci03_build/native_v1'
    if sha(old/'report.json') != r['build_report_sha256'] or sha(old/'build_log.json') != r['build_log_sha256']:
        raise ValueError('Original build evidence changed')
    files.update({p.relative_to(ROOT).as_posix():sha(p) for p in (old/'report.json',old/'build_log.json')})
    binary = shutil.which('sumo')
    if binary is None: raise ValueError('SUMO executable unavailable')
    version = subprocess.check_output([binary,'--version'],text=True,encoding='utf-8').splitlines()[0]
    if version != 'Eclipse SUMO sumo Version 1.22.0': raise ValueError('Frozen SUMO version changed')
    return {'files':files,'extension':r['extension'],'extension_sha256':r['extension_sha256'],
        'sumo_executable':str(Path(binary).resolve()),'sumo_sha256':sha(binary),'sumo_version':version,
        'build_stdout_capture_complete':False,'build_gap_explicitly_recorded':True}


def parent_inputs(config, baseline):
    if config != SETTINGS: raise ValueError('Only frozen bounded smoke configuration accepted')
    path = inside(ROOT,config['parent_request']); q = read(path); out = path.parent
    report = read(out/'report.json'); receipt = read(out/'receipt.json')
    current = shared.sources()
    if (digest(q) != config['parent_request_hash'] or report['status'] != 'complete'
            or receipt['request_hash'] != digest(q) or report['request_hash'] != digest(q)
            or sha(out/'report.json') != config['parent_report_sha256']
            or receipt['report_sha256'] != config['parent_report_sha256']
            or sha(out/'index.json') != config['parent_index_sha256']
            or receipt['index_sha256'] != config['parent_index_sha256']
            or report['index_sha256'] != config['parent_index_sha256']
            or q['adapter_contract'] != contract() or q['environment'] != shared.environment()
            or any(current.get(name) != h for name,h in q['source_hashes'].items())):
        raise ValueError('Accepted offline parent or unchanged implementation differs')
    if Path(baseline).resolve() != Path(q['baseline_root']).resolve(): raise ValueError('Do not replace baseline')
    roots,review,hashes,parents,accepted,training,episodes = cohort.prerequisites(q['config'],baseline)
    if (roots != q['roots'] or hashes != q['input_hashes'] or parents != q['parent_artifact_hashes']
            or accepted != q['training_acceptance'] or digest(review['manifest']) != q['manifest_hash']
            or episodes != q['design_episodes']):
        raise ValueError('Parent input/model identity changed')
    index = read(out/'index.json')
    if index['root_ids'] != [r['root_id'] for r in roots]: raise ValueError('Full root order changed')
    bound = {p.relative_to(ROOT).as_posix():sha(p) for p in (path,out/'report.json',out/'receipt.json',out/'index.json')}
    for name,h in index['root_files'].items():
        p = inside(out,name)
        if sha(p) != h: raise ValueError('Completed root result changed')
        bound[p.relative_to(ROOT).as_posix()] = h
    return q,accepted,training,bound


def upstream_files(baseline):
    files = {}
    for name in shared.git(ROOT,'ls-files','RL-MPC-LaneMerging-master').splitlines():
        if Path(name).suffix not in ('.py','.pyx','.xml','.sumocfg','.json'): continue
        verify_historical_text_checkout(ROOT,name,sha(inside(baseline,name)),shared.BASE)
        files[name] = sha(inside(ROOT,name))
    if not files: raise ValueError('Missing fixed upstream files')
    return files


def prepare(args):
    shared.clean(); cp = inside(ROOT,args.config); config = read(cp)
    parent,accepted,training,bound = parent_inputs(config,args.baseline_root)
    native = native_identity(config); upstream = upstream_files(Path(args.baseline_root))
    if not re.fullmatch('[A-Za-z0-9_-]{1,24}',args.run_id): raise ValueError('Use fresh short run ID')
    out = ROOT/'artifacts/rcci03'/args.run_id; out.mkdir(parents=True,exist_ok=False)
    assets = {**accepted['input_hashes'],cohort.shared.CONFIG['predictor_request']:
              sha(inside(Path(args.baseline_root),cohort.shared.CONFIG['predictor_request']))}
    q = {**shared.metadata(),'version':VERSION,'config':config,'config_path':cp.relative_to(ROOT).as_posix(),
        'config_sha256':sha(cp),'design_contract_sha256':sha(inside(ROOT,config['design_contract'])),
        'source_hashes':shared.sources(),'environment':shared.environment(),'adapter_contract':contract(),
        'baseline_root':parent['baseline_root'],'baseline_commit':shared.BASE,'model_assets':assets,
        'training_acceptance':accepted,'training_request':training,'parent_artifact_hashes':bound,
        'native_identity':native,'upstream_files':upstream,'optimizer_seed':None}
    write_once(out/'request.json',q)
    write_once(out/'preparation.json',{'request_hash':digest(q),'status':'prepared_not_executed',
        'max_sessions':10,'max_controlled_steps':1800,'max_simulation_steps':3315,
        'training_permitted':False,'policy_loading_permitted':False})
    print('rcci03_request='+str(out/'request.json')); print('confirm_request_hash='+digest(q))
    return out/'request.json',digest(q)


def check_request(path,expected):
    shared.clean(); path = inside(ROOT,path); q = read(path)
    if (path.parent.parent != ROOT/'artifacts/rcci03' or digest(q) != expected or q['version'] != VERSION
            or q['config'] != SETTINGS or q['source_hashes'] != shared.sources() or q['environment'] != shared.environment()
            or q['git_commit'] != shared.git(ROOT,'rev-parse','HEAD') or q['adapter_contract'] != contract()
            or read(path.parent/'preparation.json')['request_hash'] != expected
            or sha(inside(ROOT,q['config_path'])) != q['config_sha256']
            or sha(inside(ROOT,q['config']['design_contract'])) != q['design_contract_sha256']
            or native_identity(q['config']) != q['native_identity']):
        raise ValueError('Native smoke request/source/environment/binary changed')
    baseline = Path(q['baseline_root'])
    if shared.git(baseline,'rev-parse','HEAD') != shared.BASE or shared.git(baseline,'status','--porcelain'):
        raise ValueError('Need unchanged clean baseline')
    for names,base in ((q['upstream_files'],ROOT),(q['model_assets'],baseline)):
        for name,h in names.items():
            if sha(inside(base,name)) != h: raise ValueError('Native input/asset changed')
    return path.parent,q


def rng_hash():
    state = np.random.get_state()
    return digest({'python':repr(random.getstate()),'numpy':[state[0],state[1].tolist(),*state[2:]],
                   'torch':torch.get_rng_state().tolist()})


def tensor_hash(model):
    h = hashlib.sha256()
    for name,value in sorted(model.state_dict().items()):
        h.update(name.encode()); h.update(str(value.dtype).encode()); h.update(str(tuple(value.shape)).encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


class ForwardRecorder(torch.nn.Module):
    """Smoke-only transparent cache; one original frozen call, no new inference."""
    def __init__(self,model):
        super().__init__(); self.inner=model; self.config=model.config; self.mode=model.mode
        self.calls=0; self.last=None; self.eval(); self.requires_grad_(False)
    def forward(self,*args,**kwargs):
        self.calls += 1; self.last=None
        result = self.inner(*args,**kwargs)
        self.last=result['mean'][0].detach().cpu().numpy().copy()
        return result


def worker(path,expected,replay_id,arm):
    out,q = check_request(path,expected)
    if arm not in SETTINGS['arms']: raise ValueError('Unknown smoke arm')
    replay = next((r for r in SETTINGS['replays'] if r['id'] == replay_id),None)
    if replay is None: raise ValueError('Unknown frozen replay')
    directory = out/(replay_id+'_'+arm); directory.mkdir(exist_ok=False)
    report = {**shared.metadata(),'status':'running','request_hash':expected,'replay':replay,'arm':arm,
        'policy_loaded':False,'training_performed':False,'method_effect_established':False}
    write_once(directory/'started.json',report); previous=torch.get_num_threads(); torch.set_num_threads(1)
    try:
        provider=recorded=model=None; before_hash=None
        source=ROOT/'RL-MPC-LaneMerging-master'; network=source/'merge.net.xml'
        cfg=PredictorConfig.from_dict(q['training_request']['model'])
        if arm != 'baseline':
            mode=arm.split('_',1)[1]; entry=q['training_acceptance']['ensembles'][mode]
            model=load_trained_ensemble(inside(Path(q['baseline_root']),entry['path']),q['training_request'],mode,entry['sha256'],entry['selected'])
            before_hash=tensor_hash(model); recorded=ForwardRecorder(model)
            provider=(FrozenRCCIV2Features(network,cfg,recorded) if arm.startswith('rcci_') else
                      FrozenProbeFeatures(LaneConstantVelocity(network),cfg,recorded))
        features=[]; traces=[]; latencies=[]; saved=[]; current_frame=None
        with upstream_session(source,ROOT/SETTINGS['upstream_config'],replay['simulator_seed']) as settings:
            import traci,control,st_cy
            if Path(st_cy.__file__).resolve() != inside(ROOT,q['native_identity']['extension']):
                raise ValueError('Foreign native extension loaded')
            def reader():
                nonlocal current_frame
                clock=traci.simulation.getTime(); current_frame=read_extended_traffic()
                if clock != current_frame['simulation_time_s'] or clock != traci.simulation.getTime():
                    raise AssertionError('Traffic reading advanced simulator')
                return current_frame
            base=AuditedJerkEnv()
            env=(RCCIV2FeatureEnv(base,arm.split('_',1)[1],provider,reader) if arm.startswith('rcci_') else
                 PredictionFeatureEnv(base,'baseline' if arm == 'baseline' else arm.split('_',1)[1],provider,reader if provider else None))
            if base.raw.wait_before_start != 20 or settings.TICK_LENGTH != .2 or not settings.USE_CYTHON:
                raise ValueError('Original warmup/step/native execution contract changed')
            if env.base_dimension != 20: raise ValueError('Original observation layout changed')
            write_once(directory/'resolved_config.json',settings.export_settings())
            def observe(obs,info,episode,index,terminal):
                meta=info.pop('prediction_features'); latencies.append(meta['latency_s'])
                if obs.dtype != np.float32 or obs.shape != ((20,) if arm == 'baseline' else (169,)):
                    raise AssertionError('Wrong observation dtype/dimension')
                block=obs[20:]
                if terminal:
                    if block.any(): raise AssertionError('Terminal block is not zero')
                elif provider is not None:
                    if current_frame['simulation_time_s'] != traci.simulation.getTime():
                        raise AssertionError('Prediction advanced simulator')
                    history=list(env.history)
                    if len(history) != min(index+1,11): raise AssertionError('Wrong reset/history support')
                    if arm.startswith('rcci_'):
                        cache=provider.last
                        if cache is None or not np.array_equal(cache['neighbors'],recorded.last): raise AssertionError('Wrong forward cache')
                        expected_features=summarize_road_consequences_v2(history,cache['neighbors'],cache['ego'],provider.road)['features']
                        ego=cache['ego']
                    else:
                        inputs=build_actor_inputs(history,provider.geometry.geometry)
                        ego=rollout_ego(history[-1],probe_plans(1,cfg)[0],provider.geometry).numpy()
                        expected_features=raw_summary(inputs,torch.from_numpy(recorded.last),torch.from_numpy(ego))
                    if not np.array_equal(block,expected_features): raise AssertionError('Cached offline feature differs')
                    if index in SETTINGS['cache_observation_indices']:
                        saved.append({'episode':episode,'observation_index':index,'history':deepcopy(history),
                            'mean':recorded.last.tolist(),'ego':ego.tolist(),'features':block.tolist()})
                features.append({'episode':episode,'observation_index':index,'terminal':terminal,
                    'features':block.tolist(),'meta':meta})
            try:
                for episode in range(replay['episodes']):
                    calls_before=0 if recorded is None else recorded.calls
                    obs,info=env.reset(); observe(obs,info,episode,0,False)
                    if recorded is not None and recorded.calls != calls_before+1: raise AssertionError('Reset not one forward')
                    trace={'reset_observation':obs[:20].tolist(),'reset_info':info,'reset_rng':rng_hash(),
                        'reset_delay':float(control.delay),'steps':[]}; traces.append(trace)
                    done=False
                    for step in range(replay['step_cap']):
                        action=np.array([replay['jerks'][step%len(replay['jerks'])]],np.float32)
                        calls_before=0 if recorded is None else recorded.calls
                        obs,reward,done,info=env.step(action); observe(obs,info,episode,step+1,done)
                        if recorded is not None and recorded.calls != calls_before+int(not done):
                            raise AssertionError('Terminal/nonterminal forward contract changed')
                        trace['steps'].append({'action':action.tolist(),'observation':obs[:20].tolist(),
                            'reward':float(reward),'done':bool(done),'info':info,'traffic':traffic_snapshot(),
                            'delay':float(control.delay),'rng':rng_hash()})
                        if done: break
                    if replay['require_terminal'] and not done: raise AssertionError('Terminal required within frozen cap')
                ticks=round(traci.simulation.getTime()/.2)
                if abs(ticks*.2-traci.simulation.getTime()) > 1e-8: raise AssertionError('Unexpected simulation grid')
            finally: env.close()
        calls=0 if recorded is None else recorded.calls
        cap=320 if replay_id == 'stream' else 41
        if calls > cap: raise AssertionError('Forward budget exceeded')
        unchanged=model is None or (tensor_hash(model) == before_hash and all(p.grad is None and not p.requires_grad for p in model.parameters()))
        if not unchanged: raise AssertionError('Frozen predictor changed')
        write_once(directory/'traces.json',traces); write_once(directory/'features.json',features); write_once(directory/'cached_examples.json',saved)
        report.update(status='complete',actual_controlled_steps=sum(len(t['steps']) for t in traces),
            actual_simulation_steps=ticks,episodes=len(traces),ensemble_forward_calls=calls,member_forward_calls=calls*3,
            cached_offline_features_exact=True,predictor_unchanged=True,history_clock_terminal_checks=True,
            latency_s={'count':len(latencies),'mean':float(np.mean(latencies)),'max':max(latencies)},
            files={p.name:sha(p) for p in (directory/'traces.json',directory/'features.json',directory/'cached_examples.json',directory/'resolved_config.json')})
    except BaseException:
        report.update(status='failed',failure_reason=traceback.format_exc()); raise
    finally:
        report['finished_at']=shared.now(); write_once(directory/'report.json',report); torch.set_num_threads(previous)


def run(path,expected):
    out,q=check_request(path,expected)
    if any((out/n).exists() for n in ('started.json','report.json','receipt.json')):
        raise ValueError('Never overwrite/retry a started native smoke')
    with run_lock(out):
        report={**shared.metadata(),'status':'running','request_hash':expected,'training_performed':False,
            'policy_loaded':False,'method_effect_established':False,'online_training_released':False}
        write_once(out/'started.json',report)
        try:
            jobs={}; checks={}; files={}
            for replay in SETTINGS['replays']:
                reference=None
                for arm in SETTINGS['arms']:
                    key=replay['id']+'_'+arm; print('[rcci03] '+key,flush=True)
                    command=[sys.executable,'-B',str(Path(__file__).resolve()),'worker','--request',str(Path(path).resolve()),
                             '--confirm-request-hash',expected,'--replay',replay['id'],'--arm',arm]
                    with (out/(key+'.log')).open('xb') as log:
                        subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=SETTINGS['worker_watchdog_s'])
                    d=out/key; child=read(d/'report.json'); trace=read(d/'traces.json')
                    if child['status'] != 'complete' or child['request_hash'] != expected: raise ValueError('Child identity/status differs')
                    for name,h in child['files'].items():
                        if sha(d/name) != h: raise ValueError('Child output changed')
                    if reference is None: reference=trace
                    else: checks[key+'_exact_base_reward_traffic_rng_parity']=trace == reference
                    jobs[key]=child
                    files.update({p.relative_to(out).as_posix():sha(p) for p in d.iterdir() if p.is_file()})
                    files[key+'.log']=sha(out/(key+'.log'))
            totals={key:sum(j[key] for j in jobs.values()) for key in
                ('actual_controlled_steps','actual_simulation_steps','episodes','ensemble_forward_calls','member_forward_calls')}
            limits={'actual_controlled_steps':1800,'actual_simulation_steps':3315,'episodes':15,
                    'ensemble_forward_calls':1444,'member_forward_calls':4332}
            if len(jobs) != 10 or any(totals[k] > v for k,v in limits.items()) or not all(checks.values()):
                raise AssertionError('Native parity or fixed budget failed')
            check_request(path,expected)
            # Recheck completed evidence and baseline asset identity after all sessions.
            for name,h in q['parent_artifact_hashes'].items():
                if sha(inside(ROOT,name)) != h: raise ValueError('Parent evidence changed during smoke')
            parent_inputs(SETTINGS,q['baseline_root'])
            write_once(out/'index.json',{'files':files})
            report.update(status='complete',engineering_gate=True,checks=checks,jobs=jobs,totals=totals,
                index_sha256=sha(out/'index.json'),next_stage='RCCI04_reuse_and_training_design_requires_separate_approval')
        except BaseException:
            report.update(status='failed',engineering_gate=False,failure_reason=traceback.format_exc()); raise
        finally:
            report['finished_at']=shared.now(); write_once(out/'report.json',report)
    write_once(out/'receipt.json',{'request_hash':expected,'report_sha256':sha(out/'report.json'),'index_sha256':sha(out/'index.json')})
    print('rcci03_report='+str(out/'report.json'))


def main():
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('prepare'); a.add_argument('--baseline-root',required=True); a.add_argument('--config',default=CONFIG_PATH); a.add_argument('--run-id',required=True)
    for name in ('run','worker'):
        a=sub.add_parser(name); a.add_argument('--request',required=True); a.add_argument('--confirm-request-hash',required=True)
        if name == 'worker': a.add_argument('--replay',required=True); a.add_argument('--arm',required=True)
    args=p.parse_args()
    if args.command == 'prepare': prepare(args)
    elif args.command == 'run': run(args.request,args.confirm_request_hash)
    else: worker(args.request,args.confirm_request_hash,args.replay,args.arm)


if __name__ == '__main__': main()
