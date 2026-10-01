"""P7b independent recording request; parent P7 remains immutable, no training."""
import argparse
from collections import defaultdict
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

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
import explore_feature_ddpg as parent_tools
import train_response_predictors as shared
import smoke_feature_ddpg as smoke
from prediction_rl.data.collection_store import read_json as read,write_once,file_hash as sha,verify_episode,seal_episode,run_lock
from prediction_rl.data.dataset_contract import digest
from prediction_rl.data.frozen_dataset import inside,verify_historical_inputs
from prediction_rl.data.actor_adapter import build_actor_inputs
from prediction_rl.data.merge_geometry import read_extended_traffic
from prediction_rl.prediction.kinematic_reference import LaneConstantVelocity
from prediction_rl.evaluation.exploration import PROTOCOL as P7,aggregate as p7_aggregate,zero_channel_query
from prediction_rl.evaluation.closed_loop_diagnostics import (
    PROTOCOL,AUDIT,validate_protocol,select_jobs,CapturedProvider,Recorder,cohort,
    physical_values,ranges,vector_ranges,summarize_trace,assert_parity)
from prediction_rl.features.probe_features import NAMES
from prediction_rl.training.all_ddpg import preset_factory,checkpoint_contract,load_evaluation_weights
from prediction_rl.envs.upstream import upstream_session


def load_parent(config):
    validate_protocol(config);path=inside(ROOT,config['parent_request']);q=read(path)
    if (digest(q)!=config['parent_request_hash'] or read(path.parent/'preparation.json')['request_hash']!=digest(q)
            or q['version']!='p07_request_v1' or digest(q['protocol'])!=digest(P7)
            or sha(path.parent/'aggregate.json')!=config['parent_aggregate_sha256']):
        raise ValueError('Frozen P7 request/aggregate changed')
    if (path.parent/'writer.lock').exists():raise ValueError('P7 parent run is active or has an unresolved lock')
    # New audit files may be added, but NONE of the old executable files change.
    verify_historical_inputs(ROOT,{'python_lf:'+k:v for k,v in q['source_hashes'].items()},q['git_commit'])
    current=shared.source_hashes()
    if (any(current.get(k)!=v for k,v in q['source_hashes'].items())
            or q['upstream_hashes']!=parent_tools.upstream_hashes()
            or q['dependencies']!=smoke.dependencies() or q['environment']!=shared.env()
            or q['runtime_identity']!=parent_tools.runtime_identity()
            or q['upstream_config_sha256']!=sha(parent_tools.UPSTREAM)):
        raise ValueError('Original P7 implementation/runtime changed; no permissive loading')
    parent_tools.prerequisites(q['protocol'])
    trained={}
    for j in q['train_jobs']:
        directory=path.parent/'train'/j['id'];m=verify_episode(directory,parent_tools.binding(q,'train',j))['metadata']
        if sha(directory/'final.pt')!=m['weights_sha256']:raise ValueError('Parent weights changed')
        trained[j['id']]=m
    rows=[]
    for j in q['evaluation_jobs']:
        directory=path.parent/'evaluate'/j['id'];m=verify_episode(directory,parent_tools.binding(q,'evaluate',j))['metadata']
        row=read(directory/'episode.json')
        if (row!=m['outcome'] or m['no_policy_updates']is not True
                or row['policy_sha256']!=trained[j['training_id']]['weights_sha256']):
            raise ValueError('Parent evaluation lineage changed')
        rows.append(row)
    a=read(path.parent/'aggregate.json');recomputed=p7_aggregate(q['protocol'],rows)
    if a['request_hash']!=digest(q) or any(a[k]!=v for k,v in recomputed.items()):
        raise ValueError('Parent aggregate not reproducible')
    return path,q


def observed_train_histories(parent):
    """Read observed history.json only. Future label files are NOT parsed."""
    p=parent['predictor_training_request'];rp=inside(ROOT,p['dataset_review'])
    if sha(rp)!=p['dataset_review_sha256']:raise ValueError('Dataset review changed')
    r=read(rp)
    if r['test_locked']is not True:raise ValueError('Need locked test')
    values=[];hashes={}
    for row in r['roots']['train']:
        if not row['eligible_trajectory_root']:continue  # Frozen training eligibility, not new outcome filtering.
        hp=inside(ROOT,row['history'])
        if sha(hp)!=row['history_sha256']:raise ValueError('Observed training history changed')
        h=read(hp)
        if h['split']!='train' or h['root_id']!=row['root_id'] or h['episode_id']!=row['episode_id']:
            raise ValueError('Wrong training root')
        values.append(h);hashes[str(hp.relative_to(ROOT))]=row['history_sha256']
    if not values or len({h['root_id'] for h in values})!=len(values):raise ValueError('Missing/duplicate observed roots')
    return values,hashes


def make_reference(parent):
    histories,hashes=observed_train_histories(parent)
    geometry=LaneConstantVelocity(parent_tools.SOURCE/'merge.net.xml')
    physical=defaultdict(list);raw={'ordinary':[],'conditional':[]}
    for h in histories:
        inputs=build_actor_inputs(h['history'],geometry.geometry)
        if digest(inputs)!=digest(h['inputs']):raise ValueError('Original observed input encoding changed')
        for k,v in physical_values(inputs).items():physical[k].extend(v)
    for arm in raw:
        provider,_=parent_tools.provider_for(parent,arm);before=smoke.tensor_hash(provider.model.state_dict())
        for i,h in enumerate(histories):
            f=provider(h['history']);lane=h['history'][-1]['vehicles']['ego']['lane_id'];frames=len(h['history'])
            raw[arm].append({'root_id':h['root_id'],'episode_id':h['episode_id'],
                            'cohort':cohort(lane,frames),'features':f.tolist()})
            if (i+1)%128==0:print(f'[p7b reference] {arm} roots={i+1}/{len(histories)}',flush=True)
        if before!=smoke.tensor_hash(provider.model.state_dict()):raise ValueError('Reference changed predictor')
    result={'version':'p07b_observed_train_reference_v1','eligible_roots':len(histories),
            'feature_names':list(NAMES),'physical':{k:ranges(v) for k,v in physical.items()},
            'models':{a:{'global':vector_ranges(v),
                        'cohorts':{k:vector_ranges([r for r in v if r['cohort']==k]) for k in sorted({r['cohort'] for r in v})}}
                      for a,v in raw.items()},
            'history_hashes':hashes,'future_labels_parsed':False,'test_opened':False}
    return result,raw


def prepare(config,run_id):
    shared.clean();validate_protocol(config);path,parent=load_parent(config)
    out=shared.output_dir('p7b',run_id);started=shared.metadata();write_once(out/'started.json',started)
    try:
        torch.set_num_threads(config['torch_threads'])
        reference,raw=make_reference(parent)
        write_once(out/'reference.json',reference);write_once(out/'reference_rows.json',raw)
        q={**started,'version':'p07b_request_v1','protocol':deepcopy(config),'jobs':select_jobs(config,parent),
           'source_hashes':shared.source_hashes(),'environment':shared.env(),
           'parent_request_hash':digest(parent),'reference_sha256':sha(out/'reference.json'),
           'reference_rows_sha256':sha(out/'reference_rows.json')}
        write_once(out/'request.json',q);write_once(out/'preparation.json',{'request_hash':digest(q),'status':'prepared_not_simulated'})
    except BaseException:
        write_once(out/'failure.json',{**started,'failure_reason':traceback.format_exc()});raise
    print('diagnostic_request='+str(out/'request.json'),flush=True)
    print('confirm_request_hash='+digest(q),flush=True);return out/'request.json',digest(q)


def load_request(path,expected_hash):
    shared.clean();path=inside(ROOT,path);q=read(path);validate_protocol(q['protocol'])
    if (q['version']!='p07b_request_v1' or digest(q)!=expected_hash
            or read(path.parent/'preparation.json')['request_hash']!=expected_hash
            or q['source_hashes']!=shared.source_hashes() or q['environment']!=shared.env()
            or sha(path.parent/'reference.json')!=q['reference_sha256']
            or sha(path.parent/'reference_rows.json')!=q['reference_rows_sha256']):
        raise ValueError('Diagnostic request/source/reference changed')
    pp,parent=load_parent(q['protocol'])
    if q['jobs']!=select_jobs(q['protocol'],parent) or q['parent_request_hash']!=digest(parent):
        raise ValueError('Diagnostic roster/parent changed')
    reference=read(path.parent/'reference.json')
    if reference['future_labels_parsed']is not False or reference['feature_names']!=list(NAMES):
        raise ValueError('Invalid observed-only reference')
    for name,h in reference['history_hashes'].items():
        if sha(inside(ROOT,name))!=h:raise ValueError('Reference history changed')
    return path,q,pp,parent,reference


def binding(q,job):return {'request_hash':digest(q),'job':job,'stage':'recording'}


def run_episode(parent_path,parent,q,job,reference,out):
    train=next(j for j in parent['train_jobs'] if j['id']==job['training_id'])
    td=parent_path.parent/'train'/train['id']
    meta=verify_episode(td,parent_tools.binding(parent,'train',train))['metadata']
    base,predictor_sha=parent_tools.provider_for(parent,job['arm'])
    provider=None if base is None else CapturedProvider(base)
    geometry=LaneConstantVelocity(parent_tools.SOURCE/'merge.net.xml').geometry
    recorder=Recorder(job['arm'],geometry,read_extended_traffic,provider)
    frozen=None if base is None else smoke.tensor_hash(base.model.state_dict())
    with upstream_session(parent_tools.SOURCE,parent_tools.UPSTREAM,job['simulator_seed']) as settings:
        env=parent_tools.environment(parent,job,provider,recorder.record)
        try:
            from all.logging import DummyWriter
            checked=preset_factory('cpu',settings.LEARNING_RATE)(env,DummyWriter())
            contract=checkpoint_contract(env,parent['preset'],predictor_sha,parent['upstream_config_sha256'])
            if contract!=meta['weights_contract']:raise ValueError('Diagnostic inputs changed')
            load_evaluation_weights(checked,td/'final.pt',contract,meta['weights_sha256'])
            initial={n:smoke.tensor_hash(a.model.state_dict()) for n,a in [('policy',checked.agent.policy),('q',checked.agent.q)]}
            write_once(out/'resolved_config.json',{'upstream':settings.export_settings(),'job':job,
                'protocol':q['protocol'],'parent_preset':parent['preset'],'feature_names':list(NAMES),
                'policy_sha256':meta['weights_sha256'],'predictor_sha256':predictor_sha})
            state=env.reset();reward=0.;actions=[];sensitivity=[]
            for step in range(500):
                before=torch.zeros(1,device=state.features.device) if checked.body.timestep is None else checked.body.timestep.clone()
                recorder.mark_policy_input(state,before)
                action=checked.eval(state,reward)
                zero_action=None if provider is None else zero_channel_query(checked,state,before)
                if zero_action is not None:sensitivity.append(float((action-zero_action).abs().item()))
                state,reward=env.step(action)
                actions.append({'step':step,'jerk':float(action.item()),'reward':float(reward),'done':bool(env.done),
                                'zero_channel_query_jerk':None if zero_action is None else float(zero_action.item())})
                if env.done:
                    recorder.mark_policy_input(state,checked.body.timestep.clone())
                    checked.eval(state,reward);break
            else:raise ValueError('Original limit not honored')
            if (len(checked.agent.replay_buffer)!=0 or any(a._optimizer.state for a in [checked.agent.policy,checked.agent.q])
                    or any(initial[n]!=smoke.tensor_hash(a.model.state_dict()) for n,a in [('policy',checked.agent.policy),('q',checked.agent.q)])):
                raise ValueError('Recording updated agent/replay')
            if base is not None and (frozen!=smoke.tensor_hash(base.model.state_dict())
                    or any(p.grad is not None or p.requires_grad for p in base.model.parameters())):
                raise ValueError('Recording changed predictor')
            row={**job,**recorder.metrics.result(),
                 'mean_abs_zero_channel_action_change':None if not sensitivity else float(np.mean(sensitivity)),
                 'policy_sha256':meta['weights_sha256'],'query_is_not_executed':True}
            original=parent_path.parent/'evaluate'/job['id']
            parity=assert_parity(row,actions,read(original/'episode.json'),read(original/'actions.json'))
            summary=summarize_trace(recorder.rows,job['arm'],reference,q['protocol']['thresholds_descriptive_only'])
            write_once(out/'observations.json',recorder.rows);write_once(out/'actions.json',actions)
            write_once(out/'episode.json',row);write_once(out/'diagnostics.json',summary)
            return {'job':job,'status':'complete','exact_parent_parity':parity,'outcome':row,'diagnostics':summary,
                    'observations_sha256':sha(out/'observations.json'),'actions_sha256':sha(out/'actions.json'),
                    'policy_sha256':meta['weights_sha256'],'no_policy_updates':True}
        finally:env.close()


def worker(args):
    path,q,pp,parent,reference=load_request(args.request,args.confirm_request_hash)
    job=next(j for j in q['jobs'] if j['id']==args.job);out=path.parent/'recording'/job['id'];out.mkdir(parents=True,exist_ok=False)
    started={**shared.metadata(),'binding':binding(q,job)};write_once(out/'started.json',started)
    torch.set_num_threads(q['protocol']['torch_threads']);start=time.perf_counter()
    try:
        result=run_episode(pp,parent,q,job,reference,out)
        result.update(provenance=started,elapsed_s=time.perf_counter()-start,finished_at=shared.now())
        write_once(out/'report.json',result);seal_episode(out,binding(q,job),result)
    except BaseException:
        write_once(out/'failure.json',{**started,'failure_reason':traceback.format_exc()});raise


def aggregate_request(path,expected_hash):
    path,q,_,_,_=load_request(path,expected_hash);out=path.parent
    with run_lock(out):
        reports=[verify_episode(out/'recording'/j['id'],binding(q,j))['metadata'] for j in q['jobs']]
        if any(r['exact_parent_parity']is not True or r['no_policy_updates']is not True for r in reports):
            raise ValueError('Invalid instrumentation')
        report={**shared.metadata(),'status':'complete','all_replays_exact':True,'request_hash':digest(q),
                'diagnostic_episodes':len(reports),'jobs':reports,'method_effect_gate':None,
                'thresholds_descriptive_only':q['protocol']['thresholds_descriptive_only'],
                'test_opened':False,'formal_claim':False,'finished_at':shared.now()}
        write_once(out/'summary.json',report)
        write_once(out/'summary_receipt.json',{'request_hash':digest(q),'summary_sha256':sha(out/'summary.json')})
    print('diagnostic_summary='+str(out/'summary.json'),flush=True);return out/'summary.json'


def execute(path,expected_hash,resume=False):
    path,q,_,_,_=load_request(path,expected_hash);out=path.parent
    invocation=out/'invocations'/uuid.uuid4().hex[:12];invocation.mkdir(parents=True,exist_ok=False)
    report={**shared.metadata(),'status':'running','jobs':[],'request_hash':digest(q)}
    with run_lock(out):
        try:
            pending=[]
            for j in q['jobs']:
                directory=out/'recording'/j['id']
                if directory.exists():
                    if not resume:raise ValueError('--resume only skips verified COMPLETE recordings')
                    verify_episode(directory,binding(q,j));report['jobs'].append({'id':j['id'],'action':'reuse'})
                else:pending.append(j)
            def child(j):
                command=[sys.executable,str(Path(__file__).resolve()),'worker','--request',str(path),
                         '--confirm-request-hash',digest(q),'--job',j['id']]
                print('[p7b] recording '+j['id'],flush=True)
                with (invocation/(j['id']+'.log')).open('w',encoding='utf-8') as log:
                    subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
                verify_episode(out/'recording'/j['id'],binding(q,j));return {'id':j['id'],'action':'run'}
            workers=q['protocol']['workers']
            with ThreadPoolExecutor(max_workers=workers) as pool:
                for i in range(0,len(pending),workers):
                    report['jobs'].extend(f.result() for f in [pool.submit(child,j) for j in pending[i:i+workers]])
            report.update(status='complete',finished_at=shared.now())
        except BaseException:
            report.update(status='failed',failure_reason=traceback.format_exc(),finished_at=shared.now());raise
        finally:write_once(invocation/'report.json',report)
    # Complete-only reruns verify an existing immutable summary, never overwrite it.
    if (out/'summary.json').exists():
        s=read(out/'summary.json')
        receipt=read(out/'summary_receipt.json')
        actual=[verify_episode(out/'recording'/j['id'],binding(q,j))['metadata'] for j in q['jobs']]
        if (s['request_hash']!=digest(q) or s['all_replays_exact']is not True or s['jobs']!=actual
                or receipt!={'request_hash':digest(q),'summary_sha256':sha(out/'summary.json')}):
            raise ValueError('Invalid prior summary')
        return out/'summary.json'
    return aggregate_request(path,expected_hash)


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    c=sub.add_parser('prepare');c.add_argument('--config',default='configs/development/p07b_closed_loop_diagnostics_v1.json');c.add_argument('--run-id',required=True)
    for name in ['run','aggregate','worker']:
        c=sub.add_parser(name);c.add_argument('--request',required=True);c.add_argument('--confirm-request-hash',required=True)
        if name=='run':c.add_argument('--resume',action='store_true')
        if name=='worker':c.add_argument('--job',required=True)
    c=sub.add_parser('audit');c.add_argument('--run-id',required=True)
    args=p.parse_args()
    if args.command=='prepare':prepare(read(inside(ROOT,args.config)),args.run_id)
    elif args.command=='worker':worker(args)
    elif args.command=='run':print('diagnostic_summary='+str(execute(args.request,args.confirm_request_hash,args.resume)))
    elif args.command=='aggregate':aggregate_request(args.request,args.confirm_request_hash)
    else:
        path,h=prepare(AUDIT,args.run_id);execute(path,h)


if __name__=='__main__':main()
