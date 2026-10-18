"""RCCI04c immutable user-run matched experiment; never resumes partial training."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
import re
import subprocess
import sys
import time
import traceback
from types import FunctionType, SimpleNamespace
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'),str(ROOT/'tools')]
import torch
import audit_rcci_feasibility as shared
import audit_rcci_raw_reuse as audit
import bridge_rcci_raw_runtime as bridge
import train_matched_ddpg as kernel
import train_fixed_budget_ddpg as old_budget
import smoke_feature_ddpg as smoke
from prediction_rl.data.collection_store import read_json as read, write_once, run_lock, seal_episode
from prediction_rl.data.dataset_contract import digest
from prediction_rl.data.frozen_dataset import inside
from prediction_rl.data.merge_geometry import read_extended_traffic
from prediction_rl.envs.upstream import AuditedJerkEnv
from prediction_rl.envs.rcci_features import RCCIV2FeatureEnv
from prediction_rl.prediction.interface import PredictorConfig
from prediction_rl.prediction.kinematic_reference import LaneConstantVelocity
from prediction_rl.prediction.trained_artifacts import load_trained_ensemble
from prediction_rl.features.probe_features import FrozenProbeFeatures
from prediction_rl.features.road_consequences_v2 import contract
from prediction_rl.training import all_ddpg as raw, rcci_ddpg as rcci
from prediction_rl.training.matched_learning_curve import Jsonl
from prediction_rl.evaluation.rcci_interface_coverage import InterfaceCoverage
from prediction_rl.evaluation.rcci_matched_policy import PROTOCOL, SMOKE, validate_protocol, jobs, aggregate

require, sha = audit.require, audit.sha
CONFIG = 'configs/development/rcci04c_matched_policy_v1.json'
SMOKE_CONFIG = 'configs/development/rcci04c_entry_smoke_v1.json'
DESIGN = 'docs/design/rcci04c_entry_contract_v1.md'
BRIDGE = 'artifacts/rcci04b/raw_bridge_v1'
BRIDGE_REPORT_SHA = '49d6418ac680792a0e0bb4d5de97d0297de13085f9e66fbd4c996988b85f4583'


def parent_inputs(baseline):
    legacy, assets, variants = bridge.parent_inputs(baseline)
    root = ROOT/BRIDGE; q, report, receipt = (read(root/n) for n in ('request.json','report.json','receipt.json'))
    current=shared.sources()
    require(sha(root/'report.json') == BRIDGE_REPORT_SHA and report['status']=='complete' and report['engineering_gate']
        and report['scoped_Raw_control_reuse_supported'] and report['request_hash']==digest(q)
        and receipt == {'request_hash':digest(q),'report_sha256':BRIDGE_REPORT_SHA,'index_sha256':sha(root/'index.json')}
        and all(current.get(k)==h for k,h in q['source_hashes'].items())
        and q['environment']==shared.environment() and q['binary_variants']==variants
        and not (root/'writer.lock').exists(), 'Accepted bridge changed/active')
    for n,h in read(root/'index.json')['files'].items(): require(sha(inside(root,n))==h,'Bridge output changed')
    for n in ('request.json','report.json','receipt.json','index.json'):
        assets['worktree'][(root/n).relative_to(ROOT).as_posix()] = sha(root/n)
    baseline = Path(baseline).resolve(); old = inside(baseline,audit.SETTINGS['request']).parent
    inventory = read(ROOT/'artifacts/rcci04a_raw/raw_reuse_v1/verified_inventory.json')
    models = read(old/'aggregate.json')['training']
    selected = {k:v for k,v in models.items() if k.startswith(('ordinary_','conditional_'))}
    for model,m in selected.items():
        d=old/'train'/model
        require(sha(d/'complete.json')==inventory['train/'+model]['receipt_sha256'],'Raw training receipt changed')
        for s in m['snapshots']:
            for name,h in ((s['weights_file'],s['weights_sha256']),
                           (f'n{s["requested_steps"]}.json',inventory['train/'+model]['files'][f'n{s["requested_steps"]}.json'])):
                p=d/name; require(sha(p)==h,'Raw node changed'); assets['baseline'][p.relative_to(baseline).as_posix()]=h
    return legacy,assets,variants['current'],selected,inventory


def acceptance(path):
    path=inside(ROOT,path); q=read(path); d=path.parent
    r,receipt=read(d/'aggregate.json'),read(d/'aggregate_receipt.json')
    require(q['protocol']==SMOKE and q['source_hashes']==shared.sources() and q['environment']==shared.environment()
        and q['design_sha256']==sha(ROOT/DESIGN) and r['request_hash']==digest(q) and r['engineering_complete']
        and r['status']=='complete' and r['method_effect_gate'] is None and not r['method_comparison_allowed']
        and receipt['request_hash']==digest(q) and receipt['aggregate_sha256']==sha(d/'aggregate.json')
        and not (d/'writer.lock').exists(), 'Need accepted entry smoke from identical implementation')
    for stage in ('train','evaluate'):
        for j in jobs(SMOKE,stage): audit.sealed(d/stage/j['id'],binding(q,stage,j))
    return {p.relative_to(ROOT).as_posix():sha(p) for p in (path,d/'aggregate.json',d/'aggregate_receipt.json')}


def prepare(config,baseline,run_id,entry_smoke=None):
    shared.clean(); torch.set_num_threads(1); cp=inside(ROOT,config); p=validate_protocol(read(cp))
    require(bool(re.fullmatch('[A-Za-z0-9_-]{1,24}',run_id)), 'Use fresh short run ID')
    approved={} if p==SMOKE else acceptance(entry_smoke) if entry_smoke else None
    require(approved is not None,'Formal prepare requires --entry-smoke-request')
    legacy,assets,binary,models,inventory=parent_inputs(baseline)
    q={**shared.metadata(),'version':'rcci04c_request_v1','protocol':p,'config_path':cp.relative_to(ROOT).as_posix(),
        'config_sha256':sha(cp),'design_sha256':sha(ROOT/DESIGN),'source_hashes':shared.sources(),
        'environment':shared.environment(),'baseline_root':str(Path(baseline).resolve()),'assets':assets,
        'binary':binary,'legacy_request':legacy,'raw_training':models,'raw_inventory':inventory,
        'predictors':legacy['predictors'],'predictor_training_request':legacy['predictor_training_request'],
        'upstream_config_sha256':legacy['upstream_config_sha256'],
        'preset':raw.resolved_preset('cpu',.0002,smoke=p==SMOKE),'entry_acceptance':approved,
        'train_jobs':jobs(p,'train'),'evaluation_jobs':jobs(p,'evaluate'),'raw_policy_retraining':False}
    out=ROOT/'artifacts/rcci04c'/run_id; out.mkdir(parents=True,exist_ok=False)
    write_once(out/'request.json',q)
    write_once(out/'preparation.json',{'request_hash':digest(q),'status':'prepared_not_trained',
        'new_training_instances':len(q['train_jobs']),'requested_training_steps':len(q['train_jobs'])*p['training']['requested_frames'],
        'evaluation_episodes':len(q['evaluation_jobs']),'full_training_started':False})
    print('rcci_matched_request='+str(out/'request.json')); print('confirm_request_hash='+digest(q))
    return out/'request.json',digest(q)


def binding(q,stage,j): return {'request_hash':digest(q),'stage':stage,'job':j}


def verify_raw_training(q):
    base=inside(Path(q['baseline_root']),audit.SETTINGS['request']).parent
    for arm in ('ordinary','conditional'):
        for seed in q['protocol']['training']['run_seeds']:
            model=f'{arm}_s{seed}'; d=base/'train'/model
            job={'id':model,'arm':arm,'run_seed':seed}
            receipt=audit.sealed(d,kernel.binding(q['legacy_request'],'train',job))
            require(sha(d/'complete.json')==q['raw_inventory']['train/'+model]['receipt_sha256']
                and receipt['metadata']==q['raw_training'][model],'Raw complete training evidence changed')


def load_request(path,expected):
    shared.clean(); torch.set_num_threads(1); path=inside(ROOT,path); q=read(path)
    p=validate_protocol(q['protocol'])
    require(path.parent.parent==ROOT/'artifacts/rcci04c' and digest(q)==expected and q['version']=='rcci04c_request_v1'
        and q['source_hashes']==shared.sources() and q['environment']==shared.environment()
        and q['config_sha256']==sha(inside(ROOT,q['config_path'])) and read(inside(ROOT,q['config_path']))==p
        and q['design_sha256']==sha(ROOT/DESIGN) and q['train_jobs']==jobs(p,'train') and q['evaluation_jobs']==jobs(p,'evaluate')
        and q['preset']==raw.resolved_preset('cpu',.0002,smoke=p==SMOKE)
        and read(path.parent/'preparation.json')['request_hash']==expected,'Matched source/config/request changed')
    # A documentation-only commit is allowed: Python sources/config/design are exact, both SHAs recorded.
    base=Path(q['baseline_root'])
    require(shared.git(base,'rev-parse','HEAD')==shared.BASE and not shared.git(base,'status','--porcelain'),'Baseline changed')
    for where,root in (('baseline',base),('worktree',ROOT)):
        for n,h in q['assets'][where].items(): require(sha(inside(root,n))==h,'Bound asset changed: '+n)
    for n,h in q['entry_acceptance'].items(): require(sha(inside(ROOT,n))==h,'Entry acceptance changed')
    return path.parent,q


def expected_contract(q,arm):
    env=SimpleNamespace(feature_arm=arm,state_space=SimpleNamespace(shape=(169,)),rcci_contract=contract())
    return rcci.checkpoint_contract(env,q['preset'],q['predictors'][arm.removeprefix('rcci_')]['sha256'],q['upstream_config_sha256'])


def training_metadata(q,root,*,publish=False):
    result={}; records={}
    for j in q['train_jobs']:
        d=root/'train'/j['id']; receipt=audit.sealed(d,binding(q,'train',j)); m=receipt['metadata']
        require(m['job']==j and m['from_scratch'] and m['predictor_frozen']
            and m['initialization_matches_legacy_expanded'] and m['coverage']['sample_calls']>0,
            'Missing from-scratch matched optimizer execution')
        require([s['requested_steps'] for s in m['snapshots']]==q['protocol']['training']['checkpoint_frames'],'Node roster changed')
        for s in m['snapshots']:
            kernel.snapshot_steps(s['actual_steps'],s['requested_steps'])
            require(read(d/f'n{s["requested_steps"]}.json')==s and sha(d/s['weights_file'])==s['weights_sha256']
                and s['weights_contract']==expected_contract(q,j['arm']),'RCCI snapshot/contract changed')
        result[j['id']]=m; records[j['id']]={'receipt_sha256':sha(d/'complete.json'),'metadata':m}
    index={'request_hash':digest(q),'models':records}
    if publish:
        write_once(root/'training_index.json',index)
        write_once(root/'training_index_receipt.json',{'request_hash':digest(q),'index_sha256':sha(root/'training_index.json')})
    elif (root/'training_index.json').exists(): require(read(root/'training_index.json')==index,'Frozen training index changed')
    return result


def fast_training_receipt(q,root,directory,b):
    """Workers hash used weights separately; full inventory is checked at stage boundaries."""
    directory=Path(directory); model=directory.name
    if model.startswith('rcci_'):
        index=read(root/'training_index.json'); rc=read(root/'training_index_receipt.json')
        require(index['request_hash']==digest(q) and rc=={'request_hash':digest(q),'index_sha256':sha(root/'training_index.json')},'Training index receipt changed')
        entry=index['models'][model]; expected_hash=entry['receipt_sha256']; metadata=entry['metadata']
        require(directory.resolve()==(root/'train'/model).resolve(),'Wrong new training directory')
    else:
        expected_hash=q['raw_inventory']['train/'+model]['receipt_sha256']; metadata=q['raw_training'][model]
        require(directory.resolve()==(inside(Path(q['baseline_root']),audit.SETTINGS['request']).parent/'train'/model).resolve(),'Wrong Raw directory')
    require(sha(directory/'complete.json')==expected_hash,'Training receipt changed')
    r=read(directory/'complete.json')
    require(r['binding']==b and r['metadata']==metadata and r['status']=='complete','Training metadata/binding changed')
    names=[]
    for f in directory.rglob('*'):
        require(not f.is_symlink() and f.resolve().is_relative_to(directory.resolve()),'Training path escape')
        if f.is_file() and f!=directory/'complete.json': names.append(f.relative_to(directory).as_posix())
    require(set(names)==set(r['files']),'Training file roster changed')
    return r


class Runtime:
    def __init__(self,q,j,stage,out,root):
        self.q,self.job,self.stage,self.out,self.root=q,j,stage,out,root
        self.profile=q['protocol']; self.small=self.profile==SMOKE
        self.providers=[]; self.guards=[]; self.hooks=[]; self.steps=0; self.policy_calls=0
        self.diag=InterfaceCoverage(); self.stream=Jsonl(out/'interface_events.jsonl.gz')
        self.initial_ok=False; self.started=time.perf_counter()
        self.source=ROOT/'RL-MPC-LaneMerging-master'
        self.config=inside(Path(q['baseline_root']),'RL-MPC-LaneMerging-master/configs/train_default_1.json')

    def provider_for(self,lq,arm):
        mode=arm.removeprefix('rcci_'); e=self.q['predictors'][mode]; t=self.q['predictor_training_request']
        model=load_trained_ensemble(inside(Path(self.q['baseline_root']),e['path']),t,mode,e['sha256'],e['selected'])
        p=(rcci.MatchedRCCIProvider(self.source/'merge.net.xml',PredictorConfig.from_dict(t['model']),model)
           if arm.startswith('rcci_') else FrozenProbeFeatures(LaneConstantVelocity(self.source/'merge.net.xml'),
                PredictorConfig.from_dict(t['model']),bridge.native.ForwardRecorder(model)))
        self.providers.append(p); return p,e['sha256']

    def environment(self,lq,j,p,recorder):
        def record(event,obs,reward,done,info):
            recorder(event,obs,reward,done,info)
            self.diag.add(obs,info)
            if not done:
                meta={k:v for k,v in info['prediction_features'].items()}
                self.stream.append({'event':event,'metadata':meta})
        if not j['arm'].startswith('rcci_'): return bridge.original.environment(lq,j,p,record)
        wrapped=RCCIV2FeatureEnv(AuditedJerkEnv(),j['arm'].removeprefix('rcci_'),p,read_extended_traffic)
        env=raw.make_all_environment(wrapped,'cpu',record); env.rcci_contract=contract()
        return env

    @contextmanager
    def session(self,*args):
        require(args[0]==self.source and args[1]==self.config and sha(self.config)==self.q['upstream_config_sha256'],'Unexpected runtime config')
        with kernel.upstream_session(*args) as settings:
            import control,st,st_cy
            require(Path(st_cy.__file__).resolve()==Path(self.q['binary']['path']).resolve() and sha(st_cy.__file__)==self.q['binary']['sha256'],'Foreign native module')
            original=control.step
            def counted():
                limit=2147 if self.stage=='train' else 602
                if self.small and self.steps>=limit: raise ValueError('Entry native step budget exceeded')
                self.steps+=1; return original()
            with bridge.NoSTCalls(st,st_cy) as guard:
                self.guards.append(guard); control.step=counted
                try: yield settings
                finally: control.step=original

    def factory(self,*args,**kwargs):
        factory=rcci.preset_factory if self.job['arm'].startswith('rcci_') else raw.preset_factory
        create=factory(*args,**kwargs)
        def build(env,writer):
            checked=create(env,writer)
            if self.stage=='train':
                initial={n:smoke.tensor_hash(a.model.state_dict()) for n,a in [('policy',checked.agent.policy),('q',checked.agent.q)]}
                model=self.job['id'].removeprefix('rcci_')
                require(initial==self.q['raw_training'][model]['initial_weight_sha256'],'Expanded initialization differs before training')
                self.initial_ok=True; write_once(self.out/'initialization.json',{'matches_legacy_expanded':True,'hashes':initial})
            def count(module,inputs):
                limit=1577 if self.stage=='train' else 501
                if self.small and self.policy_calls>=limit: raise ValueError('Entry policy forward cap exceeded')
                self.policy_calls+=1
            self.hooks.append(checked.agent.policy.model.register_forward_pre_hook(count))
            target=checked.agent.policy._target._target
            require(isinstance(target,torch.nn.Module),'Unexpected original policy target implementation')
            self.hooks.append(target.register_forward_pre_hook(count))
            return checked
        return build

    def save(self,checked,path,c):
        h=rcci.save_evaluation_weights(checked,path,c)
        write_once(Path(path).with_suffix('.interface.json'),self.diag.report())
        return h

    def function(self,fn):
        is_rcci=self.job['arm'].startswith('rcci_')
        namespace=SimpleNamespace(SOURCE=self.source,UPSTREAM=self.config,provider_for=self.provider_for,environment=self.environment)
        overrides={'original':namespace,'upstream_session':self.session,'preset_factory':self.factory,'SMOKE':SMOKE,
            'checkpoint_contract':rcci.checkpoint_contract if is_rcci else raw.checkpoint_contract,
            'load_evaluation_weights':rcci.load_evaluation_weights if is_rcci else raw.load_evaluation_weights,
            'save_evaluation_weights':self.save,
            'verify_episode':lambda d,b:fast_training_receipt(self.q,self.root,d,b)}
        value=FunctionType(fn.__code__,{**fn.__globals__,**overrides},fn.__name__,fn.__defaults__,fn.__closure__)
        value.__kwdefaults__=fn.__kwdefaults__; return value

    def close(self):
        self.stream.close()
        for h in self.hooks: h.remove()

    def summary(self):
        calls=sum(p.calls if isinstance(p,rcci.MatchedRCCIProvider) else p.model.calls for p in self.providers)
        return {'actual_simulation_steps':self.steps,'ensemble_forward_calls':calls,'member_forward_calls':calls*3,
            'policy_forward_calls':self.policy_calls,'st_calls':[n for g in self.guards for n in g.calls],
            'interface_coverage':self.diag.report()}


def worker(args):
    root,q=load_request(args.request,args.confirm_request_hash)
    plan=q['train_jobs' if args.stage=='train' else 'evaluation_jobs']; j=next(x for x in plan if x['id']==args.job)
    out=root/args.stage/j['id']; out.mkdir(parents=True,exist_ok=False)
    started={**shared.metadata(),'binding':binding(q,args.stage,j)}; write_once(out/'started.json',started)
    rt=None
    try:
        bridge.load_extension(q['binary']['path'],q['binary']['sha256'])
        rt=Runtime(q,j,args.stage,out,root)
        if args.stage=='train':
            result=rt.function(kernel.train_job)(q,j,out)
            result['initialization_matches_legacy_expanded']=rt.initial_ok
            require(rt.initial_ok and result['coverage']['sample_calls']>0,'No matched initialization/optimizer updates')
        else:
            is_raw=not j['arm'].startswith('rcci_')
            lq=q['legacy_request'] if is_raw else q
            trainroot=inside(Path(q['baseline_root']),audit.SETTINGS['request']).parent if is_raw else root
            result=rt.function(kernel.evaluation_job)(lq,j,out,trainroot)
            if is_raw:
                old=trainroot/'evaluate'/j['id']; expected=q['raw_inventory']['evaluate/'+j['id']]
                require(sha(old/'complete.json')==expected['receipt_sha256'],'Raw reference receipt changed')
                for n in ('episode.json','actions.json','states.json','raw_events.json','resolved_config.json'):
                    require(sha(old/n)==expected['files'][n] and read(out/n)==read(old/n),'Raw historical parity failed: '+n)
                result['exact_Raw_historical_parity']=True
        details=rt.summary(); require(not details['st_calls'],'Unexpected ST control')
        controlled=result['actual_steps'] if args.stage=='train' else result['outcome']['steps']
        require(details['ensemble_forward_calls']==controlled,'Terminal/reset predictor call mismatch')
        if q['protocol']==SMOKE:
            require(controlled<=(515 if args.stage=='train' else 500),'Entry controlled step cap exceeded')
            if args.stage=='train': require(result['training']['completed_episodes']<=16,'Entry episode cap exceeded')
        result.update(runtime_audit=details,provenance=started,finished_at=shared.now(),elapsed_s=time.perf_counter()-rt.started)
        write_once(out/'interface_summary.json',rt.diag.report()); rt.close(); rt=None
        write_once(out/'report.json',result); seal_episode(out,binding(q,args.stage,j),result)
    except BaseException:
        write_once(out/'failure.json',{'status':'failed','failure_reason':traceback.format_exc()}); raise
    finally:
        if rt is not None: rt.close()


def run(path,expected,stage,resume=False):
    require(stage in ('train','evaluate'),'Unknown stage')
    root,q=load_request(path,expected); p=q['protocol']
    with run_lock(root):
        if stage=='evaluate': training_metadata(q,root); verify_raw_training(q)
        invocation=root/'invocations'/uuid.uuid4().hex[:10]; invocation.mkdir(parents=True)
        report={**shared.metadata(),'request_hash':expected,'stage':stage,'jobs':[]}
        try:
            pending=[]
            for j in q['train_jobs' if stage=='train' else 'evaluation_jobs']:
                d=root/stage/j['id']
                if d.exists():
                    require(resume,'Use --resume only for complete whole jobs')
                    audit.sealed(d,binding(q,stage,j)); report['jobs'].append({'id':j['id'],'action':'reuse'})
                else: pending.append(j)
            def child(j):
                print('[rcci04c] '+stage+' '+j['id'],flush=True)
                cmd=[sys.executable,'-B',str(Path(__file__).resolve()),'worker','--request',str(Path(path).resolve()),
                     '--confirm-request-hash',expected,'--stage',stage,'--job',j['id']]
                with (invocation/(j['id']+'.log')).open('xb') as log:
                    subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=180 if p==SMOKE else None)
                audit.sealed(root/stage/j['id'],binding(q,stage,j)); return {'id':j['id'],'action':'run'}
            width=p['training' if stage=='train' else 'evaluation']['workers']
            with ThreadPoolExecutor(max_workers=width) as pool:
                for i in range(0,len(pending),width):
                    report['jobs'].extend(f.result() for f in [pool.submit(child,j) for j in pending[i:i+width]])
            if stage=='train': training_metadata(q,root,publish=not (root/'training_index.json').exists())
            else: training_metadata(q,root); verify_raw_training(q)
            load_request(path,expected); report.update(status='complete',finished_at=shared.now())
        except BaseException:
            report.update(status='failed',failure_reason=traceback.format_exc()); raise
        finally: write_once(invocation/'report.json',report)
    print('stage_report='+str(invocation/'report.json'))


def aggregate_run(path,expected):
    root,q=load_request(path,expected)
    with run_lock(root):
        trained=training_metadata(q,root); verify_raw_training(q); rows=[]; audits=[]
        for j in q['evaluation_jobs']:
            d=root/'evaluate'/j['id']; m=audit.sealed(d,binding(q,'evaluate',j))['metadata']; row=read(d/'episode.json')
            tm=trained[j['training_id']] if j['arm'].startswith('rcci_') else q['raw_training'][j['training_id']]
            snap=next(s for s in tm['snapshots'] if s['requested_steps']==j['checkpoint_frames'])
            require(row==m['outcome'] and row['policy_sha256']==snap['weights_sha256']
                and row['actual_training_steps']==snap['actual_steps']
                and all(row.get(k)==v for k,v in j.items()),'Outcome/checkpoint receipt changed')
            if not j['arm'].startswith('rcci_'): require(m['exact_Raw_historical_parity'],'Raw parity missing')
            rows.append(row); audits.append(m['runtime_audit'])
        for m in trained.values(): audits.append(m['runtime_audit'])
        totals={k:sum(a[k] for a in audits) for k in ('actual_simulation_steps','ensemble_forward_calls','member_forward_calls','policy_forward_calls')}
        if q['protocol']==SMOKE:
            controls=sum(m['actual_steps'] for m in trained.values())+sum(r['steps'] for r in rows)
            episodes=sum(m['training']['completed_episodes'] for m in trained.values())+len(rows)
            require(controls<=3030 and episodes<=36 and all(totals[k]<=v for k,v in zip(totals,(6702,3030,9090,5158))),'Entry total budget exceeded')
            totals.update(controlled_steps=controls,episodes=episodes)
        result={**aggregate(q['protocol'],rows),'request_hash':expected,'provenance':shared.metadata(),
            'runtime_totals':totals,'training':trained,'finished_at':shared.now(),
            'previous_policy_weights_used_for_training':False,'automatic_budget_extension':False}
        write_once(root/'training_curves.json',old_budget.curve_index(root,trained,q))
        write_once(root/'development_curves.json',{'label':'frozen_policy_Gym20_development_NOT_test','nodes':result.get('nodes',[])})
        write_once(root/'aggregate.json',result)
        write_once(root/'aggregate_receipt.json',{'request_hash':expected,'aggregate_sha256':sha(root/'aggregate.json')})
    print('rcci_matched_aggregate='+str(root/'aggregate.json'))


def main():
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('prepare'); a.add_argument('--config',default=CONFIG); a.add_argument('--baseline-root',required=True)
    a.add_argument('--run-id',required=True); a.add_argument('--entry-smoke-request')
    for name in ('run','worker','aggregate'):
        a=sub.add_parser(name); a.add_argument('--request',required=True); a.add_argument('--confirm-request-hash',required=True)
        if name!='aggregate': a.add_argument('--stage',choices=['train','evaluate'],required=True)
        if name=='worker': a.add_argument('--job',required=True)
        if name=='run': a.add_argument('--resume',action='store_true')
    a=sub.add_parser('audit'); a.add_argument('--baseline-root',required=True); a.add_argument('--run-id',required=True)
    args=p.parse_args()
    if args.command=='prepare': prepare(args.config,args.baseline_root,args.run_id,args.entry_smoke_request)
    elif args.command=='worker': worker(args)
    elif args.command=='run': run(args.request,args.confirm_request_hash,args.stage,args.resume)
    elif args.command=='aggregate': aggregate_run(args.request,args.confirm_request_hash)
    else:
        path,h=prepare(SMOKE_CONFIG,args.baseline_root,args.run_id)
        run(path,h,'train'); run(path,h,'evaluate'); aggregate_run(path,h)


if __name__=='__main__': main()
