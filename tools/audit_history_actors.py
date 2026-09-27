"""Bounded replay of nine predeclared roots; no candidate rollout or training."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from audit_environment import sha, write
from prediction_rl.envs.upstream import upstream_session
from prediction_rl.data.branching import ReplayBrancher, fingerprint
from prediction_rl.data.reference_policy import AuthorDDPGReference
from prediction_rl.data.merge_geometry import MergeGeometry
from prediction_rl.data.actor_adapter import build_actor_inputs


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def load_config(path):
    config=read(path)
    keys={'schema_version','source_p3_report','source_p3_sha256','source_label_report','source_label_sha256',
          'network','network_sha256','simulator_seeds','history_steps','neighbor_capacity','tick_s','worker_wall_limit_s'}
    if not isinstance(config,dict) or set(config)!=keys:
        raise ValueError('Unknown/missing history pilot fields')
    for key,value in [('schema_version',1),('history_steps',11),('neighbor_capacity',12),('worker_wall_limit_s',300)]:
        if type(config[key]) is not int or config[key]!=value:raise ValueError('Frozen bounded pilot mismatch: '+key)
    if config['simulator_seeds']!=[0,1,100] or any(type(x)is not int for x in config['simulator_seeds']):
        raise ValueError('Do not replace development seeds')
    if type(config['tick_s']) not in (int,float) or config['tick_s']!=.2:
        raise ValueError('Original time grid required')
    for key,hash_key in [('source_p3_report','source_p3_sha256'),('source_label_report','source_label_sha256'),('network','network_sha256')]:
        path=Path(config[key])
        if path.is_absolute() or '..' in path.parts or not (ROOT/path).resolve().is_relative_to(ROOT):
            raise ValueError('Use project-relative inputs')
        if not isinstance(config[hash_key],str) or not re.fullmatch('[0-9a-f]{64}',config[hash_key]):
            raise ValueError('Pin input hashes')
        if sha(ROOT/path)!=config[hash_key]:raise ValueError('Pinned input changed: '+key)
    return config


def worker(config,seed,output):
    output.mkdir(parents=True,exist_ok=False)
    source=read(ROOT/config['source_p3_report'])
    label_path=ROOT/config['source_label_report']
    label_report=read(label_path)
    if label_report['source_report_sha256']!=config['source_p3_sha256'] or not label_report['label_contract_gate']:
        raise ValueError('Labels do not match accepted source')
    child=next(c for c in source['children'] if c['seed']==seed)
    if len(child['roots'])!=3 or any(r['status']!='complete' or r['prefix_steps']>250 for r in child['roots']):
        raise ValueError('Only the three accepted bounded roots may be replayed')
    geometry=MergeGeometry(ROOT/config['network'])
    report={'seed':seed,'status':'running','roots':[]}
    started=time.perf_counter()
    try:
        with upstream_session(ROOT/'RL-MPC-LaneMerging-master',ROOT/source['plan']['upstream_config'],seed) as settings:
            write(output/'resolved_upstream_config.json',settings.export_settings())
            reference=source['plan']['reference_policy']
            # Match original P3 import/checkpoint initialization before frozen RNG.
            # Replay consumes SAVED actions; the policy is never called/trained here.
            bootstrap=AuthorDDPGReference(ROOT/reference['checkpoint'],reference['sha256'],'cuda' if settings.CUDA else 'cpu')
            manager=ReplayBrancher()
            try:
                for number,expected in enumerate(child['roots']):
                    root,history=manager.capture_with_history(expected['prefix_actions'],config['history_steps'])
                    geometry.verify_runtime()
                    if (root.signature!=expected['signature'] or root.prefix_trace_hash!=expected['prefix_trace_hash']
                            or root.traffic!=expected['root_traffic']):
                        raise AssertionError('Read-only history capture changed accepted P3 replay')
                    repeat,repeated_history=manager.capture_with_history(expected['prefix_actions'],config['history_steps'])
                    if repeat.signature!=root.signature or history!=repeated_history:
                        raise AssertionError('Repeated history replay differs')
                    inputs=build_actor_inputs(history,geometry,config['neighbor_capacity'],config['history_steps'],config['tick_s'])
                    if inputs!=build_actor_inputs(repeated_history,geometry,config['neighbor_capacity'],config['history_steps'],config['tick_s']):
                        raise AssertionError('Repeated actor encoding differs')
                    matched=[]
                    for name,expected_hash in label_report['output_sha256'].items():
                        if not re.fullmatch('r[0-9]{2}.json',name):continue
                        path=label_path.parent/name
                        if sha(path)!=expected_hash:raise ValueError('Changed label pack')
                        pack=read(path)
                        if pack['simulator_seed']==seed and pack['prefix_actions']==expected['prefix_actions']:
                            if pack['root_traffic']!=root.traffic or pack['split']!='development':
                                raise AssertionError('Label/root join mismatch')
                            matched.append((path,pack))
                    if len(matched)!=1:raise ValueError('Require exactly one same-root label pack')
                    label_file,label_pack=matched[0]
                    if len(label_pack['candidates'])!=5:raise ValueError('Incomplete candidate family')
                    payload={'schema_version':1,'episode_id':label_pack['episode_id'],'root_id':label_pack['root_id'],
                             'split':'development','history':history,'inputs':inputs,
                             'input_hash':fingerprint(inputs),'shared_by_candidates':[0,1,2,3,4],
                             'label_pack':str(label_file.relative_to(ROOT)), 'label_pack_sha256':sha(label_file),
                             'formal_training_ready':False}
                    path=output/f'r{number}.json';write(path,payload)
                    report['roots'].append({'target':expected['target']['id'],'prefix_steps':expected['prefix_steps'],
                        'root_id':label_pack['root_id'],'history_frames':len(history),'replay_exact':True,
                        'repeated_history_exact':True,'selected_neighbors':sum(inputs['actor_mask']),
                        'omitted_neighbors':len(inputs['root_omitted_ids']),
                        'active_summary_groups':sum(inputs['summary_mask']),
                        'input_hash':fingerprint(inputs),'artifact_sha256':sha(path)})
                    print(f"[p4b] seed={seed} target={expected['target']['id']} exact=true selected={sum(inputs['actor_mask'])} omitted={len(inputs['root_omitted_ids'])}",flush=True)
                report['status']='complete'
            finally:manager.close()
    except Exception as error:
        report.update(status='failed',failure_reason=f'{type(error).__name__}: {error}')
        raise
    finally:
        report['elapsed_s']=time.perf_counter()-started
        write(output/'report.json',report)


def run(args,config):
    if not re.fullmatch('[A-Za-z0-9_-]{1,40}',args.run_id):raise ValueError('Use a short unique run ID')
    git=lambda *a:subprocess.check_output(['git','-C',str(ROOT),*a],text=True,encoding='utf-8')
    if git('status','--porcelain').strip():raise ValueError('Pilot requires clean source commit')
    source=read(ROOT/config['source_p3_report'])
    if source['status']!='complete' or not source['engineering_gate'] or source['plan']['simulator_seeds']!=config['simulator_seeds']:
        raise ValueError('Require accepted P3 evidence and exact seed list')
    output=ROOT/'artifacts/p4'/args.run_id;output.mkdir(parents=True,exist_ok=False)
    report={'status':'running','git_commit':git('rev-parse','HEAD').strip(),'working_tree_dirty':False,
            'command':subprocess.list2cmdline([sys.executable,*sys.argv]),'started_at':datetime.now(timezone.utc).isoformat(),
            'execution_contract':'simulation_blocking_exact_v1','config_sha256':sha(Path(args.config)),
            'source_p3_sha256':config['source_p3_sha256'],'source_label_sha256':config['source_label_sha256'],
            'formal_training_ready':False,'children':[]}
    from diagnose_branches import environment_metadata
    report['environment_versions']=environment_metadata()
    write(output/'resolved_config.json',config)
    started=time.perf_counter()
    try:
        for seed in config['simulator_seeds']:
            print(f'[p4b] seed={seed} three roots, two prefix replays/root',flush=True)
            with (output/f'seed_{seed}.log').open('w',encoding='utf-8') as log:
                subprocess.run([sys.executable,str(Path(__file__).resolve()),'--config',str(Path(args.config).resolve()),
                                '--worker','--seed',str(seed),'--output',str(output/f'seed_{seed}')],
                               cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=config['worker_wall_limit_s'])
            report['children'].append(read(output/f'seed_{seed}/report.json'))
        roots=[r for child in report['children'] for r in child['roots']]
        if len(roots)!=9:raise AssertionError('Do not discard planned roots')
        report.update(status='complete',history_actor_gate=True,root_count=9,joined_candidate_count=45,
                      next_stage='P4c_predictor_interfaces')
    except Exception as error:
        report.update(status='failed',history_actor_gate=False,failure_reason=f'{type(error).__name__}: {error}')
        raise
    finally:
        report['elapsed_s']=time.perf_counter()-started;report['finished_at']=datetime.now(timezone.utc).isoformat()
        write(output/'report.json',report)
    print('p4_history_report='+str(output/'report.json'))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True)
    parser.add_argument('--run-id');parser.add_argument('--worker',action='store_true')
    parser.add_argument('--seed',type=int);parser.add_argument('--output')
    args=parser.parse_args();config=load_config(args.config)
    if args.worker:
        if args.seed not in config['simulator_seeds']:raise ValueError('Unexpected worker seed')
        worker(config,args.seed,Path(args.output))
    else:run(args,config)
