"""P7h USER-run author DDPG/ST/RL+MPC raw terminal audit; no training."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
import subprocess
import sys
import traceback
import uuid

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'tools')]
import recover_terminal_events as recovery
from prediction_rl.data.collection_store import inventory
from prediction_rl.evaluation.external_terminal_audit import (
    PROTOCOL, AUDIT, AUTHOR, B0, B3, EXTERNAL, validate_protocol, roster, native_row, analyze, aggregate)

legacy = recovery.legacy
previous = legacy.previous
cross, external = previous.cross, previous.cross.external
shared, read, sha, digest = recovery.shared, recovery.read, recovery.sha, recovery.digest
inside, write_once, verify_episode = recovery.inside, recovery.write_once, recovery.verify_episode
seal_episode, run_lock = recovery.seal_episode, recovery.run_lock


@lru_cache(maxsize=1)
def parents():
    pp, p, ep, e, cp, cq, *_ = previous.parents()
    path = inside(ROOT, PROTOCOL['recovery_request']); q = read(path); out = path.parent
    recovery.require_design(q['recovery']); previous.historical_source(q)
    if ((out/'writer.lock').exists() or digest(q) != PROTOCOL['recovery_request_hash']
            or q['version'] != 'p07g_recovery_request_v1' or q['protocol'] != legacy.PROTOCOL
            or sha(out/'aggregate.json') != PROTOCOL['recovery_aggregate_sha256']
            or q['environment'] != shared.env() or q['runtime_identity'] != previous.p7.runtime_identity()
            or q['author_assets'] != external.assets()
            or read(out/'preparation.json')['request_hash'] != digest(q)
            or q['source_manifest_sha256'] != sha(out/'source_manifest.json')):
        raise ValueError('Completed P7g recovery changed or active')
    manifest = recovery.source_snapshot()
    if manifest != read(out/'source_manifest.json') or q['jobs'] != manifest['pending_jobs']:
        raise ValueError('P7g immutable source evidence changed')
    result = recovery.collect(q, manifest, out); saved = read(out/'aggregate.json')
    if (any(saved[k] != v for k, v in result.items()) or read(out/'aggregate_receipt.json') !=
            {'request_hash': digest(q), 'sha256': sha(out/'aggregate.json')}):
        raise ValueError('P7g aggregate/receipt not reproducible')
    return pp, p, ep, e, cp, cq, path, q, saved


def identity():
    pp,p,ep,e,cp,cq,gp,gq,_ = parents()
    return {label: {'path': str(path.relative_to(ROOT)), 'request_hash': digest(q),
                   'request_sha256': sha(path), 'aggregate_sha256': sha(path.parent/'aggregate.json')}
            for label,path,q in (('p7',pp,p),('p7d',ep,e),('p7e',cp,cq),('p7g',gp,gq))}


def original(job):
    pp,p,ep,e,cp,cq,*_ = parents()
    if job['model_id'] == B0:
        old = next(j for j in cq['jobs'] if j['model_id'] == B0 and j['driver_id'] == cross.AUTHOR_LOOP
                   and j['run_seed'] == job['run_seed'] and j['simulator_seed'] == job['simulator_seed'])
        return 'p7e_b0', old, cq, cp.parent/'evaluate'/old['id']
    arm = 'conditional' if job['model_id'] == B3 else EXTERNAL[job['model_id']]
    old = next(j for j in e['jobs'] if j['arm'] == arm and j['run_seed'] == job['run_seed']
               and j['simulator_seed'] == job['simulator_seed'])
    return 'p7d_external', old, e, ep.parent/'evaluate'/old['id']


def records(result, job):
    frames = result['actual_observed_frames']; history = result['histories']['control_history']
    if len(frames) != len(history): raise ValueError('Missing original author decision states')
    queries = result.get('decision_policy_queries')
    requested = result.get('requested_conditional_jerks')
    if job['model_id'] == B0: requested = [r['requested_jerk'] for r in result['actual_query_records']]
    if job['model_id'] == AUTHOR:
        if any(len(q) != 1 for q in queries): raise ValueError('Pure author DDPG query count changed')
        requested = [q[0] for q in queries]
    return [{'frame': previous.canonical_frame(frame), 'commanded_speed': history[i],
             'requested_jerk': None if requested is None else requested[i],
             'policy_queries': queries[i] if job['model_id'] in EXTERNAL else None}
            for i,frame in enumerate(frames)]


def reference(job):
    kind, old, q, directory = original(job)
    binding = cross.binding(q,old) if kind == 'p7e_b0' else external.binding(q,old)
    m = verify_episode(directory,binding)['metadata']; result = read(directory/'trace.json')
    outcome = read(directory/'episode.json')
    if m['no_policy_updates'] is not True or m['outcome'] != outcome or result['outcome'] != outcome:
        raise ValueError('Historical frozen reference receipt changed')
    decisions = records(result,job)
    files = {**inventory(directory), 'complete.json': sha(directory/'complete.json')}
    return {'row': native_row(job,outcome,decisions), 'decisions': decisions,
        'histories': result['histories'], 'takeovers': result.get('takeovers'),
        'kind': kind, 'original_job': old, 'original_request_hash': digest(q),
        'directory': str(directory.relative_to(ROOT)), 'files': files}


def raw_reuse(job, ref):
    *_,gp,gq,saved = parents()
    oldrow = next((r for r in saved['rows'] if r['id'] == job['id']),None)
    if oldrow is None: return None
    origin = oldrow['recovery_origin']
    directory = (gp.parent/'evaluate'/job['id'] if origin['kind'] == 'new_simulation'
                 else inside(ROOT,origin['directory']))
    trace = read(directory/'trace.json'); outcome = read(directory/'episode.json')
    decisions = [{'frame': r['frame'], 'commanded_speed': r['commanded_speed'],
        'requested_jerk': r['requested_jerk'],
        'policy_queries': [r['requested_jerk']] if job['model_id'] == AUTHOR else None}
        for r in trace['decisions']]
    result = analyze(job,outcome,decisions,trace['native_histories'],None,
                     read(directory/'raw_events.json'),ref)
    if result['independent_events'] != oldrow['independent_events']:
        raise ValueError('Reused independent event classification changed')
    return {'analysis': result, 'directory': str(directory.relative_to(ROOT)),
        'files': {**inventory(directory), 'complete.json': sha(directory/'complete.json')},
        'kind': 'verified_p7g_raw', 'source_request_hash': digest(gq)}


def binding(q,job):
    return {'request_hash': digest(q),'stage': 'external_raw_terminal_audit','job': job}


def analyze_directory(q,job,directory,ref):
    result = read(directory/'trace.json'); outcome = read(directory/'episode.json')
    resolved = read(directory/'resolved_config.json')
    if (result['outcome'] != outcome or resolved['p7h_protocol'] != q['protocol'] or resolved['p7h_job'] != job):
        raise ValueError('Native trace/resolved audit identity changed')
    return analyze(job,outcome,records(result,job),result['histories'],result.get('takeovers'),
                   read(directory/'raw_events.json'),ref)


def source_snapshot(config,audit=None):
    refs = {j['id']: reference(j) for j in roster(config)}
    reuse = {j['id']: r for j in roster(config) if (r := raw_reuse(j,refs[j['id']])) is not None}
    if audit is not None:
        ap,aq,am = load(inside(ROOT,audit['path']).parent/'request.json',audit['request_hash'])
        if aq['protocol'] != AUDIT or (ap.parent/'writer.lock').exists():
            raise ValueError('Need complete inactive interface audit')
        a = collect(aq,am,ap.parent); saved = read(ap.parent/'aggregate.json')
        if (any(saved[k] != v for k,v in a.items()) or sha(ap.parent/'aggregate.json') != audit['sha256']
                or read(ap.parent/'aggregate_receipt.json') != {'request_hash': digest(aq),'sha256': audit['sha256']}):
            raise ValueError('Interface audit not reproducible')
        for job in aq['jobs']:
            if job['id'] not in refs or job['id'] in reuse: raise ValueError('Unexpected audit reuse cell')
            directory = ap.parent/'evaluate'/job['id']; row = analyze_directory(aq,job,directory,refs[job['id']])
            reuse[job['id']] = {'analysis': row,'kind': 'verified_interface_audit',
                'source_request_hash': digest(aq),'directory': str(directory.relative_to(ROOT)),
                'files': {**inventory(directory),'complete.json': sha(directory/'complete.json')}}
    pending = [j for j in roster(config) if j['id'] not in reuse]
    return {'parent_identity': identity(),'references': refs,'reuse': reuse,'pending_jobs': pending}


def audit_evidence(path):
    path = inside(ROOT,path); q = read(path.parent/'request.json')
    rp,q,m = load(path.parent/'request.json',digest(q))
    if q['protocol'] != AUDIT or (rp.parent/'writer.lock').exists():
        raise ValueError('Need inactive P7h interface audit')
    actual = collect(q,m,rp.parent)
    if (any(read(path)[k] != v for k,v in actual.items()) or
            read(path.parent/'aggregate_receipt.json') != {'request_hash': digest(q),'sha256': sha(path)}):
        raise ValueError('P7h interface audit not complete/reproducible')
    return {'path': str(path.relative_to(ROOT)),'request_hash': digest(q),'sha256': sha(path)}


def prepare(config,run_id,audit_path=None):
    shared.clean(); validate_protocol(config)
    audit = None if config == AUDIT else audit_evidence(audit_path or 'artifacts/p7h/p7h_audit_v1/aggregate.json')
    manifest = source_snapshot(config,audit); out = shared.output_dir('p7h',run_id)
    write_once(out/'source_manifest.json',manifest)
    q = {**shared.metadata(),'version': 'p07h_request_v1','protocol': deepcopy(config),
        'jobs': manifest['pending_jobs'],'source_manifest_sha256': sha(out/'source_manifest.json'),
        'source_hashes': shared.source_hashes(),'environment': shared.env(),
        'runtime_identity': previous.p7.runtime_identity(),'author_assets': external.assets(),
        'upstream_config_hashes': external.configurations(),'engineering_audit': audit}
    write_once(out/'request.json',q)
    write_once(out/'preparation.json',{'request_hash': digest(q),'status': 'prepared_not_simulated',
        'total_cells': len(roster(config)),'verified_raw_reused': len(manifest['reuse']),
        'new_episodes': len(q['jobs']),'training_started': False,'source_artifacts_modified': False})
    print('external_terminal_request='+str(out/'request.json'),flush=True)
    print('confirm_request_hash='+digest(q),flush=True)
    return out/'request.json',digest(q)


def load(path,expected):
    shared.clean(); path = inside(ROOT,path); q = read(path); validate_protocol(q['protocol'])
    if (q['version'] != 'p07h_request_v1' or digest(q) != expected
            or q['source_hashes'] != shared.source_hashes() or q['environment'] != shared.env()
            or q['runtime_identity'] != previous.p7.runtime_identity() or q['author_assets'] != external.assets()
            or q['upstream_config_hashes'] != external.configurations()
            or read(path.parent/'preparation.json')['request_hash'] != expected
            or q['source_manifest_sha256'] != sha(path.parent/'source_manifest.json')):
        raise ValueError('P7h request/source/environment/config changed')
    if (q['protocol'] == AUDIT) != (q['engineering_audit'] is None):
        raise ValueError('Invalid/self-dependent P7h engineering audit')
    manifest = source_snapshot(q['protocol'],q['engineering_audit'])
    if manifest != read(path.parent/'source_manifest.json') or q['jobs'] != manifest['pending_jobs']:
        raise ValueError('Frozen historical evidence/raw-reuse/remaining roster changed')
    return path,q,manifest


def worker(args):
    path,q,m = load(args.request,args.confirm_request_hash)
    job = next(j for j in q['jobs'] if j['id'] == args.job)
    pp,p,*_ = parents(); kind,old,source_q,_ = original(job)
    out = path.parent/'evaluate'/job['id']; out.mkdir(parents=True,exist_ok=False)
    started = shared.metadata(); write_once(out/'started.json',started)
    torch.set_num_threads(q['protocol']['torch_threads'])
    try:
        # Original historical episode function/arguments, scoped read-only observer.
        function = cross.author_loop_episode if kind == 'p7e_b0' else external.episode
        result,events = legacy.instrument_episode(function,previous.upstream_session,
            (pp,p,old,source_q) if kind == 'p7e_b0' else (pp,p,source_q,old),job)
        resolved = {**result['resolved_config'],'p7h_protocol': q['protocol'],'p7h_job': job}
        write_once(out/'resolved_config.json',resolved)
        write_once(out/'episode.json',result['outcome']); write_once(out/'trace.json',result)
        write_once(out/'raw_events.json',events)
        row = analyze_directory(q,job,out,m['references'][job['id']]); write_once(out/'analysis.json',row)
        seal_episode(out,binding(q,job),{**started,'status': 'complete','outcome': result['outcome'],
            'no_policy_updates': True,'exact_native_trajectory_parity': True,
            'event_implementation': q['protocol']['event_implementation'],'finished_at': shared.now()})
    except BaseException:
        write_once(out/'failure.json',{**started,'failure_reason': traceback.format_exc()}); raise


def verify_new(q,job,directory,ref):
    meta = verify_episode(directory,binding(q,job))['metadata']
    if (meta['no_policy_updates'] is not True or meta['exact_native_trajectory_parity'] is not True
            or meta['event_implementation'] != q['protocol']['event_implementation']
            or meta['outcome'] != read(directory/'episode.json')):
        raise ValueError('Frozen runtime/complete receipt mismatch')
    row = analyze_directory(q,job,directory,ref)
    if read(directory/'analysis.json') != row: raise ValueError('P7h analysis not reproducible')
    return row


def collect(q,m,out):
    rows = [{**v['analysis'],'origin': {k:v[k] for k in ('kind','directory','files','source_request_hash')}}
            for v in m['reuse'].values()]
    for job in q['jobs']:
        row = verify_new(q,job,out/'evaluate'/job['id'],m['references'][job['id']])
        rows.append({**row,'origin': {'kind': 'new_raw_simulation','request_hash': digest(q)}})
    order = {j['id']: i for i,j in enumerate(roster(q['protocol']))}
    rows.sort(key=lambda r: order[r['id']])
    return {**aggregate(q['protocol'],rows),'request_hash': digest(q),
        'verified_raw_reused': len(m['reuse']),'new_episodes': len(q['jobs']),
        'source_artifacts_modified': False,'training_started': False}


def aggregate_request(path,expected):
    path,q,m = load(path,expected); out = path.parent
    with run_lock(out):
        result = collect(q,m,out)
        if (out/'aggregate.json').exists():
            if (any(read(out/'aggregate.json')[k] != v for k,v in result.items()) or
                    read(out/'aggregate_receipt.json') != {'request_hash': expected,'sha256': sha(out/'aggregate.json')}):
                raise ValueError('Immutable P7h aggregate/receipt changed')
        else:
            write_once(out/'aggregate.json',{**shared.metadata(),**result,'finished_at': shared.now()})
            write_once(out/'aggregate_receipt.json',{'request_hash': expected,'sha256': sha(out/'aggregate.json')})
    print('external_terminal_aggregate='+str(out/'aggregate.json'),flush=True)
    return out/'aggregate.json'


def execute(path,expected,resume=False):
    path,q,m = load(path,expected); out = path.parent
    invocation = out/'invocations'/uuid.uuid4().hex[:12]; invocation.mkdir(parents=True,exist_ok=False)
    report = {**shared.metadata(),'request_hash': expected,'status': 'running','jobs': []}
    with run_lock(out):
        try:
            pending = []
            for job in q['jobs']:
                directory = out/'evaluate'/job['id']
                if not directory.exists(): pending.append(job); continue
                if not resume: raise ValueError('Use --resume only for verified COMPLETE cells')
                verify_new(q,job,directory,m['references'][job['id']])
                report['jobs'].append({'id': job['id'],'action': 'reuse'})
            def child(job):
                command = [sys.executable,'-B',str(Path(__file__).resolve()),'worker','--request',str(path),
                           '--confirm-request-hash',expected,'--job',job['id']]
                print('[p7h] evaluating '+job['id'],flush=True)
                log_path = invocation/(job['id']+'.log')
                with log_path.open('w',encoding='utf-8') as log:
                    result = subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                if result.returncode: raise RuntimeError(f'Worker {job["id"]} failed; inspect {log_path}')
                verify_new(q,job,out/'evaluate'/job['id'],m['references'][job['id']])
                print('[p7h] complete '+job['id'],flush=True)
                return {'id': job['id'],'action': 'run'}
            width = q['protocol']['workers']
            with ThreadPoolExecutor(max_workers=width) as pool:
                for i in range(0,len(pending),width):
                    futures = [pool.submit(child,j) for j in pending[i:i+width]]
                    report['jobs'].extend(f.result() for f in futures)
            report.update(status='complete',finished_at=shared.now())
        except BaseException:
            report.update(status='failed',failure_reason=traceback.format_exc()); raise
        finally: write_once(invocation/'report.json',report)
    return aggregate_request(path,expected)


def main():
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest='command',required=True)
    c = sub.add_parser('audit'); c.add_argument('--run-id',default='p7h_audit_v1')
    c = sub.add_parser('prepare'); c.add_argument('--config',default='configs/development/p07h_external_terminal_audit_v1.json')
    c.add_argument('--run-id',default='p7h_external_v1'); c.add_argument('--engineering-audit')
    for name in ('run','aggregate','worker'):
        c = sub.add_parser(name); c.add_argument('--request',required=True); c.add_argument('--confirm-request-hash',required=True)
        if name == 'run': c.add_argument('--resume',action='store_true')
        if name == 'worker': c.add_argument('--job',required=True)
    args = parser.parse_args()
    if args.command == 'audit':
        path,expected = prepare(AUDIT,args.run_id); execute(path,expected)
    elif args.command == 'prepare': prepare(read(inside(ROOT,args.config)),args.run_id,args.engineering_audit)
    elif args.command == 'run': execute(args.request,args.confirm_request_hash,args.resume)
    elif args.command == 'aggregate': aggregate_request(args.request,args.confirm_request_hash)
    else: worker(args)


if __name__ == '__main__': main()
