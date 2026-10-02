"""P7g: user-run raw terminal-event audit of unchanged frozen P7f episodes."""
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
import diagnose_driver_warmup as previous
from prediction_rl.data.collection_store import (read_json as read, write_once, file_hash as sha,
    verify_episode, seal_episode, run_lock)
from prediction_rl.data.dataset_contract import digest
from prediction_rl.data.frozen_dataset import inside
from prediction_rl.evaluation.terminal_events import (PROTOCOL, AUDIT, CONDITIONS, validate_protocol, jobs,
    instrument_episode, classify, assert_reference_parity, aggregate)

shared = previous.shared


def finished_previous(path, protocol, expected_hash, expected_sha):
    path = inside(ROOT, path); q = read(path); out = path.parent
    if ((out/'writer.lock').exists() or q['version'] != 'p07f_request_v1' or q['protocol'] != protocol
            or digest(q) != expected_hash or sha(out/'aggregate.json') != expected_sha
            or q['jobs'] != previous.jobs(protocol) or q['runtime_identity'] != previous.p7.runtime_identity()
            or q['author_assets'] != previous.cross.external.assets()
            or read(out/'preparation.json')['request_hash'] != expected_hash):
        raise ValueError('Historical P7f request/report/runtime changed or active')
    previous.historical_source(q)  # Additive new files allowed; all old executable sources must match.
    rows = []; traces = {}
    for j in previous.roster(protocol):
        if previous.is_new(j):
            directory = out/'evaluate'/j['id']; m = verify_episode(directory, previous.binding(q, j))['metadata']
            outcome = read(directory/'episode.json'); trace = read(directory/'trace.json')['decisions']
            if m['outcome'] != outcome or m['no_policy_updates'] is not True or m['interface_audit_pass'] is not True:
                raise ValueError('Historical P7f frozen interface invalid')
            row = previous.row_from_native(j, outcome, trace,
                previous.evidence('new', q, directory, ('episode.json', 'trace.json', 'complete.json')))
        else: row, trace = previous.historical(j)
        rows.append(row); traces[j['id']] = trace
    a = {**previous.aggregate(protocol, rows, traces), 'request_hash': expected_hash}
    saved = read(out/'aggregate.json')
    if (any(saved[k] != v for k, v in a.items())
            or read(out/'aggregate_receipt.json') != {'request_hash': expected_hash, 'sha256': expected_sha}
            or q['historical_rows'] != [r for r in rows if not previous.is_new(r)]):
        raise ValueError('Historical P7f aggregate/receipt not reproducible')
    return path, q, {r['id']: {'row': r, 'decisions': traces[r['id']]} for r in rows}


@lru_cache(maxsize=1)
def parent():
    pp, p, *_ = previous.parents()
    path = inside(ROOT, PROTOCOL['parent_request']); q = read(path); ev = q['engineering_audit']
    ap, aq, _ = finished_previous(Path(ev['path']).parent/'request.json', previous.AUDIT,
                                  ev['request_hash'], ev['sha256'])
    if aq['engineering_audit'] is not None: raise ValueError('Self-dependent historical interface audit')
    path, q, refs = finished_previous(path, previous.PROTOCOL,
        PROTOCOL['parent_request_hash'], PROTOCOL['parent_aggregate_sha256'])
    return pp, p, path, q, refs


def parent_identity():
    _, _, path, q, _ = parent()
    return {'path': str(path.relative_to(ROOT)), 'request_hash': digest(q), 'request_sha256': sha(path),
            'aggregate_sha256': sha(path.parent/'aggregate.json')}


def binding(q, j): return {'request_hash': digest(q), 'stage': 'raw_terminal_event_audit', 'job': j}


def audit_evidence(path):
    path = inside(ROOT, path); request = path.parent/'request.json'; q = read(request)
    _, q = load(request, digest(q)); result = collect(q, path.parent)
    if (q['protocol'] != AUDIT or any(read(path)[k] != v for k, v in result.items())
            or read(path.parent/'aggregate_receipt.json') != {'request_hash': digest(q), 'sha256': sha(path)}):
        raise ValueError('Need complete exact-parity raw terminal interface audit')
    return {'path': str(path.relative_to(ROOT)), 'request_hash': digest(q), 'sha256': sha(path)}


def prepare(config, run_id, audit_path=None):
    shared.clean(); validate_protocol(config); *_, references = parent()
    accepted = None if config == AUDIT else audit_evidence(audit_path or 'artifacts/p7g/p7g_audit_v1/aggregate.json')
    out = shared.output_dir('p7g', run_id)
    write_once(out/'reference.json', {j['id']: references[j['id']] for j in jobs(config)})
    q = {**shared.metadata(), 'version': 'p07g_request_v1', 'protocol': deepcopy(config), 'jobs': jobs(config),
         'parent_identity': parent_identity(), 'reference_sha256': sha(out/'reference.json'),
         'source_hashes': shared.source_hashes(), 'environment': shared.env(),
         'runtime_identity': previous.p7.runtime_identity(), 'author_assets': previous.cross.external.assets(),
         'engineering_audit': accepted}
    write_once(out/'request.json', q)
    write_once(out/'preparation.json', {'request_hash': digest(q), 'status': 'prepared_not_simulated',
        'new_episodes': len(q['jobs']), 'historical_episodes_reused_for_raw_events': 0})
    print('terminal_event_request='+str(out/'request.json'), flush=True)
    print('confirm_request_hash='+digest(q), flush=True); return out/'request.json', digest(q)


def load(path, expected):
    shared.clean(); path = inside(ROOT, path); q = read(path); validate_protocol(q['protocol'])
    if (q['version'] != 'p07g_request_v1' or digest(q) != expected or q['jobs'] != jobs(q['protocol'])
            or read(path.parent/'preparation.json')['request_hash'] != expected
            or q['source_hashes'] != shared.source_hashes() or q['environment'] != shared.env()
            or q['runtime_identity'] != previous.p7.runtime_identity()
            or q['author_assets'] != previous.cross.external.assets() or q['parent_identity'] != parent_identity()
            or q['reference_sha256'] != sha(path.parent/'reference.json')):
        raise ValueError('Raw terminal request/source/runtime/reference changed')
    *_, refs = parent()
    if read(path.parent/'reference.json') != {j['id']: refs[j['id']] for j in q['jobs']}:
        raise ValueError('Frozen reference trace changed')
    if q['protocol'] == PROTOCOL:
        if audit_evidence(q['engineering_audit']['path']) != q['engineering_audit']: raise ValueError('Engineering audit changed')
    elif q['engineering_audit'] is not None: raise ValueError('Self-dependent engineering audit')
    return path, q


def row_and_analysis(q, job, directory):
    outcome = read(directory/'episode.json'); trace = read(directory/'trace.json')['decisions']
    row = previous.row_from_native(job, outcome, trace,
        previous.evidence('new', q, directory, ('episode.json', 'trace.json', 'raw_events.json')))
    *_, refs = parent(); ref = refs[job['id']]
    parity = assert_reference_parity(row, trace, ref['row'], ref['decisions'])
    analysis = classify(read(directory/'raw_events.json'), row, trace)
    return {**row, 'parity': parity, 'independent_events': analysis}


def worker(args):
    path, q = load(args.request, args.confirm_request_hash); pp, p, *_ = parent()
    j = next(j for j in q['jobs'] if j['id'] == args.job)
    out = path.parent/'evaluate'/j['id']; out.mkdir(parents=True, exist_ok=False)
    started = shared.metadata(); write_once(out/'started.json', started)
    torch.set_num_threads(q['protocol']['torch_threads'])
    try:
        function = previous.gym_episode if CONDITIONS[j['condition_id']]['driver'] == 'gym' else previous.author_episode
        (outcome, trace, settings), events = instrument_episode(function, previous.upstream_session, (pp, p, j), j)
        write_once(out/'episode.json', outcome); write_once(out/'trace.json', trace); write_once(out/'raw_events.json', events)
        write_once(out/'resolved_config.json', {'upstream': settings, 'protocol': q['protocol'], 'job': j})
        row = row_and_analysis(q, j, out); write_once(out/'analysis.json', row)
        seal_episode(out, binding(q, j), {**started, 'status': 'complete', 'outcome': outcome,
            'interface_audit_pass': True, 'no_policy_updates': True, 'exact_reference_parity': True,
            'finished_at': shared.now()})
    except BaseException:
        write_once(out/'failure.json', {**started, 'failure_reason': traceback.format_exc()}); raise


def collect(q, out):
    rows = []
    for j in q['jobs']:
        directory = out/'evaluate'/j['id']; m = verify_episode(directory, binding(q, j))['metadata']
        if (m['no_policy_updates'] is not True or m['interface_audit_pass'] is not True
                or m['exact_reference_parity'] is not True or m['outcome'] != read(directory/'episode.json')):
            raise ValueError('Frozen interface/receipt mismatch')
        row = row_and_analysis(q, j, directory)
        if row != read(directory/'analysis.json'): raise ValueError('Raw-event analysis not reproducible')
        rows.append(row)
    return {**aggregate(q['protocol'], rows), 'request_hash': digest(q)}


def aggregate_request(path, expected):
    path, q = load(path, expected); out = path.parent
    with run_lock(out):
        result = collect(q, out)
        if (out/'aggregate.json').exists():
            if (any(read(out/'aggregate.json')[k] != v for k, v in result.items())
                    or read(out/'aggregate_receipt.json') != {'request_hash': expected, 'sha256': sha(out/'aggregate.json')}):
                raise ValueError('Immutable raw-event aggregate changed')
        else:
            write_once(out/'aggregate.json', {**shared.metadata(), **result, 'finished_at': shared.now()})
            write_once(out/'aggregate_receipt.json', {'request_hash': expected, 'sha256': sha(out/'aggregate.json')})
    print('terminal_event_aggregate='+str(out/'aggregate.json'), flush=True); return out/'aggregate.json'


def execute(path, expected, resume=False):
    path, q = load(path, expected); out = path.parent
    invocation = out/'invocations'/uuid.uuid4().hex[:12]; invocation.mkdir(parents=True, exist_ok=False)
    report = {**shared.metadata(), 'request_hash': expected, 'jobs': [], 'status': 'running'}
    with run_lock(out):
        try:
            pending = []
            for j in q['jobs']:
                directory = out/'evaluate'/j['id']
                if directory.exists():
                    if not resume: raise ValueError('Use --resume only for verified COMPLETE jobs')
                    m = verify_episode(directory, binding(q, j))['metadata']
                    if m['exact_reference_parity'] is not True or m['no_policy_updates'] is not True:
                        raise ValueError('Incomplete exact-reference audit')
                    report['jobs'].append({'id': j['id'], 'action': 'reuse'})
                else: pending.append(j)
            def child(j):
                command = [sys.executable, '-B', str(Path(__file__).resolve()), 'worker', '--request', str(path),
                           '--confirm-request-hash', expected, '--job', j['id']]
                print('[p7g] evaluating '+j['id'], flush=True)
                with (invocation/(j['id']+'.log')).open('w', encoding='utf-8') as log:
                    subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
                verify_episode(out/'evaluate'/j['id'], binding(q, j))
                print('[p7g] complete '+j['id'], flush=True); return {'id': j['id'], 'action': 'run'}
            width = q['protocol']['workers']
            with ThreadPoolExecutor(max_workers=width) as pool:
                for i in range(0, len(pending), width):
                    report['jobs'].extend(f.result() for f in [pool.submit(child, j) for j in pending[i:i+width]])
            report.update(status='complete', finished_at=shared.now())
        except BaseException:
            report.update(status='failed', failure_reason=traceback.format_exc()); raise
        finally: write_once(invocation/'report.json', report)
    return aggregate_request(path, expected)


def main():
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest='command', required=True)
    c = sub.add_parser('audit'); c.add_argument('--run-id', default='p7g_audit_v1')
    c = sub.add_parser('prepare'); c.add_argument('--config', default='configs/development/p07g_terminal_events_v1.json')
    c.add_argument('--run-id', default='p7g_diag_v1'); c.add_argument('--engineering-audit')
    for name in ('run', 'aggregate', 'worker'):
        c = sub.add_parser(name); c.add_argument('--request', required=True); c.add_argument('--confirm-request-hash', required=True)
        if name == 'run': c.add_argument('--resume', action='store_true')
        if name == 'worker': c.add_argument('--job', required=True)
    args = parser.parse_args()
    if args.command == 'audit':
        path, expected = prepare(AUDIT, args.run_id); execute(path, expected)
    elif args.command == 'prepare': prepare(read(inside(ROOT, args.config)), args.run_id, args.engineering_audit)
    elif args.command == 'run': execute(args.request, args.confirm_request_hash, args.resume)
    elif args.command == 'aggregate': aggregate_request(args.request, args.confirm_request_hash)
    else: worker(args)


if __name__ == '__main__': main()
