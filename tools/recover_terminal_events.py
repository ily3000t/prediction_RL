"""P7g immutable v2 reanalysis and USER-run completion of the ten missing cells."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
import subprocess
import sys
import traceback
import uuid

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'tools')]
import audit_terminal_events as legacy
from prediction_rl.data.collection_store import inventory
from prediction_rl.data.frozen_dataset import verify_historical_inputs
from prediction_rl.evaluation.terminal_events_v2 import VERSION, classify

read, sha, digest = legacy.read, legacy.sha, legacy.digest
write_once, verify_episode, seal_episode, run_lock = (legacy.write_once, legacy.verify_episode,
                                                     legacy.seal_episode, legacy.run_lock)
shared, inside = legacy.shared, legacy.inside
FAILED_IDS = ['project_20k_b3_conditional_ddpg_author20_r2_v200',
              'project_20k_b3_conditional_ddpg_author20_r2_v210']
RECOVERY = {
    'version': 'p07g_terminal_events_recovery_v1', 'analysis_implementation': VERSION,
    'source_request': 'artifacts/p7g/p7g_diag_v1/request.json',
    'source_request_hash': '1d4e443824312f75d3b46831c41d9dc938892892ed005231bf6e3d408b67fcf2',
    'source_execution_commit': 'e327fb38ddf1ca7e4f329256a3b5e585087d2965',
    'source_invocation': 'artifacts/p7g/p7g_diag_v1/invocations/15ba5bc958d9/report.json',
    'expected_completed_cells': 72, 'recoverable_analysis_failure_ids': FAILED_IDS,
    'expected_new_simulations': 10, 'preserve_source_artifacts': True,
    'reward_action_termination_changes': False, 'training_permitted': False,
    'reason': 'author_time_limit_final_command_removed_before_cleanup_step',
}


def require_design(config):
    if digest(config) != digest(RECOVERY): raise ValueError('Changed recovery design; approve a new version')


def legacy_request(path, expected, protocol):
    """Allow only additive code: every historical executable must still match."""
    path = inside(ROOT, path); q = read(path); out = path.parent
    if ((out/'writer.lock').exists() or digest(q) != expected or q['version'] != 'p07g_request_v1'
            or q['protocol'] != protocol or q['jobs'] != legacy.jobs(protocol)
            or read(out/'preparation.json')['request_hash'] != expected
            or q['reference_sha256'] != sha(out/'reference.json')
            or q['runtime_identity'] != legacy.previous.p7.runtime_identity()
            or q['author_assets'] != legacy.previous.cross.external.assets()
            or q['parent_identity'] != legacy.parent_identity()):
        raise ValueError('Historical raw request/reference/runtime changed or active')
    legacy.previous.historical_source(q)
    *_, refs = legacy.parent()
    if read(out/'reference.json') != {j['id']: refs[j['id']] for j in q['jobs']}:
        raise ValueError('Historical reference trace changed')
    return path, q, refs


def validate_audit(evidence):
    path = inside(ROOT, evidence['path']); q = read(path.parent/'request.json')
    _, q, _ = legacy_request(path.parent/'request.json', evidence['request_hash'], legacy.AUDIT)
    if (q['engineering_audit'] is not None or sha(path) != evidence['sha256']
            or read(path.parent/'aggregate_receipt.json') !=
               {'request_hash': evidence['request_hash'], 'sha256': evidence['sha256']}):
        raise ValueError('Historical interface audit changed')
    actual = legacy.collect(q, path.parent)
    if any(read(path)[k] != v for k, v in actual.items()): raise ValueError('Historical audit not reproducible')


def analyze(q, job, directory, refs):
    outcome = read(directory/'episode.json'); trace = read(directory/'trace.json')['decisions']
    row = legacy.previous.row_from_native(job, outcome, trace,
        legacy.previous.evidence('new', q, directory, ('episode.json', 'trace.json', 'raw_events.json')))
    parity = legacy.assert_reference_parity(row, trace, refs[job['id']]['row'], refs[job['id']]['decisions'])
    settings = read(directory/'resolved_config.json')
    if settings['protocol'] != q['protocol'] or settings['job'] != job:
        raise ValueError('Resolved original job/protocol changed')
    return {**row, 'parity': parity, 'independent_events': classify(read(directory/'raw_events.json'), row, trace)}


def source_cell(q, job, directory, refs):
    directory = inside(ROOT, directory)
    started = read(directory/'started.json')
    if (started['working_tree_dirty'] is not False
            or started['git_commit'] != RECOVERY['source_execution_commit']):
        raise ValueError('Original cell execution provenance changed')
    files = inventory(directory)
    if (directory/'complete.json').exists():
        m = verify_episode(directory, legacy.binding(q, job))['metadata']
        if (m['no_policy_updates'] is not True or m['exact_reference_parity'] is not True
                or m['interface_audit_pass'] is not True or m['outcome'] != read(directory/'episode.json')
                or m['git_commit'] != started['git_commit'] or m['working_tree_dirty'] is not False):
            raise ValueError('Original completed cell interface invalid')
        row = analyze(q, job, directory, refs)
        if row != read(directory/'analysis.json'): raise ValueError('Completed-cell v2 analysis changed')
        kind = 'verified_complete'
        files = {**files, 'complete.json': sha(directory/'complete.json')}
    else:
        expected = {'started.json', 'episode.json', 'trace.json', 'raw_events.json', 'resolved_config.json', 'failure.json'}
        if job['id'] not in FAILED_IDS or set(files) != expected:
            raise ValueError('Unapproved/incomplete failed cell; preserve without retry')
        failure = read(directory/'failure.json')
        reason = failure.get('failure_reason')
        if (any(failure.get(k) != v for k, v in started.items()) or not isinstance(reason, str)
                or not reason.endswith('ValueError: Native cleanup/extra-step count changed\n')
                or 'analysis = classify(' not in reason):
            raise ValueError('Failure is not the approved post-simulation analysis defect')
        row = analyze(q, job, directory, refs)
        if row['time_limit'] != 1 or row['independent_events']['classification'] != 'time_limit':
            raise ValueError('Recovered result must remain a real time-limit negative result')
        kind = 'recovered_analysis_failure'
    return {'kind': kind, 'directory': directory.relative_to(ROOT).as_posix(), 'files': files, 'analysis': row}


def source_snapshot():
    path, q, refs = legacy_request(RECOVERY['source_request'], RECOVERY['source_request_hash'], legacy.PROTOCOL)
    validate_audit(q['engineering_audit'])
    verify_historical_inputs(ROOT, {'python_lf:'+k: v for k, v in q['source_hashes'].items()},
                             RECOVERY['source_execution_commit'])
    invocation = inside(ROOT, RECOVERY['source_invocation']); report = read(invocation)
    if (report['status'] != 'failed' or report['request_hash'] != digest(q)
            or report['git_commit'] != RECOVERY['source_execution_commit']
            or report['working_tree_dirty'] is not False):
        raise ValueError('Original failed invocation changed')
    out = path.parent; cells = {}; pending = []
    for job in q['jobs']:
        directory = out/'evaluate'/job['id']
        if directory.exists(): cells[job['id']] = source_cell(q, job, directory, refs)
        else: pending.append(job)
    present = {p.name for p in (out/'evaluate').iterdir()}
    completed = [k for k, c in cells.items() if c['kind'] == 'verified_complete']
    recovered = [k for k, c in cells.items() if c['kind'] == 'recovered_analysis_failure']
    if (present != set(cells) or len(completed) != RECOVERY['expected_completed_cells']
            or recovered != FAILED_IDS or len(pending) != RECOVERY['expected_new_simulations']
            or report['jobs'] != [{'id': j['id'], 'action': 'run'} for j in q['jobs'] if j['id'] in completed]):
        raise ValueError('Original 72-complete/2-analysis-failure/10-missing roster changed')
    return {'source_request_sha256': sha(path), 'reference_sha256': sha(out/'reference.json'),
            'source_preparation_sha256': sha(out/'preparation.json'), 'source_invocation_sha256': sha(invocation),
            'source_failure_log_sha256': {k: sha(invocation.parent/(k+'.log')) for k in FAILED_IDS},
            'cells': cells, 'pending_jobs': pending}


def prepare(config, run_id):
    shared.clean(); require_design(config); manifest = source_snapshot()
    out = shared.output_dir('p7g', run_id)
    write_once(out/'source_manifest.json', manifest)
    q = {**shared.metadata(), 'version': 'p07g_recovery_request_v1', 'recovery': deepcopy(config),
         'protocol': deepcopy(legacy.PROTOCOL), 'jobs': manifest['pending_jobs'],
         'source_manifest_sha256': sha(out/'source_manifest.json'), 'source_hashes': shared.source_hashes(),
         'environment': shared.env(), 'runtime_identity': legacy.previous.p7.runtime_identity(),
         'author_assets': legacy.previous.cross.external.assets()}
    write_once(out/'request.json', q)
    write_once(out/'preparation.json', {'status': 'prepared_not_simulated', 'request_hash': digest(q),
        'verified_complete_reused': 72, 'failed_raw_records_revalidated': 2, 'new_episodes': 10,
        'source_artifacts_modified': False, 'training_started': False})
    print('terminal_event_recovery_request='+str(out/'request.json'), flush=True)
    print('confirm_request_hash='+digest(q), flush=True)
    return out/'request.json', digest(q)


def load(path, expected):
    shared.clean(); path = inside(ROOT, path); q = read(path); require_design(q['recovery'])
    if (q['version'] != 'p07g_recovery_request_v1' or digest(q) != expected
            or q['protocol'] != legacy.PROTOCOL or q['source_hashes'] != shared.source_hashes()
            or q['environment'] != shared.env() or q['runtime_identity'] != legacy.previous.p7.runtime_identity()
            or q['author_assets'] != legacy.previous.cross.external.assets()
            or read(path.parent/'preparation.json')['request_hash'] != expected
            or q['source_manifest_sha256'] != sha(path.parent/'source_manifest.json')):
        raise ValueError('Recovery request/source/runtime/manifest changed')
    manifest = source_snapshot()
    if manifest != read(path.parent/'source_manifest.json') or q['jobs'] != manifest['pending_jobs']:
        raise ValueError('Original preserved evidence/remaining roster changed')
    return path, q, manifest


def binding(q, job): return {'request_hash': digest(q), 'stage': 'raw_terminal_event_recovery_v2', 'job': job}


def worker(args):
    path, q, _ = load(args.request, args.confirm_request_hash)
    job = next(j for j in q['jobs'] if j['id'] == args.job); pp, p, *_, refs = legacy.parent()
    out = path.parent/'evaluate'/job['id']; out.mkdir(parents=True, exist_ok=False)
    started = shared.metadata(); write_once(out/'started.json', started)
    torch.set_num_threads(q['protocol']['torch_threads'])
    try:
        function = legacy.previous.gym_episode if legacy.CONDITIONS[job['condition_id']]['driver'] == 'gym' else legacy.previous.author_episode
        (outcome, trace, settings), events = legacy.instrument_episode(function, legacy.previous.upstream_session, (pp, p, job), job)
        write_once(out/'episode.json', outcome); write_once(out/'trace.json', trace); write_once(out/'raw_events.json', events)
        write_once(out/'resolved_config.json', {'upstream': settings, 'protocol': q['protocol'], 'job': job})
        row = analyze(q, job, out, refs); write_once(out/'analysis.json', row)
        seal_episode(out, binding(q, job), {**started, 'status': 'complete', 'outcome': outcome,
            'interface_audit_pass': True, 'no_policy_updates': True, 'exact_reference_parity': True,
            'analysis_implementation': VERSION, 'finished_at': shared.now()})
    except BaseException:
        write_once(out/'failure.json', {**started, 'failure_reason': traceback.format_exc()}); raise


def collect(q, manifest, out):
    *_, refs = legacy.parent()
    rows = [{**c['analysis'], 'recovery_origin': {'kind': c['kind'], 'source_request_hash': RECOVERY['source_request_hash'],
             'directory': c['directory'], 'files': c['files']}} for c in manifest['cells'].values()]
    for job in q['jobs']:
        directory = out/'evaluate'/job['id']; m = verify_episode(directory, binding(q, job))['metadata']
        if (m['no_policy_updates'] is not True or m['interface_audit_pass'] is not True
                or m['exact_reference_parity'] is not True or m['analysis_implementation'] != VERSION
                or m['outcome'] != read(directory/'episode.json')):
            raise ValueError('Recovery simulation interface/receipt changed')
        row = analyze(q, job, directory, refs)
        if row != read(directory/'analysis.json'): raise ValueError('New-cell v2 analysis not reproducible')
        rows.append({**row, 'recovery_origin': {'kind': 'new_simulation', 'request_hash': digest(q)}})
    order = {j['id']: i for i, j in enumerate(legacy.jobs(legacy.PROTOCOL))}
    rows.sort(key=lambda r: order[r['id']])
    return {**legacy.aggregate(legacy.PROTOCOL, rows), 'request_hash': digest(q), 'analysis_implementation': VERSION,
            'verified_complete_reused': 72, 'failed_raw_records_revalidated': 2, 'new_episodes': 10,
            'source_artifacts_modified': False}


def aggregate_request(path, expected):
    path, q, manifest = load(path, expected); out = path.parent
    with run_lock(out):
        result = collect(q, manifest, out)
        if (out/'aggregate.json').exists():
            if (any(read(out/'aggregate.json')[k] != v for k, v in result.items())
                    or read(out/'aggregate_receipt.json') != {'request_hash': expected, 'sha256': sha(out/'aggregate.json')}):
                raise ValueError('Immutable recovery aggregate changed')
        else:
            write_once(out/'aggregate.json', {**shared.metadata(), **result, 'finished_at': shared.now()})
            write_once(out/'aggregate_receipt.json', {'request_hash': expected, 'sha256': sha(out/'aggregate.json')})
    print('terminal_event_recovery_aggregate='+str(out/'aggregate.json'), flush=True)
    return out/'aggregate.json'


def execute(path, expected, resume=False):
    path, q, manifest = load(path, expected); out = path.parent
    invocation = out/'invocations'/uuid.uuid4().hex[:12]; invocation.mkdir(parents=True, exist_ok=False)
    report = {**shared.metadata(), 'request_hash': expected, 'status': 'running', 'jobs': []}
    with run_lock(out):
        try:
            pending = []
            for job in q['jobs']:
                directory = out/'evaluate'/job['id']
                if not directory.exists(): pending.append(job); continue
                if not resume: raise ValueError('Use --resume only for verified COMPLETE recovery cells')
                m = verify_episode(directory, binding(q, job))['metadata']
                if m['exact_reference_parity'] is not True or m['no_policy_updates'] is not True or m['analysis_implementation'] != VERSION:
                    raise ValueError('Incomplete recovery cell; preserve without retry')
                report['jobs'].append({'id': job['id'], 'action': 'reuse'})
            def child(job):
                command = [sys.executable, '-B', str(Path(__file__).resolve()), 'worker', '--request', str(path),
                           '--confirm-request-hash', expected, '--job', job['id']]
                print('[p7g_v2] evaluating '+job['id'], flush=True)
                log_path = invocation/(job['id']+'.log')
                with log_path.open('w', encoding='utf-8') as log:
                    result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                if result.returncode:
                    raise RuntimeError(f"Worker {job['id']} failed; inspect {log_path}")
                verify_episode(out/'evaluate'/job['id'], binding(q, job))
                print('[p7g_v2] complete '+job['id'], flush=True)
                return {'id': job['id'], 'action': 'run'}
            with ThreadPoolExecutor(max_workers=q['protocol']['workers']) as pool:
                width = q['protocol']['workers']
                for i in range(0, len(pending), width):
                    futures = [pool.submit(child, j) for j in pending[i:i+width]]
                    report['jobs'].extend(f.result() for f in futures)
            report.update(status='complete', finished_at=shared.now())
        except BaseException:
            report.update(status='failed', failure_reason=traceback.format_exc()); raise
        finally: write_once(invocation/'report.json', report)
    return aggregate_request(path, expected)


def main():
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare'); p.add_argument('--config', default='configs/development/p07g_terminal_events_recovery_v1.json')
    p.add_argument('--run-id', default='p7g_recovery_v1')
    for name in ('run', 'aggregate', 'worker'):
        p = sub.add_parser(name); p.add_argument('--request', required=True); p.add_argument('--confirm-request-hash', required=True)
        if name == 'run': p.add_argument('--resume', action='store_true')
        if name == 'worker': p.add_argument('--job', required=True)
    args = parser.parse_args()
    if args.command == 'prepare': prepare(read(inside(ROOT, args.config)), args.run_id)
    elif args.command == 'run': execute(args.request, args.confirm_request_hash, args.resume)
    elif args.command == 'aggregate': aggregate_request(args.request, args.confirm_request_hash)
    else: worker(args)


if __name__ == '__main__': main()
