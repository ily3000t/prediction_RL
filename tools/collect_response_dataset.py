"""Prepare without SUMO; manually run train/validation/calibration; never test."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import platform
import re
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import audit_response_collection as audit
from prediction_rl.data.collection_protocol import collection_plan, selected_jobs, summarize, validate_job_metadata, SPLITS
from prediction_rl.data.collection_store import read_json as read, file_hash as sha, write_once, run_lock, verify_episode
from prediction_rl.data.dataset_contract import digest
from prediction_rl.data.episode_collector import collect_episode


def now():
    return datetime.now(timezone.utc).isoformat()


def git(*args):
    return subprocess.check_output(['git', '-C', str(ROOT), *args], text=True, encoding='utf-8').strip()


def clean_tree():
    if git('status', '--porcelain'):
        raise ValueError('Commit reviewed source/config before preparing or executing collection')


def environment():
    return {**audit.environment_metadata(), 'python_version': sys.version, 'os': platform.platform()}


def provenance():
    return {'git_commit': git('rev-parse', 'HEAD'), 'working_tree_dirty': False,
            'command': subprocess.list2cmdline([sys.executable, *sys.argv]), 'started_at': now(), 'optimizer_seed': None}


def verify_acceptance(path):
    path = Path(path).resolve()
    if not path.is_relative_to(ROOT / 'artifacts/p4'):
        raise ValueError('Collector evidence must be inside project artifacts/p4')
    report = read(path)
    if (report['status'] != 'complete' or report['collector_gate'] is not True
            or report['root_count'] != 9 or report['candidate_count'] != 45 or report['branch_executions'] != 90):
        raise ValueError('Require full accepted three-seed collector audit')
    output = path.parent.parent.parent
    request = read(output / 'request.json')
    if report['request_hash'] != digest(request):
        raise ValueError('Collector acceptance request changed')
    # The audit must exercise this exact current implementation, not an older core.
    audit.verify_inputs(request['input_hashes'])
    c, reports, hashes, _ = audit.prerequisites(ROOT / 'configs/development/p04_response_collection_v1.json')
    if request['config'] != c or request['input_hashes'] != hashes or request['environment'] != audit.environment_metadata():
        raise ValueError('Collector acceptance is stale for current source/environment')
    result = audit.aggregate(output, request, reports)
    for k in ('root_count', 'candidate_count', 'branch_executions', 'supervision_data_hash'):
        if result[k] != report[k]:
            raise ValueError('Collector aggregate differs from accepted report')
    for i in range(3):
        receipt = output / f'e{i:03d}/complete.json'
        hashes[str(receipt.relative_to(ROOT))] = sha(receipt)
    hashes[str(path.relative_to(ROOT))] = sha(path)
    hashes[str((output / 'request.json').relative_to(ROOT))] = sha(output / 'request.json')
    return reports, hashes


def prepare(args):
    clean_tree()
    config_path = Path(args.config).resolve()
    if not config_path.is_relative_to(ROOT):
        raise ValueError('Configuration must be inside project')
    protocol = read(config_path); plan = collection_plan(protocol)
    reports, hashes = verify_acceptance(args.collector_report)
    p3 = reports['mechanism']; b = protocol['baseline']; c = protocol['collection']
    if (b['upstream_config'] != p3['plan']['upstream_config'] or
            b['reference_checkpoint'] != p3['plan']['reference_policy']['checkpoint'] or
            b['reference_sha256'] != p3['policy_checkpoint_hash'] or
            b['episode_group_namespace_sha256'] != reports['label']['environment_identity_sha256'] or
            c['root_targets'] != p3['plan']['root_targets']):
        raise ValueError('Released baseline/roots differ from accepted mechanism')
    for key, kind in [('mechanism', 'mechanism'), ('training_interface', 'training')]:
        if protocol['sources'][key+'_sha256'] != audit.PINNED[kind]:
            raise ValueError('Protocol prerequisite hash changed')
        name = protocol['sources'][key+'_report']
        if sha(audit.project_path(name)) != audit.PINNED[kind]:
            raise ValueError('Protocol prerequisite report changed')
        hashes[name] = audit.PINNED[kind]
    hashes['json_canonical:' + str(config_path.relative_to(ROOT))] = digest(protocol)
    if not re.fullmatch('[A-Za-z0-9_-]{1,24}', args.run_id):
        raise ValueError('Use short run ID (24 characters maximum)')
    output = ROOT / 'artifacts/data' / args.run_id
    output.mkdir(parents=True, exist_ok=False)
    request = {'schema_version': 1, 'protocol': protocol, 'plan': plan, 'input_hashes': hashes,
               'environment': environment(), 'network': 'RL-MPC-LaneMerging-master/merge.net.xml',
               'collector_report': str(Path(args.collector_report).resolve().relative_to(ROOT)),
               'test_locked': True, 'execution_requires_manual_hash_confirmation': True}
    write_once(output / 'request.json', request)
    report = {**provenance(), 'status': 'prepared_not_executed', 'request_hash': digest(request),
              'executable_episode_budget': 384, 'locked_test_episode_budget': 128,
              'formal_training_ready': False, 'collection_started': False, 'finished_at': now()}
    # Counts come from the validated manifest, not a claim of obtained coverage.
    report['executable_episode_budget'] = sum(j['split'] in SPLITS for j in plan['jobs'])
    report['locked_test_episode_budget'] = sum(j['split'] == 'test' for j in plan['jobs'])
    write_once(output / 'preparation.json', report)
    print('request=' + str(output / 'request.json'))
    print('confirm_request_hash=' + digest(request))
    return output / 'request.json'


def load_request(path):
    path = Path(path).resolve()
    if path.name != 'request.json' or not path.is_relative_to(ROOT / 'artifacts/data'):
        raise ValueError('Use an immutable prepared request inside artifacts/data')
    request = read(path)
    if (set(request) != {'schema_version', 'protocol', 'plan', 'input_hashes', 'environment', 'network',
                        'collector_report', 'test_locked', 'execution_requires_manual_hash_confirmation'}
            or request['schema_version'] != 1 or request['test_locked'] is not True
            or request['execution_requires_manual_hash_confirmation'] is not True
            or request['network'] != 'RL-MPC-LaneMerging-master/merge.net.xml'):
        raise ValueError('Unexpected request contract')
    if request['plan'] != collection_plan(request['protocol']):
        raise ValueError('Request/manifest mismatch')
    # Receipt prevents unnoticed replacement of the prepared request.
    if read(path.parent / 'preparation.json')['request_hash'] != digest(request):
        raise ValueError('Prepared request changed')
    audit.verify_inputs(request['input_hashes'])
    if environment() != request['environment']:
        raise ValueError('Environment changed since preparation')
    return path.parent, request


def binding(request, job):
    return {'request_hash': digest(request), 'job': job}


def checked_episode(output, request, job):
    receipt = verify_episode(output / job['job_id'], binding(request, job))
    targets = [t['id'] for t in request['protocol']['collection']['root_targets']]
    validate_job_metadata(job, receipt['metadata'], targets)
    for number, row in enumerate(receipt['metadata']['roots']):
        if row['status'] == 'complete':
            # Full candidate/history contract validation is independent of receipt booleans.
            from prediction_rl.prediction.interface import PredictorConfig
            from prediction_rl.prediction.supervision import align_supervision
            directory = output / job['job_id'] / f'r{number}'
            history, pack = read(directory/'history.json'), read(directory/'labels.json')
            if history['label_pack_sha256'] != sha(directory/'labels.json'):
                raise ValueError('History-to-label reference changed')
            align_supervision([history], [pack], request['plan']['split_manifest'],
                              PredictorConfig.from_dict(request['protocol']['model']), split=job['split'])
    return receipt


def execute(args):
    clean_tree()
    output, request = load_request(args.request)
    jobs = selected_jobs(request, args.splits, args.confirm_request_hash)
    with run_lock(output):
        invocation = output/'invocations'/uuid.uuid4().hex[:12]
        invocation.mkdir(parents=True, exist_ok=False)
        authorization = {**provenance(), 'request_hash': digest(request), 'splits': args.splits,
                         'job_ids': [j['job_id'] for j in jobs], 'lock_token': read(output/'writer.lock')['token']}
        write_once(invocation/'authorization.json', authorization)
        report = {**authorization, 'status': 'running', 'executed_jobs': [], 'reused_jobs': [],
                  'completed_jobs': [], 'formal_training_ready': False, 'test_locked': True}
        try:
            for job in jobs:
                directory = output/job['job_id']
                if directory.exists():
                    if not args.resume:
                        raise ValueError('Existing episode requires explicit --resume; never overwrite it')
                    checked_episode(output, request, job)
                    report['reused_jobs'].append(job['job_id'])
                else:
                    print(f"[dataset] {job['job_id']} split={job['split']} seed={job['simulator_seed']} collect", flush=True)
                    report['executed_jobs'].append(job['job_id'])
                    with (invocation/(job['job_id']+'.log')).open('x', encoding='utf-8') as log:
                        subprocess.run([sys.executable, str(Path(__file__).resolve()), '_worker', '--request',
                            str(output/'request.json'), '--authorization', str(invocation/'authorization.json'),
                            '--job-id', job['job_id']], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                            check=True, timeout=request['protocol']['execution']['worker_wall_limit_s'])
                    checked_episode(output, request, job)
                report['completed_jobs'].append(job['job_id'])
                print(f"[dataset] completed={len(report['completed_jobs'])}/{len(jobs)}", flush=True)
            audit.verify_inputs(request['input_hashes'])
            records = [(j, checked_episode(output, request, j)['metadata']) for j in jobs]
            targets = [t['id'] for t in request['protocol']['collection']['root_targets']]
            report.update(status='complete', summaries={s: summarize([(j,m) for j,m in records if j['split']==s], targets)
                                                       for s in args.splits})
        except BaseException as error:
            report.update(status='failed', failure_reason=f'{type(error).__name__}: {error}')
            raise
        finally:
            report['finished_at'] = now()
            write_once(invocation/'report.json', report)
            print('dataset_report=' + str(invocation/'report.json'), flush=True)
    return invocation/'report.json'


def worker(args):
    output, request = load_request(args.request)
    authorization_path = Path(args.authorization).resolve()
    if (not authorization_path.is_relative_to(output/'invocations') or
            authorization_path.name != 'authorization.json'):
        raise ValueError('Worker requires a parent invocation')
    authorization = read(authorization_path)
    if authorization['lock_token'] != read(output/'writer.lock')['token']:
        raise ValueError('Worker parent lock no longer owned')
    jobs = selected_jobs(request, authorization['splits'], authorization['request_hash'])
    if authorization['job_ids'] != [j['job_id'] for j in jobs]:
        raise ValueError('Parent job selection changed')
    found = [j for j in jobs if j['job_id'] == args.job_id]
    if len(found) != 1:
        raise ValueError('Job is not authorized (test remains locked)')
    job = found[0]
    p = request['protocol']
    collect_episode(ROOT, p['baseline'], p['collection'], request['network'], job, output/job['job_id'],
                    binding(request, job), before_seal=lambda: audit.verify_inputs(request['input_hashes']))


def main():
    p = argparse.ArgumentParser(description=__doc__); sub = p.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare'); prep.add_argument('--config', required=True)
    prep.add_argument('--collector-report', required=True); prep.add_argument('--run-id', required=True)
    run = sub.add_parser('run'); run.add_argument('--request', required=True)
    run.add_argument('--splits', nargs='+', choices=SPLITS, required=True)
    run.add_argument('--confirm-request-hash', required=True); run.add_argument('--resume', action='store_true')
    child = sub.add_parser('_worker'); child.add_argument('--request', required=True)
    child.add_argument('--authorization', required=True); child.add_argument('--job-id', required=True)
    args = p.parse_args()
    if args.command == 'prepare': prepare(args)
    elif args.command == 'run': execute(args)
    else: worker(args)


if __name__ == '__main__':
    main()
