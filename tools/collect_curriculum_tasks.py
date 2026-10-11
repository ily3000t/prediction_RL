"""IC02d approved THREE-task collector; never runs the remaining pilot or DDPG."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'tools')]
import torch

from prediction_rl.curriculum.entry_smoke import PROTOCOL as ENTRY_SMOKE, rng_fingerprint
from prediction_rl.curriculum.task_characterization import (REPAIRED_PROTOCOL as PROTOCOL,
    validate_protocol, validate_actual_budget)
from prediction_rl.curriculum.task_design import validate_design
from prediction_rl.curriculum.train_reader import git, inside, strict_json, validate_config
from prediction_rl.data.collection_store import file_hash as sha, run_lock, seal_episode, verify_episode, write_once
from prediction_rl.data.dataset_contract import digest
from smoke_curriculum_entry import inputs as entry_inputs, runtime

CONFIG = 'configs/development/ic02d_task_engineering_v2.json'
PREVIOUS_FAILURE_HASHES = {
    'request.json': 'd54408e723a453db651427641b0072a7887ad724dcbdebffdbb2f3ac7fe9ca38',
    'failure.json': 'e5ec35fce5608e17d70e7f5e5392c2f8efda141b31fb747a05309ea57c563c2a',
    'postflight.json': '1f3db5ee5a03d8230e37739f5b9300d01e321a7c53b5c4d993168bce722055d9',
    't11000/worker.log': '4a640c7c17968a0181d5e5afc5b75ecf335feb329f70165492876091f7a9ac8f',
}


def verify_previous_zero_runtime_failure(repo):
    """Bind the preserved failed attempt; never erase receipts or retry its worker."""
    directory = Path(repo)/'artifacts/ic/ic02d_eng_v1'
    if {p.name for p in directory.iterdir()} != {
            'request.json', 'preparation.json', 'started.json', 'failure.json', 'postflight.json', 't11000'}:
        raise ValueError('Previous failed run inventory changed')
    if {p.name for p in (directory/'t11000').iterdir()} != {'started.json', 'launch.json', 'worker.log'}:
        raise ValueError('Previous failure no longer has zero worker execution evidence')
    hashes = {name: sha(directory/name) for name in PREVIOUS_FAILURE_HASHES}
    if hashes != PREVIOUS_FAILURE_HASHES:
        raise ValueError('Previous immutable zero-runtime failure bytes changed')
    failure, post = read(directory/'failure.json'), read(directory/'postflight.json')
    if (failure['status'] != 'failed' or failure['engineering_gate'] is not False
            or failure['tasks'] != [] or post['task_attempt_budgets'] != {}
            or post['parent_rng_unchanged'] is not True or post['immutable_input_postflight_passed'] is not True):
        raise ValueError('Previous failure consumed runtime or lacks valid postflight')
    return {'worktree:artifacts/ic/ic02d_eng_v1/'+name: value for name,value in hashes.items()}


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return strict_json(Path(path).read_bytes())


def require_clean():
    if git(ROOT, 'status', '--porcelain') or git(ROOT, 'branch', '--show-current') != 'codex/interaction-curriculum-v1':
        raise ValueError('Commit reviewed collector on the isolated clean research branch')


def prerequisites(baseline):
    design = validate_design(read(ROOT/PROTOCOL['design_config']))
    protocol = validate_protocol(read(ROOT/CONFIG), design)
    if protocol != PROTOCOL:
        raise ValueError('Only the explicitly approved repaired v2 request may be prepared')
    hashes, _, _ = entry_inputs(baseline, ENTRY_SMOKE)
    hashes.update(verify_previous_zero_runtime_failure(ROOT))
    if git(Path(baseline), 'branch', '--show-current') != 'main':
        raise ValueError('Read-only baseline must retain main')
    report_path = ROOT/protocol['design_report']; report = read(report_path)
    if (sha(report_path) != protocol['design_report_sha256'] or report['engineering_gate'] is not True
            or report['resolved_config'] != design or report['runtime_authorized'] is not False
            or report['training_ready'] is not False or report['new_counterfactual_branches'] != 0):
        raise ValueError('Need immutable offline design evidence, not invented runtime release')
    proposals = report_path.parent/'task_proposals.json'
    if sha(proposals) != report['task_proposals_sha256']:
        raise ValueError('Task proposal bytes changed')
    jobs = [{'id': r['proposed_task_id'], 'simulator_seed': r['simulator_seed'],
             'source_episode_id': r['source_episode_id'], 'source_job_id': r['source_job_id'], 'split': 'train'}
            for r in read(proposals)['records'] if r['engineering_stage']]
    if [r['simulator_seed'] for r in jobs] != protocol['simulator_seeds']:
        raise ValueError('Only the explicitly approved three tasks may execute')
    auth = read(ROOT/protocol['authorization'])
    expected = {'protocol_id': protocol['version'], 'run_id': protocol['run_id'],
        'simulator_seeds': protocol['simulator_seeds'], 'maximum_tasks': 3,
        'maximum_fresh_sumo_sessions': 114, 'maximum_control_steps': 12234,
        'maximum_candidate_branches': 90, 'maximum_raw_ticks': 23862,
        'predictor_inference_authorized': False, 'predictor_training_authorized': False,
        'ddpg_training_authorized': False, 'remaining_29_pilot_tasks_authorized': False,
        'task_layer_assignment_authorized': False, 'merge_main_authorized': False,
        'push_authorized': False, 'old_baseline_access': 'read_only'}
    expected.update(previous_failed_run='ic02d_eng_v1', previous_runtime_consumed=False,
                    previous_failure_hashes=PREVIOUS_FAILURE_HASHES)
    if any(auth.get(k) != v for k,v in expected.items()):
        raise ValueError('Human authorization scope changed')
    policy = inside(Path(baseline), design['reference_checkpoint'])
    if sha(policy) != design['reference_sha256']:
        raise ValueError('Frozen author reference checkpoint changed')
    hashes['baseline:'+design['reference_checkpoint']] = sha(policy)
    for name in (CONFIG, PROTOCOL['design_config'], PROTOCOL['design_report'],
                 proposals.relative_to(ROOT).as_posix(), PROTOCOL['authorization'],
                 'tools/collect_curriculum_tasks.py', 'tools/smoke_curriculum_entry.py',
                 'configs/development/ic01_train_interaction_v1.json'):
        hashes['worktree:'+name] = sha(ROOT/name)
    old = read(ROOT/design['ic01_report'])
    if sha(ROOT/design['ic01_report']) != design['ic01_report_sha256']:
        raise ValueError('IC01 normalization source changed')
    hashes['worktree:'+design['ic01_report']] = sha(ROOT/design['ic01_report'])
    metric = validate_config(read(ROOT/'configs/development/ic01_train_interaction_v1.json'))
    if metric != old['resolved_config']:
        raise ValueError('Root-only metric contract changed')
    return protocol, design, jobs, hashes, metric, old['summary']['classification']['scales']


def prepare(baseline):
    require_clean(); baseline = Path(baseline).resolve()
    protocol, design, jobs, hashes, metric, scales = prerequisites(baseline)
    output = ROOT/'artifacts/ic'/protocol['run_id']
    environment = runtime()
    q = {'version': 'ic02d-task-request-v2', 'git_commit_sha': git(ROOT, 'rev-parse', 'HEAD'),
        'working_tree_dirty': False, 'launch_command': subprocess.list2cmdline(sys.orig_argv),
        'prepared_at': now(), 'baseline_root': str(baseline), 'protocol': protocol, 'design': design,
        'jobs': jobs, 'input_hashes': hashes, 'runtime': environment, 'metric_config': metric,
        'normalization_scales': scales, 'human_approved_scope_only': True}
    output.mkdir(parents=True, exist_ok=False)
    write_once(output/'request.json', q)
    write_once(output/'preparation.json', {'status': 'prepared_not_executed', 'request_hash': digest(q)})
    print('task_request='+str(output/'request.json')); print('confirm_request_hash='+digest(q))
    return output/'request.json', digest(q)


def load(path, expected):
    require_clean(); path = Path(path).resolve()
    if path != ROOT/'artifacts/ic'/PROTOCOL['run_id']/'request.json':
        raise ValueError('Single approved run only; renaming cannot reset the budget')
    q = read(path)
    if (q['version'] != 'ic02d-task-request-v2' or digest(q) != expected
            or read(path.parent/'preparation.json')['request_hash'] != expected
            or q['git_commit_sha'] != git(ROOT, 'rev-parse', 'HEAD')
            or q['working_tree_dirty'] is not False or q['human_approved_scope_only'] is not True):
        raise ValueError('Request/hash/source mismatch')
    protocol, design, jobs, hashes, metric, scales = prerequisites(q['baseline_root'])
    if (q['protocol'] != protocol or q['design'] != design or q['jobs'] != jobs
            or q['input_hashes'] != hashes or q['runtime'] != runtime()
            or q['metric_config'] != metric or q['normalization_scales'] != scales):
        raise ValueError('Frozen inputs/environment changed; no automatic adaptation')
    return path, q


def binding(q, job):
    return {'request_hash': digest(q), 'job': job}


def require_fresh_worker_directory(output, expected_binding):
    # Only the parent's three setup files may exist. Prefix matching 's*.json'
    # also matches started.json, and falsely treats a fresh worker as a retry.
    output = Path(output)
    allowed = {'started.json', 'launch.json', 'worker.log'}
    if (not (output/'started.json').is_file()
            or {p.name for p in output.iterdir()}-allowed):
        raise ValueError('Started/incomplete worker cannot be automatically retried')
    if read(output/'started.json')['binding'] != expected_binding:
        raise ValueError('Worker receipt binding changed')


def worker(path, expected, job_id):
    path, q = load(path, expected)
    if (not (path.parent/'started.json').is_file() or (path.parent/'report.json').exists()
            or (path.parent/'failure.json').exists()):
        raise ValueError('Worker requires the authorized parent invocation')
    job = next(j for j in q['jobs'] if j['id'] == job_id)
    output = path.parent/job_id
    require_fresh_worker_directory(output, binding(q, job))
    write_once(output/'worker_started.json', {'binding': binding(q, job), 'started_at': now()})
    torch.set_num_threads(PROTOCOL['torch_threads'])
    from prediction_rl.curriculum.task_collector import collect_task
    collect_task(ROOT, q['baseline_root'], q['design'], job, output, binding(q, job),
        q['metric_config'], q['normalization_scales'], lambda: load(path, expected))


def checked_task(out, q, job):
    directory = out/job['id']; receipt = verify_episode(directory, binding(q, job))
    metadata = receipt['metadata']; post = read(directory/'postflight.json')
    if (metadata['job'] != job or metadata['status'] != 'complete'
            or metadata['training_ready'] is not False or metadata['budget'] != post['budget']
            or not all(post[k] for k in ('worker_session_rng_restored', 'reference_tensors_unchanged', 'immutable_input_postflight_passed'))):
        raise ValueError('Invalid task metadata/postflight')
    if read(directory/'discovery.json') != read(directory/'discovery_repeat.json'):
        raise ValueError('Sealed reference discovery parity failed')
    from prediction_rl.curriculum.task_collector import root_evidence
    from prediction_rl.data.merge_geometry import MergeGeometry
    g = MergeGeometry(ROOT/'RL-MPC-LaneMerging-master/merge.net.xml')
    roots = metadata['roots']
    if [r['target'] for r in roots] != [t['id'] for t in q['design']['checkpoint_proposal']]:
        raise ValueError('Missing/reordered checkpoints')
    for index, row in enumerate(roots):
        d = directory/f'r{index}'
        if row != read(d/'accounting.json'):
            raise ValueError('Root accounting differs from receipt')
        if row['status'] == 'unavailable':
            if row['candidate_count'] != 0 or not row['reason']: raise ValueError('Invalid unavailable root')
            continue
        if row['status'] != 'complete' or row['candidate_count'] != 5 or row['branch_executions'] != 10:
            raise ValueError('Incomplete candidate family')
        branches = {c: read(d/f'c{c}.json') for c in range(5)}
        repeats = {c: read(d/f'c{c}_repeat.json') for c in range(5)}
        if branches != repeats: raise ValueError('Sealed reverse-order branch repeat differs')
        value = root_evidence(read(d/'history.json'), read(d/'labels.json'), branches, repeats, g,
            q['metric_config'], q['normalization_scales'],
            {'task_id': job['id'], 'target': row['target'], 'simulator_seed': job['simulator_seed']})
        if value != read(d/'evidence.json'): raise ValueError('Sealed root evidence cannot be reproduced')
    budget = metadata['budget']
    validate_actual_budget(budget, q['design'])
    for index, row in enumerate(post['session_rows']):
        value = read(directory/f's{index:03d}.json'); events = value['raw_events']['steps']
        if (value['accounting'] != row or row['session_index'] != index or len(events) != row['raw_ticks']
                or row['waiting_insertion_ticks'] != 101
                or not 0 <= len(events)-101-row['control_attempts'] <= 1
                or any(e['index'] != i or abs(e['before']['simulation_time_s']-.2*i) > 1e-7
                       or abs(e['after']['simulation_time_s']-.2*(i+1)) > 1e-7 for i,e in enumerate(events))):
            raise ValueError('Sealed actual session/tick sequence differs from budget')
    if (budget['control_attempts'] != budget['completed_control_steps']
            or budget['raw_tick_attempts'] != budget['completed_raw_ticks']
            or budget['fresh_sumo_sessions'] != len(post['session_rows'])
            or budget['completed_control_steps'] != sum(sum(r['control_steps_by_phase'].values()) for r in post['session_rows'])
            or budget['completed_raw_ticks'] != sum(r['raw_ticks'] for r in post['session_rows'])):
        raise ValueError('Incomplete/different actual budget accounting')
    return metadata


def execute(path, expected, resume=False):
    path, q = load(path, expected); out = path.parent
    if (out/'report.json').exists() or (out/'failure.json').exists():
        raise ValueError('Immutable complete/failed report cannot be rerun')
    report = {'status': 'running', 'request_hash': expected, 'git_commit_sha': q['git_commit_sha'],
        'launch_command': subprocess.list2cmdline(sys.orig_argv), 'started_at': now(), 'tasks': []}
    with run_lock(out):
        if not (out/'started.json').exists(): write_once(out/'started.json', report)
        elif not resume: raise ValueError('Interrupted invocation requires explicit COMPLETE-only reuse')
        error = None; parent_before = rng_fingerprint()
        try:
            for job in q['jobs']:
                d = out/job['id']
                if d.exists():
                    if not resume: raise ValueError('Existing task cannot be overwritten')
                    verify_episode(d, binding(q, job))
                else:
                    d.mkdir(); write_once(d/'started.json', {'binding': binding(q, job), 'started_at': now()})
                    command = [sys.executable, '-B', str(Path(__file__).resolve()), 'worker',
                        '--request', str(path), '--confirm-request-hash', expected, '--job', job['id']]
                    write_once(d/'launch.json', {'command': command})
                    with (d/'worker.log').open('x', encoding='utf-8') as log:
                        subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
                row = checked_task(out, q, job); report['tasks'].append(row)
                print(f"[ic02d] task={job['id']} controls={row['budget']['completed_control_steps']} complete", flush=True)
        except BaseException as caught:
            error = caught; report['traceback'] = traceback.format_exc()
        finally:
            post = {'parent_rng_unchanged': rng_fingerprint() == parent_before, 'finished_at': now()}
            post['task_attempt_budgets'] = {job['id']: read(out/job['id']/'postflight.json')['budget']
                for job in q['jobs'] if (out/job['id']/'postflight.json').is_file()}
            try:
                load(path, expected); post['immutable_input_postflight_passed'] = True
            except BaseException as caught:
                post['immutable_input_postflight_passed'] = False; post['failure_reason'] = str(caught)
                if error is None: error = caught
            write_once(out/'postflight.json', post)
        if error is None and not all(post[k] for k in ('parent_rng_unchanged', 'immutable_input_postflight_passed')):
            error = AssertionError('Parent RNG/files changed during collection')
        if error is not None:
            report.update(status='failed', engineering_gate=False, finished_at=now(), failure_reason=str(error))
            write_once(out/'failure.json', report); raise error
        sessions = sum(r['budget']['fresh_sumo_sessions'] for r in report['tasks'])
        controls = sum(r['budget']['completed_control_steps'] for r in report['tasks'])
        ticks = sum(r['budget']['completed_raw_ticks'] for r in report['tasks'])
        roots = [r for task in report['tasks'] for r in task['roots']]
        branches = sum(r.get('branch_executions', 0) for r in roots)
        if sessions > 114 or controls > 12234 or ticks > 23862 or branches > 90:
            report.update(status='failed', engineering_gate=False, finished_at=now(),
                          failure_reason='Approved aggregate budget exceeded')
            write_once(out/'failure.json', report)
            raise ValueError('Approved budget exceeded; do not report success')
        report.update(status='complete', engineering_gate=True, finished_at=now(),
            actual_tasks=3, fresh_sumo_sessions=sessions, actual_control_steps=controls, actual_raw_ticks=ticks,
            reached_roots=sum(r['status']=='complete' for r in roots), unavailable_roots=sum(r['status']=='unavailable' for r in roots),
            candidate_branch_executions=branches, parent_postflight=post, predictors_loaded=False,
            predictor_training_steps=0, ddpg_training_steps=0, learner_instantiated=False, test_opened=False,
            task_classifier=None, target_distribution_pi=None, training_ready=False,
            remaining_gate='review_three_task_evidence_then_separately_approve_remaining_29_tasks')
        write_once(out/'report.json', report); print('task_engineering_report='+str(out/'report.json'))


def main():
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare'); p.add_argument('--baseline-root', required=True)
    for name in ('run', 'worker'):
        p = sub.add_parser(name); p.add_argument('--request', required=True); p.add_argument('--confirm-request-hash', required=True)
        if name == 'worker': p.add_argument('--job', required=True)
        else: p.add_argument('--resume', action='store_true')
    a = parser.parse_args()
    if a.command == 'prepare': prepare(a.baseline_root)
    elif a.command == 'worker': worker(a.request, a.confirm_request_hash, a.job)
    else: execute(a.request, a.confirm_request_hash, a.resume)


if __name__ == '__main__': main()
