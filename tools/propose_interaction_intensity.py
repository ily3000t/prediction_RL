"""IC02h: immutable offline L/M/H proposal. Never authorizes training or SUMO."""
import argparse
from pathlib import Path
import platform
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'tools'), str(ROOT/'src')]
import audit_curriculum_tasks as audit
import propose_curriculum_task_rules as prior
from prediction_rl.curriculum.interaction_intensity import PROTOCOL, proposal, validate_protocol
from prediction_rl.data.collection_store import file_hash, run_lock, write_once
from prediction_rl.data.dataset_contract import digest

CONFIG = 'configs/development/ic02h_interaction_intensity_v1.json'
PRIOR_HASHES = {
    'request.json':'90ad7ab968ed4a194ceff74903f19f4ae7daf6c8bf1c13b8828d73302489e86d',
    'report.json':'39782411bca6c11fa94464bcbde69ba19ea9d1852b9ca775e73700680640f92e',
    'task_proposals.json':'7df1bf72d7153db0bddbc7b4eb1556272fa2c5b10064cdb7a4889be23a2fe657',
    'postflight.json':'f46a78b2aca3e6205bab2bb542ae983fddc1e44dc0a1e5d60c2eeacfd505eb04',
}


def verify_prior_hashes(current, sealed):
    added = {'worktree:src/prediction_rl/curriculum/interaction_intensity.py'}
    if (set(current)-set(sealed) != added or set(sealed)-set(current)
            or any(current[k] != v for k,v in sealed.items())):
        raise ValueError('Changed sealed scientific source or unreviewed source addition')


def prerequisites(baseline):
    _, hashes, metric, scales, _ = audit.prerequisites(baseline)
    source = ROOT/'artifacts/ic/ic02f_audit_v1'
    if {name:file_hash(source/name) for name in prior.SOURCE_HASHES} != prior.SOURCE_HASHES:
        raise ValueError('Immutable common-window audit changed')
    qf, rf, roots, pf = [audit.pilot.eng.read(source/name) for name in
                         ('request.json','report.json','roots.json','postflight.json')]
    if (qf['protocol'] != audit.PROTOCOL or rf['request_hash'] != digest(qf)
            or rf['status'] != 'complete' or rf['engineering_gate'] is not True
            or roots['request_hash'] != digest(qf) or rf['parent_postflight'] != pf
            or rf['summary']['global_dense_prefix_ceiling_steps'] != 14
            or qf['metric_config'] != metric or qf['normalization_scales'] != scales
            or not all(pf[k] for k in ('parent_rng_unchanged','immutable_input_postflight_passed'))):
        raise ValueError('Changed source scope, scales or common-window support')
    for name,value in prior.SOURCE_HASHES.items():
        hashes['worktree:artifacts/ic/ic02f_audit_v1/'+name] = value
    for name in (prior.CONFIG, 'tools/propose_curriculum_task_rules.py',
                 'src/prediction_rl/curriculum/task_rule_proposal.py'):
        hashes['worktree:'+name] = file_hash(ROOT/name)
    directory = ROOT/'artifacts/ic/ic02g_rules_v1'
    if {name:file_hash(directory/name) for name in PRIOR_HASHES} != PRIOR_HASHES:
        raise ValueError('Old failed-coverage proposal must remain immutable')
    q, report, tasks, post = [audit.pilot.eng.read(directory/name) for name in PRIOR_HASHES]
    if (q['git_commit_sha'] != '15ef568575bf31850a2634ab56af983c919a44aa'
            or q['resolved_config'] != prior.PROTOCOL or report['request_hash'] != digest(q)
            or report['status'] != 'complete' or report['engineering_gate'] is not True
            or report['parent_postflight'] != post or tasks['request_hash'] != digest(q)
            or report['task_proposals_sha256'] != PRIOR_HASHES['task_proposals.json']
            or report['classification_applied'] is not False or report['training_ready'] is not False
            or any(t['task_level'] is not None or t['classification_applied'] is not False for t in tasks['tasks'])
            or not all(post[k] for k in ('parent_rng_unchanged','immutable_input_postflight_passed'))):
        raise ValueError('Old proposal lineage or authorization changed')
    verify_prior_hashes(hashes, q['input_hashes'])
    for name,value in PRIOR_HASHES.items():
        hashes['worktree:artifacts/ic/ic02g_rules_v1/'+name] = value
    for name in (CONFIG, 'tools/propose_interaction_intensity.py'):
        hashes['worktree:'+name] = file_hash(ROOT/name)
    return validate_protocol(audit.pilot.eng.read(ROOT/CONFIG)), hashes, roots['rows']


def prepare(baseline):
    audit.pilot.eng.require_clean()
    config, hashes, _ = prerequisites(baseline)
    q = {'version':'ic02h-interaction-intensity-request-v1',
        'git_commit_sha':audit.pilot.eng.git(ROOT,'rev-parse','HEAD'), 'working_tree_dirty':False,
        'baseline_root':str(Path(baseline).resolve()), 'resolved_config':config, 'input_hashes':hashes,
        'prepared_at':audit.now(), 'launch_command':subprocess.list2cmdline(sys.orig_argv),
        'environment':{'python':sys.version,'executable':sys.executable,'os':platform.platform()},
        'human_authorization':'implement_new_L_M_H_version_and_offline_validation_no_training',
        'source_read_only':True, 'no_policy_outcomes_read':True}
    out = ROOT/'artifacts/ic'/config['run_id']; out.mkdir(parents=True,exist_ok=False)
    write_once(out/'request.json', q)
    write_once(out/'preparation.json', {'status':'prepared_not_executed','request_hash':digest(q)})
    print('intensity_request='+str(out/'request.json'))
    print('confirm_request_hash='+digest(q))


def load(path, expected):
    audit.pilot.eng.require_clean(); path = Path(path).resolve()
    if path != ROOT/'artifacts/ic'/PROTOCOL['run_id']/'request.json':
        raise ValueError('Only the named immutable IC02h request can run')
    q = audit.pilot.eng.read(path)
    if (q['version'] != 'ic02h-interaction-intensity-request-v1' or digest(q) != expected
            or q['git_commit_sha'] != audit.pilot.eng.git(ROOT,'rev-parse','HEAD')
            or q['working_tree_dirty'] is not False or q['source_read_only'] is not True
            or q['no_policy_outcomes_read'] is not True
            or audit.pilot.eng.read(path.parent/'preparation.json')['request_hash'] != expected):
        raise ValueError('Request/hash/clean source mismatch')
    config, hashes, rows = prerequisites(q['baseline_root'])
    if q['resolved_config'] != config or q['input_hashes'] != hashes:
        raise ValueError('Frozen request inputs changed')
    return path, q, rows


def execute(path, expected):
    path, q, rows = load(path, expected); out = path.parent
    if any((out/name).exists() for name in ('report.json','failure.json','started.json')):
        raise ValueError('No overwrite or automatic retry of a completed/failed proposal')
    with run_lock(out):
        start = audit.now(); write_once(out/'started.json', {'request_hash':expected,'started_at':start})
        before = audit.pilot.eng.rng_fingerprint(); error = None
        try:
            result = proposal(rows, q['resolved_config'])
        except BaseException as caught:
            error = caught; detail = traceback.format_exc()
        post = {'finished_at':audit.now(), 'parent_rng_unchanged':audit.pilot.eng.rng_fingerprint() == before,
                'new_sumo_sessions':0, 'predictor_inference_calls':0, 'training_steps':0}
        try:
            load(path, expected); post['immutable_input_postflight_passed'] = True
        except BaseException as caught:
            post['immutable_input_postflight_passed'] = False
            if error is None: error = caught; detail = traceback.format_exc()
        write_once(out/'postflight.json', post)
        if error is None and not all(post[k] for k in ('parent_rng_unchanged','immutable_input_postflight_passed')):
            error = AssertionError('Changed RNG or inputs'); detail = str(error)
        report = {'request_hash':expected,'git_commit_sha':q['git_commit_sha'], 'started_at':start,
            'finished_at':audit.now(), 'launch_command':subprocess.list2cmdline(sys.orig_argv),
            'resolved_config':q['resolved_config'], 'parent_postflight':post,
            'new_sumo_sessions':0, 'predictor_inference_calls':0, 'ddpg_training_steps':0,
            'predictors_loaded':False, 'learner_instantiated':False, 'test_opened':False,
            'classification_applied':False, 'task_classifier':None, 'target_distribution_pi':None,
            'training_ready':False}
        if error is not None:
            write_once(out/'failure.json', {**report,'status':'failed','engineering_gate':False,
                                          'failure_reason':str(error),'traceback':detail})
            raise error
        write_once(out/'task_proposals.json', {'request_hash':expected,
                   'record_status':'new_intensity_proposal_not_live_training_labels','tasks':result.pop('tasks')})
        write_once(out/'report.json', {**report,**result,'status':'complete','engineering_gate':True,
            'task_proposals_sha256':file_hash(out/'task_proposals.json'),
            'remaining_gate':'review_new_intensity_rule_then_integrate_persistent_learner_and_bounded_engineering_smoke'})
        print('intensity_report='+str(out/'report.json'))
        print('proposed_counts='+str(result['summary']['proposed_counts']))
        print('coverage_gate='+str(result['summary']['coverage_gate']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare'); p.add_argument('--baseline-root',required=True)
    p = sub.add_parser('run'); p.add_argument('--request',required=True); p.add_argument('--confirm-request-hash',required=True)
    args = parser.parse_args()
    if args.command == 'prepare': prepare(args.baseline_root)
    else: execute(args.request, args.confirm_request_hash)


if __name__ == '__main__': main()
