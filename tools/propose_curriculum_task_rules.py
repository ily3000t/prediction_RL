"""IC02g: one fixed offline task-rule proposal, not a training manifest."""
import argparse
from pathlib import Path
import platform
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'tools'),str(ROOT/'src')]
import audit_curriculum_tasks as audit
from prediction_rl.curriculum.task_rule_proposal import PROTOCOL, proposal, validate_protocol
from prediction_rl.data.collection_store import file_hash, run_lock, write_once
from prediction_rl.data.dataset_contract import digest

CONFIG = 'configs/development/ic02g_task_rule_proposal_v1.json'
SOURCE_HASHES = {
    'request.json':'79163f9cc7dbe3e3a2399efb31689dea67733f6e955f3c34b796f916e32fe83f',
    'report.json':'17fb033e10696fcfbab4b56d5094bd13c03933988ebdf71df6117ea4914b185f',
    'roots.json':'accad86a94413240a87a9758b65e17c73c75bda52cbc84b1eb7dbf4b8542b21b',
    'postflight.json':'25d3a1a530ccdc6a724eb891ec6d6bfbb7eea99cff2c3f58355f629c11244442'}


def verify_prior_hashes(current, prior):
    # entry_inputs inventories every project source. Exactly ONE new module is
    # added by this proposal; never accept changed old bytes or arbitrary extras.
    added = {'worktree:src/prediction_rl/curriculum/task_rule_proposal.py'}
    if set(current)-set(prior) != added or set(prior)-set(current) or any(current[k] != v for k,v in prior.items()):
        raise ValueError('Changed old scientific inputs or unreviewed new source')


def prerequisites(baseline):
    _,hashes,metric,scales,_ = audit.prerequisites(baseline)
    directory = ROOT/'artifacts/ic/ic02f_audit_v1'
    if {name:file_hash(directory/name) for name in SOURCE_HASHES} != SOURCE_HASHES:
        raise ValueError('Immutable IC02f source bytes changed')
    q,r,rows,post = [audit.pilot.eng.read(directory/name) for name in ('request.json','report.json','roots.json','postflight.json')]
    if (q['git_commit_sha'] != '24b908dad8220de8c0778a730cc717a8210056ab' or q['protocol'] != audit.PROTOCOL
            or r['status'] != 'complete' or r['engineering_gate'] is not True or r['request_hash'] != digest(q)
            or rows['request_hash'] != digest(q) or r['roots_sha256'] != SOURCE_HASHES['roots.json']
            or r['parent_postflight'] != post or r['summary']['global_dense_prefix_ceiling_steps'] != 14
            or r['training_ready'] is not False
            or not all(post[k] for k in ('parent_rng_unchanged','immutable_input_postflight_passed'))):
        raise ValueError('Invalid audit source or silent replacement of observed support window')
    verify_prior_hashes(hashes,q['input_hashes'])
    if q['metric_config'] != metric or q['normalization_scales'] != scales:
        raise ValueError('Scientific source/current receipts differ from the completed audit')
    protocol = validate_protocol(audit.pilot.eng.read(ROOT/CONFIG))
    for name,value in SOURCE_HASHES.items():
        hashes['worktree:artifacts/ic/ic02f_audit_v1/'+name] = value
    for name in (CONFIG,'tools/propose_curriculum_task_rules.py','src/prediction_rl/curriculum/task_rule_proposal.py'):
        hashes['worktree:'+name] = file_hash(ROOT/name)
    return protocol,hashes,rows['rows']


def prepare(baseline):
    audit.pilot.eng.require_clean(); protocol,hashes,_ = prerequisites(baseline)
    q = {'version':'ic02g-task-rule-request-v1','git_commit_sha':audit.pilot.eng.git(ROOT,'rev-parse','HEAD'),
        'working_tree_dirty':False,'baseline_root':str(Path(baseline).resolve()),'resolved_config':protocol,
        'input_hashes':hashes,'prepared_at':audit.now(),'launch_command':subprocess.list2cmdline(sys.orig_argv),
        'environment':{'python':sys.version,'executable':sys.executable,'os':platform.platform()},
        'human_authorization':'continue_with_offline_task_rule_proposal_and_coverage_check_not_live_assignment',
        'no_policy_outcomes_read':True}
    out = ROOT/'artifacts/ic'/protocol['run_id']; out.mkdir(parents=True,exist_ok=False)
    write_once(out/'request.json',q); write_once(out/'preparation.json',{'request_hash':digest(q),'status':'prepared_not_executed'})
    print('proposal_request='+str(out/'request.json')); print('confirm_request_hash='+digest(q))


def load(path,expected):
    audit.pilot.eng.require_clean(); path = Path(path).resolve()
    if path != ROOT/'artifacts/ic'/PROTOCOL['run_id']/'request.json':
        raise ValueError('Only the fixed proposal request is permitted')
    q = audit.pilot.eng.read(path)
    if (q['version'] != 'ic02g-task-rule-request-v1' or digest(q) != expected
            or q['git_commit_sha'] != audit.pilot.eng.git(ROOT,'rev-parse','HEAD') or q['working_tree_dirty'] is not False
            or q['no_policy_outcomes_read'] is not True or audit.pilot.eng.read(path.parent/'preparation.json')['request_hash'] != expected):
        raise ValueError('Proposal request/hash/source mismatch')
    protocol,hashes,rows = prerequisites(q['baseline_root'])
    if q['resolved_config'] != protocol or q['input_hashes'] != hashes:
        raise ValueError('Frozen proposal inputs changed')
    return path,q,rows


def execute(path,expected):
    path,q,rows = load(path,expected); out = path.parent
    if any((out/name).exists() for name in ('report.json','failure.json','started.json')):
        raise ValueError('No overwrite, alternative threshold retry or automatic rerun')
    with run_lock(out):
        start = audit.now(); write_once(out/'started.json',{'request_hash':expected,'started_at':start})
        before = audit.pilot.eng.rng_fingerprint(); error = None
        try:
            result = proposal(rows,q['resolved_config'])
        except BaseException as caught:
            error = caught; detail = traceback.format_exc()
        post = {'finished_at':audit.now(),'parent_rng_unchanged':audit.pilot.eng.rng_fingerprint()==before,
                'new_sumo_sessions':0,'predictor_inference_calls':0,'training_steps':0}
        try:
            load(path,expected); post['immutable_input_postflight_passed'] = True
        except BaseException as caught:
            post['immutable_input_postflight_passed'] = False
            if error is None: error = caught; detail = traceback.format_exc()
        write_once(out/'postflight.json',post)
        if error is None and not all(post[k] for k in ('parent_rng_unchanged','immutable_input_postflight_passed')):
            error = AssertionError('Proposal audit changed RNG/inputs'); detail = str(error)
        report = {'request_hash':expected,'git_commit_sha':q['git_commit_sha'],'started_at':start,'finished_at':audit.now(),
            'launch_command':subprocess.list2cmdline(sys.orig_argv),'resolved_config':q['resolved_config'],'parent_postflight':post,
            'new_sumo_sessions':0,'predictor_inference_calls':0,'predictor_training_steps':0,'ddpg_training_steps':0,
            'predictors_loaded':False,'test_opened':False,'classification_applied':False,
            'task_classifier':None,'target_distribution_pi':None,'training_ready':False}
        if error is not None:
            write_once(out/'failure.json',{**report,'status':'failed','engineering_gate':False,'failure_reason':str(error),'traceback':detail})
            raise error
        write_once(out/'task_proposals.json',{'request_hash':expected,'record_status':'unapproved_not_executable','tasks':result.pop('tasks')})
        write_once(out/'report.json',{**report,**result,'status':'complete','engineering_gate':True,
            'task_proposals_sha256':file_hash(out/'task_proposals.json'),
            'remaining_gate':'human_review_of_task_rule_and_coverage_before_any_live_labels_or_schedule'})
        print('proposal_report='+str(out/'report.json'))
        print('proposed_counts='+str(result['summary']['proposed_counts']))
        print('coverage_gate='+str(result['summary']['coverage_gate']))


def main():
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest='command',required=True)
    p = sub.add_parser('prepare'); p.add_argument('--baseline-root',required=True)
    p = sub.add_parser('run'); p.add_argument('--request',required=True); p.add_argument('--confirm-request-hash',required=True)
    args = parser.parse_args()
    if args.command == 'prepare': prepare(args.baseline_root)
    else: execute(args.request,args.confirm_request_hash)


if __name__ == '__main__': main()
