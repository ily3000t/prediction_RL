"""IC03b repair: ONLY the three unstarted arms, six new native episodes max."""
import argparse
from copy import deepcopy
import gzip
import hashlib
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'tools')]
import numpy as np
import smoke_persistent_curriculum as original
from prediction_rl.data.collection_store import inventory, write_once
from prediction_rl.data.dataset_contract import digest
from prediction_rl.data.merge_geometry import MergeGeometry

SCOPE = 'ic03b_approved_remaining_three_arm_smoke_v1'
CONFIG = 'configs/development/ic03b_remaining_smoke_v1.json'
AUTH = 'provenance/ic03b_remaining_smoke_authorization.json'
LEGACY_DIR = ROOT/'artifacts/ic/ic03b_smoke_v1'
LEGACY_HASH = '86a5830a2f4a365eebfbd260cbb284d7febf331875a3d60b4712a4d5dc7f53d2'
LEGACY_INVENTORY = '700d1a2dd590702b92dc13c58941c04dcf1baad6ddebe93bcae9f1169b2276f2'
LEGACY_TOOL_SHA = 'b4ecddac03ca7bd36dc96e6fa72f3343d0d5469081ad8a4c881f6836d057c852'
PROTOCOL = {**deepcopy(original.PROTOCOL), 'protocol_id':SCOPE, 'run_id':'ic03b_remaining_v1',
    'scope':'approved_remaining_engineering_smoke_not_retry_or_formal_training',
    'arms':['b2_curriculum','b3_mix','b3_curriculum'], 'sumo_sessions_max':6,
    'control_steps_max':3000, 'raw_ticks_max':3612, 'feature_calls_max':3006}
CONSUMED = {'sumo_sessions':2, 'control_steps':135, 'raw_ticks':337, 'feature_calls':135,
            'optimizer_updates':0}


def validate_protocol(value):
    if not isinstance(value,dict) or digest(value) != digest(PROTOCOL):
        raise ValueError('Changed approved remaining-arm protocol; no repeat or extra budget')
    return value


def remaining_jobs(tasks):
    plan = original.jobs(tasks)
    result = [j for j in plan if j['id'] in PROTOCOL['arms']]
    if [j['id'] for j in result] != PROTOCOL['arms']:
        raise ValueError('Exactly the three originally unstarted arms are authorized')
    return result


def jsonl(path):
    with gzip.open(path,'rt',encoding='utf-8') as source:
        for line in source: yield original.entry.strict_json(line.encode('utf-8'))


def review_legacy(tasks,current_hashes):
    """Read-only forensic review; NEVER repair or validate the old bad receipt."""
    files = inventory(LEGACY_DIR)
    if len(files) != 21 or digest(files) != LEGACY_INVENTORY:
        raise ValueError('Legacy failed-run bytes/file set changed')
    q = original.entry.read(LEGACY_DIR/'request.json')
    failure = original.entry.read(LEGACY_DIR/'failure.json')
    receipt = original.entry.read(LEGACY_DIR/'b2_mix/complete.json')
    row = original.entry.read(LEGACY_DIR/'b2_mix/summary.json')
    job = original.jobs(tasks)[0]
    if (digest(q) != LEGACY_HASH or q['git_commit_sha'] != 'c64006a55a9b2a388d3b4204a338796950bebd8e'
            or q['protocol'] != original.PROTOCOL or q['jobs'] != original.jobs(tasks)
            or q['smoke_task_roster'] != tasks or failure['status'] != 'failed'
            or failure['engineering_gate'] is not False or failure['request_hash'] != LEGACY_HASH
            or failure['failure_reason'] != 'Episode artifacts changed, missing or unexpectedly added'
            or failure['jobs'] or receipt['binding'] != {'request_hash':LEGACY_HASH,'job':job}
            or receipt['metadata'] != row or row['job'] != job or len(row['episodes']) != 2):
        raise ValueError('Legacy provenance/failure/job changed')
    for key,value in q['input_hashes'].items():
        if key == 'worktree:tools/smoke_persistent_curriculum.py':
            if value != LEGACY_TOOL_SHA: raise ValueError('Unexpected old runner source')
        elif current_hashes.get(key) != value:
            raise ValueError('Legacy scientific input changed: '+key)
    actual = inventory(LEGACY_DIR/'b2_mix')
    changed = {k for k in set(actual)|set(receipt['files']) if actual.get(k) != receipt['files'].get(k)}
    data = (LEGACY_DIR/'b2_mix/worker.log').read_bytes()
    suffix = b'smoke_arm_complete=b2_mix steps=135\r\n'
    if (changed != {'worker.log'} or not data.endswith(suffix)
            or hashlib.sha256(data[:-len(suffix)]).hexdigest() != receipt['files']['worker.log']):
        raise ValueError('Legacy receipt differs by more than the known final stdout line')
    if (row['actual_control_steps'] != 135 or row['raw_ticks'] != 337 or row['feature_calls'] != 135
            or row['optimizer_updates'] != 0 or row['original_minibatches'] != 0
            or row['original_agent_calls'] != 137 or row['formal_training_authorized'] is not False
            or any(row[k] is not True for k in ('persistent_learner_verified','predictor_frozen',
                        'environment_and_features_rng_neutral','source_postflight_passed'))):
        raise ValueError('Legacy consumption or child checks changed')
    geometry = MergeGeometry(ROOT/'RL-MPC-LaneMerging-master/merge.net.xml')
    audits = [original.NativePacketAudit(geometry,
        original.entry.read(LEGACY_DIR/f'b2_mix/e{i}/runtime.json')['settings']) for i in range(2)]
    for saved in jsonl(LEGACY_DIR/'b2_mix/native.jsonl.gz'):
        packet = saved['packet']
        info = {**packet['info'],'prediction_features':saved['feature_info']}
        audits[saved['episode_index']].accept(packet,np.asarray(saved['feature_observation'],np.float32),
                                              packet['reward'],packet['done'],info)
    native_results = [a.result() for a in audits]
    if any(any(native[k] != episode[k] for k in native)
           for native,episode in zip(native_results,row['episodes'])):
        raise ValueError('Independent legacy raw-event/reward review disagrees')
    transitions,episodes = [],[]
    for saved in jsonl(LEGACY_DIR/'b2_mix/replay.jsonl.gz'):
        if saved['kind'] == 'transition': transitions.append(saved['value'])
        elif saved['kind'] == 'episode': episodes.append(saved['value'])
        else: raise ValueError('Legacy smoke unexpectedly sampled minibatches')
    if (len(transitions) != 135 or len(episodes) != 2
            or [r['metadata']['actual_environment_step'] for r in transitions] != list(range(1,136))
            or sum(r['transition']['done'] for r in transitions) != 2
            or row['replay_audit']['reset_bridge_transitions'] != 0
            or row['replay_audit']['original_replay_size'] != 135):
        raise ValueError('Legacy replay/control budget mismatch')
    return {'status':'read_only_forensic_review_passed_not_a_repaired_receipt',
        'legacy_request_hash':LEGACY_HASH, 'legacy_inventory_digest':LEGACY_INVENTORY,
        'legacy_request_remains_failed':True, 'legacy_receipt_remains_invalid':True,
        'legacy_rerun':False, 'consumed':deepcopy(CONSUMED), 'job':job,
        'independent_native_reward_and_tick_review':native_results,
        'child_preexit_tensor_and_rng_checks_recorded':True,
        'posthoc_tensor_or_rng_reconstruction':False,
        'permitted_source_difference':'only_the_reviewed_smoke_coordinator_not_scientific_sources'}


def prerequisites(baseline):
    p = validate_protocol(original.entry.read(ROOT/CONFIG))
    _,h,predictors,trained,tasks,spaces = original.prerequisites(baseline)
    auth = original.entry.read(ROOT/AUTH)
    required = {'protocol_id':SCOPE,'run_id':p['run_id'],'arms':p['arms'],
        'maximum_sumo_sessions':6,'maximum_control_steps':3000,'maximum_raw_ticks':3612,
        'maximum_feature_calls':3006,'episodes_per_arm':2,'previously_consumed_sessions':2,
        'previously_consumed_control_steps':135,'previously_consumed_raw_ticks':337,
        'previously_consumed_feature_calls':135,'prior_failed_run':'artifacts/ic/ic03b_smoke_v1',
        'prior_request_hash':LEGACY_HASH,'receipt_fix_commit':'5cd69c1edf2e14b0f5d3782879a99fe59e24b73a',
        'repeat_b2_mix_authorized':False,'optimizer_updates_authorized':False,
        'formal_training_authorized':False,'predictor_training_authorized':False,
        'new_counterfactual_branches':0,'automatic_retry_authorized':False,
        'old_baseline_access':'read_only','merge_main_authorized':False,'push_authorized':False}
    if any(auth.get(k) != v for k,v in required.items()): raise ValueError('Remaining scope approval changed')
    review = review_legacy(tasks,h)
    for name in (CONFIG,AUTH,'tools/smoke_remaining_curriculum.py'):
        h['worktree:'+name] = original.sha(ROOT/name)
    for name,value in inventory(LEGACY_DIR).items():
        h['worktree:artifacts/ic/ic03b_smoke_v1/'+name] = value
    return p,h,predictors,trained,tasks,spaces,review


def prepare(baseline):
    original.entry.require_clean()
    p,h,predictors,trained,tasks,spaces,review = prerequisites(baseline)
    q = {'version':SCOPE,'protocol':p,'baseline_root':str(Path(baseline).resolve()),
        'git_commit_sha':original.entry.git(ROOT,'rev-parse','HEAD'),'working_tree_dirty':False,
        'prepared_at':original.now(),'launch_command':subprocess.list2cmdline(sys.orig_argv),
        'input_hashes':h,'runtime':original.entry.runtime(),'predictors':predictors,
        'predictor_training_request':trained,'smoke_task_roster':tasks,'native_spaces':spaces,
        'jobs':remaining_jobs(tasks),'prior_evidence_review':review,
        'formal_training_authorized':False,'human_approved_scope_only':True,
        'resolved_ddpg_preset':original.resolved_preset('cpu',.0002,smoke=False),
        'per_job_sampler_contract':{'version':original.SCOPE,'maximum_episodes':2,
            'not_a_parent_budget_authorization':True}}
    out = ROOT/'artifacts/ic'/p['run_id']; out.mkdir(parents=True,exist_ok=False)
    write_once(out/'request.json',q)
    write_once(out/'preparation.json',{'request_hash':digest(q),'status':'prepared_not_executed'})
    print('remaining_smoke_request='+str(out/'request.json'),flush=True)
    print('confirm_request_hash='+digest(q),flush=True)


def load(path,expected):
    original.entry.require_clean(); path = Path(path).resolve()
    if path != ROOT/'artifacts/ic'/PROTOCOL['run_id']/'request.json':
        raise ValueError('One approved remaining request only; no rename/retry')
    q = original.entry.read(path)
    if (q['version'] != SCOPE or digest(q) != expected
            or original.entry.read(path.parent/'preparation.json')['request_hash'] != expected
            or q['git_commit_sha'] != original.entry.git(ROOT,'rev-parse','HEAD')
            or q['working_tree_dirty'] is not False or q['formal_training_authorized'] is not False):
        raise ValueError('Remaining request approval/clean-source/hash changed')
    p,h,predictors,trained,tasks,spaces,review = prerequisites(q['baseline_root'])
    if (q['protocol'] != p or q['input_hashes'] != h or q['runtime'] != original.entry.runtime()
            or q['predictors'] != predictors or q['predictor_training_request'] != trained
            or q['smoke_task_roster'] != tasks or q['native_spaces'] != spaces
            or q['jobs'] != remaining_jobs(tasks) or q['prior_evidence_review'] != review
            or q['resolved_ddpg_preset'] != original.resolved_preset('cpu',.0002,smoke=False)
            or q['per_job_sampler_contract'] != {'version':original.SCOPE,'maximum_episodes':2,
                                                'not_a_parent_budget_authorization':True}):
        raise ValueError('Remaining source/assets/jobs/environment or prior consumption changed')
    return path,q


def execute(path,expected):
    report = original.execute(path,expected,request_loader=load,worker_script=Path(__file__))
    path,q = load(path,expected)
    consumed = {k:CONSUMED[k]+report[{'sumo_sessions':'actual_episodes',
        'control_steps':'actual_control_steps','raw_ticks':'raw_ticks','feature_calls':'feature_calls',
        'optimizer_updates':'optimizer_updates'}[k]] for k in CONSUMED}
    if (consumed['sumo_sessions'] != 8 or consumed['control_steps'] > 4000
            or consumed['raw_ticks'] > 4816 or consumed['feature_calls'] > 4008
            or consumed['optimizer_updates']): raise ValueError('Combined original budget exceeded')
    write_once(path.parent/'coverage_review.json',{'status':'complete',
        'new_remaining_report_sha256':original.sha(path.parent/'report.json'),
        'new_request_hash':expected,'prior_evidence_review':q['prior_evidence_review'],
        'combined_consumption':consumed,'all_four_paths_have_native_integration_evidence':True,
        'legacy_request_remains_failed':True,'legacy_receipt_remains_invalid':True,
        'formal_training_authorized':False,'training_ready':False,
        'remaining_gate':'freeze_IC04_formal_request_then_human_approval'})
    print('coverage_review='+str(path.parent/'coverage_review.json'),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command',required=True)
    p = commands.add_parser('prepare'); p.add_argument('--baseline-root',required=True)
    for name in ('run','worker'):
        p = commands.add_parser(name); p.add_argument('--request',required=True)
        p.add_argument('--confirm-request-hash',required=True)
        if name == 'worker': p.add_argument('--job',required=True,choices=PROTOCOL['arms'])
    args = parser.parse_args()
    if args.command == 'prepare': prepare(args.baseline_root)
    elif args.command == 'run': execute(args.request,args.confirm_request_hash)
    else: original.worker(args.request,args.confirm_request_hash,args.job,
                          request_loader=load,sampler_protocol=original.PROTOCOL)


if __name__ == '__main__': main()
