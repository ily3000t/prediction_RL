"""IC02f: immutable, purely offline common-window/role/road-relation audit."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import platform
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'tools'), str(ROOT/'src')]
import collect_curriculum_pilot as pilot
from prediction_rl.curriculum.task_audit import (actor_sets, longest_dense_prefix, neighbor_timelines,
    relation_audit, summarize, window_audit)
from prediction_rl.data.collection_store import file_hash, run_lock, verify_episode, write_once
from prediction_rl.data.dataset_contract import digest
from prediction_rl.data.merge_geometry import MergeGeometry

CONFIG = 'configs/development/ic02f_task_comparability_audit_v1.json'
PROTOCOL = {'schema_version': 1, 'protocol_id': 'ic02f-sealed-train-task-comparability-audit-v1',
    'run_id': 'ic02f_audit_v1', 'scope': 'offline_train_only', 'tasks': 32,
    'root_checkpoints': ['reset', 'early_ramp', 'approach'], 'fixed_window_grid_steps': [5,10,15,25],
    'common_prefix_selection': 'minimum_all_candidate_all_critical_actor_dense_prefix_over_all_96_roots',
    'common_prefix_status': 'exploratory_observed_support_ceiling_not_training_contract',
    'normalization': 'unchanged_IC01_dv_ds', 'actor_sets': ['all_critical','target_front_rear'],
    'task_layer_assignment_authorized': False, 'new_sumo_sessions_authorized': False,
    'predictor_inference_authorized': False, 'predictor_training_authorized': False,
    'ddpg_training_authorized': False, 'training_ready': False, 'task_classifier': None,
    'target_distribution_pi': None}
PILOT_HASHES = {'request.json': '3a3fdafdba6b368d439d379a5b5b0cf1da3c22903cf2c802079f4af57f11a424',
    'report.json': '95511688ee4727b2c04ef4ac4029e46d782ee6d349cb1fb6ac23cd992e97023f',
    'postflight.json': '22ec08066e86ea7104961b0be78d1ebcb33e47c0fcab6b109e84542e568ae77b'}


def now():
    return datetime.now(timezone.utc).isoformat()


def validate_protocol(value):
    if not isinstance(value, dict) or digest(value) != digest(PROTOCOL):
        raise ValueError('Offline audit scope/indicator definitions changed')
    return value


def verify_source(directory, expected, protocol, source_sha, hashes):
    if {name: file_hash(directory/name) for name in expected} != expected:
        raise ValueError('Immutable source report/request/postflight changed')
    q, report = pilot.eng.read(directory/'request.json'), pilot.eng.read(directory/'report.json')
    post = pilot.eng.read(directory/'postflight.json')
    if (q['protocol'] != protocol or q['git_commit_sha'] != source_sha
            or report['request_hash'] != digest(q) or report['status'] != 'complete'
            or report['engineering_gate'] is not True or report['actual_tasks'] != len(q['jobs'])
            or report['training_ready'] is not False or report['predictors_loaded'] is not False
            or report['parent_postflight'] != post
            or not all(post[k] for k in ('parent_rng_unchanged','immutable_input_postflight_passed'))):
        raise ValueError('Incomplete or differently scoped source evidence')
    for name, value in q['input_hashes'].items():
        if name in hashes and hashes[name] != value:
            raise ValueError('Scientific source/asset differs from sealed request: '+name)
    if [row['job'] for row in report['tasks']] != q['jobs']:
        raise ValueError('Top report task roster differs from request')
    for job, row in zip(q['jobs'], report['tasks']):
        receipt = verify_episode(directory/job['id'], pilot.eng.binding(q, job))
        if receipt['metadata'] != row:
            raise ValueError('Top report task differs from sealed metadata')
        hashes['worktree:'+str((directory/job['id']/'complete.json').relative_to(ROOT)).replace('\\','/')] = file_hash(directory/job['id']/'complete.json')
    hashes.update({'worktree:'+str((directory/name).relative_to(ROOT)).replace('\\','/'): value for name,value in expected.items()})
    return q, report


def prerequisites(baseline):
    baseline = Path(baseline).resolve()
    _, design, _, hashes, metric, scales, _ = pilot.prerequisites(baseline)
    protocol = validate_protocol(pilot.eng.read(ROOT/CONFIG))
    sources = []
    for folder, expected, contract, source in (
        ('ic02d_eng_v2', pilot.ENGINEERING_HASHES, pilot.eng.PROTOCOL, '3856a78b97a87cf0cac9280161ce576f8d7f8056'),
        ('ic02e_pilot_v1', PILOT_HASHES, pilot.PROTOCOL, '22b8a7559d6fc65ce34a9f828550b85b50407497')):
        directory = ROOT/'artifacts/ic'/folder
        q, report = verify_source(directory, expected, contract, source, hashes)
        if q['design'] != design or q['metric_config'] != metric or q['normalization_scales'] != scales:
            raise ValueError('Scientific design/normalization changed across task sources')
        sources.append((directory,q,report))
    seeds = [j['simulator_seed'] for _,q,_ in sources for j in q['jobs']]
    if seeds != list(range(11000,11249,8)):
        raise ValueError('Need the complete original 32-task roster, without outcome selection')
    for name in (CONFIG, 'tools/audit_curriculum_tasks.py', 'src/prediction_rl/curriculum/task_audit.py'):
        hashes['worktree:'+name] = file_hash(ROOT/name)
    return protocol, hashes, metric, scales, sources


def prepare(baseline):
    pilot.eng.require_clean()
    protocol, hashes, metric, scales, _ = prerequisites(baseline)
    q = {'version': 'ic02f-offline-audit-request-v1', 'git_commit_sha': pilot.eng.git(ROOT,'rev-parse','HEAD'),
        'working_tree_dirty': False, 'baseline_root': str(Path(baseline).resolve()), 'protocol': protocol,
        'input_hashes': hashes, 'metric_config': metric, 'normalization_scales': scales,
        'prepared_at': now(), 'launch_command': subprocess.list2cmdline(sys.orig_argv),
        'environment': {'python': sys.version, 'executable': sys.executable, 'os': platform.platform()},
        'human_authorization': 'user_continue_after_pure_offline_common_window_role_D_audit_proposal',
        'source_read_only': True}
    out = ROOT/'artifacts/ic'/protocol['run_id']; out.mkdir(parents=True,exist_ok=False)
    write_once(out/'request.json', q)
    write_once(out/'preparation.json', {'status':'prepared_not_executed','request_hash':digest(q)})
    print('audit_request='+str(out/'request.json')); print('confirm_request_hash='+digest(q))


def load(path, expected):
    pilot.eng.require_clean(); path = Path(path).resolve()
    if path != ROOT/'artifacts/ic'/PROTOCOL['run_id']/'request.json':
        raise ValueError('Only the new named offline audit can execute')
    q = pilot.eng.read(path)
    if (q['version'] != 'ic02f-offline-audit-request-v1' or digest(q) != expected
            or q['git_commit_sha'] != pilot.eng.git(ROOT,'rev-parse','HEAD')
            or q['working_tree_dirty'] is not False or q['source_read_only'] is not True
            or pilot.eng.read(path.parent/'preparation.json')['request_hash'] != expected):
        raise ValueError('Offline request/hash/source mismatch')
    p,h,m,s,sources = prerequisites(q['baseline_root'])
    if q['protocol'] != p or q['input_hashes'] != h or q['metric_config'] != m or q['normalization_scales'] != s:
        raise ValueError('Frozen offline inputs changed')
    return path,q,sources


def execute(path, expected):
    path,q,sources = load(path,expected); out = path.parent
    if (out/'report.json').exists() or (out/'failure.json').exists() or (out/'started.json').exists():
        raise ValueError('Audit cannot overwrite or automatically retry an existing run')
    with run_lock(out):
        start = now(); write_once(out/'started.json', {'started_at':start,'request_hash':expected})
        before = pilot.eng.rng_fingerprint(); rows, prepared = [], []
        error = None
        try:
            geometry = MergeGeometry(ROOT/'RL-MPC-LaneMerging-master/merge.net.xml')
            for directory, source, report in sources:
                for job in source['jobs']:
                    pilot.eng.checked_task(directory,source,job)
                    for index,target in enumerate(PROTOCOL['root_checkpoints']):
                        d = directory/job['id']/f'r{index}'
                        evidence = pilot.eng.read(d/'evidence.json')
                        if evidence['target'] != target or evidence['task_level'] is not None or evidence['classification_applied']:
                            raise ValueError('Root identity/layer assignment changed')
                        history,pack = pilot.eng.read(d/'history.json'), pilot.eng.read(d/'labels.json')
                        timelines = neighbor_timelines(pack,{cid:pilot.eng.read(d/f'c{cid}.json') for cid in range(5)},geometry,q['metric_config'])
                        context = evidence['root_context']
                        prefix = longest_dense_prefix(timelines,actor_sets(context)['all_critical'],q['metric_config']['horizon_steps'])
                        if not prefix:
                            raise ValueError('No common dense neighbor prefix; do not fabricate a window')
                        rows.append({'task_id':job['id'],'simulator_seed':job['simulator_seed'],'target':target,
                            'root_id':evidence['root_id'], 'source_request_hash':digest(source),
                            'original_evidence_sha256':file_hash(d/'evidence.json'), 'original_evidence':evidence,
                            'longest_all_critical_dense_prefix_steps':prefix,
                            'windows':{str(n):window_audit(timelines,context,n,q['metric_config'],q['normalization_scales'])
                                for n in PROTOCOL['fixed_window_grid_steps']},
                            'road_relations':relation_audit(history,evidence,geometry)})
                        prepared.append((timelines,context))
                    print('[ic02f] verified_and_audited='+job['id'],flush=True)
            if len(rows) != 96 or len({r['task_id'] for r in rows}) != 32:
                raise ValueError('Missing task/checkpoint; cannot select an easier subset')
            common = min(r['longest_all_critical_dense_prefix_steps'] for r in rows)
            for row,(timelines,context) in zip(rows,prepared):
                row['common_prefix'] = window_audit(timelines,context,common,q['metric_config'],q['normalization_scales'])
            summary = summarize(rows,PROTOCOL['fixed_window_grid_steps'],common)
        except BaseException as caught:
            error = caught; detail = traceback.format_exc()
        post = {'finished_at':now(), 'parent_rng_unchanged':pilot.eng.rng_fingerprint()==before,
                'new_sumo_sessions':0,'predictor_inference_calls':0,'training_steps':0}
        try:
            load(path,expected); post['immutable_input_postflight_passed'] = True
        except BaseException as caught:
            post.update(immutable_input_postflight_passed=False,failure_reason=str(caught))
            if error is None: error = caught; detail = traceback.format_exc()
        write_once(out/'postflight.json',post)
        if error is None and not all(post[k] for k in ('parent_rng_unchanged','immutable_input_postflight_passed')):
            error = AssertionError('Offline audit changed RNG/inputs'); detail = str(error)
        base = {'request_hash':expected,'git_commit_sha':q['git_commit_sha'],'started_at':start,
            'finished_at':now(),'launch_command':subprocess.list2cmdline(sys.orig_argv), 'parent_postflight':post,
            'resolved_config':q['protocol'],'normalization_scales':q['normalization_scales'],
            'new_sumo_sessions':0,'predictor_inference_calls':0,'predictor_training_steps':0,'ddpg_training_steps':0,
            'predictors_loaded':False,'learner_instantiated':False,'test_opened':False,
            'task_classifier':None,'target_distribution_pi':None,'training_ready':False}
        if error is not None:
            write_once(out/'failure.json',{**base,'status':'failed','engineering_gate':False,'failure_reason':str(error),'traceback':detail})
            raise error
        write_once(out/'roots.json',{'request_hash':expected,'rows':rows})
        write_once(out/'report.json',{**base,'status':'complete','engineering_gate':True,
            'roots_sha256':file_hash(out/'roots.json'),'summary':summary,
            'remaining_gate':'propose_task_level_D_I_rules_and_named_finite_pool_distribution_for_human_review'})
        print('offline_audit_report='+str(out/'report.json'))


def main():
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest='command',required=True)
    p = sub.add_parser('prepare'); p.add_argument('--baseline-root',required=True)
    p = sub.add_parser('run'); p.add_argument('--request',required=True); p.add_argument('--confirm-request-hash',required=True)
    args = parser.parse_args()
    if args.command == 'prepare': prepare(args.baseline_root)
    else: execute(args.request,args.confirm_request_hash)


if __name__ == '__main__': main()
