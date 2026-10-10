"""Offline task-characterization proposal; never assigns a training task layer."""
from collections import Counter

from prediction_rl.curriculum.interaction import root_context
from prediction_rl.data.actor_adapter import build_actor_inputs
from prediction_rl.data.dataset_contract import digest

DESIGN_SHA256 = '6e6052947d25ce8ca45fb2185d58402440fde9717eb5eb4575978a9ebe45276c'


def validate_design(value):
    if not isinstance(value, dict) or digest(value) != DESIGN_SHA256:
        raise ValueError('Changed/unknown task design; no runtime authorization from configuration')
    if (value['scope'] != 'offline_design_only' or value['runtime_authorized'] is not False
            or value['training_ready'] is not False or value['task_classifier'] is not None
            or value['target_distribution_pi'] is not None or value['test_opened'] is not False):
        raise ValueError('This design cannot release collection or curriculum training')
    return value


def proposal_jobs(mapping, config):
    records = mapping['records']
    low, high = config['source_train_seed_range']
    by_seed = {row['simulator_seed']: row for row in records}
    if (len(by_seed) != len(records) or sorted(by_seed) != list(range(low, high+1))
            or mapping['classified_executable_tasks'] != 0 or mapping['fresh_task_target_pi'] is not None
            or any(row['task_level'] is not None or row['execution_authorized'] is not False for row in records)):
        raise ValueError('Need the complete train source inventory, not selected positive-result tasks')
    result = []
    for seed in range(low, high+1, config['pilot_seed_stride']):
        row = by_seed[seed]
        result.append({'proposed_task_id': f't{seed}', 'simulator_seed': seed, 'episode_index': 0,
            'source_episode_id': row['episode_id'], 'source_job_id': row['job_id'],
            'entry_contract': config['entry_contract'], 'task_level': None,
            'execution_authorized': False, 'engineering_stage': seed in config['engineering_seeds'],
            'selection': 'fixed_stride_in_complete_train_seed_roster_not_policy_outcomes'})
    if not set(config['engineering_seeds']) <= {row['simulator_seed'] for row in result}:
        raise ValueError('Engineering stage is not contained in the pilot')
    return result


def replay_budget(task_count, config):
    if type(task_count) is not int or task_count < 1:
        raise ValueError('Need a positive finite task count')
    # reset has prefix=0; both lane checkpoints may be at the discovery cap.
    prefix_caps = [0 if p['kind'] == 'original_reset' else config['discovery_max_steps']
                   for p in config['checkpoint_proposal']]
    roots = len(prefix_caps)
    branches_per_root = len(config['probe_jerks'])*config['branch_repeats']
    sessions = task_count*(config['discovery_repeats']
        + roots*config['root_capture_repeats'] + roots*branches_per_root)
    discovery = task_count*config['discovery_repeats']*config['discovery_max_steps']
    capture = task_count*config['root_capture_repeats']*sum(prefix_caps)
    branch_prefix = task_count*branches_per_root*sum(prefix_caps)
    futures = task_count*roots*branches_per_root*config['horizon_steps']
    controls = discovery+capture+branch_prefix+futures
    waiting = sessions*config['reset_warmup_and_insertion_ticks']
    cleanup = sessions*config['cleanup_ticks_per_session_upper_bound']
    return {'execution_authorized': False, 'unique_training_tasks': task_count,
        'reference_checkpoint_slots_max': task_count*roots,
        'candidate_branch_executions_max': task_count*roots*branches_per_root,
        'fresh_sumo_sessions_max': sessions, 'reference_discovery_control_steps_max': discovery,
        'root_capture_prefix_control_steps_max': capture, 'branch_replay_prefix_control_steps_max': branch_prefix,
        'candidate_future_control_steps_max': futures, 'total_control_steps_max': controls,
        'warmup_and_insertion_ticks_max': waiting, 'cleanup_ticks_max': cleanup,
        'total_raw_ticks_max': controls+waiting+cleanup,
        'author_reference_policy_calls_max': discovery, 'frozen_predictor_inference_calls': 0,
        'learner_transitions': 0, 'independent_unit': 'training_seed_first_episode_not_branches_or_replays'}


def existing_root_coverage(rows, report):
    if (len({r['root_id'] for r in rows}) != len(rows) or len(rows) != report['summary']['reached_roots']
            or len({r['episode_id'] for r in rows}) != report['summary']['episodes']
            or dict(Counter(r['level'] for r in rows)) != {
                k: v['roots'] for k,v in report['summary']['classification']['levels'].items() if v['roots']}):
        raise ValueError('Old train root evidence/report binding changed')
    result = {}
    for target in sorted({r['target'] for r in rows}):
        values = [r['task_mapping']['reference_prefix_steps'] for r in rows if r['target'] == target]
        if any(type(step) is not int or step < 0 for step in values):
            raise ValueError('Invalid actual reference-prefix accounting')
        result[target] = {'roots': len(values), 'prefix_steps_min': min(values), 'prefix_steps_max': max(values)}
    return {'source_episodes': report['summary']['episodes'], 'roots': len(rows), 'by_checkpoint': result,
        'roots_at_original_reset': sum(r['task_mapping']['reference_prefix_steps'] == 0 for r in rows),
        'old_root_levels_preserved': report['summary']['classification']['levels'],
        'old_root_thresholds_changed': False, 'executable_task_levels_assigned': 0,
        'target_distribution_pi': None}


def initial_description(trace, geometry, root_config):
    packet = trace['packets'][0]
    if packet['phase'] != 'reset' or packet['control_step'] != 0 or packet['done'] or packet['reward'] is not None:
        raise ValueError('Require the real nonterminal reset packet, not a later reference root')
    frame = packet['current_frame']
    inputs = build_actor_inputs([frame], geometry, capacity=12, history_steps=11, tick_s=.2)
    history = {'history': [frame], 'inputs': inputs}
    context = root_context(history, geometry, root_config)
    return {'simulator_seed': trace['job']['simulator_seed'], 'initial_traffic_hash': digest(frame),
        'initial_observation_hash': digest(packet['observation']), 'history_frames': 1,
        'root_context': context, 'task_level': None, 'neighbor_response_I': None,
        'classification_status': 'observable_entry_only_no_candidate_future_evidence',
        'not_target_distribution_estimate': True}
