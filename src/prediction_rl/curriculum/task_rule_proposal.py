"""Unapproved train-task rule proposal; never exports live curriculum labels."""
from collections import Counter
from copy import deepcopy

from prediction_rl.curriculum.interaction import distribution, quantile
from prediction_rl.data.dataset_contract import digest, finite

PROTOCOL = {
    'schema_version': 1, 'protocol_id': 'ic02g-task-rule-proposal-v1', 'run_id': 'ic02g_rules_v1',
    'scope': 'offline_train_proposal_review_only', 'task_seeds': list(range(11000,11249,8)),
    'checkpoints': ['reset','early_ramp','approach'], 'anchor': 'approach', 'window_steps': 14,
    'I_aggregation': 'maximum_over_root_critical_actors_of_pair_time_mean_normalized_response',
    'normalization': 'unchanged_IC01_dv_ds', 'noise_floor': 'maximum_original_approach_repeat_noise_bound',
    'lower_quantile': 1/3, 'upper_quantile': 2/3, 'response_epsilon': 1e-10,
    'C0': {'low_I': True, 'require_both_distinct_target_actors': True,
           'require_positive_minimum_route_projected_gap': True,
           'maximum_closing_speed_rule': 'at_most_lower_train_task_tercile'},
    'C1': {'high_I': True, 'strictly_above_repeat_noise': True, 'maximum_critical_actors': 3},
    'priority': ['C0','C1','C2'], 'boundary_ties': 'inclusive_quantiles_no_task_id_tiebreak',
    'minimum_unique_tasks_per_level': 10,
    'target_proposal': 'finite_train_task32_gym20_uniform_v1',
    'target_task_weight': '1_over_32', 'original_continuing_traffic_distribution_claim': False,
    'task_layer_assignment_authorized': False, 'schedule_execution_authorized': False,
    'new_sumo_sessions_authorized': False, 'predictor_inference_authorized': False,
    'predictor_training_authorized': False, 'ddpg_training_authorized': False,
    'training_ready': False, 'task_classifier': None, 'target_distribution_pi': None,
}


def validate_protocol(value):
    if not isinstance(value, dict) or digest(value) != digest(PROTOCOL):
        raise ValueError('Changed the single unapproved task-rule proposal')
    return value


def task_features(family, config):
    """No selection of easiest root or predictor/policy outcomes."""
    if [r['target'] for r in family] != config['checkpoints']:
        raise ValueError('Require all three original checkpoints, in frozen order')
    task = family[0]['task_id']; seed = family[0]['simulator_seed']
    if any(r['task_id'] != task or r['simulator_seed'] != seed for r in family):
        raise ValueError('Mixed source tasks')
    for r in family:
        e = r['original_evidence']
        if (e['task_id'] != task or e['simulator_seed'] != seed or e['target'] != r['target']
                or e['task_level'] is not None or e['classification_applied'] is not False
                or e['training_ready'] is not False):
            raise ValueError('Need original unclassified root evidence')
    anchor = family[config['checkpoints'].index(config['anchor'])]
    context = anchor['original_evidence']['root_context']; window = anchor['common_prefix']
    if window['steps'] != config['window_steps']:
        raise ValueError('Do not adapt the proposal window to each task')
    actors = context['actor_ids']
    if len(set(actors)) != len(actors) or 'ego' in actors or set(window['per_actor']) != set(actors):
        raise ValueError('Changed critical actor set or included ego')
    covered = not context['coverage']['key_actor_missing_slots']
    available = bool(actors) and covered and window['sets']['all_critical']['status'] == 'available'
    per_actor = {}
    for actor in actors:
        value = window['per_actor'][actor]
        if value['roles'] != context['roles_by_actor'][actor]:
            raise ValueError('Roles changed after reading the candidate future')
        response = value['response']
        if (response['actor_ids'] != [actor] or response['horizon_steps'] != config['window_steps']
                or response['status'] not in ('available','unavailable')):
            raise ValueError('Malformed per-actor response contract')
        if response['status'] == 'available':
            score = finite(response['I_dense_with_frozen_IC01_scales'])
            if score < 0 or response['interaction']['full_common_horizon'] is not True:
                raise ValueError('Invalid dense response')
            per_actor[actor] = score
        else:
            available = False; per_actor[actor] = None
    peak = max(per_actor.values()) if available else None
    maxima = sorted(a for a,v in per_actor.items() if peak is not None and v == peak)
    target_rows = anchor['road_relations']['target_relations']
    targets = [target_rows[name] for name in ('target_front','target_rear')]
    target_ids = [r['actor_id'] for r in targets]
    distinct = all(a is not None for a in target_ids) and len(set(target_ids)) == 2
    gap = min(finite(r['route_projected_gap_m']) for r in targets) if distinct else None
    closing = max(finite(r['closing_speed_mps']) for r in targets) if distinct else None
    if closing is not None and closing < 0:
        raise ValueError('Closing proxy must be nonnegative')
    difficulty = anchor['road_relations']['original_difficulty_unchanged']
    if difficulty['critical_actor_count'] != len(actors):
        raise ValueError('Critical count does not match the root-selected actor set')
    noise = anchor['original_evidence']['repeat_noise_with_frozen_IC01_scales']
    if noise is not None and finite(noise) < 0:
        raise ValueError('Invalid repeat-noise bound')
    return {'task_id': task, 'simulator_seed':seed, 'source_episode_id':anchor['original_evidence']['episode_id'],
        'anchor_root_id':anchor['root_id'], 'root_ids':{r['target']:r['root_id'] for r in family},
        'I_available':available, 'I_peak':peak, 'per_actor_I':per_actor, 'peak_actor_ids':maxima,
        'peak_actor_roles':{a:context['roles_by_actor'][a] for a in maxima}, 'repeat_noise_bound':noise,
        'distinct_target_front_rear':distinct, 'minimum_projected_opportunity_gap_m':gap,
        'maximum_target_closing_speed_mps':closing, 'critical_actor_count':len(actors),
        'target_path_relations':{name:r['path_relation'] for name,r in target_rows.items()},
        'difficulty_components':deepcopy(difficulty), 'predictor_slot_coverage':deepcopy(context['coverage']),
        'source_request_hashes':sorted({r['source_request_hash'] for r in family}),
        'reference_prefix_before_training_policy_steps':0,
        'task_level':None, 'classification_applied':False, 'execution_authorized':False, 'training_ready':False}


def proposed_decision(task, thresholds, config):
    low = task['I_available'] and thresholds['I_low'] is not None and task['I_peak'] <= thresholds['I_low']
    high = task['I_available'] and thresholds['I_high'] is not None and task['I_peak'] >= thresholds['I_high']
    gap = task['minimum_projected_opportunity_gap_m']
    closing = task['maximum_target_closing_speed_mps']
    above_noise = (task['I_available'] and thresholds['repeat_noise_floor'] is not None
                   and task['I_peak'] > thresholds['repeat_noise_floor']+config['response_epsilon'])
    criteria = {'response_available':task['I_available'], 'low_response':low, 'high_response':high,
        'distinct_target_front_rear':task['distinct_target_front_rear'],
        'open_projected_opportunity':gap is not None and gap > 0,
        'low_closing_proxy':closing is not None and thresholds['closing_low'] is not None and closing <= thresholds['closing_low'],
        'response_above_noise':above_noise,
        'few_critical_actors':task['critical_actor_count'] <= config['C1']['maximum_critical_actors']}
    c0 = all(criteria[k] for k in ('response_available','low_response','distinct_target_front_rear',
                                  'open_projected_opportunity','low_closing_proxy'))
    c1 = all(criteria[k] for k in ('response_available','high_response','response_above_noise','few_critical_actors'))
    level = 'C0' if c0 else 'C1' if c1 else 'C2'
    return {'proposed_level':level, 'criteria':criteria, 'matches_C0':c0, 'matches_C1':c1,
            'reason':'weak_response_and_open_low_closing_projection' if c0 else
                     'strong_identifiable_neighbor_response_with_few_critical_actors' if c1 else
                     'unavailable_response' if not task['I_available'] else 'remaining_complete_task'}


def proposal(rows, config):
    validate_protocol(config)
    families = {}
    for row in rows:
        families.setdefault(row['simulator_seed'], []).append(row)
    if sorted(families) != config['task_seeds'] or len(rows) != len(families)*3:
        raise ValueError('Require the full 32-task roster; no selection or replacement')
    tasks = [task_features(sorted(families[s],key=lambda r:config['checkpoints'].index(r['target'])),config)
             for s in config['task_seeds']]
    if len({t['task_id'] for t in tasks}) != len(tasks) or len({t['source_episode_id'] for t in tasks}) != len(tasks):
        raise ValueError('Duplicate task/episode cannot count as independent task coverage')
    thresholds = {'I_low':quantile([t['I_peak'] for t in tasks],config['lower_quantile']),
        'I_high':quantile([t['I_peak'] for t in tasks],config['upper_quantile']),
        'closing_low':quantile([t['maximum_target_closing_speed_mps'] for t in tasks],config['lower_quantile']),
        'repeat_noise_floor':max((t['repeat_noise_bound'] for t in tasks if t['repeat_noise_bound'] is not None),default=None),
        'projected_opportunity_gap_boundary_m':0., 'threshold_status':'unapproved_train_only_proposal'}
    for task in tasks:
        task.update(proposed_decision(task,thresholds,config))
    counts = {level:sum(t['proposed_level']==level for t in tasks) for level in ('C0','C1','C2')}
    minimum = config['minimum_unique_tasks_per_level']
    empty = [k for k,n in counts.items() if n == 0]
    sparse = [k for k,n in counts.items() if n < minimum]
    mass = {k:n/len(tasks) for k,n in counts.items()}
    mix = {'C0':.2+mass['C0']/6, 'C1':.35+mass['C1']/6, 'C2':17/60+mass['C2']/6}
    summary = {'unique_tasks':len(tasks),'proposed_counts':counts,'empty_levels':empty,
        'below_minimum_levels':sparse,'coverage_gate':not sparse,
        'fixed_three_level_schedule_mathematically_schedulable':not empty,
        'criteria_marginal_counts':dict(Counter(k for t in tasks for k,v in t['criteria'].items() if v)),
        'C0_open_and_low_closing_tasks':sum(t['criteria']['open_projected_opportunity'] and t['criteria']['low_closing_proxy'] for t in tasks),
        'proposed_uniform_task_layer_mass':mass,'proposed_Mix_nominal_layer_mass':mix,
        'distribution_proposal_name':config['target_proposal'],'distribution_frozen':False,
        'I_peak_distribution':distribution(t['I_peak'] for t in tasks),
        'closing_proxy_distribution':distribution(t['maximum_target_closing_speed_mps'] for t in tasks),
        'minimum_projected_gap_distribution':distribution(t['minimum_projected_opportunity_gap_m'] for t in tasks),
        'candidate_level_task_ids':{k:[t['task_id'] for t in tasks if t['proposed_level']==k] for k in counts},
        'manual_review_required':True,'task_classifier':None,'target_distribution_pi':None,'training_ready':False}
    return {'threshold_proposal':thresholds,'tasks':tasks,'summary':summary}
