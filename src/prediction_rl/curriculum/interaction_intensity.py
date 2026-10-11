"""Versioned L/M/H proposal and offline scheduling primitives; no learner/env imports."""
from collections import Counter, deque
from copy import deepcopy
import math
import random

from prediction_rl.curriculum.interaction import distribution, quantile
from prediction_rl.curriculum.task_rule_proposal import task_features
from prediction_rl.data.dataset_contract import digest, finite

LEVELS = ('L', 'M', 'H')
PROTOCOL = {
    'schema_version': 1, 'protocol_id': 'ic02h-interaction-intensity-proposal-v1',
    'run_id': 'ic02h_intensity_v1', 'scope': 'offline_train_proposal_and_sampler_review_only',
    'task_seeds': list(range(11000, 11249, 8)),
    'checkpoints': ['reset', 'early_ramp', 'approach'], 'anchor': 'approach', 'window_steps': 14,
    'I_aggregation': 'maximum_over_root_critical_actors_of_pair_time_mean_normalized_response',
    'normalization': 'unchanged_IC01_dv_ds',
    'label_meaning': 'reference_policy_conditioned_task_interaction_potential_not_control_difficulty',
    'noise_floor': 'maximum_original_approach_repeat_noise_bound',
    'lower_quantile': 1/3, 'upper_quantile': 2/3, 'response_epsilon': 1e-10,
    'boundary_ties': 'inclusive_outer_quantiles_no_id_tiebreak_collapsed_boundaries_rejected',
    'unavailable_response': 'unclassified_blocks_freeze_never_imputed_as_L_M_or_H',
    'D_use': 'independent_description_only_not_a_classifier_or_filter',
    'minimum_unique_tasks_per_level': 10,
    'target_proposal': 'finite_train_task32_gym20_uniform_v1',
    'target_task_weight': '1_over_32', 'within_level_sampling': 'uniform_task',
    'original_continuing_traffic_distribution_claim': False,
    'schedule': [
        {'start': 0, 'end': 20000, 'mass': {'L': .6, 'M': .3, 'H': .1}},
        {'start': 20000, 'end': 60000, 'mass': {'L': .2, 'M': .2, 'H': .6}},
        {'start': 60000, 'end': 120000, 'mass': 'finite_pool_target'},
    ],
    'schedule_clock': 'actual_control_transitions_at_episode_boundary',
    'Mix_rule': 'stage_step_length_weighted_nominal_reset_mass_not_exposure_equality',
    'matched_arms': ['b2_mix', 'b2_curriculum', 'b3_mix', 'b3_curriculum'],
    'optimizer_seeds': [0, 1, 2], 'requested_control_steps_per_model': 120000,
    'evaluation_nodes': [20000, 40000, 60000, 120000], 'evaluation_seeds': list(range(200, 220)),
    'main_endpoint': 120000, 'entry_contract': 'seeded_fresh_gym20_task_v1',
    'reference_prefix_before_training_policy_steps': 0,
    'replay_capacity': 1000000, 'replay_sampling': 'unchanged_uniform',
    'clear_replay_at_stage_change': False, 'curriculum_metadata_in_observation': False,
    'task_layer_assignment_authorized': False, 'schedule_execution_authorized': False,
    'new_sumo_sessions_authorized': False, 'predictor_inference_authorized': False,
    'predictor_training_authorized': False, 'ddpg_training_authorized': False,
    'training_ready': False, 'task_classifier': None, 'target_distribution_pi': None,
}


def validate_protocol(value):
    if not isinstance(value, dict) or digest(value) != digest(PROTOCOL):
        raise ValueError('Changed interaction-intensity proposal; create a new version')
    return value


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError('Invalid integer '+name)
    return value


def _mass(value):
    if not isinstance(value, dict) or set(value) != set(LEVELS):
        raise ValueError('Require exactly L/M/H masses')
    result = {k: finite(value[k]) for k in LEVELS}
    if any(v < 0 for v in result.values()) or not math.isclose(math.fsum(result.values()), 1., abs_tol=1e-12):
        raise ValueError('Layer masses must be nonnegative and sum to one; no silent normalization')
    return result


def schedule_proposal(target):
    target = _mass(target)
    stages = [{**deepcopy(s), 'mass': target.copy() if s['mass'] == 'finite_pool_target' else dict(s['mass'])}
              for s in PROTOCOL['schedule']]
    mix = {k: math.fsum((s['end']-s['start'])*s['mass'][k] for s in stages)/120000 for k in LEVELS}
    return {'stages': stages, 'nominal_Mix_reset_mass': mix,
            'within_level_sampling': 'uniform_task', 'actual_exposure_equality_claim': False}


def proposed_decision(task, thresholds):
    if not task['I_available']:
        return {'proposed_level': None, 'reason': 'response_unavailable_blocks_freeze'}
    if not thresholds['complete_source_support'] or not thresholds['separated_terciles']:
        return {'proposed_level': None, 'reason': 'incomplete_support_or_collapsed_boundaries'}
    value = finite(task['I_peak'])
    if value <= thresholds['I_low']:
        return {'proposed_level': 'L', 'reason': 'weak_reference_conditioned_response'}
    if value >= thresholds['I_high']:
        if value <= thresholds['repeat_noise_floor']+PROTOCOL['response_epsilon']:
            return {'proposed_level': None, 'reason': 'upper_tercile_not_distinguishable_from_repeat_noise'}
        return {'proposed_level': 'H', 'reason': 'high_identifiable_reference_conditioned_response'}
    return {'proposed_level': 'M', 'reason': 'intermediate_reference_conditioned_response'}


def proposal(rows, config):
    validate_protocol(config)
    families = {}
    for row in rows:
        families.setdefault(row['simulator_seed'], []).append(row)
    if sorted(families) != config['task_seeds'] or len(rows) != len(families)*3:
        raise ValueError('Require all 32 original tasks without selection or replacement')
    tasks = [task_features(sorted(families[s], key=lambda r:config['checkpoints'].index(r['target'])), config)
             for s in config['task_seeds']]
    if len({t['task_id'] for t in tasks}) != 32 or len({t['source_episode_id'] for t in tasks}) != 32:
        raise ValueError('Duplicate source tasks/episodes cannot count as coverage')
    complete = all(t['I_available'] and t['repeat_noise_bound'] is not None for t in tasks)
    low = quantile([t['I_peak'] for t in tasks], 1/3) if complete else None
    high = quantile([t['I_peak'] for t in tasks], 2/3) if complete else None
    thresholds = {'I_low': low, 'I_high': high,
        'repeat_noise_floor': max(t['repeat_noise_bound'] for t in tasks) if complete else None,
        'complete_source_support': complete, 'separated_terciles': complete and low < high,
        'threshold_status': 'train_only_unapproved_new_version_proposal'}
    for task in tasks:
        task.update(proposed_decision(task, thresholds))
        task['label_meaning'] = config['label_meaning']
    counts = {k: sum(t['proposed_level'] == k for t in tasks) for k in LEVELS}
    unclassified = [t['task_id'] for t in tasks if t['proposed_level'] is None]
    sparse = [k for k,n in counts.items() if n < config['minimum_unique_tasks_per_level']]
    coverage = not unclassified and not sparse
    target = {k: n/32 for k,n in counts.items()} if not unclassified else None
    schedule = schedule_proposal(target) if target is not None else None
    difficulty_names = sorted({k for t in tasks for k in t['difficulty_components']})
    by_level = {}
    for level in LEVELS:
        members = [t for t in tasks if t['proposed_level'] == level]
        by_level[level] = {'I_peak': distribution(t['I_peak'] for t in members),
            'maximum_target_closing_speed_mps': distribution(t['maximum_target_closing_speed_mps'] for t in members),
            'minimum_projected_opportunity_gap_m': distribution(t['minimum_projected_opportunity_gap_m'] for t in members),
            'difficulty_components': {k: distribution(t['difficulty_components'].get(k) for t in members)
                                      for k in difficulty_names},
            'critical_actor_counts': dict(Counter(str(t['critical_actor_count']) for t in members)),
            'key_actor_missing_task_count': sum(bool(t['predictor_slot_coverage']['key_actor_missing_slots']) for t in members)}
    return {'threshold_proposal': thresholds, 'tasks': tasks,
        'summary': {'unique_tasks': 32, 'proposed_counts': counts, 'unclassified_task_ids': unclassified,
            'below_minimum_levels': sparse, 'coverage_gate': coverage,
            'fixed_three_level_schedule_mathematically_schedulable': not unclassified and all(counts.values()),
            'proposed_uniform_task_layer_mass': target, 'schedule_proposal': schedule,
            'candidate_level_task_ids': {k:[t['task_id'] for t in tasks if t['proposed_level'] == k] for k in LEVELS},
            'independent_D_and_coverage_by_level': by_level,
            'control_difficulty_matching_claim': False, 'predictor_accuracy_by_level_measured': False,
            'distribution_frozen': False, 'manual_review_required': True,
            'task_classifier': None, 'target_distribution_pi': None, 'training_ready': False}}


class OfflineEpisodeSampler:
    """Exercise the future sampler contract without authorizing env/learner execution.

    Call begin only at an episode boundary and end with the actual transition
    count. Sampling uses a private RNG; no model, replay or simulator RNG is used.
    """
    def __init__(self, tasks, mode, sampling_seed, *, purpose):
        if purpose != 'offline_algorithm_test_only' or mode not in ('mix', 'curriculum'):
            raise ValueError('Offline sampler cannot authorize training')
        _integer(sampling_seed, 'sampling_seed')
        self._tasks = tuple(deepcopy(tasks))
        if not self._tasks or any(set(t) != {'task_id','simulator_seed','level'} for t in self._tasks):
            raise ValueError('Require explicit task ID/seed/level roster')
        if len({t['task_id'] for t in self._tasks}) != len(self._tasks) or len({t['simulator_seed'] for t in self._tasks}) != len(self._tasks):
            raise ValueError('Duplicate tasks or simulator seeds')
        for t in self._tasks:
            if not isinstance(t['task_id'], str) or not t['task_id'] or t['level'] not in LEVELS:
                raise ValueError('Malformed task/level')
            _integer(t['simulator_seed'], 'simulator_seed')
        self._pools = {k: tuple(t for t in self._tasks if t['level'] == k) for k in LEVELS}
        if not all(self._pools.values()):
            raise ValueError('Do not renormalize an empty layer away')
        self._schedule = schedule_proposal({k:len(v)/len(self._tasks) for k,v in self._pools.items()})
        self._rng = random.Random(sampling_seed)
        self._mode = mode
        self._step = 0
        self._episode = 0
        self._active = None
        self._stage_entries = {}
        self._binding = digest({'tasks':list(self._tasks), 'mode':mode, 'seed':sampling_seed, 'schedule':self._schedule})

    def begin_episode(self):
        if self._active is not None:
            raise ValueError('No task switch inside an active episode')
        if self._step >= 120000:
            raise ValueError('Requested budget reached; no automatic extension')
        stage = next(i for i,s in enumerate(self._schedule['stages']) if s['start'] <= self._step < s['end'])
        self._stage_entries.setdefault(stage, self._step)
        mass = self._schedule['stages'][stage]['mass'] if self._mode == 'curriculum' else self._schedule['nominal_Mix_reset_mass']
        draw = self._rng.random(); cumulative = 0.
        for level in LEVELS:
            cumulative += mass[level]
            if draw < cumulative:
                break
        task = deepcopy(self._rng.choice(self._pools[level]))
        self._active = {**task, 'episode_index':self._episode, 'collection_stage':stage,
            'actual_start_step':self._step, 'nominal_stage_start':self._schedule['stages'][stage]['start'],
            'actual_stage_entry_step':self._stage_entries[stage],
            'stage_boundary_overrun':self._stage_entries[stage]-self._schedule['stages'][stage]['start'],
            'sampling_mode':self._mode, 'sampling_mass':dict(mass), 'sampler_binding':self._binding,
            'execution_authorized':False}
        self._episode += 1
        return deepcopy(self._active)

    def end_episode(self, actual_control_steps):
        _integer(actual_control_steps, 'actual_control_steps', 1)
        if self._active is None:
            raise ValueError('No active episode')
        start = self._step; self._step += actual_control_steps
        crossings = [{'boundary':s['end'], 'actual_next_reset_step':self._step,
                      'overrun':self._step-s['end']} for s in self._schedule['stages']
                     if start < s['end'] <= self._step]
        result = {**deepcopy(self._active), 'actual_end_step':self._step,
                  'actual_control_steps':actual_control_steps, 'crossed_boundaries':crossings,
                  'request_budget_overrun':max(0,self._step-120000)}
        self._active = None
        return result


class ExposureAudit:
    """Observe external transition slots and actual minibatches; never sample/store replay."""
    def __init__(self, capacity=1000000, recent_window=1024):
        self._capacity = _integer(capacity, 'capacity', 1)
        self._recent = deque(maxlen=_integer(recent_window, 'recent_window', 1))
        self._slots = {}
        self._steps = 0
        self._active = None
        self._episode_start = 0
        self._completed = []
        self._collected = Counter()
        self._sampled = Counter()
        self._collected_task = Counter()
        self._sampled_task = Counter()
        self._collected_stage = Counter()
        self._sampled_stage = Counter()
        self._started = Counter()
        self._minibatches = 0

    def begin_episode(self, selection):
        if self._active is not None or selection['actual_start_step'] != self._steps:
            raise ValueError('Episode start does not align with actual control transitions')
        if selection['level'] not in LEVELS or selection['execution_authorized'] is not False:
            raise ValueError('Only offline audit metadata is accepted')
        self._active = deepcopy(selection)
        self._episode_start = self._steps
        self._started[selection['level']] += 1

    def record_transition(self, replay_index, actual_environment_step):
        _integer(replay_index, 'replay_index')
        _integer(actual_environment_step, 'actual_environment_step', 1)
        if self._active is None or actual_environment_step != self._steps+1 or replay_index >= self._capacity:
            raise ValueError('Missing episode, invalid replay slot or nonsequential real transition')
        row = {k:self._active[k] for k in ('task_id','simulator_seed','level','episode_index','collection_stage')}
        row['actual_environment_step'] = actual_environment_step
        self._slots[replay_index] = row
        self._recent.append(row)
        self._collected[row['level']] += 1
        self._collected_task[row['task_id']] += 1
        self._collected_stage[(row['collection_stage'], row['level'])] += 1
        self._steps = actual_environment_step
        return deepcopy(row)

    def record_minibatch(self, actual_replay_indices):
        indices = list(actual_replay_indices)
        if not indices or any(type(i) is not int or i not in self._slots for i in indices):
            raise ValueError('Require indices actually observed from known replay slots')
        self._sampled.update(self._slots[i]['level'] for i in indices)
        self._sampled_task.update(self._slots[i]['task_id'] for i in indices)
        self._sampled_stage.update((self._slots[i]['collection_stage'], self._slots[i]['level']) for i in indices)
        self._minibatches += 1
        return [deepcopy(self._slots[i]) for i in indices]

    def end_episode(self, outcome):
        if self._active is None or not isinstance(outcome, str) or not outcome:
            raise ValueError('Missing episode/outcome; failures must also be recorded')
        steps = self._steps-self._episode_start
        if steps <= 0:
            raise ValueError('Cannot count a reset-only episode as control exposure')
        self._completed.append({**self._active, 'actual_end_step':self._steps,
                                'actual_control_steps':steps, 'outcome':outcome})
        self._active = None

    def snapshot(self):
        def counts(items):
            result = Counter(r['level'] for r in items)
            return {k:result[k] for k in LEVELS}
        def stage_counts(result):
            return {str(i):{k:result[(i,k)] for k in LEVELS} for i in range(3)}
        def observed_stages(items):
            return stage_counts(Counter((r['collection_stage'], r['level']) for r in items))
        return {'actual_control_steps':self._steps,
            'started_episodes_by_level':{k:self._started[k] for k in LEVELS},
            'completed_episodes':deepcopy(self._completed),
            'active_episode':deepcopy(self._active),
            'collected_transition_counts':{k:self._collected[k] for k in LEVELS},
            'recent_transition_counts':counts(self._recent), 'replay_slot_counts':counts(self._slots.values()),
            'minibatch_transition_counts':{k:self._sampled[k] for k in LEVELS},
            'collected_transitions_by_task':dict(self._collected_task),
            'replay_slots_by_task':dict(Counter(r['task_id'] for r in self._slots.values())),
            'minibatch_transitions_by_task':dict(self._sampled_task),
            'collected_transitions_by_stage_and_level':stage_counts(self._collected_stage),
            'recent_transitions_by_stage_and_level':observed_stages(self._recent),
            'replay_slots_by_stage_and_level':observed_stages(self._slots.values()),
            'minibatch_transitions_by_stage_and_level':stage_counts(self._sampled_stage),
            'observed_minibatches':self._minibatches,
            'replay_resampled_or_cleared':False, 'metadata_in_network_inputs':False,
            'execution_authorized':False}
