from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'tools')]
from prediction_rl.curriculum.task_design import (validate_design, proposal_jobs, replay_budget,
    existing_root_coverage, initial_description)
from prediction_rl.data.merge_geometry import MergeGeometry
from test_actor_adapter import actor, frame


@pytest.fixture
def config():
    return json.loads((ROOT/'configs/development/ic02c_task_design_v1.json').read_text())


def inventory():
    return {'classified_executable_tasks': 0, 'fresh_task_target_pi': None,
        'records': [{'simulator_seed': seed, 'episode_id': f'e{seed}', 'job_id': f'j{seed}',
            'task_level': None, 'execution_authorized': False, 'provisional_root_levels': {'C1': 1}}
            for seed in range(11000,11256)]}


def test_reviewed_design_is_not_a_collector_or_training_request(config):
    assert validate_design(config) is config
    assert not config['runtime_authorized'] and not config['training_ready']
    assert config['task_classifier'] is config['target_distribution_pi'] is None


@pytest.mark.parametrize('field,value', [('runtime_authorized', True), ('training_ready', True),
    ('task_classifier', 'any_C1_root'), ('target_distribution_pi', [1/3]*3), ('scope', 'run'),
    ('source_train_seed_range', [200,219]), ('pilot_seed_stride', 1), ('branch_repeats', 1),
    ('discovery_max_steps', 250), ('reference_prefix_before_training_policy_steps', 20),
    ('frozen_predictor_inference_calls', 1), ('schema_version', True), ('unknown', 0)])
def test_design_cannot_silently_authorize_expand_or_change_semantics(config, field, value):
    config[field] = value
    with pytest.raises(ValueError): validate_design(config)


def test_task_proposal_is_fixed_train_stride_not_root_outcome_selection(config):
    source = inventory()
    initial = deepcopy(source)
    result = proposal_jobs(source, config)
    assert len(result) == 32
    assert [j['simulator_seed'] for j in result] == list(range(11000,11256,8))
    assert sum(j['engineering_stage'] for j in result) == 3
    assert all(j['task_level'] is None and not j['execution_authorized'] for j in result)
    assert source == initial
    for row in source['records']: row['provisional_root_levels'] = {'C2': 99}
    assert proposal_jobs(source, config) == result


@pytest.mark.parametrize('fault', ['missing', 'duplicate', 'promoted', 'pi', 'authorized'])
def test_incomplete_or_promoted_source_inventory_is_rejected(config, fault):
    source = inventory()
    if fault == 'missing': source['records'].pop()
    elif fault == 'duplicate': source['records'][-1]['simulator_seed'] = 11000
    elif fault == 'promoted': source['records'][0]['task_level'] = 'C1'
    elif fault == 'pi': source['fresh_task_target_pi'] = [0, .5, .5]
    elif fault == 'authorized': source['records'][0]['execution_authorized'] = True
    with pytest.raises(ValueError): proposal_jobs(source, config)


@pytest.mark.parametrize('count,sessions,branches,controls,waiting,raw', [
    (3,114,90,12234,11514,23862), (32,1216,960,130496,122816,254528)])
def test_budget_counts_all_discoveries_captures_prefix_replays_and_resets(config, count, sessions, branches, controls, waiting, raw):
    result = replay_budget(count, config)
    assert result['fresh_sumo_sessions_max'] == sessions
    assert result['candidate_branch_executions_max'] == branches
    assert result['total_control_steps_max'] == controls
    assert result['warmup_and_insertion_ticks_max'] == waiting
    assert result['total_raw_ticks_max'] == raw
    assert result['candidate_future_control_steps_max'] == branches*25
    assert result['total_control_steps_max'] > result['candidate_future_control_steps_max']
    assert not result['execution_authorized'] and result['learner_transitions'] == 0


@pytest.mark.parametrize('count', [True, 0, -1, 1.5])
def test_invalid_budget_counts_rejected(config, count):
    with pytest.raises(ValueError): replay_budget(count, config)


def test_old_root_labels_and_unavailable_reset_evidence_are_not_promoted():
    rows = [{'root_id': 'a', 'episode_id': 'e', 'target': 'approach', 'level': 'C1',
             'task_mapping': {'reference_prefix_steps': 32}},
            {'root_id': 'b', 'episode_id': 'e', 'target': 'internal_mid', 'level': 'C2',
             'task_mapping': {'reference_prefix_steps': 48}}]
    report = {'summary': {'reached_roots': 2, 'episodes': 1, 'classification': {'levels': {
        'C0': {'roots': 0}, 'C1': {'roots': 1}, 'C2': {'roots': 1}}}}}
    original = deepcopy(rows)
    result = existing_root_coverage(rows, report)
    assert result['roots_at_original_reset'] == result['executable_task_levels_assigned'] == 0
    assert result['target_distribution_pi'] is None and not result['old_root_thresholds_changed']
    assert rows == original
    assert result['by_checkpoint']['approach']['prefix_steps_min'] == 32


def reset_trace():
    return {'job': {'simulator_seed': 11000}, 'packets': [
        {'phase': 'reset', 'control_step': 0, 'done': False, 'reward': None, 'observation': [0.]*20,
         'current_frame': frame(20.2, {'ego': actor('ramp_0',40,speed=15),
            'front': actor(pos=100), 'rear': actor(pos=20)})},
        {'future': 'must_not_be_used'}]}


def test_initial_frame_is_observable_only_not_low_interaction_or_free_history():
    trace = reset_trace()
    config = json.loads((ROOT/'configs/development/ic01_train_interaction_v1.json').read_text())
    geometry = MergeGeometry(ROOT/'RL-MPC-LaneMerging-master/merge.net.xml')
    result = initial_description(trace, geometry, config)
    assert result['history_frames'] == 1
    assert result['task_level'] is result['neighbor_response_I'] is None
    assert all(n == 1 for n in result['root_context']['coverage']['key_history_steps'].values())
    trace['packets'][1] = {'future': 'different_future_or_policy_winner'}
    assert initial_description(trace, geometry, config) == result


@pytest.mark.parametrize('field,value', [('phase', 'step'), ('control_step', 1), ('done', True), ('reward', 0)])
def test_later_or_terminal_state_cannot_be_called_reset(field, value):
    trace = reset_trace()
    trace['packets'][0][field] = value
    with pytest.raises(ValueError): initial_description(trace, None, {})
