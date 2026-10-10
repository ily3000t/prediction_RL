from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'tools')]
from prediction_rl.curriculum.entry_design import (
    SOURCE_PATHS, inspect_source, mapping_summary, smoke_budget, validate_design)


@pytest.fixture
def config():
    return json.loads((ROOT/'configs/development/ic02a_entry_design_v1.json').read_text())


def source_inventory():
    return {p: (ROOT/p).read_text(encoding='utf-8-sig') for p in SOURCE_PATHS if p.endswith('.py')}


def inspect(sources=None):
    return inspect_source(source_inventory() if sources is None else sources,
        json.loads((ROOT/SOURCE_PATHS[1]).read_text()))


def test_checked_in_design_is_offline_unapproved(config):
    assert validate_design(config) is config
    assert config['training_ready'] is False and config['smoke_proposal']['authorized'] is False


@pytest.mark.parametrize('key,value', [
    ('training_ready', True), ('schema_version', True), ('proposal_status', 'approved'),
    ('scope', 'run_sumo'), ('reference_prefix_before_policy_steps', 10),
    ('prefill_predictor_history', True), ('target_distribution_pi', [1/3]*3),
    ('task_classifier', 'any_C1_root_makes_C1_task'),
    ('environment_rng_scope', 'learner_process'),
])
def test_cannot_turn_design_into_runtime_or_promote_old_labels(config, key, value):
    config[key] = value
    with pytest.raises(ValueError):
        validate_design(config)


def test_unknown_field_and_changed_arms_rejected(config):
    wrong = deepcopy(config); wrong['override'] = True
    with pytest.raises(ValueError): validate_design(wrong)
    config['matched_arms'].pop()
    with pytest.raises(ValueError): validate_design(config)


def test_static_source_facts_not_runtime_verification():
    before = set(sys.modules)
    result = inspect()
    assert all(result['assumptions'].values()) and result['runtime_verified'] is False
    assert result['native_continuing_reset_can_request_named_task'] is False
    assert result['learned_initial_history_frames'] == 1
    assert result['resolved_upstream_subset']['ALT_J_WEIGHT'] == .1  # config override, not default .05
    assert result['resolved_upstream_subset']['REWARD_FUNCTION'] == 'Slotted Jerk'
    assert result['resolved_upstream_subset']['START_SPEED_VARIANCE'] == 5
    assert not ({'control', 'sumo', 'traci', 'merge_gym'} & (set(sys.modules)-before))


@pytest.mark.parametrize('path,old,new', [
    (SOURCE_PATHS[2], 'def reset(self):', 'def reset(self, seed=None):'),
    (SOURCE_PATHS[2], 'config.get("wait_before_start", 20)', 'config.get("wait_before_start", 50)'),
    (SOURCE_PATHS[3], 'departPos=40', 'departPos=70'),
    (SOURCE_PATHS[5], 'torch.manual_seed(seed)', 'torch.manual_seed(seed + 1)'),
    (SOURCE_PATHS[6], 'self.history.clear()', 'self.history.append(None)'),
])
def test_detect_changed_entry_assumptions(path, old, new):
    sources = source_inventory(); assert old in sources[path]
    sources[path] = sources[path].replace(old, new, 1)
    # The seed expression is also checked; do not silently accept a new RNG contract.
    if path == SOURCE_PATHS[5]:
        assert 'seed + 1' in sources[path]
    with pytest.raises(ValueError): inspect(sources)


def test_smoke_budget_is_finite_and_not_authorization(config):
    result = smoke_budget(config['smoke_proposal'])
    assert result['episodes_max'] == 12 and result['control_steps_max'] == 144
    assert result['warmup_and_insertion_ticks_max'] == 1212
    assert result['feature_calls_per_predictor_mode_max'] == 156
    assert result['execution_authorized'] is False and result['optimizer_updates'] == 0
    config['smoke_proposal']['authorized'] = True
    with pytest.raises(ValueError): smoke_budget(config['smoke_proposal'])


def family():
    rows = [{'root_id': 'r0', 'episode_id': 'e0', 'simulator_seed': 11000, 'job_id': 'e0000',
             'level': 'C1', 'task_mapping': {'executable_as_original_reset': False}},
            {'root_id': 'r1', 'episode_id': 'e0', 'simulator_seed': 11000, 'job_id': 'e0000',
             'level': 'C2', 'task_mapping': {'executable_as_original_reset': False}}]
    report = {'summary': {'reached_roots': 2, 'episodes': 1, 'classification': {'levels': {
        'C0': {'roots': 0}, 'C1': {'roots': 1}, 'C2': {'roots': 1}}}}}
    return rows, report


def test_roots_grouped_without_assigning_episode_level_or_pi():
    rows, report = family(); original = deepcopy(rows)
    result = mapping_summary(rows, report)
    assert rows == original and result['source_episodes'] == 1 and result['source_roots'] == 2
    assert result['source_episodes_with_any_provisional_C1_root'] == 1
    assert result['classified_executable_tasks'] == 0 and result['fresh_task_target_pi'] is None
    assert result['records'][0]['task_level'] is None
    assert result['records'][0]['execution_authorized'] is False
    assert result == mapping_summary(list(reversed(rows)), report)


@pytest.mark.parametrize('change', ['test_seed', 'duplicate_root', 'promoted_reset', 'seed_join', 'summary'])
def test_reject_invalid_task_mapping(change):
    rows, report = family()
    if change == 'test_seed': rows[0]['simulator_seed'] = 14000
    elif change == 'duplicate_root': rows[1]['root_id'] = rows[0]['root_id']
    elif change == 'promoted_reset': rows[0]['task_mapping']['executable_as_original_reset'] = True
    elif change == 'seed_join': rows[1]['simulator_seed'] = 11001
    else: report['summary']['episodes'] = 2
    with pytest.raises(ValueError): mapping_summary(rows, report)
