from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'tools')]
from prediction_rl.curriculum.interaction import (
    analyze_root, distribution, pair_response, quantile, root_context, stratify, validated_timeline)
from prediction_rl.curriculum.train_reader import TrainReader, inside, strict_json, validate_config
from prediction_rl.data.actor_adapter import build_actor_inputs
from prediction_rl.data.dataset_contract import build_branch_labels
from prediction_rl.data.merge_geometry import MergeGeometry
import audit_curriculum_interaction as cli


@pytest.fixture
def config():
    return validate_config(json.loads((ROOT/'configs/development/ic01_train_interaction_v1.json').read_text()))


@pytest.fixture
def geometry():
    return MergeGeometry(ROOT/'RL-MPC-LaneMerging-master/merge.net.xml')


def base(frame):
    return {'simulation_time_s': frame['simulation_time_s'], 'vehicles': {
        a: {k: v[k] for k in ('position', 'speed', 'acceleration', 'lane_id')}
        for a, v in frame['vehicles'].items()}}


def example(g, c, responsive=False, terminal_step=None):
    def actor(x, speed):
        return {'position': [float(x), 0.], 'speed': float(speed), 'acceleration': 0.,
                'lane_id': 'highwayahead_0', 'lane_position_m': float(x), 'length_m': 5., 'width_m': 1.8}
    root = {'simulation_time_s': 20., 'vehicles': {
        'ego': actor(20, 10), 'a': actor(35, 8), 'b': actor(5, 12)}}
    h = {'root_id': 'r', 'episode_id': 'e', 'split': 'train', 'history': [root],
         'inputs': build_actor_inputs([root], g)}
    pack = {'root_id': 'r', 'episode_id': 'e', 'split': 'train', 'root_traffic': base(root),
            'prefix_actions': [], 'candidates': []}
    branches = {}
    for cid, jerk in enumerate(c['probe_jerks']):
        plan = [jerk]+[0.]*(c['horizon_steps']-1)
        trace, frames = [], []; previous = base(root)
        for t in range(c['horizon_steps']):
            elapsed = (t+1)*c['tick_s']; frame = deepcopy(root)
            frame['simulation_time_s'] += elapsed
            for a, v in frame['vehicles'].items():
                change = cid*.25 if a != 'ego' and responsive else cid*.1 if a == 'ego' else 0.
                v['speed'] += change; v['lane_position_m'] += v['speed']*elapsed
                v['position'][0] = v['lane_position_m']
            done = terminal_step is not None and t == terminal_step
            if done:
                del frame['vehicles']['ego']
            after = base(frame)
            trace.append({'done': done, 'reward': 0., 'info': {'execution_audit': {
                'before': previous, 'after': after, 'execution_contract': 'simulation_blocking_exact_v1',
                'simulation_elapsed_s': c['tick_s'], 'requested_jerk': plan[t],
                'commanded_speed_reconstructed': 10., 'upstream_reward_projected_jerk': plan[t],
                'invalid_action_reward': 0., 'measured_post_step_speed': None if done else 10.,
                'termination_reason': 'upstream_collision' if done else None}}})
            frames.append(frame); previous = after
            if done:
                break
        branches[cid] = {'trace': trace, 'future_metadata': {'contract': 'lane_position_length_width_v1',
            'frames': frames, 'trajectory_usable': [not step['done'] for step in trace]}}
        pack['candidates'].append({'candidate_id': cid, 'labels': build_branch_labels(base(root), trace, plan, .2)})
    return h, pack, branches


def root_row(g, c, responsive=False, terminal_step=None):
    h, p, branches = example(g, c, responsive, terminal_step)
    return analyze_root(h, p, branches, branches.copy(), g, c,
                        {'target': 'downstream', 'job_id': 'e0000', 'simulator_seed': 11000})


def test_ego_only_motion_cannot_create_neighbor_interaction(geometry, config):
    row = root_row(geometry, config)
    assert row['interaction']['mean_abs_dv_mps'] == row['interaction']['mean_abs_ds_m'] == 0
    assert row['interaction']['full_common_horizon']
    assert row['repeat_noise']['max_abs_ds_m'] == 0
    assert 'ego' not in row['root_context']['actor_ids']


def test_real_neighbor_change_and_order_invariance(geometry, config):
    h, p, b = example(geometry, config, responsive=True)
    original = analyze_root(h, p, b, b.copy(), geometry, config, {})
    h2, p2, b2 = deepcopy((h, p, b))
    for frame in h2['history']:
        frame['vehicles'] = dict(reversed(list(frame['vehicles'].items())))
    p2['candidates'].reverse(); b2 = dict(reversed(list(b2.items())))
    permuted = analyze_root(h2, p2, b2, b2.copy(), geometry, config, {})
    assert original == permuted
    assert original['interaction']['mean_abs_dv_mps'] > 0
    assert original['interaction']['mean_abs_ds_m'] > 0


def test_future_actor_never_selects_root_roles(geometry, config):
    h, p, _ = example(geometry, config)
    before = root_context(h, geometry, config)
    p['future_best_actor'] = 'unobserved_actor'
    assert root_context(h, geometry, config) == before


def test_disappearance_is_not_zero_response():
    r = pair_response({0: {'a': (1., 2.), 'b': (100., 100.)}, 1: {'a': (3., 4.)}},
                      {0: {'a': (2., 3.)}, 1: {'b': (5., 6.)}}, ['a', 'b'], 25, 5)
    assert r['common_cells'] == 1 and r['actor_common_steps'] == {'a': 1}
    assert r['mean_abs_ds_m'] == r['mean_abs_dv_mps'] == 1
    assert not r['supported']


@pytest.mark.parametrize('left,right,actors', [({}, {}, ['a']),
    ({0: {'a': (1, 2)}}, {1: {'a': (1, 2)}}, ['a']),
    ({0: {'ego': (1, 2)}}, {0: {'ego': (9, 9)}}, ['ego'])])
def test_no_common_support_stays_uncomputable(left, right, actors):
    result = pair_response(left, right, actors, 25, 5)
    assert result['common_cells'] == 0 and result['mean_abs_ds_m'] is None


def test_terminal_tail_is_explicitly_censored(geometry, config):
    row = root_row(geometry, config, responsive=True, terminal_step=2)
    assert row['censoring']['terminal_candidates'] == 5
    assert row['interaction']['common_cells'] == 10*2*2
    assert not row['interaction']['full_common_horizon']
    assert row['interaction']['supported_pairs'] == 0
    stratify([row], config)
    assert row['level'] == 'C2' and row['level_reason'] == 'insufficient_common_support'


def test_moving_ego_relative_labels_are_rejected(geometry, config):
    h, p, b = example(geometry, config)
    label = deepcopy(p['candidates'][0]['labels'])
    label['future'][label['actor_ids'].index('a')][0][0] -= 1.
    with pytest.raises(ValueError, match='fixed-root'):
        validated_timeline(b[0], p['root_traffic'], label, geometry, config)


def test_wrong_sidecar_and_time_grid_are_rejected(geometry, config):
    h, p, b = example(geometry, config)
    bad = deepcopy(b[0]); bad['future_metadata']['frames'][0]['vehicles']['a']['speed'] += 1.
    with pytest.raises(ValueError, match='Sidecar'):
        validated_timeline(bad, p['root_traffic'], p['candidates'][0]['labels'], geometry, config)
    bad = deepcopy(b[0]); bad['future_metadata']['trajectory_usable'][0] = False
    with pytest.raises(ValueError, match='support metadata'):
        validated_timeline(bad, p['root_traffic'], p['candidates'][0]['labels'], geometry, config)


@pytest.mark.parametrize('upstream,downstream', [
    ('highwayrear_0', ':mergenode_0_0'), (':mergenode_0_0', 'highwayahead_0'),
    ('ramp_0', ':mergenode_1_0'), (':mergenode_1_0', 'highwayahead_0')])
def test_connected_lane_progress_has_no_spurious_jump(geometry, upstream, downstream):
    a = {'lane_id': upstream, 'lane_position_m': geometry.lengths[upstream]-.5}
    b = {'lane_id': downstream, 'lane_position_m': .5}
    assert geometry.progress(b)-geometry.progress(a) == pytest.approx(1.)


def test_unknown_lane_not_silently_assigned_geometry(geometry):
    with pytest.raises(KeyError):
        geometry.progress({'lane_id': 'other_0', 'lane_position_m': 1.})


def test_repeat_noise_is_measured_not_assumed_zero():
    result = pair_response({0: {'a': (1., 2.)}}, {0: {'a': (1.01, 2.02)}}, ['a'], 25, 1)
    assert result['mean_abs_dv_mps'] == pytest.approx(.02)
    assert result['max_abs_ds_m'] == pytest.approx(.01)


def test_zero_response_cannot_be_C1_even_if_upper_quantile_is_zero(geometry, config):
    rows = [root_row(geometry, config) for _ in range(3)]
    for i, r in enumerate(rows):
        r['episode_id'] = str(i); r['root_id'] = str(i)
    report = stratify(rows, config)
    assert report['thresholds']['I_high'] == 0
    assert not report['response_available'] and report['levels']['C1']['roots'] == 0
    assert sum(v['roots'] for v in report['levels'].values()) == len(rows)
    assert report['training_ready'] is False


def test_high_response_above_repeat_noise_is_required(geometry, config):
    rows = [root_row(geometry, config, responsive=True)]
    rows[0]['repeat_noise']['max_abs_dv_mps'] = 1e6
    report = stratify(rows, config)
    assert report['levels']['C1']['roots'] == 0 and not report['response_available']


@pytest.mark.parametrize('scope', ['development', 'validation', 'calibration', 'test'])
def test_only_train_allowed(config, scope):
    config['scope'] = scope
    with pytest.raises(ValueError, match='non-train'):
        validate_config(config)


def test_configuration_unknown_field_no_silent_defaults(config):
    config['new_reward'] = True
    with pytest.raises(ValueError, match='Unknown'):
        validate_config(config)


@pytest.mark.parametrize('raw', ['{"a":1,"a":2}', '{"a":NaN}', '{"a":Infinity}', '{"a":1e999}'])
def test_json_rejects_duplicates_and_nonfinite_literals(raw):
    with pytest.raises(ValueError):
        strict_json(raw)


def test_verified_reader_rejects_changed_bytes_and_path_escape(tmp_path):
    reader = object.__new__(TrainReader); reader.baseline = tmp_path.resolve()
    reader.hashes = {}; reader.files_read = 0
    p = tmp_path/'data.json'; p.write_text('{}')
    with pytest.raises(ValueError, match='SHA256'):
        reader.bytes(p, '0'*64)
    with pytest.raises(ValueError, match='traversal'):
        inside(tmp_path, '../test/data.json')


def test_output_never_overwrites_or_writes_baseline(tmp_path):
    worktree = tmp_path/'worktree'; baseline = tmp_path/'baseline'
    out = cli.output_directory(worktree, baseline, 'ic01_v1')
    out.mkdir(parents=True)
    with pytest.raises(ValueError, match='overwrite'):
        cli.output_directory(worktree, baseline, 'ic01_v1')
    with pytest.raises(ValueError, match='baseline'):
        cli.output_directory(baseline, baseline, 'new_run')
    with pytest.raises(ValueError, match='run ID'):
        cli.output_directory(worktree, baseline, '../escape')


def test_original_observations_and_inputs_are_unchanged(geometry, config):
    h, p, b = example(geometry, config, responsive=True)
    snapshot = deepcopy((h, p, b))
    row = analyze_root(h, p, b, b.copy(), geometry, config, {})
    stratify([row], config)
    assert (h, p, b) == snapshot


def test_quantiles_unavailable_and_nonfinite():
    assert quantile([None, 1., 3.], .5) == 2.
    assert quantile([], .5) is None
    assert distribution([None])['mean'] is None
    with pytest.raises(ValueError):
        quantile([float('inf')], .5)
