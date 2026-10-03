from copy import deepcopy
import gzip
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'tools')]
from prediction_rl.evaluation.matched_learning_curve import PROTOCOL, SMOKE, validate_protocol, jobs, snapshot_steps, aggregate, METRICS
from prediction_rl.training.matched_learning_curve import StreamingReplay, train_nodes, Coverage
from prediction_rl.training.all_ddpg import make_all_environment, preset_factory, experiment_type, checkpoint_contract, save_evaluation_weights
from test_training_contract import SyntheticEnv


def test_frozen_config_and_exact_roster():
    for config, name in [(PROTOCOL, 'p07k_matched_60k_v1'), (SMOKE, 'p07k_workflow_smoke_v1')]:
        assert json.loads((ROOT/f'configs/development/{name}.json').read_text()) == config
    assert len(jobs(PROTOCOL, 'train')) == 12
    assert len(jobs(PROTOCOL, 'evaluate')) == 720
    assert 12*PROTOCOL['training']['requested_frames'] == 720000
    assert len({j['simulator_seed'] for j in jobs(PROTOCOL, 'evaluate')}) == 20
    assert PROTOCOL['training']['warm_start'] is False and PROTOCOL['automatic_budget_extension'] is False


@pytest.mark.parametrize('mutation', ['seed', 'reward', 'unknown', 'budget', 'workers'])
def test_protocol_cannot_be_silently_modified(mutation):
    p = deepcopy(PROTOCOL)
    if mutation == 'seed': p['training']['run_seeds'] = [0, 1, 3]
    elif mutation == 'reward': p['reward'] = 'modified'
    elif mutation == 'unknown': p['unrecognized'] = True
    elif mutation == 'workers': p['training']['workers'] = 2
    else: p['training']['requested_frames'] = 100000
    with pytest.raises(ValueError): validate_protocol(p)


@pytest.mark.parametrize('actual', [19999, 20500, 20000.0])
def test_snapshot_budget_rejects_invalid_node(actual):
    with pytest.raises(ValueError): snapshot_steps(actual, 20000)


def run_synthetic(directory, arm, segmented, watched):
    directory.mkdir(); torch.manual_seed(910); np.random.seed(910)
    raw = SyntheticEnv(arm); holder = {}
    def record(*args): holder['replay'].record(*args)
    env = make_all_environment(raw, 'cpu', record if watched else None)
    experiment = experiment_type(directory/'logs')(preset_factory('cpu', .0002, smoke=True), env, quiet=True)
    recorder = type('Verifier', (), {'verify_row': lambda self, row: None})()
    replay = StreamingReplay(experiment._agent, directory, recorder) if watched else None
    holder['replay'] = replay
    contract = checkpoint_contract(env, experiment._agent.preset_parameters,
        '0'*64 if arm in ('ordinary', 'conditional') else None, '1'*64)
    snaps = []
    def save(n, actual):
        sha = save_evaluation_weights(experiment._agent, directory/f'n{n}.pt', contract)
        snaps.append((n, actual, sha))
    try:
        if segmented: train_nodes(experiment, [12, 24, 36], save)
        else: experiment.train(frames=36)
    finally:
        if replay: replay.close()
        experiment._writer.close(); env.close()
    return experiment._agent, replay, snaps


@pytest.mark.parametrize('arm', ['baseline', 'zero', 'ordinary', 'conditional'])
def test_snapshots_and_streaming_leave_exact_rng_actions_replay_optimizer_targets(tmp_path, arm):
    torch.set_num_threads(1)
    plain, _, _ = run_synthetic(tmp_path/'plain', arm, False, False)
    expected_torch = torch.get_rng_state().clone(); expected_numpy = deepcopy(np.random.get_state())
    watched, replay, snaps = run_synthetic(tmp_path/'watched', arm, True, True)
    assert torch.equal(torch.get_rng_state(), expected_torch)
    actual = np.random.get_state(); np.testing.assert_array_equal(actual[1], expected_numpy[1]); assert actual[2:] == expected_numpy[2:]
    assert [r[:2] for r in snaps] == [(12, 12), (24, 24), (36, 36)]
    assert replay.summary()['transitions'] == 36
    for name in ('policy', 'q'):
        a, b = getattr(plain.agent, name), getattr(watched.agent, name)
        for k, v in a.model.state_dict().items(): assert torch.equal(v, b.model.state_dict()[k])
        for k, v in a._target._target.state_dict().items(): assert torch.equal(v, b._target._target.state_dict()[k])
        assert a._scheduler.state_dict() == b._scheduler.state_dict()
        for state, other in zip(a._optimizer.state.values(), b._optimizer.state.values()):
            for k, v in state.items(): assert torch.equal(v, other[k])
    assert len(plain.agent.replay_buffer) == len(watched.agent.replay_buffer) == 36
    for a, b in zip(plain.agent.replay_buffer, watched.agent.replay_buffer):
        assert torch.equal(a[0].features, b[0].features) and torch.equal(a[1], b[1])
        assert a[2] == b[2] and torch.equal(a[3].features, b[3].features) and torch.equal(a[3].mask, b[3].mask)
    with gzip.open(tmp_path/'watched/replay.jsonl.gz', 'rt') as h: transitions = [json.loads(line) for line in h]
    with gzip.open(tmp_path/'watched/minibatches.jsonl.gz', 'rt') as h: batches = [json.loads(line) for line in h]
    assert len(transitions) == 36 and sum(r['done'] for r in transitions) == 12
    assert all(1 <= i <= b['training_steps'] for b in batches for i in b['transition_ids'])
    assert len(batches) == replay.sample_calls and replay.sample_coverage.count == 8*len(batches)
    assert [r['transition_id'] for r in transitions] == list(range(1, 37))


def rows(protocol=SMOKE):
    values = []
    for j in jobs(protocol, 'evaluate'):
        r = {**j, **{k: 0. for k in METRICS}, 'steps': 3,
             'natural_arrival': 1, 'ego_collision': 0, 'time_limit': 0, 'native_event_agrees': True,
             'no_policy_updates': True, 'actual_training_steps': j['checkpoint_frames'],
             'initial_traffic_sha256': str(j['simulator_seed']).zfill(64)}
        values.append(r)
    return values


def test_negative_method_effect_still_completes_and_does_not_select_best():
    r = rows()
    for x in r:
        if x['arm'] == 'conditional':
            x.update(natural_arrival=0, time_limit=1, **{'return': -10.})
    result = aggregate(SMOKE, r)
    assert result['engineering_complete'] is True and result['method_effect_gate'] is None
    assert result['checkpoint_selection_by_development_results'] is False
    assert result['unique_traffic_scenarios'] == 1 and result['evaluation_episodes'] == 12
    assert result['nodes'][-1]['comparisons']['total_augmentation']['equal_run_seed_mean_delta']['return'] == -10.


@pytest.mark.parametrize('fault', ['missing', 'duplicate', 'scene', 'nonfinite', 'events', 'training_steps', 'updates'])
def test_aggregate_rejects_invalid_evidence(fault):
    r = rows()
    if fault == 'missing': r.pop()
    elif fault == 'duplicate': r[-1] = deepcopy(r[0])
    elif fault == 'scene': r[-1]['initial_traffic_sha256'] = 'f'*64
    elif fault == 'nonfinite': r[-1]['return'] = float('nan')
    elif fault == 'events': r[-1]['ego_collision'] = 1
    elif fault == 'training_steps': r[-1]['actual_training_steps'] = 1000
    else: r[-1]['no_policy_updates'] = False
    with pytest.raises(ValueError): aggregate(SMOKE, r)


def test_missing_history_and_zero_channels_not_counted_as_learned_coverage():
    row = {'ego': {'speed': 0., 'acceleration': 0., 'lane_id': 'ramp_0'},
           'history_frames': 0, 'feature_metadata': {k: 0. for k in Coverage.names[7:]}, 'background_vehicle_count': 12}
    c = Coverage(); c.add([Coverage.vector(row)])
    out = c.result('replay')['moments']
    assert out['short_history']['mean'] is None and out['selected_fraction']['count'] == 0
    assert out['low_speed']['mean'] == 1.


def test_original_schedule_is_not_shortened_for_new_training_budget():
    from prediction_rl.training.all_ddpg import resolved_preset
    p = resolved_preset('cpu', .0002)
    assert p['last_frame'] == 2e6 and p['discount_factor'] == .98
    assert p['replay_start_size'] == 5000 and p['minibatch_size'] == 100
    assert p['replay_buffer_size'] == 1e6
