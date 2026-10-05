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
from prediction_rl.data.collection_store import read_json, write_once, file_hash
from prediction_rl.data.dataset_contract import digest
from prediction_rl.evaluation.fixed_budget_extension import (
    PROTOCOL, PREVIOUS, SMOKE, METRICS, jobs, validate_protocol, validate_replay_capacity, aggregate)
from prediction_rl.evaluation.matched_learning_curve import aggregate as previous_aggregate
from prediction_rl.training.all_ddpg import (
    resolved_preset, preset_factory, make_all_environment, experiment_type, checkpoint_contract, save_evaluation_weights)
from prediction_rl.training.matched_learning_curve import StreamingReplay, train_nodes
from test_training_contract import SyntheticEnv
import train_fixed_budget_ddpg as cli


def test_only_approved_budget_and_identity_change():
    assert read_json(ROOT/'configs/development/p07n_matched_120k_v1.json') == PROTOCOL
    changed = deepcopy(PROTOCOL)
    changed['version'] = PREVIOUS['version']; changed['selection'] = PREVIOUS['selection']
    changed['training']['checkpoint_frames'] = PREVIOUS['training']['checkpoint_frames']
    changed['training']['requested_frames'] = PREVIOUS['training']['requested_frames']
    assert changed == PREVIOUS
    assert len(jobs(PROTOCOL, 'train')) == 12 and len(jobs(PROTOCOL, 'evaluate')) == 960
    assert 12*PROTOCOL['training']['requested_frames'] == 1440000
    assert len({j['simulator_seed'] for j in jobs(PROTOCOL, 'evaluate')}) == 20
    assert PROTOCOL['from_scratch'] and PROTOCOL['training']['warm_start'] is False


@pytest.mark.parametrize('mutation', ['seed', 'warm_start', 'nodes', 'reward', 'budget', 'learning_rate', 'workers', 'unknown'])
def test_protocol_rejects_unapproved_changes(mutation):
    p = deepcopy(PROTOCOL)
    if mutation == 'seed': p['training']['run_seeds'] = [0, 1, 3]
    elif mutation == 'warm_start': p['training']['warm_start'] = True
    elif mutation == 'nodes': p['training']['checkpoint_frames'] += [100000]
    elif mutation == 'budget': p['training']['requested_frames'] = 200000
    elif mutation == 'learning_rate': p['training']['learning_rate'] = .001
    elif mutation == 'workers': p['training']['workers'] = 2
    else: p[mutation] = 'new'
    with pytest.raises(ValueError): validate_protocol(p)


@pytest.mark.parametrize('capacity', [60499, 100000, 120499])
def test_new_budget_has_stronger_no_wrap_guard(capacity):
    p = resolved_preset('cpu', .0002); p['replay_buffer_size'] = capacity
    with pytest.raises(ValueError): validate_replay_capacity(PROTOCOL, p)


def test_original_preset_and_schedule_retained():
    p = resolved_preset('cpu', .0002)
    validate_replay_capacity(PROTOCOL, p)
    assert p['last_frame'] == 2000000 and p['replay_start_size'] == 5000
    assert p['minibatch_size'] == 100 and p['discount_factor'] == .98
    p['last_frame'] = 120000
    with pytest.raises(ValueError): validate_replay_capacity(PROTOCOL, p)


def rows(protocol):
    return [{**j, **{k: 0. for k in METRICS}, 'steps': 3,
        'natural_arrival': 1, 'ego_collision': 0, 'time_limit': 0, 'native_event_agrees': True,
        'no_policy_updates': True, 'actual_training_steps': j['checkpoint_frames'],
        'initial_traffic_sha256': str(j['simulator_seed']).zfill(64)} for j in jobs(protocol, 'evaluate')]


def test_same_statistics_as_immutable_kernel_smoke():
    r = rows(SMOKE)
    for i, x in enumerate(r): x['return'] = float(i % 7)
    assert aggregate(SMOKE, r) == previous_aggregate(SMOKE, r)


def test_final120k_remains_primary_even_when_worse_and_all_seeds_kept():
    r = rows(PROTOCOL)
    for x in r:
        if x['arm'] == 'conditional' and x['checkpoint_frames'] == 120000:
            x.update(natural_arrival=0, ego_collision=1, **{'return': -10.})
    a = aggregate(PROTOCOL, r)
    assert a['primary_node'] == 120000 and a['engineering_complete']
    assert a['method_effect_gate'] is None and a['checkpoint_selection_by_development_results'] is False
    assert a['evaluation_episodes'] == 960 and a['unique_traffic_scenarios'] == 20
    assert a['nodes'][-1]['comparisons']['action_conditioning']['equal_run_seed_mean_delta']['return'] == -10.
    assert set(a['nodes'][-1]['arms']['conditional']['per_run_seed']) == {'0', '1', '2'}


@pytest.mark.parametrize('fault', ['missing', 'duplicate', 'scene', 'nan', 'events', 'steps', 'updates'])
def test_aggregate_rejects_invalid_cells(fault):
    r = rows(PROTOCOL)
    if fault == 'missing': r.pop()
    elif fault == 'duplicate': r[-1] = deepcopy(r[0])
    elif fault == 'scene': r[-1]['initial_traffic_sha256'] = 'f'*64
    elif fault == 'nan': r[-1]['return'] = float('nan')
    elif fault == 'events': r[-1]['ego_collision'] = 1
    elif fault == 'steps': r[-1]['actual_training_steps'] = 120500
    else: r[-1]['no_policy_updates'] = False
    with pytest.raises(ValueError): aggregate(PROTOCOL, r)


def synthetic(directory, arm, watched):
    directory.mkdir(); torch.manual_seed(912); np.random.seed(912)
    holder = {}; raw = SyntheticEnv(arm)
    def record(*args): holder['replay'].record(*args)
    env = make_all_environment(raw, 'cpu', record if watched else None)
    experiment = experiment_type(directory/'logs')(preset_factory('cpu', .0002, smoke=True), env, quiet=True)
    recorder = type('Recorder', (), {'verify_row': lambda self, row: None})()
    replay = StreamingReplay(experiment._agent, directory, recorder) if watched else None
    holder['replay'] = replay
    contract = checkpoint_contract(env, experiment._agent.preset_parameters,
        '0'*64 if arm in ('ordinary', 'conditional') else None, '1'*64)
    snaps = []
    def snapshot(n, actual):
        snaps.append((n, actual, save_evaluation_weights(experiment._agent, directory/f'n{n}.pt', contract)))
    try:
        if watched: train_nodes(experiment, [12, 24, 36, 48], snapshot)
        else: experiment.train(frames=48)
    finally:
        if replay: replay.close()
        experiment._writer.close(); env.close()
    return experiment._agent, replay, snaps


@pytest.mark.parametrize('arm', PROTOCOL['arms'])
def test_four_node_kernel_exact_actions_rng_replay_optimizers_targets(tmp_path, arm):
    torch.set_num_threads(1)
    plain, _, _ = synthetic(tmp_path/'plain', arm, False)
    rng = torch.get_rng_state().clone(); numpy_rng = deepcopy(np.random.get_state())
    watched, replay, snaps = synthetic(tmp_path/'watched', arm, True)
    assert torch.equal(rng, torch.get_rng_state())
    state = np.random.get_state(); np.testing.assert_array_equal(state[1], numpy_rng[1]); assert state[2:] == numpy_rng[2:]
    assert [r[:2] for r in snaps] == [(12, 12), (24, 24), (36, 36), (48, 48)]
    assert replay.summary()['transitions'] == 48
    for name in ('q', 'policy'):
        a, b = getattr(plain.agent, name), getattr(watched.agent, name)
        for k, v in a.model.state_dict().items(): assert torch.equal(v, b.model.state_dict()[k])
        for k, v in a._target._target.state_dict().items(): assert torch.equal(v, b._target._target.state_dict()[k])
        assert a._scheduler.state_dict() == b._scheduler.state_dict()
        for s, other in zip(a._optimizer.state.values(), b._optimizer.state.values()):
            for k, v in s.items(): assert torch.equal(v, other[k])
    for a, b in zip(plain.agent.replay_buffer, watched.agent.replay_buffer):
        assert torch.equal(a[0].features, b[0].features) and torch.equal(a[1], b[1])
        assert a[2] == b[2] and torch.equal(a[3].features, b[3].features) and torch.equal(a[3].mask, b[3].mask)


def test_actual120k_targets_do_not_reset_or_stop_at60k():
    class FakeExperiment:
        frame = 1
        def train(self, frames, episodes):
            assert episodes == np.inf
            self.calls.append(frames); self.frame = frames+1
    x = FakeExperiment(); x.calls = []; snapshots = []
    train_nodes(x, PROTOCOL['training']['checkpoint_frames'], lambda n, actual: snapshots.append((n, actual)))
    assert x.calls == [20000, 40000, 60000, 120000]
    assert snapshots[-1] == (120000, 120000)


def test_train_dispatch_reuses_original_kernel_without_weights_loading(tmp_path, monkeypatch):
    p = resolved_preset('cpu', .0002); q = {'protocol': PROTOCOL, 'preset': p}; job = jobs(PROTOCOL, 'train')[0]
    calls = []
    monkeypatch.setattr(cli.kernel, 'train_job', lambda request, j, out: calls.append((request, j, out)) or {'from_scratch': True})
    assert cli.train_job(q, job, tmp_path) == {'from_scratch': True}
    assert calls == [(q, job, tmp_path)]


def test_curve_manifest_preserves_original_losses_without_large_copy(tmp_path):
    j = jobs(PROTOCOL, 'train')[0]; d = tmp_path/'train'/j['id']; d.mkdir(parents=True)
    write_once(d/'episodes.json', [{'return': 3.}])
    with gzip.open(d/'losses.jsonl.gz', 'xt') as h:
        for name in ('policy', 'q'): h.write(json.dumps({'component': name, 'value': .5})+'\n')
    trained = {j['id']: {'coverage': {'sample_calls': 1}}}
    r = cli.curve_index(tmp_path, trained, {'train_jobs': [j]})
    assert r['jobs'][j['id']]['optimizer_loss_counts'] == {'q': 1, 'policy': 1}
    assert r['jobs'][j['id']]['optimizer_losses_sha256'] == file_hash(d/'losses.jsonl.gz')
    assert len(json.dumps(r)) < 1000
    trained[j['id']]['coverage']['sample_calls'] = 2
    with pytest.raises(ValueError): cli.curve_index(tmp_path, trained, {'train_jobs': [j]})


def test_prepare_and_load_only_freeze_never_start_training(tmp_path, monkeypatch):
    cp = tmp_path/'config.json'; write_once(cp, PROTOCOL)
    out = tmp_path/'artifacts/p7n/new'; out.mkdir(parents=True)
    previous = {'preset': resolved_preset('cpu', .0002), 'predictors': {'ordinary': 'frozen'},
                'predictor_training_request': {'frozen': True}}
    monkeypatch.setattr(cli, 'ROOT', tmp_path)
    monkeypatch.setattr(cli, 'decision_inputs', lambda **kwargs: ({'receipt': 'hash'}, previous))
    monkeypatch.setattr(cli.shared, 'clean', lambda: None)
    monkeypatch.setattr(cli.shared, 'output_dir', lambda kind, rid: out)
    monkeypatch.setattr(cli.shared, 'source_hashes', lambda: {'source': 'frozen'})
    monkeypatch.setattr(cli.shared, 'metadata', lambda: {'git_commit': 'a'*40, 'working_tree_dirty': False})
    monkeypatch.setattr(cli.smoke, 'dependencies', lambda: {'all': 'frozen'})
    monkeypatch.setattr(cli.original, 'runtime_identity', lambda: {'runtime': 'frozen'})
    monkeypatch.setattr(cli.original, 'upstream_hashes', lambda: {'upstream': 'frozen'})
    monkeypatch.setattr(cli.kernel, 'train_job', lambda *args: pytest.fail('Prepare/load started training'))
    path, h = cli.prepare(str(cp), 'new')
    qp, q = cli.load_request(path, h)
    assert qp == path and read_json(out/'preparation.json')['requested_total_training_steps'] == 1440000
    assert q['previous_policy_weights_used_for_training'] is False
    monkeypatch.setattr(cli.shared, 'source_hashes', lambda: {'source': 'changed'})
    with pytest.raises(ValueError): cli.load_request(path, h)


def test_partial_jobs_are_not_checkpoint_resume(tmp_path, monkeypatch):
    root = tmp_path/'run'; root.mkdir(); path = root/'request.json'
    q = {'protocol': SMOKE, 'train_jobs': jobs(SMOKE, 'train')}
    d = root/'train'/q['train_jobs'][0]['id']; d.mkdir(parents=True)
    write_once(d/'failure.json', {'failed': True})
    monkeypatch.setattr(cli, 'load_request', lambda *args: (path, q))
    monkeypatch.setattr(cli, 'ROOT', tmp_path)
    monkeypatch.setattr(cli.shared, 'metadata', lambda: {})
    monkeypatch.setattr(cli.subprocess, 'run', lambda *args, **kwargs: pytest.fail('Retried incomplete training'))
    with pytest.raises(ValueError): cli.run(path, digest(q), 'train', resume=True)
