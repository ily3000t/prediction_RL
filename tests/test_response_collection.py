from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'tools'))
from prediction_rl.data.collection_store import (
    write_once, read_json, file_hash, run_lock, seal_episode, verify_episode,
)
from prediction_rl.data.response_collection import base_traffic, validate_future, branch_with_geometry, root_accounting
from prediction_rl.data.merge_geometry import MergeGeometry
import prediction_rl.data.response_collection as collection
import audit_response_collection as cli


@pytest.fixture
def geometry():
    return MergeGeometry(ROOT / 'RL-MPC-LaneMerging-master/merge.net.xml')


def frame(ego=True):
    state = {'position': [100., 0.], 'speed': 8., 'acceleration': 0.,
             'lane_id': 'highwayrear_0', 'lane_position_m': 100., 'length_m': 5., 'width_m': 1.8}
    return {'simulation_time_s': .2, 'vehicles': {'ego' if ego else 'neighbor': state}}


def transition(f, done=False):
    return {'done': done, 'info': {'execution_audit': {'after': base_traffic(f)}}}


def test_future_metadata_separate_and_no_mutation(geometry):
    f = frame(); t = transition(f); before = deepcopy(t)
    out = validate_future([t], [f], geometry)
    assert out['trajectory_usable'] == [True]
    assert t == before and 'lane_position_m' not in t['info']['execution_audit']['after']['vehicles']['ego']


def test_terminal_without_ego_preserved_but_masked(geometry):
    f = frame(False)
    assert validate_future([transition(f, True)], [f], geometry)['trajectory_usable'] == [False]
    with pytest.raises(ValueError, match='requires ego'):
        validate_future([transition(f)], [f], geometry)
    with pytest.raises(ValueError, match='requires ego'):
        geometry.validate_frame(f)  # Existing history contract is unchanged.


@pytest.mark.parametrize('change', ['time', 'speed', 'actor', 'position'])
def test_metadata_cannot_be_from_another_transition(geometry, change):
    f = frame(); t = transition(deepcopy(f))
    if change == 'time': f['simulation_time_s'] += .2
    if change == 'speed': f['vehicles']['ego']['speed'] += 1
    if change == 'actor': f['vehicles']['other'] = f['vehicles'].pop('ego')
    if change == 'position': f['vehicles']['ego']['position'][0] += 1
    with pytest.raises(ValueError, match='audited transition'):
        validate_future([t], [f], geometry)


@pytest.mark.parametrize('key,value', [('lane_position_m', -1), ('length_m', 0),
                                      ('width_m', float('nan')), ('lane_id', 'unknown')])
def test_invalid_future_geometry_rejected_even_on_terminal(geometry, key, value):
    f = frame(False); f['vehicles']['neighbor'][key] = value
    with pytest.raises(ValueError):
        validate_future([transition(f, True)], [f], geometry)


def test_metadata_requires_exact_tail(geometry):
    f = frame()
    with pytest.raises(ValueError): validate_future([], [], geometry)
    with pytest.raises(ValueError): validate_future([transition(f)], [], geometry)
    with pytest.raises(ValueError): validate_future([transition(f, True), transition(f)], [f, f], geometry)


def test_collection_reads_only_after_step_and_stops_at_terminal(geometry, monkeypatch):
    events = []; f = frame(False)
    manager = SimpleNamespace(restore=lambda root: events.append(('restore', root)))
    def rollout(env, actions):
        events.append(('step', actions))
        return [transition(f, True)]
    monkeypatch.setattr(collection, 'rollout', rollout)
    def reader():
        events.append(('read', None)); return f
    trace, future = branch_with_geometry(manager, 'root', [-5, 0, 0], geometry, reader=reader)
    assert events == [('restore', 'root'), ('step', [-5]), ('read', None)]
    assert len(trace) == 1 and future['trajectory_usable'] == [False]


def test_unavailable_roots_are_accounted_without_replacement():
    rows = [{'target': {'id': 'a'}, 'prefix_steps': None, 'reason': 'upstream_arrival'},
            {'target': {'id': 'b'}, 'prefix_steps': 1, 'prefix_actions': [0]}]
    out = root_accounting(rows)
    assert len(out) == 2 and out[0]['status'] == 'unavailable' and out[0]['candidate_count'] == 0
    assert out[1]['status'] == 'pending'
    with pytest.raises(ValueError): root_accounting([rows[0], rows[0]])
    with pytest.raises(ValueError): root_accounting([{'target': {'id': 'a'}, 'prefix_steps': None}])
    with pytest.raises(ValueError): root_accounting([{'target': {'id': 'a'}, 'prefix_steps': True, 'prefix_actions': [0]}])


def test_write_once_atomic_and_no_overwrite(tmp_path):
    path = tmp_path / 'r.json'; write_once(path, {'value': 1})
    h = file_hash(path)
    with pytest.raises(FileExistsError): write_once(path, {'value': 2})
    assert file_hash(path) == h and list(tmp_path.iterdir()) == [path]
    with pytest.raises(ValueError): write_once(tmp_path / 'nan.json', {'x': float('nan')})
    assert not (tmp_path / 'nan.json').exists()


def test_exclusive_lock_and_exception_release(tmp_path):
    with pytest.raises(RuntimeError, match='test failure'):
        with run_lock(tmp_path):
            original = read_json(tmp_path / 'writer.lock')
            with pytest.raises(FileExistsError):
                with run_lock(tmp_path): pass
            assert read_json(tmp_path / 'writer.lock') == original
            raise RuntimeError('test failure')
    assert not (tmp_path / 'writer.lock').exists()


def test_stale_lock_not_automatically_deleted(tmp_path):
    write_once(tmp_path / 'writer.lock', {'pid': -1, 'token': 'old'})
    with pytest.raises(FileExistsError):
        with run_lock(tmp_path): pass
    assert read_json(tmp_path / 'writer.lock')['token'] == 'old'


def completed(tmp_path):
    write_once(tmp_path / 'started.json', {'seed': 0})
    write_once(tmp_path / 'r0/history.json', {'a': [1, 2]})
    binding = {'request_hash': 'abc', 'seed': 0}
    seal_episode(tmp_path, binding, {'roots': []})
    return binding


def test_completed_episode_reuse_is_read_only(tmp_path):
    binding = completed(tmp_path)
    before = {str(p): file_hash(p) for p in tmp_path.rglob('*.json')}
    assert verify_episode(tmp_path, binding)['status'] == 'complete'
    assert before == {str(p): file_hash(p) for p in tmp_path.rglob('*.json')}
    with pytest.raises(ValueError, match='mismatch'): verify_episode(tmp_path, {**binding, 'seed': 1})


@pytest.mark.parametrize('mutation', ['change', 'remove', 'add'])
def test_resume_rejects_artifact_mutation(tmp_path, mutation):
    binding = completed(tmp_path)
    path = tmp_path / 'r0/history.json'
    if mutation == 'change': path.write_text('{}', encoding='utf-8')
    if mutation == 'remove': path.unlink()
    if mutation == 'add': write_once(tmp_path / 'extra.json', {})
    with pytest.raises(ValueError, match='artifacts changed'): verify_episode(tmp_path, binding)


def test_failed_incomplete_empty_never_reused_or_sealed(tmp_path):
    with pytest.raises(ValueError): seal_episode(tmp_path, {}, {})
    write_once(tmp_path / 'started.json', {})
    with pytest.raises(ValueError, match='automatic retry'): verify_episode(tmp_path, {})
    write_once(tmp_path / 'failure.json', {'reason': 'diagnostic'})
    with pytest.raises(ValueError): seal_episode(tmp_path, {}, {})
    assert read_json(tmp_path / 'failure.json')['reason'] == 'diagnostic'


@pytest.mark.parametrize('key,value', [('simulator_seeds', [11000]), ('horizon_steps', 26),
    ('branch_repeats', 1), ('worker_wall_limit_s', 301), ('schema_version', True),
    ('purpose', 'formal'), ('unknown', 1)])
def test_cli_cannot_expand_to_formal_or_silently_change_bounds(key, value):
    c = read_json(ROOT / 'configs/development/p04_response_collection_v1.json')
    c[key] = value
    with pytest.raises(ValueError): cli.validate_config(c)


def test_default_cli_config_is_valid():
    cli.validate_config(read_json(ROOT / 'configs/development/p04_response_collection_v1.json'))


def test_source_newline_normalization_does_not_hide_code_changes(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, 'ROOT', tmp_path)
    path = tmp_path / 'code.py'
    path.write_bytes(b'x = 1\n')
    hashes = {'python_lf:code.py': cli.fingerprint(path.read_text(encoding='utf-8'))}
    path.write_bytes(b'x = 1\r\n')
    cli.verify_inputs(hashes)
    path.write_bytes(b'x = 2\r\n')
    with pytest.raises(ValueError, match='Input/source changed'): cli.verify_inputs(hashes)


def test_parent_resume_never_launches_completed_workers(tmp_path, monkeypatch):
    c = {'simulator_seeds': [0, 1, 100], 'worker_wall_limit_s': 300}
    monkeypatch.setattr(cli, 'ROOT', tmp_path)
    monkeypatch.setattr(cli, 'prerequisites', lambda p: (c, {}, {}, {}))
    monkeypatch.setattr(cli, 'environment_metadata', lambda: {'test': True})
    monkeypatch.setattr(cli.subprocess, 'check_output', lambda a, **k: '' if '--porcelain' in a else 'commit')
    monkeypatch.setattr(cli, 'aggregate', lambda *a: {'status': 'complete', 'collector_gate': True})
    calls = []
    def child(command, **kwargs):
        request_path = Path(command[command.index('--request')+1])
        request = read_json(request_path); seed = int(command[command.index('--seed')+1])
        directory = request_path.parent / f'e{c["simulator_seeds"].index(seed):03d}'
        write_once(directory / 'started.json', {})
        seal_episode(directory, {'request_hash': cli.digest(request), 'seed': seed}, {})
        calls.append(seed)
    monkeypatch.setattr(cli.subprocess, 'run', child)
    args = SimpleNamespace(config='unused', run_id='audit', resume=False)
    first = cli.run(args)
    assert calls == [0, 1, 100]
    first_hash = file_hash(first)
    args.resume = True
    second = cli.run(args)
    assert calls == [0, 1, 100] and first != second and file_hash(first) == first_hash
    assert read_json(second)['reused_episodes'] == [0, 1, 100]
    assert read_json(second)['executed_episodes'] == []


def test_parent_records_failure_and_wont_retry_partial_episode(tmp_path, monkeypatch):
    c = {'simulator_seeds': [0, 1, 100], 'worker_wall_limit_s': 300}
    monkeypatch.setattr(cli, 'ROOT', tmp_path)
    monkeypatch.setattr(cli, 'prerequisites', lambda p: (c, {}, {}, {}))
    monkeypatch.setattr(cli, 'environment_metadata', lambda: {})
    monkeypatch.setattr(cli.subprocess, 'check_output', lambda a, **k: '' if '--porcelain' in a else 'commit')
    calls = []
    def child(command, **kwargs):
        directory = Path(command[command.index('--request')+1]).parent / 'e000'
        write_once(directory / 'started.json', {})
        calls.append(0)
        raise RuntimeError('worker failed')
    monkeypatch.setattr(cli.subprocess, 'run', child)
    args = SimpleNamespace(config='unused', run_id='audit', resume=False)
    with pytest.raises(RuntimeError, match='worker failed'): cli.run(args)
    args.resume = True
    with pytest.raises(ValueError, match='automatic retry'): cli.run(args)
    assert calls == [0]
    reports = list((tmp_path / 'artifacts/p4/audit/invocations').glob('*/report.json'))
    assert len(reports) == 2 and all(read_json(p)['status'] == 'failed' for p in reports)
