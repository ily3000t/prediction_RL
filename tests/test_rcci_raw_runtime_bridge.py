from copy import deepcopy
import json
from pathlib import Path
import sys
from types import FunctionType, SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'tools')]
import bridge_rcci_raw_runtime as cli


def test_frozen_roster_budget_and_no_training_or_test():
    c = cli.SETTINGS
    assert json.loads((ROOT/cli.CONFIG).read_text()) == c
    jobs = cli.jobs()
    assert len(jobs) == len({j['id'] for j in jobs}) == c['max_sessions'] == 12
    assert len({j['model'] for j in jobs}) == 6
    assert {j['legacy_job']['simulator_seed'] for j in jobs} == {200}
    assert {j['legacy_job']['checkpoint_frames'] for j in jobs} == {120000}
    assert {j['legacy_job']['run_seed'] for j in jobs} == {0, 1, 2}
    assert len(jobs)*500 == c['max_controlled_steps'] == c['max_ensemble_forward_calls']
    assert len(jobs)*(101+500+1) == c['max_simulation_steps'] == 7224
    assert c['max_member_forward_calls'] == 3*c['max_ensemble_forward_calls']
    assert c['max_policy_forward_calls'] == 12*501
    assert not any(c[k] for k in ('training_permitted', 'formal_evaluation', 'test_opened'))
    jobs[0]['legacy_job']['simulator_seed'] = 999
    assert cli.jobs()[0]['legacy_job']['simulator_seed'] == 200


def module(name):
    def original(x): return x+1
    fn = FunctionType(original.__code__, {'__name__': name}, original.__name__)
    return SimpleNamespace(__name__=name, helper=fn)


def test_st_guard_fails_without_running_original_and_restores():
    st, cy = module('st'), module('st_cy')
    before = (st.helper, cy.helper)
    with cli.NoSTCalls(st, cy) as guard:
        assert guard.names == ['st.helper', 'st_cy.helper'] and not guard.calls
        with pytest.raises(ValueError):
            with cli.NoSTCalls(st, cy): pass
        with pytest.raises(RuntimeError, match='Unexpected ST planning'): cy.helper(1)
        assert guard.calls == ['st_cy.helper']
    assert (st.helper, cy.helper) == before and st.helper(1) == cy.helper(1) == 2
    assert not hasattr(st, '_rcci04b_guard') and not hasattr(cy, '_rcci04b_guard')


def test_guard_refuses_unobservable_module_without_partial_patching():
    st = module('st'); empty = SimpleNamespace(__name__='st_cy')
    before = st.helper
    with pytest.raises(ValueError):
        with cli.NoSTCalls(st, empty): pass
    assert st.helper is before and not hasattr(st, '_rcci04b_guard')


def test_evaluator_bytecode_preserved_dependencies_scoped(monkeypatch):
    def original_evaluator():
        return original, upstream_session, preset_factory
    monkeypatch.setattr(cli.kernel, 'evaluation_job', original_evaluator)
    wrapped = cli.bind_evaluation('assets', 'session', 'factory')
    assert wrapped.__code__ is original_evaluator.__code__
    assert wrapped() == ('assets', 'session', 'factory')
    assert 'original' not in original_evaluator.__globals__


def test_native_hash_or_existing_import_refused_before_loading(tmp_path, monkeypatch):
    p = tmp_path/'st_cy.pyd'; p.write_bytes(b'not-a-native-module')
    monkeypatch.setattr(cli.importlib.util, 'spec_from_file_location', lambda *a: pytest.fail('Imported invalid native module'))
    with pytest.raises(ValueError): cli.load_extension(p, '0'*64)
    monkeypatch.setitem(sys.modules, 'st_cy', object())
    with pytest.raises(ValueError): cli.load_extension(p, cli.sha(p))


def test_stale_source_rejected_before_native_asset_reads(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, 'ROOT', tmp_path)
    monkeypatch.setattr(cli.shared, 'clean', lambda: None)
    monkeypatch.setattr(cli.shared, 'sources', lambda: {'new': 'hash'})
    q = {'config': deepcopy(cli.SETTINGS), 'jobs': cli.jobs(), 'source_hashes': {'old': 'hash'}}
    p = tmp_path/'artifacts/rcci04b/stale/request.json'
    monkeypatch.setattr(cli, 'read', lambda path: deepcopy(cli.SETTINGS) if path == tmp_path/cli.CONFIG else q)
    monkeypatch.setattr(cli, 'load_extension', lambda *a: pytest.fail('Loaded extension for stale request'))
    with pytest.raises(ValueError): cli.check_request(p, cli.digest(q))
