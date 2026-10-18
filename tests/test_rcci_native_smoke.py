from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'),str(ROOT/'tools')]
import smoke_rcci_native as cli
from prediction_rl.prediction.model import PredictorEnsemble
from prediction_rl.prediction.interface import PredictorConfig,collate_inputs,probe_plans
from test_road_consequences import scene
from test_rcci_provider import provider


def test_fixed_config_and_full_budget_including_warmup_and_terminal():
    c=cli.SETTINGS
    assert json.loads((ROOT/cli.CONFIG_PATH).read_text()) == c
    assert len(c['arms'])*len(c['replays']) == c['max_sessions'] == 10
    episodes=sum(r['episodes'] for r in c['replays'])*len(c['arms'])
    controlled=sum(r['episodes']*r['step_cap'] for r in c['replays'])*len(c['arms'])
    assert episodes == 15 and controlled == 1800
    assert episodes*101+controlled == c['max_simulation_steps'] == 3315
    assert (320+41)*4 == c['max_ensemble_forward_calls'] == 1444
    assert not c['training'] and not c['policy_loading'] and not c['model_updates']


def test_forward_cache_transparent_one_call_frozen_no_rng():
    p=provider(); h,_,_,_=scene(p.road); cfg=p.config
    x=cli.build_actor_inputs(h,p.road.geometry); batch=collate_inputs([x],cfg); plans=probe_plans(1,cfg)
    recorder=cli.ForwardRecorder(p.model); before=cli.tensor_hash(p.model); rng=cli.rng_hash()
    with torch.inference_mode():
        expected=p.model(batch,plans); actual=recorder(batch,plans)
    assert recorder.calls == 1
    assert cli.tensor_hash(p.model) == before and cli.rng_hash() == rng
    np.testing.assert_array_equal(recorder.last,expected['mean'][0].numpy())
    for key,value in expected.items():
        if isinstance(value,torch.Tensor): torch.testing.assert_close(actual[key],value,rtol=0,atol=0)


def test_unknown_settings_fail_before_reading_parent(monkeypatch):
    bad=deepcopy(cli.SETTINGS); bad['policy_loading']=True
    monkeypatch.setattr(cli,'read',lambda *a:pytest.fail('Opened assets with invalid config'))
    with pytest.raises(ValueError): cli.parent_inputs(bad,'unused')


def test_stale_source_rejected_before_native_loading(tmp_path,monkeypatch):
    monkeypatch.setattr(cli,'ROOT',tmp_path); monkeypatch.setattr(cli.shared,'clean',lambda:None)
    monkeypatch.setattr(cli.shared,'sources',lambda:{'new':'hash'})
    monkeypatch.setattr(cli,'native_identity',lambda *a:pytest.fail('Loaded native asset for stale source'))
    out=tmp_path/'artifacts/rcci03/old'; out.mkdir(parents=True)
    q={'version':cli.VERSION,'config':deepcopy(cli.SETTINGS),'source_hashes':{'old':'hash'}}
    h=cli.digest(q); (out/'request.json').write_text(json.dumps(q))
    with pytest.raises(ValueError): cli.check_request(out/'request.json',h)


def test_native_acceptance_explicit_log_gap_and_binary_identity(tmp_path,monkeypatch):
    monkeypatch.setattr(cli,'ROOT',tmp_path)
    source=tmp_path/'RL-MPC-LaneMerging-master'; source.mkdir()
    for name in ('st_cy.pyd','st_cy.c','st_cy.pyx','setup.py'): (source/name).write_bytes(name.encode())
    old=tmp_path/'artifacts/rcci03_build/native_v1'; old.mkdir(parents=True)
    (old/'report.json').write_text('{}'); (old/'build_log.json').write_text('{}')
    report={'status':'complete_with_build_stdout_capture_gap','import_path_exact':True,
        'extension':'RL-MPC-LaneMerging-master/st_cy.pyd','extension_sha256':cli.sha(source/'st_cy.pyd'),
        'generated_c_sha256':cli.sha(source/'st_cy.c'),'inputs':{n:cli.sha(source/n) for n in ('st_cy.pyx','setup.py')},
        'build_report_sha256':cli.sha(old/'report.json'),'build_log_sha256':cli.sha(old/'build_log.json')}
    path=tmp_path/'native.json'; path.write_text(json.dumps(report))
    config={'native_acceptance':'native.json','native_acceptance_sha256':cli.sha(path)}
    binary=tmp_path/'sumo.exe'; binary.write_bytes(b'fake')
    monkeypatch.setattr(cli.shutil,'which',lambda *a:str(binary))
    monkeypatch.setattr(cli.subprocess,'check_output',lambda *a,**k:'Eclipse SUMO sumo Version 1.22.0\n')
    value=cli.native_identity(config)
    assert not value['build_stdout_capture_complete'] and value['build_gap_explicitly_recorded']
    (source/'st_cy.pyd').write_bytes(b'changed')
    with pytest.raises(ValueError): cli.native_identity(config)


def test_native_worker_has_no_policy_or_optimizer_entry():
    import inspect
    text=inspect.getsource(cli.worker)
    assert 'preset_factory' not in text and 'load_evaluation_weights' not in text and '.train(' not in text
    assert 'load_trained_ensemble' in text and 'cached_offline_features_exact=True' in text
