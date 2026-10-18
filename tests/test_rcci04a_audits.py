from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'tools')]
from audit_rcci_cached_geometry import equal_results, SETTINGS as CACHE
from audit_rcci_raw_reuse import check_weights, sealed, sha, SETTINGS as RAW
from prediction_rl.data.collection_store import write_once, seal_episode


def test_complete_diagnostics_not_only_float_tolerance():
    a = {'features': np.arange(149, dtype=np.float32), 'candidates': [{'actor': 'a', 'prefix': .2}]}
    b = {'features': a['features'].copy(), 'candidates': [{'actor': 'a', 'prefix': .2}]}
    equal_results(a, b)
    b['candidates'][0]['prefix'] = .4
    with pytest.raises(ValueError): equal_results(a, b)
    b['candidates'][0]['prefix'] = .2
    b['features'][0] = np.nextafter(np.float32(0), np.float32(1))
    with pytest.raises(AssertionError): equal_results(a, b)
    assert CACHE['full_cached_summaries']+CACHE['native_cached_summaries'] == 1798
    assert CACHE['predictor_forward_calls'] == RAW['simulation_episodes'] == 0


def test_streaming_receipt_verification_rejects_mutation_extra_and_binding(tmp_path):
    d = tmp_path/'run'; d.mkdir()
    binding = {'job': 'frozen'}; metadata = {'status': 'complete'}
    write_once(d/'started.json', {'status': 'running'}); write_once(d/'report.json', metadata)
    expected = seal_episode(d, binding, metadata)
    assert sealed(d, binding) == expected
    assert sha(d/'report.json') == expected['files']['report.json']
    with pytest.raises(ValueError): sealed(d, {'job': 'other'})
    write_once(d/'extra.json', {})
    with pytest.raises(ValueError): sealed(d, binding)


def test_raw_weights_are_inspected_without_agent_or_partial_loading():
    contract = {'arm': 'ordinary'}
    states = {}
    for name, n in (('policy', 170), ('q', 171)):
        states[name] = {f'model.{i}.{kind}': torch.zeros(shape)
            for i, input_size, output_size in ((0, n, 400), (2, 400, 300), (4, 300, 1))
            for kind, shape in (('weight', (output_size, input_size)), ('bias', (output_size,)))}
    payload = {'contract': contract, 'state_dicts': states}
    check_weights(payload, contract)
    with pytest.raises(ValueError): check_weights(payload, {'arm': 'rcci_ordinary'})
    payload['state_dicts']['q']['model.4.bias'][0] = float('nan')
    with pytest.raises(ValueError): check_weights(payload, contract)
