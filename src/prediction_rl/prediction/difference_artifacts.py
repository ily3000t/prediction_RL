"""Explicit opt-in loader. Legacy trajectory-only consumers remain unchanged."""
import torch

from prediction_rl.data.collection_store import file_hash
from prediction_rl.data.dataset_contract import digest
from .candidate_difference import DIFFERENCE_LOSS_CONTRACT, DIFFERENCE_TRAINING_VERSION, checked_lambda
from .checkpoint import CONTRACT
from .interface import PredictorConfig
from .model import PredictorEnsemble
from .trained_artifacts import checked_state


def load_difference_ensemble(path, request, expected_sha256, selected):
    if (request['version'] != DIFFERENCE_TRAINING_VERSION
            or request['loss_contract'] != DIFFERENCE_LOSS_CONTRACT
            or request['training']['loss_contract'] != DIFFERENCE_LOSS_CONTRACT['version']
            or request['training']['methods'] != ['conditional']):
        raise ValueError('Explicit difference training identity required')
    checked_lambda(request['training']['difference_lambda'])
    seeds = request['training']['member_seeds']
    if len(seeds) != 3 or len(set(seeds)) != 3 or [r['seed'] for r in selected] != seeds:
        raise ValueError('Require all three ordered selected members')
    if file_hash(path) != expected_sha256:
        raise ValueError('Difference ensemble artifact changed')
    payload = torch.load(path, map_location='cpu', weights_only=True)
    expected = {'version': DIFFERENCE_TRAINING_VERSION, 'loss_contract': DIFFERENCE_LOSS_CONTRACT,
        'purpose': 'formal_validation_selected_ensemble', 'contract': CONTRACT,
        'request_hash': digest(request), 'model': request['model'], 'mode': 'conditional',
        'member_seeds': seeds, 'selected': selected, 'data_hashes': request['data_hashes'],
        'calibrated': False, 'test_evaluated': False}
    if (not isinstance(payload, dict) or set(payload) != set(expected) | {'state_dict'}
            or any(payload[k] != v for k, v in expected.items())):
        raise ValueError('Difference ensemble identity/selection mismatch')
    model = PredictorEnsemble(PredictorConfig.from_dict(request['model']), 'conditional', seeds)
    checked_state(model, payload['state_dict'])
    model.load_state_dict(payload['state_dict'], strict=True)
    model.eval(); model.requires_grad_(False)
    return model
