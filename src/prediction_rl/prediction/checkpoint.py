"""Versioned state-dict-only checkpoints; no partial/legacy model loading."""
import json
from pathlib import Path
import uuid

import torch

from prediction_rl.data.actor_adapter import FEATURES, SUMMARY_FEATURES
from prediction_rl.data.merge_geometry import GROUPS
from .interface import PredictorConfig
from .model import PredictorEnsemble


CONTRACT = {'version': 'p4c_response_v1', 'features': list(FEATURES),
            'summary_features': list(SUMMARY_FEATURES), 'summary_groups': list(GROUPS),
            'trajectory': ['root_relative_x_m', 'root_relative_y_m', 'speed_mps', 'acceleration_mps2'],
            'plan': 'one_step_jerk_then_zero_v1', 'scaling': 'fixed_SI_v1'}


def save_checkpoint(path, model, provenance):
    path = Path(path)
    if path.exists():
        raise FileExistsError('Never overwrite predictor evidence: ' + str(path))
    if not isinstance(model, PredictorEnsemble):
        raise ValueError('Expected complete three-member ensemble')
    if not isinstance(provenance, dict) or provenance.get('training_status') != 'untrained_interface_smoke':
        raise ValueError('P4c only serializes explicitly untrained interface models')
    # Only JSON-safe metadata, never pickled arbitrary user classes.
    clean_provenance = json.loads(json.dumps(provenance, allow_nan=False))
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if any(not torch.isfinite(v).all() for v in state.values()):
        raise ValueError('Nonfinite checkpoint weights')
    payload = {'schema_version': 1, 'contract': CONTRACT, 'config': model.config.to_dict(),
               'mode': model.mode, 'initialization_seeds': list(model.initialization_seeds),
               'provenance': clean_provenance, 'state_dict': state}
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / (uuid.uuid4().hex[:12] + '.tmp')
    try:
        torch.save(payload, temporary)
        # Hard-link publication is atomic and refuses a concurrent existing target.
        path.hardlink_to(temporary)
    finally:
        temporary.unlink(missing_ok=True)


def load_checkpoint(path, *, expected_mode, expected_config):
    payload = torch.load(Path(path), map_location='cpu', weights_only=True)
    keys = {'schema_version', 'contract', 'config', 'mode', 'initialization_seeds', 'provenance', 'state_dict'}
    if not isinstance(payload, dict) or set(payload) != keys or payload['schema_version'] != 1:
        raise ValueError('Unsupported checkpoint envelope')
    if payload['contract'] != CONTRACT or payload['mode'] != expected_mode:
        raise ValueError('Checkpoint feature/plan/mode contract mismatch')
    config = PredictorConfig.from_dict(payload['config'])
    if config != expected_config:
        raise ValueError('Checkpoint architecture/config mismatch')
    provenance = payload['provenance']
    if not isinstance(provenance, dict) or provenance.get('training_status') != 'untrained_interface_smoke':
        raise ValueError('Unsupported checkpoint training provenance')
    state = payload['state_dict']
    if not isinstance(state, dict) or any(not isinstance(v, torch.Tensor) or not torch.isfinite(v).all() for v in state.values()):
        raise ValueError('Invalid checkpoint tensor state')
    model = PredictorEnsemble(config, expected_mode, payload['initialization_seeds'])
    expected = model.state_dict()
    if set(state) != set(expected) or any(state[k].shape != expected[k].shape or state[k].dtype != expected[k].dtype for k in expected):
        raise ValueError('Incomplete or incompatible ensemble state; partial loads forbidden')
    # Fixed conversion buffers belong to the contract, not learned parameters.
    for key, value in model.named_buffers():
        if not torch.equal(state[key], value):
            raise ValueError('Changed fixed unit conversion: ' + key)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, provenance
