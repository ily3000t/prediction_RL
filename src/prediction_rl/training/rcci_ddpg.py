"""Dedicated RCCI identity, reusing the unchanged ALL DDPG and strict weight I/O."""
import numpy as np

from prediction_rl.training import all_ddpg as raw
from prediction_rl.features.rcci_provider_cached import FrozenCachedRCCIV2Features
from prediction_rl.features.road_consequences_v2 import DIMENSION, NAMES, VERSION, contract
from prediction_rl.data.dataset_contract import digest

ARMS = ('rcci_ordinary', 'rcci_conditional')
RELEASE = 'rcci04c_matched_policy_interface_v1'
KEYS = frozenset(('version purpose arm observation_dimension time_feature_scale all_version preset '
    'feature_version feature_names_sha256 adapter_contract_sha256 provider_release raw_weights_compatible '
    'predictor_sha256 upstream_config_sha256 execution_contract action_bounds').split())


class MatchedRCCIProvider(FrozenCachedRCCIV2Features):
    release = RELEASE


def preset_factory(device, learning_rate, *, smoke=False):
    from all.presets.continuous import ddpg
    parameters = raw.resolved_preset(device, learning_rate, smoke=smoke)
    def build(env, writer):
        if (env.feature_arm not in ARMS or env.state_space.shape != (169,)
                or env.state_space.dtype != np.float32 or env.action_space.shape != (1,)
                or getattr(env, 'rcci_contract', None) != contract()):
            raise ValueError('Require dedicated RCCI observation identity; no Raw alias')
        low, high = float(env.action_space.low[0]), float(env.action_space.high[0])
        if (low, high) != (-5., 5.): raise ValueError('Original continuous jerk changed')
        body = ddpg(**parameters)(env, writer)
        body.agent.replay_buffer = raw.CheckedReplayBuffer(body.agent.replay_buffer, 170, low, high)
        checked = raw.CheckedAgent(body, 169, low, high)
        checked.arm, checked.preset_parameters = env.feature_arm, dict(parameters)
        checked.rcci_identity = digest(contract())
        if raw.network_shapes(checked) != {'policy': [(170,400),(400,300),(300,1)],
                                           'q': [(171,400),(400,300),(300,1)]}:
            raise ValueError('Matched policy/Q architecture changed')
        return checked
    return build


def checkpoint_contract(env, parameters, predictor_sha256, upstream_config_sha256):
    valid = lambda h: isinstance(h, str) and len(h) == 64 and all(x in '0123456789abcdef' for x in h)
    if (env.feature_arm not in ARMS or env.state_space.shape != (169,)
            or getattr(env, 'rcci_contract', None) != contract()
            or not valid(predictor_sha256) or not valid(upstream_config_sha256)):
        raise ValueError('Invalid dedicated RCCI checkpoint provenance')
    return {'version': 'rcci_ddpg_feature_weights_v1', 'purpose': 'evaluation_only_not_training_resume',
        'arm': env.feature_arm, 'observation_dimension': 20+DIMENSION, 'time_feature_scale': .001,
        'all_version': raw.ALL_VERSION, 'preset': parameters, 'feature_version': VERSION,
        'feature_names_sha256': digest(list(NAMES)), 'adapter_contract_sha256': digest(contract()),
        'provider_release': RELEASE, 'raw_weights_compatible': False, 'predictor_sha256': predictor_sha256,
        'upstream_config_sha256': upstream_config_sha256,
        'execution_contract': 'simulation_blocking_exact_v1', 'action_bounds': [-5.,5.]}


def validate(checked, c):
    raw.validate_agent_contract(checked, c)
    if (set(c)!=KEYS or checked.arm not in ARMS or getattr(checked, 'rcci_identity', None) != digest(contract())
            or c.get('version') != 'rcci_ddpg_feature_weights_v1' or c.get('feature_version') != VERSION
            or c.get('feature_names_sha256') != digest(list(NAMES))
            or c.get('adapter_contract_sha256') != digest(contract()) or c.get('provider_release') != RELEASE
            or c.get('raw_weights_compatible') is not False or c.get('all_version')!=raw.ALL_VERSION
            or c.get('execution_contract')!='simulation_blocking_exact_v1' or c.get('action_bounds')!=[-5.,5.]):
        raise ValueError('RCCI/Raw contract mismatch; no channel relabeling')


def save_evaluation_weights(checked, path, c):
    validate(checked, c)
    return raw.save_evaluation_weights(checked, path, c)


def load_evaluation_weights(checked, path, c, expected_sha256):
    validate(checked, c)
    return raw.load_evaluation_weights(checked, path, c, expected_sha256)
