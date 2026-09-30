"""Reuse ALL 0.5.3 DDPG; only boundary guards and observation adaptation are new."""
import importlib.metadata
import inspect
import math
from pathlib import Path

import numpy as np
import torch

from prediction_rl.data.dataset_contract import digest
from prediction_rl.data.collection_store import file_hash
from prediction_rl.features.probe_features import DIMENSION, NAMES, VERSION
from prediction_rl.prediction.epoch_training import atomic_torch

ALL_VERSION = '0.5.3'
DEFAULTS = {'device':'cuda', 'discount_factor':.98, 'last_frame':2e6,
            'lr_q':1e-3, 'lr_pi':1e-3, 'minibatch_size':100, 'update_frequency':1,
            'polyak_rate':.005, 'replay_start_size':5000, 'replay_buffer_size':1e6, 'noise':.1}
SMOKE_OVERRIDES = {'replay_start_size':8, 'minibatch_size':8}


class LegacyFeatureView:
    """ALL expects reset->obs, not reset->(obs,info).

    Detailed observed-traffic audits are available to an external recorder, not
    copied into every replay State.info. Original upstream info is preserved.
    """
    def __init__(self, feature_env, recorder=None):
        self.feature_env, self.recorder = feature_env, recorder
        self.observation_space = feature_env.observation_space
        self.action_space = feature_env.action_space
        self.last_info = None

    def reset(self):
        obs, info = self.feature_env.reset(); self.last_info = info
        if self.recorder is not None: self.recorder('reset',obs,None,False,info)
        return obs

    def step(self, action):
        obs, reward, done, info = self.feature_env.step(action); self.last_info = info
        if self.recorder is not None: self.recorder('step',obs,reward,done,info)
        lean = {k:v for k,v in info.items() if k not in ('execution_audit','prediction_features')}
        return obs, reward, done, lean

    def close(self): self.feature_env.close()


def make_all_environment(feature_env, device, recorder=None):
    from all.environments import GymEnvironment
    env = GymEnvironment(LegacyFeatureView(feature_env,recorder),device=device)
    env._name = 'sumo-jerk-continuous-v0'
    env.feature_arm = feature_env.mode
    return env


def validate_state(state, dimension, size=None):
    if (not isinstance(state.features,torch.Tensor) or state.features.ndim!=2
            or state.features.shape[1]!=dimension or state.features.shape[0]<1
            or (size is not None and state.features.shape[0]!=size)
            or state.features.dtype!=torch.float32 or not torch.isfinite(state.features).all()
            or state.mask.shape!=(state.features.shape[0],) or state.mask.dtype!=torch.uint8
            or not ((state.mask==0)|(state.mask==1)).all()):
        raise ValueError('Invalid DDPG observation/time/mask contract')


def validate_action(action, size, low, high):
    if (not isinstance(action,torch.Tensor) or action.shape!=(size,1)
            or action.dtype!=torch.float32 or not torch.isfinite(action).all()
            or (action<low).any() or (action>high).any()):
        raise ValueError('Invalid continuous jerk; do not repair or snap it')


class CheckedReplayBuffer:
    """Delegates ALL storage/sampling once; does not alter sampling or returns."""
    def __init__(self, buffer, dimension, low, high):
        self.inner, self.dimension, self.low, self.high = buffer, dimension, low, high
    def store(self, state, action, reward, next_state):
        validate_state(next_state,self.dimension,1)
        if not math.isfinite(float(reward)): raise ValueError('Nonfinite reward')
        if state is not None:
            validate_state(state,self.dimension,1)
            validate_action(action,1,self.low,self.high)
        return self.inner.store(state,action,reward,next_state)
    def sample(self, batch_size):
        result = self.inner.sample(batch_size)
        states, actions, rewards, next_states, weights = result
        validate_state(states,self.dimension,batch_size)
        validate_state(next_states,self.dimension,batch_size)
        validate_action(actions,batch_size,self.low,self.high)
        if (rewards.shape!=(batch_size,) or weights.shape!=(batch_size,)
                or not torch.isfinite(rewards).all() or not torch.isfinite(weights).all()):
            raise ValueError('Invalid replay minibatch')
        return result
    def update_priorities(self,*args): return self.inner.update_priorities(*args)
    def __len__(self): return len(self.inner)
    def __iter__(self): return iter(self.inner)


class CheckedAgent:
    def __init__(self, body, dimension, low, high):
        self.body, self.dimension, self.low, self.high = body, dimension, low, high
        self.evaluation_only = False
    @property
    def agent(self): return self.body.agent
    def act(self, state, reward):
        if self.evaluation_only: raise RuntimeError('Weights-only checkpoint cannot resume training')
        validate_state(state,self.dimension,1)
        if not math.isfinite(float(reward)): raise ValueError('Nonfinite reward')
        action = self.body.act(state,reward)
        validate_action(action,1,self.low,self.high)
        return action
    def eval(self, state, reward):
        validate_state(state,self.dimension,1)
        if not math.isfinite(float(reward)): raise ValueError('Nonfinite reward')
        action = self.body.eval(state,reward)
        validate_action(action,1,self.low,self.high)
        return action


def resolved_preset(device, learning_rate, *, smoke=False):
    from all.presets.continuous import ddpg
    defaults = {k:v.default for k,v in inspect.signature(ddpg).parameters.items()}
    if importlib.metadata.version('autonomous-learning-library')!=ALL_VERSION or defaults!=DEFAULTS:
        raise ValueError('Installed upstream DDPG preset changed; re-audit before training')
    if device not in ('cpu','cuda') or not math.isfinite(learning_rate) or learning_rate<=0 or type(smoke)is not bool:
        raise ValueError('Invalid DDPG configuration')
    return {**defaults,'device':device,'lr_q':learning_rate,'lr_pi':learning_rate,
            **(SMOKE_OVERRIDES if smoke else {})}


def network_shapes(agent):
    return {name:[(layer.in_features,layer.out_features) for layer in approximation.model.modules()
                  if isinstance(layer,torch.nn.Linear)]
            for name,approximation in (('policy',agent.agent.policy),('q',agent.agent.q))}


def preset_factory(device, learning_rate, *, smoke=False):
    from all.presets.continuous import ddpg
    parameters = resolved_preset(device,learning_rate,smoke=smoke)
    def _ddpg(env, writer):
        if env.feature_arm not in ('baseline','zero','ordinary','conditional'):
            raise ValueError('Unknown DDPG comparison arm')
        size=env.state_space.shape[0]
        expected=20 if env.feature_arm=='baseline' else 20+DIMENSION
        if size!=expected or env.state_space.dtype!=np.float32 or env.action_space.shape!=(1,):
            raise ValueError('Wrong original/expanded policy observation layout')
        low,high = float(env.action_space.low[0]),float(env.action_space.high[0])
        if (low,high)!=(-5.,5.): raise ValueError('Original continuous jerk bounds changed')
        body = ddpg(**parameters)(env,writer)
        body.agent.replay_buffer = CheckedReplayBuffer(body.agent.replay_buffer,size+1,low,high)
        result = CheckedAgent(body,size,low,high)
        result.arm=env.feature_arm
        result.preset_parameters=dict(parameters)
        if network_shapes(result)!={'policy':[(size+1,400),(400,300),(300,1)],
                                   'q':[(size+2,400),(400,300),(300,1)]}:
            raise ValueError('Original network architecture/time feature changed')
        return result
    return _ddpg


def checkpoint_contract(env, parameters, predictor_sha256, upstream_config_sha256):
    arm=env.feature_arm
    if arm not in ('baseline','zero','ordinary','conditional'):
        raise ValueError('Unknown checkpoint arm')
    learned=arm in ('ordinary','conditional')
    def valid_sha(value): return isinstance(value,str) and len(value)==64 and all(x in '0123456789abcdef' for x in value)
    if (learned and not valid_sha(predictor_sha256)) or (not learned and predictor_sha256 is not None) or not valid_sha(upstream_config_sha256):
        raise ValueError('Missing/inapplicable checkpoint provenance')
    return {'version':'ddpg_feature_weights_v1','purpose':'evaluation_only_not_training_resume',
            'arm':arm,'observation_dimension':env.state_space.shape[0],
            'time_feature_scale':.001,'all_version':ALL_VERSION,'preset':parameters,
            'feature_version':None if arm=='baseline' else VERSION,
            'feature_names_sha256':None if arm=='baseline' else digest(list(NAMES)),
            'predictor_sha256':predictor_sha256,'upstream_config_sha256':upstream_config_sha256,
            'execution_contract':'simulation_blocking_exact_v1','action_bounds':[-5.,5.]}


def save_evaluation_weights(agent, path, contract):
    if agent.evaluation_only: raise ValueError('Do not relabel loaded evaluation weights as trained')
    validate_agent_contract(agent,contract)
    state = {name:{k:v.detach().cpu().clone() for k,v in approximation.model.state_dict().items()}
             for name,approximation in (('policy',agent.agent.policy),('q',agent.agent.q))}
    if any(not torch.isfinite(v).all() for component in state.values() for v in component.values()):
        raise ValueError('Do not save invalid DDPG weights')
    atomic_torch(path,{'contract':contract,'state_dicts':state})
    return file_hash(path)


def load_evaluation_weights(agent, path, contract, expected_sha256):
    validate_agent_contract(agent,contract)
    if file_hash(path)!=expected_sha256: raise ValueError('DDPG weights hash changed')
    payload=torch.load(Path(path),map_location='cpu',weights_only=True)
    if (not isinstance(payload,dict) or set(payload)!={'contract','state_dicts'}
            or payload['contract']!=contract or set(payload['state_dicts'])!={'policy','q'}):
        raise ValueError('Checkpoint arm/features/predictor/preset identity mismatch')
    # Check BOTH components completely BEFORE mutating either; no partial loading.
    for name,approximation in (('policy',agent.agent.policy),('q',agent.agent.q)):
        state=payload['state_dicts'][name]; template=approximation.model.state_dict()
        if (not isinstance(state,dict) or set(state)!=set(template) or any(not isinstance(state[k],torch.Tensor)
                or state[k].shape!=v.shape or state[k].dtype!=v.dtype
                or not torch.isfinite(state[k]).all() for k,v in template.items())):
            raise ValueError('Incomplete/incompatible/nonfinite '+name+' weights')
    for name,approximation in (('policy',agent.agent.policy),('q',agent.agent.q)):
        approximation.model.load_state_dict(payload['state_dicts'][name],strict=True)
    agent.evaluation_only=True


def validate_agent_contract(agent,contract):
    if (contract.get('arm')!=agent.arm or contract.get('observation_dimension')!=agent.dimension
            or contract.get('purpose')!='evaluation_only_not_training_resume'
            or contract.get('time_feature_scale')!=agent.body.scale
            or contract.get('preset')!=agent.preset_parameters):
        raise ValueError('Checkpoint metadata does not describe this agent')


def experiment_type(output):
    """Only writer initialization differs; ALL episode/train logic is inherited."""
    from all.experiments import SingleEnvExperiment
    from all.experiments.writer import ExperimentWriter
    from tensorboardX import SummaryWriter
    directory=Path(output).resolve()
    class OutputWriter(ExperimentWriter):
        def __init__(self,experiment,agent_name,env_name,loss=True):
            (directory/env_name).mkdir(parents=True,exist_ok=False)
            self.env_name,self.log_dir=env_name,str(directory)
            self._experiment,self._loss=experiment,loss
            SummaryWriter.__init__(self,log_dir=self.log_dir)
    class FeatureExperiment(SingleEnvExperiment):
        def _make_writer(self,agent_name,env_name,write_loss):
            return OutputWriter(self,agent_name,env_name,write_loss)
    return FeatureExperiment
