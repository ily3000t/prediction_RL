from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import sys

import gym
import numpy as np
import pytest
import torch
from all.environments import State
from all.logging import DummyWriter

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from prediction_rl.evaluation.training_contract import TransitionWitness, install_witness, coverage, verify_reward
from prediction_rl.training.all_ddpg import preset_factory, make_all_environment, experiment_type
from prediction_rl.envs.upstream import upstream_session


class SyntheticEnv:
    """Known observations/actions/outcomes; original ALL does all storage/training."""
    def __init__(self, arm):
        self.mode=arm; self.size=20 if arm=='baseline' else 169; self.episode=-1
        self.observation_space=gym.spaces.Box(-np.inf,np.inf,(self.size,),np.float32)
        self.action_space=gym.spaces.Box(-5,5,(1,),np.float32)
    def info(self, action=0., reason=None):
        frame={'simulation_time_s':float(self.tick)*.2,'vehicles':{'ego':{'speed':1.,'acceleration':0.,
               'position':[float(self.tick),0.],'lane_id':'ramp_0'}}}
        return {'traffic':frame,'prediction_features':{'history_frames':1 if self.mode in ('ordinary','conditional') else 0},
                'execution_audit':{'after':frame,'requested_jerk':action,'measured_jerk':0.,
                                   'termination_reason':reason,'invalid_action_reward':0.}}
    def reset(self):
        self.episode+=1;self.tick=0
        return np.full(self.size,self.episode,np.float32),self.info()
    def step(self, action):
        self.tick+=1; done=self.tick==3
        reason=('upstream_collision','upstream_arrival','upstream_time_limit')[self.episode%3] if done else None
        reward=-10. if reason=='upstream_collision' else 10. if reason=='upstream_arrival' else -.02
        obs=np.full(self.size,float(action[0])+self.tick,np.float32)
        return obs,reward,done,self.info(float(action[0]),reason)
    def close(self): pass


def synthetic(tmp_path, arm, watched):
    torch.manual_seed(555);np.random.seed(555)
    witness=TransitionWitness();raw=SyntheticEnv(arm)
    env=make_all_environment(raw,'cpu',witness.record if watched else None)
    experiment=experiment_type(tmp_path)(preset_factory('cpu',2e-4,smoke=True),env,quiet=True)
    wrapper=install_witness(experiment._agent,witness) if watched else None
    experiment.train(frames=35,episodes=12)
    experiment._writer.close()
    return experiment._agent,wrapper,witness


@pytest.mark.parametrize('arm',['baseline','zero','ordinary','conditional'])
def test_witness_does_not_change_actions_rng_optimizers_targets_or_replay(tmp_path,arm):
    torch.set_num_threads(1)
    plain,_,_=synthetic(tmp_path/'plain',arm,False)
    expected_rng=torch.get_rng_state().clone();expected_numpy=deepcopy(np.random.get_state())
    checked,wrapper,witness=synthetic(tmp_path/'watched',arm,True)
    assert torch.equal(torch.get_rng_state(),expected_rng)
    actual_numpy=np.random.get_state();np.testing.assert_array_equal(actual_numpy[1],expected_numpy[1]);assert actual_numpy[2:]==expected_numpy[2:]
    assert witness.finish()=={'transitions':36,'reset_calls':12,'terminal_transitions':12,'reset_bridge_transitions':0,'aligned':True}
    assert wrapper.losses and sum(x['terminal_samples'] for x in wrapper.samples)>0
    for name in ('policy','q'):
        a,b=getattr(plain.agent,name),getattr(checked.agent,name)
        for k,v in a.model.state_dict().items():assert torch.equal(v,b.model.state_dict()[k])
        for k,v in a._target._target.state_dict().items():assert torch.equal(v,b._target._target.state_dict()[k])
        assert a._scheduler.state_dict()==b._scheduler.state_dict()
        for state,other in zip(a._optimizer.state.values(),b._optimizer.state.values()):
            for k,v in state.items():assert torch.equal(v,other[k])
    for a,b in zip(plain.agent.replay_buffer,checked.agent.replay_buffer):
        assert torch.equal(a[0].features,b[0].features) and torch.equal(a[1],b[1])
        assert a[2]==b[2] and torch.equal(a[3].features,b[3].features) and torch.equal(a[3].mask,b[3].mask)


@pytest.mark.parametrize('fault',['action','reward','next_observation','time','mask','reset_bridge'])
def test_witness_rejects_corrupt_alignment(tmp_path,fault):
    raw=SyntheticEnv('baseline');w=TransitionWitness()
    obs,info=raw.reset();w.record('reset',obs,None,False,info)
    initial=State(torch.tensor(np.r_[obs,0.]).float().view(1,-1),torch.tensor([1],dtype=torch.uint8))
    w.on_store(None,None,0.,initial)
    action=torch.tensor([[.7]],dtype=torch.float32)
    obs,reward,done,info=raw.step(np.array([.7],np.float32));w.record('step',obs,reward,done,info)
    following=State(torch.tensor(np.r_[obs,np.float32(.001)]).view(1,-1),torch.tensor([1],dtype=torch.uint8))
    if fault=='action': action=torch.tensor([[.8]])
    if fault=='reward': reward+=1
    if fault=='next_observation': following.features[0,0]+=1
    if fault=='time': following.features[0,-1]+=.001
    if fault=='mask': following.mask[0]=0
    if fault=='reset_bridge': initial.mask[0]=0
    with pytest.raises(ValueError):w.on_store(initial,action,reward,following)


@pytest.mark.parametrize('reason',['collision','arrival','deadline','normal'])
def test_actual_upstream_reward_done_and_cleanup(monkeypatch,reason):
    with upstream_session(ROOT/'RL-MPC-LaneMerging-master',ROOT/'RL-MPC-LaneMerging-master/configs/train_default_1.json',0) as settings:
        import merge_gym,control,dqn,prediction
        raw=object.__new__(merge_gym.ContinuousJerkEnv)
        raw.control_history=[];raw.current_episode_ticks=0;raw.max_episode_ticks=1 if reason=='deadline' else 500
        raw.observation_dim=20;raw.previous_acceleration=0.;raw.previous_state=SimpleNamespace(ego_speed=7.)
        raw.penalty_for_invalid_action=settings.INVALID_ACTION_PENALTY;raw.crashed=raw.merged=False
        raw.reward_function=dqn.get_reward_function()
        for name in ('speed_history','position_history','acceleration_history','jerk_history','state_history'):setattr(raw,name,[])
        state=SimpleNamespace(ego_speed=7.,ego_acceleration=.02,ego_position=(0.,0.))
        calls=[]
        monkeypatch.setattr(control,'set_ego_jerk',lambda action:calls.append('command'))
        monkeypatch.setattr(control,'step',lambda:calls.append('step'))
        monkeypatch.setattr(control,'remove_ego_car',lambda:calls.append('remove'))
        monkeypatch.setattr(control,'just_had_collision',lambda:reason=='collision')
        monkeypatch.setattr(control,'ego_just_arrived',lambda:reason=='arrival')
        monkeypatch.setattr(prediction.HighwayState,'from_sumo',staticmethod(lambda:state))
        monkeypatch.setattr(prediction.HighwayState,'empty_state',staticmethod(lambda:state))
        monkeypatch.setattr(dqn,'get_state_vector_from_base_state',lambda state:np.ones(20))
        obs,reward,done,info=raw.step(np.array([0.],np.float32))
        assert done==(reason!='normal')
        expected=-10 if reason=='collision' else 10 if reason=='arrival' else settings.TIME_REWARD*.2-settings.ALT_J_WEIGHT*(.02/.2)**2*.2
        assert float(reward)==pytest.approx(expected+raw.invalid_action_reward)
        assert calls==(['command','step','remove','step'] if reason=='deadline' else ['command','step'])
        assert np.all(obs==(0 if reason in ('collision','arrival') else 1))


def test_registered_time_limit_also_masks_bootstrap():
    sys.path.insert(0,str(ROOT/'RL-MPC-LaneMerging-master'))
    from legacy_gym_compat import LegacyTimeLimit
    class Raw(gym.Env):
        observation_space=gym.spaces.Box(-1,1,(20,),np.float32)
        action_space=gym.spaces.Box(-5,5,(1,),np.float32)
        def reset(self):return np.zeros(20,np.float32)
        def step(self,a):return np.ones(20,np.float32),-.1,False,{}
    from all.environments import GymEnvironment
    env=GymEnvironment(LegacyTimeLimit(Raw(),1));state=env.reset()
    following,reward=env.step(torch.zeros(1,1))
    assert following.done and following.info[0]['TimeLimit.truncated'] is True
    model=SimpleNamespace(feature_arm='baseline',state_space=env.state_space,action_space=env.action_space)
    a=preset_factory('cpu',2e-4,smoke=True)(model,DummyWriter())
    terminal=State(torch.cat([following.features,torch.zeros(1,1)],1),following.mask)
    assert a.agent.q.target(terminal,a.agent.policy.target(terminal)).item()==0.


def test_reward_mismatch_is_not_repaired():
    settings=SimpleNamespace(REWARD_FUNCTION='Slotted Jerk',CRASH_REWARD=-10,SUCCESS_REWARD=10,TIME_REWARD=-.1,ALT_J_WEIGHT=.1,TICK_LENGTH=.2)
    rows=[{'reason':'upstream_collision','reward':-10.,'execution_audit':{'invalid_action_reward':0.,'measured_jerk':None}}]
    assert verify_reward(rows,settings)['checked_transitions']==1
    rows[0]['reward']=10.
    with pytest.raises(ValueError):verify_reward(rows,settings)


def test_coverage_missing_history_is_not_zero_filled_as_learning_evidence():
    raw=SyntheticEnv('baseline');obs,info=raw.reset()
    from prediction_rl.evaluation.training_contract import state_measurement
    c=coverage([state_measurement(obs,info)])
    assert c['short_history_fraction'] is None
    assert c['scope']=='new_bounded_trajectory_not_historical_20k_replay'


def test_cli_budget_is_bounded_and_config_matches():
    import json
    sys.path.insert(0,str(ROOT/'tools'))
    from audit_training_contract import CONFIG, JOBS
    assert json.loads((ROOT/'configs/development/p07j_training_contract_v1.json').read_text())==CONFIG
    assert CONFIG['training_60k_authorized'] is False
    assert CONFIG['max_training_transitions_per_job']==CONFIG['requested_frames']+499
    assert CONFIG['max_frozen_control_steps']==2*3*500
    assert len(JOBS)==7


def test_original_replay_sampling_shares_numpy_rng_with_later_reset_speed():
    from all.memory import ExperienceReplayBuffer
    b=ExperienceReplayBuffer(20)
    s=State(torch.zeros(1,1),torch.ones(1,dtype=torch.uint8))
    for _ in range(10): b.store(s,torch.zeros(1,1),0.,s)
    saved=np.random.get_state()
    try:
        np.random.seed(200); reference=[float(np.random.normal(15,5)) for _ in range(3)]
        np.random.seed(200); first=float(np.random.normal(15,5)); b.sample(8)
        sampled=[first,float(np.random.normal(15,5)),float(np.random.normal(15,5))]
        # NumPy's second cached normal draw is unchanged; later draws depend on
        # replay sampling. This is inherited behavior, not an audit side effect.
        assert sampled[:2]==reference[:2] and sampled[2]!=reference[2]
    finally: np.random.set_state(saved)


def test_reset_jump_does_not_count_as_recovery_acceleration():
    raw=SyntheticEnv('baseline');obs,info=raw.reset()
    from prediction_rl.evaluation.training_contract import state_measurement
    first={**state_measurement(obs,info),'episode':0,'event':'step'}
    first['ego']['speed']=0.
    second=deepcopy(first);second['ego']['speed']=15.;second['episode']=1;second['event']='reset'
    assert coverage([first,second])['low_speed_to_moving_transitions']==0
