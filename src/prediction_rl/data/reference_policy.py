"""Pinned author DDPG inference for development root discovery only."""
import hashlib
from pathlib import Path

import numpy as np

from prediction_rl.data.branching import rng_fingerprint
from prediction_rl.envs.upstream import validate_action


def verify_checkpoint(path, expected_sha256):
    path = Path(path)
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected_sha256:
        raise ValueError('Reference policy checkpoint SHA256 mismatch; refuse pickle load')
    return path


class AuthorDDPGReference:
    """Use the author's GreedyAgent + TimeFeature without creating another SUMO.

    The hash-pinned full-module pickle was supplied with the audited upstream.
    No optimizer, exploration noise, q network, ST takeover or model updates.
    Two independent TimeFeature wrappers check observation/control-path parity.
    """
    def __init__(self, checkpoint, expected_sha256, device):
        import torch
        import ddpg  # Complete upstream imports BEFORE ReplayBrancher freezes RNG.
        from all.policies.deterministic import DeterministicPolicyNetwork
        self.upstream_agent_class = ddpg.DDPGAgent
        self.device = device
        path = verify_checkpoint(checkpoint, expected_sha256)
        self.model = torch.load(path, map_location=device, weights_only=False).to(device)
        if type(self.model) is not DeterministicPolicyNetwork:
            raise TypeError('Expected the author deterministic DDPG policy network')
        if any(isinstance(m, (torch.nn.modules.dropout._DropoutNd,
                              torch.nn.modules.batchnorm._BatchNorm)) for m in self.model.modules()):
            raise TypeError('Stateful/stochastic policy layers require a separate inference audit')
        self.env = None

    def bind(self, env):
        from all.bodies.time import TimeFeature
        from all.environments import GymEnvironment
        from all.experiments.watch import GreedyAgent
        self.env = env
        self.codec = GymEnvironment(env.env, device=self.device)
        self.codec._lazy_init()  # State masks only; no reset/step/start/close.
        self.agent = TimeFeature(GreedyAgent(env.action_space, policy=self.model))
        # Bypass constructor solely to avoid its second SUMO connection. Execute
        # the unchanged upstream get_control on raw.previous_state each step.
        self.oracle = object.__new__(self.upstream_agent_class)
        self.oracle.env = self.codec
        self.oracle.agent = TimeFeature(GreedyAgent(env.action_space, policy=self.model))
        self.checked_steps = 0

    def __call__(self, observation, step):
        if self.env is None or step != self.checked_steps:
            raise RuntimeError('Reference policy requires sequential first-episode calls')
        before_rng = rng_fingerprint()
        state = self.codec._make_state(observation, False)
        action = self.agent.eval(state, 0).detach().cpu().numpy().reshape(-1)
        original = self.oracle.get_control(self.env.raw.previous_state)
        if action.shape != (1,) or float(action[0]) != original:
            raise AssertionError('Audited observation policy differs from upstream get_control')
        if rng_fingerprint() != before_rng:
            raise AssertionError('Deterministic reference inference consumed RNG')
        action = validate_action(action, self.env.action_space)
        self.checked_steps += 1
        return float(action[0])
