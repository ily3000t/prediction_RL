"""IC02b prototype: one native Gym20 first episode in an isolated worker.

This does not integrate a learner or implement curriculum sampling. Creating a
new task never calls upstream_session in the parent. Transport failures are
errors, not synthetic terminal transitions or wall-clock observation fallback.
"""
from copy import deepcopy
from contextlib import redirect_stderr, redirect_stdout
import multiprocessing as mp
from pathlib import Path
import traceback

import numpy as np

from prediction_rl.envs.upstream import observation_at_policy_dtype, validate_action, validate_seed

ENTRY_CONTRACT = 'seeded_fresh_gym20_task_v1'
EXECUTION_CONTRACT = 'simulation_blocking_exact_v1'


def observer_delta(observer, cursors):
    counts = (len(observer.events), len(observer.creations), len(observer.removals))
    result = {'steps': deepcopy(observer.events[cursors[0]:]),
              'creations': deepcopy(observer.creations[cursors[1]:]),
              'explicit_removals': deepcopy(observer.removals[cursors[2]:])}
    return result, counts


def native_packet(base, observer, cursors, phase, control_step, action=None):
    """Delegate exactly once, then read the CURRENT full frame and raw events."""
    from prediction_rl.data.merge_geometry import read_extended_traffic
    if phase == 'reset':
        observation, info = base.reset()
        reward, done = None, False
    elif phase == 'step':
        observation, reward, done, info = base.step(action)
    else:
        raise ValueError('Unknown native packet phase')
    events, counts = observer_delta(observer, cursors)
    return {'phase': phase, 'observation': observation.copy(), 'reward': reward,
            'done': bool(done), 'info': info, 'current_frame': read_extended_traffic(),
            'control_step': control_step, 'raw_step_count': counts[0],
            'raw_events': events}, counts


def space_description(base):
    return {name: {'low': space.low.copy(), 'high': space.high.copy(),
                   'dtype': str(space.dtype), 'shape': list(space.shape)}
            for name, space in (('observation', base.observation_space), ('action', base.action_space))}


def runtime_description(source, settings, geometry):
    import os
    import sys
    names = ('config', 'control', 'merge_gym', 'sumo', 'prediction', 'dqn', 'st', 'st_cy')
    modules = {name: str(Path(sys.modules[name].__file__).resolve()) for name in names}
    if any(Path(value).parent != Path(source).resolve() for value in modules.values()):
        raise ValueError('Worker imported source/extension outside the isolated checkout')
    if settings.USE_CYTHON is not True or settings.TICK_LENGTH != .2:
        raise ValueError('Do not disable Cython or change the native tick')
    geometry.verify_runtime()
    return {'pid': os.getpid(), 'module_paths': modules, 'settings': settings.export_settings(),
            'geometry_lengths': geometry.lengths, 'execution_contract': EXECUTION_CONTRACT}


def _worker(connection, source, config, seed, log_path, control_cap):
    """Spawn target, never imported/executed as an upstream session in parent."""
    base = None
    try:
        with Path(log_path).open('x', encoding='utf-8') as log, redirect_stdout(log), redirect_stderr(log):
            from prediction_rl.data.merge_geometry import MergeGeometry
            from prediction_rl.envs.upstream import AuditedJerkEnv, upstream_session
            from prediction_rl.evaluation.terminal_events import StepObserver
            with upstream_session(source, config, seed) as settings:
                import control
                import traci
                with StepObserver(control, traci, {'id': f'fresh_{seed}', 'condition_id': 'gym20'}) as observer:
                    # One reset, capped controls, and at most one native deadline cleanup tick.
                    observer.cap = 101 + control_cap + 1
                    base = AuditedJerkEnv()
                    geometry = MergeGeometry(Path(source)/'merge.net.xml')
                    description = runtime_description(source, settings, geometry)
                    connection.send({'ok': True, 'sequence': 0, 'value': {
                        'spaces': space_description(base), 'runtime': description}})
                    ready, reset_used, steps, sequence = False, False, 0, 0
                    cursors = (0, 0, 0)
                    while True:
                        request = connection.recv()
                        sequence += 1
                        if (not isinstance(request, dict) or set(request) != {'sequence', 'op', 'action'}
                                or request['sequence'] != sequence):
                            raise ValueError('Malformed/out-of-order worker request')
                        op = request['op']
                        if op == 'close':
                            if request['action'] is not None:
                                raise ValueError('Close cannot carry an action')
                            base.close()
                            connection.send({'ok': True, 'sequence': sequence, 'value': {'closed': True}})
                            break
                        if op == 'reset':
                            if ready or reset_used or request['action'] is not None:
                                raise RuntimeError('Fresh task permits exactly one native reset')
                            packet, cursors = native_packet(base, observer, cursors, 'reset', 0)
                            reset_used = True
                        elif op == 'step':
                            if not ready or steps >= control_cap:
                                raise RuntimeError('No step before reset/after terminal/engineering cap')
                            action = validate_action(request['action'], base.action_space)
                            steps += 1
                            packet, cursors = native_packet(base, observer, cursors, 'step', steps, action)
                        else:
                            raise ValueError('Unknown worker operation')
                        geometry.validate_frame(packet['current_frame'], require_ego=not packet['done'])
                        ready = not packet['done']
                        connection.send({'ok': True, 'sequence': sequence, 'value': packet})
    except BaseException as error:
        try:
            connection.send({'ok': False, 'sequence': locals().get('sequence', 0),
                             'error_type': type(error).__name__, 'error': str(error),
                             'traceback': traceback.format_exc()})
        except (OSError, EOFError):
            pass
    finally:
        if base is not None:
            base.close()
        connection.close()


class FreshGym20TaskEnv:
    """Single-task prototype with native four-result stepping.

    The owner may construct the next task ONLY after the previous one finishes
    and closes. The learner is not owned or reseeded here. The engineering cap
    refuses further calls, but never changes a returned done or observation.
    """
    execution_contract = EXECUTION_CONTRACT
    reset_contract = ENTRY_CONTRACT

    def __init__(self, source, config, seed, log_path, *, control_cap=500, transport_timeout_s=120.):
        validate_seed(seed)
        if type(control_cap) is not int or not 1 <= control_cap <= 500:
            raise ValueError('Invalid native control cap')
        if (isinstance(transport_timeout_s, bool) or not isinstance(transport_timeout_s, (int, float))
                or not np.isfinite(transport_timeout_s) or transport_timeout_s <= 0):
            raise ValueError('Invalid transport liveness watchdog')
        self.source, self.config = Path(source).resolve(), Path(config).resolve()
        if (not (self.source/'merge_gym.py').is_file() or not self.config.is_relative_to(self.source)
                or not self.config.is_file() or Path(log_path).exists()):
            raise ValueError('Need isolated source, its config, and a new worker log')
        self.seed, self.control_cap, self.timeout = seed, control_cap, float(transport_timeout_s)
        self._sequence = self._steps = 0
        self._ready = self._reset_used = self._faulted = self._closed = False
        self.last_packet = None
        context = mp.get_context('spawn')
        self._connection, child = context.Pipe()
        self._process = context.Process(target=_worker, args=(child, str(self.source), str(self.config),
            seed, str(Path(log_path).resolve()), control_cap), name=f'gym20-task-{seed}')
        self._process.start()
        child.close()
        try:
            description = self._receive(0)
            from gym.spaces import Box
            for name in ('observation', 'action'):
                values = description['spaces'][name]
                if values['dtype'] != 'float32':
                    raise ValueError('Original float32 space changed')
                space = Box(values['low'], values['high'], dtype=np.float32)
                if list(space.shape) != values['shape']:
                    raise ValueError('Invalid transported space')
                setattr(self, name+'_space', space)
            if self.observation_space.shape != (20,) or self.action_space.shape != (1,):
                raise ValueError('Original policy observation/action shape changed')
            if not np.array_equal(self.action_space.low, [-5]) or not np.array_equal(self.action_space.high, [5]):
                raise ValueError('Original continuous jerk bounds changed')
            self.runtime = description['runtime']
            if self.runtime['pid'] != self._process.pid:
                raise ValueError('Worker identity mismatch')
        except BaseException:
            self._faulted = True
            self.close()
            raise

    def _receive(self, sequence):
        # Liveness only: an expired watchdog invalidates this task. It never
        # returns stale features, zero reward or a synthetic terminal sample.
        if not self._connection.poll(self.timeout):
            raise TimeoutError('Environment transport stalled; task is invalid')
        reply = self._connection.recv()
        if not isinstance(reply, dict) or reply.get('sequence') != sequence:
            raise RuntimeError('Invalid worker reply sequence')
        if reply.get('ok') is not True:
            raise RuntimeError('Environment worker failed: '+str(reply.get('error_type'))+': '
                               +str(reply.get('error'))+'\n'+str(reply.get('traceback')))
        if set(reply) != {'ok', 'sequence', 'value'}:
            raise RuntimeError('Unknown worker reply fields')
        return reply['value']

    def _exchange(self, op, action=None):
        try:
            self._sequence += 1
            self._connection.send({'sequence': self._sequence, 'op': op, 'action': action})
            return self._receive(self._sequence)
        except BaseException:
            self._faulted = True
            raise

    def _accept(self, packet, phase, step):
        try:
            if (set(packet) != {'phase', 'observation', 'reward', 'done', 'info', 'current_frame',
                               'control_step', 'raw_step_count', 'raw_events'}
                    or packet['phase'] != phase or packet['control_step'] != step or type(packet['done']) is not bool):
                raise RuntimeError('Native packet boundary mismatch')
            observation = observation_at_policy_dtype(packet['observation'], self.observation_space)
            if phase == 'reset' and (packet['done'] or packet['reward'] is not None):
                raise RuntimeError('Reset cannot contain a transition or terminal')
            if phase == 'step' and (isinstance(packet['reward'], (bool, np.bool_))
                                   or not np.isfinite(packet['reward'])):
                raise ValueError('Invalid native reward')
            self.last_packet = deepcopy(packet)
            self._ready = not packet['done']
            return observation, deepcopy(packet['info'])
        except BaseException:
            self._faulted = True
            raise

    def reset(self):
        if self._closed or self._faulted or self._ready or self._reset_used:
            raise RuntimeError('Fresh task reset is single-use and must be healthy')
        packet = self._exchange('reset')
        self._reset_used = True
        return self._accept(packet, 'reset', 0)

    def step(self, action):
        if self._closed or self._faulted or not self._ready or self._steps >= self.control_cap:
            raise RuntimeError('No step before reset/after terminal/engineering cap')
        try:
            action = validate_action(action, self.action_space)
            packet = self._exchange('step', action)
            self._steps += 1
            observation, info = self._accept(packet, 'step', self._steps)
            return observation, packet['reward'], packet['done'], info
        except BaseException:
            self._faulted = True
            raise

    def current_frame(self):
        if self._closed or self._faulted or self.last_packet is None:
            raise RuntimeError('Current frame requires a healthy current native packet')
        return deepcopy(self.last_packet['current_frame'])

    def close(self):
        if self._closed:
            return
        try:
            if self._process.is_alive() and not self._faulted:
                self._exchange('close')
        finally:
            self._closed, self._ready = True, False
            self._connection.close()
            self._process.join(timeout=5.)
            if self._process.is_alive():
                # Only the owned worker; a failure remains visible, never retried.
                self._process.terminate()
                self._process.join(timeout=5.)
                self._faulted = True
                raise RuntimeError('Owned environment worker did not close cleanly')
            if self._process.exitcode != 0 and not self._faulted:
                self._faulted = True
                raise RuntimeError('Owned environment worker exited abnormally')
