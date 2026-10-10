"""IC02d worker engine. Original replay/reward/action core remains unchanged."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
from pathlib import Path

from prediction_rl.curriculum.task_characterization import BudgetLedger, root_evidence, scan_checkpoints
from prediction_rl.data.actor_adapter import build_actor_inputs
from prediction_rl.data.branching import ReplayBrancher, fingerprint, restore_rng, rng_fingerprint
from prediction_rl.data.collection_store import seal_episode, write_once
from prediction_rl.data.dataset_contract import build_branch_labels, digest
from prediction_rl.data.merge_geometry import MergeGeometry, read_extended_traffic
from prediction_rl.data.reference_policy import AuthorDDPGReference
from prediction_rl.data.response_collection import branch_with_geometry, root_accounting
from prediction_rl.envs.upstream import AuditedJerkEnv, upstream_session
from prediction_rl.envs.fresh_task import runtime_description
from prediction_rl.evaluation.terminal_events import StepObserver


def now():
    return datetime.now(timezone.utc).isoformat()


def model_hash(model):
    result = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        result.update(name.encode()); result.update(str((tensor.dtype, tuple(tensor.shape))).encode())
        result.update(tensor.detach().cpu().numpy().tobytes())
    return result.hexdigest()


class CountedEnv:
    def __init__(self, env, manager):
        self.base, self.manager = env, manager
    def __getattr__(self, name):
        return getattr(self.base, name)
    def step(self, action):
        self.manager.ledger.control(self.manager.phase)
        self.manager.session_controls[self.manager.phase] += 1
        value = self.base.step(action)
        self.manager.ledger.completed_controls += 1
        return value


class CountedObserver(StepObserver):
    def __init__(self, control, traci, job, ledger):
        super().__init__(control, traci, job); self.ledger = ledger
        self.cap = 101+128+25+1
    def step(self):
        self.ledger.tick()
        value = super().step()
        self.ledger.completed_raw_ticks += 1
        return value


class TaskBrancher(ReplayBrancher):
    """Only instrumentation/new checkpoint discovery; existing restoration unchanged."""
    def __init__(self, output, design, job):
        self.output, self.design, self.job = Path(output), design, job
        self.ledger = BudgetLedger(design); self.observer = None
        self.phase = None; self.session_controls = Counter(); self.session_rows = []
        super().__init__()

    def _fresh(self):
        # Same initialization sequence as ReplayBrancher._fresh, with pre-step caps.
        from config import Settings
        import control, traci
        if fingerprint(Settings.export_settings()) != self.settings_hash:
            raise RuntimeError('Configuration changed after manager initialization')
        self.close()
        self.ledger.session()
        restore_rng(self.initial_rng); control.delay = self.initial_delay
        self.session_controls = Counter()
        self.observer = CountedObserver(control, traci,
            {'id': self.job['id'], 'condition_id': 'gym20'}, self.ledger)
        self.observer.__enter__()
        self.env = CountedEnv(AuditedJerkEnv(), self)
        observation, _ = self.env.reset()
        if len(self.observer.events) != 101 or len(self.observer.creations) != 1:
            raise AssertionError('Fresh original reset must contain 100+1 raw ticks and one ego insertion')
        creation = self.observer.creations[0]
        if (creation['depart_position_m'] != 40 or creation['arrival_position_m'] != 50
                or creation['simulation_time_s'] != 20. or creation['route_id'] != 'rampRoute'):
            raise AssertionError('Original reset/insertion contract changed')
        return observation

    def discover(self, policy, geometry):
        self.phase = 'reference_discovery'
        observation = self._fresh(); geometry.verify_runtime(); policy.bind(self.env)
        return scan_checkpoints(self.env, self.design, geometry, policy, observation, read_extended_traffic)

    def capture_with_history(self, prefix, history_steps=11):
        self.phase = 'root_capture_prefix'
        return super().capture_with_history(prefix, history_steps)

    def restore(self, root):
        self.phase = 'branch_replay_prefix'
        value = super().restore(root)
        self.phase = 'candidate_future'
        return value

    def close(self):
        observer = self.observer
        try:
            super().close()
        finally:
            if observer is not None:
                try:
                    events = observer.export()
                finally:
                    observer.__exit__(None, None, None); self.observer = None
                number = len(self.session_rows)
                row = {'session_index': number, 'raw_ticks': len(events['steps']),
                    'control_steps_by_phase': dict(self.session_controls),
                    'control_attempts': sum(self.session_controls.values()),
                    'waiting_insertion_ticks': min(len(events['steps']), 101),
                    'native_terminal_events_recorded': True}
                self.session_rows.append(row)
                write_once(self.output/f's{number:03d}.json', {'accounting': row, 'raw_events': events})


def collect_task(repo, baseline, design, job, output, binding, metric_config, scales, postflight):
    repo, baseline, output = Path(repo), Path(baseline), Path(output)
    geometry = MergeGeometry(repo/'RL-MPC-LaneMerging-master/merge.net.xml')
    progress = []; manager = policy = None
    started = now(); error = None; reference_before = reference_after = None
    # Prime the same Torch/CUDA state reader before the saved session boundary.
    rng_before = rng_fingerprint()
    try:
        with upstream_session(repo/'RL-MPC-LaneMerging-master', repo/design['upstream_config'], job['simulator_seed']) as settings:
            write_once(output/'settings.json', settings.export_settings())
            low, high = settings.MINIMUM_NEGATIVE_JERK, settings.MAXIMUM_POSITIVE_JERK
            if ([low+f*(high-low) for f in (0,.25,.5,.75,1)] != design['probe_jerks']
                    or settings.TICK_LENGTH != .2 or settings.USE_CYTHON is not True):
                raise ValueError('Keep original controls, tick and enabled Cython')
            policy = AuthorDDPGReference(baseline/design['reference_checkpoint'], design['reference_sha256'],
                                         'cuda' if settings.CUDA else 'cpu')
            reference_before = model_hash(policy.model)
            manager = TaskBrancher(output, design, job)
            try:
                discovery = manager.discover(policy, geometry)
                description = runtime_description(repo/'RL-MPC-LaneMerging-master', settings, geometry)
                description.update(reference_checkpoint_sha256=design['reference_sha256'], reference_device=policy.device)
                write_once(output/'runtime.json', description)
                repeated_discovery = manager.discover(policy, geometry)
                write_once(output/'discovery.json', discovery)
                write_once(output/'discovery_repeat.json', repeated_discovery)
                if discovery != repeated_discovery:
                    raise AssertionError('Repeated reference discovery differs')
                progress = root_accounting(discovery['targets'])
                for number, planned in enumerate(discovery['targets']):
                    directory = output/f'r{number}'
                    if planned['prefix_steps'] is None:
                        write_once(directory/'accounting.json', progress[number]); continue
                    root, history = manager.capture_with_history(planned['prefix_actions'], 11)
                    geometry.verify_runtime()
                    if root.traffic != planned['selection']['traffic'] or history[-1] != planned['selection']['frame']:
                        raise AssertionError('Discovered checkpoint changed during replay')
                    repeated_root, repeated_history = manager.capture_with_history(root.prefix, 11)
                    if (root.signature != repeated_root.signature or root.prefix_trace_hash != repeated_root.prefix_trace_hash
                            or history != repeated_history):
                        raise AssertionError('Repeated root/history differs')
                    root_id = digest(['ic02d-task-engineering-v1', job['id'], planned['target']['id'], root.signature])
                    inputs = build_actor_inputs(history, geometry, 12, 11, .2)
                    branches, repeats, candidates = {}, {}, []
                    for cid in [*range(5), *reversed(range(5))]:
                        actions = [design['probe_jerks'][cid]]+[0.] * 24
                        trace, future = branch_with_geometry(manager, root, actions, geometry)
                        value = {'trace': trace, 'future_metadata': future}
                        repeated = cid in branches
                        write_once(directory/f'c{cid}{"_repeat" if repeated else ""}.json', value)
                        if repeated:
                            if value != branches[cid]:
                                raise AssertionError('Reverse-order repeated branch differs')
                            repeats[cid] = value
                        else:
                            branches[cid] = value
                            candidates.append({'candidate_id': cid, 'labels': build_branch_labels(root.traffic, trace, actions, .2)})
                    h = {'root_id': root_id, 'episode_id': job['source_episode_id'], 'split': 'train',
                        'history': history, 'inputs': inputs}
                    pack = {'root_id': root_id, 'episode_id': job['source_episode_id'], 'split': 'train',
                        'root_traffic': root.traffic, 'prefix_actions': list(root.prefix), 'candidates': candidates}
                    evidence = root_evidence(h, pack, branches, repeats, geometry, metric_config, scales,
                        {'task_id': job['id'], 'target': planned['target']['id'], 'simulator_seed': job['simulator_seed']})
                    write_once(directory/'history.json', h); write_once(directory/'labels.json', pack)
                    write_once(directory/'evidence.json', evidence)
                    progress[number].update(status='complete', prefix_steps=len(root.prefix), root_id=root_id,
                        candidate_count=5, branch_executions=10, repeated_exact=True,
                        history_frames=len(history), normalized_I=evidence['I_with_frozen_IC01_scales'],
                        full_common_horizon=evidence['interaction']['full_common_horizon'],
                        support_qualified=evidence['support_qualified'], task_level=None)
                    write_once(directory/'accounting.json', progress[number])
                    print(f"[ic02d] task={job['id']} root={planned['target']['id']} repeats=exact", flush=True)
            finally:
                manager.close()
                reference_after = model_hash(policy.model)
    except BaseException as caught:
        error = caught
    finally:
        # Capture BEFORE aggregate comparison, on success and failure, without new ticks.
        if policy is not None:
            reference_after = model_hash(policy.model)
        post = {'worker_session_rng_restored': rng_fingerprint() == rng_before,
            'reference_tensor_before': reference_before, 'reference_tensor_after': reference_after,
            'reference_tensors_unchanged': reference_before is not None and reference_before == reference_after,
            'predictor_loaded': False, 'learner_instantiated': False,
            'budget': manager.ledger.export() if manager is not None else None,
            'session_rows': manager.session_rows if manager is not None else [], 'finished_at': now()}
        try:
            postflight(); post['immutable_input_postflight_passed'] = True
        except BaseException as caught:
            post['immutable_input_postflight_passed'] = False; post['input_failure'] = str(caught)
            if error is None: error = caught
        write_once(output/'postflight.json', post)
    if error is None and not all(post[k] for k in ('worker_session_rng_restored', 'reference_tensors_unchanged', 'immutable_input_postflight_passed')):
        error = AssertionError('Reference tensors/RNG/files changed during collection')
    if error is not None:
        write_once(output/'failure.json', {'status': 'failed', 'binding': binding, 'roots': progress,
            'failure_reason': f'{type(error).__name__}: {error}', 'postflight_recorded': True, 'finished_at': now()})
        raise error
    return seal_episode(output, binding, {'job': job, 'status': 'complete', 'roots': progress,
        'budget': post['budget'], 'postflight_passed': True, 'training_ready': False,
        'started_at': started, 'finished_at': now()})
