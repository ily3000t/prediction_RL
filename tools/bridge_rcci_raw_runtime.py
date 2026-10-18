"""Bounded cross-build Raw runtime bridge, not training or method evaluation."""
import argparse
from contextlib import contextmanager
from copy import deepcopy
import importlib.util
from pathlib import Path
import re
import subprocess
import sys
import traceback
from types import FunctionType, SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'tools')]
import numpy as np
import torch

import audit_rcci_feasibility as shared
import audit_rcci_raw_reuse as audit
import explore_feature_ddpg as original
import smoke_feature_ddpg as smoke
import smoke_rcci_native as native
import train_matched_ddpg as kernel
import train_response_predictors as training
from prediction_rl.data.branching import fingerprint
from prediction_rl.data.collection_store import read_json as read, write_once, run_lock
from prediction_rl.data.dataset_contract import digest
from prediction_rl.data.frozen_dataset import inside
from prediction_rl.features.probe_features import FrozenProbeFeatures
from prediction_rl.prediction.interface import PredictorConfig
from prediction_rl.prediction.kinematic_reference import LaneConstantVelocity
from prediction_rl.prediction.trained_artifacts import load_trained_ensemble

CONFIG = 'configs/development/rcci04b_raw_runtime_bridge_v1.json'
SETTINGS = {'version': 'rcci04b_raw_runtime_bridge_v1',
    'design': 'docs/design/rcci04b_raw_runtime_bridge_v1.md',
    'raw_audit_request': 'artifacts/rcci04a_raw/raw_reuse_v1/request.json',
    'raw_audit_report_sha256': '0f85dbea6694b455559e228b5e9e2168e8851c7731899b38aa3bef35b85f5afb',
    'raw_audit_inventory_sha256': '84febff249293ada0ccac1efae4c61d31d26df979388a519d00dd874b50e57d3',
    'arms': ['ordinary', 'conditional'], 'run_seeds': [0, 1, 2], 'checkpoint_frames': 120000,
    'simulator_seed': 200, 'binary_variants': ['historical', 'current'],
    'max_sessions': 12, 'max_episodes': 12, 'max_controlled_steps': 6000,
    'max_simulation_steps': 7224, 'max_ensemble_forward_calls': 6000,
    'max_member_forward_calls': 18000, 'max_policy_forward_calls': 6012,
    'worker_watchdog_s': 180, 'device': 'cpu', 'torch_threads': 1,
    'execution_contract': 'simulation_blocking_exact_v1', 'warmup_s': 20,
    'training_permitted': False, 'formal_evaluation': False, 'test_opened': False}
require, sha = audit.require, audit.sha


def jobs():
    result = []
    for arm in SETTINGS['arms']:
        for seed in SETTINGS['run_seeds']:
            model = f'{arm}_s{seed}'
            legacy = {'arm': arm, 'run_seed': seed, 'training_id': model,
                'checkpoint_frames': 120000, 'simulator_seed': 200, 'id': model+'_n120000_v200'}
            result.extend({'id': model+'_'+variant, 'model': model, 'binary_variant': variant,
                           'legacy_job': legacy} for variant in SETTINGS['binary_variants'])
    return result


class NoSTCalls:
    """Worker-local diagnostic guard; no installed package/source is patched."""
    def __init__(self, *modules):
        self.modules = modules; self.originals = []; self.calls = []; self.names = []

    def __enter__(self):
        require(not self.originals and all(not getattr(m, '_rcci04b_guard', False) for m in self.modules),
                'Nested ST guard is forbidden')
        entries = []
        for module in self.modules:
            names = [(name, value) for name, value in vars(module).items()
                     if not name.startswith('__') and callable(value) and getattr(value, '__module__', None) == module.__name__]
            require(bool(names), 'No auditable local ST entrypoints')
            entries.extend((module, name, value) for name, value in names)
        for module in self.modules: module._rcci04b_guard = True
        for module, name, value in entries:
            label = module.__name__+'.'+name
            def forbidden(*args, _label=label, **kwargs):
                self.calls.append(_label)
                raise RuntimeError('Unexpected ST planning in pure Raw DDPG bridge: '+_label)
            self.names.append(label); self.originals.append((module, name, value))
            setattr(module, name, forbidden)
        return self

    def __exit__(self, *error):
        for module, name, value in self.originals: setattr(module, name, value)
        for module in self.modules: del module._rcci04b_guard
        self.originals.clear()


def load_extension(path, expected_sha256):
    path = Path(path).resolve()
    require(sha(path) == expected_sha256 and 'st_cy' not in sys.modules,
            'Native hash changed or extension was already imported')
    spec = importlib.util.spec_from_file_location('st_cy', path)
    require(spec is not None and spec.loader is not None, 'Cannot resolve selected native extension')
    module = importlib.util.module_from_spec(spec)
    sys.modules['st_cy'] = module
    spec.loader.exec_module(module)
    require(Path(module.__file__).resolve() == path and sha(module.__file__) == expected_sha256,
            'Wrong native extension loaded; no fallback')
    return module


def bind_evaluation(namespace, session, factory):
    """Same frozen evaluator bytecode, explicit scoped dependencies only."""
    fn = FunctionType(kernel.evaluation_job.__code__, {**kernel.evaluation_job.__globals__,
        'original': namespace, 'upstream_session': session, 'preset_factory': factory},
        kernel.evaluation_job.__name__, kernel.evaluation_job.__defaults__, kernel.evaluation_job.__closure__)
    fn.__kwdefaults__ = kernel.evaluation_job.__kwdefaults__
    return fn


def parent_inputs(baseline):
    baseline = Path(baseline).resolve()
    require(baseline == Path('E:/Prediction_RL').resolve() and
        shared.git(baseline, 'rev-parse', 'HEAD') == shared.BASE and not shared.git(baseline, 'status', '--porcelain'),
        'Require clean unchanged read-only baseline')
    path = inside(ROOT, SETTINGS['raw_audit_request']); parent = path.parent
    aq, ar, receipt = (read(p) for p in (path, parent/'report.json', parent/'receipt.json'))
    current = shared.sources()
    require(aq['config'] == audit.SETTINGS and Path(aq['baseline_root']).resolve() == baseline
        and sha(parent/'report.json') == SETTINGS['raw_audit_report_sha256']
        and ar['request_hash'] == digest(aq) and ar['status'] == 'complete' and ar['historical_evidence_complete']
        and receipt == {'request_hash': digest(aq), 'report_sha256': SETTINGS['raw_audit_report_sha256']}
        and ar['verified_inventory_sha256'] == SETTINGS['raw_audit_inventory_sha256']
        and sha(parent/'verified_inventory.json') == SETTINGS['raw_audit_inventory_sha256']
        and all(current.get(k) == h for k, h in aq['source_hashes'].items())
        and aq['environment'] == shared.environment() and not (parent/'writer.lock').exists(),
        'Completed Raw audit changed or active')
    legacy_path = inside(baseline, audit.SETTINGS['request']); root = legacy_path.parent; legacy = read(legacy_path)
    require(digest(legacy) == audit.SETTINGS['request_hash']
        and sha(root/'aggregate.json') == audit.SETTINGS['aggregate_sha256']
        and legacy['environment'] == training.env() and legacy['dependencies'] == smoke.dependencies()
        and original.runtime_identity() == ar['current_runtime'] and not (root/'writer.lock').exists(),
        'Historical request/runtime changed')
    for mapping in (legacy['source_hashes'], legacy['upstream_hashes']):
        require(all(fingerprint(inside(ROOT, n).read_text(encoding='utf-8')) == h for n, h in mapping.items()),
                'Original execution source changed')
    inventory = read(parent/'verified_inventory.json')
    assets = {'baseline': {}, 'worktree': {}}
    def record(where, p, expected=None):
        base = baseline if where == 'baseline' else ROOT
        p = Path(p); h = sha(p)
        if expected is not None: require(h == expected, 'Frozen bridge asset changed: '+str(p))
        assets[where][p.relative_to(base).as_posix()] = h
    for p in (path, parent/'report.json', parent/'receipt.json', parent/'verified_inventory.json'):
        record('worktree', p)
    for name, h in aq['parent_files'].items(): record('baseline', root/name, h)
    for mode in SETTINGS['arms']:
        entry = legacy['predictors'][mode]
        record('baseline', inside(baseline, entry['path']), entry['sha256'])
    config = inside(baseline, 'RL-MPC-LaneMerging-master/configs/train_default_1.json')
    record('baseline', config, legacy['upstream_config_sha256'])
    for model in sorted({j['model'] for j in jobs()}):
        d = root/'train'/model; selected = ar['models'][model]
        require(selected['weights_sha256'] == selected['snapshots'][-1]['weights_sha256'], 'Final model selection differs')
        recorded = inventory['train/'+model]
        record('baseline', d/'complete.json', recorded['receipt_sha256'])
        for name in ('report.json', 'n120000.json', 'n120000.pt'):
            record('baseline', d/name, recorded['files'][name])
        job = next(j['legacy_job'] for j in jobs() if j['model'] == model)
        require(job in legacy['evaluation_jobs'], 'Bridge job absent from frozen protocol')
        ed = root/'evaluate'/job['id']; recorded = inventory['evaluate/'+job['id']]
        record('baseline', ed/'complete.json', recorded['receipt_sha256'])
        for name, h in recorded['files'].items(): record('baseline', ed/name, h)
    local = native.native_identity(native.SETTINGS)
    for name, h in local['files'].items(): record('worktree', inside(ROOT, name), h)
    for name, h in native.upstream_files(baseline).items(): record('worktree', inside(ROOT, name), h)
    require(len(ar['original_runtime']['upstream_compiled_modules']) == 1, 'Unknown historical native roster')
    old_name, old_sha = next(iter(ar['original_runtime']['upstream_compiled_modules'].items()))
    record('baseline', inside(baseline, old_name), old_sha)
    variants = {'historical': {'path': str(inside(baseline, old_name)), 'sha256': old_sha},
        'current': {'path': str(inside(ROOT, local['extension'])), 'sha256': local['extension_sha256']}}
    return legacy, assets, variants


def prepare(baseline, run_id):
    shared.clean(); torch.set_num_threads(1)
    require(bool(re.fullmatch('[A-Za-z0-9_-]{1,24}', run_id)), 'Use fresh short run ID')
    require(read(ROOT/CONFIG) == SETTINGS, 'Only frozen bridge config is allowed')
    legacy, assets, variants = parent_inputs(baseline)
    q = {**shared.metadata(), 'config': SETTINGS, 'config_sha256': sha(ROOT/CONFIG),
        'design_sha256': sha(ROOT/SETTINGS['design']), 'source_hashes': shared.sources(),
        'environment': shared.environment(), 'baseline_root': str(Path(baseline).resolve()),
        'assets': assets, 'binary_variants': variants, 'jobs': jobs(), 'legacy_request': legacy,
        'legacy_upstream_config': 'RL-MPC-LaneMerging-master/configs/train_default_1.json'}
    out = ROOT/'artifacts/rcci04b'/run_id; out.mkdir(parents=True, exist_ok=False)
    write_once(out/'request.json', q)
    write_once(out/'preparation.json', {'request_hash': digest(q), 'status': 'prepared_not_executed',
        'sessions': 12, 'training_permitted': False, 'formal_evaluation': False})
    print('bridge_request='+str(out/'request.json')); print('confirm_request_hash='+digest(q))
    return out/'request.json', digest(q)


def check_request(path, expected):
    shared.clean(); torch.set_num_threads(1); path = inside(ROOT, path); q = read(path)
    require(path.parent.parent == ROOT/'artifacts/rcci04b' and digest(q) == expected
        and q['config'] == SETTINGS and q['jobs'] == jobs() and read(ROOT/CONFIG) == SETTINGS
        and q['source_hashes'] == shared.sources() and q['environment'] == shared.environment()
        and q['git_commit'] == shared.git(ROOT, 'rev-parse', 'HEAD')
        and q['config_sha256'] == sha(ROOT/CONFIG) and q['design_sha256'] == sha(ROOT/SETTINGS['design'])
        and read(path.parent/'preparation.json')['request_hash'] == expected, 'Bridge request/source changed')
    baseline = Path(q['baseline_root'])
    require(shared.git(baseline, 'rev-parse', 'HEAD') == shared.BASE and not shared.git(baseline, 'status', '--porcelain'),
            'Read-only baseline changed')
    for where, base in (('baseline', baseline), ('worktree', ROOT)):
        for name, h in q['assets'][where].items(): require(sha(inside(base, name)) == h, 'Bridge asset changed: '+name)
    return path.parent, q


def worker(path, expected, job_id):
    out, q = check_request(path, expected)
    job = next((j for j in q['jobs'] if j['id'] == job_id), None); require(job is not None, 'Unknown bridge worker')
    directory = out/job_id; directory.mkdir(exist_ok=False)
    report = {**shared.metadata(), 'request_hash': expected, 'job': job, 'status': 'running',
        'training_performed': False, 'method_effect_established': False}
    write_once(directory/'started.json', report)
    providers, hooks, guards, observations, policy_inputs = [], [], [], [], []
    try:
        binary = q['binary_variants'][job['binary_variant']]
        extension = load_extension(binary['path'], binary['sha256'])
        baseline = Path(q['baseline_root']); legacy = q['legacy_request']
        source = ROOT/'RL-MPC-LaneMerging-master'; config = inside(baseline, q['legacy_upstream_config'])
        require(sha(config) == legacy['upstream_config_sha256'], 'Actual config bytes differ from checkpoint')
        def provider_for(lq, arm):
            require(lq == legacy and arm == job['legacy_job']['arm'], 'Unexpected asset resolution')
            entry = lq['predictors'][arm]; training_request = lq['predictor_training_request']
            model = load_trained_ensemble(inside(baseline, entry['path']), training_request, arm, entry['sha256'], entry['selected'])
            recorded = native.ForwardRecorder(model)
            p = FrozenProbeFeatures(LaneConstantVelocity(source/'merge.net.xml'), PredictorConfig.from_dict(training_request['model']), recorded)
            providers.append(p); return p, entry['sha256']
        def environment(lq, lj, provider, recorder):
            def record(event, obs, reward, done, info):
                recorder(event, obs, reward, done, info)
                require(obs.shape == (169,) and obs.dtype == np.float32, 'Wrong Raw observation layout')
                if done: require(not obs[20:].any(), 'Terminal predictor block not zero')
                meta = {k: v for k, v in info['prediction_features'].items() if k != 'latency_s'}
                observations.append({'event': event, 'observation': obs.tolist(),
                    'reward': None if reward is None else float(reward), 'done': bool(done),
                    'traffic': deepcopy(info.get('traffic')), 'execution_audit': deepcopy(info.get('execution_audit')),
                    'prediction_features': meta, 'rng': native.rng_hash()})
            return original.environment(lq, lj, provider, record)
        @contextmanager
        def session(*args):
            require(args == (source, config, 200), 'Unexpected session path/seed')
            with kernel.upstream_session(*args) as settings:
                import st, st_cy
                require(st_cy is extension and settings.USE_CYTHON and settings.TICK_LENGTH == .2, 'Native path/step changed')
                with NoSTCalls(st, st_cy) as guard:
                    guards.append(guard); yield settings
        def factory(*args, **kwargs):
            build = kernel.preset_factory(*args, **kwargs)
            def create(env, writer):
                checked = build(env, writer)
                def observe(module, inputs):
                    state, = inputs
                    require(state.features.shape == (1, 170) and state.features.dtype == torch.float32
                        and torch.isfinite(state.features).all(), 'Wrong actual policy/TimeFeature input')
                    policy_inputs.append({'features': state.features[0].detach().cpu().tolist(),
                                          'mask': state.mask.tolist(), 'rng': native.rng_hash()})
                hooks.append(checked.agent.policy.model.register_forward_pre_hook(observe))
                return checked
            return create
        namespace = SimpleNamespace(SOURCE=source, UPSTREAM=config, provider_for=provider_for, environment=environment)
        evaluate = bind_evaluation(namespace, session, factory)
        require(evaluate.__code__ is kernel.evaluation_job.__code__, 'Frozen evaluator bytecode changed')
        result = evaluate(legacy, job['legacy_job'], directory, inside(baseline, audit.SETTINGS['request']).parent)
        historical = inside(baseline, audit.SETTINGS['request']).parent/'evaluate'/job['legacy_job']['id']
        comparisons = {name: read(directory/name) == read(historical/name) for name in
                      ('episode.json', 'actions.json', 'states.json', 'raw_events.json', 'resolved_config.json')}
        require(all(comparisons.values()), 'Bridge differs from sealed historical trajectory/config/outcome')
        steps = result['outcome']['steps']
        require(len(providers) == len(guards) == 1 and not guards[0].calls and len(observations) == len(policy_inputs) == steps+1,
                'Unexpected forwards/records/ST call count')
        calls = providers[0].model.calls
        require(calls == steps <= 500 and result['no_policy_updates'], 'Predictor/episode cap or freeze failed')
        for i, (obs, policy) in enumerate(zip(observations, policy_inputs)):
            require(policy['features'][:169] == obs['observation'] and policy['rng'] == obs['rng']
                and policy['features'][-1] == float(torch.tensor(i, dtype=torch.float32)*.001)
                and policy['mask'] == [int(i < steps)], 'Policy input/time/mask/RNG alignment changed')
        write_once(directory/'full_observations.json', observations)
        write_once(directory/'policy_inputs.json', policy_inputs)
        report.update(status='complete', exact_historical_parity=comparisons, no_updates_verified=True,
            actual_controlled_steps=steps, actual_simulation_steps=len(read(directory/'raw_events.json')['steps']),
            episodes=1, ensemble_forward_calls=calls, member_forward_calls=calls*3, policy_forward_calls=len(policy_inputs),
            selected_extension_path=str(Path(extension.__file__).resolve()), selected_extension_sha256=sha(extension.__file__),
            st_guard_entrypoints=guards[0].names, st_calls=guards[0].calls,
            files={p.name: sha(p) for p in directory.iterdir() if p.is_file() and p.name != 'started.json'})
    except BaseException:
        report.update(status='failed', failure_reason=traceback.format_exc()); raise
    finally:
        for hook in hooks: hook.remove()
        report['finished_at'] = shared.now(); write_once(directory/'report.json', report)


def run(path, expected):
    out, q = check_request(path, expected)
    require(not any((out/n).exists() for n in ('started.json', 'report.json')), 'Never retry/overwrite a started bridge')
    with run_lock(out):
        report = {**shared.metadata(), 'request_hash': expected, 'status': 'running', 'training_performed': False,
                  'formal_evaluation': False, 'method_effect_established': False, 'training_released': False}
        write_once(out/'started.json', report)
        try:
            results, files, checks = {}, {}, {}
            for job in q['jobs']:
                print('[rcci04b] '+job['id'], flush=True)
                command = [sys.executable, '-B', str(Path(__file__).resolve()), 'worker', '--request', str(Path(path).resolve()),
                    '--confirm-request-hash', expected, '--job', job['id']]
                with (out/(job['id']+'.log')).open('xb') as log:
                    subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=180)
                d = out/job['id']; child = read(d/'report.json')
                require(child['status'] == 'complete' and child['request_hash'] == expected and child['job'] == job
                    and not child['st_calls'] and all(child['exact_historical_parity'].values()), 'Invalid bridge worker')
                for name, h in child['files'].items(): require(sha(d/name) == h, 'Completed bridge child changed')
                results[job['id']] = child
                for p in d.iterdir():
                    if p.is_file(): files[p.relative_to(out).as_posix()] = sha(p)
                files[job['id']+'.log'] = sha(out/(job['id']+'.log'))
                if job['binary_variant'] == 'current':
                    other = out/(job['model']+'_historical')
                    checks[job['model']] = {name: read(d/name) == read(other/name) for name in
                        ('episode.json', 'actions.json', 'states.json', 'raw_events.json', 'resolved_config.json',
                         'full_observations.json', 'policy_inputs.json')}
                    require(all(checks[job['model']].values()), 'Cross-build exact trajectory/input/RNG parity failed')
            totals = {k: sum(r[k] for r in results.values()) for k in ('episodes', 'actual_controlled_steps',
                'actual_simulation_steps', 'ensemble_forward_calls', 'member_forward_calls', 'policy_forward_calls')}
            limits = dict(zip(totals, (12, 6000, 7224, 6000, 18000, 6012)))
            require(len(results) == 12 and len(checks) == 6 and all(totals[k] <= v for k, v in limits.items()), 'Bridge budget/roster exceeded')
            check_request(path, expected)
            write_once(out/'index.json', {'files': files})
            report.update(status='complete', engineering_gate=True, checks=checks, jobs=results, totals=totals,
                index_sha256=sha(out/'index.json'), binary_byte_identity_equal=False,
                scoped_Raw_control_reuse_supported=True, all_MPC_native_functions_equivalent=False,
                scope='six_frozen_Raw_policies_first_development_scene_pure_Gym20_DDPG_only')
        except BaseException:
            report.update(status='failed', engineering_gate=False, failure_reason=traceback.format_exc()); raise
        finally:
            report['finished_at'] = shared.now(); write_once(out/'report.json', report)
    write_once(out/'receipt.json', {'request_hash': expected, 'report_sha256': sha(out/'report.json'), 'index_sha256': sha(out/'index.json')})
    print('bridge_report='+str(out/'report.json'))


def main():
    p = argparse.ArgumentParser(description=__doc__); sub = p.add_subparsers(dest='command', required=True)
    a = sub.add_parser('prepare'); a.add_argument('--baseline-root', required=True); a.add_argument('--run-id', required=True)
    for name in ('run', 'worker'):
        a = sub.add_parser(name); a.add_argument('--request', required=True); a.add_argument('--confirm-request-hash', required=True)
        if name == 'worker': a.add_argument('--job', required=True)
    args = p.parse_args()
    if args.command == 'prepare': prepare(args.baseline_root, args.run_id)
    elif args.command == 'run': run(args.request, args.confirm_request_hash)
    else: worker(args.request, args.confirm_request_hash, args.job)


if __name__ == '__main__': main()
