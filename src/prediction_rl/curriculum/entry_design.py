"""IC02a static entry inspection. Does not import or implement a simulator."""
import ast
from collections import Counter, defaultdict

from prediction_rl.data.dataset_contract import digest

SOURCE_PATHS = (
    'RL-MPC-LaneMerging-master/config.py',
    'RL-MPC-LaneMerging-master/configs/train_default_1.json',
    'RL-MPC-LaneMerging-master/merge_gym.py',
    'RL-MPC-LaneMerging-master/control.py',
    'RL-MPC-LaneMerging-master/sumo.py',
    'src/prediction_rl/envs/upstream.py',
    'src/prediction_rl/envs/prediction_features.py',
    'src/prediction_rl/training/all_ddpg.py',
)
ARMS = ['b2_mix', 'b2_curriculum', 'b3_mix', 'b3_curriculum']
SMOKE = {
    'authorized': False, 'simulator_seeds': [11000, 11001, 11002],
    'adapters': ['original_direct', 'proposed_worker_entry'], 'repeats': 2,
    'requested_jerks': [0, 2.5, -2.5, 5, -5, 0, 0, 2.5, -2.5, 0, 0, 0],
    'max_control_steps_per_episode': 12, 'optimizer_updates': 0,
    'predictor_modes': ['ordinary', 'conditional'],
    'bounded_cap_is_not_mdp_terminal': True,
}


def validate_design(config):
    expected = {
        'schema_version': 1, 'protocol_id': 'ic02a-entry-feasibility-v1',
        'scope': 'offline_source_and_mapping_only',
        'baseline_references': 'provenance/interaction_curriculum_baseline_v1.json',
        'ic01_report': 'artifacts/ic/ic01_v1/report.json',
        'ic01_report_sha256': '5479b3deac1bebad28101afc50c1b9668aa39afd581d971b77e3bb27228acb7b',
        'proposed_entry_contract': 'seeded_fresh_gym20_task_v1',
        'proposal_status': 'unapproved_not_implemented',
        'policy_start': 'original_reset_first_observation',
        'reference_prefix_before_policy_steps': 0, 'prefill_predictor_history': False,
        'environment_rng_scope': 'environment_worker_only',
        'learner_rng_scope': 'persistent_learner_not_reseeded_by_task',
        'matched_arms': ARMS, 'target_distribution_pi': None,
        'task_classifier': 'unresolved_no_root_label_promoted_to_episode',
        'smoke_proposal': SMOKE, 'training_ready': False,
    }
    if not isinstance(config, dict) or digest(config) != digest(expected):
        raise ValueError('Unknown/changed IC02a design; this is not a runtime approval')
    return config


def method(tree, name, owner=None):
    body = tree.body
    if owner:
        owners = [n for n in body if isinstance(n, ast.ClassDef) and n.name == owner]
        if len(owners) != 1:
            raise ValueError('Missing/ambiguous source class '+owner)
        body = owners[0].body
    found = [n for n in body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name]
    if len(found) != 1:
        raise ValueError('Missing/ambiguous source method '+name)
    return found[0]


def calls(node, name):
    return [n for n in ast.walk(node) if isinstance(n, ast.Call) and ast.unparse(n.func) == name]


def inspect_source(sources, upstream_config):
    """Check pinned statements by AST, never exec/import upstream source."""
    if set(sources) != set(SOURCE_PATHS)-{SOURCE_PATHS[1]}:
        raise ValueError('Explicit complete source inventory required')
    trees = {p: ast.parse(text, filename=p) for p, text in sources.items()}
    gym = trees[SOURCE_PATHS[2]]; control = trees[SOURCE_PATHS[3]]
    reset = method(gym, 'reset', 'JerkEnv'); init = method(gym, '__init__', 'JerkEnv')
    add = method(control, 'add_ego_car')
    speed = method(control, 'get_ego_start_speed')
    step = method(control, 'step')
    session = method(trees[SOURCE_PATHS[5]], 'upstream_session')
    audited_init = method(trees[SOURCE_PATHS[5]], '__init__', 'AuditedJerkEnv')
    limits = calls(audited_init, 'LegacyTimeLimit')
    history = method(trees[SOURCE_PATHS[6]], 'reset', 'PredictionFeatureEnv')
    augment = method(trees[SOURCE_PATHS[6]], '_augment', 'PredictionFeatureEnv')
    settings = next(n for n in trees[SOURCE_PATHS[0]].body if isinstance(n, ast.ClassDef) and n.name == 'Settings')
    assignments = {n.targets[0].id: n.value for n in settings.body if isinstance(n, ast.Assign)
                   and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)}
    if not isinstance(upstream_config, dict) or set(upstream_config)-set(assignments):
        raise ValueError('Unknown upstream configuration field')
    selected = ('TICK_LENGTH', 'MAX_EPISODE_LENGTH', 'START_SPEED', 'START_SPEED_VARIANCE',
        'RANDOMIZE_START_SPEED', 'MIN_START_SPEED', 'MAX_START_SPEED',
        'USE_SIMPLE_TRAFFIC_DISTRIBUTION', 'USE_ALTERNATE_TRAFFIC_DISTRIBUTION',
        'BASE_TRAFFIC_INTERVAL', 'VARY_TRAFFIC_START_TIMES', 'OTHER_CAR_SPEED',
        'REWARD_FUNCTION', 'CRASH_REWARD', 'SUCCESS_REWARD', 'TIME_REWARD', 'ALT_J_WEIGHT',
        'MINIMUM_NEGATIVE_JERK', 'MAXIMUM_POSITIVE_JERK')
    resolved = {k: upstream_config[k] if k in upstream_config else ast.literal_eval(assignments[k]) for k in selected}
    add_calls = calls(add, 'traci.vehicle.add')
    if len(add_calls) != 1:
        raise ValueError('Unknown ego insertion path')
    insert = add_calls[0]; kw = {k.arg: k.value for k in insert.keywords}
    mode = calls(add, 'traci.vehicle.setSpeedMode')
    wait = [n.value for n in ast.walk(init) if isinstance(n, ast.Assign)
            and any(ast.unparse(t) == 'self.wait_before_start' for t in n.targets)]
    reset_signature = (len(reset.args.args) == 1 and reset.args.args[0].arg == 'self'
        and not reset.args.posonlyargs and not reset.args.kwonlyargs
        and reset.args.vararg is None and reset.args.kwarg is None)
    warmup = [n for n in ast.walk(reset) if isinstance(n, ast.For)
              and ast.unparse(n.iter) == 'range(int(self.wait_before_start / Settings.TICK_LENGTH))']
    clear = calls(history, 'self.history.clear'); base = calls(history, 'self.base.reset')
    assumptions = {
        'original_reset_has_no_task_or_seed_argument': reset_signature,
        'sumo_start_is_constructor_not_reset': len(calls(init, 'sumo.start_sumo')) == 1
            and not calls(reset, 'sumo.start_sumo') and not calls(reset, 'traci.load'),
        'reset_waits_original_20_seconds': len(wait) == 1 and ast.unparse(wait[0]) == "config.get('wait_before_start', 20)"
            and len(warmup) == 1 and len(calls(warmup[0], 'control.step')) == 1,
        'reset_draws_then_inserts_original_start_speed': len(calls(reset, 'control.get_ego_start_speed')) == 1
            and len(calls(reset, 'control.add_ego_car')) == 1,
        'reset_has_one_insertion_tick_after_warmup': len(calls(reset, 'control.step')) == 2
            and calls(reset, 'control.add_ego_car')[0].lineno < max(n.lineno for n in calls(reset, 'control.step')),
        'ego_insertion_route_position_and_mode_unchanged':
            [ast.literal_eval(n) for n in insert.args] == ['ego', 'rampRoute', 'egoCar']
            and ast.literal_eval(kw['departPos']) == 40 and ast.literal_eval(kw['arrivalPos']) == 50
            and ast.unparse(kw['departSpeed']) == 'start_velocity'
            and len(mode) == 1 and ast.literal_eval(mode[0].args[1]) == 22,
        'start_speed_uses_global_numpy_normal': len(calls(speed, 'np.random.normal')) == 1
            and [ast.unparse(n) for n in calls(speed, 'np.random.normal')[0].args]
                == ['Settings.START_SPEED', 'Settings.START_SPEED_VARIANCE'],
        'background_jitter_uses_global_python_rng': len(calls(step, 'random.random')) == 1,
        'session_reseeds_python_numpy_and_torch': all(len(calls(session, n)) == 1
            and [ast.unparse(a) for a in calls(session, n)[0].args] == ['seed']
            for n in ('random.seed', 'np.random.seed', 'torch.manual_seed')),
        'upstream_tick_and_raw_deadline_unchanged': resolved['TICK_LENGTH'] == .2
            and resolved['MAX_EPISODE_LENGTH'] == 100,
        'registered_wrapper_is_original_500_ticks': len(limits) == 1
            and [ast.unparse(a) for a in limits[0].args] == ['self.raw', '500'],
        'learned_history_is_cleared_before_base_reset': len(clear) == len(base) == 1
            and clear[0].lineno < base[0].lineno,
        'learned_observation_appends_one_current_frame_not_prefill': len(calls(augment, 'self.history.append')) == 1
            and len(calls(history, 'self._augment')) == 1 and not calls(history, 'self.history.append'),
        'all_checkpoint_time_feature_scale_remains_0_001': any(
            isinstance(n, ast.Dict) and any(isinstance(k, ast.Constant) and k.value == 'time_feature_scale'
                and isinstance(v, ast.Constant) and v.value == .001 for k, v in zip(n.keys, n.values))
            for n in ast.walk(trees[SOURCE_PATHS[7]])),
    }
    if not all(assumptions.values()):
        raise ValueError('Entry source assumptions changed: '+str([k for k,v in assumptions.items() if not v]))
    return {'method': 'static_ast_checks_not_runtime_parity', 'runtime_verified': False,
        'assumptions': assumptions, 'resolved_upstream_subset': resolved,
        'start_speed_variance_name_is_numpy_normal_standard_deviation': True,
        'reset_entry_control_steps': 0, 'reset_warmup_ticks': 100,
        'reset_insertion_ticks': 1, 'learned_initial_history_frames': 1,
        'native_continuing_reset_can_request_named_task': False}


def mapping_summary(rows, report):
    """Group source roots; deliberately do NOT classify executable episodes."""
    groups = defaultdict(list); roots = set()
    for row in rows:
        if (row['root_id'] in roots or type(row['simulator_seed']) is not int
                or not 11000 <= row['simulator_seed'] <= 11255
                or row['level'] not in ('C0', 'C1', 'C2')
                or row['task_mapping']['executable_as_original_reset'] is not False):
            raise ValueError('Invalid IC01 train root mapping')
        roots.add(row['root_id']); groups[row['episode_id']].append(row)
    records = []
    for episode, family in sorted(groups.items()):
        if len({r['simulator_seed'] for r in family}) != 1 or len({r['job_id'] for r in family}) != 1:
            raise ValueError('Episode seed/job identity changed between roots')
        records.append({'episode_id': episode, 'simulator_seed': family[0]['simulator_seed'],
            'job_id': family[0]['job_id'], 'reference_roots': len(family),
            'provisional_root_levels': dict(Counter(r['level'] for r in family)),
            'task_level': None, 'entry_verified': False, 'execution_authorized': False})
    expected = report['summary']
    if (len(rows) != expected['reached_roots'] or len(records) != expected['episodes']
            or {k: sum(r['level'] == k for r in rows) for k in ('C0','C1','C2')}
               != {k: expected['classification']['levels'][k]['roots'] for k in ('C0','C1','C2')}):
        raise ValueError('IC01 summary/root family mismatch')
    return {'source_episodes': len(records), 'source_roots': len(rows),
        'source_episodes_with_any_provisional_C1_root': sum('C1' in r['provisional_root_levels'] for r in records),
        'classified_executable_tasks': 0, 'original_continuing_target_pi': None,
        'fresh_task_target_pi': None, 'old_root_thresholds_changed': False,
        'scope': 'source_episode_inventory_not_task_pool', 'records': records}


def smoke_budget(proposal):
    if digest(proposal) != digest(SMOKE):
        raise ValueError('Unapproved or altered smoke proposal')
    episodes = len(proposal['simulator_seeds'])*len(proposal['adapters'])*proposal['repeats']
    return {'execution_authorized': False, 'episodes_max': episodes,
        'control_steps_max': episodes*proposal['max_control_steps_per_episode'],
        'warmup_and_insertion_ticks_max': episodes*101,
        'feature_calls_per_predictor_mode_max': episodes*(1+proposal['max_control_steps_per_episode']),
        'optimizer_updates': 0, 'new_counterfactual_branches': 0}
