"""Read-only terminal evidence; never change upstream control or native scores."""
from contextlib import contextmanager
from copy import deepcopy
import math
from types import FunctionType

from prediction_rl.data.dataset_contract import digest
from prediction_rl.evaluation.driver_warmup import (PROTOCOL as P7F, CONDITIONS, AUTHOR, B0, B3, roster)

PROTOCOL = {
    'version': 'p07g_terminal_events_v1', 'purpose': 'development_raw_terminal_event_diagnosis',
    'parent_request': 'artifacts/p7f/p7f_diag_v1/request.json',
    'parent_request_hash': 'a87464d61fcf154bbc81444267ff8bf10e13ab80c09a077d72ccd2e45d76b917',
    'parent_aggregate_sha256': '9a1a37d30ec0c2181999570f9766655bc5a28179f968215ec56a57cf2f302d9e',
    'models': [AUTHOR, B0, B3], 'project_run_seeds': [0, 1, 2], 'simulator_seeds': [200, 210, 219],
    'conditions': deepcopy(CONDITIONS), 'selection': 'unchanged_complete_p7f_roster',
    'execution_contract': 'simulation_blocking_exact_v1', 'workers': 2, 'torch_threads': 1,
    'device': 'cpu', 'new_episodes': 84, 'strict_native_trajectory_parity': True,
    'native_reward_recomputed_with_new_labels': False, 'training_permitted': False,
    'formal_claim': False, 'test_opened': False,
    'capture_boundary': 'after_original_control_step_before_native_branch_cleanup_or_extra_step',
    'arrival_position_source': 'recorded_original_vehicle_add_argument_not_a_traci_getter',
    'ego_creation_contract': {'vehicle_id': 'ego', 'route_id': 'rampRoute', 'type_id': 'egoCar',
                              'depart_position_m': 40., 'arrival_position_m': 50.},
}
AUDIT = deepcopy(PROTOCOL)
AUDIT.update(version='p07g_terminal_events_audit_v1', purpose='bounded_read_only_instrumentation_audit',
             simulator_seeds=[200], selection='author_asset_b0_seed2_b3_seed0_mechanism_checks_not_model_selection',
             workers=1, new_episodes=12)


def validate_protocol(value):
    if digest(value) not in (digest(PROTOCOL), digest(AUDIT)):
        raise ValueError('Changed terminal-event design; approve a new version')
    return value


def jobs(config):
    validate_protocol(config)
    values = roster(P7F)
    if config == AUDIT:
        values = [j for j in values if j['simulator_seed'] == 200 and
                  (j['model_id'] == AUTHOR or (j['model_id'], j['run_seed']) in ((B0, 2), (B3, 0)))]
    return values


def number(x):
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x):
        raise ValueError('Invalid raw numeric evidence')
    return x


def snapshot(traci):
    ids = sorted(traci.vehicle.getIDList()); ego = None
    if 'ego' in ids:
        v = traci.vehicle
        ego = {'position': list(v.getPosition('ego')), 'speed': float(v.getSpeed('ego')),
               'acceleration': float(v.getAcceleration('ego')), 'lane_id': v.getLaneID('ego'),
               'lane_position_m': float(v.getLanePosition('ego')), 'route_id': v.getRouteID('ego'),
               'route': list(v.getRoute('ego')), 'route_index': int(v.getRouteIndex('ego'))}
    return {'simulation_time_s': float(traci.simulation.getTime()), 'live_vehicle_ids': ids, 'ego': ego}


def collision_object(c):
    return {'collider': c.collider, 'victim': c.victim, 'collider_type': c.colliderType,
            'victim_type': c.victimType, 'collider_speed': float(c.colliderSpeed),
            'victim_speed': float(c.victimSpeed), 'type': c.type, 'lane': c.lane, 'position_m': float(c.pos)}


class VehicleProxy:
    """Only the PROJECT-LOCAL control.traci reference uses this delegating proxy."""
    def __init__(self, observer): self.observer = observer
    def __getattr__(self, name): return getattr(self.observer.traci.vehicle, name)
    def add(self, *args, **kwargs):
        result = self.observer.traci.vehicle.add(*args, **kwargs)  # Same arguments, exactly once.
        if args and args[0] == 'ego':
            if len(args) != 3 or set(kwargs) != {'departSpeed', 'departPos', 'arrivalPos'}:
                raise ValueError('Unrecognized original ego add signature')
            self.observer.creations.append({
                'simulation_time_s': float(self.observer.traci.simulation.getTime()),
                'vehicle_id': args[0], 'route_id': args[1], 'type_id': args[2],
                'depart_speed_mps': float(kwargs['departSpeed']),
                'depart_position_m': float(kwargs['departPos']), 'arrival_position_m': float(kwargs['arrivalPos'])})
        return result


class TraciProxy:
    def __init__(self, observer): self.original = observer.traci; self.vehicle = VehicleProxy(observer)
    def __getattr__(self, name): return getattr(self.original, name)


class StepObserver:
    """Scoped upstream-module hooks. No installed TraCI globals are patched."""
    def __init__(self, control, traci, job):
        self.control, self.traci, self.job = control, traci, job
        self.events = []; self.removals = []; self.creations = []; self.entered = False
        self.cap = CONDITIONS[job['condition_id']]['warmup_s']*5 + 505

    def step(self):
        if len(self.events) >= self.cap: raise ValueError('Bounded original simulation-step cap exceeded')
        before = snapshot(self.traci)
        result = self.original_step()  # Includes original traffic insertion; never add a simulation step.
        after = snapshot(self.traci); s = self.traci.simulation
        self.events.append({'index': len(self.events), 'before': before, 'after': after,
            'arrived_vehicle_ids': list(s.getArrivedIDList()),
            'colliding_vehicle_ids': list(s.getCollidingVehiclesIDList()),
            'colliding_vehicle_number': int(s.getCollidingVehiclesNumber()),
            'collisions': [collision_object(c) for c in s.getCollisions()]})
        return result

    def remove(self):
        before = snapshot(self.traci); result = self.original_remove(); after = snapshot(self.traci)
        self.removals.append({'after_step_count': len(self.events), 'before': before, 'after': after,
                             'source': 'original_control_remove_ego_car'})
        return result

    def __enter__(self):
        if self.entered or getattr(self.control.step, '_terminal_event_hook', False):
            raise ValueError('Nested terminal instrumentation is forbidden')
        if self.control.traci is not self.traci: raise ValueError('Foreign control TraCI reference')
        self.original_step = self.control.step; self.original_remove = self.control.remove_ego_car
        self.control.step = self.step; self.control.remove_ego_car = self.remove
        self.control.traci = TraciProxy(self); self.entered = True
        return self

    def __exit__(self, *error):
        self.control.step = self.original_step; self.control.remove_ego_car = self.original_remove
        self.control.traci = self.traci; self.entered = False

    def export(self):
        return deepcopy({'version': 'raw_terminal_events_v1', 'job': self.job,
                         'capture_boundary': PROTOCOL['capture_boundary'],
                         'arrival_position_source': PROTOCOL['arrival_position_source'],
                         'creations': self.creations, 'explicit_removals': self.removals, 'steps': self.events})


def instrument_episode(function, session, args, job):
    """Reuse unchanged P7f bytecode with one scoped session dependency override."""
    observers = []
    @contextmanager
    def observed_session(*a, **kw):
        with session(*a, **kw) as settings:
            import control, traci
            with StepObserver(control, traci, job) as observer:
                observers.append(observer); yield settings
    wrapped = FunctionType(function.__code__, {**function.__globals__, 'upstream_session': observed_session},
                           function.__name__, function.__defaults__, function.__closure__)
    wrapped.__kwdefaults__ = function.__kwdefaults__
    result = wrapped(*args)
    if len(observers) != 1: raise ValueError('Original episode must own exactly one session')
    return result, observers[0].export()


def validate_snapshot(value):
    if set(value) != {'simulation_time_s', 'live_vehicle_ids', 'ego'}: raise ValueError('Invalid snapshot fields')
    number(value['simulation_time_s']); ids = value['live_vehicle_ids']; e = value['ego']
    if ids != sorted(set(ids)) or any(not isinstance(i, str) or not i for i in ids):
        raise ValueError('Invalid actual live IDs')
    if (e is None) != ('ego' not in ids): raise ValueError('Ego presence disagrees with actual IDs')
    if e is not None:
        if (set(e) != {'position', 'speed', 'acceleration', 'lane_id', 'lane_position_m', 'route_id', 'route', 'route_index'}
                or len(e['position']) != 2 or e['route_id'] != 'rampRoute'
                or e['route'] != ['ramp', 'highwayahead'] or type(e['route_index']) is not int
                or e['route_index'] not in (0, 1) or not isinstance(e['lane_id'], str)):
            raise ValueError('Unknown actual ego route/state')
        for x in (*e['position'], e['speed'], e['acceleration'], e['lane_position_m']): number(x)
        if e['speed'] < 0 or e['lane_position_m'] < 0: raise ValueError('Invalid ego motion')


def classify(sidecar, native_row, decisions):
    """Raw ego involvement wins over arrived flags; native labels/scores stay intact."""
    if (set(sidecar) != {'version', 'job', 'capture_boundary', 'arrival_position_source', 'creations', 'explicit_removals', 'steps'}
            or sidecar['version'] != 'raw_terminal_events_v1'
            or set(sidecar['job']) != set(roster(P7F)[0])
            or sidecar['job'] != {k: native_row[k] for k in sidecar['job']}
            or sidecar['capture_boundary'] != PROTOCOL['capture_boundary']
            or sidecar['arrival_position_source'] != PROTOCOL['arrival_position_source']):
        raise ValueError('Invalid sidecar identity/contract')
    created = sidecar['creations']; events = sidecar['steps']; removals = sidecar['explicit_removals']
    job = sidecar['job']; warmup = CONDITIONS[job['condition_id']]['warmup_s']
    if len(created) != 1 or any(created[0].get(k) != v for k, v in PROTOCOL['ego_creation_contract'].items()):
        raise ValueError('Original ego creation arguments changed')
    if not math.isclose(number(created[0]['simulation_time_s']), warmup, abs_tol=1e-7):
        raise ValueError('Ego creation time changed')
    number(created[0]['depart_speed_mps'])
    if not warmup*5 < len(events) <= warmup*5+505: raise ValueError('Missing or unbounded raw steps')
    extra = (int(bool(native_row['time_limit'])) if CONDITIONS[job['condition_id']]['driver'] == 'gym'
             else 2 if native_row['reported_collision'] else 1)
    if len(events) != warmup*5+1+native_row['steps']+extra:
        raise ValueError('Native cleanup/extra-step count changed')
    ego_collision_indices = []; disappearance_indices = []; background_indices = []
    for i, event in enumerate(events):
        if set(event) != {'index', 'before', 'after', 'arrived_vehicle_ids', 'colliding_vehicle_ids',
                          'colliding_vehicle_number', 'collisions'} or event['index'] != i:
            raise ValueError('Raw event sequence changed')
        b, a = event['before'], event['after']; validate_snapshot(b); validate_snapshot(a)
        if (not math.isclose(a['simulation_time_s']-b['simulation_time_s'], .2, abs_tol=1e-7)
                or not math.isclose(b['simulation_time_s'], .2*i, abs_tol=1e-7)):
            raise ValueError('Observer added/lost a simulation step')
        for field in ('arrived_vehicle_ids', 'colliding_vehicle_ids'):
            if not isinstance(event[field], list) or any(not isinstance(x, str) or not x for x in event[field]):
                raise ValueError('Invalid raw event IDs')
        if (type(event['colliding_vehicle_number']) is not int
                or event['colliding_vehicle_number'] != len(event['colliding_vehicle_ids'])):
            raise ValueError('Collision ID/count disagreement')
        participants = set(event['colliding_vehicle_ids'])
        for c in event['collisions']:
            if set(c) != {'collider', 'victim', 'collider_type', 'victim_type', 'collider_speed',
                          'victim_speed', 'type', 'lane', 'position_m'}:
                raise ValueError('Incomplete collision object')
            if any(not isinstance(c[k], str) or not c[k] for k in ('collider', 'victim', 'type', 'lane')):
                raise ValueError('Invalid collision participant/type')
            for k in ('collider_speed', 'victim_speed', 'position_m'): number(c[k])
            participants.update((c['collider'], c['victim']))
        if 'ego' in participants: ego_collision_indices.append(i)
        elif participants and a['simulation_time_s'] > warmup: background_indices.append(i)
        if b['ego'] is not None and a['ego'] is None: disappearance_indices.append(i)
    for r in removals:
        if set(r) != {'after_step_count', 'before', 'after', 'source'} or r['source'] != 'original_control_remove_ego_car':
            raise ValueError('Invalid explicit removal evidence')
        validate_snapshot(r['before']); validate_snapshot(r['after'])
        n = r['after_step_count']
        if type(n) is not int or not 1 <= n <= len(events) or r['before']['simulation_time_s'] != r['after']['simulation_time_s']:
            raise ValueError('Removal performed an extra simulation step')
        if not math.isclose(r['before']['simulation_time_s'], events[n-1]['after']['simulation_time_s'], abs_tol=1e-7):
            raise ValueError('Removal boundary not bound to raw step')
    # Match every actual action to the pre-step ego evidence, before cleanup.
    control_indices = []
    for decision in decisions:
        matches = [e for e in events if math.isclose(e['before']['simulation_time_s'],
                   decision['frame']['simulation_time_s'], abs_tol=1e-7)]
        if len(matches) != 1 or matches[0]['before']['ego'] is None: raise ValueError('Missing actual control boundary')
        ego = matches[0]['before']['ego']; expected = decision['frame']['vehicles']['ego']
        if any(ego[k] != expected[k] for k in ('position', 'speed', 'acceleration', 'lane_id')):
            raise ValueError('Sidecar does not match actual decision state')
        control_indices.append(matches[0]['index'])
    if len(control_indices) != native_row['steps'] or any(b != a+1 for a, b in zip(control_indices, control_indices[1:])):
        raise ValueError('Control steps not contiguous')
    overlap = [i for i in ego_collision_indices if 'ego' in events[i]['arrived_vehicle_ids']]
    terminal_index = None; label = 'unexplained_terminal'
    if ego_collision_indices:
        label = 'ego_collision'; terminal_index = ego_collision_indices[0]
    elif disappearance_indices:
        terminal_index = disappearance_indices[0]; event = events[terminal_index]; e = event['before']['ego']
        near_endpoint = (e['lane_id'] == 'highwayahead_0' and e['route_index'] == 1 and
                         e['lane_position_m'] >= created[0]['arrival_position_m']-30*.2-1.)
        if 'ego' in event['arrived_vehicle_ids'] and near_endpoint: label = 'natural_arrival'
        elif 'ego' in event['arrived_vehicle_ids']: label = 'arrived_far_before_endpoint'
        else: label = 'unexplained_ego_disappearance'
    elif removals and native_row['time_limit']:
        label = 'time_limit'
    elif removals and background_indices and native_row['reported_collision']:
        label = 'background_collision_stop'
    return {'classification': label, 'terminal_step_index': terminal_index,
            'ego_collision_step_indices': ego_collision_indices, 'ego_collision_arrived_overlap_indices': overlap,
            'background_collision_step_indices': background_indices,
            'native_arrival_with_ego_collision': bool(native_row['arrival'] and ego_collision_indices),
            'native_event_agrees': bool((native_row['arrival'] and label == 'natural_arrival') or
                (native_row['reported_collision'] and label == 'ego_collision') or (native_row['time_limit'] and label == 'time_limit')),
            'raw_simulation_steps': len(events), 'actual_control_steps': len(control_indices),
            'arrival_position_m': created[0]['arrival_position_m'],
            'endpoint_check': 'recorded_arrival_flag_no_recorded_ego_collision_and_route_endpoint_consistency_not_a_safety_guarantee'}


def assert_reference_parity(row, decisions, old_row, old_decisions):
    keys = ('arrival', 'reported_collision', 'time_limit', 'steps', 'simulation_duration_s', 'native_score',
            'native_mean_abs_jerk', 'policy_sha256', 'predictor_sha256', 'common_initial_traffic_sha256')
    if any(row[k] != old_row[k] for k in keys) or len(decisions) != len(old_decisions):
        raise ValueError('Instrumentation changed native outcome; do not relax parity')
    observations = 0
    for now, old in zip(decisions, old_decisions):
        if any(now[k] != old[k] for k in ('frame', 'requested_jerk', 'commanded_speed', 'time_feature')):
            raise ValueError('Instrumentation changed native trajectory/action/time; do not relax parity')
        if old['observation'] is not None:
            observations += 1
            if now['observation'] != old['observation']: raise ValueError('Instrumentation changed native observation')
    return {'exact_native_outcome_parity': True, 'exact_available_trajectory_parity': True,
            'compared_control_steps': len(decisions), 'historical_observations_compared': observations,
            'historical_missing_observations_not_reconstructed': True}


def aggregate(config, rows):
    expected = jobs(config); index = {r['id']: r for r in rows}
    if len(index) != len(rows) or set(index) != {j['id'] for j in expected}: raise ValueError('Missing/extra/duplicate cells')
    for j in expected:
        r = index[j['id']]
        if (any(r.get(k) != v for k, v in j.items()) or r['parity']['exact_native_outcome_parity'] is not True
                or r['parity']['exact_available_trajectory_parity'] is not True):
            raise ValueError('Cell identity/parity changed')
    matrix = {}
    for model in PROTOCOL['models']:
        matrix[model] = {}
        for condition in CONDITIONS:
            values = [r for r in rows if r['model_id'] == model and r['condition_id'] == condition]
            counts = {k: sum(r['independent_events']['classification'] == k for r in values)
                      for k in ('natural_arrival', 'ego_collision', 'time_limit', 'background_collision_stop',
                                'arrived_far_before_endpoint', 'unexplained_ego_disappearance', 'unexplained_terminal')}
            if sum(counts.values()) != len(values): raise ValueError('Unknown independent category')
            matrix[model][condition] = {'episodes': len(values), 'native_arrivals': sum(r['arrival'] for r in values),
                'native_collisions': sum(r['reported_collision'] for r in values), 'independent_events': counts,
                'native_arrival_with_ego_collision': sum(r['independent_events']['native_arrival_with_ego_collision'] for r in values)}
    return {'status': 'complete', 'engineering_complete': True, 'method_effect_gate': None, 'rows': rows,
            'episodes': len(rows), 'independent_simulator_seed_ids': len(config['simulator_seeds']), 'matrix': matrix,
            'native_reward_recomputed_with_new_labels': False, 'formal_claim': False, 'test_opened': False}
