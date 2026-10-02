"""P7g analysis repair; keep the pinned v1 recorder and all raw evidence intact.

Only the author's time-limit path differs: its final control command is followed
by remove_ego_car and one cleanup step, NOT a step with ego still present.
"""
import math

from .terminal_events import PROTOCOL, CONDITIONS, P7F, roster, number, validate_snapshot

VERSION = 'p07g_raw_terminal_analysis_v2'


def classify(sidecar, native_row, decisions):
    """Validate actual events, including the cancelled final author command.

This separate implementation preserves v1 source provenance. Normal branches
retain v1's exact output; no synthetic steps or reconstructed snapshots are used.
"""
    if (set(sidecar) != {'version', 'job', 'capture_boundary', 'arrival_position_source', 'creations', 'explicit_removals', 'steps'}
            or sidecar['version'] != 'raw_terminal_events_v1'
            or set(sidecar['job']) != set(roster(P7F)[0])
            or sidecar['job'] != {k: native_row[k] for k in sidecar['job']}
            or sidecar['capture_boundary'] != PROTOCOL['capture_boundary']
            or sidecar['arrival_position_source'] != PROTOCOL['arrival_position_source']):
        raise ValueError('Invalid sidecar identity/contract')
    created = sidecar['creations']; events = sidecar['steps']; removals = sidecar['explicit_removals']
    job = sidecar['job']; condition = CONDITIONS[job['condition_id']]; warmup = condition['warmup_s']
    author_limit = condition['driver'] == 'author' and native_row['time_limit'] == 1
    if len(created) != 1 or any(created[0].get(k) != v for k, v in PROTOCOL['ego_creation_contract'].items()):
        raise ValueError('Original ego creation arguments changed')
    if not math.isclose(number(created[0]['simulation_time_s']), warmup, abs_tol=1e-7):
        raise ValueError('Ego creation time changed')
    number(created[0]['depart_speed_mps'])
    if not warmup*5 < len(events) <= warmup*5+505: raise ValueError('Missing or unbounded raw steps')
    # For author time-limit, the final command is cancelled by removal. Its
    # cleanup step replaces (rather than follows) the final ego dynamics step.
    extra = (int(bool(native_row['time_limit'])) if condition['driver'] == 'gym'
             else 0 if author_limit else 2 if native_row['reported_collision'] else 1)
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
    if author_limit:
        if (native_row['steps'] != condition['max_control_calls'] or native_row['arrival']
                or native_row['reported_collision'] or len(removals) != 1
                or len(decisions) != native_row['steps']):
            raise ValueError('Invalid author time-limit branch')
        removal = removals[0]; cleanup = events[-1]
        if (removal['after_step_count'] != len(events)-1 or removal['before']['ego'] is None
                or removal['after']['ego'] is not None or cleanup['before'] != removal['after']
                or events[-2]['after'] != removal['before']
                or decisions[-1]['frame']['simulation_time_s'] != removal['before']['simulation_time_s']):
            raise ValueError('Author final command not bound to actual removal/cleanup')
    control_indices = []
    for index, decision in enumerate(decisions):
        matches = [e for e in events if math.isclose(e['before']['simulation_time_s'],
                   decision['frame']['simulation_time_s'], abs_tol=1e-7)]
        if len(matches) != 1: raise ValueError('Missing actual control boundary')
        ego = matches[0]['before']['ego']
        if author_limit and index == len(decisions)-1:
            # Actual pre-removal state, not an invented ego in the cleanup step.
            if matches[0]['index'] != len(events)-1: raise ValueError('Author final control boundary changed')
            ego = removals[0]['before']['ego']
        if ego is None: raise ValueError('Missing actual control boundary')
        expected = decision['frame']['vehicles']['ego']
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
    elif removals and native_row['time_limit']: label = 'time_limit'
    elif removals and background_indices and native_row['reported_collision']: label = 'background_collision_stop'
    return {'classification': label, 'terminal_step_index': terminal_index,
            'ego_collision_step_indices': ego_collision_indices, 'ego_collision_arrived_overlap_indices': overlap,
            'background_collision_step_indices': background_indices,
            'native_arrival_with_ego_collision': bool(native_row['arrival'] and ego_collision_indices),
            'native_event_agrees': bool((native_row['arrival'] and label == 'natural_arrival') or
                (native_row['reported_collision'] and label == 'ego_collision') or (native_row['time_limit'] and label == 'time_limit')),
            'raw_simulation_steps': len(events), 'actual_control_steps': len(control_indices),
            'arrival_position_m': created[0]['arrival_position_m'],
            'endpoint_check': 'recorded_arrival_flag_no_recorded_ego_collision_and_route_endpoint_consistency_not_a_safety_guarantee'}
