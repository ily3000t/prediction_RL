from copy import deepcopy
from contextlib import contextmanager
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'tools')]
import audit_terminal_events as runner
from prediction_rl.evaluation.terminal_events import (PROTOCOL, AUDIT, AUTHOR, B0, B3, CONDITIONS,
    validate_protocol, jobs, StepObserver, snapshot, classify, assert_reference_parity, aggregate, instrument_episode)


def ego(lane='highwayahead_0', position=49.):
    return {'position': [position+1.5, -1.6], 'speed': 7., 'acceleration': 0., 'lane_id': lane,
            'lane_position_m': position, 'route_id': 'rampRoute', 'route': ['ramp', 'highwayahead'], 'route_index': 1}


def snap(t, e=None):
    return {'simulation_time_s': t, 'live_vehicle_ids': ['ego', 'traffic'] if e else ['traffic'], 'ego': deepcopy(e)}


def fixture(label='arrival', condition='gym20', background=False):
    j = next(j for j in jobs(AUDIT) if j['model_id'] == AUTHOR and j['condition_id'] == condition)
    warm = CONDITIONS[condition]['warmup_s']; events = []
    for i in range(warm*5):
        events.append({'index': i, 'before': snap(.2*i), 'after': snap(.2*(i+1)), 'arrived_vehicle_ids': [],
                       'colliding_vehicle_ids': [], 'colliding_vehicle_number': 0, 'collisions': []})
    events.append({'index': len(events), 'before': snap(warm), 'after': snap(warm+.2, ego()),
                   'arrived_vehicle_ids': [], 'colliding_vehicle_ids': [], 'colliding_vehicle_number': 0, 'collisions': []})
    e = ego(position=10. if label == 'far' else 49.)
    if label == 'far': events[-1]['after']['ego'] = deepcopy(e)
    collision = label in ('collision', 'overlap') or background
    participants = ['traffic', 'traffic2'] if background else ['ego', 'traffic']
    c = {'collider': participants[0], 'victim': participants[1], 'collider_type': 'egoCar',
         'victim_type': 'normal', 'collider_speed': 7., 'victim_speed': 7.,
         'type': 'junction' if not background else 'collision', 'lane': 'highwayahead_0', 'position_m': 49.}
    disappears = label in ('arrival', 'collision', 'overlap', 'far', 'unknown')
    events.append({'index': len(events), 'before': snap(warm+.2, e), 'after': snap(warm+.4, None if disappears else e),
        'arrived_vehicle_ids': ['ego'] if label in ('arrival', 'overlap', 'far') else [],
        'colliding_vehicle_ids': participants if collision else [], 'colliding_vehicle_number': 2 if collision else 0,
        'collisions': [c] if collision else []})
    arrival = label in ('arrival', 'overlap', 'far', 'unknown')
    native = {**j, 'arrival': int(arrival), 'reported_collision': int(label == 'collision' or background),
              'time_limit': int(label == 'limit'), 'steps': 1}
    removals = []
    if background or label == 'limit':
        removals.append({'after_step_count': len(events), 'before': snap(warm+.4, e), 'after': snap(warm+.4),
                         'source': 'original_control_remove_ego_car'})
    extra = (int(label == 'limit') if condition.startswith('gym') else 2 if native['reported_collision'] else 1)
    for _ in range(extra):
        t = events[-1]['after']['simulation_time_s']
        events.append({'index': len(events), 'before': snap(t), 'after': snap(t+.2),
            'arrived_vehicle_ids': [], 'colliding_vehicle_ids': [], 'colliding_vehicle_number': 0, 'collisions': []})
    sidecar = {'version': 'raw_terminal_events_v1', 'job': j, 'capture_boundary': PROTOCOL['capture_boundary'],
        'arrival_position_source': PROTOCOL['arrival_position_source'],
        'creations': [{**PROTOCOL['ego_creation_contract'], 'simulation_time_s': warm, 'depart_speed_mps': 7.}],
        'explicit_removals': removals, 'steps': events}
    record = {'frame': {'simulation_time_s': warm+.2, 'vehicles': {'ego': {k:e[k] for k in
               ('position', 'speed', 'acceleration', 'lane_id')}}}}
    return sidecar, native, [record]


def test_frozen_design_and_all_cells_reexecuted_without_training():
    assert json.loads((ROOT/'configs/development/p07g_terminal_events_v1.json').read_text()) == PROTOCOL
    assert len(jobs(PROTOCOL)) == 84 and len(jobs(AUDIT)) == 12
    assert {(j['model_id'], j['run_seed']) for j in jobs(AUDIT)} == {(AUTHOR, None), (B0, 2), (B3, 0)}
    assert PROTOCOL['simulator_seeds'] == [200, 210, 219]
    assert not PROTOCOL['training_permitted'] and not PROTOCOL['native_reward_recomputed_with_new_labels']


@pytest.mark.parametrize('key,value', [('workers', True), ('simulator_seeds', [201]), ('project_run_seeds', [0,1]),
    ('strict_native_trajectory_parity', False), ('new_episodes', 42), ('training_permitted', True), ('extra', 1)])
def test_protocol_amendments_rejected(key, value):
    p = deepcopy(PROTOCOL); p[key] = value
    with pytest.raises(ValueError): validate_protocol(p)


@pytest.mark.parametrize('condition', list(CONDITIONS))
def test_native_arrival_collision_overlap_classified_without_native_relabel(condition):
    s, row, d = fixture('overlap', condition); before = deepcopy(row)
    a = classify(s, row, d)
    assert a['classification'] == 'ego_collision' and a['native_arrival_with_ego_collision']
    assert len(a['ego_collision_arrived_overlap_indices']) == 1 and not a['native_event_agrees']
    assert row == before


@pytest.mark.parametrize('label,expected', [('arrival','natural_arrival'),('far','arrived_far_before_endpoint'),
    ('collision','ego_collision'),('limit','time_limit'),('unknown','unexplained_ego_disappearance')])
def test_independent_categories_do_not_synthesize_success(label, expected):
    s, r, d = fixture(label); assert classify(s, r, d)['classification'] == expected


def test_background_collision_is_not_ego_collision():
    s, r, d = fixture('background', 'author20', True)
    a = classify(s, r, d); assert a['classification'] == 'background_collision_stop'
    assert not a['ego_collision_step_indices'] and not a['native_event_agrees']


@pytest.mark.parametrize('fault', ['count','gap','nan','missing_object','presence','creation','extra_step','state','missing_job_field'])
def test_incomplete_or_mutated_evidence_rejected(fault):
    s, r, d = fixture('overlap'); e = s['steps'][-1]
    if fault == 'count': e['colliding_vehicle_number'] = 1
    elif fault == 'gap': e['after']['simulation_time_s'] += .2
    elif fault == 'nan': e['collisions'][0]['position_m'] = float('nan')
    elif fault == 'missing_object': del e['collisions'][0]['victim']
    elif fault == 'presence': e['after']['live_vehicle_ids'].append('ego')
    elif fault == 'creation': s['creations'][0]['arrival_position_m'] = 55.
    elif fault == 'extra_step': s['steps'].append(deepcopy(e))
    elif fault == 'state': d[0]['frame']['vehicles']['ego']['speed'] = 8.
    elif fault == 'missing_job_field': del s['job']['training_id']
    with pytest.raises(ValueError): classify(s, r, d)


def fake_runtime():
    state = {'time':0., 'live':False, 'calls':0, 'adds':0, 'removes':0}
    v = SimpleNamespace(getIDList=lambda:['ego'] if state['live'] else [], getPosition=lambda _: (1.,2.),
        getSpeed=lambda _:7., getAcceleration=lambda _:0., getLaneID=lambda _:'highwayahead_0',
        getLanePosition=lambda _:49., getRouteID=lambda _:'rampRoute', getRoute=lambda _:('ramp','highwayahead'),
        getRouteIndex=lambda _:1)
    def add(*a, **kw): state.update(live=True, adds=state['adds']+1); return 'add-result'
    v.add = add
    s = SimpleNamespace(getTime=lambda:state['time'], getArrivedIDList=lambda:('ego',) if not state['live'] and state['calls'] else (),
        getCollidingVehiclesIDList=lambda:('ego','traffic') if state['calls'] else (),
        getCollidingVehiclesNumber=lambda:2 if state['calls'] else 0, getCollisions=lambda:())
    traci = SimpleNamespace(vehicle=v, simulation=s)
    def step(): state.update(time=state['time']+.2, live=False, calls=state['calls']+1); return 'step-result'
    def remove(): state.update(live=False, removes=state['removes']+1); return 'remove-result'
    control = SimpleNamespace(step=step, remove_ego_car=remove, traci=traci)
    return state, traci, control


def test_scoped_observer_original_calls_once_and_restores_without_installed_patch():
    state, traci, control = fake_runtime(); original_step, original_add = control.step, traci.vehicle.add
    with StepObserver(control, traci, jobs(AUDIT)[0]) as o:
        assert control.traci.vehicle.add('ego','rampRoute','egoCar',departSpeed=7.,departPos=40,arrivalPos=50) == 'add-result'
        assert control.step() == 'step-result'
        assert o.events[0]['before']['ego'] is not None and o.events[0]['after']['ego'] is None
        assert o.events[0]['arrived_vehicle_ids'] == ['ego']
        assert control.remove_ego_car() == 'remove-result'
        assert o.removals[0]['after_step_count'] == 1
        assert traci.vehicle.add is original_add  # Installed module reference untouched.
    assert control.step is original_step and control.traci is traci
    assert state['calls'] == state['adds'] == state['removes'] == 1


def test_scoped_observer_restores_after_exception_and_rejects_nested():
    _, traci, control = fake_runtime(); original = control.step
    with pytest.raises(RuntimeError):
        with StepObserver(control, traci, jobs(AUDIT)[0]):
            with pytest.raises(ValueError):
                with StepObserver(control, traci, jobs(AUDIT)[0]): pass
            raise RuntimeError('model error')
    assert control.step is original and control.traci is traci


def test_episode_dependency_injection_does_not_modify_previous_globals(monkeypatch):
    _, traci, control = fake_runtime()
    monkeypatch.setitem(sys.modules,'control',control); monkeypatch.setitem(sys.modules,'traci',traci)
    @contextmanager
    def session(): yield 'original-settings'
    namespace = {'upstream_session':session, 'control':control}
    exec('def episode():\n    with upstream_session() as s:\n        control.step()\n        return s\n',namespace)
    original = namespace['upstream_session']
    result, sidecar = instrument_episode(namespace['episode'],session,(),jobs(AUDIT)[0])
    assert result == 'original-settings' and len(sidecar['steps']) == 1
    assert namespace['upstream_session'] is original


@pytest.mark.parametrize('fault', [None,'reward','action','frame','observation','time'])
def test_exact_parity_ignores_only_unrecorded_historical_observations(fault):
    keys = ('arrival','reported_collision','time_limit','steps','simulation_duration_s','native_score',
            'native_mean_abs_jerk','policy_sha256','predictor_sha256','common_initial_traffic_sha256')
    old = dict.fromkeys(keys,1); row = deepcopy(old)
    a = [{'frame':{'x':1},'requested_jerk':1.,'commanded_speed':7.,'time_feature':0.,'observation':[1.]}]
    b = deepcopy(a)
    if fault == 'reward': row['native_score'] = 2
    elif fault == 'action': a[0]['requested_jerk'] = 2.
    elif fault == 'frame': a[0]['frame']['x'] = 2
    elif fault == 'observation': a[0]['observation'] = [2.]
    elif fault == 'time': a[0]['time_feature'] = .001
    if fault:
        with pytest.raises(ValueError): assert_reference_parity(row,a,old,b)
    else:
        b[0]['observation'] = None
        p = assert_reference_parity(row,a,old,b)
        assert p['historical_observations_compared'] == 0 and p['exact_available_trajectory_parity']


def test_negative_events_complete_without_method_effect_gate():
    rows = [{**j,'arrival':1,'reported_collision':0,'parity':{'exact_native_outcome_parity':True,
        'exact_available_trajectory_parity':True}, 'independent_events':{'classification':'ego_collision',
        'native_arrival_with_ego_collision':True}} for j in jobs(AUDIT)]
    a = aggregate(AUDIT,rows); assert a['status'] == 'complete' and a['method_effect_gate'] is None
    assert sum(c['native_arrival_with_ego_collision'] for m in a['matrix'].values() for c in m.values()) == 12
    with pytest.raises(ValueError): aggregate(AUDIT,rows[:-1])


def test_resume_preserves_incomplete_result_without_retry(tmp_path,monkeypatch):
    from prediction_rl.data.collection_store import write_once
    j = jobs(AUDIT)[0]; q = {'jobs':[j],'protocol':AUDIT}
    directory = tmp_path/'evaluate'/j['id']; directory.mkdir(parents=True)
    write_once(directory/'started.json',{'status':'incomplete'})
    monkeypatch.setattr(runner,'ROOT',tmp_path); monkeypatch.setattr(runner,'load',lambda *a:(tmp_path/'request.json',q))
    monkeypatch.setattr(runner.shared,'metadata',lambda:{'git_commit':'a'*40})
    monkeypatch.setattr(runner.subprocess,'run',lambda *a,**k:pytest.fail('Must not retry incomplete episode'))
    with pytest.raises(ValueError,match='Incomplete/failed'): runner.execute(tmp_path/'request.json','frozen',True)
    assert (directory/'started.json').exists()
