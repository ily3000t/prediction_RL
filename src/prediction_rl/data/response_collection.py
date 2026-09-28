"""Read-only future geometry sidecars; original observations/rewards stay intact."""
from prediction_rl.data.branching import rollout
from prediction_rl.data.merge_geometry import read_extended_traffic


BASE_FIELDS = ('position', 'speed', 'acceleration', 'lane_id')


def base_traffic(frame):
    return {'simulation_time_s': frame['simulation_time_s'],
            'vehicles': {actor: {key: state[key] for key in BASE_FIELDS}
                         for actor, state in frame['vehicles'].items()}}


def validate_future(trace, frames, geometry):
    if not trace or len(trace) != len(frames):
        raise ValueError('Future metadata must cover exactly the observed transitions')
    for index, (step, frame) in enumerate(zip(trace, frames)):
        if step['done'] and index != len(trace) - 1:
            raise ValueError('No metadata may follow a terminal transition')
        if base_traffic(frame) != step['info']['execution_audit']['after']:
            raise ValueError('Future metadata is not from the audited transition')
        geometry.validate_frame(frame, require_ego=not step['done'])
    # Kept separate from history and model inputs. Terminal geometry is audit-only.
    return {'contract': 'lane_position_length_width_v1', 'frames': frames,
            'trajectory_usable': [not step['done'] for step in trace]}


def branch_with_geometry(manager, root, actions, geometry, *, reader=read_extended_traffic):
    env = manager.restore(root)
    trace, frames = [], []
    for action in actions:
        step = rollout(env, [action])[0]
        trace.append(step)
        frames.append(reader())
        if step['done']:
            break
    return trace, validate_future(trace, frames, geometry)


def root_accounting(targets):
    """Unavailable predeclared roots are explicit, never replaced or discarded."""
    ids = [row['target']['id'] for row in targets]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError('Expected distinct predeclared root targets')
    rows = []
    for row in targets:
        steps = row['prefix_steps']
        if steps is None:
            if not row.get('reason'):
                raise ValueError('Unavailable root requires a reason')
            rows.append({'target': row['target']['id'], 'status': 'unavailable',
                         'reason': row['reason'], 'candidate_count': 0})
        elif type(steps) is not int or steps < 0 or len(row['prefix_actions']) != steps:
            raise ValueError('Invalid reference prefix')
        else:
            rows.append({'target': row['target']['id'], 'status': 'pending', 'candidate_count': 0})
    return rows
