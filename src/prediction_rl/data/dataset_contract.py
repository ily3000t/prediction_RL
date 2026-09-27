"""Episode-group splits and masked branch labels; no learned-model inputs yet."""
import hashlib
import json
import math

SPLITS = ('development', 'train', 'validation', 'calibration', 'test')
TERMINATIONS = {'upstream_collision', 'upstream_arrival', 'upstream_time_limit', 'registered_time_limit'}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                     separators=(',', ':')).encode()).hexdigest()


def episode_id(environment_hash, simulator_seed, episode_index):
    if (not isinstance(environment_hash, str) or len(environment_hash) != 64
            or any(c not in '0123456789abcdef' for c in environment_hash)):
        raise ValueError('Expected a frozen environment SHA256')
    if type(simulator_seed) is not int or not 0 <= simulator_seed < 2**32:
        raise ValueError('Invalid simulator seed')
    if type(episode_index) is not int or episode_index < 0:
        raise ValueError('Invalid episode index')
    # Reference controller, candidate and root time deliberately NOT included.
    return digest([environment_hash, simulator_seed, episode_index])


def validate_split_manifest(manifest):
    if not isinstance(manifest, dict) or set(manifest) != set(SPLITS):
        raise ValueError('Explicit development/train/validation/calibration/test lists required')
    assignments = {}
    for split, episodes in manifest.items():
        if not isinstance(episodes, list):
            raise ValueError('Split entries must be episode ID lists')
        for episode in episodes:
            if not isinstance(episode, str) or len(episode) != 64 or any(c not in '0123456789abcdef' for c in episode):
                raise ValueError('Invalid episode ID')
            if episode in assignments:
                raise ValueError('Duplicate episode across/within splits')
            assignments[episode] = split
    return assignments


def validate_rows(rows, manifest):
    assignments = validate_split_manifest(manifest)
    root_episodes, seen = {}, set()
    for row in rows:
        ep, root, candidate = row['episode_id'], row['root_id'], row['candidate_id']
        if assignments.get(ep) != row['split'] or ep not in assignments:
            raise ValueError('Row split not assigned by frozen episode manifest')
        if root in root_episodes and root_episodes[root] != ep:
            raise ValueError('One root cannot belong to multiple episodes')
        root_episodes[root] = ep
        key = (ep, root, candidate)
        if key in seen:
            raise ValueError('Duplicate candidate; repeats are verification, not new data')
        seen.add(key)


def finite(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError('Expected a finite numeric measurement')
    return float(value)


def validate_traffic(traffic):
    finite(traffic['simulation_time_s'])
    if not isinstance(traffic['vehicles'], dict):
        raise ValueError('Expected vehicle dictionary')
    for actor_id, state in traffic['vehicles'].items():
        if not isinstance(actor_id, str) or not actor_id:
            raise ValueError('Invalid actor ID')
        position = state['position']
        if not isinstance(position, list) or len(position) != 2:
            raise ValueError('Expected absolute XY position')
        for value in [*position, state['speed'], state['acceleration']]:
            finite(value)
        if not isinstance(state['lane_id'], str):
            raise ValueError('Expected lane ID')


def build_branch_labels(root_traffic, trace, requested_plan, tick_s):
    """Root actors only; future membership must never select input actors.

    Terminal-frame coordinates are conservatively masked (upstream can remove
    ego on that frame). Events retain the observed original terminal reason.
    Missing trajectories and censored events are not negative/zero targets.
    """
    tick_s = finite(tick_s)
    if tick_s <= 0 or not isinstance(requested_plan, list) or not requested_plan:
        raise ValueError('Positive tick and complete requested jerk plan required')
    plan = [finite(v) for v in requested_plan]
    horizon = len(plan)
    if not isinstance(trace, list) or not 1 <= len(trace) <= horizon:
        raise ValueError('Nonempty trace must not exceed the frozen horizon')
    validate_traffic(root_traffic)
    if 'ego' not in root_traffic['vehicles']:
        raise ValueError('Root must contain ego')
    actors = ['ego', *sorted(set(root_traffic['vehicles']) - {'ego'})]
    origin = root_traffic['vehicles']['ego']['position']
    values = [[[0.0] * 4 for _ in range(horizon)] for _ in actors]
    masks = [[False] * horizon for _ in actors]
    transition_mask = [False] * horizon
    commands, projected, invalid_rewards, measured_speeds = ([None] * horizon for _ in range(4))
    reason, previous = None, root_traffic
    for step, frame in enumerate(trace):
        if type(frame['done']) is not bool:
            raise ValueError('Expected Boolean terminal flag')
        audit = frame['info']['execution_audit']
        if audit['execution_contract'] != 'simulation_blocking_exact_v1':
            raise ValueError('Unexpected execution contract')
        if audit['before'] != previous:
            raise ValueError('Discontinuous or wrong-root branch trace')
        after = audit['after']
        validate_traffic(after)
        delta = finite(after['simulation_time_s']) - finite(previous['simulation_time_s'])
        reason = audit['termination_reason']
        if frame['done']:
            if step != len(trace) - 1 or reason not in TERMINATIONS:
                raise ValueError('Invalid terminal tail or reason')
        elif reason is not None:
            raise ValueError('Nonterminal frame has terminal reason')
        timeout_extra = frame['done'] and reason == 'upstream_time_limit' and math.isclose(delta, 2*tick_s, abs_tol=1e-8)
        if (not math.isclose(delta, tick_s, abs_tol=1e-8) and not timeout_extra
                or not math.isclose(finite(audit['simulation_elapsed_s']), delta, abs_tol=1e-8)):
            raise ValueError('Unexpected simulator time grid')
        if not math.isclose(finite(audit['requested_jerk']), plan[step], rel_tol=0, abs_tol=1e-6):
            raise ValueError('Requested action differs from declared candidate plan')
        finite(frame['reward'])
        transition_mask[step] = True
        commands[step] = finite(audit['commanded_speed_reconstructed'])
        projected[step] = finite(audit['upstream_reward_projected_jerk'])
        invalid_rewards[step] = finite(audit['invalid_action_reward'])
        measured_speeds[step] = (None if audit['measured_post_step_speed'] is None
                                 else finite(audit['measured_post_step_speed']))
        if not frame['done']:
            for index, actor in enumerate(actors):
                state = after['vehicles'].get(actor)
                if state is not None:
                    values[index][step] = [state['position'][0] - origin[0],
                                           state['position'][1] - origin[1],
                                           state['speed'], state['acceleration']]
                    masks[index][step] = True
        previous = after
    if len(trace) < horizon and not trace[-1]['done']:
        raise ValueError('Incomplete nonterminal rollout is an error, not censoring')
    covered = len(trace) == horizon and math.isclose(
        previous['simulation_time_s'] - root_traffic['simulation_time_s'], horizon*tick_s, abs_tol=1e-8)
    events = {}
    for event in ('upstream_collision', 'upstream_arrival'):
        observed = reason == event
        events[event] = {'value': int(observed), 'mask': observed or covered}
    return {'actor_ids': actors, 'feature_names': ['root_ego_relative_x_m', 'root_ego_relative_y_m',
                                                'speed_mps', 'acceleration_mps2'],
            'future': values, 'future_mask': masks, 'transition_mask': transition_mask,
            'requested_plan': plan, 'commanded_speed_reconstructed': commands,
            'upstream_reward_projected_jerk': projected, 'invalid_action_reward': invalid_rewards,
            'measured_post_step_speed': measured_speeds, 'events': events,
            'termination_reason': reason, 'observed_steps': len(trace),
            'full_time_horizon_observed': covered,
            'new_actor_ids': sorted(set().union(*(set(f['info']['execution_audit']['after']['vehicles'])
                                                  for f in trace)) - set(actors))}
