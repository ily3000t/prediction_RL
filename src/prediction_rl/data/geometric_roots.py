"""Result-independent lane-position roots along a fixed reference trajectory."""
from copy import deepcopy
import math
import re

import numpy as np

from prediction_rl.envs.upstream import traffic_snapshot


def validate_targets(targets):
    if not isinstance(targets, list) or not 1 <= len(targets) <= 3:
        raise ValueError('Use one to three geometric root targets')
    ids, lanes = set(), set()
    for target in targets:
        if not isinstance(target, dict) or set(target) != {'id', 'lane_id', 'reference', 'value'}:
            raise ValueError('Unknown/missing geometric target field')
        if not isinstance(target['id'], str) or not re.fullmatch(r'[a-z][a-z0-9_]{0,30}', target['id']):
            raise ValueError('Use a short target identifier')
        if not isinstance(target['lane_id'], str) or not target['lane_id']:
            raise ValueError('lane_id must be a nonempty string')
        if target['id'] in ids or target['lane_id'] in lanes:
            raise ValueError('Duplicate root IDs/lanes are not allowed')
        ids.add(target['id'])
        lanes.add(target['lane_id'])
        if target['reference'] not in ('from_start_m', 'before_end_m', 'fraction'):
            raise ValueError('Unknown lane-position reference')
        value = target['value']
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError('Target position must be finite and nonnegative')
        if target['reference'] == 'fraction' and value > 1:
            raise ValueError('Lane fraction must be in [0,1]')


def target_position(target, length):
    if not math.isfinite(length) or length <= 0:
        raise ValueError('Invalid lane length')
    value = target['value']
    position = {'from_start_m': lambda: value,
                'before_end_m': lambda: length - value,
                'fraction': lambda: length * value}[target['reference']]()
    if not 0 <= position <= length:
        raise ValueError('Geometric target lies outside the configured lane')
    return position


def read_ego_progress():
    import traci
    lane = traci.vehicle.getLaneID('ego')
    return {'lane_id': lane, 'lane_position_m': float(traci.vehicle.getLanePosition('ego')),
            'lane_length_m': float(traci.lane.getLength(lane)), 'traffic': traffic_snapshot()}


def scan_lane_roots(env, targets, max_steps, jerk, read_progress=read_ego_progress):
    """First qualifying nonterminal state per target. No probe results as input.

    env has already been reset. The original prefix is not altered at targets.
    Terminal states and unreachable targets are recorded, never replaced.
    """
    validate_targets(targets)
    if type(max_steps) is not int or not 0 <= max_steps <= 250:
        raise ValueError('Discovery budget must be 0..250 steps')
    selected = {}
    visited = []
    stop_reason = 'discovery_budget_exhausted'
    for step in range(max_steps + 1):
        progress = read_progress()
        if not visited or progress['lane_id'] != visited[-1]:
            visited.append(progress['lane_id'])
        for target in targets:
            if target['id'] in selected or progress['lane_id'] != target['lane_id']:
                continue
            position = target_position(target, progress['lane_length_m'])
            if progress['lane_position_m'] >= position:
                selected[target['id']] = {
                    'target': deepcopy(target), 'prefix_steps': step,
                    'selection': deepcopy(progress), 'target_position_m': position,
                    'overshoot_m': progress['lane_position_m'] - position,
                    'selection_rule': 'first_nonterminal_lane_threshold_crossing_v1'}
        if len(selected) == len(targets):
            stop_reason = 'all_targets_reached'
            break
        if step == max_steps:
            break
        _, _, done, info = env.step(np.array([jerk], dtype=env.action_space.dtype))
        if done:
            stop_reason = 'episode_terminated_before_remaining_targets'
            break
    return {
        'targets': [selected.get(target['id'], {
            'target': deepcopy(target), 'prefix_steps': None, 'reason': stop_reason,
            'selection_rule': 'first_nonterminal_lane_threshold_crossing_v1'})
                    for target in targets],
        'visited_lane_ids': visited, 'stop_reason': stop_reason,
    }
