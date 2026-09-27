"""Root-only physical slots and masked omitted-actor history summaries."""
import math
import numpy as np

from prediction_rl.data.merge_geometry import GROUPS

FEATURES = ('relative_x_m','relative_y_m','speed_mps','acceleration_mps2','merge_end_progress_m',
            'length_m','width_m','is_ramp','is_main_approach','is_main_downstream')
SUMMARY_FEATURES = ('count','min_same_path_gap_m','min_cv_ttc_s','min_merge_end_eta_s',
                    'mean_relative_speed_mps','max_abs_relative_speed_mps','capacity_overflow')


class MandatoryOverflow(ValueError):
    pass


def select_actors(root, geometry, capacity):
    geometry.validate_frame(root)
    ego = root['vehicles']['ego']
    metrics = {key: geometry.pair(ego,value) for key,value in root['vehicles'].items() if key != 'ego'}
    roles = {}
    def choose(role, keys, metric):
        keys = list(keys)
        if keys:
            roles[role] = min(keys, key=lambda key: (metric(metrics[key]), key))
    main = [key for key in metrics if geometry.group(root['vehicles'][key]) != 'ramp']
    choose('target_front', (key for key in main if metrics[key]['progress_delta_m'] >= 0),
           lambda m:m['progress_delta_m'])
    choose('target_rear', (key for key in main if metrics[key]['progress_delta_m'] < 0),
           lambda m:-m['progress_delta_m'])
    choose('nearest_cross_path_arrival', (key for key,m in metrics.items() if m['cross_path_eta_difference_s'] is not None),
           lambda m:m['cross_path_eta_difference_s'])
    choose('lowest_same_path_ttc', (key for key,m in metrics.items() if m['cv_ttc_s'] is not None), lambda m:m['cv_ttc_s'])
    choose('earliest_merge_end_arrival', (key for key,m in metrics.items() if m['merge_end_eta_s'] is not None),
           lambda m:m['merge_end_eta_s'])
    selected = list(dict.fromkeys(roles.values()))
    if len(selected) > capacity:
        raise MandatoryOverflow(f'{len(selected)} mandatory actors exceed capacity {capacity}')
    nullable = lambda value: math.inf if value is None else value
    ordered = sorted(metrics, key=lambda key: (
        nullable(metrics[key]['merge_end_eta_s']), nullable(metrics[key]['cv_ttc_s']),
        abs(metrics[key]['progress_delta_m']), metrics[key]['front_bumper_distance_m'], key))
    selected += [key for key in ordered if key not in selected][:capacity-len(selected)]
    return selected, roles, metrics


def build_actor_inputs(history, geometry, capacity=12, history_steps=11, tick_s=.2):
    if type(capacity) is not int or not 1 <= capacity <= 64:
        raise ValueError('Invalid fixed neighbor capacity')
    if type(history_steps) is not int or not 1 <= history_steps <= 64 or not math.isfinite(tick_s) or tick_s <= 0:
        raise ValueError('Invalid history shape/time contract')
    if not isinstance(history, list) or not 1 <= len(history) <= history_steps:
        raise ValueError('History must be a nonempty bounded chronological list')
    for index, frame in enumerate(history):
        geometry.validate_frame(frame)
        if index and not math.isclose(frame['simulation_time_s']-history[index-1]['simulation_time_s'], tick_s, rel_tol=0, abs_tol=1e-8):
            raise ValueError('History gap, duplicate time or future leakage')
    root = history[-1]
    selected, roles, metrics = select_actors(root, geometry, capacity)
    origin = root['vehicles']['ego']['position']
    actor = np.zeros((capacity,history_steps,len(FEATURES)), np.float32)
    actor_valid = np.zeros((capacity,history_steps), bool)
    ego = np.zeros((history_steps,len(FEATURES)), np.float32)
    ego_valid = np.zeros(history_steps, bool)
    summary = np.zeros((len(GROUPS),history_steps,len(SUMMARY_FEATURES)), np.float32)
    summary_valid = np.zeros((len(GROUPS),history_steps), bool)
    summary_features_valid = np.zeros(summary.shape, bool)
    def encode(state):
        return [state['position'][0]-origin[0],state['position'][1]-origin[1],state['speed'],
                state['acceleration'],geometry.progress(state),state['length_m'],state['width_m'],
                *[float(geometry.group(state)==group) for group in GROUPS]]
    for offset, frame in enumerate(history, history_steps-len(history)):
        vehicles = frame['vehicles']
        ego[offset], ego_valid[offset] = encode(vehicles['ego']), True
        for slot, actor_id in enumerate(selected):
            if actor_id in vehicles:
                actor[slot,offset], actor_valid[slot,offset] = encode(vehicles[actor_id]), True
        for group_index, group in enumerate(GROUPS):
            omitted = [key for key in sorted(vehicles) if key != 'ego' and key not in selected
                       and geometry.group(vehicles[key]) == group]
            if not omitted:
                continue
            pairs = [geometry.pair(vehicles['ego'],vehicles[key]) for key in omitted]
            relative = [vehicles[key]['speed']-vehicles['ego']['speed'] for key in omitted]
            summary_valid[group_index,offset] = True
            row = [float(len(omitted)),0.,0.,0.,float(np.mean(relative)),max(abs(v) for v in relative),
                   float(len(vehicles)-1 > capacity)]
            masks = [True,False,False,False,True,True,True]
            for col, name in enumerate(('same_path_gap_m','cv_ttc_s','merge_end_eta_s'),1):
                observed = [pair[name] for pair in pairs if pair[name] is not None]
                if observed:
                    row[col], masks[col] = min(observed), True
            summary[group_index,offset] = row
            summary_features_valid[group_index,offset] = masks
    if not all(np.isfinite(array).all() for array in (actor,ego,summary)):
        raise ValueError('Nonfinite encoded inputs')
    return {'schema_version':1, 'actor_ids':selected+[None]*(capacity-len(selected)),
            'mandatory_roles':roles, 'root_actor_metrics':metrics,
            'root_omitted_ids':sorted(set(root['vehicles'])-{'ego'}-set(selected)),
            'actor_features':actor.tolist(), 'actor_history_mask':actor_valid.tolist(),
            'actor_mask':actor_valid[:,-1].tolist(), 'ego_features':ego.tolist(),
            'ego_history_mask':ego_valid.tolist(), 'summary_features':summary.tolist(),
            'summary_history_mask':summary_valid.tolist(), 'summary_mask':summary_valid[:,-1].tolist(),
            'summary_feature_mask':summary_features_valid.tolist(),
            'feature_names':list(FEATURES), 'summary_feature_names':list(SUMMARY_FEATURES),
            'summary_groups':list(GROUPS), 'history_steps':history_steps, 'neighbor_capacity':capacity,
            'tick_s':tick_s, 'root_time_s':root['simulation_time_s'], 'ego_separate':True}
