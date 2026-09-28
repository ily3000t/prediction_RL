"""Pinned upstream lane-chain geometry, not a general collision predictor."""
import math
import xml.etree.ElementTree as ET

from prediction_rl.data.dataset_contract import finite, validate_traffic

LANES = ('ramp_0', ':mergenode_1_0', 'highwayrear_0', ':mergenode_0_0', 'highwayahead_0')
GROUPS = ('ramp', 'main_approach', 'main_downstream')


class MergeGeometry:
    def __init__(self, network):
        net = ET.parse(network).getroot()
        lanes = {node.attrib['id']: node for node in net.findall('./edge/lane')}
        if set(lanes) != set(LANES):
            raise ValueError('Geometry contract supports only the audited five-lane network')
        links = {(x.get('from'), x.get('to'), x.get('via')) for x in net.findall('connection')}
        if not {('ramp','highwayahead',':mergenode_1_0'),
                ('highwayrear','highwayahead',':mergenode_0_0'),
                (':mergenode_1','highwayahead',None),
                (':mergenode_0','highwayahead',None)} <= links:
            raise ValueError('Unexpected lane connectivity')
        self.lengths = {lane: finite(float(node.get('length'))) for lane,node in lanes.items()}
        if min(self.lengths.values()) <= 0:
            raise ValueError('Invalid lane lengths')
        self.groups = dict(zip(LANES, ('ramp','ramp','main_approach','main_approach','main_downstream')))
        self.offsets = {'highwayahead_0': 0., ':mergenode_1_0': -self.lengths[':mergenode_1_0'],
                        ':mergenode_0_0': -self.lengths[':mergenode_0_0']}
        self.offsets['ramp_0'] = self.offsets[':mergenode_1_0'] - self.lengths['ramp_0']
        self.offsets['highwayrear_0'] = self.offsets[':mergenode_0_0'] - self.lengths['highwayrear_0']

    def verify_runtime(self):
        import traci
        for lane, length in self.lengths.items():
            if not math.isclose(traci.lane.getLength(lane), length, rel_tol=0, abs_tol=1e-6):
                raise ValueError('Loaded simulator geometry differs from pinned network')

    def validate_frame(self, frame, *, require_ego=True):
        validate_traffic(frame)
        if require_ego and 'ego' not in frame['vehicles']:
            raise ValueError('Nonterminal history requires ego')
        for state in frame['vehicles'].values():
            lane = state['lane_id']
            if lane not in self.lengths:
                raise ValueError('Unknown lane; do not silently omit actor')
            position = finite(state['lane_position_m'])
            if not -1e-6 <= position <= self.lengths[lane] + 1e-6:
                raise ValueError('Actor lane position outside network bounds')
            if finite(state['length_m']) <= 0 or finite(state['width_m']) <= 0 or state['speed'] < 0:
                raise ValueError('Invalid vehicle extent/speed')

    def progress(self, state):
        return self.offsets[state['lane_id']] + state['lane_position_m']

    def group(self, state):
        return self.groups[state['lane_id']]

    def same_path(self, a, b):
        ga, gb = self.group(a), self.group(b)
        return ga == gb or 'main_downstream' in (ga, gb)

    def eta(self, state):
        distance = -self.progress(state)
        return distance / state['speed'] if distance > 0 and state['speed'] > 1e-6 else None

    def pair(self, ego, other):
        delta = self.progress(other) - self.progress(ego)
        gap = ttc = None
        if self.same_path(ego, other):
            gap = delta - other['length_m'] if delta >= 0 else -delta - ego['length_m']
            closing = ego['speed']-other['speed'] if delta >= 0 else other['speed']-ego['speed']
            if gap <= 0:
                ttc = 0.
            elif closing > 1e-6:
                ttc = gap / closing
        ego_eta, other_eta = self.eta(ego), self.eta(other)
        conflict = (abs(ego_eta-other_eta) if not self.same_path(ego, other)
                    and ego_eta is not None and other_eta is not None else None)
        return {'progress_delta_m': delta, 'same_path_gap_m': gap, 'cv_ttc_s': ttc,
                'merge_end_eta_s': other_eta, 'cross_path_eta_difference_s': conflict,
                'front_bumper_distance_m': math.dist(ego['position'], other['position'])}


def read_extended_traffic():
    """Read-only metadata; base fields remain identical to P2/P3 snapshots."""
    import traci
    from prediction_rl.envs.upstream import traffic_snapshot
    frame = traffic_snapshot()
    for actor, state in frame['vehicles'].items():
        state.update(lane_position_m=float(traci.vehicle.getLanePosition(actor)),
                     length_m=float(traci.vehicle.getLength(actor)),
                     width_m=float(traci.vehicle.getWidth(actor)))
    return frame
