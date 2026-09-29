"""Root-only lane-chain constant-speed reference; no future or candidate input."""
import math
import xml.etree.ElementTree as ET

import torch

from prediction_rl.data.merge_geometry import MergeGeometry

NEXT={'ramp_0':':mergenode_1_0',':mergenode_1_0':'highwayahead_0',
      'highwayrear_0':':mergenode_0_0',':mergenode_0_0':'highwayahead_0'}


class LaneConstantVelocity:
    def __init__(self,network):
        self.geometry=MergeGeometry(network)
        self.points={x.attrib['id']:[tuple(map(float,p.split(','))) for p in x.attrib['shape'].split()]
                     for x in ET.parse(network).getroot().findall('./edge/lane')}
        self.arc={k:[math.dist(a,b) for a,b in zip(v,v[1:])] for k,v in self.points.items()}
        if any(not v or min(v)<=0 for v in self.arc.values()): raise ValueError('Invalid map polyline')

    def xy(self,lane,position):
        if lane not in self.points or not math.isfinite(position) or position<0: raise ValueError('Invalid route position')
        while position>self.geometry.lengths[lane] and lane in NEXT:
            position-=self.geometry.lengths[lane];lane=NEXT[lane]
        # SUMO lane length can differ slightly from rounded polyline arc length.
        distance=position/self.geometry.lengths[lane]*sum(self.arc[lane])
        points=self.points[lane]; lengths=self.arc[lane]
        for i,length in enumerate(lengths):
            if distance<=length or i==len(lengths)-1:
                a,b=points[i],points[i+1];fraction=distance/length
                return (a[0]+fraction*(b[0]-a[0]),a[1]+fraction*(b[1]-a[1]))
            distance-=length
        raise AssertionError('Unreachable polyline state')

    def predict(self,histories,config):
        result=torch.zeros(len(histories),5,config.neighbor_capacity,config.horizon_steps,4)
        for i,h in enumerate(histories):
            # Only observed root/history metadata is accepted, never label packs.
            root=h['history'][-1]; self.geometry.validate_frame(root);ego=root['vehicles']['ego']['position']
            for k,actor in enumerate(h['inputs']['actor_ids']):
                if actor is None: continue
                state=root['vehicles'][actor];lane=state['lane_id'];s=state['lane_position_m'];v=state['speed']
                anchor=self.xy(lane,s)
                for t in range(config.horizon_steps):
                    x,y=self.xy(lane,s+v*(t+1)*config.tick_s)
                    value=[state['position'][0]+x-anchor[0]-ego[0],state['position'][1]+y-anchor[1]-ego[1],v,0.]
                    result[i,:,k,t]=torch.tensor(value)
        return result
