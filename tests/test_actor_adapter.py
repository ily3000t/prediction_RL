from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from prediction_rl.data.merge_geometry import MergeGeometry
from prediction_rl.data.actor_adapter import build_actor_inputs, MandatoryOverflow


@pytest.fixture
def geometry():
    return MergeGeometry(ROOT/'RL-MPC-LaneMerging-master/merge.net.xml')


def actor(lane='highwayrear_0',pos=100.,speed=7.):
    return {'position':[pos,0.], 'lane_id':lane, 'lane_position_m':pos,
            'speed':speed,'acceleration':0.,'length_m':5.,'width_m':1.8}


def frame(t=0.,vehicles=None):
    return {'simulation_time_s':t,'vehicles':vehicles or {'ego':actor()}}


def test_merge_chain_coordinate_continuity(geometry):
    for lane,internal in [('ramp_0',':mergenode_1_0'),('highwayrear_0',':mergenode_0_0')]:
        assert geometry.progress(actor(lane,geometry.lengths[lane])) == pytest.approx(geometry.progress(actor(internal,0)))
        assert geometry.progress(actor(internal,geometry.lengths[internal])) == 0
    assert geometry.progress(actor('highwayahead_0',0)) == 0


def test_same_path_gap_uses_front_bumper_and_leader_length(geometry):
    a,b=actor(pos=100,speed=10),actor(pos=120,speed=5)
    pair=geometry.pair(a,b)
    assert pair['same_path_gap_m']==15
    assert pair['cv_ttc_s']==3
    assert geometry.pair(a,actor(pos=80,speed=15))['cv_ttc_s']==3


def test_cross_path_eta_is_not_ttc(geometry):
    pair=geometry.pair(actor('ramp_0',100),actor(pos=100))
    assert pair['same_path_gap_m'] is None and pair['cv_ttc_s'] is None
    assert pair['cross_path_eta_difference_s'] is not None


def test_empty_neighbors_and_padding_masks(geometry):
    x=build_actor_inputs([frame()],geometry)
    assert np.shape(x['actor_features'])==(12,11,10)
    assert np.shape(x['summary_features'])==(3,11,7)
    assert not any(x['actor_mask']) and not any(x['summary_mask'])
    assert x['ego_history_mask']==[False]*10+[True]
    assert not np.asarray(x['actor_features']).any()
    assert not np.asarray(x['summary_feature_mask']).any()


def test_root_actor_absent_in_past_is_masked(geometry):
    past=frame();root=frame(.2,{'ego':actor(),'n':actor(pos=120)})
    x=build_actor_inputs([past,root],geometry)
    slot=x['actor_ids'].index('n')
    assert x['actor_history_mask'][slot][-2:]==[False,True]
    assert x['actor_features'][slot][-2]==[0.]*10


def test_history_actor_gone_at_root_never_becomes_slot(geometry):
    past=frame(0,{'ego':actor(),'gone':actor(pos=120)})
    x=build_actor_inputs([past,frame(.2)],geometry)
    assert 'gone' not in x['actor_ids']
    assert x['summary_history_mask'][1][-2:] == [True,False]
    assert not x['summary_mask'][1]


def test_mandatory_front_rear_preserved_and_overflow_refused(geometry):
    root=frame(0,{'ego':actor(':mergenode_1_0',20),
                  'front':actor(':mergenode_0_0',40),'rear':actor(':mergenode_0_0',10)})
    x=build_actor_inputs([root],geometry)
    assert x['mandatory_roles']['target_front']=='front'
    assert x['mandatory_roles']['target_rear']=='rear'
    with pytest.raises(MandatoryOverflow):build_actor_inputs([root],geometry,capacity=1)


def test_actor_order_invariance_and_omission_counts(geometry):
    vehicles={'ego':actor(':mergenode_1_0',20)}
    vehicles.update({f'n{i:02d}':actor(pos=50+i*7) for i in range(16)})
    root=frame(0,vehicles)
    x=build_actor_inputs([root],geometry)
    y=build_actor_inputs([frame(0,dict(reversed(list(vehicles.items()))))],geometry)
    assert x==y
    assert sum(x['actor_mask'])==12 and len(x['root_omitted_ids'])==4
    assert x['summary_features'][1][-1][0]==4
    assert x['summary_features'][1][-1][-1]==1
    assert not x['summary_feature_mask'][1][-1][1]  # Cross-path gap unavailable.
    assert not x['summary_feature_mask'][1][-1][2]


def test_vehicle_id_breaks_exact_ties(geometry):
    x=build_actor_inputs([frame(0,{'ego':actor(),'z':actor(pos=120),'a':actor(pos=120)})],geometry)
    assert x['mandatory_roles']['target_front']=='a'


@pytest.mark.parametrize('change',[
    {'lane_id':'unknown'}, {'lane_position_m':1000}, {'speed':float('nan')},
    {'length_m':0}, {'width_m':-1}, {'speed':-1},
])
def test_invalid_geometry_refuses_to_silently_drop_actor(geometry,change):
    other=actor();other.update(change)
    with pytest.raises(ValueError):build_actor_inputs([frame(0,{'ego':actor(),'n':other})],geometry)


@pytest.mark.parametrize('times',[[.2,0],[0,0],[0,.4]])
def test_bad_history_time_grid_rejected(geometry,times):
    with pytest.raises(ValueError):build_actor_inputs([frame(t) for t in times],geometry)


def test_same_root_inputs_independent_of_future_candidates(geometry):
    history=[frame()]
    before=deepcopy(history)
    first=build_actor_inputs(history,geometry)
    # API has no future labels, candidate outcome, or candidate plan argument.
    assert build_actor_inputs(history,geometry)==first
    assert history==before


def test_full_history_has_no_padding(geometry):
    x=build_actor_inputs([frame(i*.2) for i in range(11)],geometry)
    assert all(x['ego_history_mask'])
    assert x['root_time_s']==2.
