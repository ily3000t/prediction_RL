from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
from prediction_rl.evaluation.rcci_feasibility import (
    AuditMap, AuditAccumulator, CONTRACT, audit_branch, summarize, distribution)
import audit_rcci_feasibility as cli
import prediction_rl.data.frozen_dataset as frozen


@pytest.fixture
def geometry():return AuditMap(ROOT/'RL-MPC-LaneMerging-master/merge.net.xml')


def frame(g,t=0.,s=180.):
    def state(lane,position):
        return {'position':g.lane_xy(lane,position).tolist(),'lane_id':lane,
                'lane_position_m':position,'length_m':5.,'width_m':1.8,'speed':10.,'acceleration':0.}
    return {'simulation_time_s':t,'vehicles':{'ego':state('ramp_0',s),'other':state('highwayrear_0',s)}}


def branch(g,*,terminal=False):
    frames=[frame(g,.2,182.),frame(g,.4,184.)]
    steps=[{'done':False,'info':{'execution_audit':{'after':deepcopy(f)}}} for f in frames]
    if terminal:steps[-1]['done']=True
    return {'trace':steps,'future_metadata':{'frames':frames,'trajectory_usable':[not s['done'] for s in steps]}}


def test_independent_explicit_configuration():
    assert json.loads((ROOT/'configs/development/rcci01_feasibility_v1.json').read_text())==CONTRACT
    assert CONTRACT['splits']==['train','validation'] and not CONTRACT['predictor_inference']
    assert CONTRACT['projection_acceptance_threshold'] is None


@pytest.mark.parametrize('lane',['ramp_0',':mergenode_1_0','highwayrear_0',':mergenode_0_0','highwayahead_0'])
def test_known_lane_roundtrip_including_endpoints(geometry,lane):
    for s in (0.,geometry.geometry.lengths[lane]*.37,geometry.geometry.lengths[lane]):
        p=geometry.project(geometry.lane_xy(lane,s),[lane])
        assert p['distance_m']==pytest.approx(0,abs=1e-10)
        assert p['lane_position_m']==pytest.approx(s,abs=1e-8)
        assert not p['ambiguous_progress']


@pytest.mark.parametrize('xy,lanes', [([float('nan'),0],['ramp_0']),([0,0],[]),([0,0],['unknown']),([0,0],['ramp_0','ramp_0'])])
def test_invalid_projection_not_turned_into_missing_or_safe(geometry,xy,lanes):
    with pytest.raises(ValueError):geometry.project(xy,lanes)


def test_offroad_and_endpoint_clamps_remain_diagnostic(geometry):
    p=geometry.project([150.,40.],['highwayahead_0'])
    assert p['distance_m']>50 and p['endpoint_clamped']
    assert p['diagnostic_only_no_acceptance_threshold']


def test_projection_respects_explicit_lane_whitelist(geometry):
    point=geometry.lane_xy(':mergenode_1_0',20.)
    p=geometry.project(point,[':mergenode_0_0'])
    assert p['lane']==':mergenode_0_0' and p['distance_m']>0


def test_common_rigid_transform_invariance_of_adapter_only(geometry):
    moved=deepcopy(geometry); angle=.71
    rot=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
    translation=np.array([103.,-72.])
    for k,p in moved.points.items():moved.points[k]=p@rot.T+translation
    original=frame(geometry); transformed=deepcopy(original)
    for state in transformed['vehicles'].values():state['position']=(np.array(state['position'])@rot.T+translation).tolist()
    for a,b in zip(geometry.audit_frame(original),moved.audit_frame(transformed)):
        for key in ('lane_position_xy_error_m','known_lane_projection_error_m','projected_lane_position_error_m'):
            assert a[key]==pytest.approx(b[key],abs=1e-8)


def test_terminal_geometry_excluded_not_fabricated(geometry):
    b=branch(geometry,terminal=True); b['future_metadata']['frames'][-1]['vehicles'].pop('ego')
    b['trace'][-1]['info']['execution_audit']['after']['vehicles'].pop('ego')
    result=audit_branch(b,frame(geometry),geometry)
    assert result['horizon_censored'] and result['nonterminal_frames']==1 and len(result['states'])==2


@pytest.mark.parametrize('fault',['time','mask','base','after_terminal'])
def test_branch_inconsistency_invalidates_audit(geometry,fault):
    b=branch(geometry)
    if fault=='time':b['future_metadata']['frames'][0]['simulation_time_s']=.3
    elif fault=='mask':b['future_metadata']['trajectory_usable'][0]=False
    elif fault=='base':b['trace'][0]['info']['execution_audit']['after']['vehicles']['ego']['speed']=11.
    else:
        b['trace'][0]['done']=True;b['future_metadata']['trajectory_usable'][0]=False
    with pytest.raises(ValueError):audit_branch(b,frame(geometry),geometry)


def test_exact_streaming_statistics_without_state_dump(geometry):
    row={'episode_id':'same','root_states':geometry.audit_frame(frame(geometry)),
         'omitted_actors':1,'ego_history_fraction':.5,'branches':[audit_branch(branch(geometry),frame(geometry),geometry)]}
    acc=AuditAccumulator();acc.add(row);acc.add(row)
    report=acc.report(CONTRACT['diagnostic_distance_bins_m'])
    assert report==summarize([row,row],CONTRACT['diagnostic_distance_bins_m'])
    assert report['unique_episodes']==1 and report['roots']==2
    assert report['heading_recorded_states']==0 and not report['native_footprint_ground_truth_available']
    assert report['observed_relation_counts']['cross_approach']==4
    assert not report['event_extraction_implemented'] and not report['forecast_projection_evaluated']
    assert report['sparse_transition_counts_are_not_missed_conflict_events']


def test_histograms_do_not_drop_zero_or_large_values():
    result=distribution([0.,.001,1.,100.],[.001,1.,2.])
    assert sum(h['count'] for h in result['histogram'])==4
    assert result['max']==100.
    with pytest.raises(ValueError):distribution([float('nan')],[1.])


def test_cohort_includes_empty_roots_without_outcome_filter():
    review={'manifest':{'train':['a'],'validation':['b'],'test':['NEVER_OPEN']},
            'roots':{'train':[{'root_id':'t'+str(i),'episode_id':'a','eligible_trajectory_root':False} for i in range(4)],
                     'validation':[{'root_id':'v'+str(i),'episode_id':'b'} for i in range(4)],'test':None}}
    rows=cli.select_roots(review,'bounded')
    assert len(rows)==6 and [r['root_id'] for r in rows[:3]]==['t0','t1','t2']
    assert len(cli.select_roots(review,'full'))==8
    with pytest.raises(ValueError):cli.select_roots(review,'test')
    review['manifest']['validation']=['a']
    with pytest.raises(ValueError):cli.select_roots(review,'full')


def test_unapproved_config_rejected_before_any_dataset_read(tmp_path,monkeypatch):
    monkeypatch.setattr(cli,'git',lambda root,*args:cli.BASE if args[0]=='rev-parse' else '')
    bad=deepcopy(CONTRACT);bad['splits'].append('test')
    monkeypatch.setattr(cli,'read',lambda p:pytest.fail('Opened dataset before rejecting config'))
    with pytest.raises(ValueError):cli.inputs(tmp_path,bad,'full')


def test_missing_or_nonfinite_geometry_is_not_legitimate_mask(geometry):
    f=frame(geometry);f['vehicles']['other']['width_m']=float('nan')
    with pytest.raises(ValueError):geometry.audit_frame(f)
    f=frame(geometry);f['vehicles']['ego']['heading_rad']=float('nan')
    with pytest.raises(ValueError):geometry.audit_frame(f)


def test_worktree_map_accepts_only_git_line_endings_not_semantic_changes(tmp_path,monkeypatch):
    baseline=tmp_path/'base';work=tmp_path/'work';baseline.mkdir();work.mkdir()
    (baseline/'map.xml').write_bytes(b'<map/>\r\n');(work/'map.xml').write_bytes(b'<map/>\n')
    monkeypatch.setattr(cli,'ROOT',work)
    monkeypatch.setattr(frozen.subprocess,'check_output',lambda *a,**k:b'<map/>\n')
    assert cli.verify_map_identity(baseline,'map.xml')==cli.sha(work/'map.xml')
    (work/'map.xml').write_bytes(b'<map changed="yes"/>\n')
    with pytest.raises(ValueError):cli.verify_map_identity(baseline,'map.xml')
