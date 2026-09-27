from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'tools'))
from prediction_rl.data.dataset_contract import episode_id, validate_split_manifest, validate_rows, build_branch_labels, SPLITS


def traffic(t, actors=('ego', 'neighbor')):
    return {'simulation_time_s': t, 'vehicles': {
        actor: {'position': [t+10, 2.0], 'speed': 7.0, 'acceleration': 0.0, 'lane_id': 'main'} for actor in actors}}


def frame(before, after, done=False, reason=None, jerk=0):
    return {'done': done, 'reward': 0., 'info': {'execution_audit': {
        'before': before, 'after': after, 'requested_jerk': jerk,
        'simulation_elapsed_s': after['simulation_time_s']-before['simulation_time_s'],
        'termination_reason': reason, 'execution_contract': 'simulation_blocking_exact_v1',
        'commanded_speed_reconstructed': 7., 'upstream_reward_projected_jerk': jerk,
        'invalid_action_reward': 0., 'measured_post_step_speed': None if done else 7.}}}


def test_full_horizon_negative_events_are_known():
    a, b, c = traffic(0), traffic(.2), traffic(.4)
    labels = build_branch_labels(a, [frame(a,b), frame(b,c)], [0,0], .2)
    assert labels['events']['upstream_collision'] == {'value': 0, 'mask': True}
    assert labels['full_time_horizon_observed']
    assert all(all(row) for row in labels['future_mask'])
    assert labels['future'][0][0][0] == pytest.approx(.2)


@pytest.mark.parametrize('reason,event', [('upstream_collision', 'upstream_collision'), ('upstream_arrival','upstream_arrival')])
def test_early_terminal_keeps_event_masks_unknown_future(reason,event):
    a, b = traffic(0), traffic(.2, ('neighbor',))
    labels = build_branch_labels(a, [frame(a,b,True,reason)], [0,0], .2)
    assert labels['events'][event] == {'value': 1, 'mask': True}
    other = 'upstream_arrival' if event == 'upstream_collision' else 'upstream_collision'
    assert labels['events'][other] == {'value': 0, 'mask': False}
    assert labels['future_mask'] == [[False,False],[False,False]]
    assert labels['transition_mask'] == [True,False]
    assert labels['measured_post_step_speed'] == [None,None]


def test_missing_actor_not_zero_motion_and_new_actor_not_input():
    a, b = traffic(0), traffic(.2, ('ego','new'))
    labels = build_branch_labels(a, [frame(a,b)], [0], .2)
    assert labels['actor_ids'] == ['ego','neighbor']
    assert labels['future_mask'] == [[True],[False]]
    assert labels['new_actor_ids'] == ['new']


def test_truncated_nonterminal_rejected():
    a,b=traffic(0),traffic(.2)
    with pytest.raises(ValueError, match='Incomplete'):
        build_branch_labels(a,[frame(a,b)],[0,0],.2)


@pytest.mark.parametrize('kind', ['plan','time','root','nan','done','reason','contract'])
def test_invalid_trace_rejected(kind):
    a,b=traffic(0),traffic(.2)
    f=frame(a,b)
    audit=f['info']['execution_audit']
    if kind=='plan': audit['requested_jerk']=1
    if kind=='time': audit['after']['simulation_time_s']=.3
    if kind=='root': audit['before']=traffic(-.2)
    if kind=='nan': audit['after']['vehicles']['ego']['speed']=float('nan')
    if kind=='done': f['done']=1
    if kind=='reason': audit['termination_reason']='upstream_collision'
    if kind=='contract': audit['execution_contract']='soft_realtime'
    with pytest.raises(ValueError): build_branch_labels(traffic(0),[f],[0],.2)


def test_frames_after_terminal_rejected():
    a,b,c=traffic(0),traffic(.2),traffic(.4)
    with pytest.raises(ValueError):
        build_branch_labels(a,[frame(a,b,True,'upstream_collision'),frame(b,c)],[0,0],.2)


def test_original_timeout_extra_tick_is_censored():
    a,b=traffic(0),traffic(.4,('neighbor',))
    labels=build_branch_labels(a,[frame(a,b,True,'upstream_time_limit')],[0],.2)
    assert not labels['full_time_horizon_observed']
    assert not labels['events']['upstream_collision']['mask']
    assert not any(labels['future_mask'][0])


def test_episode_split_groups_all_roots_and_candidates():
    ep=episode_id('a'*64,100,0)
    manifest={s:[] for s in SPLITS}; manifest['development']=[ep]
    rows=[{'episode_id':ep,'root_id':r,'candidate_id':c,'split':'development'} for r in ['r0','r1'] for c in range(5)]
    validate_rows(rows,manifest)
    rows[-1]['split']='test'
    with pytest.raises(ValueError): validate_rows(rows,manifest)
    manifest['test']=[ep]
    with pytest.raises(ValueError): validate_split_manifest(manifest)


def test_repeated_branch_not_new_sample():
    ep=episode_id('a'*64,0,0)
    manifest={s:[] for s in SPLITS};manifest['train']=[ep]
    row={'episode_id':ep,'root_id':'r','candidate_id':0,'split':'train'}
    with pytest.raises(ValueError):validate_rows([row,row],manifest)


def test_environment_seed_episode_define_split_unit():
    assert episode_id('a'*64,0,0)==episode_id('a'*64,0,0)
    assert episode_id('a'*64,0,0)!=episode_id('a'*64,1,0)
    assert episode_id('a'*64,0,0)!=episode_id('a'*64,0,1)


@pytest.mark.parametrize('change', [{'split':'train'}, {'training_ready':True}, {'tick_s':True}, {'extends':'x'}, {'source_report':'../report.json'}])
def test_audit_config_never_promotes_development_data(tmp_path,change):
    import json
    from audit_dataset_labels import load_config
    config=json.loads((ROOT/'configs/development/p04_label_audit_v1.json').read_text())
    config.update(change)
    path=tmp_path/'config.json';path.write_text(json.dumps(config))
    with pytest.raises(ValueError):load_config(path)
