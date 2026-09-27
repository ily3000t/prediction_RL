from copy import deepcopy
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from prediction_rl.data.actor_adapter import build_actor_inputs
from prediction_rl.data.merge_geometry import MergeGeometry
from prediction_rl.prediction.interface import PredictorConfig, collate_inputs, probe_plans
from prediction_rl.prediction.model import PredictorEnsemble
from prediction_rl.prediction.checkpoint import save_checkpoint, load_checkpoint


@pytest.fixture(autouse=True, scope='module')
def threads():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


@pytest.fixture
def cfg():
    return PredictorConfig(hidden_dim=16, attention_heads=4)


def inputs(neighbors=2):
    def actor(pos):
        return {'position': [pos, 0.], 'lane_id': 'highwayrear_0', 'lane_position_m': pos,
                'speed': 7., 'acceleration': 0., 'length_m': 5., 'width_m': 1.8}
    history = [{'simulation_time_s': 0., 'vehicles': {'ego': actor(90)}},
               {'simulation_time_s': .2, 'vehicles': {'ego': actor(91), **{f'n{i}': actor(100+i*4) for i in range(neighbors)}}}]
    geometry = MergeGeometry(ROOT / 'RL-MPC-LaneMerging-master/merge.net.xml')
    return build_actor_inputs(history, geometry)


def model(cfg, mode='conditional'):
    return PredictorEnsemble(cfg, mode, [71, 72, 73]).eval()


def test_config_strict_fields_and_bounds():
    d = PredictorConfig().to_dict()
    assert PredictorConfig.from_dict(d) == PredictorConfig()
    with pytest.raises(ValueError): PredictorConfig.from_dict({**d, 'unknown': 1})
    with pytest.raises(ValueError): PredictorConfig(ensemble_size=1)
    with pytest.raises(ValueError): PredictorConfig(tick_s=.1)
    with pytest.raises(ValueError): PredictorConfig(hidden_dim=True)


def test_rng_preserved_and_members_distinct(cfg):
    before = torch.get_rng_state().clone()
    net = model(cfg)
    assert torch.equal(before, torch.get_rng_state())
    weights = [next(m.parameters()) for m in net.members]
    assert all(weights[0].data_ptr() != x.data_ptr() for x in weights[1:])
    # First parameter is zero-initialized ego role; use learned response weights.
    assert not torch.equal(net.members[0].response_head[0].weight, net.members[1].response_head[0].weight)
    with pytest.raises(ValueError): PredictorEnsemble(cfg, 'conditional', [1, 1, 2])


@pytest.mark.parametrize('mode', ['ordinary', 'conditional'])
@pytest.mark.parametrize('count', [0, 2, 20])
def test_masks_empty_history_overflow_and_shape(cfg, mode, count):
    batch = collate_inputs([inputs(count)], cfg)
    output = model(cfg, mode)(batch, probe_plans(1, cfg))
    assert output['member_trajectories'].shape == (3, 1, 5, 12, 25, 4)
    assert torch.isfinite(output['mean']).all()
    assert (output['disagreement'] >= 0).all()
    invalid = ~batch['actor_mask'][0]
    assert torch.count_nonzero(output['member_trajectories'][:, 0, :, invalid]) == 0


@pytest.mark.parametrize('mode', ['ordinary', 'conditional'])
def test_masked_finite_garbage_cannot_change_prediction(cfg, mode):
    batch = collate_inputs([inputs(2)], cfg)
    other = {k: v.clone() for k, v in batch.items()}
    for value, mask in [('actor_features','actor_history_mask'), ('ego_features','ego_history_mask'),
                        ('summary_features','summary_feature_mask')]:
        valid = other[mask]
        if valid.ndim < other[value].ndim: valid = valid.unsqueeze(-1).expand_as(other[value])
        other[value][~valid] = 99999
    net, plans = model(cfg, mode), probe_plans(1, cfg)
    assert torch.equal(net(batch, plans)['mean'], net(other, plans)['mean'])


def test_b2_never_receives_plan_values_and_b3_does(cfg):
    batch, plans = collate_inputs([inputs()], cfg), probe_plans(1, cfg)
    captured = []
    ordinary, conditional = model(cfg, 'ordinary'), model(cfg)
    hook = ordinary.members[0].plan_encoder.register_forward_pre_hook(lambda _, args: captured.append(args[0].detach()))
    out2, out3 = ordinary(batch, plans)['mean'], conditional(batch, plans)['mean']
    hook.remove()
    assert torch.count_nonzero(captured[0]) == 0
    assert torch.equal(out2[:, :1].expand_as(out2), out2)
    assert not torch.allclose(out3[:, 0], out3[:, -1])
    assert sum(p.numel() for p in ordinary.parameters()) == sum(p.numel() for p in conditional.parameters())
    altered = plans.clone(); altered[:, :, 0] *= -.5
    assert torch.equal(out2, ordinary(batch, altered)['mean'])


@pytest.mark.parametrize('mode', ['ordinary', 'conditional'])
def test_candidate_batch_and_scene_batch_match_serial(cfg, mode):
    batch = collate_inputs([inputs(), inputs(20)], cfg)
    net, plans = model(cfg, mode), probe_plans(2, cfg)
    calls = []
    hook = net.members[0].scene_encoder.register_forward_hook(lambda *args: calls.append(1))
    output = net(batch, plans)['member_trajectories']
    hook.remove()
    assert len(calls) == 1
    serial = torch.cat([net(batch, plans[:, i:i+1])['member_trajectories'] for i in range(5)], 2)
    torch.testing.assert_close(output, serial, rtol=1e-5, atol=1e-5)
    root_serial = torch.cat([net({k:v[i:i+1] for k,v in batch.items()}, plans[i:i+1])['member_trajectories'] for i in range(2)], 1)
    torch.testing.assert_close(output, root_serial, rtol=1e-5, atol=1e-5)


def test_actor_permutation_equivariance(cfg):
    batch = collate_inputs([inputs(20)], cfg)
    perm = torch.arange(11, -1, -1)
    other = {k: v[:, perm] if k.startswith('actor_') else v for k,v in batch.items()}
    net, plans = model(cfg), probe_plans(1, cfg)
    expected = net(batch, plans)['mean'][:, :, perm]
    torch.testing.assert_close(expected, net(other, plans)['mean'], rtol=1e-5, atol=1e-5)


def test_backward_reaches_b3_plan_and_summary_without_training(cfg):
    batch, plans = collate_inputs([inputs(20)], cfg), probe_plans(1, cfg).requires_grad_()
    net = model(cfg)
    net(batch, plans)['mean'].square().mean().backward()
    assert plans.grad[:, :, 0].abs().sum() > 0
    for member in net.members:
        assert member.scene_encoder.summary_history.cell.weight_ih.grad.abs().sum() > 0
        assert member.plan_encoder[0].weight.grad[:, 0].abs().sum() > 0
    ordinary = model(cfg, 'ordinary')
    plans2 = plans.detach().clone().requires_grad_()
    ordinary(batch, plans2)['mean'].sum().backward()
    assert plans2.grad is None


def test_reject_nan_invalid_mask_bad_metadata_extra_tensor(cfg):
    row = inputs()
    bad = deepcopy(row); bad['actor_features'][11][0][0] = float('nan')
    with pytest.raises(ValueError, match='Nonfinite'): collate_inputs([bad], cfg)
    bad = deepcopy(row); bad['actor_mask'] = [int(x) for x in bad['actor_mask']]
    with pytest.raises(ValueError, match='booleans'): collate_inputs([bad], cfg)
    bad = deepcopy(row); bad['feature_names'][0] = 'future_x'
    with pytest.raises(ValueError, match='metadata'): collate_inputs([bad], cfg)
    batch = collate_inputs([row], cfg); batch['future_labels'] = torch.ones(1)
    with pytest.raises(ValueError, match='fixed observation'): model(cfg)(batch, probe_plans(1,cfg))
    batch = collate_inputs([row], cfg); batch['actor_mask'][0, 11] = True
    with pytest.raises(ValueError, match='disagreement'): model(cfg)(batch, probe_plans(1,cfg))


@pytest.mark.parametrize('index,value', [(0, 6.), (1, 1.), (0, float('nan'))])
def test_invalid_plans_rejected_in_both_modes(cfg, index, value):
    batch, plans = collate_inputs([inputs()], cfg), probe_plans(1,cfg)
    plans[0, 0, index] = value
    for mode in ('ordinary', 'conditional'):
        with pytest.raises(ValueError): model(cfg, mode)(batch, plans)


@pytest.mark.parametrize('mode', ['ordinary', 'conditional'])
def test_checkpoint_exact_roundtrip_and_no_overwrite(cfg, tmp_path, mode):
    net, path = model(cfg, mode), tmp_path / 'model.pt'
    save_checkpoint(path, net, {'training_status':'untrained_interface_smoke','seed':[71,72,73]})
    loaded, provenance = load_checkpoint(path, expected_mode=mode, expected_config=cfg)
    batch, plans = collate_inputs([inputs(20)],cfg), probe_plans(1,cfg)
    assert torch.equal(net(batch,plans)['member_trajectories'], loaded(batch,plans)['member_trajectories'])
    assert provenance['training_status'] == 'untrained_interface_smoke'
    with pytest.raises(FileExistsError): save_checkpoint(path,net,provenance)
    with pytest.raises(ValueError): load_checkpoint(path, expected_mode='wrong', expected_config=cfg)


@pytest.mark.parametrize('mutation', ['missing_member','feature_order','nan','fixed_scale','shape','dtype'])
def test_checkpoint_corruption_fails_closed(cfg, tmp_path, mutation):
    path = tmp_path / 'model.pt'
    save_checkpoint(path,model(cfg),{'training_status':'untrained_interface_smoke'})
    payload = torch.load(path,weights_only=True)
    state = payload['state_dict']
    key = 'members.0.plan_encoder.0.weight'
    if mutation == 'missing_member': state.pop(key)
    elif mutation == 'feature_order': payload['contract']['features'].reverse()
    elif mutation == 'nan': state[key][0,0] = float('nan')
    elif mutation == 'fixed_scale': state['members.0.scene_encoder.physical_scale'][0] = 1
    elif mutation == 'shape': state[key] = state[key][:-1]
    elif mutation == 'dtype': state[key] = state[key].double()
    changed = tmp_path / 'changed.pt'; torch.save(payload,changed)
    with pytest.raises(ValueError): load_checkpoint(changed, expected_mode='conditional', expected_config=cfg)
