from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'tests')]
from test_masked_training import batch, compare_tree, fixture_data
from prediction_rl.prediction.candidate_difference import (
    DIFFERENCE_LOSS_CONTRACT, candidate_difference_objective, difference_training,
    response_difference_loss,
)
from prediction_rl.prediction.epoch_training import MemberTrainer, subset, evaluate, evaluate_difference
from prediction_rl.prediction.interface import PredictorConfig
from prediction_rl.prediction.losses import trajectory_loss
from prediction_rl.prediction.model import ResponsePredictor
from prediction_rl.prediction.supervision import align_supervision


@pytest.fixture(autouse=True, scope='module')
def threads():
    previous = torch.get_num_threads(); torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def tensors():
    target = torch.zeros(2, 5, 2, 3, 4)
    mask = torch.ones(target.shape[:-1], dtype=torch.bool)
    scale = torch.tensor([10., 3., 5., 2.])
    prediction = torch.arange(5.).reshape(1, 5, 1, 1, 1)*scale
    return prediction.expand_as(target).clone(), target, mask


def test_complete_mask_five_candidate_variance_identity():
    p, y, m = tensors()
    absolute, _ = trajectory_loss(p, y, m)
    difference, coverage = response_difference_loss(p, y, m)
    # Candidate errors 0,1,2,3,4: mean squared=4, variance=2.
    assert absolute.item() == pytest.approx(6.)
    assert difference.item() == pytest.approx(2.5*2.)
    assert coverage['valid_candidate_pairs'] == 20
    assert coverage['pair_common_actor_time_cells'] == 120


def test_common_error_not_removed_from_absolute_anchor():
    p, y, m = tensors()
    p[:] = torch.tensor([10., 3., 5., 2.])
    difference, _ = response_difference_loss(p, y, m)
    objective, metrics = candidate_difference_objective(p, y, m, .2)
    assert difference == 0 and objective == 1
    assert metrics['trajectory_loss'] == 1.


def test_zero_true_response_is_not_forced_apart_and_wrong_sign_is_penalized():
    p, y, m = tensors()
    p.zero_()
    assert response_difference_loss(p, y, m)[0] == 0
    y.copy_(tensors()[0])
    assert response_difference_loss(y, y, m)[0] == 0
    assert response_difference_loss(-y, y, m)[0] > response_difference_loss(p, y, m)[0]


def test_intersection_mask_ignores_disappearance_and_has_no_invalid_gradient():
    p, y, m = tensors()
    m[:, 1:, :, 1:] = False
    p.requires_grad_()
    first, _ = response_difference_loss(p, y, m)
    changed = p.detach().clone(); changed[~m] = 99999.
    y2 = y.clone(); y2[~m] = -99999.
    second, _ = response_difference_loss(changed, y2, m)
    assert torch.equal(first, second)
    first.backward()
    assert torch.count_nonzero(p.grad[~m]) == 0


def test_no_common_pair_keeps_absolute_supervised_root_in_denominator():
    p, y, m = tensors()
    m[1] = False; m[1, 0, 0, 0] = True
    objective, metrics = candidate_difference_objective(p, y, m, .2)
    assert metrics['roots_with_valid_pairs'] == metrics['roots_without_valid_pairs'] == 1
    assert metrics['difference_root_denominator'] == 2
    assert metrics['difference_loss'] == pytest.approx(2.5)
    assert objective.item() == pytest.approx(metrics['trajectory_loss']+.2*2.5)
    m[0] = False
    p.requires_grad_()
    difference, coverage = response_difference_loss(p, y, m)
    difference.backward()
    assert difference == 0 and torch.count_nonzero(p.grad) == 0
    assert coverage['roots_without_valid_pairs'] == 1


def test_pair_average_not_duration_or_pair_count_weighted():
    p = torch.zeros(2, 3, 1, 2, 4); y = torch.zeros_like(p)
    m = torch.zeros(p.shape[:-1], dtype=torch.bool)
    scale = torch.tensor([10., 3., 5., 2.])
    p[0, 1] = scale; p[0, 2] = 3*scale; m[0] = True
    m[0, 2, :, 1:] = False
    p[1, 1] = 2*scale; m[1, :2] = True
    value, _ = response_difference_loss(p, y, m)
    assert value.item() == pytest.approx(((1+9+4)/3+4)/2)


@pytest.mark.parametrize('weight', [-1, float('nan'), float('inf'), True, '0.2', None])
def test_invalid_weight_refused(weight):
    with pytest.raises(ValueError, match='difference_lambda'):
        candidate_difference_objective(*tensors(), weight)


def test_empty_and_nonfinite_labels_remain_explicit_errors():
    p, y, m = tensors()
    with pytest.raises(ValueError, match='No supervised'):
        candidate_difference_objective(p, y, torch.zeros_like(m), .2)
    m[0, 0, 0, 0] = False; y[0, 0, 0, 0, 0] = float('nan')
    with pytest.raises(ValueError, match='Nonfinite'):
        candidate_difference_objective(p, y, m, .2)


def test_zero_lambda_exact_scalar_and_gradient():
    p, y, m = tensors(); m[1, 3:] = False
    p.requires_grad_(); q = p.detach().clone().requires_grad_()
    old, _ = trajectory_loss(p, y, m)
    new, _ = candidate_difference_objective(q, y, m, 0.)
    old.backward(); new.backward()
    assert torch.equal(old, new) and torch.equal(p.grad, q.grad)


def test_label_and_candidate_permutation_keeps_id_alignment():
    cfg = PredictorConfig(hidden_dim=16)
    histories, packs, manifest = fixture_data(cfg)
    a = align_supervision(histories, packs, manifest, cfg, split='development')
    for c in packs[0]['candidates']:
        for key in ('actor_ids', 'future', 'future_mask'): c['labels'][key].reverse()
    packs[0]['candidates'].reverse()
    b = align_supervision(histories, packs, manifest, cfg, split='development')
    p = a.targets.clone()+torch.randn_like(a.targets)
    assert torch.equal(response_difference_loss(p, a.targets, a.target_mask)[0],
                       response_difference_loss(p, b.targets, b.target_mask)[0])


def setup_training():
    cfg = PredictorConfig(hidden_dim=16)
    b = subset(batch(cfg), [0, 0, 0])
    b = replace(b, root_ids=('r0','r1','r2'), episode_ids=('e0','e1','e2'), split='train')
    b.targets[:, 1:] += torch.arange(1., 5.).view(1, 4, 1, 1, 1)
    v = replace(deepcopy(b), root_ids=('v0','v1','v2'), episode_ids=('v0','v1','v2'), split='validation')
    t = json.loads((ROOT/'configs/formal/p04_response_dataset_v1.json').read_text())['training']
    t.update(max_epochs=2, early_stop_patience=2, batch_roots=2)
    return cfg, t, b, v


def test_zero_lambda_exact_epoch_parameters_adam_rng_and_selection():
    cfg, t, b, v = setup_training()
    old = MemberTrainer(cfg, 'conditional', 4101, t, b, v)
    new = MemberTrainer(cfg, 'conditional', 4101, difference_training(t, 0.), b, v)
    for _ in range(2):
        a, c = old.run_epoch(), new.run_epoch()
        assert {k:c[k] for k in a} == a
        compare_tree(old.model.state_dict(), new.model.state_dict())
        compare_tree(old.optimizer.state_dict(), new.optimizer.state_dict())
        compare_tree(old.selection, new.selection)
        assert torch.equal(old.generator.get_state(), new.generator.get_state())
        assert torch.equal(old.cpu_rng, new.cpu_rng)


def test_new_training_checkpoint_resume_exact_and_old_identity_refused(tmp_path):
    cfg, t, b, v = setup_training(); delta = difference_training(t, .2)
    a = MemberTrainer(cfg, 'conditional', 4101, delta, b, v)
    a.run_epoch(); path = tmp_path/'e1.pt'; a.save(path, {'run':'delta'})
    resumed = MemberTrainer.load(path, cfg, 'conditional', 4101, delta, b, v, {'run':'delta'})
    assert a.run_epoch() == resumed.run_epoch()
    compare_tree(a.model.state_dict(), resumed.model.state_dict())
    compare_tree(a.optimizer.state_dict(), resumed.optimizer.state_dict())
    with pytest.raises(ValueError, match='identity/contract'):
        MemberTrainer.load(path, cfg, 'conditional', 4101, t, b, v, {'run':'delta'})
    with pytest.raises(ValueError, match='identity/contract'):
        MemberTrainer.load(path, cfg, 'conditional', 4101, difference_training(t, .4), b, v, {'run':'delta'})


def test_selection_remains_mse_and_objective_metrics_root_weighted():
    cfg, t, b, v = setup_training()
    trainer = MemberTrainer(cfg, 'conditional', 4101, difference_training(t, .2), b, v)
    row = trainer.run_epoch()
    assert row['validation_loss'] == evaluate(trainer.model, v, 2)
    assert row['validation_objective_loss'] == pytest.approx(row['validation_loss']+.2*row['validation_difference_loss'])
    individual = [evaluate_difference(trainer.model, subset(v, [i]), 1, .2) for i in range(3)]
    aggregate = evaluate_difference(trainer.model, v, 2, .2)
    for key in ('trajectory_loss','difference_loss','objective_loss'):
        assert aggregate[key] == pytest.approx(sum(x[key] for x in individual)/3, rel=1e-6)


def test_ordinary_pair_objective_has_zero_gradient_and_training_is_refused():
    cfg, t, b, v = setup_training()
    model = ResponsePredictor(cfg, 'ordinary')
    difference, _ = response_difference_loss(model(b.observations, b.plans), b.targets, b.target_mask)
    difference.backward()
    assert difference > 0
    # Shared-output gradient cancellation can leave float32 accumulation noise.
    assert all(p.grad is not None and torch.allclose(p.grad, torch.zeros_like(p.grad), atol=1e-7, rtol=0)
               for p in model.parameters())
    with pytest.raises(ValueError, match='conditional'):
        MemberTrainer(cfg, 'ordinary', 4101, difference_training(t, .2), b, v)


def test_only_declared_settings_change():
    _, t, _, _ = setup_training(); new = difference_training(t, .2)
    assert {k for k in t if t[k]!=new[k]} == {'methods','loss_contract'}
    assert set(new)-set(t) == {'difference_lambda'}
    assert new['loss_contract'] == DIFFERENCE_LOSS_CONTRACT['version']
