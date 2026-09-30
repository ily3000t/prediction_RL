import ast
from argparse import Namespace
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'tools')]
from test_masked_training import fixture_data
from prediction_rl.prediction.interface import PredictorConfig, probe_plans
from prediction_rl.prediction.supervision import align_supervision
from prediction_rl.prediction.kinematic_reference import LaneConstantVelocity
from prediction_rl.prediction.candidate_ranking import (CONTROL, FIELDS, command_speed, rollout_ego,
    ego_labels, parity_metrics, score_metrics, ranking_rows, summarize_ranking)
import diagnose_candidate_ranking as cli


@pytest.fixture(autouse=True, scope='module')
def threads():
    old = torch.get_num_threads(); torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


@pytest.fixture
def geometry():
    return LaneConstantVelocity(ROOT/'RL-MPC-LaneMerging-master/merge.net.xml')


def example(geometry):
    cfg = PredictorConfig(hidden_dim=16); h, packs, manifest = fixture_data(cfg)
    ego = rollout_ego(h[0]['history'][-1], probe_plans(1, cfg)[0], geometry)
    for i, candidate in enumerate(packs[0]['candidates']):
        candidate['labels']['future'][0] = ego[i].tolist()
    b = align_supervision(h, packs, manifest, cfg, split='development')
    return h, packs, manifest, b


def test_command_matches_original_helper_without_importing_simulator():
    source = (ROOT/'RL-MPC-LaneMerging-master/control.py').read_text(encoding='utf-8')
    fn = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == 'get_ego_speed_from_jerk')
    module = ast.Module(body=[fn], type_ignores=[]); scope = {'Settings': SimpleNamespace(**CONTROL)}
    exec(compile(ast.fix_missing_locations(module), '<isolated original helper>', 'exec'), scope)
    for speed in (0., .01, 7., 29.99, 30.):
        for acceleration in (-9., -6., 0., 4.5):
            for jerk in (-5., -2.5, 0., 2.5, 5.):
                assert command_speed(speed, acceleration, jerk) == scope[fn.name](speed, acceleration, jerk)


@pytest.mark.parametrize('values', [(-1, 0, 0), (31, 0, 0), (1, float('nan'), 0), (1, 0, 6)])
def test_invalid_state_or_action_rejected(values):
    with pytest.raises(ValueError): command_speed(*values)


def test_root_only_ego_has_no_label_channel_and_retains_continuation_acceleration(geometry):
    cfg = PredictorConfig(hidden_dim=16); h, packs, _ = fixture_data(cfg)
    plans = probe_plans(1, cfg)[0]; first = rollout_ego(h[0]['history'][-1], plans, geometry)
    packs[0]['candidates'] = None; h[0]['label_pack'] = 'not_read.json'
    assert torch.equal(first, rollout_ego(h[0]['history'][-1], plans, geometry))
    assert first[4, 0, 2:].tolist() == pytest.approx([7.2, 1.])
    assert first[4, 1, 2:].tolist() == pytest.approx([7.4, 1.])
    assert first[2, 0, 0] == pytest.approx(1.4)
    plans[0, 1] = -5
    with pytest.raises(ValueError, match='continuation'): rollout_ego(h[0]['history'][-1], plans, geometry)


def test_speed_saturation_recurrence_and_lane_crossing(geometry):
    cfg = PredictorConfig(hidden_dim=16); h, _, _ = fixture_data(cfg); root = h[0]['history'][-1]
    ego = root['vehicles']['ego']; ego.update(lane_id=':mergenode_1_0', lane_position_m=52., speed=30., acceleration=4.5)
    result = rollout_ego(root, probe_plans(1, cfg)[0], geometry)
    assert result[4, 0, 2:].tolist() == [30., 0.]
    assert result[4, 1, 2:].tolist() == [30., 0.]
    assert result[4, -1, 0] > result[4, 0, 0]
    assert torch.isfinite(result).all()


def test_proxy_ties_order_and_regret_are_explicit():
    truth = torch.tensor([1., 2., 3., 4., 5.])
    exact = score_metrics(truth, truth)
    assert exact['pair_order_accuracy'] == 1 and exact['chosen_proxy_regret_m'] == 0
    wrong = score_metrics(truth.flip(0), truth)
    assert wrong['pair_order_accuracy'] == 0 and wrong['chosen_proxy_regret_m'] == 4
    tied_prediction = score_metrics(torch.zeros(5), truth)
    assert tied_prediction['chosen_candidate_id'] == 0 and tied_prediction['pair_order_accuracy'] == 0
    no_signal = score_metrics(torch.zeros(5), torch.ones(5))
    assert no_signal['pair_order_accuracy'] is None and no_signal['informative_pairs'] == 0
    assert no_signal['best_proxy_tie_hit_rate'] == 1


def test_exact_geometry_zero_errors_and_same_support(geometry):
    h, packs, manifest, b = example(geometry)
    rows = ranking_rows({'ordinary': b.targets, 'conditional': b.targets}, b, h, packs, geometry)
    assert rows[0]['ego_parity']['gate'] is True and rows[0]['full_5s_all_selected_actors'] is True
    for metric in ('score_mae_m', 'pair_delta_mae_m', 'chosen_proxy_regret_m'):
        assert rows[0]['methods']['conditional'][metric] == 0
    assert rows[0]['common_actor_time_cells'] == 50
    summary = summarize_ranking(rows, manifest['development'], ['ordinary', 'conditional'])
    assert summary['ego_parity_gate'] is True
    assert summary['support_views']['complete_5s_all_selected_actors']['roots'] == 1


def test_future_ego_changes_labels_not_ego_rollout(geometry):
    h, packs, _, b = example(geometry)
    before = rollout_ego(h[0]['history'][-1], b.plans[0], geometry)
    for c in packs[0]['candidates']:
        for state in c['labels']['future'][0]: state[0] += 1.
    after = rollout_ego(h[0]['history'][-1], b.plans[0], geometry)
    assert torch.equal(before, after)
    rows = ranking_rows({'conditional': b.targets}, b, h, packs, geometry)
    assert rows[0]['ego_parity']['gate'] is False
    assert rows[0]['ego_parity']['position_max_m'] == pytest.approx(1.)


def test_common_mask_excludes_tail_in_every_branch_no_imputation(geometry):
    h, packs, manifest, b = example(geometry)
    b.target_mask[0, 0, :, 2:] = False
    first = ranking_rows({'conditional': b.targets}, b, h, packs, geometry)
    assert first[0]['common_actor_time_cells'] == 4 and not first[0]['full_5s_all_selected_actors']
    changed = b.targets.clone(); changed[:, :, :, 2:, :2] = 1e6
    second = ranking_rows({'conditional': changed}, b, h, packs, geometry)
    assert first[0]['methods'] == second[0]['methods']
    assert summarize_ranking(first, manifest['development'], ['conditional'])['support_views']['complete_5s_all_selected_actors']['roots'] == 0


def test_ego_censoring_and_empty_roots_counted_not_safe(geometry):
    h, packs, manifest, b = example(geometry)
    packs[0]['candidates'][0]['labels']['future_mask'][0] = [False]*25
    rows = ranking_rows({'conditional': b.targets}, b, h, packs, geometry)
    assert rows[0]['methods'] == {} and rows[0]['common_actor_time_cells'] == 0
    assert summarize_ranking(rows, manifest['development'], ['conditional'])['empty_support_roots'] == 1
    observed, mask = ego_labels(packs[0]); mask[:] = False
    assert parity_metrics(observed, observed, mask)['gate'] is False


def test_wrong_identity_or_nonfinite_prediction_rejected(geometry):
    h, packs, _, b = example(geometry); changed = b.targets.clone(); changed[0, 0, 0, 0, 0] = float('nan')
    with pytest.raises(ValueError, match='predictions'): ranking_rows({'conditional': changed}, b, h, packs, geometry)
    packs[0]['root_id'] = 'wrong'
    with pytest.raises(ValueError, match='order'): ranking_rows({'conditional': b.targets}, b, h, packs, geometry)


def test_summary_episode_weighting_paired_sign_and_empty_episode(geometry):
    h, packs, _, b = example(geometry)
    base = ranking_rows({'ordinary': b.targets, 'conditional': b.targets}, b, h, packs, geometry)[0]
    rows = []
    for i, (ep, value) in enumerate([('one', 2.), ('one', 2.), ('two', 8.)]):
        row = deepcopy(base); row.update(root_id=str(i), episode_id=ep)
        for name in FIELDS:
            row['methods']['ordinary'][name] = value; row['methods']['conditional'][name] = 0.
        rows.append(row)
    s = summarize_ranking(rows, ['one', 'two', 'empty'], ['ordinary', 'conditional'])['support_views']['all_common_observed_support']
    assert s['methods']['ordinary']['root_equal']['score_mae_m'] == 4
    assert s['methods']['ordinary']['episode_equal']['score_mae_m'] == 5
    assert s['methods']['ordinary']['episode_denominators']['score_mae_m'] == 2
    assert s['paired_differences']['ordinary__to__conditional']['score_mae_m']['episode_mean_right_minus_left'] == -5


def test_manual_confirmation_precedes_ranking(tmp_path, monkeypatch):
    monkeypatch.setattr(cli.shared, 'clean', lambda: None)
    monkeypatch.setattr(cli, 'load_request', lambda _: (tmp_path, {}, None, None, None))
    monkeypatch.setattr(cli, 'evaluate', lambda *a: pytest.fail('Must not evaluate'))
    with pytest.raises(ValueError, match='Confirm'): cli.run(Namespace(request='none', confirm_request_hash='wrong'))
    assert not list(tmp_path.iterdir())


def test_no_overwrite_completed_or_started_diagnostic(tmp_path, monkeypatch):
    monkeypatch.setattr(cli.shared, 'clean', lambda: None)
    monkeypatch.setattr(cli, 'load_request', lambda _: (tmp_path, {}, None, None, None))
    monkeypatch.setattr(cli, 'evaluate', lambda *a: pytest.fail('Must not evaluate'))
    (tmp_path/'started.json').write_text('{}')
    with pytest.raises(ValueError, match='overwrite'):
        cli.run(Namespace(request='none', confirm_request_hash=cli.digest({})))


def test_protocol_file_and_no_future_training_or_calibration_flags():
    assert cli.read(ROOT/'configs/development/p05_candidate_ranking_v1.json') == cli.CONFIG
    assert cli.CONFIG['split'] == 'validation' and cli.CONFIG['ddpg_release'] is False
    assert cli.CONFIG['calibrated_bands_used'] is False and cli.CONFIG['test_release'] is False


def test_prepare_never_calls_models(tmp_path, monkeypatch, geometry):
    _, _, _, b = example(geometry)
    monkeypatch.setattr(cli, 'ROOT', tmp_path); monkeypatch.setattr(cli.shared, 'ROOT', tmp_path)
    monkeypatch.setattr(cli.shared, 'clean', lambda: None)
    monkeypatch.setattr(cli.shared, 'metadata', lambda: {'git_commit': 'a'*40})
    monkeypatch.setattr(cli, 'prerequisites', lambda _: (cli.CONFIG, {}, {}, {}, {}))
    monkeypatch.setattr(cli, 'validation_data', lambda _: (b, [], [], []))
    monkeypatch.setattr(cli, 'evaluate', lambda *a: pytest.fail('Prepare cannot evaluate'))
    cli.write_once(tmp_path/'roots.json', [])
    cli.write_once(tmp_path/'audit.json', {'status': 'complete', 'purpose': 'development_smoke', 'interface_gate': True,
        'source_hashes': cli.shared.source_hashes(), 'environment': cli.shared.env(), 'inputs': {},
        'config_hash': cli.digest(cli.CONFIG), 'roots_sha256': cli.sha(tmp_path/'roots.json'), 'development_evidence_sha256': cli.DEV_SHA})
    cli.prepare(Namespace(config='config.json', audit='audit.json', run_id='test'))
    out = tmp_path/'artifacts/ranking/test'
    assert sorted(p.name for p in out.iterdir()) == ['preparation.json', 'request.json']
    q = cli.read(out/'request.json'); assert q['ddpg_ready'] is False and q['test_locked'] is True


def test_negative_results_complete_without_changing_gate(tmp_path, monkeypatch, geometry):
    _, _, _, b = example(geometry)
    q = {'config': cli.CONFIG, 'data_hash': b.fingerprint()}
    monkeypatch.setattr(cli.shared, 'clean', lambda: None)
    monkeypatch.setattr(cli.shared, 'metadata', lambda: {'git_commit': 'a'*40})
    monkeypatch.setattr(cli, 'load_request', lambda _: (tmp_path, q, {}, {}, {}))
    monkeypatch.setattr(cli, 'validation_data', lambda _: (b, [], [], []))
    monkeypatch.setattr(cli, 'evaluate', lambda *a: ([], {'ego_parity_gate': False, 'negative_result': True}))
    cli.run(Namespace(request='none', confirm_request_hash=cli.digest(q)))
    report = cli.read(tmp_path/'report.json')
    assert report['status'] == 'complete' and report['summary']['ego_parity_gate'] is False
    assert report['ddpg_ready'] is False and report['test_evaluated'] is False
