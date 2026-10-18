import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('rcci_review',ROOT/'reports/scripts/review_rcci02_v2_full_v1.py')
review = importlib.util.module_from_spec(spec); spec.loader.exec_module(review)


def row(episode,ordinary,conditional):
    record = {m:{'errors':{'front_neighbor_only_min_gap_m':value}} for m,value in
              (('ordinary',ordinary),('conditional',conditional))}
    return {'episode_id':episode,'branches':[{'following':record,
        'fixed_horizons':{'1.0':{'methods':record},'5.0':{'methods':{}}}}]}


def test_scalar_availability_intersection_not_different_candidate_denominators():
    result = review.paired_gap([row('e',[1.],[2.]),row('e',[],[100.])],'front')
    assert result['paired_candidates'] == 1 and result['ordinary_mae_m'] == 1
    assert result['conditional_mae_m'] == 2


def test_episode_equal_weights_not_root_equal_weights():
    result = review.paired_gap([row('one',[1.],[2.])]*3+[row('two',[5.],[8.])],'front')
    assert result['paired_candidates'] == 4 and result['episodes'] == 2
    assert result['ordinary_mae_m'] == pytest.approx(2.)
    assert result['ordinary_equal_episode_mae_m'] == pytest.approx(3.)
    assert result['conditional_equal_episode_mae_m'] == pytest.approx(5.)


def test_zero_is_observed_value_missing_is_not_zero():
    result = review.paired_gap([row('e',[0.],[0.]),row('absent',[],[])],'front','1.0')
    assert result['paired_candidates'] == 1 and result['ordinary_mae_m'] == 0
    absent = review.paired_gap([row('e',[1.],[2.])],'front','5.0')
    assert absent['paired_candidates'] == 0 and absent['ordinary_mae_m'] is None


def test_identity_check_rejects_failure():
    with pytest.raises(ValueError): review.require(False)
