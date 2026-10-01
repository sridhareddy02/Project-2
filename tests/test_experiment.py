import math

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize, proportions_ztest

from lifecycle_lab import experiment as ex


def test_sample_size_agrees_with_statsmodels_and_hits_the_target_power():
    for p0, mde in [(0.05, 0.01), (0.10, 0.02), (0.30, 0.05)]:
        n = ex.sample_size_per_arm(p0, mde)
        sm_n = NormalIndPower().solve_power(effect_size=proportion_effectsize(p0 + mde, p0), alpha=0.05, power=0.8, alternative="two-sided")
        assert abs(n - sm_n) / sm_n < 0.05
        assert 0.80 <= ex.power_for_n(p0, mde, n) < 0.805


def test_power_rises_with_sample_size_and_mde_inverts_it():
    assert ex.power_for_n(0.05, 0.01, 500) < ex.power_for_n(0.05, 0.01, 5000)
    n = 3000
    mde = ex.minimum_detectable_effect(0.05, n)
    assert ex.power_for_n(0.05, mde, n) == pytest.approx(0.80, abs=1e-3)


def test_two_proportion_test_matches_statsmodels():
    r = ex.two_proportion_test(120, 2000, 90, 2000)
    z, p = proportions_ztest([120, 90], [2000, 2000])
    assert r["z"] == pytest.approx(z) and r["p_value"] == pytest.approx(p)
    assert r["ci_low"] < r["absolute_lift"] < r["ci_high"]
    assert r["relative_lift"] == pytest.approx((0.06 - 0.045) / 0.045)


def test_no_difference_gives_zero_z_and_symmetric_interval():
    r = ex.two_proportion_test(100, 1000, 100, 1000)
    assert r["z"] == 0 and r["p_value"] == pytest.approx(1.0)
    assert abs(r["ci_low"] + r["ci_high"]) < 1e-12


def test_welch_test_matches_scipy():
    rng = np.random.default_rng(0)
    a, b = rng.normal(10.4, 3, 400), rng.normal(10, 5, 300)
    r = ex.welch_mean_test(a, b)
    assert r["p_value"] == pytest.approx(stats.ttest_ind(a, b, equal_var=False).pvalue)
    assert r["ci_low"] < r["difference"] < r["ci_high"]


def test_cuped_reduces_variance_by_about_rho_squared_and_keeps_the_mean():
    rng = np.random.default_rng(1)
    x = rng.normal(100, 20, 20_000)
    y = 0.8 * x + rng.normal(0, 12, 20_000)
    adj, theta, reduction = ex.cuped_adjust(y, x)
    rho2 = np.corrcoef(x, y)[0, 1] ** 2
    assert reduction == pytest.approx(rho2, abs=0.01)
    assert theta == pytest.approx(0.8, abs=0.02)
    assert adj.mean() == pytest.approx(y.mean(), abs=1e-9)


def test_cuped_does_nothing_when_the_covariate_is_unrelated():
    rng = np.random.default_rng(2)
    _, _, reduction = ex.cuped_adjust(rng.normal(0, 1, 5000), rng.normal(0, 1, 5000))
    assert reduction < 0.01


def test_srm_flags_a_broken_split_but_not_a_balanced_one():
    assert ex.srm_check(5000, 5010)["ok"]
    assert not ex.srm_check(5000, 5600)["ok"]


def test_bayesian_probability_is_sensible():
    assert ex.beta_binomial(100, 1000, 100, 1000)["prob_treatment_better"] == pytest.approx(0.5, abs=0.03)
    assert ex.beta_binomial(200, 1000, 100, 1000)["prob_treatment_better"] > 0.999
    r = ex.beta_binomial(130, 1000, 100, 1000)
    assert r["credible_low"] < r["expected_absolute_lift"] < r["credible_high"]


def test_benjamini_hochberg_matches_statsmodels():
    p = np.array([0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, 0.212, 0.216])
    expected = multipletests(p, method="fdr_bh")[1]
    assert np.allclose(ex.benjamini_hochberg(p), expected)
    shuffled = np.random.default_rng(0).permutation(p)
    assert np.allclose(ex.benjamini_hochberg(shuffled), multipletests(shuffled, method="fdr_bh")[1])


def test_the_test_has_the_right_false_positive_rate_coverage_and_power():
    v = ex.validate_methods(p0=0.03, effect=0.015, n_per_arm=2000, reps=4000, seed=3)
    assert 0.04 <= v["aa_false_positive_rate"] <= 0.06
    assert 0.94 <= v["aa_ci_coverage"] <= 0.96
    assert 0.94 <= v["ab_ci_coverage"] <= 0.96
    assert abs(v["ab_simulated_power"] - v["ab_analytic_power"]) < 0.03


def test_heterogeneity_test_separates_equal_from_unequal_effects():
    rng = np.random.default_rng(4)
    n = 20_000
    group = pd.Series(rng.choice(["a", "b", "c"], n))
    treated = rng.random(n) < 0.5
    same = rng.random(n) < (0.05 + 0.02 * treated)
    different = rng.random(n) < (0.05 + np.where(group == "a", 0.0, 0.06) * treated)
    assert ex.heterogeneity_test(same, treated, group)["p_value"] > 0.01
    assert ex.heterogeneity_test(different, treated, group)["p_value"] < 1e-6


def test_winback_experiment_is_internally_consistent(scored):
    r = ex.run_winback_experiment(scored, true_effect=0.02, seed=0)
    assert r["srm"]["ok"] and r["n_treatment"] + r["n_control"] == r["population"]
    assert r["primary"]["ci_low"] < r["primary"]["absolute_lift"] < r["primary"]["ci_high"]
    assert 0 <= r["bayesian"]["prob_treatment_better"] <= 1
    assert r["heterogeneity"]["q_value"].between(0, 1).all()
    assert (r["heterogeneity"]["q_value"] >= r["heterogeneity"]["p_value"] - 1e-12).all()
    assert r["planned_n_per_arm"] > 0 and 0 < r["power_with_available_n"] <= 1
