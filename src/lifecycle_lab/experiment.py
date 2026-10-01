"""A/B testing toolkit: planning, analysis, variance reduction, diagnostics and self-validation.

Every statistical function here is checked in ``tests/`` against either a closed form, scipy/statsmodels
or a Monte Carlo simulation, so the numbers in the report can be trusted.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

# ------------------------------------------------------------------ planning


def sample_size_per_arm(p0: float, mde_abs: float, alpha: float = 0.05, power: float = 0.80) -> int:
    """Users needed in each arm to detect an absolute lift of ``mde_abs`` over baseline ``p0`` (two-sided z test)."""
    p1 = p0 + mde_abs
    pbar = (p0 + p1) / 2
    z_a, z_b = stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(power)
    num = (z_a * math.sqrt(2 * pbar * (1 - pbar)) + z_b * math.sqrt(p0 * (1 - p0) + p1 * (1 - p1))) ** 2
    return math.ceil(num / mde_abs**2)


def power_for_n(p0: float, mde_abs: float, n_per_arm: int, alpha: float = 0.05) -> float:
    """Probability of detecting the lift with ``n_per_arm`` users per arm."""
    p1 = p0 + mde_abs
    se_alt = math.sqrt(p0 * (1 - p0) / n_per_arm + p1 * (1 - p1) / n_per_arm)
    pbar = (p0 + p1) / 2
    se_null = math.sqrt(2 * pbar * (1 - pbar) / n_per_arm)
    z_a = stats.norm.ppf(1 - alpha / 2)
    return float(stats.norm.sf((z_a * se_null - mde_abs) / se_alt) + stats.norm.cdf((-z_a * se_null - mde_abs) / se_alt))


def minimum_detectable_effect(p0: float, n_per_arm: int, alpha: float = 0.05, power: float = 0.80) -> float:
    """Smallest absolute lift detectable with the available sample (found by bisection)."""
    lo, hi = 1e-6, 1 - p0 - 1e-6
    for _ in range(80):
        mid = (lo + hi) / 2
        if power_for_n(p0, mid, n_per_arm, alpha) < power:
            lo = mid
        else:
            hi = mid
    return hi


# ------------------------------------------------------------------ analysis


def two_proportion_test(x_t: int, n_t: int, x_c: int, n_c: int, alpha: float = 0.05) -> dict:
    """Difference in proportions: pooled z test for the p-value, unpooled standard error for the interval."""
    p_t, p_c = x_t / n_t, x_c / n_c
    diff = p_t - p_c
    pooled = (x_t + x_c) / (n_t + n_c)
    se_pooled = math.sqrt(pooled * (1 - pooled) * (1 / n_t + 1 / n_c))
    z = diff / se_pooled if se_pooled > 0 else 0.0
    se = math.sqrt(p_t * (1 - p_t) / n_t + p_c * (1 - p_c) / n_c)
    crit = stats.norm.ppf(1 - alpha / 2)
    return {
        "rate_treatment": p_t,
        "rate_control": p_c,
        "absolute_lift": diff,
        "relative_lift": diff / p_c if p_c > 0 else float("nan"),
        "z": z,
        "p_value": float(2 * stats.norm.sf(abs(z))),
        "ci_low": diff - crit * se,
        "ci_high": diff + crit * se,
    }


def welch_mean_test(treatment: np.ndarray, control: np.ndarray, alpha: float = 0.05) -> dict:
    """Welch's t test for a difference in means with a confidence interval."""
    t = stats.ttest_ind(treatment, control, equal_var=False)
    diff = float(np.mean(treatment) - np.mean(control))
    v_t, v_c = np.var(treatment, ddof=1) / len(treatment), np.var(control, ddof=1) / len(control)
    se = math.sqrt(v_t + v_c)
    df = (v_t + v_c) ** 2 / (v_t**2 / (len(treatment) - 1) + v_c**2 / (len(control) - 1))
    crit = stats.t.ppf(1 - alpha / 2, df)
    return {"difference": diff, "se": se, "p_value": float(t.pvalue), "ci_low": diff - crit * se, "ci_high": diff + crit * se}


def cuped_adjust(y: np.ndarray, covariate: np.ndarray) -> tuple[np.ndarray, float, float]:
    """CUPED: remove the part of the outcome explained by a pre-experiment covariate.

    ``theta = cov(y, x) / var(x)`` is estimated on the pooled data; the adjusted outcome has the same mean
    as ``y`` in expectation but lower variance. Returns (adjusted outcome, theta, variance reduction share).
    """
    x = np.asarray(covariate, dtype=float)
    y = np.asarray(y, dtype=float)
    theta = float(np.cov(y, x, ddof=1)[0, 1] / np.var(x, ddof=1)) if np.var(x) > 0 else 0.0
    adjusted = y - theta * (x - x.mean())
    reduction = 1.0 - float(np.var(adjusted, ddof=1) / np.var(y, ddof=1))
    return adjusted, theta, reduction


def srm_check(n_treatment: int, n_control: int, expected_treatment_share: float = 0.5) -> dict:
    """Sample ratio mismatch: a tiny p-value means assignment or logging is broken and results cannot be trusted."""
    total = n_treatment + n_control
    chi = stats.chisquare([n_treatment, n_control], [total * expected_treatment_share, total * (1 - expected_treatment_share)])
    return {"chi2": float(chi.statistic), "p_value": float(chi.pvalue), "ok": bool(chi.pvalue > 0.001)}


def beta_binomial(x_t: int, n_t: int, x_c: int, n_c: int, draws: int = 200_000, seed: int = 0) -> dict:
    """Bayesian read-out with uniform Beta(1, 1) priors: probability that treatment beats control."""
    rng = np.random.default_rng(seed)
    post_t = rng.beta(1 + x_t, 1 + n_t - x_t, draws)
    post_c = rng.beta(1 + x_c, 1 + n_c - x_c, draws)
    lift = post_t - post_c
    lo, hi = np.percentile(lift, [2.5, 97.5])
    return {
        "prob_treatment_better": float((lift > 0).mean()),
        "expected_absolute_lift": float(lift.mean()),
        "credible_low": float(lo),
        "credible_high": float(hi),
    }


def benjamini_hochberg(p_values: list[float] | np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values (q-values) controlling the false discovery rate."""
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    adjusted = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(adjusted, 0, 1)
    return out


def heterogeneity_test(outcome: np.ndarray, treated: np.ndarray, group: pd.Series) -> dict:
    """Likelihood-ratio test of whether the treatment effect differs between groups.

    Compares a logistic model with a common treatment effect against one with a treatment-by-group
    interaction. A large p-value means there is no evidence that the effect varies by group. This is the right
    test for "does it work differently by channel?"; testing each group against zero is not.
    """
    dummies = pd.get_dummies(group.reset_index(drop=True), drop_first=True).astype(float)
    t = pd.Series(treated.astype(float), name="treated")
    base = pd.concat([t, dummies], axis=1)
    full = pd.concat([base, dummies.mul(t, axis=0).add_prefix("treated_x_")], axis=1)
    y = np.asarray(outcome, dtype=float)
    m0 = sm.Logit(y, sm.add_constant(base)).fit(disp=0)
    m1 = sm.Logit(y, sm.add_constant(full)).fit(disp=0)
    lr = max(2 * (m1.llf - m0.llf), 0.0)
    df = dummies.shape[1]
    return {"lr_statistic": float(lr), "df": int(df), "p_value": float(stats.chi2.sf(lr, df))}


# ------------------------------------------------------------------ self-validation


def simulate_tests(p0: float, effect: float, n_per_arm: int, reps: int = 4000, seed: int = 0, alpha: float = 0.05) -> dict:
    """Monte Carlo check of a two-proportion test: rejection rate and coverage of the true difference."""
    rng = np.random.default_rng(seed)
    x_c = rng.binomial(n_per_arm, p0, reps)
    x_t = rng.binomial(n_per_arm, p0 + effect, reps)
    p_c, p_t = x_c / n_per_arm, x_t / n_per_arm
    diff = p_t - p_c
    pooled = (x_c + x_t) / (2 * n_per_arm)
    se_pooled = np.sqrt(pooled * (1 - pooled) * 2 / n_per_arm)
    z = np.divide(diff, se_pooled, out=np.zeros_like(diff), where=se_pooled > 0)
    reject = np.abs(z) > stats.norm.ppf(1 - alpha / 2)
    se = np.sqrt(p_t * (1 - p_t) / n_per_arm + p_c * (1 - p_c) / n_per_arm)
    crit = stats.norm.ppf(1 - alpha / 2)
    covered = (diff - crit * se <= effect) & (effect <= diff + crit * se)
    return {"rejection_rate": float(reject.mean()), "ci_coverage": float(covered.mean()), "reps": reps}


# ------------------------------------------------------------------ the win-back experiment


def run_winback_experiment(snapshot: pd.DataFrame, true_effect: float = 0.02, seed: int = 0, order_value: float = 50.0) -> dict:
    """Randomised win-back offer for lapsed customers (recency score 1-2) at the later snapshot.

    Outcomes are the *observed* 30-day behaviour from the simulated orders. The treatment effect is then
    layered on top by flipping a share of treated non-purchasers to purchasers so that the expected absolute lift
    equals ``true_effect``. The assignment, the effect and the extra revenue are synthetic.
    """
    rng = np.random.default_rng(seed)
    pop = snapshot.loc[snapshot["r_score"] <= 2].reset_index(drop=True)
    n = len(pop)
    treated = np.zeros(n, dtype=bool)
    treated[rng.permutation(n)[: n // 2]] = True

    base_y = pop["repurchased_30d"].to_numpy().astype(bool)
    p0 = float(base_y.mean())
    flip_prob = true_effect / (1 - p0)
    extra = treated & ~base_y & (rng.random(n) < flip_prob)
    y = base_y | extra
    mu = math.log(order_value) - 0.45**2 / 2
    revenue = pop["revenue_30d"].to_numpy() + extra * np.exp(rng.normal(mu, 0.45, n))

    x_t, n_t = int(y[treated].sum()), int(treated.sum())
    x_c, n_c = int(y[~treated].sum()), int((~treated).sum())
    primary = two_proportion_test(x_t, n_t, x_c, n_c)
    bayes = beta_binomial(x_t, n_t, x_c, n_c, seed=seed)
    srm = srm_check(n_t, n_c)

    plain = welch_mean_test(revenue[treated], revenue[~treated])
    adjusted, theta, reduction = cuped_adjust(revenue, pop["monetary"].to_numpy())
    cuped = welch_mean_test(adjusted[treated], adjusted[~treated])

    rows = []
    for ch in sorted(pop["channel"].unique()):
        m = (pop["channel"] == ch).to_numpy()
        t_ok, c_ok = treated & m, ~treated & m
        r = two_proportion_test(int(y[t_ok].sum()), int(t_ok.sum()), int(y[c_ok].sum()), int(c_ok.sum()))
        rows.append({"channel": ch, "n": int(m.sum()), "absolute_lift": r["absolute_lift"], "ci_low": r["ci_low"],
                     "ci_high": r["ci_high"], "p_value": r["p_value"]})
    het = pd.DataFrame(rows)
    het["q_value"] = benjamini_hochberg(het["p_value"].to_numpy())

    interaction = heterogeneity_test(y, treated, pop["channel"])
    planned_n = sample_size_per_arm(p0, true_effect)
    return {
        "population": n,
        "baseline_rate": p0,
        "true_effect": true_effect,
        "n_treatment": n_t,
        "n_control": n_c,
        "primary": primary,
        "bayesian": bayes,
        "srm": srm,
        "revenue_plain": plain,
        "revenue_cuped": cuped,
        "cuped_theta": theta,
        "cuped_variance_reduction": reduction,
        "heterogeneity": het,
        "interaction_test": interaction,
        "planned_n_per_arm": planned_n,
        "power_with_available_n": power_for_n(p0, true_effect, n_t),
        "mde_with_available_n": minimum_detectable_effect(p0, n_t),
    }


def validate_methods(p0: float, effect: float, n_per_arm: int, reps: int = 4000, seed: int = 0) -> dict:
    """A/A and A/B simulations that test the test: false positive rate, coverage and power."""
    aa = simulate_tests(p0, 0.0, n_per_arm, reps, seed)
    ab = simulate_tests(p0, effect, n_per_arm, reps, seed + 1)
    return {
        "aa_false_positive_rate": aa["rejection_rate"],
        "aa_ci_coverage": aa["ci_coverage"],
        "ab_simulated_power": ab["rejection_rate"],
        "ab_ci_coverage": ab["ci_coverage"],
        "ab_analytic_power": power_for_n(p0, effect, n_per_arm),
        "reps": reps,
    }
