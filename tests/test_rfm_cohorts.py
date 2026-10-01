import numpy as np
import pytest

from lifecycle_lab.cohorts import average_retention_curve, cohort_activity, retention_matrix
from lifecycle_lab.rfm import SEGMENT_ORDER, choose_k, cluster_customers, rfm_scores, segment_summary
from lifecycle_lab.sqlcohort import sql_cohort_activity


def test_rfm_scores_are_one_to_five_and_quintiles_are_balanced(scored):
    for col in ("r_score", "f_score", "m_score"):
        counts = scored[col].value_counts()
        assert set(counts.index) == {1, 2, 3, 4, 5}
        assert counts.max() - counts.min() <= 1


def test_recent_buyers_score_higher_on_recency(scored):
    best = scored.loc[scored.r_score == 5, "recency_days"].max()
    worst = scored.loc[scored.r_score == 1, "recency_days"].min()
    assert best <= worst


def test_segments_cover_everyone_and_shares_sum_to_one(scored):
    assert set(scored.rfm_segment) <= set(SEGMENT_ORDER)
    summary = segment_summary(scored)
    assert summary.customers.sum() == len(scored)
    assert summary.customer_share.sum() == pytest.approx(1.0)
    assert summary.revenue_share.sum() == pytest.approx(1.0)


def test_champions_repurchase_more_than_hibernating_customers(scored):
    s = segment_summary(scored).set_index("segment")
    assert s.loc["Champions", "repurchase_rate_90d"] > 3 * s.loc["Hibernating", "repurchase_rate_90d"]


def test_kmeans_is_deterministic_and_ari_is_bounded(scored):
    a, ari_a = cluster_customers(scored, 4, seed=1)
    b, ari_b = cluster_customers(scored, 4, seed=1)
    assert a["cluster"].equals(b["cluster"]) and ari_a == ari_b
    assert -1 <= ari_a <= 1
    ks = choose_k(scored, k_range=range(3, 5), sample=500)
    assert list(ks.k) == [3, 4] and ks.silhouette.between(-1, 1).all()


def test_month_zero_retention_is_one_and_rates_are_probabilities(world):
    act = cohort_activity(world.customers, world.orders)
    m0 = act[act.months_since_acquisition == 0]
    assert (m0["retention"] == 1.0).all()
    assert act["retention"].between(0, 1).all()


def test_only_fully_observed_ages_are_kept(world):
    act = cohort_activity(world.customers, world.orders)
    last = world.orders.order_date.max().to_period("M")
    age_limit = (last - act.acq_month.dt.to_period("M")).apply(lambda x: x.n)
    assert (act.months_since_acquisition <= age_limit).all()


def test_retention_declines_after_the_first_month(world):
    curve = average_retention_curve(cohort_activity(world.customers, world.orders))
    assert curve.iloc[1] > curve.iloc[12]
    assert retention_matrix(cohort_activity(world.customers, world.orders)).shape[0] == 36


def test_sql_cohort_query_matches_pandas(world):
    py = cohort_activity(world.customers, world.orders, max_age=40)
    sql = sql_cohort_activity(world.customers, world.orders)
    sql["acq_month"] = sql["acq_month"].astype("datetime64[ns]")
    merged = py.merge(sql, on=["acq_month", "months_since_acquisition"], suffixes=("_py", "_sql"))
    assert len(merged) == len(py)
    assert (merged.active_customers_py == merged.active_customers_sql).all()
    assert np.allclose(merged.revenue_py, merged.revenue_sql)
    assert np.allclose(merged.retention_py, merged.retention_sql)
