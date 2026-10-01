import json

import pandas as pd
import pytest

from lifecycle_lab.cli import run
from lifecycle_lab.forecast import backtest, mape, monthly_revenue
from lifecycle_lab.predict import ACTIVE_WINDOW_DAYS, active_customers, churn_models, value_model


def test_churn_model_beats_chance_out_of_time(train, test_snap):
    res = churn_models(train, test_snap)
    for name, m in res["metrics"].items():
        assert m["roc_auc"] > 0.6, name
        assert m["top_decile_lift"] > 1.0, name
    assert res["gb_params"]["max_depth"] in (2, 3) and res["gb_params"]["n_estimators"] in (50, 100, 200)
    assert res["n_test"] == len(active_customers(test_snap))


def test_churn_population_is_recently_active_customers(test_snap):
    assert (active_customers(test_snap)["recency_days"] <= ACTIVE_WINDOW_DAYS).all()


def test_value_model_beats_the_run_rate_baseline(train, test_snap):
    res = value_model(train, test_snap)
    assert res["model_metrics"]["mae"] < res["baseline_metrics"]["mae"]
    assert (res["predictions"] >= 0).all()
    d = res["deciles"]
    assert len(d) == 10 and d["avg_predicted"].is_monotonic_increasing
    assert d["avg_actual"].iloc[-1] > 5 * d["avg_actual"].iloc[0]  # the top decile really is worth far more


def test_mape_is_correct():
    import numpy as np

    assert mape(np.array([100.0, 200.0]), np.array([110.0, 180.0])) == pytest.approx(0.10)


def test_holt_winters_backtest_returns_all_models(world):
    bt = backtest(monthly_revenue(world.orders))
    assert set(bt["mape"]) == {"Holt-Winters", "Seasonal naive (growth adjusted)", "Naive (last month)"}
    assert len(bt["test"]) == 6 and all(len(p) == 6 for p in bt["predictions"].values())


def test_end_to_end_run_writes_everything(tmp_path):
    res = run(seed=2, out=tmp_path / "r", bi=tmp_path / "b", new_base=120)
    for name in ("report.html", "results.json", "segments.png", "cohorts.png", "churn.png", "value.png", "forecast.png",
                 "experiment.png", "validation.png"):
        assert (tmp_path / "r" / name).stat().st_size > 0
    for name in ("dim_customer_segments.csv", "fact_cohort_retention.csv", "fact_segment_summary.csv", "fact_monthly_revenue.csv"):
        assert not pd.read_csv(tmp_path / "b" / name).empty
    assert len(res["findings"]) == 8
    assert "seeded simulation" in (tmp_path / "r" / "report.html").read_text()
    assert json.loads((tmp_path / "r" / "results.json").read_text())["meta"]["seed"] == 2
