"""Command line entry point: ``python -m lifecycle_lab run``."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from . import plots
from .cohorts import average_retention_curve, cohort_activity, retention_matrix
from .experiment import run_winback_experiment, validate_methods
from .features import labelled_snapshot
from .forecast import backtest, monthly_revenue
from .predict import active_customers, churn_models, value_model
from .report import build_report
from .rfm import choose_k, cluster_customers, cluster_profiles, rfm_scores, segment_summary
from .simulate import N_MONTHS, simulate

TRAIN_SNAPSHOT = "2024-06-30"
TEST_SNAPSHOT = "2025-06-30"


def run(seed: int, out: Path, bi: Path, new_base: int = 400) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    bi.mkdir(parents=True, exist_ok=True)
    world = simulate(seed, new_base=new_base)
    customers, orders = world.customers, world.orders

    train = labelled_snapshot(customers, orders, TRAIN_SNAPSHOT)
    test = labelled_snapshot(customers, orders, TEST_SNAPSHOT)
    scored = rfm_scores(test)
    seg = segment_summary(scored)

    ks = choose_k(scored)
    best_k = int(ks.loc[ks["silhouette"].idxmax(), "k"])
    clustered, ari = cluster_customers(scored, best_k)
    clustered4, ari4 = cluster_customers(scored, 4)
    profiles = cluster_profiles(clustered)

    activity = cohort_activity(customers, orders)
    matrix = retention_matrix(activity)
    curve = average_retention_curve(activity)

    churn = churn_models(train, test)
    value = value_model(train, test)
    bt = backtest(monthly_revenue(orders))
    exp = run_winback_experiment(scored, true_effect=0.02, seed=seed)
    val = validate_methods(exp["baseline_rate"], exp["true_effect"], exp["n_treatment"], reps=4000, seed=seed)

    plots.segment_chart(seg, out / "segments.png")
    plots.cohort_heatmap(matrix, out / "cohorts.png")
    plots.churn_chart(churn, out / "churn.png")
    plots.value_chart(value["deciles"], out / "value.png")
    plots.forecast_chart(bt, out / "forecast.png")
    plots.experiment_chart(exp, out / "experiment.png")
    plots.validation_chart(val, out / "validation.png")

    top = seg.set_index("segment")
    het = exp["heterogeneity"]
    n_sig = int((het["q_value"] < 0.05).sum())
    inter = exp["interaction_test"]
    inter_text = (
        f"a formal interaction test finds no evidence that the effect differs by channel (p = {inter['p_value']:.2f})"
        if inter["p_value"] >= 0.05
        else f"a formal interaction test finds the effect differs by channel (p = {inter['p_value']:.3f})"
    )
    channel_note = (
        f"{n_sig} of {len(het)} channels show a lift on their own after false-discovery correction, and {inter_text}."
    )
    agreement = "weakly" if ari < 0.3 else "moderately" if ari < 0.6 else "closely"
    p_value = exp["primary"]["p_value"]
    p_text = "p < 0.001" if p_value < 0.001 else f"p = {p_value:.3f}"
    g_lr, g_gb = churn["metrics"]["logistic_regression"], churn["metrics"]["gradient_boosting"]
    vm, vb = value["model_metrics"], value["baseline_metrics"]
    p = exp["primary"]
    findings = [
        f"Champions are {top.loc['Champions', 'customer_share'] * 100:.0f}% of customers but {top.loc['Champions', 'revenue_share'] * 100:.0f}% "
        f"of revenue to date, and {top.loc['Champions', 'repurchase_rate_90d'] * 100:.0f}% of them buy again within 90 days, against "
        f"{top.loc['Hibernating', 'repurchase_rate_90d'] * 100:.0f}% for Hibernating customers.",
        f"K-means on log recency, frequency and monetary value picks {best_k} clusters and agrees {agreement} with the four hidden "
        f"behavioural segments (adjusted Rand index {ari:.2f}; {ari4:.2f} when forced to 4): spend and frequency overlap between segments.",
        f"Retention settles near {curve.iloc[12] * 100:.0f}% of a cohort ordering in a given month a year after acquisition (month 1: {curve.iloc[1] * 100:.0f}%).",
        f"The churn model, trained on {TRAIN_SNAPSHOT} and tested on {TEST_SNAPSHOT}, reaches AUC {g_lr['roc_auc']:.2f} (logistic regression) "
        f"and {g_gb['roc_auc']:.2f} (tuned gradient boosting); the riskiest decile churns at {g_lr['top_decile_churn_rate'] * 100:.0f}% versus "
        f"{churn['base_rate'] * 100:.0f}% overall.",
        f"Predicted six-month value beats a run-rate baseline: MAE {vm['mae']:.0f} versus {vb['mae']:.0f}, and the top decile captures "
        f"{vm['top_decile_revenue_capture'] * 100:.0f}% of future revenue versus {vb['top_decile_revenue_capture'] * 100:.0f}%.",
        f"Holt-Winters forecasts the held-out six months with a MAPE of {bt['mape']['Holt-Winters'] * 100:.1f}%, against "
        f"{bt['mape']['Naive (last month)'] * 100:.1f}% for a naive forecast.",
        f"The win-back test measured {p['absolute_lift'] * 100:+.2f} pp (95% CI {p['ci_low'] * 100:.2f} to {p['ci_high'] * 100:.2f}, "
        f"{p_text}); "
        f"the Bayesian probability that the offer helps is {exp['bayesian']['prob_treatment_better'] * 100:.0f}%. {channel_note}",
        f"The testing method checks out in simulation: {val['aa_false_positive_rate'] * 100:.1f}% false positives on A/A tests (target 5%), "
        f"{val['ab_ci_coverage'] * 100:.1f}% interval coverage, and simulated power {val['ab_simulated_power'] * 100:.0f}% versus "
        f"{val['ab_analytic_power'] * 100:.0f}% from the formula.",
    ]

    tables = {
        "segments": seg,
        "clusters": profiles,
        "churn": pd.DataFrame(churn["metrics"]).T.reset_index().rename(columns={"index": "model"}),
        "value": pd.DataFrame({"metric": list(vm), "model": list(vm.values()), "run-rate baseline": list(vb.values())}),
        "forecast": pd.DataFrame({"model": list(bt["mape"]), "MAPE": list(bt["mape"].values())}),
        "experiment": exp["heterogeneity"],
    }
    meta = {"seed": seed, "customers": int(len(customers)), "orders": int(len(orders)), "months": N_MONTHS}
    build_report(out, meta, findings, tables)

    # BI-ready extracts
    scored.assign(snapshot=TEST_SNAPSHOT)[["customer_id", "channel", "r_score", "f_score", "m_score", "rfm_segment", "recency_days",
                                           "frequency", "monetary", "snapshot"]].merge(
        customers[["customer_id", "acq_month"]], on="customer_id").to_csv(bi / "dim_customer_segments.csv", index=False)
    activity.to_csv(bi / "fact_cohort_retention.csv", index=False)
    seg.to_csv(bi / "fact_segment_summary.csv", index=False)
    monthly_revenue(orders).rename("revenue").reset_index().rename(columns={"order_date": "month"}).to_csv(
        bi / "fact_monthly_revenue.csv", index=False)
    value["deciles"].to_csv(bi / "fact_value_deciles.csv", index=False)
    exp["heterogeneity"].to_csv(bi / "fact_experiment_by_channel.csv", index=False)

    results = {
        "meta": meta, "findings": findings,
        "segments": seg.to_dict(orient="records"),
        "kmeans": {"k": best_k, "ari": ari, "ari_k4": ari4, "silhouette": ks.to_dict(orient="records")},
        "retention_curve": curve.round(4).to_dict(),
        "churn": {"metrics": churn["metrics"], "base_rate": churn["base_rate"], "gb_params": churn["gb_params"],
                  "n_train": churn["n_train"], "n_test": churn["n_test"]},
        "value": {"model": vm, "baseline": vb},
        "forecast_mape": bt["mape"],
        "experiment": {k: v for k, v in exp.items() if k != "heterogeneity"},
        "experiment_by_channel": exp["heterogeneity"].to_dict(orient="records"),
        "validation": val,
    }
    (out / "results.json").write_text(json.dumps(results, indent=2, default=float))
    return results


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="lifecycle_lab", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="simulate, analyse and write reports")
    r.add_argument("--seed", type=int, default=21)
    r.add_argument("--out", type=Path, default=Path("reports"))
    r.add_argument("--bi", type=Path, default=Path("bi/data"))
    args = parser.parse_args(argv)
    res = run(args.seed, args.out, args.bi)
    print(f"Wrote report to {args.out / 'report.html'}")
    for line in res["findings"]:
        print(" -", line)


if __name__ == "__main__":
    main()
