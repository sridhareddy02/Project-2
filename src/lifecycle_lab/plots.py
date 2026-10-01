"""Charts for the report (matplotlib only)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

from .experiment import power_for_n  # noqa: E402

INK, MUTED = "#1b1f2a", "#6b7280"
VIOLET, GREEN, AMBER, GREY = "#6366f1", "#16a34a", "#f59e0b", "#cbd5e1"


def _style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#d1d5db")
    ax.tick_params(colors=MUTED)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)


def segment_chart(summary: pd.DataFrame, path: Path):
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    x = np.arange(len(summary))
    ax.bar(x - 0.2, summary["customer_share"] * 100, 0.4, color=GREY, label="Share of customers")
    ax.bar(x + 0.2, summary["revenue_share"] * 100, 0.4, color=VIOLET, label="Share of revenue to date")
    ax.set_xticks(x, summary["segment"], rotation=0, fontsize=9)
    ax.set_ylabel("%")
    ax.set_title("RFM segments: who the customers are and where the revenue comes from", loc="left", color=INK, fontsize=11)
    ax.legend(frameon=False, fontsize=9)
    _style(ax)
    _save(fig, path)


def cohort_heatmap(matrix: pd.DataFrame, path: Path, max_age: int = 18):
    data = matrix.iloc[:, : max_age + 1] * 100
    fig, ax = plt.subplots(figsize=(9.5, 6))
    cmap = LinearSegmentedColormap.from_list("ret", ["#f8fafc", "#c7d2fe", "#6366f1", "#312e81"])
    shown = data.copy()
    shown.iloc[:, 0] = np.nan  # month 0 is 100% by construction
    im = ax.imshow(shown.to_numpy(dtype=float), aspect="auto", cmap=cmap, vmin=0, vmax=30)
    ax.set_xticks(range(0, data.shape[1], 2), data.columns[::2])
    ax.set_yticks(range(0, len(data), 3), [d.strftime("%Y-%m") for d in data.index[::3]])
    ax.set_xlabel("Months since acquisition")
    ax.set_ylabel("Acquisition cohort")
    ax.set_title("Share of each cohort that orders in the month (month 0 hidden)", loc="left", color=INK, fontsize=11)
    cb = fig.colorbar(im, ax=ax, fraction=0.03)
    cb.set_label("% of cohort active", color=MUTED)
    _save(fig, path)


def churn_chart(churn: dict, path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    ax = axes[0]
    for name, color in (("logistic_regression", VIOLET), ("gradient_boosting", GREEN)):
        fpr, tpr, _ = roc_curve(churn["y_test"], churn["scores"][name])
        ax.plot(fpr, tpr, color=color, label=f"{name.replace('_', ' ').title()} (AUC {churn['metrics'][name]['roc_auc']:.2f})")
    ax.plot([0, 1], [0, 1], color=GREY, linestyle="--")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Churn model, tested on a later snapshot", loc="left", color=INK, fontsize=11)
    ax.legend(frameon=False, fontsize=8)
    _style(ax)

    ax = axes[1]
    s, y = churn["scores"]["logistic_regression"], churn["y_test"]
    deciles = pd.qcut(pd.Series(s).rank(method="first"), 10, labels=range(1, 11)).astype(int)
    rate = pd.Series(y).groupby(deciles.to_numpy()).mean() * 100
    ax.bar(rate.index, rate.to_numpy(), color=VIOLET)
    ax.axhline(churn["base_rate"] * 100, color=AMBER, linestyle="--", label=f"Average {churn['base_rate'] * 100:.0f}%")
    ax.set_xlabel("Predicted risk decile (10 = highest risk)")
    ax.set_ylabel("Actual churn rate (%)")
    ax.set_title("Higher scores really do churn more", loc="left", color=INK, fontsize=11)
    ax.legend(frameon=False, fontsize=8)
    _style(ax)
    _save(fig, path)


def value_chart(deciles: pd.DataFrame, path: Path):
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(deciles["decile"], deciles["avg_actual"], marker="o", color=INK, label="Actual six-month revenue")
    ax.plot(deciles["decile"], deciles["avg_predicted"], marker="s", color=VIOLET, label="Predicted")
    ax.set_xlabel("Predicted-value decile (10 = highest)")
    ax.set_ylabel("Average revenue per customer")
    ax.set_title("Predicted versus actual value on the later snapshot", loc="left", color=INK, fontsize=11)
    ax.legend(frameon=False, fontsize=9)
    _style(ax)
    _save(fig, path)


def forecast_chart(bt: dict, path: Path):
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.plot(bt["train"].index, bt["train"] / 1000, color=INK, label="History")
    ax.plot(bt["test"].index, bt["test"] / 1000, color=INK, linestyle="--", label="Held-out actual")
    colors = {"Holt-Winters": GREEN, "Seasonal naive (growth adjusted)": AMBER, "Naive (last month)": "#94a3b8"}
    for name, pred in bt["predictions"].items():
        ax.plot(pred.index, pred / 1000, color=colors[name], marker="o", markersize=4, label=f"{name} (MAPE {bt['mape'][name] * 100:.1f}%)")
    ax.set_ylabel("Monthly revenue (thousands)")
    ax.set_title("Revenue forecast back-test: last six months held out", loc="left", color=INK, fontsize=11)
    ax.legend(frameon=False, fontsize=8)
    _style(ax)
    _save(fig, path)


def experiment_chart(exp: dict, path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    ax = axes[0]
    rows = [("All lapsed customers", exp["primary"]["absolute_lift"], exp["primary"]["ci_low"], exp["primary"]["ci_high"])]
    for _, r in exp["heterogeneity"].iterrows():
        rows.append((f"{r['channel'].replace('_', ' ')} (q={r['q_value']:.2f})", r["absolute_lift"], r["ci_low"], r["ci_high"]))
    y = np.arange(len(rows))[::-1]
    for yy, (label, est, lo, hi) in zip(y, rows):
        ax.plot([lo * 100, hi * 100], [yy, yy], color=VIOLET if yy == y[0] else GREY, linewidth=3)
        ax.plot(est * 100, yy, "o", color=INK)
    ax.axvline(0, color=MUTED, linewidth=1)
    ax.axvline(exp["true_effect"] * 100, color=AMBER, linestyle="--", label="True effect")
    ax.set_yticks(y, [r[0] for r in rows], fontsize=8)
    ax.set_xlabel("Lift in 30-day repurchase rate (percentage points, 95% CI)")
    ax.set_title("Win-back offer: overall and by channel", loc="left", color=INK, fontsize=11)
    ax.legend(frameon=False, fontsize=8)
    _style(ax)

    ax = axes[1]
    n = np.arange(100, 4001, 50)
    p0, eff = exp["baseline_rate"], exp["true_effect"]
    ax.plot(n, [power_for_n(p0, eff, int(k)) * 100 for k in n], color=VIOLET)
    ax.axhline(80, color=GREY, linestyle="--")
    ax.axvline(exp["planned_n_per_arm"], color=AMBER, linestyle="--", label=f"Needed for 80%: {exp['planned_n_per_arm']:,} per arm")
    ax.axvline(exp["n_treatment"], color=GREEN, linestyle=":", label=f"Available: {exp['n_treatment']:,} per arm")
    ax.set_xlabel("Customers per arm")
    ax.set_ylabel("Power (%)")
    ax.set_title(f"Power to detect +{eff * 100:.0f} pp from a {p0 * 100:.1f}% baseline", loc="left", color=INK, fontsize=11)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    _style(ax)
    _save(fig, path)


def validation_chart(val: dict, path: Path):
    fig, ax = plt.subplots(figsize=(7.5, 4))
    labels = ["A/A false\npositive rate", "A/A CI\ncoverage", "A/B power\n(simulated)", "A/B power\n(formula)", "A/B CI\ncoverage"]
    values = [val["aa_false_positive_rate"], val["aa_ci_coverage"], val["ab_simulated_power"], val["ab_analytic_power"], val["ab_ci_coverage"]]
    targets = [0.05, 0.95, None, None, 0.95]
    ax.bar(labels, np.array(values) * 100, color=[GREY, GREY, VIOLET, VIOLET, GREY])
    for i, (v, t) in enumerate(zip(values, targets)):
        ax.text(i, v * 100 + 1.5, f"{v * 100:.1f}%", ha="center", color=INK, fontsize=9)
        if t is not None:
            ax.plot([i - 0.4, i + 0.4], [t * 100] * 2, color=AMBER, linewidth=2)
    ax.set_ylim(0, 112)
    ax.set_ylabel("%")
    ax.set_title(f"Testing the test: {val['reps']:,} simulated experiments (amber line = target)", loc="left", color=INK, fontsize=10.5)
    _style(ax)
    _save(fig, path)
