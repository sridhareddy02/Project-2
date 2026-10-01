"""Churn and six-month value models, trained on one snapshot and tested on a later one."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, mean_absolute_error, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

NUMERIC = ["recency_days", "frequency", "monetary", "orders_last_90d", "revenue_last_90d", "tenure_days",
           "avg_order_value", "orders_per_month"]
CHANNELS = ["organic", "paid_search", "paid_social", "referral"]


def design_matrix(df: pd.DataFrame) -> pd.DataFrame:
    x = df[NUMERIC].astype(float).copy()
    for c in CHANNELS:
        x[f"channel_{c}"] = (df["channel"] == c).astype(float)
    return x


ACTIVE_WINDOW_DAYS = 180


def active_customers(df: pd.DataFrame) -> pd.DataFrame:
    """Customers who bought in the last 180 days: the population a retention team can still act on."""
    return df.loc[df["recency_days"] <= ACTIVE_WINDOW_DAYS].reset_index(drop=True)


def _tune_gradient_boosting(x: pd.DataFrame, y: pd.Series, seed: int):
    """Pick depth and size on a validation split of the *training* snapshot, then refit on all of it.

    The later test snapshot is never used for tuning, so its scores are an honest out-of-time estimate.
    """
    rng = np.random.default_rng(seed)
    mask = rng.random(len(x)) < 0.8
    best, best_auc = None, -1.0
    for depth in (2, 3):
        for n in (50, 100, 200):
            model = GradientBoostingClassifier(n_estimators=n, max_depth=depth, learning_rate=0.05, subsample=0.8, random_state=seed)
            model.fit(x[mask], y[mask])
            auc = roc_auc_score(y[~mask], model.predict_proba(x[~mask])[:, 1])
            if auc > best_auc:
                best, best_auc = {"max_depth": depth, "n_estimators": n}, auc
    final = GradientBoostingClassifier(learning_rate=0.05, subsample=0.8, random_state=seed, **best).fit(x, y)
    return final, {**best, "validation_auc": float(best_auc)}


def churn_models(train: pd.DataFrame, test: pd.DataFrame, seed: int = 0) -> dict:
    """Predict *no purchase in the next 90 days* among recently active customers."""
    train, test = active_customers(train), active_customers(test)
    y_train = (~train["repurchased_90d"]).astype(int)
    y_test = (~test["repurchased_90d"]).astype(int)
    xtr, xte = design_matrix(train), design_matrix(test)

    logit = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(np.log1p(xtr.clip(lower=0)), y_train)
    gb, gb_params = _tune_gradient_boosting(xtr, y_train, seed)
    scores = {
        "logistic_regression": logit.predict_proba(np.log1p(xte.clip(lower=0)))[:, 1],
        "gradient_boosting": gb.predict_proba(xte)[:, 1],
    }
    base_rate = float(y_test.mean())
    metrics = {}
    for name, s in scores.items():
        order = np.argsort(-s)
        top = order[: max(len(s) // 10, 1)]
        metrics[name] = {
            "roc_auc": float(roc_auc_score(y_test, s)),
            "average_precision": float(average_precision_score(y_test, s)),
            "top_decile_churn_rate": float(y_test.to_numpy()[top].mean()),
            "top_decile_lift": float(y_test.to_numpy()[top].mean() / base_rate),
        }
    return {"metrics": metrics, "scores": scores, "y_test": y_test.to_numpy(), "base_rate": base_rate, "model": gb,
            "gb_params": gb_params, "n_train": int(len(train)), "n_test": int(len(test))}


def value_model(train: pd.DataFrame, test: pd.DataFrame, seed: int = 0) -> dict:
    """Predict revenue over the next six months and compare with a naive run-rate baseline."""
    xtr, xte = design_matrix(train), design_matrix(test)
    reg = GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=seed)
    reg.fit(xtr, train["revenue_6m"])
    pred = np.clip(reg.predict(xte), 0, None)
    actual = test["revenue_6m"].to_numpy()
    baseline = (test["monetary"] / np.maximum(test["tenure_days"] / 30.4, 1.0) * 6).to_numpy()  # historic run rate

    def summary(p):
        order = np.argsort(-p)
        top = order[: len(p) // 10]
        return {
            "mae": float(mean_absolute_error(actual, p)),
            "rmse": float(np.sqrt(np.mean((actual - p) ** 2))),
            "spearman": float(pd.Series(p).corr(pd.Series(actual), method="spearman")),
            "top_decile_revenue_capture": float(actual[top].sum() / actual.sum()),
        }

    frame = pd.DataFrame({"predicted": pred, "actual": actual})
    frame["decile"] = pd.qcut(frame["predicted"].rank(method="first"), 10, labels=range(1, 11)).astype(int)
    deciles = frame.groupby("decile").agg(avg_predicted=("predicted", "mean"), avg_actual=("actual", "mean"), customers=("actual", "size"))
    return {
        "model_metrics": summary(pred),
        "baseline_metrics": summary(baseline),
        "deciles": deciles.reset_index(),
        "predictions": pred,
    }
