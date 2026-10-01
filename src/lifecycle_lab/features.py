"""Point-in-time customer features. Everything is computed using orders on or before the snapshot."""

from __future__ import annotations

import numpy as np
import pandas as pd

CHURN_DAYS = 90
VALUE_DAYS = 182  # six months


def customer_features(customers: pd.DataFrame, orders: pd.DataFrame, snapshot: pd.Timestamp) -> pd.DataFrame:
    """Recency, frequency, monetary value, tenure and recent activity as of ``snapshot``."""
    past = orders.loc[orders["order_date"] <= snapshot]
    g = past.groupby("customer_id")
    feats = pd.DataFrame(
        {
            "frequency": g.size(),
            "monetary": g["revenue"].sum(),
            "first_order": g["order_date"].min(),
            "last_order": g["order_date"].max(),
        }
    )
    recent = past.loc[past["order_date"] > snapshot - pd.Timedelta(days=90)].groupby("customer_id")
    feats["orders_last_90d"] = recent.size().reindex(feats.index).fillna(0)
    feats["revenue_last_90d"] = recent["revenue"].sum().reindex(feats.index).fillna(0.0)
    feats["recency_days"] = (snapshot - feats["last_order"]).dt.days
    feats["tenure_days"] = (snapshot - feats["first_order"]).dt.days
    feats["avg_order_value"] = feats["monetary"] / feats["frequency"]
    feats["orders_per_month"] = feats["frequency"] / np.maximum(feats["tenure_days"] / 30.4, 1.0)
    out = feats.reset_index().merge(customers[["customer_id", "channel", "true_segment"]], on="customer_id", how="left")
    return out


def future_outcomes(orders: pd.DataFrame, snapshot: pd.Timestamp) -> pd.DataFrame:
    """What happens after the snapshot: any order within 90 days, and revenue within six months."""
    after = orders.loc[orders["order_date"] > snapshot]
    near = after.loc[after["order_date"] <= snapshot + pd.Timedelta(days=CHURN_DAYS)]
    far = after.loc[after["order_date"] <= snapshot + pd.Timedelta(days=VALUE_DAYS)]
    out = pd.DataFrame(index=pd.Index(orders["customer_id"].unique(), name="customer_id"))
    out["repurchased_90d"] = near.groupby("customer_id").size().reindex(out.index).fillna(0) > 0
    out["revenue_6m"] = far.groupby("customer_id")["revenue"].sum().reindex(out.index).fillna(0.0)
    out["repurchased_30d"] = (
        after.loc[after["order_date"] <= snapshot + pd.Timedelta(days=30)].groupby("customer_id").size().reindex(out.index).fillna(0) > 0
    )
    out["revenue_30d"] = (
        after.loc[after["order_date"] <= snapshot + pd.Timedelta(days=30)].groupby("customer_id")["revenue"].sum().reindex(out.index).fillna(0.0)
    )
    return out.reset_index()


def labelled_snapshot(customers: pd.DataFrame, orders: pd.DataFrame, snapshot: str | pd.Timestamp) -> pd.DataFrame:
    snap = pd.Timestamp(snapshot)
    feats = customer_features(customers, orders, snap)
    return feats.merge(future_outcomes(orders, snap), on="customer_id", how="left")
