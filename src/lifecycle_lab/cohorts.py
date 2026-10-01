"""Acquisition-cohort retention."""

from __future__ import annotations

import pandas as pd


def cohort_activity(customers: pd.DataFrame, orders: pd.DataFrame, max_age: int = 24) -> pd.DataFrame:
    """Long table: acquisition cohort, months since acquisition, active customers, revenue, retention rates.

    A customer is *active* in a month if they placed at least one order in it. Month 0 is the acquisition
    month, so retention at month 0 is 100% by construction.
    """
    o = orders.merge(customers[["customer_id", "acq_month"]], on="customer_id")
    o["order_month"] = o["order_date"].values.astype("datetime64[M]")
    o["age"] = (o["order_month"].dt.year - o["acq_month"].dt.year) * 12 + (o["order_month"].dt.month - o["acq_month"].dt.month)
    o = o.loc[o["age"] <= max_age]
    act = (
        o.groupby(["acq_month", "age"])
        .agg(active_customers=("customer_id", "nunique"), revenue=("revenue", "sum"), orders=("order_id", "size"))
        .reset_index()
    )
    size = customers.groupby("acq_month").size().rename("cohort_size").reset_index()
    act = act.merge(size, on="acq_month")
    # keep only ages that are fully observed for the cohort
    last_month = orders["order_date"].max().to_period("M").to_timestamp()
    observed_age = (last_month.year - act["acq_month"].dt.year) * 12 + (last_month.month - act["acq_month"].dt.month)
    act = act.loc[act["age"] <= observed_age].copy()
    act["retention"] = act["active_customers"] / act["cohort_size"]
    act["revenue_per_customer"] = act["revenue"] / act["cohort_size"]
    return act.rename(columns={"age": "months_since_acquisition"})


def retention_matrix(activity: pd.DataFrame, value: str = "retention") -> pd.DataFrame:
    return activity.pivot(index="acq_month", columns="months_since_acquisition", values=value)


def average_retention_curve(activity: pd.DataFrame) -> pd.Series:
    """Customer-weighted retention by age across cohorts that have reached that age."""
    g = activity.groupby("months_since_acquisition")
    return g["active_customers"].sum() / g["cohort_size"].sum()
