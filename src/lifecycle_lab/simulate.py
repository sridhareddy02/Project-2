"""Seeded simulator of a repeat-purchase customer base with four hidden behavioural segments.

Customers are acquired every month for 36 months, so later cohorts are observed for less time. Each customer belongs to a latent
segment that sets how often they buy, how quickly they leave and how much they spend. The segment
is kept as ``true_segment`` so that segmentation methods can be scored against it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

SEGMENTS = {
    "loyalist": dict(share=0.15, rate=0.55, dropout=0.010, aov=85.0),
    "bargain_hunter": dict(share=0.30, rate=0.30, dropout=0.035, aov=42.0),
    "occasional": dict(share=0.35, rate=0.10, dropout=0.025, aov=64.0),
    "one_and_done": dict(share=0.20, rate=0.03, dropout=0.100, aov=50.0),
}
CHANNELS = ["organic", "paid_search", "paid_social", "referral"]
CHANNEL_MIX = {
    "loyalist": [0.30, 0.15, 0.10, 0.45],
    "bargain_hunter": [0.15, 0.25, 0.50, 0.10],
    "occasional": [0.40, 0.30, 0.20, 0.10],
    "one_and_done": [0.20, 0.35, 0.40, 0.05],
}
# multiplier on purchase rate by calendar month (January first)
SEASONALITY = np.array([0.90, 0.85, 0.95, 0.95, 1.00, 1.00, 0.95, 0.95, 1.00, 1.05, 1.35, 1.50])
START_MONTH = "2023-01-01"
N_MONTHS = 36
ACQUISITION_MONTHS = 36
ORDER_VALUE_SIGMA = 0.45


@dataclass
class World:
    customers: pd.DataFrame  # customer_id, acq_month, channel, true_segment
    orders: pd.DataFrame     # order_id, customer_id, order_date, revenue
    seed: int


def simulate(seed: int = 21, new_base: int = 400, growth: int = 5) -> World:
    rng = np.random.default_rng(seed)
    months = pd.date_range(START_MONTH, periods=N_MONTHS, freq="MS")

    acq_idx = np.concatenate([np.full(new_base + growth * m, m) for m in range(ACQUISITION_MONTHS)])
    n = len(acq_idx)
    seg_names = list(SEGMENTS)
    seg = rng.choice(len(seg_names), size=n, p=[SEGMENTS[s]["share"] for s in seg_names])
    channel = np.array([rng.choice(CHANNELS, p=CHANNEL_MIX[seg_names[s]]) for s in seg])

    rate = np.array([SEGMENTS[s]["rate"] for s in seg_names])[seg]
    dropout = np.array([SEGMENTS[s]["dropout"] for s in seg_names])[seg]
    aov = np.array([SEGMENTS[s]["aov"] for s in seg_names])[seg]

    alive = np.ones(n, dtype=bool)
    rows_c, rows_m, rows_k = [], [], []
    for t in range(N_MONTHS):
        existing = acq_idx < t
        alive &= ~(existing & (rng.random(n) < dropout))
        counts = np.zeros(n, dtype=int)
        counts[acq_idx == t] = 1  # first order lands in the acquisition month
        counts[existing & alive] = rng.poisson(rate[existing & alive] * SEASONALITY[months[t].month - 1])
        who = np.where(counts > 0)[0]
        rows_c.append(np.repeat(who, counts[who]))
        rows_m.append(np.full(counts[who].sum(), t))
    cust = np.concatenate(rows_c)
    month = np.concatenate(rows_m)

    days_in_month = months[month].days_in_month.to_numpy()
    day = (rng.random(len(cust)) * days_in_month).astype(int)
    order_date = months[month] + pd.to_timedelta(day, unit="D")
    mu = np.log(aov[cust]) - ORDER_VALUE_SIGMA**2 / 2
    revenue = np.exp(rng.normal(mu, ORDER_VALUE_SIGMA))

    orders = (
        pd.DataFrame({"customer_id": cust, "order_date": order_date, "revenue": revenue})
        .sort_values(["order_date", "customer_id"], kind="stable")
        .reset_index(drop=True)
    )
    orders.insert(0, "order_id", np.arange(len(orders)))

    customers = pd.DataFrame(
        {
            "customer_id": np.arange(n),
            "acq_month": months[acq_idx],
            "channel": channel,
            "true_segment": np.array(seg_names)[seg],
        }
    )
    return World(customers=customers, orders=orders, seed=seed)
