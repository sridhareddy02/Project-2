import numpy as np
import pandas as pd

from lifecycle_lab.features import customer_features, future_outcomes, labelled_snapshot
from lifecycle_lab.simulate import SEGMENTS, simulate


def test_same_seed_same_world():
    a, b = simulate(3, new_base=40), simulate(3, new_base=40)
    assert a.orders["revenue"].round(8).equals(b.orders["revenue"].round(8))
    assert a.customers.equals(b.customers)


def test_every_customer_has_a_first_order_in_the_acquisition_month(world):
    first = world.orders.groupby("customer_id")["order_date"].min().dt.to_period("M")
    acq = world.customers.set_index("customer_id")["acq_month"].dt.to_period("M")
    assert (first == acq.reindex(first.index)).all()
    assert world.orders["customer_id"].nunique() == len(world.customers)


def test_segment_mix_is_close_to_the_design(world):
    share = world.customers["true_segment"].value_counts(normalize=True)
    for name, cfg in SEGMENTS.items():
        assert abs(share[name] - cfg["share"]) < 0.03


def test_loyalists_are_worth_the_most_per_customer(world):
    o = world.orders.merge(world.customers, on="customer_id")
    per_customer = o.groupby("true_segment")["revenue"].sum() / world.customers["true_segment"].value_counts()
    assert per_customer.idxmax() == "loyalist" and per_customer.idxmin() == "one_and_done"


def test_features_use_only_orders_up_to_the_snapshot(world):
    snap = pd.Timestamp("2024-06-30")
    full = customer_features(world.customers, world.orders, snap).set_index("customer_id")
    truncated = customer_features(world.customers, world.orders[world.orders.order_date <= snap], snap).set_index("customer_id")
    cols = ["frequency", "monetary", "recency_days", "orders_last_90d", "tenure_days"]
    pd.testing.assert_frame_equal(full[cols].sort_index(), truncated[cols].sort_index())


def test_outcomes_only_count_orders_after_the_snapshot(world):
    snap = pd.Timestamp("2024-06-30")
    out = future_outcomes(world.orders, snap).set_index("customer_id")
    after = world.orders[(world.orders.order_date > snap) & (world.orders.order_date <= snap + pd.Timedelta(days=182))]
    expected = after.groupby("customer_id")["revenue"].sum().reindex(out.index).fillna(0)
    assert np.allclose(out["revenue_6m"], expected)
    assert (out["repurchased_30d"] <= out["repurchased_90d"]).all()


def test_customers_acquired_after_the_snapshot_are_not_in_the_features(world):
    snap = pd.Timestamp("2024-06-30")
    feats = labelled_snapshot(world.customers, world.orders, snap)
    late = world.customers.loc[world.customers.acq_month > snap, "customer_id"]
    assert not set(late) & set(feats["customer_id"])
