"""Run the cohort SQL against an in-memory SQLite database."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

SQL_PATH = Path(__file__).resolve().parents[2] / "sql" / "cohort_retention.sql"


def sql_cohort_activity(customers: pd.DataFrame, orders: pd.DataFrame, sql_path: Path = SQL_PATH) -> pd.DataFrame:
    con = sqlite3.connect(":memory:")
    try:
        customers[["customer_id", "acq_month"]].assign(acq_month=lambda d: d["acq_month"].dt.strftime("%Y-%m-%d")).to_sql(
            "customers", con, index=False
        )
        orders.assign(order_date=orders["order_date"].dt.strftime("%Y-%m-%d")).to_sql("orders", con, index=False)
        con.execute("CREATE INDEX idx_orders_customer ON orders(customer_id)")
        return pd.read_sql_query(Path(sql_path).read_text(), con)
    finally:
        con.close()
