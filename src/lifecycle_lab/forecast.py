"""Monthly revenue forecast with a back-test against simple baselines."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing


def monthly_revenue(orders: pd.DataFrame) -> pd.Series:
    s = orders.set_index("order_date")["revenue"].resample("MS").sum()
    s.index.freq = "MS"
    return s


def mape(actual: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(np.abs((actual - pred) / actual)))


def backtest(series: pd.Series, horizon: int = 6) -> dict:
    """Hold out the last ``horizon`` months; compare Holt-Winters with naive and seasonal-naive forecasts."""
    train, test = series.iloc[:-horizon], series.iloc[-horizon:]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        hw = ExponentialSmoothing(
            train, trend="add", damped_trend=True, seasonal="mul", seasonal_periods=12, initialization_method="estimated"
        ).fit()
        hw_pred = hw.forecast(horizon)
    naive = np.repeat(train.iloc[-1], horizon)
    # same calendar months a year earlier, scaled by growth of the last 12 months over the 12 before
    last12, prev12 = train.iloc[-12:].sum(), train.iloc[-24:-12].sum()
    template = train.reindex([d - pd.DateOffset(years=1) for d in test.index]).to_numpy()
    seasonal = template * (last12 / prev12 if prev12 else 1.0)
    out = {
        "train": train,
        "test": test,
        "predictions": {"Holt-Winters": hw_pred, "Seasonal naive (growth adjusted)": pd.Series(seasonal, index=test.index),
                        "Naive (last month)": pd.Series(naive, index=test.index)},
    }
    out["mape"] = {k: mape(test.to_numpy(), v.to_numpy()) for k, v in out["predictions"].items()}
    return out
