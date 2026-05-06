"""
ETS — Exponential Smoothing (Holt's Trend / Holt-Winters).
Uses statsmodels ExponentialSmoothing with automatic damped trend.
"""
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from statsmodels.tsa.holtwinters import ExponentialSmoothing


def ets_forecast(series, steps: int = 30, seasonal_periods: int = None):
    """
    Fit ETS model and return forecast array.

    Args:
        series: pd.Series of prices
        steps:  forecast horizon
        seasonal_periods: set to e.g. 5 (weekly) or 252 (yearly) if seasonal

    Returns:
        pred (np.ndarray), model_params (dict)
    """
    trend = "add"
    seasonal = None
    sp = None

    if seasonal_periods and len(series) > 2 * seasonal_periods:
        seasonal = "add"
        sp = seasonal_periods

    try:
        model = ExponentialSmoothing(
            series,
            trend=trend,
            seasonal=seasonal,
            seasonal_periods=sp,
            damped_trend=True,
            initialization_method="estimated",
        ).fit(optimized=True)
    except Exception:
        # fallback: simple exponential smoothing
        model = ExponentialSmoothing(series, trend=None, seasonal=None).fit(optimized=True)

    pred = model.forecast(steps)
    params = {
        "alpha": round(float(model.params.get("smoothing_level", 0)), 4),
        "beta":  round(float(model.params.get("smoothing_trend", 0)), 4),
        "gamma": round(float(model.params.get("smoothing_seasonal", 0)), 4),
        "phi":   round(float(model.params.get("damping_trend", 1.0)), 4),
        "AIC":   round(model.aic, 2),
    }
    return np.array(pred), params
