"""ETS / Holt-Winters on log-returns with in-sample fitted values."""
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from statsmodels.tsa.holtwinters import ExponentialSmoothing


def _fit_ets(series, seasonal_periods=None):
    try:
        return ExponentialSmoothing(
            series, trend="add", damped_trend=True,
            initialization_method="estimated",
        ).fit(optimized=True)
    except Exception:
        return ExponentialSmoothing(series).fit(optimized=True)


def ets_forecast(series, steps=30, seasonal_periods=None):
    model    = _fit_ets(series, seasonal_periods)
    pred     = model.forecast(steps)
    insample = model.fittedvalues.values
    actual   = series.values
    params   = {
        "alpha": round(float(model.params.get("smoothing_level", 0)), 4),
        "beta":  round(float(model.params.get("smoothing_trend", 0)), 4),
        "phi":   round(float(model.params.get("damping_trend", 1.0)), 4),
        "AIC":   round(model.aic, 2),
    }
    return np.array(pred), params, insample, actual


def ets_ic(series):
    m   = _fit_ets(series)
    n   = len(series)
    k   = len(m.params)
    llf = float(m.llf) if hasattr(m, "llf") else float(-m.sse)
    bic = -2 * llf + k * np.log(n)
    return {"AIC": round(m.aic, 2), "BIC": round(bic, 2),
            "LogL": round(llf, 2), "detail": "ETS(A,Ad,N)"}
