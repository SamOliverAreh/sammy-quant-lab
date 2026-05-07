“””
ETS — Exponential Smoothing (Holt’s Trend / Holt-Winters).
Uses statsmodels ExponentialSmoothing with automatic damped trend.
“””
import numpy as np
import warnings
warnings.filterwarnings(“ignore”)

from statsmodels.tsa.holtwinters import ExponentialSmoothing

def _fit_ets(series, seasonal_periods=None):
“”“Internal: fit and return the HW model object.”””
trend, seasonal, sp = “add”, None, None
if seasonal_periods and len(series) > 2 * seasonal_periods:
seasonal, sp = “add”, seasonal_periods
try:
return ExponentialSmoothing(
series, trend=trend, seasonal=seasonal, seasonal_periods=sp,
damped_trend=True, initialization_method=“estimated”,
).fit(optimized=True)
except Exception:
return ExponentialSmoothing(series, trend=None, seasonal=None).fit(optimized=True)

def ets_forecast(series, steps: int = 30, seasonal_periods: int = None):
model  = _fit_ets(series, seasonal_periods)
pred   = model.forecast(steps)
params = {
“alpha”: round(float(model.params.get(“smoothing_level”, 0)), 4),
“beta”:  round(float(model.params.get(“smoothing_trend”, 0)), 4),
“gamma”: round(float(model.params.get(“smoothing_seasonal”, 0)), 4),
“phi”:   round(float(model.params.get(“damping_trend”, 1.0)), 4),
“AIC”:   round(model.aic, 2),
}
return np.array(pred), params

def ets_ic(series):
“”“Return {AIC, BIC, LogL, detail} for the best ETS fit.”””
m   = _fit_ets(series)
n   = len(series)
k   = len(m.params)
llf = float(m.llf) if hasattr(m, “llf”) else float(-m.sse)
bic = -2 * llf + k * np.log(n)
return {
“AIC”:    round(m.aic, 2),
“BIC”:    round(bic, 2),
“LogL”:   round(llf, 2),
“detail”: “ETS(A,Ad,N)”,
}