"""ARIMA on log-returns — auto order selection + ARIMAX (exog support)."""
import numpy as np
import itertools
import warnings
import pandas as pd
warnings.filterwarnings("ignore")

from statsmodels.tsa.arima.model import ARIMA


def find_best_arima(series, max_p=3, max_d=1, max_q=3):
    """On log-returns, d is usually 0 (already stationary)."""
    best_aic, best_order = float("inf"), (1, 0, 1)
    for p, d, q in itertools.product(range(max_p + 1), range(max_d + 1), range(max_q + 1)):
        if p == 0 and d == 0 and q == 0:
            continue
        try:
            res = ARIMA(series, order=(p, d, q)).fit()
            if res.aic < best_aic:
                best_aic, best_order = res.aic, (p, d, q)
        except Exception:
            continue
    return best_order


def arima_forecast(series, steps=30, order=None, exog=None, exog_future=None):
    if order is None:
        order = find_best_arima(series)
    fitted = ARIMA(series, order=order, exog=exog).fit()
    fc     = fitted.forecast(steps=steps, exog=exog_future)
    ci     = fitted.get_forecast(steps=steps, exog=exog_future).conf_int()
    # In-sample fitted values
    insample = fitted.fittedvalues.values
    actual   = series.values[len(series) - len(insample):]
    return np.array(fc), order, ci, insample, actual


def arima_ic(series, order=None):
    if order is None:
        order = find_best_arima(series)
    fitted = ARIMA(series, order=order).fit()
    return {
        "AIC":    round(fitted.aic, 2),
        "BIC":    round(fitted.bic, 2),
        "LogL":   round(fitted.llf, 2),
        "detail": f"ARIMA{order}",
    }
