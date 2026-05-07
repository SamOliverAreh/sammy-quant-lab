import numpy as np
import itertools
import warnings
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings(“ignore”)

def find_best_arima(series, max_p=3, max_d=2, max_q=3):
best_aic, best_order = float(“inf”), (1, 1, 1)
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

def arima_forecast(series, steps=30, order=None):
if order is None:
order = find_best_arima(series)
fitted = ARIMA(series, order=order).fit()
fc     = fitted.forecast(steps=steps)
ci     = fitted.get_forecast(steps=steps).conf_int()
return np.array(fc), order, ci

def arima_ic(series, order=None):
“”“Return {AIC, BIC, LogL, order} for the best (or given) ARIMA fit.”””
if order is None:
order = find_best_arima(series)
fitted = ARIMA(series, order=order).fit()
return {
“AIC”:  round(fitted.aic, 2),
“BIC”:  round(fitted.bic, 2),
“LogL”: round(fitted.llf, 2),
“detail”: f”ARIMA{order}”,
}