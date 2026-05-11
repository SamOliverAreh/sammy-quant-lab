"""Hybrid ETS+GRU — corrected for updated function signatures (v4.2)."""
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from models.ml.gru import train_gru, forecast_gru

def ets_gru_forecast(series: pd.Series, steps: int = 30, exog=None):
    try:
        ets_m = ExponentialSmoothing(series, trend="add", damped_trend=True,
                                     initialization_method="estimated").fit(optimized=True)
    except Exception:
        ets_m = ExponentialSmoothing(series).fit(optimized=True)
    ets_pred = np.array(ets_m.forecast(steps))
    resid    = pd.Series(series.values - ets_m.fittedvalues.values, index=series.index).dropna()
    # train_gru → (model, scaler_y, scaler_X, losses, insample, actual)
    gru_m, sc_y, sc_X, losses, ins, act = train_gru(resid, epochs=30, exog=exog)
    gru_pred = forecast_gru(gru_m, resid, sc_y, steps=steps, scaler_X=sc_X, exog=exog)
    final = ets_pred + gru_pred[:len(ets_pred)]
    return final, ets_pred, gru_pred
