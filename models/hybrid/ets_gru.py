"""Hybrid ETS + GRU: Holt-trend baseline + GRU residual learning."""
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from models.ml.gru import train_gru, forecast_gru


def ets_gru_forecast(series: pd.Series, steps: int = 30):
    # Step 1: ETS
    try:
        ets_model  = ExponentialSmoothing(series, trend="add", damped_trend=True,
                                          initialization_method="estimated").fit(optimized=True)
    except Exception:
        ets_model  = ExponentialSmoothing(series).fit(optimized=True)
    ets_pred   = np.array(ets_model.forecast(steps))

    # Step 2: residuals
    resid = pd.Series(series.values - ets_model.fittedvalues.values, index=series.index).dropna()

    # Step 3: GRU on residuals
    gru_model, scaler, _ = train_gru(resid, epochs=30)
    gru_pred = forecast_gru(gru_model, resid, scaler, steps=steps)

    final = ets_pred + gru_pred[: len(ets_pred)]
    return final, ets_pred, gru_pred
