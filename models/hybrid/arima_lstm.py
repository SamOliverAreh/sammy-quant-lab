"""Hybrid ARIMA+LSTM — corrected for updated function signatures (v4.2)."""
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from statsmodels.tsa.arima.model import ARIMA as _ARIMA
from models.statistical.arima import arima_forecast, find_best_arima
from models.ml.lstm import train_lstm, forecast_lstm

def hybrid_forecast(series: pd.Series, steps: int = 30, exog=None):
    # arima_forecast → (pred, order, ci, insample, actual)
    arima_pred, order, ci, ins, act = arima_forecast(series, steps=steps, exog=exog)
    fitted = _ARIMA(series, order=order).fit()
    resid  = pd.Series(fitted.resid, index=series.index).dropna()
    # train_lstm → (model, scaler_y, scaler_X, losses, insample, actual)
    lstm_m, sc_y, sc_X, losses, ins2, act2 = train_lstm(resid, epochs=30, exog=exog)
    lstm_pred = forecast_lstm(lstm_m, resid, sc_y, steps=steps, scaler_X=sc_X, exog=exog)
    final = arima_pred + lstm_pred[:len(arima_pred)]
    return final, arima_pred, lstm_pred, order
