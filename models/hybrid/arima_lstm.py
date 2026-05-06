"""Hybrid ARIMA + LSTM: linear baseline + nonlinear residual learning."""
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from statsmodels.tsa.arima.model import ARIMA as _ARIMA
from models.statistical.arima import arima_forecast
from models.ml.lstm import train_lstm, forecast_lstm


def hybrid_forecast(series: pd.Series, steps: int = 30):
    arima_pred, order, conf_int = arima_forecast(series, steps=steps)

    fitted  = _ARIMA(series, order=order).fit()
    resid   = pd.Series(fitted.resid, index=series.index).dropna()

    lstm_model, scaler, _ = train_lstm(resid, epochs=30)
    lstm_pred = forecast_lstm(lstm_model, resid, scaler, steps=steps)

    final = arima_pred + lstm_pred[: len(arima_pred)]
    return final, arima_pred, lstm_pred, order
