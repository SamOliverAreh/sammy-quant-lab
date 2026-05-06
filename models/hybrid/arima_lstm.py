import numpy as np
import pandas as pd
from models.statistical.arima import arima_forecast
from models.ml.lstm import train_lstm, forecast_lstm


def hybrid_forecast(series, steps=30):
    """
    Hybrid ARIMA + LSTM model.
    ARIMA captures linear structure; LSTM learns nonlinear residuals.
    """
    # Step 1: ARIMA on original series
    arima_pred, order, conf_int = arima_forecast(series, steps=steps)

    # Step 2: Compute in-sample ARIMA residuals
    from statsmodels.tsa.arima.model import ARIMA as _ARIMA
    arima_fitted = _ARIMA(series, order=order).fit()
    residuals = pd.Series(arima_fitted.resid, index=series.index)
    residuals = residuals.dropna()

    # Step 3: LSTM on residuals
    lstm_model, scaler, _ = train_lstm(residuals, epochs=30)
    lstm_pred = forecast_lstm(lstm_model, residuals, scaler, steps=steps)

    # Step 4: Combine
    final = arima_pred + lstm_pred[: len(arima_pred)]

    return final, arima_pred, lstm_pred, order
