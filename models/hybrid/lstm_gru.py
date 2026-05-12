"""
Hybrid LSTM + GRU
=================
LSTM captures long-range dependencies; GRU refines on LSTM residuals.
Widely used in financial deep-learning literature (e.g. Kim & Won 2018,
Baek & Kim 2018) for combining complementary recurrent architectures.
"""
import numpy as np
import pandas as pd
import warnings; warnings.filterwarnings("ignore")
from models.ml.lstm import train_lstm, forecast_lstm
from models.ml.gru  import train_gru,  forecast_gru


def lstm_gru_forecast(series: pd.Series, steps: int = 30,
                      epochs: int = 30, hidden: int = 64,
                      window: int = 20, exog=None):
    """
    Step 1: Train LSTM on log-returns → get in-sample predictions
    Step 2: Compute LSTM residuals
    Step 3: Train GRU on residuals → forecast residual correction
    Step 4: Final = LSTM forecast + GRU residual forecast

    Returns: final, lstm_pred, gru_residual_pred
    """
    # LSTM
    lstm_m, sc_y, sc_X, _, ins_lstm, act = train_lstm(
        series, window=window, epochs=epochs, hidden=hidden, exog=exog)
    lstm_fc = forecast_lstm(lstm_m, series, sc_y, steps=steps,
                             window=window, scaler_X=sc_X, exog=exog)

    # Residuals from LSTM in-sample
    n_ins = min(len(ins_lstm), len(act))
    resid = pd.Series(
        act[-n_ins:] - ins_lstm[-n_ins:],
        index=series.index[-n_ins:]
    ).dropna()

    if len(resid) < window + 10:
        return lstm_fc, lstm_fc, np.zeros(steps)

    # GRU on residuals
    gru_m, sc_y2, sc_X2, _, _, _ = train_gru(
        resid, window=window, epochs=epochs, hidden=hidden, exog=None)
    gru_res_fc = forecast_gru(gru_m, resid, sc_y2, steps=steps, window=window)

    final = lstm_fc + gru_res_fc[:len(lstm_fc)]
    return final, lstm_fc, gru_res_fc
