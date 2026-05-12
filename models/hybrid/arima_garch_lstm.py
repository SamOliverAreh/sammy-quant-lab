"""
Hybrid ARIMA-GARCH + LSTM (Three-stage)
========================================
Stage 1: ARIMA removes linear mean structure
Stage 2: GARCH models the conditional volatility of residuals
Stage 3: LSTM learns remaining nonlinear structure in standardised residuals
Final forecast recombines all three components.

This follows the ARIMA-GARCH-Neural Network decomposition framework
referenced in Zhang (2003), Khashei & Bijari (2010), and the
volatility-filtered hybrid approach in Ding et al. (2015).
"""
import numpy as np
import pandas as pd
import warnings; warnings.filterwarnings("ignore")
from statsmodels.tsa.arima.model import ARIMA as _ARIMA
from models.statistical.arima import arima_forecast, find_best_arima
from models.statistical.garch import garch_insample_vol, garch_volatility
from models.ml.lstm import train_lstm, forecast_lstm


def arima_garch_lstm_forecast(series: pd.Series, steps: int = 30,
                               epochs: int = 30, hidden: int = 64,
                               window: int = 20, exog=None):
    """
    Returns: final, arima_pred, garch_vol_fc, lstm_vol_adj_pred, order
    """
    # Stage 1: ARIMA
    arima_pred, order, ci, ins, act = arima_forecast(series, steps=steps, exog=exog)
    fitted_arima = _ARIMA(series, order=order).fit()
    arima_resid  = pd.Series(fitted_arima.resid, index=series.index).dropna()

    # Stage 2: GARCH vol on ARIMA residuals
    try:
        garch_vol_insample = garch_insample_vol(arima_resid)
        garch_vol_fc       = garch_volatility(arima_resid, horizon=steps)
        # Standardise residuals by conditional vol
        std_resid = arima_resid / (garch_vol_insample + 1e-8)
        std_resid = std_resid.dropna().replace([np.inf, -np.inf], 0)
    except Exception:
        std_resid    = arima_resid.copy()
        garch_vol_fc = np.full(steps, float(arima_resid.std()))

    if len(std_resid) < window + 10:
        return arima_pred, arima_pred, garch_vol_fc, np.zeros(steps), order

    # Stage 3: LSTM on standardised residuals
    lstm_m, sc_y, sc_X, _, _, _ = train_lstm(
        std_resid, window=window, epochs=epochs, hidden=hidden, exog=None)
    lstm_std_fc = forecast_lstm(lstm_m, std_resid, sc_y, steps=steps, window=window)

    # Rescale LSTM output by GARCH forecast vol
    lstm_adj = lstm_std_fc[:steps] * garch_vol_fc[:steps]

    # Final: ARIMA mean + GARCH-scaled LSTM nonlinear component
    final = arima_pred + lstm_adj[:len(arima_pred)]
    return final, arima_pred, garch_vol_fc, lstm_adj, order
