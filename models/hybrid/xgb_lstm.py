"""
Hybrid XGBoost + LSTM
=====================
XGBoost captures nonlinear lag-feature structure;
LSTM corrects residuals by learning temporal patterns XGB misses.
Referenced in Qiu et al. 2020 and used in competitive Kaggle TS solutions.
"""
import numpy as np
import pandas as pd
import warnings; warnings.filterwarnings("ignore")
from models.ml.xgboost_model import train_xgboost, forecast_xgboost, XGB_AVAILABLE
from models.ml.lstm import train_lstm, forecast_lstm


def xgb_lstm_forecast(series: pd.Series, steps: int = 30,
                       epochs: int = 30, hidden: int = 64,
                       window: int = 20, exog=None):
    """
    Step 1: XGBoost on log-returns with lag features
    Step 2: Compute XGB in-sample residuals
    Step 3: LSTM on residuals
    Step 4: Final = XGB forecast + LSTM residual forecast
    """
    if not XGB_AVAILABLE:
        raise ImportError("xgboost not installed — pip install xgboost")

    # XGBoost
    xgb_m, lags, ins_xgb, act = train_xgboost(series, exog=exog)
    xgb_fc = forecast_xgboost(xgb_m, series, steps=steps, lags=lags, exog=exog)

    # Residuals
    n_ins = min(len(ins_xgb), len(act))
    resid = pd.Series(
        act[-n_ins:] - ins_xgb[-n_ins:],
        index=series.index[-n_ins:]
    ).dropna()

    if len(resid) < window + 10:
        return xgb_fc, xgb_fc, np.zeros(steps)

    # LSTM on residuals
    lstm_m, sc_y, sc_X, _, _, _ = train_lstm(
        resid, window=window, epochs=epochs, hidden=hidden, exog=None)
    lstm_res_fc = forecast_lstm(lstm_m, resid, sc_y, steps=steps, window=window)

    final = xgb_fc + lstm_res_fc[:len(xgb_fc)]
    return final, xgb_fc, lstm_res_fc
