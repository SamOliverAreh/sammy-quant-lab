"""
Hybrid Transformer + LSTM
=========================
Transformer captures global attention patterns across the sequence;
LSTM corrects the Transformer residuals with its sequential memory.
Used in Wu et al. 2021 (Autoformer) inspired decomposition approach,
and widely cited in financial TS forecasting literature.
"""
import numpy as np
import pandas as pd
import warnings; warnings.filterwarnings("ignore")
from models.ml.transformer_model import train_transformer, forecast_transformer
from models.ml.lstm import train_lstm, forecast_lstm


def transformer_lstm_forecast(series: pd.Series, steps: int = 30,
                               epochs: int = 30, d_model: int = 32,
                               hidden: int = 64, window: int = 20,
                               exog=None):
    """
    Step 1: Train Transformer on log-returns
    Step 2: Compute Transformer residuals
    Step 3: Train LSTM on residuals
    Step 4: Final = Transformer forecast + LSTM residual forecast
    """
    # Transformer
    tr_m, sc_y, sc_X, _, ins_tr, act = train_transformer(
        series, window=window, epochs=epochs, d_model=d_model, exog=exog)
    tr_fc = forecast_transformer(tr_m, series, sc_y, steps=steps,
                                  window=window, scaler_X=sc_X, exog=exog)

    # Residuals
    n_ins = min(len(ins_tr), len(act))
    resid = pd.Series(
        act[-n_ins:] - ins_tr[-n_ins:],
        index=series.index[-n_ins:]
    ).dropna()

    if len(resid) < window + 10:
        return tr_fc, tr_fc, np.zeros(steps)

    # LSTM on residuals
    lstm_m, sc_y2, sc_X2, _, _, _ = train_lstm(
        resid, window=window, epochs=epochs, hidden=hidden, exog=None)
    lstm_res_fc = forecast_lstm(lstm_m, resid, sc_y2, steps=steps, window=window)

    final = tr_fc + lstm_res_fc[:len(tr_fc)]
    return final, tr_fc, lstm_res_fc
