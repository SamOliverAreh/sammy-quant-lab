"""Stacked Ensemble v4.2 — multivariate exog + corrected signatures."""
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler as SKScaler


def _pad(arr, steps):
    arr = np.array(arr)[:steps]
    if len(arr) < steps:
        arr = np.pad(arr, (0, steps-len(arr)), mode="edge")
    return arr

def _arima(train, steps, exog=None):
    from models.statistical.arima import arima_forecast
    try:
        p, _, _, _, _ = arima_forecast(train, steps=steps, exog=exog)
        return p
    except Exception:
        return np.full(steps, float(train.iloc[-1]))

def _ets(train, steps, exog=None):
    from models.statistical.ets import ets_forecast
    try:
        p, _, _, _ = ets_forecast(train, steps=steps)
        return p
    except Exception:
        return np.full(steps, float(train.iloc[-1]))

def _lstm(train, steps, exog=None):
    from models.ml.lstm import train_lstm, forecast_lstm
    try:
        m, sy, sx, _, _, _ = train_lstm(train, epochs=20, exog=exog)
        return forecast_lstm(m, train, sy, steps=steps, scaler_X=sx, exog=exog)
    except Exception:
        return np.full(steps, float(train.iloc[-1]))

def _gru(train, steps, exog=None):
    from models.ml.gru import train_gru, forecast_gru
    try:
        m, sy, sx, _, _, _ = train_gru(train, epochs=20, exog=exog)
        return forecast_gru(m, train, sy, steps=steps, scaler_X=sx, exog=exog)
    except Exception:
        return np.full(steps, float(train.iloc[-1]))

def _xgb(train, steps, exog=None):
    from models.ml.xgboost_model import train_xgboost, forecast_xgboost, XGB_AVAILABLE
    if not XGB_AVAILABLE:
        return np.full(steps, float(train.iloc[-1]))
    try:
        m, lags, _, _ = train_xgboost(train, exog=exog)
        return forecast_xgboost(m, train, steps=steps, lags=lags, exog=exog)
    except Exception:
        return np.full(steps, float(train.iloc[-1]))

def _rf(train, steps, exog=None):
    from models.ml.random_forest import train_rf, forecast_rf
    try:
        m, lags, _, _ = train_rf(train, exog=exog)
        return forecast_rf(m, train, steps=steps, lags=lags, exog=exog)
    except Exception:
        return np.full(steps, float(train.iloc[-1]))

BASE = {"ARIMA": _arima, "ETS": _ets, "LSTM": _lstm,
        "GRU":   _gru,   "XGB": _xgb, "RF":   _rf}


def stacked_ensemble_forecast(series, steps=30, n_splits=3,
                               min_train=200, exog=None):
    n            = len(series)
    meta_X_rows, meta_y_rows = [], []
    fold_size    = max(steps, (n-min_train)//(n_splits+1))
    fold_starts  = [min_train + i*fold_size for i in range(n_splits)]

    for fs in fold_starts:
        if fs + steps > n:
            break
        tr  = series.iloc[:fs]
        ex_tr = exog.iloc[:fs] if exog is not None else None
        row = [_pad(fn(tr, steps, ex_tr), steps) for fn in BASE.values()]
        meta_X_rows.append(np.stack(row, axis=1))
        meta_y_rows.append(series.iloc[fs:fs+steps].values)

    if not meta_X_rows:
        return _simple_avg(series, steps, exog)

    meta_X = np.vstack(meta_X_rows)
    meta_y = np.concatenate(meta_y_rows)
    sc     = SKScaler()
    meta   = Ridge(alpha=1.0)
    meta.fit(sc.fit_transform(meta_X), meta_y)

    base_preds = {nm: _pad(fn(series, steps, exog), steps) for nm, fn in BASE.items()}
    final_X    = np.stack(list(base_preds.values()), axis=1)
    final_pred = meta.predict(sc.transform(final_X))
    return final_pred, base_preds


def _simple_avg(series, steps, exog=None):
    bp    = {nm: _pad(fn(series, steps, exog), steps) for nm, fn in BASE.items()}
    final = np.mean(np.stack(list(bp.values())), axis=0)
    return final, bp
