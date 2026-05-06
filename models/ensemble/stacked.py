"""
Stacked Ensemble Forecaster.

Level-0 base models: ARIMA, ETS, LSTM, GRU, XGBoost, Random Forest.
Level-1 meta-learner: Ridge regression trained on OOS base-model predictions.

Walk-forward cross-validation is used to generate the meta-training data
so there is no look-ahead bias.
"""
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler as SKScaler


# ── helpers ────────────────────────────────────────────────────────────────────

def _arima_pred(train, steps):
    from models.statistical.arima import arima_forecast
    try:
        p, _, _ = arima_forecast(train, steps=steps)
        return p
    except Exception:
        return np.full(steps, train.iloc[-1])


def _ets_pred(train, steps):
    from models.statistical.ets import ets_forecast
    try:
        p, _ = ets_forecast(train, steps=steps)
        return p
    except Exception:
        return np.full(steps, train.iloc[-1])


def _lstm_pred(train, steps):
    from models.ml.lstm import train_lstm, forecast_lstm
    try:
        m, sc, _ = train_lstm(train, epochs=20)
        return forecast_lstm(m, train, sc, steps=steps)
    except Exception:
        return np.full(steps, train.iloc[-1])


def _gru_pred(train, steps):
    from models.ml.gru import train_gru, forecast_gru
    try:
        m, sc, _ = train_gru(train, epochs=20)
        return forecast_gru(m, train, sc, steps=steps)
    except Exception:
        return np.full(steps, train.iloc[-1])


def _xgb_pred(train, steps):
    from models.ml.xgboost_model import train_xgboost, forecast_xgboost, XGB_AVAILABLE
    if not XGB_AVAILABLE:
        return np.full(steps, train.iloc[-1])
    try:
        m, lags = train_xgboost(train)
        return forecast_xgboost(m, train, steps=steps, lags=lags)
    except Exception:
        return np.full(steps, train.iloc[-1])


def _rf_pred(train, steps):
    from models.ml.random_forest import train_rf, forecast_rf
    try:
        m, lags = train_rf(train)
        return forecast_rf(m, train, steps=steps, lags=lags)
    except Exception:
        return np.full(steps, train.iloc[-1])


BASE_MODELS = {
    "ARIMA": _arima_pred,
    "ETS":   _ets_pred,
    "LSTM":  _lstm_pred,
    "GRU":   _gru_pred,
    "XGB":   _xgb_pred,
    "RF":    _rf_pred,
}


# ── public API ─────────────────────────────────────────────────────────────────

def stacked_ensemble_forecast(
    series: pd.Series,
    steps: int = 30,
    n_splits: int = 3,
    min_train: int = 200,
) -> tuple[np.ndarray, dict]:
    """
    Train a stacked ensemble and return final forecast + per-model predictions.

    Returns:
        final_pred (np.ndarray shape [steps])
        base_preds (dict model_name -> np.ndarray shape [steps])
    """
    n = len(series)

    # ── Step 1: walk-forward meta-dataset ─────────────────────────────────────
    meta_X_rows = []
    meta_y_rows = []

    fold_size = max(steps, (n - min_train) // (n_splits + 1))
    fold_starts = [min_train + i * fold_size for i in range(n_splits)]

    for fold_start in fold_starts:
        if fold_start + steps > n:
            break
        train_s = series.iloc[:fold_start]
        true_s  = series.iloc[fold_start: fold_start + steps].values

        row = []
        for name, fn in BASE_MODELS.items():
            pred = fn(train_s, steps)[:steps]
            if len(pred) < steps:
                pred = np.pad(pred, (0, steps - len(pred)), mode="edge")
            row.append(pred)
        meta_X_rows.append(np.stack(row, axis=1))  # (steps, n_models)
        meta_y_rows.append(true_s)

    if not meta_X_rows:
        # Not enough data for meta-training → simple average
        final_preds, base_preds = _simple_average(series, steps)
        return final_preds, base_preds

    meta_X = np.vstack(meta_X_rows)   # (n_splits * steps, n_models)
    meta_y = np.concatenate(meta_y_rows)

    # ── Step 2: fit meta-learner ───────────────────────────────────────────────
    sc = SKScaler()
    meta_X_sc = sc.fit_transform(meta_X)
    meta = Ridge(alpha=1.0)
    meta.fit(meta_X_sc, meta_y)

    # ── Step 3: final base predictions on full series ─────────────────────────
    base_preds: dict[str, np.ndarray] = {}
    final_X = []
    for name, fn in BASE_MODELS.items():
        p = fn(series, steps)[:steps]
        if len(p) < steps:
            p = np.pad(p, (0, steps - len(p)), mode="edge")
        base_preds[name] = p
        final_X.append(p)

    final_matrix = np.stack(final_X, axis=1)          # (steps, n_models)
    final_pred   = meta.predict(sc.transform(final_matrix))

    return final_pred, base_preds


def _simple_average(series, steps):
    """Fallback when not enough data for meta-training."""
    base_preds = {}
    for name, fn in BASE_MODELS.items():
        p = fn(series, steps)
        if len(p) < steps:
            p = np.pad(p, (0, steps - len(p)), mode="edge")
        base_preds[name] = p[:steps]
    final = np.mean(np.stack(list(base_preds.values())), axis=0)
    return final, base_preds
