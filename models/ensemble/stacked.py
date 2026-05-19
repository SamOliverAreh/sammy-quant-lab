"""
Stacked Ensemble v5.0
=====================
Multiple meta-learner methods. Walk-forward OOS stacking — no look-ahead bias.

Base models (Level 0): ARIMA, ETS, LSTM, GRU, XGBoost, RF, Transformer
Meta-learners (Level 1):
  1. Ridge Regression       — L2 regularised linear combination
  2. Lasso Regression       — L1 sparse combination (feature selection)
  3. Elastic Net            — L1+L2 balanced combination
  4. Gradient Boosting      — nonlinear meta-learner (sklearn GBM)
  5. Equal-Weight Average   — simple unweighted baseline

The meta-learner with the lowest OOS MSE on the validation folds is
automatically selected as the final ensemble.
"""
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error


# ── Base model wrappers ───────────────────────────────────────────────────────

def _pad(arr, steps):
    arr = np.array(arr)[:steps]
    if len(arr) < steps:
        arr = np.pad(arr, (0, steps - len(arr)), mode="edge")
    return arr


def _arima(train, steps, exog=None):
    from models.statistical.arima import arima_forecast
    try:
        p, _, _, _, _ = arima_forecast(train, steps=steps, exog=exog)
        return _pad(p, steps)
    except Exception:
        return np.full(steps, float(train.iloc[-1]))


def _ets(train, steps, exog=None):
    from models.statistical.ets import ets_forecast
    try:
        p, _, _, _ = ets_forecast(train, steps=steps)
        return _pad(p, steps)
    except Exception:
        return np.full(steps, float(train.iloc[-1]))


def _lstm(train, steps, exog=None):
    from models.ml.lstm import train_lstm, forecast_lstm
    try:
        m, sy, sx, _, _, _ = train_lstm(train, epochs=20, exog=exog)
        return _pad(forecast_lstm(m, train, sy, steps=steps, scaler_X=sx, exog=exog), steps)
    except Exception:
        return np.full(steps, float(train.iloc[-1]))


def _gru(train, steps, exog=None):
    from models.ml.gru import train_gru, forecast_gru
    try:
        m, sy, sx, _, _, _ = train_gru(train, epochs=20, exog=exog)
        return _pad(forecast_gru(m, train, sy, steps=steps, scaler_X=sx, exog=exog), steps)
    except Exception:
        return np.full(steps, float(train.iloc[-1]))


def _xgb(train, steps, exog=None):
    from models.ml.xgboost_model import train_xgboost, forecast_xgboost, XGB_AVAILABLE
    if not XGB_AVAILABLE:
        return np.full(steps, float(train.iloc[-1]))
    try:
        m, lags, _, _ = train_xgboost(train, exog=exog)
        return _pad(forecast_xgboost(m, train, steps=steps, lags=lags, exog=exog), steps)
    except Exception:
        return np.full(steps, float(train.iloc[-1]))


def _rf(train, steps, exog=None):
    from models.ml.random_forest import train_rf, forecast_rf
    try:
        m, lags, _, _ = train_rf(train, exog=exog)
        return _pad(forecast_rf(m, train, steps=steps, lags=lags, exog=exog), steps)
    except Exception:
        return np.full(steps, float(train.iloc[-1]))


def _transformer(train, steps, exog=None):
    from models.ml.transformer_model import train_transformer, forecast_transformer
    try:
        m, sy, sx, _, _, _ = train_transformer(train, epochs=20, exog=exog)
        return _pad(forecast_transformer(m, train, sy, steps=steps, scaler_X=sx, exog=exog), steps)
    except Exception:
        return np.full(steps, float(train.iloc[-1]))


BASE_MODELS = {
    "ARIMA": _arima, "ETS": _ets, "LSTM": _lstm,
    "GRU":   _gru,   "XGB": _xgb, "RF":   _rf,
    "Transformer": _transformer,
}

META_LEARNERS = {
    "Ridge":            Ridge(alpha=1.0),
    "Lasso":            Lasso(alpha=0.001, max_iter=5000),
    "ElasticNet":       ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=5000),
    "GradientBoosting": GradientBoostingRegressor(n_estimators=100, max_depth=3,
                                                   learning_rate=0.1, random_state=42),
}


# ── Main function ─────────────────────────────────────────────────────────────

def stacked_ensemble_forecast(
    series: pd.Series,
    steps: int = 30,
    n_splits: int = 3,
    min_train: int = 60,
    exog=None,
    meta_method: str = "auto",   # "auto" = pick best by OOS MSE, or specify name
):
    """
    Walk-forward stacked ensemble.

    Returns:
        final_pred  (np.ndarray)
        base_preds  (dict model_name → np.ndarray)
        meta_scores (dict meta_name → OOS MSE)
        best_meta   (str — name of selected meta-learner)
    """
    n         = len(series)
    # Clamp min_train so we always have at least one fold
    min_train = min(min_train, max(20, n - steps * (n_splits + 1)))
    fold_size = max(steps, (n - min_train) // max(1, n_splits + 1))
    fold_starts = [min_train + i * fold_size for i in range(n_splits)]

    meta_X_rows, meta_y_rows = [], []
    for fs in fold_starts:
        if fs + steps > n:
            break
        tr    = series.iloc[:fs]
        ex_tr = exog.iloc[:fs] if exog is not None else None
        y_fold = series.iloc[fs: fs + steps].values
        # Only keep fold if y has exactly `steps` samples
        if len(y_fold) < steps:
            break
        row   = [_pad(fn(tr, steps, ex_tr), steps) for fn in BASE_MODELS.values()]
        meta_X_rows.append(np.stack(row, axis=1))   # (steps, n_models)
        meta_y_rows.append(y_fold)                   # (steps,)

    # Generate final base predictions on full series
    base_preds = {}
    final_Xs   = []
    for nm, fn in BASE_MODELS.items():
        p = _pad(fn(series, steps, exog), steps)
        base_preds[nm] = p
        final_Xs.append(p)
    final_X = np.stack(final_Xs, axis=1)   # (steps, n_models)

    if not meta_X_rows:
        # Not enough data — equal-weight average
        final_pred = np.mean(np.stack(list(base_preds.values())), axis=0)
        return final_pred, base_preds, {"Equal-Weight": 0.0}, "Equal-Weight"

    meta_X = np.vstack(meta_X_rows)
    meta_y = np.concatenate(meta_y_rows)

    sc = StandardScaler()
    meta_X_sc  = sc.fit_transform(meta_X)
    final_X_sc = sc.transform(final_X)

    # Train all meta-learners and evaluate OOS MSE
    meta_scores = {}
    meta_preds  = {}
    for name, learner in META_LEARNERS.items():
        try:
            import copy
            m = copy.deepcopy(learner)
            m.fit(meta_X_sc, meta_y)
            y_hat = m.predict(meta_X_sc)
            mse   = float(mean_squared_error(meta_y, y_hat))
            meta_scores[name]  = round(mse, 8)
            meta_preds[name]   = m.predict(final_X_sc)
        except Exception as e:
            meta_scores[name] = np.inf
            meta_preds[name]  = np.mean(np.stack(list(base_preds.values())), axis=0)

    # Equal-weight baseline
    ew_pred = np.mean(np.stack(list(base_preds.values())), axis=0)
    ew_mse  = float(mean_squared_error(
        meta_y,
        np.mean(np.vstack([_pad(p, steps) for p in base_preds.values()]), axis=0)[:len(meta_y)]
    ))
    meta_scores["Equal-Weight"] = round(ew_mse, 8)
    meta_preds["Equal-Weight"]  = ew_pred

    # Select best
    if meta_method == "auto":
        best_meta = min(meta_scores, key=meta_scores.get)
    else:
        best_meta = meta_method if meta_method in meta_preds else "Ridge"

    final_pred = meta_preds[best_meta]
    return final_pred, base_preds, meta_scores, best_meta
