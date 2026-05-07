"""
Information-Criteria Model Selection
=====================================
For every model that exposes a log-likelihood (statistical models),
we compute AIC / BIC / LogL directly from the fitted model.

For ML / tree models that don't have a closed-form likelihood,
we compute a *pseudo-likelihood* from in-sample residuals assuming
Gaussian errors:

    LogL_pseudo = -n/2 * log(2π σ²) - SSR / (2σ²)
    AIC_pseudo  = -2 * LogL + 2k
    BIC_pseudo  = -2 * LogL + k * log(n)

where σ² = SSR/n and k = number of model parameters (estimated).

Lower AIC/BIC = better fit (penalised for complexity).
Less negative LogL = better fit.

The function `run_model_selection(series, steps, models_to_run, ...)` returns:
    - ic_table  : pd.DataFrame with columns [Model, AIC, BIC, LogL, Detail]
    - best_model: str — name of the model with lowest AIC
    - results   : dict model_name → forecast array (so we can reuse without re-running)
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")


# ── Pseudo-IC for ML models ────────────────────────────────────────────────────

def _pseudo_ic(y_true: np.ndarray, y_pred: np.ndarray, n_params: int) -> dict:
    """Gaussian pseudo information criteria from in-sample residuals."""
    n   = len(y_true)
    ssr = np.sum((y_true - y_pred) ** 2)
    s2  = ssr / n
    if s2 <= 0:
        s2 = 1e-10
    llf = -n / 2 * np.log(2 * np.pi * s2) - ssr / (2 * s2)
    aic = -2 * llf + 2 * n_params
    bic = -2 * llf + n_params * np.log(n)
    return {"AIC": round(aic, 2), "BIC": round(bic, 2), "LogL": round(llf, 2)}


def _insample_pred_lstm(series, window=20, epochs=20, hidden=64):
    from models.ml.lstm import prepare_sequences, LSTMModel
    import torch, torch.nn as nn
    from sklearn.preprocessing import MinMaxScaler
    X, y, scaler = prepare_sequences(series, window)
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)
    model = LSTMModel(hidden=hidden)
    opt   = torch.optim.Adam(model.parameters(), lr=0.001)
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        p = model(X_t)
        nn.MSELoss()(p, y_t).backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        preds = scaler.inverse_transform(model(X_t).numpy()).flatten()
    true  = scaler.inverse_transform(y).flatten()
    k     = hidden * hidden * 4 * 2  # rough LSTM param count
    return true, preds, k


def _insample_pred_gru(series, window=20, epochs=20, hidden=64):
    from models.ml.gru import train_gru
    import torch
    from sklearn.preprocessing import MinMaxScaler
    # reuse train_gru with short epochs
    m, sc, _ = train_gru(series, window=window, epochs=epochs, hidden=hidden)
    from models.ml.gru import _prepare
    X, y, scaler = _prepare(series, window)
    import torch
    X_t = torch.tensor(X, dtype=torch.float32)
    m.eval()
    with torch.no_grad():
        preds = scaler.inverse_transform(m(X_t).numpy()).flatten()
    true  = scaler.inverse_transform(y).flatten()
    k     = hidden * hidden * 3 * 2
    return true, preds, k


def _insample_pred_transformer(series, window=20, epochs=20, d_model=32):
    from models.ml.transformer_model import train_transformer, _prepare
    import torch
    m, sc, _ = train_transformer(series, window=window, epochs=epochs, d_model=d_model)
    X, y, scaler = _prepare(series, window)
    X_t = torch.tensor(X, dtype=torch.float32)
    m.eval()
    with torch.no_grad():
        preds = scaler.inverse_transform(m(X_t).numpy()).flatten()
    true  = scaler.inverse_transform(y).flatten()
    k     = d_model * d_model * 4
    return true, preds, k


def _insample_pred_xgb(series, lags=20):
    from models.ml.xgboost_model import _make_features
    import xgboost as xgb
    X, y = _make_features(series, lags)
    model = xgb.XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.05,
                              verbosity=0, random_state=42)
    model.fit(X, y)
    preds = model.predict(X)
    k     = 100 * 4  # rough
    return y, preds, k


def _insample_pred_rf(series, lags=20):
    from models.ml.random_forest import _make_features
    from sklearn.ensemble import RandomForestRegressor
    X, y = _make_features(series, lags)
    model = RandomForestRegressor(n_estimators=50, n_jobs=-1, random_state=42)
    model.fit(X, y)
    preds = model.predict(X)
    k     = 50 * 10
    return y, preds, k


# ── Main entry point ──────────────────────────────────────────────────────────

def run_model_selection(
    series: pd.Series,
    models_to_evaluate: list,
    dl_epochs: int = 20,
    dl_hidden: int = 64,
    dl_window: int = 20,
) -> tuple[pd.DataFrame, str]:
    """
    Evaluate AIC / BIC / LogL for each model in `models_to_evaluate`.

    Returns:
        ic_table   : pd.DataFrame — one row per model, sorted by AIC ascending
        best_model : str — model name with lowest AIC
    """
    rows = []

    for model_name in models_to_evaluate:
        try:
            ic = _compute_ic(model_name, series, dl_epochs, dl_hidden, dl_window)
            ic["Model"] = model_name
            rows.append(ic)
        except Exception as e:
            rows.append({
                "Model": model_name,
                "AIC": np.nan, "BIC": np.nan, "LogL": np.nan,
                "detail": f"Error: {e}"
            })

    df = pd.DataFrame(rows)[["Model", "AIC", "BIC", "LogL", "detail"]]
    df = df.rename(columns={"detail": "Notes"})

    # Sort: lowest AIC first; NaN last
    df = df.sort_values("AIC", na_position="last").reset_index(drop=True)
    df["Rank"] = df.index + 1
    df = df[["Rank", "Model", "AIC", "BIC", "LogL", "Notes"]]

    best = df.dropna(subset=["AIC"])
    best_model = best.iloc[0]["Model"] if len(best) > 0 else models_to_evaluate[0]

    return df, best_model


def _compute_ic(model_name, series, dl_epochs, dl_hidden, dl_window):
    """Dispatch to the right IC computation per model."""

    # ── Statistical — exact likelihood ────────────────────────────────────────
    if model_name == "ARIMA":
        from models.statistical.arima import arima_ic
        return arima_ic(series)

    elif model_name == "ETS":
        from models.statistical.ets import ets_ic
        return ets_ic(series)

    elif model_name == "GARCH":
        from models.statistical.garch import garch_summary
        g = garch_summary(series)
        if "error" in g:
            raise RuntimeError(g["error"])
        return {"AIC": round(g["aic"], 2), "BIC": round(g["bic"], 2),
                "LogL": round(g["loglikelihood"], 2), "detail": "GARCH(1,1)"}

    elif model_name == "Prophet":
        # Prophet doesn't expose AIC; use MAP log-likelihood approximation
        from models.statistical.prophet_model import PROPHET_AVAILABLE
        if not PROPHET_AVAILABLE:
            raise RuntimeError("prophet not installed")
        from prophet import Prophet
        df_p = pd.DataFrame({"ds": series.index, "y": series.values}).reset_index(drop=True)
        m = Prophet(yearly_seasonality=True, weekly_seasonality=True,
                    daily_seasonality=False, uncertainty_samples=0)
        m.fit(df_p)
        fitted = m.predict(df_p)["yhat"].values
        n_params = 6  # intercept, trend rate, 2 yearly + 2 weekly harmonics (approx)
        ic = _pseudo_ic(series.values, fitted, n_params)
        ic["detail"] = "Prophet (pseudo-IC)"
        return ic

    # ── ML — pseudo-likelihood ────────────────────────────────────────────────
    elif model_name == "LSTM":
        true, preds, k = _insample_pred_lstm(series, window=dl_window,
                                             epochs=dl_epochs, hidden=dl_hidden)
        ic = _pseudo_ic(true, preds, k)
        ic["detail"] = f"LSTM pseudo-IC (h={dl_hidden})"
        return ic

    elif model_name == "GRU":
        true, preds, k = _insample_pred_gru(series, window=dl_window,
                                            epochs=dl_epochs, hidden=dl_hidden)
        ic = _pseudo_ic(true, preds, k)
        ic["detail"] = f"GRU pseudo-IC (h={dl_hidden})"
        return ic

    elif model_name == "Transformer":
        true, preds, k = _insample_pred_transformer(series, window=dl_window,
                                                    epochs=dl_epochs)
        ic = _pseudo_ic(true, preds, k)
        ic["detail"] = "Transformer pseudo-IC"
        return ic

    elif model_name == "XGBoost":
        from models.ml.xgboost_model import XGB_AVAILABLE
        if not XGB_AVAILABLE:
            raise RuntimeError("xgboost not installed")
        true, preds, k = _insample_pred_xgb(series, lags=20)
        ic = _pseudo_ic(true, preds, k)
        ic["detail"] = "XGBoost pseudo-IC"
        return ic

    elif model_name == "Random Forest":
        true, preds, k = _insample_pred_rf(series, lags=20)
        ic = _pseudo_ic(true, preds, k)
        ic["detail"] = "RF pseudo-IC"
        return ic

    # ── Hybrids — combine component ICs ──────────────────────────────────────
    elif model_name == "Hybrid ARIMA+LSTM":
        from models.statistical.arima import arima_ic
        ic_a = arima_ic(series)
        true, preds, k = _insample_pred_lstm(series, window=dl_window, epochs=dl_epochs, hidden=dl_hidden)
        ic_l = _pseudo_ic(true, preds, k)
        return {"AIC":  round((ic_a["AIC"]  + ic_l["AIC"])  / 2, 2),
                "BIC":  round((ic_a["BIC"]  + ic_l["BIC"])  / 2, 2),
                "LogL": round((ic_a["LogL"] + ic_l["LogL"]) / 2, 2),
                "detail": f"Hybrid: ARIMA+LSTM (avg IC)"}

    elif model_name == "Hybrid ETS+GRU":
        from models.statistical.ets import ets_ic
        ic_e = ets_ic(series)
        true, preds, k = _insample_pred_gru(series, window=dl_window, epochs=dl_epochs, hidden=dl_hidden)
        ic_g = _pseudo_ic(true, preds, k)
        return {"AIC":  round((ic_e["AIC"]  + ic_g["AIC"])  / 2, 2),
                "BIC":  round((ic_e["BIC"]  + ic_g["BIC"])  / 2, 2),
                "LogL": round((ic_e["LogL"] + ic_g["LogL"]) / 2, 2),
                "detail": "Hybrid: ETS+GRU (avg IC)"}

    elif model_name == "Hybrid Prophet+XGB":
        from models.ml.xgboost_model import XGB_AVAILABLE
        if not XGB_AVAILABLE:
            raise RuntimeError("xgboost not installed")
        from models.statistical.prophet_model import PROPHET_AVAILABLE
        if not PROPHET_AVAILABLE:
            raise RuntimeError("prophet not installed")
        # prophet pseudo-IC
        from prophet import Prophet
        df_p = pd.DataFrame({"ds": series.index, "y": series.values}).reset_index(drop=True)
        m = Prophet(uncertainty_samples=0)
        m.fit(df_p)
        fitted_p = m.predict(df_p)["yhat"].values
        ic_p = _pseudo_ic(series.values, fitted_p, 6)
        true, preds, k = _insample_pred_xgb(series)
        ic_x = _pseudo_ic(true, preds, k)
        return {"AIC":  round((ic_p["AIC"]  + ic_x["AIC"])  / 2, 2),
                "BIC":  round((ic_p["BIC"]  + ic_x["BIC"])  / 2, 2),
                "LogL": round((ic_p["LogL"] + ic_x["LogL"]) / 2, 2),
                "detail": "Hybrid: Prophet+XGB (avg IC)"}

    elif model_name == "Stacked Ensemble":
        # Use average IC of the 6 base models as a proxy
        base = ["ARIMA", "ETS", "LSTM", "GRU", "XGBoost", "Random Forest"]
        ics  = []
        for b in base:
            try:
                ics.append(_compute_ic(b, series, dl_epochs, dl_hidden, dl_window))
            except Exception:
                pass
        if not ics:
            raise RuntimeError("No base model ICs computed")
        return {"AIC":  round(np.mean([x["AIC"]  for x in ics]), 2),
                "BIC":  round(np.mean([x["BIC"]  for x in ics]), 2),
                "LogL": round(np.mean([x["LogL"] for x in ics]), 2),
                "detail": "Ensemble (avg base IC)"}

    else:
        raise ValueError(f"Unknown model: {model_name}")
        
