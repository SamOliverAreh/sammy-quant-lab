"""
XGBoost forecasting via recursive multi-step prediction.
Features: lagged prices + rolling stats.
"""
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False


def _make_features(series: pd.Series, lags=20) -> tuple:
    df = pd.DataFrame({"y": series.values})
    for i in range(1, lags + 1):
        df[f"lag_{i}"] = df["y"].shift(i)
    df["roll_mean_5"]  = df["y"].shift(1).rolling(5).mean()
    df["roll_mean_10"] = df["y"].shift(1).rolling(10).mean()
    df["roll_std_5"]   = df["y"].shift(1).rolling(5).std()
    df["roll_std_10"]  = df["y"].shift(1).rolling(10).std()
    df.dropna(inplace=True)
    X = df.drop(columns=["y"]).values
    y = df["y"].values
    return X, y


def train_xgboost(series: pd.Series, lags: int = 20, n_estimators: int = 300,
                  max_depth: int = 4, learning_rate: float = 0.05):
    if not XGB_AVAILABLE:
        raise ImportError("xgboost not installed. Run: pip install xgboost")
    X, y = _make_features(series, lags)
    model = xgb.XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0,
    )
    model.fit(X, y)
    return model, lags


def forecast_xgboost(model, series: pd.Series, steps: int = 30, lags: int = 20) -> np.ndarray:
    history = list(series.values)
    preds   = []
    for _ in range(steps):
        lag_vals  = history[-lags:][::-1]
        roll5     = np.mean(history[-5:])
        roll10    = np.mean(history[-10:])
        std5      = np.std(history[-5:])
        std10     = np.std(history[-10:])
        row       = np.array(lag_vals + [roll5, roll10, std5, std10]).reshape(1, -1)
        p         = float(model.predict(row)[0])
        preds.append(p)
        history.append(p)
    return np.array(preds)
    
