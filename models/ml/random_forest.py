"""
Random Forest forecasting via recursive multi-step prediction.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings("ignore")


def _make_features(series: pd.Series, lags: int = 20) -> tuple:
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


def train_rf(series: pd.Series, lags: int = 20, n_estimators: int = 200,
             max_depth: int = None, min_samples_leaf: int = 5):
    X, y = _make_features(series, lags)
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        n_jobs=-1,
        random_state=42,
    )
    model.fit(X, y)
    return model, lags


def forecast_rf(model, series: pd.Series, steps: int = 30, lags: int = 20) -> np.ndarray:
    history = list(series.values)
    preds   = []
    for _ in range(steps):
        lag_vals = history[-lags:][::-1]
        roll5    = np.mean(history[-5:])
        roll10   = np.mean(history[-10:])
        std5     = np.std(history[-5:])
        std10    = np.std(history[-10:])
        row      = np.array(lag_vals + [roll5, roll10, std5, std10]).reshape(1, -1)
        p        = float(model.predict(row)[0])
        preds.append(p)
        history.append(p)
    return np.array(preds)
