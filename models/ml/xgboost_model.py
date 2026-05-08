"""XGBoost — multivariate recursive forecasting on log-returns."""
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False


def _make_features(series, lags=20, exog=None):
    df = pd.DataFrame({"y": series.values})
    for i in range(1, lags + 1):
        df[f"lag_{i}"] = df["y"].shift(i)
    df["roll_mean_5"]  = df["y"].shift(1).rolling(5).mean()
    df["roll_mean_10"] = df["y"].shift(1).rolling(10).mean()
    df["roll_std_5"]   = df["y"].shift(1).rolling(5).std()
    df["roll_std_10"]  = df["y"].shift(1).rolling(10).std()
    if exog is not None:
        for col in exog.columns:
            df[col] = exog[col].values[:len(df)]
    df.dropna(inplace=True)
    X = df.drop(columns=["y"]).values
    y = df["y"].values
    return X, y, list(df.drop(columns=["y"]).columns)


def train_xgboost(series, lags=20, n_estimators=300, max_depth=4,
                  learning_rate=0.05, exog=None):
    if not XGB_AVAILABLE:
        raise ImportError("xgboost not installed")
    X, y, feat_names = _make_features(series, lags, exog)
    model = xgb.XGBRegressor(
        n_estimators=n_estimators, max_depth=max_depth,
        learning_rate=learning_rate, subsample=0.8,
        colsample_bytree=0.8, random_state=42, verbosity=0,
    )
    model.fit(X, y)
    insample = model.predict(X)
    actual   = y
    return model, lags, insample, actual


def forecast_xgboost(model, series, steps=30, lags=20, exog=None):
    history = list(series.values)
    preds   = []
    for i in range(steps):
        lag_vals = history[-lags:][::-1]
        roll5    = np.mean(history[-5:])
        roll10   = np.mean(history[-10:])
        std5     = np.std(history[-5:])
        std10    = np.std(history[-10:])
        base     = lag_vals + [roll5, roll10, std5, std10]
        if exog is not None:
            exog_row = exog.iloc[i].values.tolist() if i < len(exog) else [0] * len(exog.columns)
            base += exog_row
        row = np.array(base).reshape(1, -1)
        p   = float(model.predict(row)[0])
        preds.append(p)
        history.append(p)
    return np.array(preds)