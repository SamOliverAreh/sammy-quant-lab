import pandas as pd
import numpy as np


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Moving averages
    for w in [7, 21, 30, 50, 200]:
        df[f"ma_{w}"] = df["Close"].rolling(w).mean()

    # EMA
    for w in [12, 26, 50]:
        df[f"ema_{w}"] = df["Close"].ewm(span=w).mean()

    # Volatility
    df["volatility_10"] = df["returns"].rolling(10).std()
    df["volatility_30"] = df["returns"].rolling(30).std()

    # RSI
    df["rsi"] = _rsi(df["Close"], 14)

    # Bollinger Bands
    bb_mid           = df["Close"].rolling(20).mean()
    bb_std           = df["Close"].rolling(20).std()
    df["bb_upper"]   = bb_mid + 2 * bb_std
    df["bb_lower"]   = bb_mid - 2 * bb_std
    df["bb_width"]   = df["bb_upper"] - df["bb_lower"]
    df["bb_pct"]     = (df["Close"] - df["bb_lower"]) / (df["bb_width"] + 1e-12)

    # MACD
    ema12            = df["Close"].ewm(span=12).mean()
    ema26            = df["Close"].ewm(span=26).mean()
    df["macd"]       = ema12 - ema26
    df["macd_signal"]= df["macd"].ewm(span=9).mean()
    df["macd_hist"]  = df["macd"] - df["macd_signal"]

    # ATR
    df["atr"] = _atr(df, 14)

    # Stochastic
    df["stoch_k"], df["stoch_d"] = _stochastic(df, 14)

    df.dropna(inplace=True)
    return df


def _rsi(series, period=14):
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / (loss + 1e-12)
    return 100 - (100 / (1 + rs))


def _atr(df, period=14):
    high, low, close = df["High"], df["Low"], df["Close"]
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def _stochastic(df, period=14):
    low_min  = df["Low"].rolling(period).min()
    high_max = df["High"].rolling(period).max()
    k = 100 * (df["Close"] - low_min) / (high_max - low_min + 1e-12)
    d = k.rolling(3).mean()
    return k, d
