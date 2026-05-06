import pandas as pd
import numpy as np


def create_features(df):
    df = df.copy()

    # Moving averages
    df["ma_7"] = df["Close"].rolling(7).mean()
    df["ma_21"] = df["Close"].rolling(21).mean()
    df["ma_30"] = df["Close"].rolling(30).mean()
    df["ma_50"] = df["Close"].rolling(50).mean()

    # Volatility
    df["volatility_10"] = df["returns"].rolling(10).std()
    df["volatility_30"] = df["returns"].rolling(30).std()

    # RSI
    df["rsi"] = _rsi(df["Close"], 14)

    # Bollinger Bands
    bb_mid = df["Close"].rolling(20).mean()
    bb_std = df["Close"].rolling(20).std()
    df["bb_upper"] = bb_mid + 2 * bb_std
    df["bb_lower"] = bb_mid - 2 * bb_std
    df["bb_width"] = df["bb_upper"] - df["bb_lower"]

    # MACD
    ema12 = df["Close"].ewm(span=12).mean()
    ema26 = df["Close"].ewm(span=26).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9).mean()

    df.dropna(inplace=True)
    return df


def _rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))
