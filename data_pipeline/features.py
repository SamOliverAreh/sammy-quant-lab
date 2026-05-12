"""
features.py v5.0
All indicator features computed from log_returns (stationary) where applicable.
Price-level indicators (MA, BB) kept on Close for visualisation but
the PRIMARY features used by models are log_return-based.
"""
import pandas as pd
import numpy as np


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # ── Log-return based features (used by models) ────────────────────────────
    lr = df["log_returns"]

    # Rolling mean / std of log-returns
    for w in [5, 10, 21, 63]:
        df[f"lr_mean_{w}"]  = lr.rolling(w).mean()
        df[f"lr_std_{w}"]   = lr.rolling(w).std()

    # Momentum (log-return over past k periods)
    for w in [5, 10, 21]:
        df[f"lr_mom_{w}"]   = lr.rolling(w).sum()

    # Z-score of log-return
    df["lr_zscore_21"] = (lr - lr.rolling(21).mean()) / (lr.rolling(21).std() + 1e-12)

    # Log-return RSI proxy (on log-returns instead of price)
    df["lr_rsi"] = _rsi(lr, 14)

    # MACD on log-returns
    ema12             = lr.ewm(span=12).mean()
    ema26             = lr.ewm(span=26).mean()
    df["lr_macd"]     = ema12 - ema26
    df["lr_macd_sig"] = df["lr_macd"].ewm(span=9).mean()
    df["lr_macd_hist"]= df["lr_macd"] - df["lr_macd_sig"]

    # Realised volatility (annualised)
    df["rv_10"]  = lr.rolling(10).std() * np.sqrt(252)
    df["rv_21"]  = lr.rolling(21).std() * np.sqrt(252)
    df["rv_63"]  = lr.rolling(63).std() * np.sqrt(252)

    # Volatility ratio (short/long — measures vol regime)
    df["vol_ratio"] = df["rv_10"] / (df["rv_63"] + 1e-12)

    # ── Price-level features (for visualisation / chart overlays only) ─────────
    for w in [21, 50, 200]:
        df[f"ma_{w}"] = df["Close"].rolling(w).mean()

    bb_mid         = df["Close"].rolling(20).mean()
    bb_std         = df["Close"].rolling(20).std()
    df["bb_upper"] = bb_mid + 2 * bb_std
    df["bb_lower"] = bb_mid - 2 * bb_std
    df["bb_width"] = df["bb_upper"] - df["bb_lower"]
    df["bb_pct"]   = (df["Close"] - df["bb_lower"]) / (df["bb_width"] + 1e-12)

    # ATR (normalised by close — unit-free)
    df["atr_norm"] = _atr(df, 14) / (df["Close"] + 1e-12)

    # Stochastic
    df["stoch_k"], df["stoch_d"] = _stochastic(df, 14)

    df.dropna(inplace=True)
    return df


# ── Model feature columns (log-return based, suitable for Granger + PCA) ─────
MODEL_FEATURE_COLS = [
    "lr_mean_5","lr_mean_10","lr_mean_21","lr_mean_63",
    "lr_std_5","lr_std_10","lr_std_21","lr_std_63",
    "lr_mom_5","lr_mom_10","lr_mom_21",
    "lr_zscore_21","lr_rsi",
    "lr_macd","lr_macd_sig","lr_macd_hist",
    "rv_10","rv_21","rv_63","vol_ratio",
    "bb_pct","atr_norm","stoch_k","stoch_d",
]


def get_model_features(df_feat: pd.DataFrame) -> pd.DataFrame:
    """Return only the log-return-based feature columns available in df_feat."""
    cols = [c for c in MODEL_FEATURE_COLS if c in df_feat.columns]
    return df_feat[cols].dropna()


def _rsi(series, period=14):
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / (loss + 1e-12)
    return 100 - (100 / (1 + rs))


def _atr(df, period=14):
    if "High" not in df.columns or "Low" not in df.columns:
        return df["Close"].rolling(period).std()
    high, low, close = df["High"], df["Low"], df["Close"]
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def _stochastic(df, period=14):
    if "High" not in df.columns or "Low" not in df.columns:
        return pd.Series(50, index=df.index), pd.Series(50, index=df.index)
    low_min  = df["Low"].rolling(period).min()
    high_max = df["High"].rolling(period).max()
    k = 100 * (df["Close"] - low_min) / (high_max - low_min + 1e-12)
    d = k.rolling(3).mean()
    return k, d
