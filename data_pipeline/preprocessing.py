"""
preprocessing.py — log-returns transformation + inverse helpers.
ALL downstream models operate on log_returns (stationary, comparable across assets).
Close price is reconstructed post-forecast via cumulative sum of log-returns.
"""
import numpy as np
import pandas as pd


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["returns"]     = df["Close"].pct_change()
    df["log_returns"] = np.log(df["Close"] / df["Close"].shift(1))
    df.dropna(inplace=True)
    return df


def get_target_series(df: pd.DataFrame) -> pd.Series:
    """Return log_returns as the modelling target (stationary, unit-free)."""
    return df["log_returns"].copy()


def logret_to_price(log_ret_forecast: np.ndarray, last_price: float) -> np.ndarray:
    """
    Reconstruct price path from forecasted log-returns.

    P_t = P_{t-1} * exp(r_t)  ↔  cumsum of log-returns + log(last_price)
    """
    cumulative = np.cumsum(log_ret_forecast)
    return last_price * np.exp(cumulative)


def price_to_logret(prices: pd.Series) -> pd.Series:
    """Convert a price series to log-returns (drops first NaN)."""
    return np.log(prices / prices.shift(1)).dropna()
