"""
Prophet forecasting model v5.2
================================
Fixed:
- Frequency detection: auto-detect business day vs calendar day vs weekly
- Log-returns scale: Prophet works well on any scale; no rescaling needed
- Exog via add_regressor with proper future filling
- Suppress verbose output
- Seasonality: disable yearly/weekly for short series (<2 years / <2 weeks)
"""
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    try:
        from fbprophet import Prophet
        PROPHET_AVAILABLE = True
    except ImportError:
        PROPHET_AVAILABLE = False

import logging
logging.getLogger("prophet").setLevel(logging.ERROR)
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)


def _detect_freq(index: pd.DatetimeIndex) -> str:
    """
    Auto-detect the dominant frequency of the series index.
    Returns a pandas frequency string suitable for make_future_dataframe.
    """
    if len(index) < 2:
        return "B"
    gaps = pd.Series(index).diff().dropna().dt.days
    median_gap = int(gaps.median())
    if median_gap <= 1:
        return "D"    # daily including weekends (crypto)
    elif median_gap <= 3:
        return "B"    # business days
    elif median_gap <= 5:
        return "W"    # weekly
    else:
        return "MS"   # monthly


def prophet_forecast(
    series: pd.Series,
    steps: int = 30,
    freq: str = None,          # None = auto-detect
    exog: pd.DataFrame = None,
) -> tuple:
    """
    Fit Prophet on log-returns and forecast.

    Args:
        series: pd.Series with DatetimeIndex of log-returns
        steps:  forecast horizon
        freq:   pandas frequency string (None = auto-detect)
        exog:   optional pd.DataFrame of exogenous PCA components

    Returns:
        pred (np.ndarray), lower (np.ndarray), upper (np.ndarray), future_df
    """
    if not PROPHET_AVAILABLE:
        raise ImportError("prophet not installed. Run: pip install prophet")

    # ── Auto-detect frequency ────────────────────────────────────────────────
    if freq is None:
        freq = _detect_freq(series.index)

    # ── Check series length for seasonality ──────────────────────────────────
    n_days = (series.index[-1] - series.index[0]).days
    use_yearly  = n_days >= 365 * 2       # need ≥ 2 years for yearly seasonality
    use_weekly  = n_days >= 14            # need ≥ 2 weeks for weekly

    # ── Build training dataframe ─────────────────────────────────────────────
    df_p = pd.DataFrame({
        "ds": series.index,
        "y":  series.values,
    }).reset_index(drop=True)

    exog_cols = []
    if exog is not None and isinstance(exog, pd.DataFrame) and len(exog.columns) > 0:
        common       = series.index.intersection(exog.index)
        exog_aligned = exog.loc[common].reset_index(drop=True)
        df_p         = df_p.iloc[:len(exog_aligned)].copy().reset_index(drop=True)
        for col in exog.columns:
            if len(exog_aligned) == len(df_p):
                df_p[col] = exog_aligned[col].values
                exog_cols.append(col)

    # ── Fit Prophet ──────────────────────────────────────────────────────────
    model = Prophet(
        yearly_seasonality  = use_yearly,
        weekly_seasonality  = use_weekly,
        daily_seasonality   = False,
        changepoint_prior_scale    = 0.05,
        seasonality_prior_scale    = 1.0,
        interval_width      = 0.95,
        uncertainty_samples = 100,
    )

    for col in exog_cols:
        model.add_regressor(col, standardize=True)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(df_p)

    # ── Make future dataframe ────────────────────────────────────────────────
    future = model.make_future_dataframe(periods=steps, freq=freq)

    # Fill future exog with last observed value (flat extrapolation)
    for col in exog_cols:
        last_val = float(df_p[col].iloc[-1])
        future[col] = last_val

    # ── Predict ──────────────────────────────────────────────────────────────
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        forecast = model.predict(future)

    future_fc = forecast.tail(steps)

    pred  = np.array(future_fc["yhat"].values,       dtype=float)
    lower = np.array(future_fc["yhat_lower"].values,  dtype=float)
    upper = np.array(future_fc["yhat_upper"].values,  dtype=float)

    out_df = future_fc[["ds", "yhat", "yhat_lower", "yhat_upper", "trend"]].copy()

    return pred, lower, upper, out_df
