"""
Prophet forecasting model (Meta/Facebook).
Handles trend, seasonality, holidays automatically.
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


def prophet_forecast(series: pd.Series, steps: int = 30, freq: str = "B") -> tuple:
    """
    Fit Prophet model and return (predictions, lower, upper, components_df).

    Args:
        series: pd.Series with DatetimeIndex
        steps:  forecast horizon (business days by default)
        freq:   pandas frequency string for future dates

    Returns:
        pred (np.ndarray), lower (np.ndarray), upper (np.ndarray), forecast_df (pd.DataFrame)
    """
    if not PROPHET_AVAILABLE:
        raise ImportError("prophet not installed. Run: pip install prophet")

    df_p = pd.DataFrame({"ds": series.index, "y": series.values}).reset_index(drop=True)

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=10,
        interval_width=0.95,
        uncertainty_samples=200,
    )
    model.fit(df_p)

    future = model.make_future_dataframe(periods=steps, freq=freq)
    forecast = model.predict(future)

    future_fc = forecast.tail(steps)
    pred  = future_fc["yhat"].values
    lower = future_fc["yhat_lower"].values
    upper = future_fc["yhat_upper"].values

    return pred, lower, upper, future_fc[["ds", "yhat", "yhat_lower", "yhat_upper", "trend"]]
    
