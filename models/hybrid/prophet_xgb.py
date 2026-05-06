"""Hybrid Prophet + XGBoost: trend/seasonality + ML residuals."""
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")


def prophet_xgb_forecast(series: pd.Series, steps: int = 30):
    from models.statistical.prophet_model import prophet_forecast, PROPHET_AVAILABLE
    from models.ml.xgboost_model import train_xgboost, forecast_xgboost, XGB_AVAILABLE

    if not PROPHET_AVAILABLE:
        raise ImportError("prophet not installed")
    if not XGB_AVAILABLE:
        raise ImportError("xgboost not installed")

    # Step 1: Prophet forecast + in-sample fit
    prophet_pred, lower, upper, fc_df = prophet_forecast(series, steps=steps)

    # Reconstruct in-sample residuals from Prophet trend
    df_p = pd.DataFrame({"ds": series.index, "y": series.values}).reset_index(drop=True)
    from prophet import Prophet
    m = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
    m.fit(df_p)
    in_sample = m.predict(df_p)
    fitted_vals = in_sample["yhat"].values[: len(series)]
    resid = pd.Series(series.values - fitted_vals, index=series.index)

    # Step 2: XGBoost on residuals
    xgb_model, lags = train_xgboost(resid)
    xgb_pred = forecast_xgboost(xgb_model, resid, steps=steps, lags=lags)

    final = prophet_pred + xgb_pred[: len(prophet_pred)]
    return final, prophet_pred, xgb_pred, lower, upper
