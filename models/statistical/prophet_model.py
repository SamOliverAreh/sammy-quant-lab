"""Prophet — multivariate exog support via additional_regressors."""
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


def prophet_forecast(series: pd.Series, steps: int = 30,
                     freq: str = "B", exog: pd.DataFrame = None):
    if not PROPHET_AVAILABLE:
        raise ImportError("prophet not installed")

    df_p = pd.DataFrame({"ds": series.index, "y": series.values}).reset_index(drop=True)
    exog_cols = []

    if exog is not None and len(exog.columns) > 0:
        common       = series.index.intersection(exog.index)
        exog_aligned = exog.loc[common].reset_index(drop=True)
        df_p         = df_p.iloc[:len(exog_aligned)].copy()
        for col in exog.columns:
            df_p[col] = exog_aligned[col].values
            exog_cols.append(col)

    model = Prophet(yearly_seasonality=True, weekly_seasonality=True,
                    daily_seasonality=False, changepoint_prior_scale=0.05,
                    interval_width=0.95, uncertainty_samples=200)
    for col in exog_cols:
        model.add_regressor(col)
    model.fit(df_p)

    future = model.make_future_dataframe(periods=steps, freq=freq)
    for col in exog_cols:
        future[col] = float(df_p[col].iloc[-1])  # flat extrapolation

    forecast  = model.predict(future)
    future_fc = forecast.tail(steps)
    return (future_fc["yhat"].values,
            future_fc["yhat_lower"].values,
            future_fc["yhat_upper"].values,
            future_fc[["ds","yhat","yhat_lower","yhat_upper","trend"]])
