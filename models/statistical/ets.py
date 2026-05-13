"""
ETS / Exponential Smoothing v5.2
=================================
Univariate: standard Holt damped trend.
Multivariate: ETS with exogenous regression (ETSx).
  Approach: fit a linear regression of log-returns on PCA components,
  then fit ETS on the residuals. Forecast = ETS residual forecast +
  regression prediction on future exogenous values.

  This is the "Regression + ETS on residuals" approach (also called
  ETSX in some literature), which is simpler and more stable than
  full Vector-ETS (which requires all series to be jointly stationary
  and has O(n²) parameters). The regression-ETS approach:
    1. Removes explainable variance via regression on exog
    2. Applies ETS to the unexplained residuals
    3. Combines both for the final forecast

  Recommended over Vector-ETS because:
    - PCA components are orthogonal → no multicollinearity in regression
    - ETS on residuals is more stable (smaller variance)
    - Interpretable: regression part = linear exog effects,
      ETS part = temporal persistence in unexplained component
"""
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


def _fit_ets(series):
    """Fit ETS with damped trend. Falls back to simple ETS on failure."""
    try:
        return ExponentialSmoothing(
            series, trend="add", damped_trend=True,
            initialization_method="estimated",
        ).fit(optimized=True)
    except Exception:
        try:
            return ExponentialSmoothing(
                series, trend="add", initialization_method="estimated",
            ).fit(optimized=True)
        except Exception:
            return ExponentialSmoothing(series).fit(optimized=True)


# ── Univariate ETS ────────────────────────────────────────────────────────────

def ets_forecast(series, steps=30, exog=None):
    """
    Univariate ETS or regression-ETS (ETSx) depending on whether exog provided.

    Returns: (pred, params_dict, insample_fitted, actual_values)
    """
    if exog is not None and isinstance(exog, pd.DataFrame) and len(exog.columns) > 0:
        return _etsx_forecast(series, exog, steps)

    # Pure univariate ETS
    model    = _fit_ets(series)
    pred     = np.array(model.forecast(steps))
    insample = model.fittedvalues.values
    actual   = series.values
    params   = {
        "alpha":  round(float(model.params.get("smoothing_level", 0)), 4),
        "beta":   round(float(model.params.get("smoothing_trend", 0)), 4),
        "phi":    round(float(model.params.get("damping_trend", 1.0)), 4),
        "AIC":    round(model.aic, 2),
        "mode":   "Univariate ETS",
    }
    return pred, params, insample, actual


def _etsx_forecast(series: pd.Series, exog: pd.DataFrame, steps: int):
    """
    Regression-ETS (ETSx) for multivariate mode.

    Step 1: Ridge regression of log-returns on PCA exog components
    Step 2: ETS on regression residuals
    Step 3: Forecast = ETS residual forecast + regression(last exog)
    """
    # Align series and exog
    common = series.index.intersection(exog.index)
    y      = series.loc[common].values
    X      = exog.loc[common].values

    # Step 1: Ridge regression
    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)
    reg    = Ridge(alpha=1.0)
    reg.fit(X_sc, y)
    y_hat_reg = reg.predict(X_sc)
    resid     = y - y_hat_reg

    # Step 2: ETS on residuals
    resid_series = pd.Series(resid, index=common)
    ets_model    = _fit_ets(resid_series)
    ets_resid_fc = np.array(ets_model.forecast(steps))

    # Step 3: future exog = last observed exog values (flat extrapolation of PCA components)
    X_future     = np.tile(X[-1], (steps, 1))
    X_future_sc  = scaler.transform(X_future)
    reg_future   = reg.predict(X_future_sc)

    pred     = reg_future + ets_resid_fc
    insample = y_hat_reg + ets_model.fittedvalues.values
    actual   = y

    params = {
        "alpha":       round(float(ets_model.params.get("smoothing_level", 0)), 4),
        "beta":        round(float(ets_model.params.get("smoothing_trend", 0)), 4),
        "reg_coefs":   len(reg.coef_),
        "mode":        "ETSx (Regression + ETS residuals)",
        "exog_dim":    exog.shape[1],
    }
    return pred, params, insample, actual


def ets_ic(series):
    m   = _fit_ets(series)
    n   = len(series)
    k   = len(m.params)
    llf = float(m.llf) if hasattr(m, "llf") else float(-m.sse)
    bic = -2 * llf + k * np.log(n)
    return {
        "AIC":    round(m.aic, 2),
        "BIC":    round(bic, 2),
        "LogL":   round(llf, 2),
        "detail": "ETS(A,Ad,N)",
    }
