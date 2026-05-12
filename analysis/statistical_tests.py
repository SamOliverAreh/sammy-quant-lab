"""
statistical_tests.py v5.0
All tests on log-returns. Fixed ADF/KPSS logic and verdict table.

ADF:  H0 = unit root (NON-stationary). Reject H0 → STATIONARY.
KPSS: H0 = STATIONARY.                 Reject H0 → NON-stationary.

Conflict resolution table:
  ADF reject + KPSS fail-to-reject  → Strongly Stationary
  ADF reject + KPSS reject          → Trend-Stationary (detrend recommended)
  ADF fail   + KPSS fail-to-reject  → Difference-Stationary (difference recommended)
  ADF fail   + KPSS reject          → Non-Stationary (both agree on non-stationarity)
"""
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from scipy.stats import jarque_bera


def adf_test(series, alpha=0.05):
    """
    H0: Series has a unit root (NON-stationary).
    Reject H0 (p < alpha) → series IS stationary.
    """
    res  = adfuller(series.dropna(), autolag="AIC")
    stat, p, _, nobs, cv, _ = res
    reject = p < alpha   # reject H0 → stationary
    return {
        "test":        "ADF (Augmented Dickey-Fuller)",
        "H0":          "Series has a unit root (Non-stationary)",
        "statistic":   round(stat, 4),
        "p_value":     round(p, 4),
        "n_obs":       nobs,
        "crit_1pct":   round(cv["1%"], 4),
        "crit_5pct":   round(cv["5%"], 4),
        "crit_10pct":  round(cv["10%"], 4),
        "reject_H0":   reject,
        "conclusion":  "✅ Stationary (reject H0)" if reject else "⚠️ Non-stationary (fail to reject H0)",
        "interpretation": (
            "ADF rejects the unit root hypothesis → series is stationary. "
            "Mean and variance are stable over time — suitable for modelling without differencing."
            if reject else
            "ADF fails to reject the unit root → evidence of non-stationarity. "
            "Log-returns are usually stationary; if this fails, inspect for structural breaks or outliers."
        ),
    }


def kpss_test(series, alpha=0.05):
    """
    H0: Series IS stationary.
    Reject H0 (p < alpha) → series is NON-stationary.
    Note: KPSS p-values are bounded [0.01, 0.10] by statsmodels interpolation.
    """
    stat, p, lags, cv = kpss(series.dropna(), regression="c", nlags="auto")
    reject = p < alpha   # reject H0 → NON-stationary
    return {
        "test":       "KPSS",
        "H0":         "Series is stationary",
        "statistic":  round(stat, 4),
        "p_value":    round(p, 4),
        "lags_used":  lags,
        "crit_5pct":  round(cv["5%"], 4),
        "reject_H0":  reject,
        "conclusion": "⚠️ Non-stationary (reject H0)" if reject else "✅ Stationary (fail to reject H0)",
        "interpretation": (
            "KPSS rejects stationarity (p < α). "
            "This contradicts an ADF rejection → likely trend-stationary: the series is "
            "stationary around a deterministic trend. Consider detrending."
            if reject else
            "KPSS fails to reject stationarity (p ≥ α). "
            "Combined with ADF rejection this is strong evidence the series is truly stationary."
        ),
    }


def ljungbox_test(series, lags=10, alpha=0.05):
    """H0: No serial autocorrelation up to lag k."""
    res    = acorr_ljungbox(series.dropna(), lags=[lags], return_df=True)
    stat   = float(res["lb_stat"].iloc[-1])
    p      = float(res["lb_pvalue"].iloc[-1])
    reject = p < alpha
    return {
        "test":        f"Ljung-Box (lag={lags})",
        "H0":          "No serial autocorrelation",
        "statistic":   round(stat, 4),
        "p_value":     round(p, 4),
        "lags_tested": lags,
        "reject_H0":   reject,
        "conclusion":  "⚠️ Autocorrelation detected" if reject else "✅ No significant autocorrelation",
        "interpretation": (
            f"Significant serial autocorrelation at lag {lags} (p={p:.4f}). "
            "Log-returns have exploitable linear structure — AR/ARIMA, LSTM, GRU models benefit."
            if reject else
            f"No significant autocorrelation (p={p:.4f}). "
            "Returns appear close to white noise; nonlinear models (XGB, RF) or volatility "
            "models (GARCH) may outperform linear AR approaches."
        ),
    }


def jarquebera_test(series, alpha=0.05):
    """H0: Returns are normally distributed (skewness=0, excess kurtosis=0)."""
    clean  = series.dropna()
    stat, p = jarque_bera(clean)
    skew   = float(clean.skew())
    kurt   = float(clean.kurtosis())
    reject = p < alpha
    return {
        "test":             "Jarque-Bera (Normality)",
        "H0":               "Returns are normally distributed",
        "statistic":        round(stat, 4),
        "p_value":          round(p, 4),
        "skewness":         round(skew, 4),
        "excess_kurtosis":  round(kurt, 4),
        "reject_H0":        reject,
        "conclusion":       "⚠️ Non-normal (fat tails / skew)" if reject else "✅ Approximately normal",
        "interpretation": (
            f"Non-normal distribution confirmed (skew={skew:.3f}, excess kurt={kurt:.3f}). "
            "Fat tails mean extreme events are more frequent than Gaussian models assume. "
            "Use Student-t GARCH for risk, CVaR instead of VaR, and robust loss functions in ML."
            if reject else
            f"Returns approximately normal (skew={skew:.3f}, excess kurt={kurt:.3f}). "
            "Gaussian assumptions are reasonable for this series."
        ),
    }


def archlm_test(series, lags=5, alpha=0.05):
    """H0: No ARCH effects (homoscedastic residuals)."""
    clean                       = series.dropna()
    lm_stat, lm_p, f_stat, f_p = het_arch(clean, nlags=lags)
    reject                      = lm_p < alpha
    return {
        "test":         f"ARCH-LM (lag={lags})",
        "H0":           "No ARCH effects (constant variance)",
        "lm_statistic": round(lm_stat, 4),
        "lm_p_value":   round(lm_p, 4),
        "f_statistic":  round(f_stat, 4),
        "f_p_value":    round(f_p, 4),
        "reject_H0":    reject,
        "conclusion":   "⚠️ ARCH effects (volatility clustering)" if reject else "✅ No ARCH effects",
        "interpretation": (
            f"Volatility clustering confirmed (LM p={lm_p:.4f}). "
            "Variance is time-varying — GARCH modelling is strongly recommended. "
            "GARCH-based position sizing (reduce exposure during high-vol regimes) is warranted."
            if reject else
            f"No significant ARCH effects (LM p={lm_p:.4f}). "
            "Constant-variance (homoscedastic) assumption is acceptable for this series."
        ),
    }


def run_all_tests(log_returns, lb_lags=10, arch_lags=5):
    return {
        "ADF":        adf_test(log_returns),
        "KPSS":       kpss_test(log_returns),
        "LjungBox":   ljungbox_test(log_returns, lags=lb_lags),
        "JarqueBera": jarquebera_test(log_returns),
        "ARCHLM":     archlm_test(log_returns, lags=arch_lags),
    }


def stationarity_verdict(adf_res, kpss_res):
    """
    Combined ADF + KPSS verdict using correct null hypotheses.

    ADF:  reject_H0=True  → stationary
    KPSS: reject_H0=False → stationary (H0 is stationarity, so fail-to-reject = stationary)
    """
    adf_says_stationary  = adf_res["reject_H0"]       # True = ADF says stationary
    kpss_says_stationary = not kpss_res["reject_H0"]   # True = KPSS says stationary

    if adf_says_stationary and kpss_says_stationary:
        return {
            "verdict": "✅ Strongly Stationary",
            "detail":  "Both ADF and KPSS agree the series is stationary. "
                       "Log-returns are well-behaved — proceed directly to modelling.",
            "action":  "No transformation needed.",
        }
    elif adf_says_stationary and not kpss_says_stationary:
        return {
            "verdict": "⚠️ Trend-Stationary",
            "detail":  "ADF rejects unit root (stationary) but KPSS rejects stationarity. "
                       "This conflicting result typically means the series is stationary "
                       "around a deterministic trend. The series reverts to a trending mean, "
                       "not a fixed mean.",
            "action":  "Consider detrending (e.g. subtract linear trend) before modelling. "
                       "ARIMA with d=0 on detrended series or use models robust to trend.",
        }
    elif not adf_says_stationary and kpss_says_stationary:
        return {
            "verdict": "⚠️ Difference-Stationary",
            "detail":  "ADF fails to reject unit root (non-stationary) but KPSS confirms stationarity. "
                       "Series may have a stochastic trend (random walk component). "
                       "Differencing should resolve this.",
            "action":  "Apply first differencing. ARIMA with d=1 is appropriate.",
        }
    else:
        return {
            "verdict": "❌ Non-Stationary",
            "detail":  "Both ADF and KPSS indicate non-stationarity. "
                       "The series has neither stable mean nor stable variance.",
            "action":  "Apply differencing and/or variance-stabilising transform. "
                       "Investigate structural breaks.",
        }
