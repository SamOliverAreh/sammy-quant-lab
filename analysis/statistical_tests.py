"""
statistical_tests.py — Full battery of statistical tests on log-returns.
  1. ADF   — stationarity (unit root)
  2. KPSS  — stationarity (level)
  3. Ljung-Box — autocorrelation
  4. Jarque-Bera — normality
  5. ARCH-LM — volatility clustering
"""
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from scipy.stats import jarque_bera


def adf_test(series, alpha=0.05):
    res = adfuller(series.dropna(), autolag="AIC")
    stat, p, _, nobs, cv, _ = res
    reject = p < alpha
    return {
        "test": "ADF (Augmented Dickey-Fuller)",
        "statistic": round(stat, 4), "p_value": round(p, 4), "n_obs": nobs,
        "crit_1pct": round(cv["1%"], 4), "crit_5pct": round(cv["5%"], 4),
        "reject_H0": reject,
        "conclusion": "Stationary" if reject else "Non-stationary (unit root)",
        "interpretation": (
            "Log-returns are stationary — mean/variance stable over time. Suitable for modelling."
            if reject else
            "Non-stationary detected. Log-returns are usually stationary — check data quality."
        ),
    }


def kpss_test(series, alpha=0.05):
    stat, p, lags, cv = kpss(series.dropna(), regression="c", nlags="auto")
    reject = p < alpha
    return {
        "test": "KPSS",
        "statistic": round(stat, 4), "p_value": round(p, 4), "lags_used": lags,
        "crit_5pct": round(cv["5%"], 4),
        "reject_H0": reject,
        "conclusion": "Non-stationary" if reject else "Stationary",
        "interpretation": (
            "KPSS rejects stationarity. If ADF also fails, the series needs differencing."
            if reject else
            "KPSS confirms stationarity. Combined with ADF, this is strong evidence."
        ),
    }


def ljungbox_test(series, lags=10, alpha=0.05):
    res  = acorr_ljungbox(series.dropna(), lags=[lags], return_df=True)
    stat = float(res["lb_stat"].iloc[-1])
    p    = float(res["lb_pvalue"].iloc[-1])
    reject = p < alpha
    return {
        "test": f"Ljung-Box (lag={lags})",
        "statistic": round(stat, 4), "p_value": round(p, 4), "lags_tested": lags,
        "reject_H0": reject,
        "conclusion": "Autocorrelation detected" if reject else "No significant autocorrelation",
        "interpretation": (
            "Significant serial autocorrelation. Log-returns have predictable linear structure — AR/MA or LSTM models can exploit this."
            if reject else
            "No significant autocorrelation. Returns appear close to white noise — nonlinear (ML) or volatility models may perform better."
        ),
    }


def jarquebera_test(series, alpha=0.05):
    clean = series.dropna()
    stat, p = jarque_bera(clean)
    skew = float(clean.skew())
    kurt = float(clean.kurtosis())
    reject = p < alpha
    return {
        "test": "Jarque-Bera (Normality)",
        "statistic": round(stat, 4), "p_value": round(p, 4),
        "skewness": round(skew, 4), "excess_kurtosis": round(kurt, 4),
        "reject_H0": reject,
        "conclusion": "Non-normal (fat tails/skew)" if reject else "Approximately normal",
        "interpretation": (
            f"Fat tails present (excess kurtosis={kurt:.2f}, skew={skew:.2f}). "
            "Standard VaR underestimates tail risk. Use Student-t GARCH or CVaR."
            if reject else
            f"Returns approximately normal (excess kurtosis={kurt:.2f}). "
            "Gaussian assumptions are reasonable."
        ),
    }


def archlm_test(series, lags=5, alpha=0.05):
    clean = series.dropna()
    lm_stat, lm_p, f_stat, f_p = het_arch(clean, nlags=lags)
    reject = lm_p < alpha
    return {
        "test": f"ARCH-LM (lag={lags})",
        "lm_statistic": round(lm_stat, 4), "lm_p_value": round(lm_p, 4),
        "f_statistic": round(f_stat, 4), "f_p_value": round(f_p, 4),
        "reject_H0": reject,
        "conclusion": "ARCH effects present (volatility clustering)" if reject else "No ARCH effects",
        "interpretation": (
            "Volatility clustering confirmed. GARCH modelling is warranted. "
            "Large moves tend to follow large moves — position sizing should adapt."
            if reject else
            "No volatility clustering. Constant-variance assumptions may suffice."
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
    a = adf_res["reject_H0"]
    k = not kpss_res["reject_H0"]
    if a and k:   return "✅ Strongly Stationary (ADF & KPSS agree)"
    if a and not k: return "⚠️ Trend-Stationary — consider detrending"
    if not a and k: return "⚠️ Difference-Stationary — differencing may help"
    return "❌ Non-Stationary — series needs transformation"
