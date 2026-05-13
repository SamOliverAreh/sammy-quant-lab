"""
statistical_tests.py v5.1
=========================
All tests run on log-returns (already stationary by construction).

IMPORTANT NOTES ON P-VALUES:
─────────────────────────────
ADF p-value:
  • Computed via MacKinnon (1994) response surface regression.
  • For very stationary series (log-returns), the ADF statistic is
    highly negative (e.g. -20 to -40). The MacKinnon surface maps
    this to p ≈ 0.0000 — this is CORRECT and expected. It does NOT
    mean something went wrong. It simply means overwhelming evidence
    of stationarity. The response surface is well-behaved.

KPSS p-value:
  • Statsmodels interpolates from only 4 critical value points
    (Kwiatkowski et al. 1992 Table 1): 10%, 5%, 2.5%, 1%.
  • p-value is BOUNDED to [0.01, 0.10] — a known statsmodels limitation.
  • For log-returns, KPSS statistic is typically 0.05–0.15, which is
    BELOW the 10% critical value (0.347). Statsmodels reports p = 0.10
    but the TRUE p-value is >> 0.10 (could be 0.5–0.9).
  • p = 0.10 means "fail to reject H0 (stationarity)" — the series IS
    stationary. The bound just means we cannot be more precise.
  • We display this correctly with a note and compare stat vs crits directly.

EXPECTED RESULT FOR LOG-RETURNS:
  ADF:  stat very negative, p ≈ 0.000 → reject H0 → STATIONARY ✅
  KPSS: stat very small, p = 0.10 (bounded) → fail to reject H0 → STATIONARY ✅
  Both agree → "Strongly Stationary" verdict.
  This is exactly what you would expect from a log(Pₜ/Pₜ₋₁) series.

NULL HYPOTHESES (opposite direction):
  ADF:  H0 = series HAS a unit root (non-stationary). Reject H0 → stationary.
  KPSS: H0 = series IS stationary.                    Reject H0 → non-stationary.
"""
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from scipy.stats import jarque_bera, chi2, norm


# ── KPSS critical value table (Kwiatkowski et al. 1992, Table 1, level/c) ────
_KPSS_CRITS = {
    "10%": 0.347, "5%": 0.463, "2.5%": 0.574, "1%": 0.739
}


def _kpss_p_note(stat, cv):
    """
    Return a human-readable note about the KPSS p-value bounding.
    Since statsmodels bounds p to [0.01, 0.10], we explain what it means
    by directly comparing the statistic to critical values.
    """
    if stat < cv.get("10%", 0.347):
        return (f"stat={stat:.4f} < crit(10%)={cv.get('10%',0.347):.4f}. "
                f"True p >> 0.10 (far from rejection). Strongly fail to reject H₀.")
    elif stat < cv.get("5%", 0.463):
        return (f"stat={stat:.4f} between crit(10%) and crit(5%). p ∈ (0.05, 0.10).")
    elif stat < cv.get("2.5%", 0.574):
        return (f"stat={stat:.4f} between crit(5%) and crit(2.5%). p ∈ (0.025, 0.05). "
                f"Marginal — borderline rejection at 5%.")
    elif stat < cv.get("1%", 0.739):
        return (f"stat={stat:.4f} between crit(2.5%) and crit(1%). p ∈ (0.01, 0.025). "
                f"Reject H₀ at 5% level.")
    else:
        return (f"stat={stat:.4f} > crit(1%)={cv.get('1%',0.739):.4f}. "
                f"True p << 0.01. Strong rejection of H₀.")


def _adf_p_note(stat, p, cv):
    """
    For ADF: stat is very negative for stationary series.
    p ≈ 0 is correct and expected for log-returns.
    """
    if p < 0.001:
        return (f"stat={stat:.4f} (very negative) → overwhelming evidence of stationarity. "
                f"p≈0 is correct for log-returns and not a numerical error.")
    elif p < 0.05:
        return f"Reject H₀ at {int(round(p,2)*100)}% level. Series is stationary."
    else:
        return f"Fail to reject H₀ (p={p:.4f} ≥ 0.05). Evidence of non-stationarity."


# ─────────────────────────────────────────────────────────────────────────────

def adf_test(series, alpha=0.05):
    """
    Augmented Dickey-Fuller test.
    H₀: series has a unit root (NON-stationary).
    Reject H₀ (p < α) → series IS stationary.

    p-value: MacKinnon (1994) response surface — well-defined even at p≈0.
    For log-returns, expect stat ≈ -10 to -40, p ≈ 0.0000 → strongly stationary.
    """
    clean = series.dropna()
    res   = adfuller(clean, autolag="AIC")
    stat, p, lags_used, nobs, cv, _ = res

    # Clamp p to [0, 1] — MacKinnon surface can theoretically exceed bounds
    p = float(np.clip(p, 0.0, 1.0))

    reject = p < alpha
    note   = _adf_p_note(stat, p, cv)

    # p display: show actual value but flag if ≈ 0
    p_display = f"< 0.0001" if p < 0.0001 else f"{p:.4f}"

    return {
        "test":        "ADF (Augmented Dickey-Fuller)",
        "H0":          "Series has a unit root (Non-stationary)",
        "statistic":   round(stat, 4),
        "p_value":     p,
        "p_display":   p_display,
        "lags_used":   lags_used,
        "n_obs":       nobs,
        "crit_1pct":   round(cv["1%"],  4),
        "crit_5pct":   round(cv["5%"],  4),
        "crit_10pct":  round(cv["10%"], 4),
        "reject_H0":   reject,
        "conclusion":  "✅ Stationary (reject H₀)" if reject else "⚠️ Non-stationary (fail to reject H₀)",
        "p_note":      note,
        "interpretation": (
            f"ADF strongly rejects the unit root — series is stationary. "
            f"For log-returns this is the expected result: log(Pₜ/Pₜ₋₁) is "
            f"a first-differenced log-price series, which eliminates the unit root "
            f"present in raw prices. {note}"
            if reject else
            f"ADF fails to reject the unit root (p={p_display}). "
            f"This is unusual for log-returns. Inspect for structural breaks, "
            f"outliers, or very persistent autocorrelation. {note}"
        ),
    }


def kpss_test(series, alpha=0.05):
    """
    KPSS test.
    H₀: series IS stationary.
    Reject H₀ (p < α) → series is NON-stationary.

    CRITICAL NOTE ON P-VALUES:
    statsmodels bounds KPSS p-values to [0.01, 0.10] because the original
    Kwiatkowski et al. (1992) table only covers 4 quantiles (10%, 5%, 2.5%, 1%).
    For log-returns with very small KPSS statistics, p will be reported as 0.10
    (the floor), but the TRUE p-value is >> 0.10.
    We therefore compare stat vs critical values directly for interpretation.
    """
    clean = series.dropna()
    stat, p, lags_used, cv = kpss(clean, regression="c", nlags="auto")

    # p is bounded [0.01, 0.10] by statsmodels — do NOT compare naively to 0.05
    # Instead use stat vs critical values for the actual decision
    reject_by_stat = stat > cv.get("5%", 0.463)   # true 5% decision
    reject_by_p    = p < alpha                      # bounded p — less reliable

    # Use stat-based decision as primary
    reject = reject_by_stat
    note   = _kpss_p_note(stat, cv)

    # Explain the p-value bounding to the user
    if p <= 0.01:
        p_display = "≤ 0.01 (bounded ceiling — true p << 0.01)"
    elif p >= 0.10:
        p_display = "≥ 0.10 (bounded floor — true p >> 0.10)"
    else:
        p_display = f"{p:.4f} (interpolated)"

    return {
        "test":         "KPSS",
        "H0":           "Series is stationary",
        "statistic":    round(stat, 4),
        "p_value":      p,
        "p_display":    p_display,
        "lags_used":    lags_used,
        "crit_10pct":   round(cv.get("10%", 0.347), 4),
        "crit_5pct":    round(cv.get("5%",  0.463), 4),
        "crit_2_5pct":  round(cv.get("2.5%",0.574), 4),
        "crit_1pct":    round(cv.get("1%",  0.739), 4),
        "reject_H0":    reject,
        "reject_by_stat": reject_by_stat,
        "conclusion":   ("⚠️ Non-stationary (reject H₀ — stat > crit 5%)"
                         if reject else
                         "✅ Stationary (fail to reject H₀ — stat < crit 5%)"),
        "p_note":       note,
        "p_bound_warning": (
            "⚠️ KPSS p-values are bounded to [0.01, 0.10] by statsmodels due to "
            "limited critical value tables. Compare statistic directly to critical "
            "values above for the correct decision."
        ),
        "interpretation": (
            f"KPSS statistic ({stat:.4f}) is {'above' if reject else 'below'} the 5% "
            f"critical value ({cv.get('5%', 0.463):.4f}) → "
            f"{'reject' if reject else 'fail to reject'} H₀ (stationarity). "
            f"{note} "
            f"For log-returns, expect stat << 0.347 (10% crit), confirming stationarity. "
            f"The reported p={p_display} reflects statsmodels' table-interpolation bounds, "
            f"not a numerical error."
        ),
    }


def ljungbox_test(series, lags=10, alpha=0.05):
    """
    Ljung-Box test for serial autocorrelation.
    H₀: No autocorrelation up to lag k.
    Test statistic: Q = n(n+2) Σ(ρ̂²ₖ/(n-k)) ~ χ²(lags).
    p-value is exact chi-squared probability — no bounding issues.
    """
    clean  = series.dropna()
    res    = acorr_ljungbox(clean, lags=[lags], return_df=True)
    stat   = float(res["lb_stat"].iloc[-1])
    p      = float(np.clip(res["lb_pvalue"].iloc[-1], 0.0, 1.0))
    reject = p < alpha

    return {
        "test":        f"Ljung-Box (lag={lags})",
        "H0":          "No serial autocorrelation up to lag k",
        "statistic":   round(stat, 4),
        "p_value":     p,
        "p_display":   f"< 0.0001" if p < 0.0001 else f"{p:.4f}",
        "lags_tested": lags,
        "df":          lags,
        "reject_H0":   reject,
        "conclusion":  "⚠️ Autocorrelation detected" if reject else "✅ No significant autocorrelation",
        "p_note":      (f"Q({lags})={stat:.4f} ~ χ²({lags}). "
                        f"Exact chi-squared p-value — no bounding issues."),
        "interpretation": (
            f"Significant autocorrelation in log-returns (Q={stat:.2f}, p={p:.4f}). "
            "There is exploitable linear structure — ARIMA, LSTM, GRU models can leverage this. "
            "This is common for some asset classes (equity index momentum, FX carry)."
            if reject else
            f"No significant autocorrelation (Q={stat:.2f}, p={p:.4f}). "
            "Log-returns are close to white noise in the linear sense. "
            "Nonlinear models (XGBoost, RF, Transformer) or volatility models (GARCH) "
            "may capture higher-order structure better than linear AR models."
        ),
    }


def jarquebera_test(series, alpha=0.05):
    """
    Jarque-Bera normality test.
    H₀: Returns are normally distributed (skewness=0, excess kurtosis=0).
    Test statistic: JB = n/6 * (S² + K²/4) ~ χ²(2).
    p-value is exact chi-squared — no bounding issues.
    For financial returns, rejection is almost always expected (fat tails).
    """
    clean    = series.dropna()
    stat, p  = jarque_bera(clean)
    p        = float(np.clip(p, 0.0, 1.0))
    skew     = float(clean.skew())
    kurt     = float(clean.kurtosis())   # excess kurtosis
    reject   = p < alpha

    return {
        "test":            "Jarque-Bera (Normality)",
        "H0":              "Returns are normally distributed",
        "statistic":       round(stat, 4),
        "p_value":         p,
        "p_display":       f"< 0.0001" if p < 0.0001 else f"{p:.4f}",
        "skewness":        round(skew, 4),
        "excess_kurtosis": round(kurt, 4),
        "reject_H0":       reject,
        "conclusion":      "⚠️ Non-normal (fat tails / skew)" if reject else "✅ Approximately normal",
        "p_note":          f"JB={stat:.4f} ~ χ²(2). Exact chi-squared p-value.",
        "interpretation": (
            f"Non-normality confirmed (skew={skew:.3f}, excess kurt={kurt:.3f}). "
            f"{'Positive skew → more frequent small losses, occasional large gains. ' if skew > 0 else 'Negative skew → left-tail risk, occasional large losses. '}"
            f"Excess kurtosis={kurt:.2f} {'> 0 → fatter tails than Gaussian (leptokurtic).' if kurt > 0 else '< 0 → thinner tails than Gaussian (platykurtic).'} "
            "This is the norm for financial returns — use Student-t GARCH, CVaR, and "
            "robust ML loss functions accordingly."
            if reject else
            f"Log-returns appear approximately normal (skew={skew:.3f}, excess kurt={kurt:.3f}). "
            "Gaussian assumptions are reasonable here — unusual for financial data."
        ),
    }


def archlm_test(series, lags=5, alpha=0.05):
    """
    ARCH-LM (Engle 1982) test for conditional heteroscedasticity.
    H₀: No ARCH effects (residuals are homoscedastic).
    LM statistic: n·R² from regression of squared residuals on lags ~ χ²(lags).
    p-value is exact chi-squared — no bounding issues.
    """
    clean                       = series.dropna()
    lm_stat, lm_p, f_stat, f_p = het_arch(clean, nlags=lags)
    lm_p = float(np.clip(lm_p, 0.0, 1.0))
    f_p  = float(np.clip(f_p,  0.0, 1.0))
    reject = lm_p < alpha

    return {
        "test":         f"ARCH-LM (Engle 1982, lag={lags})",
        "H0":           "No ARCH effects (homoscedastic variance)",
        "lm_statistic": round(lm_stat, 4),
        "lm_p_value":   lm_p,
        "lm_p_display": f"< 0.0001" if lm_p < 0.0001 else f"{lm_p:.4f}",
        "f_statistic":  round(f_stat, 4),
        "f_p_value":    f_p,
        "reject_H0":    reject,
        "conclusion":   "⚠️ ARCH effects (volatility clustering)" if reject else "✅ No ARCH effects",
        "p_note":       f"LM={lm_stat:.4f} ~ χ²({lags}). Exact chi-squared p-value.",
        "interpretation": (
            f"Volatility clustering confirmed (LM={lm_stat:.2f}, p={lm_p:.4f}). "
            "Squared returns are autocorrelated — large moves cluster together. "
            "GARCH modelling is strongly warranted. "
            "GARCH-based position sizing (reduce exposure during high-vol regimes) is recommended."
            if reject else
            f"No significant ARCH effects (LM={lm_stat:.2f}, p={lm_p:.4f}). "
            "Variance appears approximately constant. "
            "Simple constant-volatility models may suffice for this series."
        ),
    }


# ── Combined runner ───────────────────────────────────────────────────────────

def run_all_tests(log_returns, lb_lags=10, arch_lags=5):
    """Run the full pre-modelling test battery on log-returns."""
    return {
        "ADF":        adf_test(log_returns),
        "KPSS":       kpss_test(log_returns),
        "LjungBox":   ljungbox_test(log_returns, lags=lb_lags),
        "JarqueBera": jarquebera_test(log_returns),
        "ARCHLM":     archlm_test(log_returns, lags=arch_lags),
    }


# ── Stationarity verdict ──────────────────────────────────────────────────────

def stationarity_verdict(adf_res, kpss_res):
    """
    Combined ADF + KPSS verdict with correct null hypotheses.

    Decision rule:
      ADF:  reject_H0 = True  → ADF says STATIONARY
      KPSS: reject_H0 = False → KPSS says STATIONARY
            (KPSS uses stat vs critical value, not bounded p-value)

    For log-returns, the expected and typical outcome is:
      ADF reject + KPSS fail-to-reject → Strongly Stationary.
    This is because log(Pₜ/Pₜ₋₁) is a first-differenced log-price series
    which eliminates the I(1) unit root present in raw prices.
    """
    adf_stationary  = adf_res["reject_H0"]          # True = ADF → stationary
    kpss_stationary = not kpss_res["reject_H0"]      # True = KPSS → stationary

    if adf_stationary and kpss_stationary:
        return {
            "verdict":  "✅ Strongly Stationary",
            "css_class": "verdict-strong",
            "detail":   (
                "Both ADF and KPSS confirm stationarity — the expected result for log-returns. "
                "log(Pₜ/Pₜ₋₁) eliminates the I(1) unit root in prices, yielding a stationary, "
                "mean-reverting series. Proceed directly to modelling without transformation."
            ),
            "action":   "No further transformation needed. Series is model-ready.",
        }

    elif adf_stationary and not kpss_stationary:
        return {
            "verdict":  "⚠️ Trend-Stationary",
            "css_class": "verdict-warn",
            "detail":   (
                "ADF rejects the unit root (stationary) but KPSS rejects stationarity. "
                "Conflicting results typically indicate the series is stationary around a "
                "deterministic trend (trend-stationary), not a fixed mean. "
                "For log-returns this is uncommon and may signal a structural trend "
                "(e.g. persistent drift in FX pairs during a prolonged depreciation episode)."
            ),
            "action":   (
                "Consider detrending: fit a linear trend and model residuals, or use "
                "ARIMA with d=0 on detrended series. Models robust to drift (LSTM, GRU) "
                "may handle this without explicit detrending."
            ),
        }

    elif not adf_stationary and kpss_stationary:
        return {
            "verdict":  "⚠️ Difference-Stationary",
            "css_class": "verdict-diff",
            "detail":   (
                "ADF fails to reject the unit root but KPSS confirms stationarity. "
                "This unusual combination for log-returns may indicate a near-unit-root "
                "process or very high persistence. ADF has low power in short samples "
                "or with structural breaks."
            ),
            "action":   (
                "Check sample size (ADF needs ≥ 100 obs). Inspect ACF for slow decay. "
                "Consider fractional integration (ARFIMA). KPSS confirmation of stationarity "
                "gives reasonable confidence to proceed with modelling."
            ),
        }

    else:
        return {
            "verdict":  "❌ Non-Stationary",
            "css_class": "verdict-bad",
            "detail":   (
                "Both ADF and KPSS indicate non-stationarity in the log-returns. "
                "This is very unusual for a log-return series and warrants investigation. "
                "Possible causes: structural breaks, outliers, very small sample, or "
                "the series contains level shifts (e.g. market suspension periods)."
            ),
            "action":   (
                "Inspect time series plot for breaks. Remove outliers or add dummy variables. "
                "Consider second differencing or regime-switching models. "
                "Verify data quality."
            ),
        }
