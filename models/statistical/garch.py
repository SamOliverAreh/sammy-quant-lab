"""
garch.py — Flexible GARCH model selection.
Searches over p, q in [1..2] and distributions ['normal','t','skewt']
Selects best specification by BIC.
"""
import numpy as np
import warnings
warnings.filterwarnings("ignore")

try:
    from arch import arch_model
    ARCH_AVAILABLE = True
except ImportError:
    ARCH_AVAILABLE = False


def fit_best_garch(returns, max_p=2, max_q=2, dists=("normal", "t", "skewt")):
    """
    Grid-search best GARCH(p,q) + distribution by BIC.
    Returns fitted model result.
    """
    if not ARCH_AVAILABLE:
        return None, (1, 1), "normal"

    scaled  = returns * 100
    best_bic, best_res, best_order, best_dist = np.inf, None, (1, 1), "normal"

    for p in range(1, max_p + 1):
        for q in range(1, max_q + 1):
            for dist in dists:
                try:
                    res = arch_model(
                        scaled, vol="Garch", p=p, q=q, dist=dist
                    ).fit(disp="off", show_warning=False)
                    if res.bic < best_bic:
                        best_bic, best_res, best_order, best_dist = res.bic, res, (p, q), dist
                except Exception:
                    continue

    return best_res, best_order, best_dist


def garch_volatility(returns, horizon=10):
    """Forecast volatility using best GARCH specification."""
    if not ARCH_AVAILABLE:
        vol = returns.rolling(horizon).std().iloc[-1]
        return np.array([vol] * horizon)
    res, _, _ = fit_best_garch(returns)
    if res is None:
        vol = float(returns.std())
        return np.array([vol] * horizon)
    return np.sqrt(res.forecast(horizon=horizon).variance.values[-1]) / 100


def garch_summary(returns):
    """Return summary stats for best GARCH fit."""
    if not ARCH_AVAILABLE:
        return {"error": "arch package not installed"}
    res, order, dist = fit_best_garch(returns)
    if res is None:
        return {"error": "GARCH fit failed"}
    return {
        "best_order":    order,
        "best_dist":     dist,
        "aic":           res.aic,
        "bic":           res.bic,
        "loglikelihood": res.loglikelihood,
        "params":        res.params.to_dict(),
    }


def garch_insample_vol(returns):
    """Return in-sample conditional volatility series (for plots)."""
    if not ARCH_AVAILABLE:
        return returns.rolling(10).std()
    res, _, _ = fit_best_garch(returns)
    if res is None:
        return returns.rolling(10).std()
    return res.conditional_volatility / 100
