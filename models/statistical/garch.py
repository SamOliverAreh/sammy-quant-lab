import numpy as np
import warnings
warnings.filterwarnings("ignore")

try:
    from arch import arch_model
    ARCH_AVAILABLE = True
except ImportError:
    ARCH_AVAILABLE = False


def garch_volatility(returns, horizon=10):
    if not ARCH_AVAILABLE:
        vol = returns.rolling(horizon).std().iloc[-1]
        return np.array([vol] * horizon)
    scaled = returns * 100
    res = arch_model(scaled, vol="Garch", p=1, q=1, dist="normal").fit(disp="off", show_warning=False)
    return np.sqrt(res.forecast(horizon=horizon).variance.values[-1]) / 100


def garch_summary(returns):
    if not ARCH_AVAILABLE:
        return {"error": "arch package not installed"}
    scaled = returns * 100
    res = arch_model(scaled, vol="Garch", p=1, q=1).fit(disp="off", show_warning=False)
    return {"aic": res.aic, "bic": res.bic, "loglikelihood": res.loglikelihood, "params": res.params.to_dict()}
