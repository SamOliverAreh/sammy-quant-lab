"""
evaluation/metrics.py — Extended evaluation suite.

Standard:  RMSE, MAE, MAPE, Direction Accuracy
New:       Diebold-Mariano test, Hit Ratio (on returns sign),
           Sortino Ratio, Information Ratio,
           Sharpe Ratio, Max Drawdown
Benchmark: Random Walk (y_hat_t = y_{t-1})
"""
import numpy as np
import pandas as pd
from scipy import stats
import warnings
warnings.filterwarnings("ignore")


# ── Basic metrics ─────────────────────────────────────────────────────────────

def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.array(y_true) - np.array(y_pred)) ** 2)))

def mae(y_true, y_pred):
    return float(np.mean(np.abs(np.array(y_true) - np.array(y_pred))))

def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)

def direction_accuracy(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    if len(y_true) < 2:
        return 50.0
    return float(np.mean(np.sign(np.diff(y_true)) == np.sign(np.diff(y_pred))) * 100)


# ── Hit ratio (sign accuracy on returns directly) ─────────────────────────────

def hit_ratio(y_true_returns, y_pred_returns):
    """
    Fraction of periods where the predicted return sign matches actual sign.
    Works directly on log-returns (no diff needed — sign IS the direction).
    Target: >50% to beat random, >55% for practical edge.
    """
    y_true = np.array(y_true_returns)
    y_pred = np.array(y_pred_returns)
    return float(np.mean(np.sign(y_true) == np.sign(y_pred)) * 100)


# ── Diebold-Mariano test ──────────────────────────────────────────────────────

def diebold_mariano(y_true, pred_a, pred_b, h=1, criterion="mse"):
    """
    Diebold-Mariano test: is model A significantly better than model B?
    H0: equal predictive accuracy.
    Returns: dm_stat, p_value, conclusion.

    criterion: 'mse' or 'mae'
    h: forecast horizon (1 for one-step-ahead)
    """
    y_true = np.array(y_true)
    pred_a = np.array(pred_a)
    pred_b = np.array(pred_b)
    n      = min(len(y_true), len(pred_a), len(pred_b))
    y_true, pred_a, pred_b = y_true[:n], pred_a[:n], pred_b[:n]

    if criterion == "mse":
        e_a = (y_true - pred_a) ** 2
        e_b = (y_true - pred_b) ** 2
    else:
        e_a = np.abs(y_true - pred_a)
        e_b = np.abs(y_true - pred_b)

    d   = e_a - e_b
    d_bar = d.mean()
    # HAC variance with Newey-West (h-1 lags)
    n_lags = h - 1
    gamma0 = np.var(d, ddof=1)
    gamma  = gamma0
    for k in range(1, n_lags + 1):
        cov_k  = np.cov(d[k:], d[:-k])[0, 1]
        gamma += 2 * (1 - k / (n_lags + 1)) * cov_k
    var_d  = gamma / n
    if var_d <= 0:
        var_d = 1e-12
    dm_stat = d_bar / np.sqrt(var_d)
    p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))

    if p_value < 0.05:
        winner = "Model A better" if dm_stat < 0 else "Model B better"
        conclusion = f"✅ Significant difference (p={p_value:.4f}) — {winner}"
    else:
        conclusion = f"❌ No significant difference (p={p_value:.4f})"

    return {
        "dm_statistic": round(dm_stat, 4),
        "p_value":      round(p_value, 4),
        "conclusion":   conclusion,
    }


# ── Risk-adjusted metrics ─────────────────────────────────────────────────────

def sharpe_ratio(returns, risk_free=0.0, periods=252):
    r = np.array(returns)
    excess = r - risk_free / periods
    std = excess.std()
    return float((excess.mean() / std) * np.sqrt(periods)) if std > 0 else 0.0


def sortino_ratio(returns, risk_free=0.0, periods=252, target=0.0):
    """
    Sortino ratio — only penalises downside deviation below `target`.
    Better than Sharpe for asymmetric return distributions (fat tails).
    """
    r = np.array(returns)
    excess  = r - risk_free / periods
    downside = np.minimum(r - target, 0)
    downside_std = np.sqrt(np.mean(downside ** 2))
    if downside_std <= 0:
        return 0.0
    return float((excess.mean() / downside_std) * np.sqrt(periods))


def information_ratio(returns, benchmark_returns, periods=252):
    """
    Information Ratio = (strategy excess over benchmark) / tracking error.
    Benchmark = buy-and-hold (passive) return series.
    """
    r = np.array(returns)
    b = np.array(benchmark_returns)
    n = min(len(r), len(b))
    active   = r[:n] - b[:n]
    te       = active.std()
    if te <= 0:
        return 0.0
    return float((active.mean() / te) * np.sqrt(periods))


def max_drawdown(prices):
    prices = np.array(prices)
    peak   = np.maximum.accumulate(prices)
    return float(((prices - peak) / (peak + 1e-12)).min())


# ── Transaction-cost equity curve ─────────────────────────────────────────────

def equity_curve_with_tc(
    log_ret_true: np.ndarray,
    log_ret_pred: np.ndarray,
    initial_capital: float = 10_000.0,
    tc_rate: float = 0.001,       # 0.1% per trade
    garch_vol: np.ndarray = None, # optional GARCH vol for position sizing
    vol_target: float = 0.02,     # target daily vol for GARCH sizing
) -> dict:
    """
    Simulate a long/short equity curve based on predicted return signs.

    Position sizing:
      - Base: +1 (long) if pred > 0, -1 (short) if pred < 0
      - GARCH sizing: scale position by vol_target / garch_vol_t
        (reduce size when volatility is high)

    Transaction costs: deducted each time position CHANGES sign.

    Returns dict with:
        equity          : np.ndarray (portfolio value over time)
        benchmark       : np.ndarray (buy-and-hold)
        positions       : np.ndarray (+1 / -1)
        turnover        : float (fraction of periods with position change)
        total_tc_cost   : float (total transaction cost paid)
    """
    n       = min(len(log_ret_true), len(log_ret_pred))
    y_true  = np.array(log_ret_true[:n])
    y_pred  = np.array(log_ret_pred[:n])

    # Raw positions: +1 long, -1 short
    positions = np.sign(y_pred)
    positions[positions == 0] = 1  # default to long on zero

    # GARCH-based position sizing
    if garch_vol is not None and len(garch_vol) >= n:
        gv = np.array(garch_vol[:n])
        gv = np.where(gv <= 0, 1e-6, gv)
        scale = np.clip(vol_target / gv, 0.1, 3.0)  # cap at 3x, floor at 0.1x
        positions = positions * scale

    # Transaction costs: pay tc_rate whenever position SIGN changes
    pos_sign = np.sign(positions)
    changes  = np.diff(pos_sign, prepend=pos_sign[0])
    tc_mask  = changes != 0
    tc_costs = tc_rate * np.abs(positions) * tc_mask

    # Strategy log-returns (after TC)
    strat_lr = positions * y_true - tc_costs
    bench_lr = y_true  # buy-and-hold

    equity    = initial_capital * np.exp(np.cumsum(strat_lr))
    benchmark = initial_capital * np.exp(np.cumsum(bench_lr))

    turnover      = tc_mask.sum() / n
    total_tc_cost = (tc_costs * initial_capital).sum()

    return {
        "equity":        equity,
        "benchmark":     benchmark,
        "positions":     positions,
        "strat_returns": strat_lr,
        "bench_returns": bench_lr,
        "turnover":      round(turnover, 4),
        "total_tc_cost": round(total_tc_cost, 2),
    }


# ── Random Walk benchmark ─────────────────────────────────────────────────────

def random_walk_forecast(series: pd.Series, steps: int) -> np.ndarray:
    """
    Random Walk: y_hat_t = y_{t-1}   (last observed value repeated).
    On log-returns this means predicting 0 change every period.
    """
    last_val = float(series.iloc[-1])
    return np.full(steps, last_val)


def random_walk_insample(series: pd.Series) -> np.ndarray:
    """In-sample random walk: predicted[t] = actual[t-1]."""
    return np.array(series.values[:-1])


# ── evaluate_all (updated) ────────────────────────────────────────────────────

def evaluate_all(y_true, y_pred, is_returns=True):
    """
    Full evaluation dictionary.
    is_returns=True: y_true/y_pred are log-returns → use hit_ratio directly.
    is_returns=False: use direction_accuracy on levels.
    """
    y_t, y_p = np.array(y_true), np.array(y_pred)
    out = {
        "RMSE": rmse(y_t, y_p),
        "MAE":  mae(y_t, y_p),
        "MAPE": mape(y_t, y_p),
    }
    if is_returns:
        out["Hit Ratio (%)"]        = hit_ratio(y_t, y_p)
        out["Direction Acc. (%)"]   = hit_ratio(y_t, y_p)   # same on returns
        out["Sortino Ratio"]        = sortino_ratio(y_t)
        out["Sharpe Ratio"]         = sharpe_ratio(y_t)
    else:
        out["Direction Acc. (%)"]   = direction_accuracy(y_t, y_p)
    return out


# ── Full model comparison table ───────────────────────────────────────────────

def build_comparison_table(
    y_true: np.ndarray,
    model_preds: dict,         # {model_name: np.ndarray}
    benchmark_returns: np.ndarray = None,
    is_returns: bool = True,
) -> pd.DataFrame:
    """
    Build a full comparison table for all models + random walk.
    Adds DM test vs Random Walk for each model.
    Marks best model per metric with ★.
    """
    rw_pred = np.zeros(len(y_true)) if is_returns else np.full(len(y_true), y_true[0])

    rows = {}
    all_preds = {"Random Walk": rw_pred, **model_preds}

    for name, pred in all_preds.items():
        n   = min(len(y_true), len(pred))
        row = evaluate_all(y_true[:n], pred[:n], is_returns=is_returns)

        if benchmark_returns is not None:
            row["Info. Ratio"] = information_ratio(
                y_true[:n], benchmark_returns[:n]
            )

        # DM test vs Random Walk (skip for RW itself)
        if name != "Random Walk":
            n2 = min(len(y_true), len(rw_pred), len(pred))
            dm = diebold_mariano(y_true[:n2], pred[:n2], rw_pred[:n2])
            row["DM vs RW (p)"] = dm["p_value"]
            row["DM conclusion"] = "✅ Sig." if dm["p_value"] < 0.05 else "—"
        else:
            row["DM vs RW (p)"]  = "—"
            row["DM conclusion"] = "baseline"

        rows[name] = row

    df = pd.DataFrame(rows).T.reset_index().rename(columns={"index": "Model"})

    # Mark best per numeric column
    numeric_cols = [c for c in df.columns if c not in ("Model", "DM conclusion")]
    for col in numeric_cols:
        try:
            vals = pd.to_numeric(df[col], errors="coerce")
            if col in ("RMSE", "MAE", "MAPE", "DM vs RW (p)"):
                best_idx = vals.idxmin()
            else:
                best_idx = vals.idxmax()
            df.loc[best_idx, col] = str(df.loc[best_idx, col]) + " ★"
        except Exception:
            pass

    return df