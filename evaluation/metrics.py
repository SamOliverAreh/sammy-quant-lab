"""
evaluation/metrics.py v5.2
===========================
Fixed:
- random_walk_forecast: correct lag-1 implementation (y_hat_t = y_{t-1})
- random_walk_backtest: produces a true shifted series, not a flat line
- build_comparison_table: RW uses proper shifted predictions
- All other metrics unchanged
"""
import numpy as np
import pandas as pd
from scipy import stats
import warnings
warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════════════════════
#  BASIC FORECAST METRICS
# ══════════════════════════════════════════════════════════════════════════════

def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.array(y_true) - np.array(y_pred)) ** 2)))

def mae(y_true, y_pred):
    return float(np.mean(np.abs(np.array(y_true) - np.array(y_pred))))

def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)

def direction_accuracy(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    if len(y_true) < 2:
        return 50.0
    return float(np.mean(np.sign(np.diff(y_true)) == np.sign(np.diff(y_pred))) * 100)

def hit_ratio(y_true_returns, y_pred_returns):
    """Sign accuracy on log-returns. Target >55% for practical edge."""
    y_true = np.array(y_true_returns)
    y_pred = np.array(y_pred_returns)
    return float(np.mean(np.sign(y_true) == np.sign(y_pred)) * 100)


# ══════════════════════════════════════════════════════════════════════════════
#  RANDOM WALK — CORRECT IMPLEMENTATION
# ══════════════════════════════════════════════════════════════════════════════

def random_walk_forecast(series: pd.Series, steps: int) -> np.ndarray:
    """
    TRUE Random Walk forecast: y_hat_{t+k} = y_t for all k (multi-step ahead).
    The forecast for ALL future steps is simply the last observed value.
    On log-returns this means: predicting the last observed log-return
    for every future period (NOT zero — that is a different model).

    For a proper walk-forward backtest comparison use random_walk_insample()
    which gives y_hat_t = y_{t-1} for each t in the test window.
    """
    last_val = float(series.iloc[-1])
    return np.full(steps, last_val)


def random_walk_insample(true_series: pd.Series) -> np.ndarray:
    """
    In-sample Random Walk predictions: y_hat_t = y_{t-1}.
    For a test window of length N, returns N predictions where each
    prediction is the previous actual value.

    This is the CORRECT way to evaluate RW against other models
    in a backtest: for every period t in [1..N],
      predict = actual[t-1]

    Args:
        true_series: the actual log-return test window (pd.Series)
    Returns:
        np.ndarray of length len(true_series),
        where pred[0] = last training value (not available here so use series.iloc[0]),
        pred[t] = true_series.iloc[t-1] for t >= 1
    """
    vals = np.array(true_series)
    # pred[0] = first value (best we can do without the training tail)
    # pred[t] = actual[t-1]  for t in 1..N-1
    preds = np.empty_like(vals)
    preds[0] = vals[0]       # can't lag further without training tail
    preds[1:] = vals[:-1]    # y_hat_t = y_{t-1}
    return preds


def random_walk_insample_with_tail(
    train_tail: float, true_series: np.ndarray
) -> np.ndarray:
    """
    Random Walk with the last training observation available.

    pred[0] = train_tail     (last value from training set)
    pred[t] = actual[t-1]   for t >= 1

    This gives the proper lag-1 series aligned with the test window.

    Example:
      train: [..., 0.5]   test: [0.76, 0.4, 0.86]
      RW:   [0.5, 0.76, 0.4]
              ↑ from train  ↑lag  ↑lag
    """
    preds    = np.empty(len(true_series))
    preds[0] = train_tail
    if len(true_series) > 1:
        preds[1:] = true_series[:-1]
    return preds


# ══════════════════════════════════════════════════════════════════════════════
#  DRAWDOWN METRICS
# ══════════════════════════════════════════════════════════════════════════════

def max_drawdown(prices):
    prices = np.array(prices, dtype=float)
    peak   = np.maximum.accumulate(prices)
    dd     = (prices - peak) / (peak + 1e-12)
    return float(dd.min())

def drawdown_series(prices):
    prices = np.array(prices, dtype=float)
    peak   = np.maximum.accumulate(prices)
    return (prices - peak) / (peak + 1e-12)

def max_drawdown_duration(prices):
    prices     = np.array(prices, dtype=float)
    peak       = np.maximum.accumulate(prices)
    underwater = prices < peak
    max_dur, cur_dur = 0, 0
    for uw in underwater:
        if uw:
            cur_dur += 1
            max_dur  = max(max_dur, cur_dur)
        else:
            cur_dur  = 0
    return int(max_dur)

def mean_daily_drawdown(prices):
    return float(np.mean(drawdown_series(prices)))

def median_daily_drawdown(prices):
    return float(np.median(drawdown_series(prices)))

def ulcer_index(prices):
    dd = drawdown_series(prices) * 100
    return float(np.sqrt(np.mean(dd ** 2)))


# ══════════════════════════════════════════════════════════════════════════════
#  RISK-ADJUSTED METRICS
# ══════════════════════════════════════════════════════════════════════════════

def sharpe_ratio(returns, risk_free=0.0, periods=252):
    r      = np.array(returns, dtype=float)
    excess = r - risk_free / periods
    std    = excess.std()
    return float((excess.mean() / std) * np.sqrt(periods)) if std > 0 else 0.0

def sortino_ratio(returns, risk_free=0.0, periods=252, target=0.0):
    r        = np.array(returns, dtype=float)
    excess   = r - risk_free / periods
    downside = np.minimum(r - target, 0.0)
    ds_std   = np.sqrt(np.mean(downside ** 2))
    return float((excess.mean() / ds_std) * np.sqrt(periods)) if ds_std > 0 else 0.0

def calmar_ratio(returns, periods=252):
    r      = np.array(returns, dtype=float)
    equity = np.exp(np.cumsum(r))
    ann_r  = r.mean() * periods
    mdd    = abs(max_drawdown(equity))
    return float(ann_r / mdd) if mdd > 1e-8 else 0.0

def sterling_ratio(returns, periods=252, top_n_dd=3):
    r        = np.array(returns, dtype=float)
    equity   = np.exp(np.cumsum(r))
    ann_r    = r.mean() * periods
    dd_arr   = drawdown_series(equity)
    episodes = []
    in_dd, trough = False, 0.0
    for v in dd_arr:
        if v < 0:
            in_dd  = True
            trough = min(trough, v)
        else:
            if in_dd:
                episodes.append(abs(trough))
                trough = 0.0
            in_dd = False
    if in_dd:
        episodes.append(abs(trough))
    if not episodes:
        return 0.0
    episodes.sort(reverse=True)
    avg_top = np.mean(episodes[:top_n_dd])
    return float(ann_r / avg_top) if avg_top > 1e-8 else 0.0

def information_ratio(returns, benchmark_returns, periods=252):
    r      = np.array(returns, dtype=float)
    b      = np.array(benchmark_returns, dtype=float)
    n      = min(len(r), len(b))
    active = r[:n] - b[:n]
    te     = active.std()
    return float((active.mean() / te) * np.sqrt(periods)) if te > 0 else 0.0


# ══════════════════════════════════════════════════════════════════════════════
#  DIEBOLD-MARIANO TEST
# ══════════════════════════════════════════════════════════════════════════════

def diebold_mariano(y_true, pred_a, pred_b, h=1, criterion="mse"):
    """
    H₀: Equal predictive accuracy between model A and B.
    DM stat < 0 and p < 0.05 → A significantly better.
    DM stat > 0 and p < 0.05 → B significantly better.
    """
    y_true = np.array(y_true, dtype=float)
    pred_a = np.array(pred_a, dtype=float)
    pred_b = np.array(pred_b, dtype=float)
    n      = min(len(y_true), len(pred_a), len(pred_b))
    y_true, pred_a, pred_b = y_true[:n], pred_a[:n], pred_b[:n]

    if criterion == "mse":
        e_a = (y_true - pred_a) ** 2
        e_b = (y_true - pred_b) ** 2
    else:
        e_a = np.abs(y_true - pred_a)
        e_b = np.abs(y_true - pred_b)

    d      = e_a - e_b
    d_bar  = d.mean()
    n_lags = max(h - 1, 0)
    gamma  = np.var(d, ddof=1)
    for k in range(1, n_lags + 1):
        cov_k  = np.cov(d[k:], d[:-k])[0, 1]
        gamma += 2 * (1 - k / (n_lags + 1)) * cov_k
    var_d   = gamma / n
    if var_d <= 0:
        var_d = 1e-12
    dm_stat = d_bar / np.sqrt(var_d)
    p_value = float(2 * (1 - stats.norm.cdf(abs(dm_stat))))
    p_value = float(np.clip(p_value, 0.0, 1.0))

    if p_value < 0.05:
        winner     = "Model A better" if dm_stat < 0 else "Model B better"
        conclusion = f"✅ Sig. (p={p_value:.4f}) — {winner}"
    else:
        conclusion = f"— Not significant (p={p_value:.4f})"

    return {
        "dm_statistic": round(dm_stat, 4),
        "p_value":      round(p_value, 4),
        "conclusion":   conclusion,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  MONTE CARLO
# ══════════════════════════════════════════════════════════════════════════════

def monte_carlo_simulation(
    log_returns: np.ndarray,
    n_simulations: int = 500,
    horizon: int = 30,
    initial_capital: float = 10_000.0,
    seed: int = 42,
) -> dict:
    rng     = np.random.default_rng(seed)
    r       = np.array(log_returns, dtype=float)
    r       = r[np.isfinite(r)]
    samples = rng.choice(r, size=(n_simulations, horizon), replace=True)
    paths   = initial_capital * np.exp(np.cumsum(samples, axis=1))
    final   = paths[:, -1]
    max_dds = np.array([abs(max_drawdown(paths[i])) for i in range(n_simulations)])
    pct     = np.percentile(final, [5, 25, 50, 75, 95])
    losses  = initial_capital - final
    var_95  = float(np.percentile(losses, 95))
    cvar_95 = float(losses[losses >= var_95].mean()) if (losses >= var_95).any() else var_95

    return {
        "paths":          paths,
        "final_values":   final,
        "percentiles":    {"p5": pct[0], "p25": pct[1], "p50": pct[2],
                           "p75": pct[3], "p95": pct[4]},
        "prob_loss":      float((final < initial_capital).mean() * 100),
        "mean_final":     float(final.mean()),
        "median_final":   float(np.median(final)),
        "mean_max_dd":    float(max_dds.mean() * 100),
        "var_95":         var_95,
        "cvar_95":        cvar_95,
        "n_simulations":  n_simulations,
        "horizon":        horizon,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  EQUITY CURVE WITH TC + GARCH SIZING
# ══════════════════════════════════════════════════════════════════════════════

def equity_curve_with_tc(
    log_ret_true: np.ndarray,
    log_ret_pred: np.ndarray,
    initial_capital: float = 10_000.0,
    tc_rate: float = 0.001,
    garch_vol: np.ndarray = None,
    vol_target: float = 0.02,
) -> dict:
    n       = min(len(log_ret_true), len(log_ret_pred))
    y_true  = np.array(log_ret_true[:n], dtype=float)
    y_pred  = np.array(log_ret_pred[:n], dtype=float)

    positions = np.sign(y_pred)
    positions[positions == 0] = 1

    if garch_vol is not None and len(garch_vol) >= n:
        gv    = np.array(garch_vol[:n], dtype=float)
        gv    = np.where(gv <= 0, 1e-6, gv)
        scale = np.clip(vol_target / gv, 0.1, 3.0)
        positions = positions * scale

    pos_sign  = np.sign(positions)
    changes   = np.diff(pos_sign, prepend=pos_sign[0])
    tc_mask   = changes != 0
    tc_costs  = tc_rate * np.abs(positions) * tc_mask
    strat_lr  = positions * y_true - tc_costs
    bench_lr  = y_true

    equity    = initial_capital * np.exp(np.cumsum(strat_lr))
    benchmark = initial_capital * np.exp(np.cumsum(bench_lr))

    return {
        "equity":        equity,
        "benchmark":     benchmark,
        "positions":     positions,
        "strat_returns": strat_lr,
        "bench_returns": bench_lr,
        "turnover":      round(float(tc_mask.sum() / n), 4),
        "total_tc_cost": round(float((tc_costs * initial_capital).sum()), 2),
    }


def full_equity_metrics(equity, strat_returns, bench_returns,
                        initial_capital=10_000.0):
    return {
        "Final ($)":           round(float(equity[-1]), 0),
        "Ann. Return (%)":     round(float(np.mean(strat_returns) * 252 * 100), 2),
        "Sharpe":              round(sharpe_ratio(strat_returns), 3),
        "Sortino":             round(sortino_ratio(strat_returns), 3),
        "Calmar":              round(calmar_ratio(strat_returns), 3),
        "Sterling":            round(sterling_ratio(strat_returns), 3),
        "Info Ratio":          round(information_ratio(strat_returns, bench_returns), 3),
        "Max DD (%)":          round(max_drawdown(equity) * 100, 2),
        "Max DD Duration":     max_drawdown_duration(equity),
        "Mean Daily DD (%)":   round(mean_daily_drawdown(equity) * 100, 3),
        "Median Daily DD (%)": round(median_daily_drawdown(equity) * 100, 3),
        "Ulcer Index":         round(ulcer_index(equity), 3),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  evaluate_all
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_all(y_true, y_pred, is_returns=True):
    y_t, y_p = np.array(y_true, dtype=float), np.array(y_pred, dtype=float)
    out = {
        "RMSE": round(rmse(y_t, y_p), 6),
        "MAE":  round(mae(y_t, y_p), 6),
        "MAPE": round(mape(y_t, y_p), 4),
    }
    if is_returns:
        out["Hit Ratio (%)"]      = round(hit_ratio(y_t, y_p), 2)
        out["Direction Acc. (%)"] = round(hit_ratio(y_t, y_p), 2)
        out["Sharpe"]             = round(sharpe_ratio(y_t), 3)
        out["Sortino"]            = round(sortino_ratio(y_t), 3)
    else:
        out["Direction Acc. (%)"] = round(direction_accuracy(y_t, y_p), 2)
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  FULL COMPARISON TABLE
# ══════════════════════════════════════════════════════════════════════════════

def build_comparison_table(
    y_true: np.ndarray,
    model_preds: dict,
    train_tail: float = None,      # last value from training set for proper RW lag
    benchmark_returns: np.ndarray = None,
    is_returns: bool = True,
) -> pd.DataFrame:
    """
    Build ranked comparison including proper Random Walk.

    Random Walk here uses random_walk_insample_with_tail() so it produces
    a TRUE lag-1 series (not a flat line) for fair comparison.

    train_tail: last log-return from training window (for RW pred[0]).
                If None, uses y_true[0] as approximation.
    """
    y_true = np.array(y_true, dtype=float)
    n      = len(y_true)

    # Proper RW: lag-1 of actuals
    tail   = train_tail if train_tail is not None else y_true[0]
    rw_pred = random_walk_insample_with_tail(tail, y_true)

    rows  = {}
    all_p = {"Random Walk": rw_pred, **model_preds}

    for name, pred in all_p.items():
        pred = np.array(pred, dtype=float)
        k    = min(n, len(pred))
        row  = evaluate_all(y_true[:k], pred[:k], is_returns=is_returns)

        if benchmark_returns is not None:
            nb = min(k, len(benchmark_returns))
            row["Info Ratio"] = round(
                information_ratio(y_true[:nb], benchmark_returns[:nb]), 3)

        if name != "Random Walk":
            nr  = min(n, len(rw_pred), len(pred))
            dm  = diebold_mariano(y_true[:nr], pred[:nr], rw_pred[:nr])
            row["DM vs RW (p)"] = dm["p_value"]
            row["DM Result"]    = "✅ Sig." if dm["p_value"] < 0.05 else "—"
        else:
            row["DM vs RW (p)"] = "—"
            row["DM Result"]    = "baseline"

        rows[name] = row

    df = pd.DataFrame(rows).T.reset_index().rename(columns={"index": "Model"})

    lower_better = ("RMSE", "MAE", "MAPE", "DM vs RW (p)")
    for col in df.columns:
        if col in ("Model", "DM Result"):
            continue
        try:
            vals = pd.to_numeric(df[col], errors="coerce")
            idx  = vals.idxmin() if col in lower_better else vals.idxmax()
            if not pd.isna(vals[idx]):
                df.at[idx, col] = str(df.at[idx, col]) + " ★"
        except Exception:
            pass

    return df
