"""
feature_selection.py — Granger Causality + Correlation-based feature selection.

Granger causality: does variable X help predict Y beyond Y's own past?
Combined with |corr| > threshold, builds the exogenous feature set for
multivariate time-series models.
"""
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from statsmodels.tsa.stattools import grangercausalitytests


def granger_test(
    df: pd.DataFrame,
    target_col: str = "log_returns",
    max_lag: int = 5,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Test whether each column in df Granger-causes target_col.

    Returns a DataFrame with columns:
        feature, best_lag, f_stat, p_value, significant, conclusion
    """
    rows = []
    candidates = [c for c in df.columns if c != target_col]
    y = df[[target_col]].dropna()

    for feat in candidates:
        try:
            sub = df[[target_col, feat]].dropna()
            if len(sub) < max_lag * 4:
                continue
            res = grangercausalitytests(sub, maxlag=max_lag, verbose=False)
            # Pick the lag with the lowest p-value (F-test)
            best_lag, best_p, best_f = 1, 1.0, 0.0
            for lag, test_dict in res.items():
                p = test_dict[0]["ssr_ftest"][1]
                f = test_dict[0]["ssr_ftest"][0]
                if p < best_p:
                    best_lag, best_p, best_f = lag, p, f
            rows.append({
                "feature":     feat,
                "best_lag":    best_lag,
                "f_stat":      round(best_f, 4),
                "p_value":     round(best_p, 4),
                "significant": best_p < alpha,
                "conclusion":  "✅ Granger-causes target" if best_p < alpha else "❌ No Granger causality",
            })
        except Exception:
            continue

    return pd.DataFrame(rows).sort_values("p_value").reset_index(drop=True)


def select_features(
    df: pd.DataFrame,
    target_col: str = "log_returns",
    corr_threshold: float = 0.3,
    granger_alpha: float = 0.05,
    max_lag: int = 5,
) -> tuple[list[str], pd.DataFrame, pd.DataFrame]:
    """
    Select exogenous features for multivariate modelling.

    Selection rule:
        Feature included if EITHER:
          - Granger-causes target (p < granger_alpha), OR
          - |correlation with target| > corr_threshold

    Returns:
        selected_features : list of column names
        granger_df        : full Granger results table
        corr_series       : correlation with target (sorted by |corr|)
    """
    # Correlation
    corr = df.corr()[target_col].drop(target_col).sort_values(key=abs, ascending=False)

    # Granger
    granger_df = granger_test(df, target_col=target_col, max_lag=max_lag, alpha=granger_alpha)

    granger_sig = set(granger_df[granger_df["significant"]]["feature"].tolist())
    corr_sig    = set(corr[corr.abs() > corr_threshold].index.tolist())

    selected = sorted(granger_sig | corr_sig)

    # Add lag info to corr df
    corr_df = corr.reset_index()
    corr_df.columns = ["feature", "correlation"]
    corr_df["|corr|"]     = corr_df["correlation"].abs().round(4)
    corr_df["corr_flag"]  = corr_df["|corr|"] > corr_threshold
    corr_df["correlation"] = corr_df["correlation"].round(4)

    return selected, granger_df, corr_df
    