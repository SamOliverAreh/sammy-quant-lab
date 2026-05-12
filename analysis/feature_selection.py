"""
feature_selection.py v5.0
Granger Causality → Correlation filter → PCA dimensionality reduction.
All features are log-return-based (stationary) from features.MODEL_FEATURE_COLS.

Pipeline:
  1. Granger causality test (α threshold)
  2. Correlation filter (|r| threshold)
  3. PCA on union of selected features (user-controlled explained variance %)
  4. Returns PCA components as exog inputs for multivariate models
"""
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from statsmodels.tsa.stattools import grangercausalitytests
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# ── Granger causality ─────────────────────────────────────────────────────────

def granger_test(df, target_col="log_returns", max_lag=5, alpha=0.05):
    rows = []
    candidates = [c for c in df.columns if c != target_col]

    for feat in candidates:
        try:
            sub = df[[target_col, feat]].dropna()
            if len(sub) < max_lag * 5:
                continue
            res = grangercausalitytests(sub, maxlag=max_lag, verbose=False)
            best_lag, best_p, best_f = 1, 1.0, 0.0
            for lag, td in res.items():
                p = td[0]["ssr_ftest"][1]
                f = td[0]["ssr_ftest"][0]
                if p < best_p:
                    best_lag, best_p, best_f = lag, p, f
            rows.append({
                "feature":     feat,
                "best_lag":    best_lag,
                "F-stat":      round(best_f, 4),
                "p-value":     round(best_p, 4),
                "Significant": best_p < alpha,
                "Conclusion":  "✅ Granger-causes" if best_p < alpha else "❌ No causality",
            })
        except Exception:
            continue

    return pd.DataFrame(rows).sort_values("p-value").reset_index(drop=True)


def correlation_filter(df, target_col="log_returns", threshold=0.3):
    corr = df.corr()[target_col].drop(target_col).sort_values(key=abs, ascending=False)
    df_c = corr.reset_index()
    df_c.columns = ["feature", "correlation"]
    df_c["|corr|"]  = df_c["correlation"].abs().round(4)
    df_c["Selected"]= df_c["|corr|"] > threshold
    return df_c


# ── PCA dimensionality reduction ──────────────────────────────────────────────

def run_pca(feature_df, explained_variance_target=0.90):
    """
    Fit PCA on feature_df, retaining components that explain
    >= explained_variance_target of total variance.

    Returns:
        components_df : pd.DataFrame of principal components (named PC1, PC2, …)
        pca           : fitted sklearn PCA object
        scaler        : fitted StandardScaler
        loadings_df   : feature loadings per component
        n_components  : number of components retained
        explained     : cumulative explained variance ratios
    """
    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(feature_df.dropna())

    # Fit full PCA first to get explained variance curve
    pca_full = PCA(random_state=42)
    pca_full.fit(X_sc)
    cum_var  = np.cumsum(pca_full.explained_variance_ratio_)
    n_comps  = int(np.searchsorted(cum_var, explained_variance_target) + 1)
    n_comps  = min(n_comps, X_sc.shape[1])

    # Refit with chosen n_components
    pca = PCA(n_components=n_comps, random_state=42)
    comps = pca.fit_transform(X_sc)

    idx   = feature_df.dropna().index
    col_names = [f"PC{i+1}" for i in range(n_comps)]
    components_df = pd.DataFrame(comps, index=idx, columns=col_names)

    # Loadings (feature contribution per PC)
    loadings_df = pd.DataFrame(
        pca.components_.T,
        index=feature_df.columns,
        columns=col_names,
    ).round(4)

    return (
        components_df, pca, scaler,
        loadings_df, n_comps,
        pca_full.explained_variance_ratio_,
        cum_var,
    )


# ── Main pipeline ─────────────────────────────────────────────────────────────

def select_features_with_pca(
    df_feat,
    target_col="log_returns",
    granger_alpha=0.05,
    corr_threshold=0.3,
    pca_variance=0.90,
    max_lag=5,
):
    """
    Full pipeline:
      1. Granger test all MODEL_FEATURE_COLS against log_returns
      2. Correlation filter
      3. Union of significant features → PCA
      4. Return PCA components as multivariate inputs

    Returns:
        exog_components : pd.DataFrame — PC columns aligned to df_feat index
        granger_df      : Granger results table
        corr_df         : correlation table
        loadings_df     : PCA loadings table
        pca_meta        : dict with n_components, explained variance info
        feature_pool    : list of features that entered PCA
    """
    from data_pipeline.features import MODEL_FEATURE_COLS, get_model_features

    feat_df = get_model_features(df_feat)
    if target_col in df_feat.columns:
        full_df = pd.concat([df_feat[[target_col]], feat_df], axis=1).dropna()
    else:
        full_df = feat_df.dropna()

    # Step 1: Granger
    granger_df = granger_test(full_df, target_col=target_col,
                               max_lag=max_lag, alpha=granger_alpha)
    granger_sig = set(granger_df[granger_df["Significant"]]["feature"].tolist())

    # Step 2: Correlation
    corr_df  = correlation_filter(full_df, target_col=target_col, threshold=corr_threshold)
    corr_sig = set(corr_df[corr_df["Selected"]]["feature"].tolist())

    # Step 3: Union
    feature_pool = sorted(granger_sig | corr_sig)

    if not feature_pool:
        return None, granger_df, corr_df, None, {}, []

    pca_input = full_df[feature_pool].dropna()

    # Step 4: PCA
    (comps, pca, scaler, loadings, n_comps,
     ev_ratios, cum_ev) = run_pca(pca_input, pca_variance)

    pca_meta = {
        "n_components":        n_comps,
        "n_features_in":       len(feature_pool),
        "explained_variance":  [round(float(v), 4) for v in ev_ratios[:n_comps]],
        "cumulative_variance": round(float(cum_ev[n_comps - 1]), 4),
        "target_variance":     pca_variance,
        "all_ev_ratios":       [round(float(v), 4) for v in ev_ratios],
        "all_cum_ev":          [round(float(v), 4) for v in cum_ev],
    }

    return comps, granger_df, corr_df, loadings, pca_meta, feature_pool


# Keep backward-compat alias
def select_features(df, target_col="log_returns",
                    corr_threshold=0.3, granger_alpha=0.05, max_lag=5):
    """Legacy wrapper — returns (selected_cols, granger_df, corr_df)."""
    granger_df = granger_test(df, target_col=target_col,
                               max_lag=max_lag, alpha=granger_alpha)
    corr_df    = correlation_filter(df, target_col=target_col, threshold=corr_threshold)
    g_sig = set(granger_df[granger_df["Significant"]]["feature"].tolist())
    c_sig = set(corr_df[corr_df["Selected"]]["feature"].tolist())
    return sorted(g_sig | c_sig), granger_df, corr_df
