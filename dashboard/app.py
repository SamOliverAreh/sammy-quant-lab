"""
Sammy Quant Lab — Dashboard v4.0
Log-returns target · Statistical tests · Granger feature selection
Multivariate models · Full evaluation suite · Equity curve with TC + GARCH sizing
Dark / Light mode · Multi-model forecast · Random Walk benchmark
"""
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# ── Pipeline ──────────────────────────────────────────────────────────────────
from data_pipeline.ingestion      import fetch_data, get_categories, get_category
from data_pipeline.preprocessing  import preprocess, get_target_series, logret_to_price
from data_pipeline.features       import create_features

# ── Analysis ──────────────────────────────────────────────────────────────────
from analysis.statistical_tests   import run_all_tests, stationarity_verdict
from analysis.feature_selection   import select_features

# ── Models ────────────────────────────────────────────────────────────────────
from models.statistical.arima     import arima_forecast, find_best_arima
from models.statistical.ets       import ets_forecast
from models.statistical.garch     import garch_volatility, garch_summary, garch_insample_vol
from models.ml.lstm               import train_lstm, forecast_lstm
from models.ml.gru                import train_gru, forecast_gru
from models.ml.xgboost_model      import train_xgboost, forecast_xgboost, XGB_AVAILABLE
from models.ml.random_forest      import train_rf, forecast_rf

# ── Evaluation ────────────────────────────────────────────────────────────────
from evaluation.metrics import (
    evaluate_all, build_comparison_table,
    random_walk_forecast, sortino_ratio, information_ratio,
    sharpe_ratio, max_drawdown, equity_curve_with_tc, diebold_mariano
)

# ── Optional ─────────────────────────────────────────────────────────────────
try:
    from models.statistical.prophet_model import prophet_forecast, PROPHET_AVAILABLE
except Exception:
    PROPHET_AVAILABLE = False
try:
    from models.ml.transformer_model import train_transformer, forecast_transformer
    TRANSFORMER_AVAILABLE = True
except Exception:
    TRANSFORMER_AVAILABLE = False
try:
    from models.hybrid.arima_lstm  import hybrid_forecast
    from models.hybrid.ets_gru     import ets_gru_forecast
    HYBRID_AVAILABLE = True
except Exception:
    HYBRID_AVAILABLE = False
try:
    from models.ensemble.stacked import stacked_ensemble_forecast
    ENSEMBLE_AVAILABLE = True
except Exception:
    ENSEMBLE_AVAILABLE = False

# ─── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="Sammy Quant Lab", page_icon="⚡",
                   layout="wide", initial_sidebar_state="collapsed")

# ─── Dark / Light mode toggle ─────────────────────────────────────────────────
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True

# ─── CSS ─────────────────────────────────────────────────────────────────────
def get_css(dark: bool) -> str:
    if dark:
        bg, s1, s2 = "#060a14", "#090d1c", "#0c1225"
        border, border2 = "#142030", "#1e3050"
        text, muted = "#c8d8e8", "#4a6a8a"
        accent, a2, green = "#00d4ff", "#9b4dff", "#00e676"
        card_bg = "linear-gradient(135deg,#0b1220,#0d1830)"
        plot_bg = "rgba(9,13,28,0.95)"
        grid_col = "#0e1e35"
        input_bg = "#0b1220"
        winner_bg = "rgba(0,230,118,.08)"
        winner_border = "rgba(0,230,118,.25)"
    else:
        bg, s1, s2 = "#f0f4f8", "#ffffff", "#e8edf5"
        border, border2 = "#d0dae8", "#b0c0d8"
        text, muted = "#1a2a3a", "#5a7090"
        accent, a2, green = "#0060cc", "#6020cc", "#009940"
        card_bg = "linear-gradient(135deg,#ffffff,#f0f4f8)"
        plot_bg = "rgba(248,250,255,0.95)"
        grid_col = "#d8e4f0"
        input_bg = "#ffffff"
        winner_bg = "rgba(0,153,64,.06)"
        winner_border = "rgba(0,153,64,.3)"

    return f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Outfit:wght@300;400;600;700;800&display=swap');
html,body,[class*="css"]{{font-family:'Outfit',sans-serif;}}
.main,.stApp{{background:{bg};}}
section[data-testid="stSidebar"]{{background:{s1};border-right:1px solid {border};}}
h1,h2,h3{{font-family:'Outfit',sans-serif;color:{text};}}
p,li,label{{color:{text};}}
.ql-title{{
  font-size:clamp(1.4rem,5vw,2.4rem);font-weight:800;
  background:linear-gradient(135deg,{accent} 0%,{a2} 100%);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;
  line-height:1.15;
}}
.ql-sub{{color:{muted};font-size:clamp(.75rem,2.5vw,.9rem);margin-top:4px;}}
.metric-card{{
  background:{card_bg};border:1px solid {border};border-radius:14px;
  padding:clamp(12px,3vw,20px);text-align:center;min-height:90px;
}}
.metric-val{{font-family:'DM Mono',monospace;font-size:clamp(1.1rem,3.5vw,1.7rem);font-weight:500;color:{accent};}}
.metric-lbl{{font-size:clamp(.65rem,2vw,.75rem);color:{muted};text-transform:uppercase;letter-spacing:.08em;margin-top:4px;}}
.metric-delta{{font-size:.82rem;margin-top:5px;}}
.pos{{color:{green};}} .neg{{color:#ff4466;}}
.sec-hdr{{
  font-size:clamp(.85rem,2.5vw,1rem);font-weight:600;color:{text};
  border-left:3px solid {accent};padding-left:10px;margin:22px 0 12px 0;
}}
.badge{{display:inline-block;padding:3px 11px;border-radius:20px;font-size:.72rem;font-family:'DM Mono',monospace;}}
.badge-live  {{background:{"#001a0f" if dark else "#e6fff2"};color:{green};border:1px solid {green};}}
.badge-model {{background:{"#001526" if dark else "#e6f0ff"};color:{accent};border:1px solid {accent};}}
.badge-cat   {{background:{"#1a0d30" if dark else "#f0e6ff"};color:{a2};border:1px solid {a2};}}
.test-pass{{background:{"#001a0f" if dark else "#e6fff0"};color:{green};border:1px solid {green};
  border-radius:8px;padding:8px 14px;font-size:.82rem;margin:4px 0;}}
.test-warn{{background:{"#1a0c00" if dark else "#fff8e6"};color:#ff9500;border:1px solid #ff9500;
  border-radius:8px;padding:8px 14px;font-size:.82rem;margin:4px 0;}}
.test-fail{{background:{"#1a0010" if dark else "#fff0f0"};color:#ff4466;border:1px solid #ff4466;
  border-radius:8px;padding:8px 14px;font-size:.82rem;margin:4px 0;}}
.ic-winner{{
  background:{winner_bg};border:1px solid {winner_border};border-radius:12px;padding:16px 20px;margin:14px 0;
}}
.ic-winner-title{{font-size:1rem;font-weight:700;color:{green};margin-bottom:4px;}}
.ic-winner-sub{{font-size:.82rem;color:{muted};}}
.stButton>button{{
  background:linear-gradient(135deg,{"#0055cc,#0080ff" if dark else "#0066dd,#0099ff"});
  color:white;border:none;border-radius:10px;padding:10px 24px;
  font-family:'Outfit',sans-serif;font-weight:700;width:100%;transition:.2s;
}}
.stButton>button:hover{{opacity:.85;transform:translateY(-1px);}}
.stSelectbox label,.stSlider label,.stCheckbox label,.stRadio label{{color:{muted}!important;font-size:.83rem;}}
div[data-testid="stMetric"]{{background:{input_bg};border:1px solid {border};border-radius:10px;padding:14px;}}
div[data-testid="stMetric"] label{{color:{muted}!important;}}
div[data-testid="stMetric"] div[data-testid="stMetricValue"]{{color:{accent}!important;font-family:'DM Mono',monospace;}}
.stDataFrame{{border-radius:10px;overflow:hidden;}}
@media(max-width:768px){{
  .modebar{{display:none!important;}}
  .main .block-container{{padding:.75rem .75rem 4rem!important;}}
}}
hr{{border-color:{border};}}
.footer{{text-align:center;color:{muted};font-size:.72rem;padding:24px 0 8px;}}
.footer a{{color:{accent};text-decoration:none;}}
.dm-note{{font-size:.74rem;color:{muted};margin-top:4px;}}
</style>"""

st.markdown(get_css(st.session_state.dark_mode), unsafe_allow_html=True)

# Plot palette based on mode
DARK = st.session_state.dark_mode
PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(9,13,28,0.95)" if DARK else "rgba(248,250,255,0.95)",
    font=dict(family="Outfit, sans-serif",
              color="#7a9abb" if DARK else "#2a4060", size=11),
    xaxis=dict(gridcolor="#0e1e35" if DARK else "#d8e8f0", showgrid=True, zeroline=False,
               color="#4a6a8a" if DARK else "#3a5070"),
    yaxis=dict(gridcolor="#0e1e35" if DARK else "#d8e8f0", showgrid=True, zeroline=False,
               color="#4a6a8a" if DARK else "#3a5070"),
    margin=dict(l=10, r=10, t=40, b=10),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#162035" if DARK else "#c0d0e0",
                borderwidth=1, font=dict(size=10)),
)

# Model colour palette
MODEL_COLORS = {
    "ARIMA": "#00d4ff", "ETS": "#29b6f6", "Prophet": "#4fc3f7",
    "LSTM": "#9b4dff", "GRU": "#b06fff", "Transformer": "#ce93d8",
    "XGBoost": "#ff9500", "Random Forest": "#ffc107",
    "Hybrid ARIMA+LSTM": "#ff4081", "Hybrid ETS+GRU": "#f48fb1",
    "Stacked Ensemble": "#00e676", "Random Walk": "#607080",
}

CATEGORIES = get_categories()

# ─────────────────────────────────────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    # Dark/Light toggle
    mode_lbl = "☀️ Light Mode" if DARK else "🌙 Dark Mode"
    if st.button(mode_lbl, use_container_width=True):
        st.session_state.dark_mode = not st.session_state.dark_mode
        st.rerun()

    st.markdown('<div class="ql-title">⚡ Sammy Quant</div>', unsafe_allow_html=True)
    st.markdown('<div class="ql-sub">Multi-Asset Forecasting Lab v4.0</div>', unsafe_allow_html=True)
    st.markdown("---")

    st.markdown('<div class="sec-hdr">📊 Asset Universe</div>', unsafe_allow_html=True)
    cat_choice  = st.selectbox("Asset Class", list(CATEGORIES.keys()))
    asset_names = CATEGORIES[cat_choice]
    ticker      = st.selectbox("Instrument", asset_names)

    st.markdown('<div class="sec-hdr">📅 Data Range</div>', unsafe_allow_html=True)
    start_date     = st.date_input("Start Date", value=pd.Timestamp("2021-01-01"))
    forecast_steps = st.slider("Forecast Horizon (days)", 5, 90, 30)

    st.markdown('<div class="sec-hdr">🤖 Models to Run</div>', unsafe_allow_html=True)
    st.caption("Select one or more models. All selected models appear in forecast + comparison table.")

    model_selections = {}
    model_defs = {
        "ARIMA": True, "ETS": True,
        "LSTM": False, "GRU": False,
        "XGBoost": XGB_AVAILABLE, "Random Forest": True,
        "Transformer": TRANSFORMER_AVAILABLE and False,
        "Hybrid ARIMA+LSTM": HYBRID_AVAILABLE and False,
        "Hybrid ETS+GRU": HYBRID_AVAILABLE and False,
        "Stacked Ensemble": ENSEMBLE_AVAILABLE and False,
    }
    if PROPHET_AVAILABLE:
        model_defs["Prophet"] = False

    for m, default in model_defs.items():
        model_selections[m] = st.checkbox(m, value=default, key=f"sel_{m}")

    selected_models = [m for m, v in model_selections.items() if v]

    st.markdown('<div class="sec-hdr">📐 Feature Selection</div>', unsafe_allow_html=True)
    use_multivariate  = st.checkbox("Enable Multivariate (Granger + Corr)", value=False)
    granger_alpha     = st.slider("Granger α", 0.01, 0.10, 0.05, 0.01) if use_multivariate else 0.05
    corr_threshold    = st.slider("Correlation threshold |r|", 0.1, 0.8, 0.3, 0.05) if use_multivariate else 0.3

    st.markdown('<div class="sec-hdr">⚙️ DL Hyperparams</div>', unsafe_allow_html=True)
    dl_epochs = st.slider("Epochs", 10, 100, 30)
    dl_hidden = st.selectbox("Hidden Units", [32, 64, 128], index=1)
    dl_window = st.slider("Look-back Window", 10, 60, 20)

    st.markdown('<div class="sec-hdr">💰 Simulation</div>', unsafe_allow_html=True)
    tc_rate       = st.slider("Transaction Cost (%)", 0.0, 0.5, 0.1, 0.01) / 100
    use_garch_sz  = st.checkbox("GARCH Position Sizing", value=True)
    vol_target    = st.slider("Vol Target (daily %)", 0.5, 5.0, 2.0, 0.5) / 100 if use_garch_sz else 0.02

    st.markdown("---")
    run_btn = st.button("🚀 Run Analysis", use_container_width=True)

    st.markdown(f"""
    <div style="text-align:center;margin-top:16px;font-size:.73rem;color:{'#2a4060' if DARK else '#5a7090'};">
      Built by <b style="color:{'#00d4ff' if DARK else '#0060cc'};">Sammy</b><br>
      Data Scientist · ML Engineer<br>
      <a href="https://github.com/SamOliverAreh" style="color:{'#0070f3' if DARK else '#0050c0'};">GitHub</a> ·
      <a href="https://www.linkedin.com/in/sam-oliver-areh/" style="color:{'#0070f3' if DARK else '#0050c0'};">LinkedIn</a><br><br>
      <span class="badge badge-live">● LIVE DATA</span>
    </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────────────────────────────────────────
col_t, col_b = st.columns([3, 1])
with col_t:
    cat = get_category(ticker)
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;">
      <div class="ql-title">⚡ Sammy Quant Lab</div>
      <span class="badge badge-model">v4.0 · Log-Returns</span>
      <span class="badge badge-cat">{cat}</span>
    </div>
    <div class="ql-sub">Multi-Asset · {ticker} · {cat} · Log-Returns Target · Real-time Data</div>
    """, unsafe_allow_html=True)
with col_b:
    st.markdown('<div style="padding-top:10px;text-align:right;"><span class="badge badge-live">● LIVE</span></div>',
                unsafe_allow_html=True)
st.markdown("---")

if not run_btn:
    st.info("👈 Open the **sidebar** (☰) — select models, configure settings, then tap **Run Analysis**.")
    with st.expander("📘 What's new in v4.0"):
        st.markdown("""
**v4.0 key changes:**
- 🎯 **Log-returns as target** — all models predict log(P_t/P_{t-1}), not raw price. Forecasts are then inverse-transformed back to price.
- 🧪 **Statistical tests** — ADF, KPSS, Ljung-Box, Jarque-Bera, ARCH-LM run on your data.
- 🔗 **Granger causality + correlation** — feature selection for multivariate models.
- 📊 **Multi-model comparison** — run multiple models at once, all plotted together.
- 🎯 **Random Walk benchmark** — every model compared against the naive baseline.
- 📈 **Diebold-Mariano test** — statistical significance of model differences.
- 💰 **Equity curve** with transaction costs (0.1%) + GARCH-based position sizing.
- 🌓 **Dark / Light mode** toggle.
        """)
    tabs = st.tabs(list(CATEGORIES.keys()))
    for tab, (cn, names) in zip(tabs, CATEGORIES.items()):
        with tab:
            st.dataframe(pd.DataFrame({"Instrument": names, "Class": [cn]*len(names)}),
                         use_container_width=True, hide_index=True)
    st.stop()

if not selected_models:
    st.error("Select at least one model in the sidebar."); st.stop()

# ─────────────────────────────────────────────────────────────────────────────
#  DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────
with st.spinner(f"📡 Fetching {ticker} data..."):
    try:
        df_raw  = fetch_data(ticker, start=str(start_date))
        df      = preprocess(df_raw.copy())
        df_feat = create_features(df.copy())
        series_price    = df["Close"]
        series_logret   = df["log_returns"]   # ← modelling target
        last_price      = float(series_price.iloc[-1])
    except Exception as e:
        st.error(f"Data fetch failed: {e}"); st.stop()

if len(series_logret) < 60:
    st.warning("⚠️ < 60 data points — results may be unreliable.")

# ─────────────────────────────────────────────────────────────────────────────
#  KPI ROW
# ─────────────────────────────────────────────────────────────────────────────
cp    = float(series_price.iloc[-1])
pp    = float(series_price.iloc[-2]) if len(series_price) > 1 else cp
pchg  = (cp - pp) / (pp + 1e-12) * 100
vol30 = float(series_logret.rolling(30).std().iloc[-1] * 100 * np.sqrt(252))
sr    = sharpe_ratio(series_logret.dropna())
mdd   = max_drawdown(series_price.values) * 100
mean_lr = float(series_logret.mean() * 252 * 100)

def kpi(val, lbl, delta=None):
    dh = ""
    if delta is not None:
        cls = "pos" if delta >= 0 else "neg"
        dh = f'<div class="metric-delta {cls}">{delta:+.3f}%</div>'
    return f'<div class="metric-card"><div class="metric-val">{val}</div><div class="metric-lbl">{lbl}</div>{dh}</div>'

c1,c2,c3,c4,c5,c6 = st.columns(6)
with c1: st.markdown(kpi(f"{cp:.4f}", "Price", pchg), unsafe_allow_html=True)
with c2: st.markdown(kpi(f"{mean_lr:.2f}%", "Ann. Log-Ret"), unsafe_allow_html=True)
with c3: st.markdown(kpi(f"{vol30:.2f}%", "Ann. Vol (30d)"), unsafe_allow_html=True)
with c4: st.markdown(kpi(f"{sr:.2f}", "Sharpe Ratio"), unsafe_allow_html=True)
with c5: st.markdown(kpi(f"{mdd:.2f}%", "Max Drawdown"), unsafe_allow_html=True)
with c6: st.markdown(kpi(f"{len(series_logret):,}", "Data Points"), unsafe_allow_html=True)
st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
#  PRICE + LOG-RETURNS CHARTS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="sec-hdr">📈 Price History & Log-Returns</div>', unsafe_allow_html=True)
t_price, t_lr, t_corr = st.tabs(["Price + Indicators", "Log-Returns Series", "Correlation"])

with t_price:
    has_vol = "Volume" in df_raw.columns and df_raw["Volume"].sum() > 0
    nrows   = 2 if has_vol else 1
    fig     = make_subplots(rows=nrows, cols=1, shared_xaxes=True,
                            row_heights=[0.72, 0.28][:nrows], vertical_spacing=0.05)
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Close",
                             line=dict(color="#00d4ff", width=1.5)), row=1, col=1)
    if "ma_21" in df_feat.columns:
        fig.add_trace(go.Scatter(x=df_feat.index, y=df_feat["ma_21"], name="MA21",
                                 line=dict(color="#ff9500", width=1, dash="dot")), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_feat.index, y=df_feat["ma_50"], name="MA50",
                                 line=dict(color="#9b4dff", width=1, dash="dot")), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_feat.index, y=df_feat["bb_upper"],
                                 line=dict(color="#1e3a55", width=1), showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_feat.index, y=df_feat["bb_lower"],
                                 line=dict(color="#1e3a55", width=1),
                                 fill="tonexty", fillcolor="rgba(0,100,200,0.06)",
                                 name="BB Band"), row=1, col=1)
    if has_vol and nrows == 2:
        fig.add_trace(go.Bar(x=df_raw.index, y=df_raw["Volume"], name="Volume",
                             marker_color="rgba(0,212,255,0.18)"), row=2, col=1)
    fig.update_layout(**PLOT_LAYOUT, height=400,
                      title=dict(text=f"{ticker} — Price History", font=dict(size=14)))
    st.plotly_chart(fig, use_container_width=True)

with t_lr:
    fig_lr = make_subplots(rows=2, cols=1, shared_xaxes=True,
                           row_heights=[0.65, 0.35], vertical_spacing=0.06)
    lr_col = "#00d4ff" if DARK else "#0055cc"
    colors = [("#00e676" if DARK else "#009940") if v >= 0 else ("#ff4466" if DARK else "#cc2244")
              for v in series_logret.values]
    fig_lr.add_trace(go.Bar(x=series_logret.index, y=series_logret.values,
                            marker_color=colors, name="Log-Returns"), row=1, col=1)
    fig_lr.add_trace(go.Scatter(x=series_logret.index,
                                y=series_logret.rolling(21).std() * np.sqrt(252),
                                name="Rolling Vol (21d, ann.)",
                                line=dict(color="#ff9500", width=1.5)), row=2, col=1)
    # Mean and Median rolling vol
    rv = series_logret.rolling(21).std() * np.sqrt(252)
    fig_lr.add_hline(y=float(rv.mean()), line=dict(color="#ff9500", dash="dash", width=1),
                     annotation_text=f"Mean Vol: {rv.mean():.3f}", row=2, col=1)
    fig_lr.add_hline(y=float(rv.median()), line=dict(color="#ffc107", dash="dot", width=1),
                     annotation_text=f"Median Vol: {rv.median():.3f}", row=2, col=1)
    fig_lr.update_layout(**PLOT_LAYOUT, height=380,
                         title=dict(text="Log-Returns & Rolling Annualised Volatility", font=dict(size=14)))
    st.plotly_chart(fig_lr, use_container_width=True)

with t_corr:
    fcols = [c for c in ["log_returns","ma_21","ma_50","rsi","macd","bb_width","volatility_30","atr"]
             if c in df_feat.columns]
    cdf   = df_feat[fcols].corr()
    fig_c = go.Figure(go.Heatmap(z=cdf.values, x=cdf.columns, y=cdf.columns,
                                 colorscale="RdBu", zmid=0,
                                 text=np.round(cdf.values, 2), texttemplate="%{text}", showscale=True))
    fig_c.update_layout(**PLOT_LAYOUT, height=360,
                        title=dict(text="Feature Correlation Matrix", font=dict(size=14)))
    st.plotly_chart(fig_c, use_container_width=True)

st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
#  STATISTICAL TESTS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="sec-hdr">🧪 Statistical Tests on Log-Returns</div>', unsafe_allow_html=True)

with st.spinner("Running statistical tests..."):
    test_results = run_all_tests(series_logret)

verdict = stationarity_verdict(test_results["ADF"], test_results["KPSS"])
st.markdown(f'<div class="ic-winner"><div class="ic-winner-title">Stationarity Verdict</div>'
            f'<div class="ic-winner-sub">{verdict}</div></div>', unsafe_allow_html=True)

def test_badge(res, pass_key, pass_val, warn_msg=""):
    reject = res.get("reject_H0", False)
    conc   = res.get("conclusion", "")
    interp = res.get("interpretation", "")
    p      = res.get("p_value") or res.get("lm_p_value", 0)
    stat   = res.get("statistic") or res.get("lm_statistic", 0)
    if pass_val == "reject" and reject:
        cls = "test-pass"
    elif pass_val == "fail" and not reject:
        cls = "test-pass"
    else:
        cls = "test-warn"
    return f'<div class="{cls}"><b>{res["test"]}</b> — stat={stat:.4f} | p={p:.4f}<br><b>{conc}</b><br><small>{interp}</small></div>'

col_t1, col_t2 = st.columns(2)
with col_t1:
    st.markdown(test_badge(test_results["ADF"],   "reject_H0", "reject"), unsafe_allow_html=True)
    st.markdown(test_badge(test_results["KPSS"],  "reject_H0", "fail"),   unsafe_allow_html=True)
    st.markdown(test_badge(test_results["LjungBox"], "reject_H0", "reject"), unsafe_allow_html=True)
with col_t2:
    st.markdown(test_badge(test_results["JarqueBera"], "reject_H0", "fail"), unsafe_allow_html=True)
    arch_r = test_results["ARCHLM"]
    arch_p = arch_r.get("lm_p_value", 1.0)
    cls    = "test-pass" if arch_r["reject_H0"] else "test-warn"
    st.markdown(f'<div class="{cls}"><b>{arch_r["test"]}</b> — LM={arch_r["lm_statistic"]:.4f} | p={arch_p:.4f}<br>'
                f'<b>{arch_r["conclusion"]}</b><br><small>{arch_r["interpretation"]}</small></div>',
                unsafe_allow_html=True)

st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
#  FEATURE SELECTION (GRANGER + CORRELATION)
# ─────────────────────────────────────────────────────────────────────────────
selected_exog_cols = []
exog_df = None

if use_multivariate:
    st.markdown('<div class="sec-hdr">🔗 Feature Selection — Granger Causality & Correlation</div>',
                unsafe_allow_html=True)
    with st.spinner("Running Granger causality tests..."):
        try:
            feat_cols = [c for c in df_feat.columns if c not in
                         ("Close", "Open", "High", "Low", "Volume", "returns", "log_returns")]
            df_for_granger = df_feat[["log_returns"] + feat_cols].dropna()
            selected_exog_cols, granger_df, corr_df = select_features(
                df_for_granger, target_col="log_returns",
                corr_threshold=corr_threshold, granger_alpha=granger_alpha
            )

            g1, g2 = st.columns(2)
            with g1:
                st.markdown("**Granger Causality Results**")
                st.dataframe(granger_df.head(15), use_container_width=True, hide_index=True)
            with g2:
                st.markdown("**Correlation with Log-Returns**")
                st.dataframe(corr_df.head(15), use_container_width=True, hide_index=True)

            if selected_exog_cols:
                st.success(f"✅ Selected exogenous features: {', '.join(selected_exog_cols)}")
                exog_df = df_feat[selected_exog_cols].dropna()
                # Align with series_logret
                common_idx = series_logret.index.intersection(exog_df.index)
                series_logret_mv = series_logret.loc[common_idx]
                exog_df          = exog_df.loc[common_idx]
            else:
                st.warning("No features passed the selection criteria — running univariate.")
                series_logret_mv = series_logret
        except Exception as ge:
            st.warning(f"Granger test failed: {ge}")
            series_logret_mv = series_logret
    st.markdown("---")
else:
    series_logret_mv = series_logret

# ─────────────────────────────────────────────────────────────────────────────
#  FLEXIBLE GARCH
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="sec-hdr">📉 GARCH — Flexible Order Selection</div>', unsafe_allow_html=True)
with st.spinner("Fitting best GARCH specification..."):
    try:
        gsum = garch_summary(series_logret)
        gvol = garch_volatility(series_logret, horizon=forecast_steps)
        gvol_insample = garch_insample_vol(series_logret)

        if "best_order" in gsum:
            best_g = gsum["best_order"]
            best_d = gsum["best_dist"]
            st.markdown(f"""<div class="ic-winner">
              <div class="ic-winner-title">Best GARCH: GARCH{best_g} — Distribution: {best_d}</div>
              <div class="ic-winner-sub">BIC = {gsum['bic']:.2f} · LogL = {gsum['loglikelihood']:.2f}</div>
            </div>""", unsafe_allow_html=True)

        fg = make_subplots(rows=1, cols=2,
                           subplot_titles=["In-Sample Conditional Vol.", f"Forward Vol Forecast ({forecast_steps}d)"])
        fg.add_trace(go.Scatter(x=series_logret.index[-len(gvol_insample):],
                                y=np.array(gvol_insample) * 100,
                                line=dict(color="#ff9500", width=1), name="Cond. Vol"), row=1, col=1)

        # Mean & median lines on in-sample vol
        gv_arr = np.array(gvol_insample) * 100
        fg.add_hline(y=float(np.nanmean(gv_arr)), row=1, col=1,
                     line=dict(color="#ff9500", dash="dash", width=1),
                     annotation_text=f"Mean: {np.nanmean(gv_arr):.2f}%")
        fg.add_hline(y=float(np.nanmedian(gv_arr)), row=1, col=1,
                     line=dict(color="#ffc107", dash="dot", width=1),
                     annotation_text=f"Median: {np.nanmedian(gv_arr):.2f}%")

        fg.add_trace(go.Scatter(y=gvol * 100, mode="lines+markers",
                                line=dict(color="#ff9500", width=2), marker=dict(size=4),
                                name="Fwd Vol"), row=1, col=2)
        fg.update_layout(**PLOT_LAYOUT, height=300)
        st.plotly_chart(fg, use_container_width=True)
    except Exception as ge:
        st.warning(f"GARCH failed: {ge}")
        gvol = None

st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
#  MODEL EXECUTION — ALL SELECTED MODELS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f'<div class="sec-hdr">🤖 Running {len(selected_models)} Model(s) on Log-Returns</div>',
            unsafe_allow_html=True)

model_forecasts   = {}   # model_name → forecast log-returns (np.ndarray, steps)
model_insample    = {}   # model_name → (insample_pred, insample_actual)
model_bt_preds    = {}   # model_name → backtest log-returns

# Backtest split
tw       = min(120, len(series_logret) - forecast_steps - 10)
train_bt = series_logret.iloc[-tw - forecast_steps: -forecast_steps] if tw >= 20 else None
true_bt  = series_logret.iloc[-forecast_steps:] if tw >= 20 else None
exog_bt  = exog_df.iloc[-tw - forecast_steps: -forecast_steps] if exog_df is not None and tw >= 20 else None

progress_bar = st.progress(0)
for i, model_name in enumerate(selected_models):
    progress_bar.progress((i + 1) / len(selected_models), text=f"Training {model_name}...")
    exog_in = exog_df if (use_multivariate and exog_df is not None) else None

    try:
        # ── ARIMA ─────────────────────────────────────────────────────────────
        if model_name == "ARIMA":
            fc, order, ci, ins, act = arima_forecast(series_logret_mv, steps=forecast_steps)
            model_forecasts[model_name] = fc
            model_insample[model_name]  = (ins[-min(len(ins),250):], act[-min(len(act),250):])
            if train_bt is not None:
                bt_fc, _, _, _, _ = arima_forecast(train_bt, steps=forecast_steps)
                model_bt_preds[model_name] = bt_fc

        # ── ETS ───────────────────────────────────────────────────────────────
        elif model_name == "ETS":
            fc, params, ins, act = ets_forecast(series_logret_mv, steps=forecast_steps)
            model_forecasts[model_name] = fc
            model_insample[model_name]  = (ins[-min(len(ins),250):], act[-min(len(act),250):])
            if train_bt is not None:
                bt_fc, _, _, _ = ets_forecast(train_bt, steps=forecast_steps)
                model_bt_preds[model_name] = bt_fc

        # ── LSTM ──────────────────────────────────────────────────────────────
        elif model_name == "LSTM":
            m, sc_y, sc_X, losses, ins, act = train_lstm(
                series_logret_mv, window=dl_window, epochs=dl_epochs, hidden=dl_hidden, exog=exog_in)
            fc = forecast_lstm(m, series_logret_mv, sc_y, steps=forecast_steps,
                               window=dl_window, scaler_X=sc_X, exog=exog_in)
            model_forecasts[model_name] = fc
            model_insample[model_name]  = (ins[-min(len(ins),250):], act[-min(len(act),250):])
            if train_bt is not None:
                bm, bsy, bsx, _, _, _ = train_lstm(train_bt, window=dl_window, epochs=dl_epochs,
                                                    hidden=dl_hidden, exog=exog_bt)
                model_bt_preds[model_name] = forecast_lstm(bm, train_bt, bsy, steps=forecast_steps,
                                                           window=dl_window, scaler_X=bsx, exog=exog_bt)

        # ── GRU ───────────────────────────────────────────────────────────────
        elif model_name == "GRU":
            m, sc_y, sc_X, losses, ins, act = train_gru(
                series_logret_mv, window=dl_window, epochs=dl_epochs, hidden=dl_hidden, exog=exog_in)
            fc = forecast_gru(m, series_logret_mv, sc_y, steps=forecast_steps,
                              window=dl_window, scaler_X=sc_X, exog=exog_in)
            model_forecasts[model_name] = fc
            model_insample[model_name]  = (ins[-min(len(ins),250):], act[-min(len(act),250):])
            if train_bt is not None:
                bm, bsy, bsx, _, _, _ = train_gru(train_bt, window=dl_window, epochs=dl_epochs,
                                                   hidden=dl_hidden, exog=exog_bt)
                model_bt_preds[model_name] = forecast_gru(bm, train_bt, bsy, steps=forecast_steps,
                                                          window=dl_window, scaler_X=bsx, exog=exog_bt)

        # ── Transformer ───────────────────────────────────────────────────────
        elif model_name == "Transformer" and TRANSFORMER_AVAILABLE:
            from models.ml.transformer_model import train_transformer, forecast_transformer
            m, sc_y, losses = train_transformer(series_logret_mv, window=dl_window, epochs=dl_epochs)
            fc = forecast_transformer(m, series_logret_mv, sc_y, steps=forecast_steps, window=dl_window)
            model_forecasts[model_name] = fc
            model_insample[model_name]  = (np.zeros(10), np.zeros(10))  # placeholder
            if train_bt is not None:
                bm, bsy, _ = train_transformer(train_bt, window=dl_window, epochs=dl_epochs)
                model_bt_preds[model_name] = forecast_transformer(bm, train_bt, bsy, steps=forecast_steps, window=dl_window)

        # ── XGBoost ───────────────────────────────────────────────────────────
        elif model_name == "XGBoost" and XGB_AVAILABLE:
            m, lags, ins, act = train_xgboost(series_logret_mv, exog=exog_in)
            fc = forecast_xgboost(m, series_logret_mv, steps=forecast_steps, lags=lags, exog=exog_in)
            model_forecasts[model_name] = fc
            model_insample[model_name]  = (ins[-min(len(ins),250):], act[-min(len(act),250):])
            if train_bt is not None:
                bm, blags, _, _ = train_xgboost(train_bt, exog=exog_bt)
                model_bt_preds[model_name] = forecast_xgboost(bm, train_bt, steps=forecast_steps, lags=blags, exog=exog_bt)

        # ── Random Forest ─────────────────────────────────────────────────────
        elif model_name == "Random Forest":
            m, lags, ins, act = train_rf(series_logret_mv, exog=exog_in)
            fc = forecast_rf(m, series_logret_mv, steps=forecast_steps, lags=lags, exog=exog_in)
            model_forecasts[model_name] = fc
            model_insample[model_name]  = (ins[-min(len(ins),250):], act[-min(len(act),250):])
            if train_bt is not None:
                bm, blags, _, _ = train_rf(train_bt, exog=exog_bt)
                model_bt_preds[model_name] = forecast_rf(bm, train_bt, steps=forecast_steps, lags=blags, exog=exog_bt)

        # ── Prophet ───────────────────────────────────────────────────────────
        elif model_name == "Prophet" and PROPHET_AVAILABLE:
            fc, lo, hi, _ = prophet_forecast(series_logret_mv, steps=forecast_steps)
            model_forecasts[model_name] = fc
            model_insample[model_name]  = (np.zeros(10), np.zeros(10))
            if train_bt is not None:
                bt_fc, _, _, _ = prophet_forecast(train_bt, steps=forecast_steps)
                model_bt_preds[model_name] = bt_fc

        # ── Hybrids ───────────────────────────────────────────────────────────
        elif model_name == "Hybrid ARIMA+LSTM" and HYBRID_AVAILABLE:
            fc, _, _, _ = hybrid_forecast(series_logret_mv, steps=forecast_steps)
            model_forecasts[model_name] = fc
            model_insample[model_name]  = (np.zeros(10), np.zeros(10))
            if train_bt is not None:
                bt_fc, _, _, _ = hybrid_forecast(train_bt, steps=forecast_steps)
                model_bt_preds[model_name] = bt_fc

        elif model_name == "Hybrid ETS+GRU" and HYBRID_AVAILABLE:
            fc, _, _ = ets_gru_forecast(series_logret_mv, steps=forecast_steps)
            model_forecasts[model_name] = fc
            model_insample[model_name]  = (np.zeros(10), np.zeros(10))
            if train_bt is not None:
                bt_fc, _, _ = ets_gru_forecast(train_bt, steps=forecast_steps)
                model_bt_preds[model_name] = bt_fc

        elif model_name == "Stacked Ensemble" and ENSEMBLE_AVAILABLE:
            with st.spinner("Stacked Ensemble — training base models..."):
                fc, _ = stacked_ensemble_forecast(series_logret_mv, steps=forecast_steps)
            model_forecasts[model_name] = fc
            model_insample[model_name]  = (np.zeros(10), np.zeros(10))
            if train_bt is not None:
                bt_fc, _ = stacked_ensemble_forecast(train_bt, steps=forecast_steps)
                model_bt_preds[model_name] = bt_fc

    except Exception as ex:
        st.warning(f"{model_name} failed: {ex}")

progress_bar.empty()

if not model_forecasts:
    st.error("All models failed."); st.stop()

st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
#  IN-SAMPLE: ACTUAL VS PREDICTED (LOG-RETURNS)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="sec-hdr">🔬 In-Sample: Actual vs Predicted Log-Returns</div>',
            unsafe_allow_html=True)
st.caption("Checks the model is not just lagging — a good model should track turns, not just shift the series.")

models_with_insample = {k: v for k, v in model_insample.items()
                        if len(v[0]) > 2 and not np.all(v[0] == 0)}
if models_with_insample:
    fig_ins = go.Figure()
    # Actual once
    first_act = list(models_with_insample.values())[0][1]
    fig_ins.add_trace(go.Scatter(y=first_act, name="Actual",
                                 line=dict(color="#00d4ff", width=1.5)))
    for m_name, (ins_pred, ins_act) in models_with_insample.items():
        n = min(len(ins_pred), len(ins_act))
        fig_ins.add_trace(go.Scatter(y=ins_pred[:n], name=f"{m_name} (pred)",
                                     line=dict(color=MODEL_COLORS.get(m_name, "#888"), width=1, dash="dash")))
    fig_ins.update_layout(**PLOT_LAYOUT, height=340,
                          title=dict(text="In-Sample Fitted vs Actual Log-Returns (last 250 obs)", font=dict(size=14)),
                          xaxis_title="Observation", yaxis_title="Log-Return")
    st.plotly_chart(fig_ins, use_container_width=True)
else:
    st.info("In-sample plots available for ARIMA, ETS, LSTM, GRU, XGBoost, RF.")

st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
#  MULTI-MODEL FORECAST — LOG-RETURNS + PRICE RECONSTRUCTION
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="sec-hdr">🔮 Forecast — Log-Returns & Reconstructed Price</div>',
            unsafe_allow_html=True)

forecast_dates = pd.bdate_range(start=series_logret.index[-1] + pd.Timedelta(days=1),
                                periods=forecast_steps)
if cat_choice == "Crypto":
    forecast_dates = pd.date_range(start=series_logret.index[-1] + pd.Timedelta(days=1),
                                   periods=forecast_steps)

tab_lr_fc, tab_px_fc = st.tabs(["Log-Returns Forecast", "Reconstructed Price Forecast"])

with tab_lr_fc:
    fig_lr_fc = go.Figure()
    # Historical log-returns tail
    hist_n = min(60, len(series_logret))
    fig_lr_fc.add_trace(go.Scatter(
        x=series_logret.index[-hist_n:], y=series_logret.values[-hist_n:],
        name="Historical", line=dict(color="#00d4ff", width=1.2), opacity=0.6))
    # Zero line
    fig_lr_fc.add_hline(y=0, line=dict(color="#607080", width=1, dash="dot"))
    # Each model
    for m_name, fc in model_forecasts.items():
        n = min(len(forecast_dates), len(fc))
        fig_lr_fc.add_trace(go.Scatter(
            x=forecast_dates[:n], y=fc[:n], name=m_name,
            line=dict(color=MODEL_COLORS.get(m_name, "#aaaaaa"), width=2)))
    fig_lr_fc.update_layout(**PLOT_LAYOUT, height=380,
                            title=dict(text=f"Forecasted Log-Returns — {forecast_steps} days", font=dict(size=14)),
                            yaxis_title="Log-Return")
    st.plotly_chart(fig_lr_fc, use_container_width=True)

with tab_px_fc:
    fig_px_fc = go.Figure()
    # Historical price tail
    hist_n = min(120, len(series_price))
    fig_px_fc.add_trace(go.Scatter(
        x=series_price.index[-hist_n:], y=series_price.values[-hist_n:],
        name="Historical Price", line=dict(color="#00d4ff", width=1.5)))
    # Each model reconstructed price
    for m_name, fc in model_forecasts.items():
        price_fc = logret_to_price(fc, last_price)
        n        = min(len(forecast_dates), len(price_fc))
        fig_px_fc.add_trace(go.Scatter(
            x=forecast_dates[:n], y=price_fc[:n], name=m_name,
            line=dict(color=MODEL_COLORS.get(m_name, "#aaaaaa"), width=2)))
        # Connector dot
        fig_px_fc.add_trace(go.Scatter(
            x=[series_price.index[-1], forecast_dates[0]],
            y=[last_price, float(price_fc[0])],
            line=dict(color=MODEL_COLORS.get(m_name, "#aaaaaa"), width=1.2, dash="dot"),
            showlegend=False))
    fig_px_fc.update_layout(**PLOT_LAYOUT, height=400,
                            title=dict(text=f"Reconstructed Price Forecast — {forecast_steps} days", font=dict(size=14)),
                            yaxis_title="Price")
    st.plotly_chart(fig_px_fc, use_container_width=True)

# Forecast table (price)
with st.expander("📋 Forecast Table (Reconstructed Price)"):
    tbl_rows = {"Date": [d.date() for d in forecast_dates[:forecast_steps]]}
    for m_name, fc in model_forecasts.items():
        px_fc = logret_to_price(fc, last_price)
        n     = min(forecast_steps, len(px_fc))
        tbl_rows[m_name] = [f"{v:.4f}" for v in px_fc[:n]] + ["—"] * (forecast_steps - n)
    st.dataframe(pd.DataFrame(tbl_rows), use_container_width=True, hide_index=True)

st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
#  BACKTEST + FULL EVALUATION TABLE
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="sec-hdr">📐 Backtest & Evaluation — All Models vs Random Walk</div>',
            unsafe_allow_html=True)

if train_bt is None or true_bt is None:
    st.warning("Not enough data for backtesting (need >80 observations).")
else:
    # Backtest plot
    fig_bt = go.Figure()
    fig_bt.add_trace(go.Scatter(y=true_bt.values, name="Actual",
                                line=dict(color="#00d4ff", width=2)))
    rw_pred = random_walk_forecast(train_bt, forecast_steps)
    fig_bt.add_trace(go.Scatter(y=rw_pred, name="Random Walk",
                                line=dict(color="#607080", width=1.5, dash="dot")))
    for m_name, bt_fc in model_bt_preds.items():
        n = min(len(true_bt), len(bt_fc))
        fig_bt.add_trace(go.Scatter(y=bt_fc[:n], name=m_name,
                                    line=dict(color=MODEL_COLORS.get(m_name, "#aaa"), width=1.8, dash="dash")))
    fig_bt.update_layout(**PLOT_LAYOUT, height=320,
                         title=dict(text="Backtest: Actual vs All Models (Log-Returns)", font=dict(size=14)),
                         yaxis_title="Log-Return")
    st.plotly_chart(fig_bt, use_container_width=True)

    # Full comparison table
    preds_for_table = dict(model_bt_preds)
    comp_df = build_comparison_table(
        y_true=true_bt.values,
        model_preds=preds_for_table,
        benchmark_returns=true_bt.values,
        is_returns=True,
    )

    st.markdown("**📊 Full Model Comparison Table** — ★ marks best per metric")
    st.dataframe(comp_df, use_container_width=True, hide_index=True)

    # DM test pairwise between models
    if len(model_bt_preds) > 1:
        with st.expander("🔬 Diebold-Mariano Pairwise Tests"):
            st.caption("Tests whether model pairs have statistically different predictive accuracy.")
            dm_rows = []
            model_names_bt = list(model_bt_preds.keys())
            for i in range(len(model_names_bt)):
                for j in range(i+1, len(model_names_bt)):
                    a, b = model_names_bt[i], model_names_bt[j]
                    pa   = model_bt_preds[a]
                    pb   = model_bt_preds[b]
                    n    = min(len(true_bt), len(pa), len(pb))
                    dm   = diebold_mariano(true_bt.values[:n], pa[:n], pb[:n])
                    dm_rows.append({"Model A": a, "Model B": b,
                                    "DM stat": dm["dm_statistic"],
                                    "p-value": dm["p_value"],
                                    "Result": dm["conclusion"]})
            if dm_rows:
                st.dataframe(pd.DataFrame(dm_rows), use_container_width=True, hide_index=True)

st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
#  EQUITY CURVE + RISK DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="sec-hdr">💰 Equity Curve Simulation (with TC + GARCH Sizing)</div>',
            unsafe_allow_html=True)
st.caption(f"Transaction cost: {tc_rate*100:.2f}% per trade · GARCH sizing: {'On' if use_garch_sz else 'Off'}")

if train_bt is not None and true_bt is not None and model_bt_preds:
    fig_eq = go.Figure()

    # Buy-and-hold benchmark
    bh_curve = 10_000 * np.exp(np.cumsum(true_bt.values))
    fig_eq.add_trace(go.Scatter(y=bh_curve, name="Buy & Hold",
                                line=dict(color="#607080", width=1.5, dash="dot")))
    # Random Walk equity
    rw_eq = equity_curve_with_tc(true_bt.values, rw_pred[:len(true_bt)],
                                 tc_rate=tc_rate,
                                 garch_vol=gvol[:len(true_bt)] if (use_garch_sz and gvol is not None) else None,
                                 vol_target=vol_target)
    fig_eq.add_trace(go.Scatter(y=rw_eq["equity"], name="Random Walk",
                                line=dict(color="#607080", width=1.5)))

    ec_metrics = []
    for m_name, bt_fc in model_bt_preds.items():
        n   = min(len(true_bt), len(bt_fc))
        gv  = gvol[:n] if (use_garch_sz and gvol is not None) else None
        ec  = equity_curve_with_tc(true_bt.values[:n], bt_fc[:n],
                                   tc_rate=tc_rate, garch_vol=gv, vol_target=vol_target)
        fig_eq.add_trace(go.Scatter(y=ec["equity"], name=m_name,
                                    line=dict(color=MODEL_COLORS.get(m_name, "#aaa"), width=2)))
        sr_  = sharpe_ratio(ec["strat_returns"])
        so_  = sortino_ratio(ec["strat_returns"])
        mdd_ = max_drawdown(ec["equity"]) * 100
        ir_  = information_ratio(ec["strat_returns"], ec["bench_returns"])
        ec_metrics.append({
            "Model": m_name,
            "Final Value": f"${ec['equity'][-1]:,.0f}",
            "Sharpe":    round(sr_, 3),
            "Sortino":   round(so_, 3),
            "Info Ratio":round(ir_, 3),
            "Max DD (%)":round(mdd_, 2),
            "Turnover":  f"{ec['turnover']*100:.1f}%",
            "TC Cost":   f"${ec['total_tc_cost']:.2f}",
        })

    fig_eq.update_layout(**PLOT_LAYOUT, height=360,
                         title=dict(text="Equity Curve — $10,000 Initial Capital", font=dict(size=14)),
                         yaxis_title="Portfolio Value ($)")
    st.plotly_chart(fig_eq, use_container_width=True)

    # Underwater / Drawdown
    fig_dd = go.Figure()
    for m_name, bt_fc in model_bt_preds.items():
        n    = min(len(true_bt), len(bt_fc))
        gv   = gvol[:n] if (use_garch_sz and gvol is not None) else None
        ec   = equity_curve_with_tc(true_bt.values[:n], bt_fc[:n],
                                    tc_rate=tc_rate, garch_vol=gv, vol_target=vol_target)
        eq   = ec["equity"]
        peak = np.maximum.accumulate(eq)
        dd   = (eq - peak) / (peak + 1e-12) * 100
        fig_dd.add_trace(go.Scatter(y=dd, name=m_name, fill="tozeroy",
                                    line=dict(color=MODEL_COLORS.get(m_name, "#aaa"), width=1),
                                    fillcolor=f"rgba{(*bytes.fromhex(MODEL_COLORS.get(m_name,'#888888').lstrip('#')),20)}"))
        # Mean + Median drawdown
    bh_peak = np.maximum.accumulate(bh_curve)
    bh_dd   = (bh_curve - bh_peak) / (bh_peak + 1e-12) * 100
    fig_dd.add_trace(go.Scatter(y=bh_dd, name="Buy & Hold",
                                line=dict(color="#607080", dash="dot", width=1.5)))
    # Mean/median on drawdown for Buy & Hold
    fig_dd.add_hline(y=float(np.mean(bh_dd)),
                     line=dict(color="#607080", dash="dash", width=1),
                     annotation_text=f"BH Mean DD: {np.mean(bh_dd):.1f}%")
    fig_dd.update_layout(**PLOT_LAYOUT, height=280,
                         title=dict(text="Underwater Equity Curve (Drawdown %)", font=dict(size=14)),
                         yaxis_title="Drawdown (%)")
    st.plotly_chart(fig_dd, use_container_width=True)

    # Equity metrics table
    if ec_metrics:
        st.markdown("**Strategy Performance Summary**")
        st.dataframe(pd.DataFrame(ec_metrics), use_container_width=True, hide_index=True)

st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
#  MALAYSIA EDUCATION
# ─────────────────────────────────────────────────────────────────────────────
with st.expander("📚 Malaysia Investor Education — Interpreting Results"):
    st.markdown("""
**Bacaan untuk Pelabur Malaysia / Reading for Malaysian Investors**

| Metric | Meaning (EN) | Interpretasi (BM) |
|--------|-------------|-------------------|
| **Log-Returns** | log(P_t/P_{t-1}) — additive, unit-free, stationary | Pulangan log — stabil, tanpa unit |
| **ADF / KPSS** | Stationarity tests — stationary series is easier to model | Ujian pegun — siri pegun lebih mudah dimodelkan |
| **Ljung-Box** | Autocorrelation — if present, AR models can exploit it | Autokorelasi — AR/LSTM boleh mengeksploitasi ini |
| **ARCH-LM** | Volatility clustering — justifies GARCH modelling | Pengelompokan turun naik — GARCH diperlukan |
| **Hit Ratio** | % correct directional calls — >55% has practical edge | >55% = kelebihan ramalan arah yang bermakna |
| **Sortino** | Only penalises downside — better than Sharpe for asymmetric returns | Hanya menghukum risiko negatif |
| **Info Ratio** | Active return vs benchmark tracking error | Pulangan aktif berbanding penanda aras |
| **DM Test** | Statistical test — is model A *significantly* better than B? | Adakah model A secara statistik lebih baik? |
| **Granger Causality** | Does X help predict Y beyond Y's own history? | Adakah X membantu meramalkan Y? |

**Malaysia-specific:** USDMYR influenced by BNM policy + oil prices. KLCI correlates with Hang Seng.
**Resources:** [BNM](https://www.bnm.gov.my) · [Bursa](https://www.bursamalaysia.com) · [SC Malaysia](https://www.sc.com.my)
    """)

st.markdown("---")
st.markdown("""
<div class="footer">
  ⚠️ Educational & portfolio purposes only. Not financial advice.<br>
  Built by <b>Sammy</b> ·
  <a href="https://github.com/SamOliverAreh">GitHub</a> ·
  <a href="https://www.linkedin.com/in/sam-oliver-areh/">LinkedIn</a> ·
  <a href="https://sammy-quant-lab.streamlit.app">Live Dashboard</a>
</div>""", unsafe_allow_html=True)
