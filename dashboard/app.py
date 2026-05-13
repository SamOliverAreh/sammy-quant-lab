"""
Sammy Quant Lab — Dashboard v5.0
Master's-level time series analysis portfolio dashboard.

Key fixes & enhancements vs v4.2:
[1]  All models truly uni/multivariate depending on toggle
[2]  All analysis (tests, features, charts) on log_returns only
[3]  ADF/KPSS logic fixed with correct H0 and conflict table
[4]  Instrument & model info tables + full glossary added
[5]  Forecast table, Monte Carlo, equity table all restored
[6]  Model dropdown grouped by category (Statistical/ML/Hybrid/Ensemble)
[7]  4 new hybrids: LSTM+GRU, Transformer+LSTM, XGB+LSTM, ARIMA-GARCH-LSTM
[8]  4 meta-learners in ensemble: Ridge, Lasso, ElasticNet, GBM + Equal-Weight
[9]  PCA after Granger+Corr with user-controlled explained variance slider
[10] Portfolio/research framing throughout
[11] Dark/Light mode full CSS coverage
"""
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# ── Pipeline ──────────────────────────────────────────────────────────────────
from data_pipeline.ingestion     import fetch_data, get_categories, get_category, TICKERS
from data_pipeline.preprocessing import preprocess, logret_to_price
from data_pipeline.features      import create_features, get_model_features, MODEL_FEATURE_COLS

# ── Analysis ──────────────────────────────────────────────────────────────────
from analysis.statistical_tests  import run_all_tests, stationarity_verdict
from analysis.feature_selection  import select_features_with_pca

# ── Statistical models ────────────────────────────────────────────────────────
from models.statistical.arima    import arima_forecast
from models.statistical.ets      import ets_forecast
from models.statistical.garch    import garch_volatility, garch_summary, garch_insample_vol

# ── ML models ─────────────────────────────────────────────────────────────────
from models.ml.lstm              import train_lstm, forecast_lstm
from models.ml.gru               import train_gru,  forecast_gru
from models.ml.xgboost_model     import train_xgboost, forecast_xgboost, XGB_AVAILABLE
from models.ml.random_forest     import train_rf, forecast_rf

# ── Hybrids ───────────────────────────────────────────────────────────────────
from models.hybrid.arima_lstm    import hybrid_forecast          as arima_lstm_fc
from models.hybrid.ets_gru       import ets_gru_forecast         as ets_gru_fc
from models.hybrid.lstm_gru      import lstm_gru_forecast
from models.hybrid.transformer_lstm import transformer_lstm_forecast
from models.hybrid.xgb_lstm      import xgb_lstm_forecast
from models.hybrid.arima_garch_lstm import arima_garch_lstm_forecast

# ── Ensemble ──────────────────────────────────────────────────────────────────
from models.ensemble.stacked     import stacked_ensemble_forecast

# ── Evaluation ────────────────────────────────────────────────────────────────
from evaluation.metrics import (
    evaluate_all, build_comparison_table, random_walk_forecast,
    sharpe_ratio, sortino_ratio, calmar_ratio, sterling_ratio,
    information_ratio, max_drawdown, max_drawdown_duration,
    mean_daily_drawdown, median_daily_drawdown, ulcer_index,
    equity_curve_with_tc, full_equity_metrics,
    monte_carlo_simulation, diebold_mariano,
)

# ── Optional ──────────────────────────────────────────────────────────────────
try:
    from models.statistical.prophet_model import prophet_forecast, PROPHET_AVAILABLE
except Exception:
    PROPHET_AVAILABLE = False

try:
    from models.ml.transformer_model import train_transformer, forecast_transformer
    TRANSFORMER_AVAILABLE = True
except Exception:
    TRANSFORMER_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────────────────
#  MODEL REGISTRY
# ─────────────────────────────────────────────────────────────────────────────
MODEL_REGISTRY = {
    "Statistical": {
        "ARIMA": {
            "desc": "Auto-Regressive Integrated Moving Average. Auto order selection via AIC. "
                    "Best for linear, near-stationary series.",
            "multivariate": True, "paper": "Box & Jenkins (1970)",
        },
        "ETS": {
            "desc": "Exponential Smoothing (Holt damped trend). AIC-optimised. "
                    "Robust to trending series.",
            "multivariate": False, "paper": "Holt (1957), Gardner (1985)",
        },
        "Prophet": {
            "desc": "Facebook/Meta Prophet with trend, seasonality, and changepoints. "
                    "Handles missing data and holidays natively.",
            "multivariate": True, "paper": "Taylor & Letham (2018)",
            "requires": "prophet",
        },
    },
    "Machine Learning": {
        "LSTM": {
            "desc": "Long Short-Term Memory network (PyTorch). 2-layer stacked with dropout "
                    "and gradient clipping. Captures long-range dependencies.",
            "multivariate": True, "paper": "Hochreiter & Schmidhuber (1997)",
        },
        "GRU": {
            "desc": "Gated Recurrent Unit (PyTorch). Lighter than LSTM — often converges "
                    "faster on financial log-returns.",
            "multivariate": True, "paper": "Cho et al. (2014)",
        },
        "Transformer": {
            "desc": "Encoder-only Transformer with positional encoding and multi-head "
                    "self-attention. Captures global sequence dependencies.",
            "multivariate": True, "paper": "Vaswani et al. (2017)",
        },
        "XGBoost": {
            "desc": "Gradient-boosted trees on lag + rolling features of log-returns. "
                    "Fast, interpretable, handles non-linearity well.",
            "multivariate": True, "paper": "Chen & Guestrin (2016)",
            "requires": "xgboost",
        },
        "Random Forest": {
            "desc": "Bootstrap-aggregated regression trees on lag features. "
                    "Strong variance reducer, robust baseline.",
            "multivariate": True, "paper": "Breiman (2001)",
        },
    },
    "Hybrid": {
        "Hybrid ARIMA+LSTM": {
            "desc": "ARIMA captures linear structure; LSTM learns nonlinear residuals. "
                    "Classic decomposition hybrid.",
            "multivariate": True, "paper": "Zhang (2003)",
        },
        "Hybrid ETS+GRU": {
            "desc": "ETS extracts trend/level; GRU corrects nonlinear deviations in residuals.",
            "multivariate": True, "paper": "Smyl (2020)",
        },
        "Hybrid LSTM+GRU": {
            "desc": "LSTM forecasts log-returns; GRU refines on LSTM residuals. "
                    "Combines complementary recurrent architectures.",
            "multivariate": True, "paper": "Kim & Won (2018)",
        },
        "Hybrid Transformer+LSTM": {
            "desc": "Transformer captures global attention patterns; LSTM corrects residuals "
                    "with sequential memory.",
            "multivariate": True, "paper": "Wu et al. (2021), Ding et al. (2022)",
        },
        "Hybrid XGB+LSTM": {
            "desc": "XGBoost captures lag-feature nonlinearity; LSTM learns remaining "
                    "temporal patterns in XGB residuals.",
            "multivariate": True, "paper": "Qiu et al. (2020)",
        },
        "Hybrid ARIMA-GARCH-LSTM": {
            "desc": "Three-stage: ARIMA removes linear mean, GARCH models conditional vol, "
                    "LSTM learns remaining nonlinear structure in vol-standardised residuals.",
            "multivariate": True, "paper": "Ding et al. (2015), Khashei & Bijari (2010)",
        },
    },
    "Ensemble": {
        "Stacked Ensemble": {
            "desc": "Walk-forward stacking of 7 base models (ARIMA, ETS, LSTM, GRU, XGB, RF, "
                    "Transformer). Meta-learner automatically selected by OOS MSE from: "
                    "Ridge, Lasso, ElasticNet, GradientBoosting, Equal-Weight.",
            "multivariate": True, "paper": "Wolpert (1992), Breiman (1996)",
        },
    },
}

# Flat lookup
MODEL_FLAT = {}
for grp, mods in MODEL_REGISTRY.items():
    for mn, meta in mods.items():
        MODEL_FLAT[mn] = {**meta, "group": grp}

# ── Colour palette — maximally distinct ──────────────────────────────────────
MODEL_COLORS = {
    "ARIMA":                    "#00d4ff",
    "ETS":                      "#0044ff",
    "Prophet":                  "#00ff99",
    "LSTM":                     "#cc00ff",
    "GRU":                      "#ff00bb",
    "Transformer":              "#ff6600",
    "XGBoost":                  "#ffdd00",
    "Random Forest":            "#33cc33",
    "Hybrid ARIMA+LSTM":        "#ff2222",
    "Hybrid ETS+GRU":           "#ff8800",
    "Hybrid LSTM+GRU":          "#ff66cc",
    "Hybrid Transformer+LSTM":  "#aa44ff",
    "Hybrid XGB+LSTM":          "#88ff00",
    "Hybrid ARIMA-GARCH-LSTM":  "#ff4444",
    "Stacked Ensemble":         "#00ffdd",
    "Random Walk":              "#888888",
    "Buy & Hold":               "#bbbbbb",
}

CATEGORIES = get_categories()

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Sammy Quant Lab · v5.0",
                   page_icon="⚡", layout="wide",
                   initial_sidebar_state="collapsed")

if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True

# ── CSS ───────────────────────────────────────────────────────────────────────
def get_css(dark):
    B   = "#060a14" if dark else "#f2f5fb"
    S1  = "#090d1c" if dark else "#ffffff"
    BD  = "#142030" if dark else "#d0dae8"
    BD2 = "#1e3050" if dark else "#b0c4da"
    TX  = "#c8d8e8" if dark else "#1a2a3a"
    MU  = "#4a6a8a" if dark else "#5a7090"
    AC  = "#00d4ff" if dark else "#0055cc"
    A2  = "#9b4dff" if dark else "#6020cc"
    GR  = "#00e676" if dark else "#007730"
    CB  = ("linear-gradient(135deg,#0b1220,#0d1830)"
           if dark else "linear-gradient(135deg,#fff,#f0f4f8)")
    IB  = "#0b1220" if dark else "#ffffff"
    TB  = "#090d1c" if dark else "#ffffff"
    TH  = "#0c1428" if dark else "#e8edf5"
    EB  = "#090d1c" if dark else "#f5f8fc"
    IFB = "#001830" if dark else "#e6f0ff"
    WRB = "#1a0c00" if dark else "#fff8e6"
    WB  = "rgba(0,230,118,.08)" if dark else "rgba(0,119,48,.06)"
    WBR = "rgba(0,230,118,.25)" if dark else "rgba(0,119,48,.25)"
    return f"""<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Outfit:wght@300;400;600;700;800&display=swap');
html,body,[class*="css"],.main,.stApp{{font-family:'Outfit',sans-serif!important;background:{B}!important;color:{TX}!important;}}
.main .block-container{{background:{B}!important;padding-top:1.2rem!important;}}
section[data-testid="stSidebar"]{{background:{S1}!important;border-right:1px solid {BD}!important;}}
section[data-testid="stSidebar"] *{{color:{TX}!important;}}
section[data-testid="stSidebar"] .stSelectbox>div>div{{background:{IB}!important;color:{TX}!important;border-color:{BD2}!important;}}
header[data-testid="stHeader"]{{background:{S1}!important;border-bottom:1px solid {BD}!important;}}
header[data-testid="stHeader"] *{{color:{TX}!important;background:{S1}!important;}}
.stDeployButton{{display:none;}}
h1,h2,h3,h4,h5,p,li,span,label{{color:{TX};}}
.ql-title{{font-size:clamp(1.3rem,4vw,2.2rem);font-weight:800;
  background:linear-gradient(135deg,{AC},{A2});
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;line-height:1.15;}}
.ql-sub{{color:{MU};font-size:clamp(.72rem,2vw,.88rem);margin-top:4px;}}
.metric-card{{background:{CB};border:1px solid {BD};border-radius:12px;
  padding:clamp(10px,2.5vw,18px);text-align:center;min-height:84px;}}
.metric-val{{font-family:'DM Mono',monospace;font-size:clamp(1rem,3vw,1.55rem);font-weight:500;color:{AC};}}
.metric-lbl{{font-size:clamp(.6rem,1.8vw,.72rem);color:{MU};text-transform:uppercase;letter-spacing:.08em;margin-top:4px;}}
.metric-delta{{font-size:.8rem;margin-top:4px;}}
.pos{{color:{GR};}} .neg{{color:#ff4466;}}
.sec-hdr{{font-size:clamp(.82rem,2.2vw,.96rem);font-weight:700;color:{TX};
  border-left:3px solid {AC};padding-left:10px;margin:20px 0 10px 0;}}
.badge{{display:inline-block;padding:3px 10px;border-radius:20px;font-size:.7rem;font-family:'DM Mono',monospace;}}
.badge-live {{background:{"#001a0f" if dark else "#e6fff0"};color:{GR};border:1px solid {GR};}}
.badge-model{{background:{"#001526" if dark else "#e6f0ff"};color:{AC};border:1px solid {AC};}}
.badge-cat  {{background:{"#1a0d30" if dark else "#f0e6ff"};color:{A2};border:1px solid {A2};}}
.test-box{{border-radius:8px;padding:10px 14px;font-size:.82rem;margin:4px 0;border:1px solid;}}
.test-green{{background:{"#001a0f" if dark else "#e6fff0"};color:{GR};border-color:{GR};}}
.test-amber{{background:{"#1a1000" if dark else "#fff8e6"};color:#ff9500;border-color:#ff9500;}}
.test-red  {{background:{"#1a0000" if dark else "#fff0f0"};color:#ff4444;border-color:#ff4444;}}
.verdict-box{{border-radius:12px;padding:14px 18px;margin:12px 0;border:1px solid;}}
.verdict-strong{{background:{WB};border-color:{WBR};}}
.verdict-warn  {{background:{"#1a1000" if dark else "#fff8e6"};border-color:#ff9500;}}
.verdict-diff  {{background:{"#001530" if dark else "#e6efff"};border-color:{AC};}}
.verdict-bad   {{background:{"#1a0000" if dark else "#fff0f0"};border-color:#ff4444;}}
.verdict-title{{font-size:1rem;font-weight:700;margin-bottom:4px;}}
.verdict-sub  {{font-size:.82rem;color:{MU};}}
.info-card{{background:{S1};border:1px solid {BD};border-radius:10px;padding:14px 16px;margin:6px 0;}}
.ic-winner{{background:{WB};border:1px solid {WBR};border-radius:12px;padding:14px 18px;margin:12px 0;}}
.ic-winner-title{{font-size:.96rem;font-weight:700;color:{GR};margin-bottom:3px;}}
.ic-winner-sub{{font-size:.8rem;color:{MU};}}
.stButton>button{{background:linear-gradient(135deg,{AC},{A2})!important;
  color:{"#000" if not dark else "#fff"}!important;border:none!important;border-radius:10px!important;
  padding:9px 22px!important;font-family:'Outfit',sans-serif!important;font-weight:700!important;
  width:100%!important;transition:.2s!important;}}
.stButton>button:hover{{opacity:.85!important;transform:translateY(-1px)!important;}}
.stSelectbox label,.stSlider label,.stCheckbox label,.stRadio label,
.stDateInput label,.stNumberInput label{{color:{MU}!important;font-size:.82rem!important;}}
.stSelectbox>div>div{{background:{IB}!important;color:{TX}!important;border-color:{BD2}!important;}}
div[data-baseweb="select"]>div{{background:{IB}!important;color:{TX}!important;border-color:{BD2}!important;}}
div[data-baseweb="popover"] ul{{background:{S1}!important;color:{TX}!important;}}
div[data-baseweb="popover"] li:hover{{background:{BD}!important;}}
div[data-testid="stRadio"] label,div[data-testid="stCheckbox"] label{{color:{TX}!important;}}
div[data-testid="stDateInput"] input{{background:{IB}!important;color:{TX}!important;border-color:{BD2}!important;}}
div[data-testid="stMetric"]{{background:{IB}!important;border:1px solid {BD}!important;border-radius:10px!important;padding:13px!important;}}
div[data-testid="stMetric"] label{{color:{MU}!important;}}
div[data-testid="stMetric"] div[data-testid="stMetricValue"]{{color:{AC}!important;font-family:'DM Mono',monospace!important;}}
.stDataFrame,div[data-testid="stDataFrame"]{{background:{TB}!important;border:1px solid {BD}!important;border-radius:10px!important;overflow:hidden!important;}}
.stDataFrame table,.stDataFrame th,.stDataFrame td{{background:{TB}!important;color:{TX}!important;border-color:{BD}!important;}}
.stDataFrame thead tr th{{background:{TH}!important;color:{TX}!important;font-weight:600;}}
.stDataFrame tbody tr:hover td{{background:{BD}!important;}}
div[data-testid="stTabs"] button{{background:{"#090d1c" if dark else "#f0f4f8"}!important;color:{MU}!important;border-bottom:2px solid transparent!important;font-weight:600;}}
div[data-testid="stTabs"] button[aria-selected="true"]{{background:{S1}!important;color:{AC}!important;border-bottom:2px solid {AC}!important;}}
div[data-testid="stTabContent"]{{background:{B}!important;border:1px solid {BD}!important;border-top:none!important;border-radius:0 0 10px 10px!important;padding:12px!important;}}
div[data-testid="stExpander"]{{background:{EB}!important;border:1px solid {BD}!important;border-radius:10px!important;}}
div[data-testid="stExpander"] summary{{color:{TX}!important;background:{EB}!important;font-weight:600;}}
div[data-testid="stExpander"] div{{background:{EB}!important;color:{TX}!important;}}
div[data-testid="stInfo"]{{background:{IFB}!important;border:1px solid {AC}!important;border-radius:10px!important;color:{TX}!important;}}
div[data-testid="stWarning"]{{background:{WRB}!important;border:1px solid #ff9500!important;border-radius:10px!important;color:{TX}!important;}}
div[data-testid="stSuccess"]{{background:{"#001a0f" if dark else "#e6fff0"}!important;border:1px solid {GR}!important;border-radius:10px!important;color:{TX}!important;}}
div[data-testid="stProgressBar"]>div{{background:{AC}!important;}}
::-webkit-scrollbar{{width:6px;height:6px;}}
::-webkit-scrollbar-track{{background:{B};}}
::-webkit-scrollbar-thumb{{background:{BD2};border-radius:3px;}}
@media(max-width:768px){{.modebar{{display:none!important;}}.main .block-container{{padding:.6rem .6rem 4rem!important;}}}}
hr{{border-color:{BD};margin:18px 0;}}
.footer{{text-align:center;color:{MU};font-size:.7rem;padding:20px 0 6px;}}
.footer a{{color:{AC};text-decoration:none;}}
.glossary-term{{font-weight:700;color:{AC};font-family:'DM Mono',monospace;}}
.research-tag{{display:inline-block;padding:2px 8px;border-radius:4px;
  font-size:.68rem;font-weight:600;background:rgba(155,77,255,.12);color:{A2};
  border:1px solid rgba(155,77,255,.3);margin-left:6px;}}
</style>"""

st.markdown(get_css(st.session_state.dark_mode), unsafe_allow_html=True)
DARK = st.session_state.dark_mode

PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(9,13,28,0.95)" if DARK else "rgba(248,250,255,0.97)",
    font=dict(family="Outfit,sans-serif",
              color="#7a9abb" if DARK else "#2a4060", size=11),
    xaxis=dict(gridcolor="#0e1e35" if DARK else "#d0dde8",
               showgrid=True, zeroline=False),
    yaxis=dict(gridcolor="#0e1e35" if DARK else "#d0dde8",
               showgrid=True, zeroline=False),
    margin=dict(l=10, r=10, t=40, b=10),
    legend=dict(bgcolor="rgba(0,0,0,0)",
                bordercolor="#1e3050" if DARK else "#c0d0e0",
                borderwidth=1, font=dict(size=10)),
)

# ─────────────────────────────────────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    if st.button("☀️ Light Mode" if DARK else "🌙 Dark Mode",
                 use_container_width=True):
        st.session_state.dark_mode = not st.session_state.dark_mode
        st.rerun()

    st.markdown('<div class="ql-title">⚡ Sammy Quant Lab</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="ql-sub">Multi-Asset Time Series · v5.0 · Master\'s Portfolio</div>',
                unsafe_allow_html=True)
    st.markdown("---")

    # Asset
    st.markdown('<div class="sec-hdr">📊 Asset</div>', unsafe_allow_html=True)
    cat_choice = st.selectbox("Asset Class", list(CATEGORIES.keys()))
    ticker     = st.selectbox("Instrument", CATEGORIES[cat_choice])

    st.markdown('<div class="sec-hdr">📅 Data Range</div>', unsafe_allow_html=True)
    start_date     = st.date_input("Start Date", value=pd.Timestamp("2021-01-01"))
    forecast_steps = st.slider("Forecast Horizon (days)", 5, 90, 30)

    # Model selection — grouped dropdowns
    st.markdown('<div class="sec-hdr">🤖 Model Selection</div>', unsafe_allow_html=True)
    st.caption("Models grouped by category. Multi-select within each group.")

    selected_models = []
    for grp, mods in MODEL_REGISTRY.items():
        available = []
        for mn, meta in mods.items():
            req = meta.get("requires", "")
            if req == "prophet" and not PROPHET_AVAILABLE:
                continue
            if req == "xgboost" and not XGB_AVAILABLE:
                continue
            if mn in ("Transformer",
                      "Hybrid Transformer+LSTM") and not TRANSFORMER_AVAILABLE:
                continue
            available.append(mn)

        if not available:
            continue

        defaults = {
            "Statistical":   ["ARIMA", "ETS"],
            "Machine Learning": ["Random Forest"],
            "Hybrid": [],
            "Ensemble": [],
        }.get(grp, [])

        chosen = st.multiselect(
            f"{grp}",
            available,
            default=[m for m in defaults if m in available],
            key=f"grp_{grp}",
        )
        selected_models.extend(chosen)

    # Multivariate
    st.markdown('<div class="sec-hdr">📐 Multivariate Mode</div>',
                unsafe_allow_html=True)
    use_mv = st.checkbox("Enable Multivariate (Granger → PCA)", value=False,
                         help="When enabled: Granger causality → correlation filter → PCA. "
                              "PCA components become exogenous inputs for all capable models.")
    if use_mv:
        g_alpha   = st.slider("Granger significance α", 0.01, 0.10, 0.05, 0.01)
        c_thr     = st.slider("Correlation threshold |r|", 0.1, 0.8, 0.3, 0.05)
        pca_var   = st.slider("PCA explained variance target (%)", 70, 95, 85, 5) / 100
    else:
        g_alpha, c_thr, pca_var = 0.05, 0.3, 0.85

    # DL params
    st.markdown('<div class="sec-hdr">⚙️ DL Hyperparams</div>',
                unsafe_allow_html=True)
    dl_epochs = st.slider("Epochs",       10, 100, 30)
    dl_hidden = st.selectbox("Hidden Units", [32, 64, 128], index=1)
    dl_window = st.slider("Look-back Window", 10, 60, 20)

    # Simulation
    st.markdown('<div class="sec-hdr">💰 Simulation</div>', unsafe_allow_html=True)
    tc_rate      = st.slider("Transaction Cost (%)", 0.0, 0.5, 0.1, 0.01) / 100
    use_garch_sz = st.checkbox("GARCH Position Sizing", value=True)
    vol_target   = (st.slider("Vol Target (daily %)", 0.5, 5.0, 2.0, 0.5) / 100
                    if use_garch_sz else 0.02)
    mc_sims      = st.slider("Monte Carlo Paths", 200, 1000, 500, 100)

    st.markdown("---")
    run_btn = st.button("🚀 Run Analysis", use_container_width=True)

    st.markdown(f"""
    <div style="text-align:center;margin-top:14px;font-size:.71rem;color:{'#2a4060' if DARK else '#5a7090'};">
      <b style="color:{'#00d4ff' if DARK else '#0055cc'};">Sammy Oliver Areh</b><br>
      MSc Statistics · Time Series Research<br>
      <a href="https://github.com/SamOliverAreh" style="color:{'#0070f3' if DARK else '#0050c0'};">GitHub</a> ·
      <a href="https://www.linkedin.com/in/sam-oliver-areh/" style="color:{'#0070f3' if DARK else '#0050c0'};">LinkedIn</a>
      <br><br><span class="badge badge-live">● LIVE MARKET DATA</span>
    </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────────────────────────────────────────
col_t, col_b = st.columns([4, 1])
with col_t:
    cat   = get_category(ticker)
    mv_lbl = "Multivariate + PCA ✅" if use_mv else "Univariate"
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;">
      <div class="ql-title">⚡ Sammy Quant Lab</div>
      <span class="badge badge-model">v5.0 · Log-Returns</span>
      <span class="badge badge-cat">{cat}</span>
      <span class="badge badge-model">{mv_lbl}</span>
    </div>
    <div class="ql-sub">
      Multi-Asset Time Series Forecasting · {ticker} · {cat} ·
      Master's Research Portfolio · MSc Statistics
    </div>""", unsafe_allow_html=True)
with col_b:
    st.markdown('<div style="padding-top:12px;text-align:right;">'
                '<span class="badge badge-live">● LIVE</span></div>',
                unsafe_allow_html=True)
st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
#  LANDING PAGE (before run)
# ─────────────────────────────────────────────────────────────────────────────
if not run_btn:
    # Research framing
    st.markdown('<div class="sec-hdr">🎓 About This Project</div>',
                unsafe_allow_html=True)
    st.markdown(f"""
    <div class="info-card">
    <b>Sammy Oliver Areh</b> <span class="research-tag">MSc Statistics</span>
    <span class="research-tag">Time Series Research</span><br><br>
    This platform is a <b>master's-level research and portfolio project</b> demonstrating
    end-to-end proficiency in quantitative financial time series analysis — from raw data
    engineering to publication-quality statistical testing, multi-paradigm forecasting,
    and practical trading simulation.<br><br>
    <b>Research focus:</b> Comparative evaluation of statistical, machine learning, hybrid and
    ensemble forecasting methods on log-returns of global financial instruments, with rigorous
    pre-modelling validation (ADF, KPSS, Ljung-Box, Jarque-Bera, ARCH-LM), multivariate
    feature selection (Granger causality + PCA), and economic evaluation (Diebold-Mariano,
    Sortino, Calmar, Ulcer Index, Monte Carlo).
    </div>""", unsafe_allow_html=True)

    # Instrument table
    st.markdown('<div class="sec-hdr">🌐 Instrument Universe</div>',
                unsafe_allow_html=True)
    tabs_inst = st.tabs(list(TICKERS.keys()))
    for tab, (cat_n, pairs) in zip(tabs_inst, TICKERS.items()):
        with tab:
            rows = [{"Display Name": nm, "Yahoo Finance Symbol": sym,
                     "Asset Class": cat_n}
                    for nm, sym in pairs.items()]
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # Model catalogue
    st.markdown('<div class="sec-hdr">🤖 Model Catalogue</div>',
                unsafe_allow_html=True)
    model_rows = []
    for grp, mods in MODEL_REGISTRY.items():
        for mn, meta in mods.items():
            model_rows.append({
                "Model":          mn,
                "Category":       grp,
                "Multivariate":   "✅" if meta["multivariate"] else "—",
                "Description":    meta["desc"][:90] + "…",
                "Reference":      meta.get("paper", "—"),
            })
    st.dataframe(pd.DataFrame(model_rows), use_container_width=True, hide_index=True)

    # Glossary
    with st.expander("📖 Glossary of All Terms Used"):
        st.markdown("""
**Statistical Tests**
- <span class="glossary-term">ADF</span> — Augmented Dickey-Fuller. H₀: series has a unit root (non-stationary). Reject H₀ → stationary.
- <span class="glossary-term">KPSS</span> — Kwiatkowski-Phillips-Schmidt-Shin. H₀: series is stationary. Reject H₀ → non-stationary. *Opposite null to ADF.*
- <span class="glossary-term">Trend-Stationary</span> — ADF rejects but KPSS also rejects. Series is stationary around a trend (not a fixed mean). Detrend before modelling.
- <span class="glossary-term">Ljung-Box</span> — Tests for serial autocorrelation. H₀: no autocorrelation. Reject → exploitable linear structure exists.
- <span class="glossary-term">Jarque-Bera</span> — Tests normality via skewness and excess kurtosis. Reject → fat tails; use CVaR, not VaR.
- <span class="glossary-term">ARCH-LM</span> — Tests for volatility clustering. Reject → GARCH modelling is warranted.

**Feature Selection**
- <span class="glossary-term">Granger Causality</span> — Does variable X help predict Y beyond Y's own past values? Based on F-test of restricted vs unrestricted VAR.
- <span class="glossary-term">PCA</span> — Principal Component Analysis. Dimensionality reduction. Retains components explaining ≥ target% of variance. Prevents overfitting from too many correlated exogenous variables.

**Models**
- <span class="glossary-term">ARIMA(p,d,q)</span> — AutoRegressive Integrated Moving Average. p=AR order, d=differencing, q=MA order.
- <span class="glossary-term">ETS</span> — Error-Trend-Seasonality exponential smoothing. Holt damped trend variant used here.
- <span class="glossary-term">GARCH(p,q)</span> — Generalised AutoRegressive Conditional Heteroscedasticity. Models time-varying volatility.
- <span class="glossary-term">LSTM</span> — Long Short-Term Memory. Recurrent neural network with memory gates. Captures long-range temporal dependencies.
- <span class="glossary-term">GRU</span> — Gated Recurrent Unit. Simplified LSTM variant; faster training, comparable performance.
- <span class="glossary-term">Transformer</span> — Attention-based model. Captures global dependencies across the entire sequence simultaneously.
- <span class="glossary-term">XGBoost</span> — Extreme Gradient Boosting. Ensemble of decision trees on lag features; fast and interpretable.
- <span class="glossary-term">Stacked Ensemble</span> — Meta-learning: base model predictions become features for a meta-learner trained on out-of-sample predictions (walk-forward CV).

**Evaluation Metrics**
- <span class="glossary-term">RMSE/MAE/MAPE</span> — Forecast accuracy measures. Lower = better.
- <span class="glossary-term">Hit Ratio</span> — % of periods where predicted return sign matches actual. >55% = practical edge.
- <span class="glossary-term">Sharpe Ratio</span> — (Return - Risk-free) / Std Dev, annualised. >1 acceptable, >2 strong.
- <span class="glossary-term">Sortino Ratio</span> — Like Sharpe but only penalises downside deviation. Better for fat-tailed returns.
- <span class="glossary-term">Calmar Ratio</span> — Annualised return / Max Drawdown. Rewards steady growth relative to worst loss.
- <span class="glossary-term">Sterling Ratio</span> — Annualised return / Average of top-3 drawdowns. More robust than Calmar.
- <span class="glossary-term">Ulcer Index</span> — √(mean of squared drawdowns). Combines drawdown depth and duration.
- <span class="glossary-term">Max DD Duration</span> — Longest consecutive periods spent below the previous peak.
- <span class="glossary-term">Information Ratio</span> — Active return over buy-and-hold / tracking error.
- <span class="glossary-term">Diebold-Mariano</span> — Statistical test of equal predictive accuracy between two models. p<0.05 → significant difference.
- <span class="glossary-term">VaR 95%</span> — Value at Risk: loss exceeded in worst 5% of scenarios.
- <span class="glossary-term">CVaR / ES</span> — Conditional VaR / Expected Shortfall: *average* loss in worst 5% of scenarios. Stricter than VaR.

**Simulation**
- <span class="glossary-term">Transaction Cost</span> — Fee applied each time position sign changes (long ↔ short).
- <span class="glossary-term">GARCH Position Sizing</span> — Scale position inversely with GARCH forecast vol. Reduce exposure when volatility is predicted high.
- <span class="glossary-term">Monte Carlo</span> — Bootstrap resample historical log-returns to simulate thousands of future equity paths.
- <span class="glossary-term">Random Walk</span> — Naive benchmark: ŷₜ = yₜ₋₁ (predict last observed value). On log-returns = predict zero change.
        """, unsafe_allow_html=True)

    st.info("👈 Open the **sidebar** (☰) — select instruments, models, and tap **🚀 Run Analysis**.")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
#  VALIDATION
# ─────────────────────────────────────────────────────────────────────────────
if not selected_models:
    st.error("❌ No models selected. Open the sidebar and choose at least one model.")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
#  DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────
with st.spinner(f"📡 Fetching {ticker} ({cat})..."):
    try:
        df_raw  = fetch_data(ticker, start=str(start_date))
        df      = preprocess(df_raw.copy())
        df_feat = create_features(df.copy())
        s_price = df["Close"]
        s_lr    = df["log_returns"]   # ← modelling target throughout
        last_px = float(s_price.iloc[-1])
    except Exception as e:
        st.error(f"Data fetch failed: {e}")
        st.stop()

if len(s_lr) < 80:
    st.warning("⚠️ < 80 data points — some models may be unreliable.")

# ─────────────────────────────────────────────────────────────────────────────
#  KPI ROW (all from log_returns)
# ─────────────────────────────────────────────────────────────────────────────
cp    = float(s_price.iloc[-1])
pp    = float(s_price.iloc[-2]) if len(s_price) > 1 else cp
pchg  = (cp - pp) / (pp + 1e-12) * 100
vol30 = float(s_lr.rolling(30).std().iloc[-1] * np.sqrt(252) * 100)
sr    = sharpe_ratio(s_lr.dropna())
mdd   = max_drawdown(s_price.values) * 100
annr  = float(s_lr.mean() * 252 * 100)
ui    = ulcer_index(s_price.values)
n_pts = len(s_lr)

def kpi(val, lbl, delta=None, small=False):
    dh = ""
    if delta is not None:
        cls = "pos" if delta >= 0 else "neg"
        dh  = f'<div class="metric-delta {cls}">{delta:+.3f}%</div>'
    sz  = "font-size:.85rem;" if small else ""
    return (f'<div class="metric-card"><div class="metric-val" style="{sz}">{val}</div>'
            f'<div class="metric-lbl">{lbl}</div>{dh}</div>')

c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
with c1: st.markdown(kpi(f"{cp:.4f}", "Last Price", pchg),   unsafe_allow_html=True)
with c2: st.markdown(kpi(f"{annr:.2f}%", "Ann. Return"),     unsafe_allow_html=True)
with c3: st.markdown(kpi(f"{vol30:.2f}%", "Ann. Vol (30d)"), unsafe_allow_html=True)
with c4: st.markdown(kpi(f"{sr:.2f}", "Sharpe Ratio"),       unsafe_allow_html=True)
with c5: st.markdown(kpi(f"{mdd:.2f}%", "Max Drawdown"),     unsafe_allow_html=True)
with c6: st.markdown(kpi(f"{ui:.2f}", "Ulcer Index"),        unsafe_allow_html=True)
with c7: st.markdown(kpi(f"{n_pts:,}", "Data Points"),       unsafe_allow_html=True)
st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
#  PRICE & LOG-RETURNS CHARTS (all on log_returns)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="sec-hdr">📈 Price History & Log-Returns Analysis</div>',
            unsafe_allow_html=True)

t_px, t_lr, t_feat, t_corr = st.tabs(
    ["Price + Overlays", "Log-Returns", "LR Features", "Correlation"])

with t_px:
    has_vol = "Volume" in df_raw.columns and df_raw["Volume"].sum() > 0
    nr = 2 if has_vol else 1
    fig = make_subplots(rows=nr, cols=1, shared_xaxes=True,
                        row_heights=[0.75, 0.25][:nr], vertical_spacing=0.04)
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Close",
                             line=dict(color="#00d4ff", width=1.5)), row=1, col=1)
    if "ma_21" in df_feat.columns:
        fig.add_trace(go.Scatter(x=df_feat.index, y=df_feat["ma_21"], name="MA21",
                                 line=dict(color="#ff9500", width=1, dash="dot")), row=1, col=1)
    if "ma_50" in df_feat.columns:
        fig.add_trace(go.Scatter(x=df_feat.index, y=df_feat["ma_50"], name="MA50",
                                 line=dict(color="#cc44ff", width=1, dash="dot")), row=1, col=1)
    if "bb_upper" in df_feat.columns:
        fig.add_trace(go.Scatter(x=df_feat.index, y=df_feat["bb_upper"],
                                 line=dict(color="#1e3a55" if DARK else "#b0c8e0", width=1),
                                 showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_feat.index, y=df_feat["bb_lower"],
                                 line=dict(color="#1e3a55" if DARK else "#b0c8e0", width=1),
                                 fill="tonexty", fillcolor="rgba(0,100,200,0.05)",
                                 name="BB Band"), row=1, col=1)
    if has_vol and nr == 2:
        fig.add_trace(go.Bar(x=df_raw.index, y=df_raw["Volume"], name="Volume",
                             marker_color="rgba(0,212,255,0.15)"), row=2, col=1)
    fig.update_layout(**PLOT_LAYOUT, height=420,
                      title=dict(text=f"{ticker} — Price History", font=dict(size=14)))
    st.plotly_chart(fig, use_container_width=True)

with t_lr:
    fig_lr = make_subplots(rows=2, cols=1, shared_xaxes=True,
                           row_heights=[0.6, 0.4], vertical_spacing=0.06)
    colors_bar = ["#00e676" if v >= 0 else "#ff4466" for v in s_lr.values]
    fig_lr.add_trace(go.Bar(x=s_lr.index, y=s_lr.values,
                            marker_color=colors_bar, name="Log-Returns"), row=1, col=1)
    fig_lr.add_hline(y=float(s_lr.mean()), row=1, col=1,
                     line=dict(color="#00d4ff", dash="dash", width=1),
                     annotation_text=f"Mean:{s_lr.mean():.4f}")

    rv = s_lr.rolling(21).std() * np.sqrt(252)
    fig_lr.add_trace(go.Scatter(x=s_lr.index, y=rv, name="Ann. Vol (21d)",
                                line=dict(color="#ff9500", width=1.5)), row=2, col=1)
    fig_lr.add_hline(y=float(rv.mean()), row=2, col=1,
                     line=dict(color="#ff9500", dash="dash", width=1),
                     annotation_text=f"Mean:{rv.mean():.3f}")
    fig_lr.add_hline(y=float(rv.median()), row=2, col=1,
                     line=dict(color="#ffd600", dash="dot", width=1),
                     annotation_text=f"Median:{rv.median():.3f}")
    fig_lr.update_layout(**PLOT_LAYOUT, height=380,
                         title=dict(text="Log-Returns & Rolling Annualised Volatility",
                                    font=dict(size=14)))
    st.plotly_chart(fig_lr, use_container_width=True)

with t_feat:
    # Show log-return based features (stationary)
    feat_for_plot = get_model_features(df_feat)
    if not feat_for_plot.empty:
        show_cols = [c for c in ["rv_21","lr_macd","lr_rsi","lr_zscore_21","vol_ratio"]
                     if c in feat_for_plot.columns]
        if show_cols:
            fig_feat = make_subplots(rows=len(show_cols), cols=1,
                                     shared_xaxes=True, vertical_spacing=0.04,
                                     subplot_titles=show_cols)
            colors_ = ["#00d4ff","#ff9500","#cc44ff","#00e676","#ff4466"]
            for r, (col, clr) in enumerate(zip(show_cols, colors_), 1):
                fig_feat.add_trace(
                    go.Scatter(x=feat_for_plot.index, y=feat_for_plot[col],
                               name=col, line=dict(color=clr, width=1)),
                    row=r, col=1)
            fig_feat.update_layout(**PLOT_LAYOUT, height=80 + 100 * len(show_cols),
                                   title=dict(text="Log-Return Based Features",
                                              font=dict(size=14)))
            st.plotly_chart(fig_feat, use_container_width=True)

with t_corr:
    feat_corr_cols = [c for c in MODEL_FEATURE_COLS
                      if c in df_feat.columns and c != "log_returns"]
    avail_corr = ["log_returns"] + feat_corr_cols[:15]
    avail_corr = [c for c in avail_corr if c in df_feat.columns]
    cdf = df_feat[avail_corr].corr()
    fig_c = go.Figure(go.Heatmap(
        z=cdf.values, x=cdf.columns, y=cdf.columns,
        colorscale="RdBu", zmid=0,
        text=np.round(cdf.values, 2), texttemplate="%{text}", showscale=True))
    fig_c.update_layout(**PLOT_LAYOUT, height=420,
                        title=dict(text="Feature Correlation Matrix (Log-Return Features)",
                                   font=dict(size=14)))
    st.plotly_chart(fig_c, use_container_width=True)

st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
#  STATISTICAL TESTS (on log_returns)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="sec-hdr">🧪 Pre-Modelling Statistical Tests (Log-Returns)</div>',
            unsafe_allow_html=True)

with st.spinner("Running statistical battery..."):
    tests = run_all_tests(s_lr)

verd = stationarity_verdict(tests["ADF"], tests["KPSS"])

# Verdict banner
v_css = {"✅ Strongly Stationary":   ("verdict-box verdict-strong", "#00e676"),
          "⚠️ Trend-Stationary":      ("verdict-box verdict-warn",   "#ff9500"),
          "⚠️ Difference-Stationary": ("verdict-box verdict-diff",   "#00d4ff"),
          "❌ Non-Stationary":         ("verdict-box verdict-bad",    "#ff4444"),
         }.get(verd["verdict"], ("verdict-box verdict-warn", "#ff9500"))

st.markdown(f"""
<div class="{v_css[0]}">
  <div class="verdict-title" style="color:{v_css[1]};">{verd['verdict']}</div>
  <div class="verdict-sub">{verd['detail']}</div>
  <div class="verdict-sub" style="margin-top:6px;"><b>Recommended action:</b> {verd['action']}</div>
</div>""", unsafe_allow_html=True)

# ADF / KPSS conflict explanation table
with st.expander("ℹ️ How to read ADF + KPSS together"):
    st.markdown("""
| ADF result | KPSS result | Interpretation | Action |
|---|---|---|---|
| Reject H₀ (stationary) | Fail to reject H₀ (stationary) | **Strongly Stationary** | Proceed directly |
| Reject H₀ (stationary) | Reject H₀ (non-stationary) | **Trend-Stationary** — stationary around a trend | Detrend series |
| Fail to reject H₀ (non-stationary) | Fail to reject H₀ (stationary) | **Difference-Stationary** — stochastic trend | Apply differencing (d=1) |
| Fail to reject H₀ (non-stationary) | Reject H₀ (non-stationary) | **Non-Stationary** — both tests agree | Difference + detrend |

*ADF H₀: series has a unit root (non-stationary). KPSS H₀: series is stationary. These are opposite null hypotheses.*
    """)

# ── tbox: render one test result card ────────────────────────────────────────
def tbox(r, positive_means_reject=True):
    """
    Render a styled test result card.
    positive_means_reject=True:  green when reject_H0 (ADF, LjungBox, ARCH-LM)
    positive_means_reject=False: green when fail-to-reject (KPSS, JarqueBera)
    """
    reject  = r.get("reject_H0", False)
    p_disp  = r.get("p_display") or r.get("lm_p_display") or str(
                round(r.get("p_value") or r.get("lm_p_value", 1.0), 4))
    stat    = r.get("statistic") or r.get("lm_statistic", 0.0)
    h0      = r.get("H0", "")
    conc    = r.get("conclusion", "")
    interp  = r.get("interpretation", "")
    p_note  = r.get("p_note", "")
    p_warn  = r.get("p_bound_warning", "")

    # Colour: green = good result (series is stationary / test confirms expectation)
    if positive_means_reject:
        css = "test-green" if reject else "test-amber"
    else:
        css = "test-green" if not reject else "test-amber"

    # KPSS: show stat vs critical values prominently
    kpss_crits = ""
    if "crit_5pct" in r and "crit_10pct" in r:
        kpss_crits = (f"<br><small>Critical values: 10%={r.get('crit_10pct','—')} · "
                      f"5%={r.get('crit_5pct','—')} · "
                      f"2.5%={r.get('crit_2_5pct', r.get('crit_5pct','—'))} · "
                      f"1%={r.get('crit_1pct','—')}</small>")

    p_warn_html = (f"<br><small style='color:#ff9500;'>{p_warn}</small>"
                   if p_warn else "")

    return (
        f'<div class="test-box {css}">'
        f'<b>{r["test"]}</b><br>'
        f'<small>H₀: {h0}</small><br>'
        f'stat = {float(stat):.4f} &nbsp;·&nbsp; p = {p_disp}'
        f'{kpss_crits}'
        f'{p_warn_html}'
        f'<br><b>{conc}</b>'
        f'<br><small>{interp}</small>'
        f'<br><small style="opacity:.7;">{p_note}</small>'
        f'</div>'
    )

# Test result cards
co1, co2 = st.columns(2)
with co1:
    st.markdown(tbox(tests["ADF"],       positive_means_reject=True),  unsafe_allow_html=True)
    st.markdown(tbox(tests["LjungBox"],  positive_means_reject=True),  unsafe_allow_html=True)
    st.markdown(tbox(tests["ARCHLM"],    positive_means_reject=True),  unsafe_allow_html=True)
with co2:
    st.markdown(tbox(tests["KPSS"],      positive_means_reject=False), unsafe_allow_html=True)
    st.markdown(tbox(tests["JarqueBera"],positive_means_reject=False), unsafe_allow_html=True)

# Summary table
st.markdown("**Test Summary Table**")
summary_rows = []
for key in ["ADF","KPSS","LjungBox","JarqueBera","ARCHLM"]:
    r    = tests[key]
    stat = r.get("statistic") or r.get("lm_statistic", 0.0)
    p_disp = r.get("p_display") or r.get("lm_p_display") or str(
        round(r.get("p_value") or r.get("lm_p_value", 1.0), 4))
    bound = "⚠️ Bounded [0.01,0.10]" if r.get("p_bound_warning") else "Exact"
    summary_rows.append({
        "Test":         r["test"],
        "H₀":           r.get("H0","—"),
        "Statistic":    round(float(stat), 4),
        "p-value":      p_disp,
        "p-value Type": bound,
        "Reject H₀":   "✅ Yes" if r["reject_H0"] else "❌ No",
        "Conclusion":   r["conclusion"],
        "Decision Basis": ("stat vs crit" if key == "KPSS" else "p vs α=0.05"),
    })
st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)
st.caption(
    "⚠️ KPSS p-values are bounded to [0.01, 0.10] by statsmodels — compare statistic "
    "to critical values directly (shown in the test card above) for the correct decision. "
    "ADF p ≈ 0.0000 for log-returns is correct and expected — it reflects overwhelming "
    "evidence of stationarity, not a numerical error."
)

st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
#  MULTIVARIATE: GRANGER → CORRELATION → PCA
# ─────────────────────────────────────────────────────────────────────────────
s_lr_model = s_lr          # univariate target series for models
exog_df    = None          # None = univariate mode

if use_mv:
    st.markdown('<div class="sec-hdr">🔗 Feature Selection: Granger → Corr → PCA</div>',
                unsafe_allow_html=True)
    with st.spinner("Granger causality + PCA pipeline..."):
        try:
            (pca_comps, granger_df, corr_df, loadings_df,
             pca_meta, feature_pool) = select_features_with_pca(
                df_feat,
                target_col="log_returns",
                granger_alpha=g_alpha,
                corr_threshold=c_thr,
                pca_variance=pca_var,
            )

            tab_gr, tab_co, tab_pca, tab_load = st.tabs(
                ["Granger Causality", "Correlation", "PCA Summary", "PCA Loadings"])

            with tab_gr:
                st.caption(f"Testing {len(granger_df)} log-return features against log_returns.")
                st.dataframe(granger_df, use_container_width=True, hide_index=True)

            with tab_co:
                st.dataframe(corr_df, use_container_width=True, hide_index=True)

            with tab_pca:
                if pca_meta:
                    st.markdown(f"""
                    <div class="ic-winner">
                      <div class="ic-winner-title">
                        PCA: {pca_meta['n_components']} components retained
                        (from {pca_meta['n_features_in']} features)
                      </div>
                      <div class="ic-winner-sub">
                        Cumulative explained variance:
                        <b>{pca_meta['cumulative_variance']*100:.1f}%</b>
                        (target: {pca_var*100:.0f}%)
                        · Features in pool: {', '.join(feature_pool[:8])}
                        {'…' if len(feature_pool) > 8 else ''}
                      </div>
                    </div>""", unsafe_allow_html=True)

                    # Scree plot
                    all_ev = pca_meta["all_ev_ratios"]
                    fig_sc = go.Figure()
                    fig_sc.add_trace(go.Bar(
                        x=[f"PC{i+1}" for i in range(len(all_ev))],
                        y=[v * 100 for v in all_ev],
                        marker_color=["#00d4ff" if i < pca_meta["n_components"]
                                      else "#1e3050" for i in range(len(all_ev))],
                        name="Explained Variance (%)"))
                    fig_sc.add_trace(go.Scatter(
                        x=[f"PC{i+1}" for i in range(len(pca_meta["all_cum_ev"]))],
                        y=[v * 100 for v in pca_meta["all_cum_ev"]],
                        name="Cumulative %",
                        line=dict(color="#ff9500", width=2),
                        yaxis="y2"))
                    fig_sc.add_hline(y=pca_var * 100,
                                     line=dict(color="#00e676", dash="dash", width=1),
                                     annotation_text=f"Target {pca_var*100:.0f}%")
                    fig_sc.update_layout(
                        **PLOT_LAYOUT, height=300,
                        title=dict(text="PCA Scree Plot — Explained Variance", font=dict(size=14)),
                        yaxis=dict(title="Var %", gridcolor="#0e1e35" if DARK else "#d0dde8"),
                        yaxis2=dict(title="Cumulative %", overlaying="y", side="right"),
                    )
                    st.plotly_chart(fig_sc, use_container_width=True)

            with tab_load:
                if loadings_df is not None:
                    st.caption("Feature loadings per principal component. "
                               "Larger |value| = stronger contribution.")
                    st.dataframe(loadings_df.style.background_gradient(
                        cmap="coolwarm", axis=None),
                        use_container_width=True)

            if pca_comps is not None and not pca_comps.empty:
                common      = s_lr.index.intersection(pca_comps.index)
                s_lr_model  = s_lr.loc[common]
                exog_df     = pca_comps.loc[common]
                st.success(
                    f"✅ Multivariate mode: {pca_meta['n_components']} PCA components "
                    f"({pca_meta['cumulative_variance']*100:.1f}% variance explained) "
                    f"will be used as exogenous inputs.")
            else:
                st.warning("No features passed selection — running univariate.")

        except Exception as fe:
            st.warning(f"Feature selection failed: {fe}. Running univariate.")

    st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
#  FLEXIBLE GARCH
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="sec-hdr">📉 GARCH — Flexible Order & Distribution Selection</div>',
            unsafe_allow_html=True)
gvol = None
with st.spinner("Fitting best GARCH(p,q)..."):
    try:
        gsum  = garch_summary(s_lr)
        gvol  = garch_volatility(s_lr, horizon=forecast_steps)
        gv_is = garch_insample_vol(s_lr)

        if "best_order" in gsum:
            st.markdown(
                f'<div class="ic-winner">'
                f'<div class="ic-winner-title">'
                f'Best GARCH{gsum["best_order"]} — Distribution: {gsum["best_dist"]}'
                f'</div>'
                f'<div class="ic-winner-sub">'
                f'BIC = {gsum["bic"]:.2f} · Log-Likelihood = {gsum["loglikelihood"]:.2f}'
                f'</div></div>', unsafe_allow_html=True)

        fg = make_subplots(rows=1, cols=2,
                           subplot_titles=["In-Sample Conditional Volatility",
                                           f"Forward Volatility Forecast ({forecast_steps}d)"])
        gva = np.array(gv_is) * 100
        fg.add_trace(go.Scatter(x=s_lr.index[-len(gva):], y=gva,
                                line=dict(color="#ff9500", width=1), name="Cond. Vol"),
                     row=1, col=1)
        fg.add_hline(y=float(np.nanmean(gva)), row=1, col=1,
                     line=dict(color="#ff9500", dash="dash", width=1),
                     annotation_text=f"Mean: {np.nanmean(gva):.2f}%")
        fg.add_hline(y=float(np.nanmedian(gva)), row=1, col=1,
                     line=dict(color="#ffd600", dash="dot", width=1),
                     annotation_text=f"Median: {np.nanmedian(gva):.2f}%")
        fg.add_trace(go.Scatter(y=gvol * 100, mode="lines+markers",
                                line=dict(color="#ff9500", width=2),
                                marker=dict(size=4), name="Fwd Vol"),
                     row=1, col=2)
        fg.update_layout(**PLOT_LAYOUT, height=280)
        st.plotly_chart(fg, use_container_width=True)
    except Exception as ge:
        st.warning(f"GARCH failed: {ge}")

st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
#  MODEL EXECUTION
# ─────────────────────────────────────────────────────────────────────────────
mode_str = ("Multivariate + PCA" if (use_mv and exog_df is not None)
            else "Univariate")
st.markdown(
    f'<div class="sec-hdr">🤖 Running {len(selected_models)} Model(s) — {mode_str}</div>',
    unsafe_allow_html=True)

mforecasts = {}   # model → forecast np.ndarray (log-returns)
minsample  = {}   # model → (fitted, actual) log-returns
mbt_preds  = {}   # model → backtest preds

# Backtest split
tw       = min(120, len(s_lr_model) - forecast_steps - 10)
train_bt = s_lr_model.iloc[-tw - forecast_steps: -forecast_steps] if tw >= 20 else None
true_bt  = s_lr_model.iloc[-forecast_steps:] if tw >= 20 else None
exog_bt  = (exog_df.iloc[-tw - forecast_steps: -forecast_steps]
            if (exog_df is not None and tw >= 20) else None)

prog = st.progress(0)
for i, mn in enumerate(selected_models):
    prog.progress((i + 1) / len(selected_models), text=f"Training {mn}...")
    ex = exog_df   # None = univariate; DataFrame = multivariate PCA components

    try:
        # ── Statistical ───────────────────────────────────────────────────────
        if mn == "ARIMA":
            fc, order, ci, ins, act = arima_forecast(
                s_lr_model, steps=forecast_steps, exog=ex)
            mforecasts[mn] = fc
            minsample[mn]  = (ins[-250:], act[-250:])
            if train_bt is not None:
                bt, _, _, _, _ = arima_forecast(
                    train_bt, steps=forecast_steps, exog=exog_bt)
                mbt_preds[mn] = bt

        elif mn == "ETS":
            fc, params, ins, act = ets_forecast(s_lr_model, steps=forecast_steps)
            mforecasts[mn] = fc
            minsample[mn]  = (ins[-250:], act[-250:])
            if train_bt is not None:
                bt, _, _, _ = ets_forecast(train_bt, steps=forecast_steps)
                mbt_preds[mn] = bt

        elif mn == "Prophet" and PROPHET_AVAILABLE:
            from models.statistical.prophet_model import prophet_forecast
            fc, _, _, _ = prophet_forecast(s_lr_model, steps=forecast_steps, exog=ex)
            mforecasts[mn] = fc
            minsample[mn]  = (np.zeros(5), np.zeros(5))
            if train_bt is not None:
                bt, _, _, _ = prophet_forecast(train_bt, steps=forecast_steps, exog=exog_bt)
                mbt_preds[mn] = bt

        # ── ML ────────────────────────────────────────────────────────────────
        elif mn == "LSTM":
            m, sy, sx, _, ins, act = train_lstm(
                s_lr_model, window=dl_window, epochs=dl_epochs, hidden=dl_hidden, exog=ex)
            fc = forecast_lstm(m, s_lr_model, sy, steps=forecast_steps,
                               window=dl_window, scaler_X=sx, exog=ex)
            mforecasts[mn] = fc
            minsample[mn]  = (ins[-250:], act[-250:])
            if train_bt is not None:
                bm, bsy, bsx, _, _, _ = train_lstm(
                    train_bt, window=dl_window, epochs=dl_epochs, hidden=dl_hidden, exog=exog_bt)
                mbt_preds[mn] = forecast_lstm(bm, train_bt, bsy, steps=forecast_steps,
                                              window=dl_window, scaler_X=bsx, exog=exog_bt)

        elif mn == "GRU":
            m, sy, sx, _, ins, act = train_gru(
                s_lr_model, window=dl_window, epochs=dl_epochs, hidden=dl_hidden, exog=ex)
            fc = forecast_gru(m, s_lr_model, sy, steps=forecast_steps,
                              window=dl_window, scaler_X=sx, exog=ex)
            mforecasts[mn] = fc
            minsample[mn]  = (ins[-250:], act[-250:])
            if train_bt is not None:
                bm, bsy, bsx, _, _, _ = train_gru(
                    train_bt, window=dl_window, epochs=dl_epochs, hidden=dl_hidden, exog=exog_bt)
                mbt_preds[mn] = forecast_gru(bm, train_bt, bsy, steps=forecast_steps,
                                             window=dl_window, scaler_X=bsx, exog=exog_bt)

        elif mn == "Transformer" and TRANSFORMER_AVAILABLE:
            m, sy, sx, _, ins, act = train_transformer(
                s_lr_model, window=dl_window, epochs=dl_epochs, exog=ex)
            fc = forecast_transformer(m, s_lr_model, sy, steps=forecast_steps,
                                      window=dl_window, scaler_X=sx, exog=ex)
            mforecasts[mn] = fc
            minsample[mn]  = (ins[-250:], act[-250:])
            if train_bt is not None:
                bm, bsy, bsx, _, _, _ = train_transformer(
                    train_bt, window=dl_window, epochs=dl_epochs, exog=exog_bt)
                mbt_preds[mn] = forecast_transformer(bm, train_bt, bsy, steps=forecast_steps,
                                                     window=dl_window, scaler_X=bsx, exog=exog_bt)

        elif mn == "XGBoost" and XGB_AVAILABLE:
            m, lags, ins, act = train_xgboost(s_lr_model, exog=ex)
            fc = forecast_xgboost(m, s_lr_model, steps=forecast_steps, lags=lags, exog=ex)
            mforecasts[mn] = fc
            minsample[mn]  = (ins[-250:], act[-250:])
            if train_bt is not None:
                bm, bl, _, _ = train_xgboost(train_bt, exog=exog_bt)
                mbt_preds[mn] = forecast_xgboost(bm, train_bt, steps=forecast_steps,
                                                  lags=bl, exog=exog_bt)

        elif mn == "Random Forest":
            m, lags, ins, act = train_rf(s_lr_model, exog=ex)
            fc = forecast_rf(m, s_lr_model, steps=forecast_steps, lags=lags, exog=ex)
            mforecasts[mn] = fc
            minsample[mn]  = (ins[-250:], act[-250:])
            if train_bt is not None:
                bm, bl, _, _ = train_rf(train_bt, exog=exog_bt)
                mbt_preds[mn] = forecast_rf(bm, train_bt, steps=forecast_steps,
                                            lags=bl, exog=exog_bt)

        # ── Hybrids ───────────────────────────────────────────────────────────
        elif mn == "Hybrid ARIMA+LSTM":
            fc, _, _, _ = arima_lstm_fc(s_lr_model, steps=forecast_steps, exog=ex)
            mforecasts[mn] = fc; minsample[mn] = (np.zeros(5), np.zeros(5))
            if train_bt is not None:
                bt, _, _, _ = arima_lstm_fc(train_bt, steps=forecast_steps, exog=exog_bt)
                mbt_preds[mn] = bt

        elif mn == "Hybrid ETS+GRU":
            fc, _, _ = ets_gru_fc(s_lr_model, steps=forecast_steps, exog=ex)
            mforecasts[mn] = fc; minsample[mn] = (np.zeros(5), np.zeros(5))
            if train_bt is not None:
                bt, _, _ = ets_gru_fc(train_bt, steps=forecast_steps, exog=exog_bt)
                mbt_preds[mn] = bt

        elif mn == "Hybrid LSTM+GRU":
            fc, _, _ = lstm_gru_forecast(
                s_lr_model, steps=forecast_steps,
                epochs=dl_epochs, hidden=dl_hidden, window=dl_window, exog=ex)
            mforecasts[mn] = fc; minsample[mn] = (np.zeros(5), np.zeros(5))
            if train_bt is not None:
                bt, _, _ = lstm_gru_forecast(
                    train_bt, steps=forecast_steps,
                    epochs=dl_epochs, hidden=dl_hidden, window=dl_window, exog=exog_bt)
                mbt_preds[mn] = bt

        elif mn == "Hybrid Transformer+LSTM" and TRANSFORMER_AVAILABLE:
            fc, _, _ = transformer_lstm_forecast(
                s_lr_model, steps=forecast_steps,
                epochs=dl_epochs, hidden=dl_hidden, window=dl_window, exog=ex)
            mforecasts[mn] = fc; minsample[mn] = (np.zeros(5), np.zeros(5))
            if train_bt is not None:
                bt, _, _ = transformer_lstm_forecast(
                    train_bt, steps=forecast_steps,
                    epochs=dl_epochs, hidden=dl_hidden, window=dl_window, exog=exog_bt)
                mbt_preds[mn] = bt

        elif mn == "Hybrid XGB+LSTM" and XGB_AVAILABLE:
            fc, _, _ = xgb_lstm_forecast(
                s_lr_model, steps=forecast_steps,
                epochs=dl_epochs, hidden=dl_hidden, window=dl_window, exog=ex)
            mforecasts[mn] = fc; minsample[mn] = (np.zeros(5), np.zeros(5))
            if train_bt is not None:
                bt, _, _ = xgb_lstm_forecast(
                    train_bt, steps=forecast_steps,
                    epochs=dl_epochs, hidden=dl_hidden, window=dl_window, exog=exog_bt)
                mbt_preds[mn] = bt

        elif mn == "Hybrid ARIMA-GARCH-LSTM":
            fc, _, _, _, _ = arima_garch_lstm_forecast(
                s_lr_model, steps=forecast_steps,
                epochs=dl_epochs, hidden=dl_hidden, window=dl_window, exog=ex)
            mforecasts[mn] = fc; minsample[mn] = (np.zeros(5), np.zeros(5))
            if train_bt is not None:
                bt, _, _, _, _ = arima_garch_lstm_forecast(
                    train_bt, steps=forecast_steps,
                    epochs=dl_epochs, hidden=dl_hidden, window=dl_window, exog=exog_bt)
                mbt_preds[mn] = bt

        # ── Ensemble ──────────────────────────────────────────────────────────
        elif mn == "Stacked Ensemble":
            with st.spinner("Stacked Ensemble — training 7 base models + meta-learners..."):
                fc, base_p, meta_scores, best_meta = stacked_ensemble_forecast(
                    s_lr_model, steps=forecast_steps, exog=ex)
            mforecasts[mn] = fc
            minsample[mn]  = (np.zeros(5), np.zeros(5))
            st.markdown(
                f'<div class="ic-winner">'
                f'<div class="ic-winner-title">Ensemble: Best meta-learner = {best_meta}</div>'
                f'<div class="ic-winner-sub">'
                f'OOS MSE scores — ' +
                " · ".join(f"{k}: {v:.6f}" for k, v in meta_scores.items()) +
                f'</div></div>', unsafe_allow_html=True)
            if train_bt is not None:
                bt, _, _, _ = stacked_ensemble_forecast(
                    train_bt, steps=forecast_steps, exog=exog_bt)
                mbt_preds[mn] = bt

    except Exception as ex_:
        st.warning(f"⚠️ {mn} failed: {ex_}")

prog.empty()

if not mforecasts:
    st.error("All selected models failed. Check sidebar configuration."); st.stop()

st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
#  IN-SAMPLE: ACTUAL vs PREDICTED
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="sec-hdr">🔬 In-Sample Fit: Actual vs Predicted Log-Returns</div>',
            unsafe_allow_html=True)
st.caption("A good model tracks turning points — not just a lag-shifted copy of actual returns.")

valid_ins = {k: v for k, v in minsample.items()
             if len(v[0]) > 5 and not np.all(v[0] == 0)}
if valid_ins:
    fig_ins = go.Figure()
    act_ref = list(valid_ins.values())[0][1]
    fig_ins.add_trace(go.Scatter(
        y=act_ref, name="Actual Log-Returns",
        line=dict(color="#ffffff" if DARK else "#222222", width=1.5)))
    for mn, (ip, ia) in valid_ins.items():
        n = min(len(ip), len(ia))
        fig_ins.add_trace(go.Scatter(
            y=ip[:n], name=f"{mn}",
            line=dict(color=MODEL_COLORS.get(mn, "#888"), width=1.2, dash="dash")))
    fig_ins.update_layout(
        **PLOT_LAYOUT, height=340,
        title=dict(text="In-Sample Fitted vs Actual (last 250 obs)", font=dict(size=14)),
        xaxis_title="Observation", yaxis_title="Log-Return")
    st.plotly_chart(fig_ins, use_container_width=True)
else:
    st.info("In-sample plots available for: ARIMA, ETS, LSTM, GRU, Transformer, XGBoost, RF.")

st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
#  FORECAST: LOG-RETURNS + RECONSTRUCTED PRICE
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f'<div class="sec-hdr">🔮 Forecast — {forecast_steps}-Day Horizon</div>',
            unsafe_allow_html=True)

fc_dates = pd.bdate_range(
    start=s_lr_model.index[-1] + pd.Timedelta(days=1), periods=forecast_steps)
if cat_choice == "Crypto":
    fc_dates = pd.date_range(
        start=s_lr_model.index[-1] + pd.Timedelta(days=1), periods=forecast_steps)

t_lr_fc, t_px_fc = st.tabs(["Log-Returns Forecast", "Reconstructed Price Forecast"])

with t_lr_fc:
    flr = go.Figure()
    hn  = min(60, len(s_lr_model))
    flr.add_trace(go.Scatter(
        x=s_lr_model.index[-hn:], y=s_lr_model.values[-hn:],
        name="Historical", line=dict(color="#ffffff" if DARK else "#333", width=1),
        opacity=0.45))
    flr.add_hline(y=0, line=dict(color="#607080", width=1, dash="dot"))
    for mn, fc in mforecasts.items():
        n = min(len(fc_dates), len(fc))
        flr.add_trace(go.Scatter(
            x=fc_dates[:n], y=fc[:n], name=mn,
            line=dict(color=MODEL_COLORS.get(mn, "#aaa"), width=2)))
    flr.update_layout(**PLOT_LAYOUT, height=360,
                      title=dict(text="Forecasted Log-Returns", font=dict(size=14)),
                      yaxis_title="Log-Return")
    st.plotly_chart(flr, use_container_width=True)

with t_px_fc:
    fpx = go.Figure()
    hn  = min(120, len(s_price))
    fpx.add_trace(go.Scatter(
        x=s_price.index[-hn:], y=s_price.values[-hn:],
        name="Historical Price",
        line=dict(color="#ffffff" if DARK else "#333", width=1.5)))
    for mn, fc in mforecasts.items():
        px_fc = logret_to_price(fc, last_px)
        n     = min(len(fc_dates), len(px_fc))
        fpx.add_trace(go.Scatter(
            x=fc_dates[:n], y=px_fc[:n], name=mn,
            line=dict(color=MODEL_COLORS.get(mn, "#aaa"), width=2)))
        fpx.add_trace(go.Scatter(
            x=[s_price.index[-1], fc_dates[0]],
            y=[last_px, float(px_fc[0])],
            line=dict(color=MODEL_COLORS.get(mn, "#aaa"), width=1, dash="dot"),
            showlegend=False))
    fpx.update_layout(**PLOT_LAYOUT, height=400,
                      title=dict(text="Reconstructed Price Forecast (from Log-Returns)",
                                 font=dict(size=14)),
                      yaxis_title="Price")
    st.plotly_chart(fpx, use_container_width=True)

# Forecast table
with st.expander("📋 Forecast Table — Reconstructed Price"):
    tbl = {"Date": [d.date() for d in fc_dates[:forecast_steps]]}
    for mn, fc in mforecasts.items():
        px  = logret_to_price(fc, last_px)
        n   = min(forecast_steps, len(px))
        chg = [(v - last_px) / last_px * 100 for v in px[:n]]
        tbl[f"{mn} Price"]  = [f"{v:.4f}" for v in px[:n]]  + ["—"] * (forecast_steps - n)
        tbl[f"{mn} Chg%"]   = [f"{c:+.2f}%" for c in chg]   + ["—"] * (forecast_steps - n)
    st.dataframe(pd.DataFrame(tbl), use_container_width=True, hide_index=True)

st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
#  BACKTEST & EVALUATION
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="sec-hdr">📐 Backtest & Evaluation — vs Random Walk Benchmark</div>',
            unsafe_allow_html=True)

if train_bt is None or true_bt is None:
    st.warning("Not enough data for backtesting (need >80 obs).")
else:
    rw_pred = random_walk_forecast(train_bt, forecast_steps)

    fig_bt = go.Figure()
    fig_bt.add_trace(go.Scatter(
        y=true_bt.values, name="Actual",
        line=dict(color="#ffffff" if DARK else "#111", width=2)))
    fig_bt.add_trace(go.Scatter(
        y=rw_pred, name="Random Walk",
        line=dict(color=MODEL_COLORS["Random Walk"], width=1.5, dash="dot")))
    for mn, bt in mbt_preds.items():
        n = min(len(true_bt), len(bt))
        fig_bt.add_trace(go.Scatter(
            y=bt[:n], name=mn,
            line=dict(color=MODEL_COLORS.get(mn, "#aaa"), width=1.8, dash="dash")))
    fig_bt.update_layout(
        **PLOT_LAYOUT, height=320,
        title=dict(text="Backtest: Actual vs All Models (Log-Returns)", font=dict(size=14)),
        yaxis_title="Log-Return")
    st.plotly_chart(fig_bt, use_container_width=True)

    # Full comparison table
    comp_df = build_comparison_table(
        y_true=true_bt.values,
        model_preds=dict(mbt_preds),
        benchmark_returns=true_bt.values,
        is_returns=True,
    )
    st.markdown("**📊 Full Model Comparison — ★ marks best per metric (vs Random Walk baseline)**")
    st.dataframe(comp_df, use_container_width=True, hide_index=True)

    # DM tests
    if len(mbt_preds) > 1:
        with st.expander("🔬 Diebold-Mariano Pairwise Significance Tests"):
            st.caption("H₀: Equal predictive accuracy. p<0.05 → statistically significant difference.")
            dm_rows = []
            nms = list(mbt_preds.keys())
            for ii in range(len(nms)):
                for jj in range(ii + 1, len(nms)):
                    a, b = nms[ii], nms[jj]
                    n    = min(len(true_bt), len(mbt_preds[a]), len(mbt_preds[b]))
                    dm   = diebold_mariano(true_bt.values[:n],
                                           mbt_preds[a][:n], mbt_preds[b][:n])
                    dm_rows.append({
                        "Model A": a, "Model B": b,
                        "DM stat": dm["dm_statistic"],
                        "p-value": dm["p_value"],
                        "Result":  dm["conclusion"],
                    })
            if dm_rows:
                st.dataframe(pd.DataFrame(dm_rows), use_container_width=True, hide_index=True)

st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
#  EQUITY SIMULATION
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    f'<div class="sec-hdr">💰 Equity Simulation — $10,000 Initial Capital</div>',
    unsafe_allow_html=True)
st.caption(f"TC: {tc_rate*100:.2f}%/trade · GARCH sizing: {'On' if use_garch_sz else 'Off'} "
           f"· All models vs Buy & Hold & Random Walk")

if train_bt is not None and true_bt is not None and mbt_preds:
    fig_eq = go.Figure()

    bh_curve = 10_000 * np.exp(np.cumsum(true_bt.values))
    fig_eq.add_trace(go.Scatter(y=bh_curve, name="Buy & Hold",
                                line=dict(color=MODEL_COLORS["Buy & Hold"],
                                         width=2, dash="dot")))

    rw_ec = equity_curve_with_tc(
        true_bt.values, rw_pred[:len(true_bt)], tc_rate=tc_rate,
        garch_vol=(gvol[:len(true_bt)] if (use_garch_sz and gvol is not None) else None),
        vol_target=vol_target)
    fig_eq.add_trace(go.Scatter(y=rw_ec["equity"], name="Random Walk",
                                line=dict(color=MODEL_COLORS["Random Walk"],
                                         width=1.5, dash="dash")))

    all_ec    = {}
    perf_rows = []

    # Buy & Hold row
    perf_rows.append({
        "Model": "Buy & Hold",
        "Final ($)":          f"${bh_curve[-1]:,.0f}",
        "Ann. Return (%)":    round(float(np.mean(true_bt.values) * 252 * 100), 2),
        "Sharpe":             round(sharpe_ratio(true_bt.values), 3),
        "Sortino":            round(sortino_ratio(true_bt.values), 3),
        "Calmar":             round(calmar_ratio(true_bt.values), 3),
        "Sterling":           round(sterling_ratio(true_bt.values), 3),
        "Info Ratio":         "—",
        "Max DD (%)":         round(max_drawdown(bh_curve) * 100, 2),
        "Max DD Duration":    max_drawdown_duration(bh_curve),
        "Mean DD (%)":        round(mean_daily_drawdown(bh_curve) * 100, 3),
        "Median DD (%)":      round(median_daily_drawdown(bh_curve) * 100, 3),
        "Ulcer Index":        round(ulcer_index(bh_curve), 3),
        "Turnover":           "100%",
        "TC Cost":            "$0.00",
    })

    # Random Walk row
    rw_em = full_equity_metrics(rw_ec["equity"], rw_ec["strat_returns"], rw_ec["bench_returns"])
    perf_rows.append({
        "Model": "Random Walk",
        "Final ($)":       f"${rw_em['Final ($)']:,.0f}",
        "Ann. Return (%)": rw_em["Ann. Return (%)"],
        "Sharpe":          rw_em["Sharpe"], "Sortino": rw_em["Sortino"],
        "Calmar":          rw_em["Calmar"], "Sterling": rw_em["Sterling"],
        "Info Ratio":      round(information_ratio(rw_ec["strat_returns"],
                                                    rw_ec["bench_returns"]), 3),
        "Max DD (%)":      rw_em["Max DD (%)"],
        "Max DD Duration": rw_em["Max DD Duration"],
        "Mean DD (%)":     rw_em["Mean Daily DD (%)"],
        "Median DD (%)":   rw_em["Median Daily DD (%)"],
        "Ulcer Index":     rw_em["Ulcer Index"],
        "Turnover":        f"{rw_ec['turnover']*100:.1f}%",
        "TC Cost":         f"${rw_ec['total_tc_cost']:.2f}",
    })

    for mn, bt in mbt_preds.items():
        n  = min(len(true_bt), len(bt))
        gv = gvol[:n] if (use_garch_sz and gvol is not None) else None
        ec = equity_curve_with_tc(
            true_bt.values[:n], bt[:n],
            tc_rate=tc_rate, garch_vol=gv, vol_target=vol_target)
        all_ec[mn] = ec
        fig_eq.add_trace(go.Scatter(
            y=ec["equity"], name=mn,
            line=dict(color=MODEL_COLORS.get(mn, "#aaa"), width=2)))
        em = full_equity_metrics(ec["equity"], ec["strat_returns"], ec["bench_returns"])
        perf_rows.append({
            "Model": mn,
            "Final ($)":       f"${em['Final ($)']:,.0f}",
            "Ann. Return (%)": em["Ann. Return (%)"],
            "Sharpe":          em["Sharpe"], "Sortino": em["Sortino"],
            "Calmar":          em["Calmar"], "Sterling": em["Sterling"],
            "Info Ratio":      round(information_ratio(ec["strat_returns"],
                                                        ec["bench_returns"]), 3),
            "Max DD (%)":      em["Max DD (%)"],
            "Max DD Duration": em["Max DD Duration"],
            "Mean DD (%)":     em["Mean Daily DD (%)"],
            "Median DD (%)":   em["Median Daily DD (%)"],
            "Ulcer Index":     em["Ulcer Index"],
            "Turnover":        f"{ec['turnover']*100:.1f}%",
            "TC Cost":         f"${ec['total_tc_cost']:.2f}",
        })

    fig_eq.update_layout(
        **PLOT_LAYOUT, height=360,
        title=dict(text="Equity Curve — $10,000 (Buy&Hold · Random Walk · All Models)",
                   font=dict(size=14)),
        yaxis_title="Portfolio Value ($)")
    st.plotly_chart(fig_eq, use_container_width=True)

    # Underwater — pure line chart
    fig_dd = go.Figure()
    for label, curve in [("Buy & Hold", bh_curve), ("Random Walk", rw_ec["equity"])]:
        pk = np.maximum.accumulate(curve)
        dd = (curve - pk) / (pk + 1e-12) * 100
        fig_dd.add_trace(go.Scatter(
            y=dd, name=label, mode="lines",
            line=dict(color=MODEL_COLORS[label], width=1.5,
                      dash="dot" if label == "Buy & Hold" else "dash")))

    bh_pk = np.maximum.accumulate(bh_curve)
    bh_dd = (bh_curve - bh_pk) / (bh_pk + 1e-12) * 100
    fig_dd.add_hline(y=float(np.mean(bh_dd)),
                     line=dict(color="#aaa", dash="dash", width=1),
                     annotation_text=f"BH Mean DD: {np.mean(bh_dd):.1f}%")
    fig_dd.add_hline(y=float(np.median(bh_dd)),
                     line=dict(color="#888", dash="dot", width=1),
                     annotation_text=f"BH Median DD: {np.median(bh_dd):.1f}%")

    for mn, ec in all_ec.items():
        eq = ec["equity"]
        pk = np.maximum.accumulate(eq)
        dd = (eq - pk) / (pk + 1e-12) * 100
        fig_dd.add_trace(go.Scatter(
            y=dd, name=mn, mode="lines",
            line=dict(color=MODEL_COLORS.get(mn, "#aaa"), width=2)))

    fig_dd.update_layout(
        **PLOT_LAYOUT, height=310,
        title=dict(text="Underwater Equity — Drawdown % (Line Chart)", font=dict(size=14)),
        yaxis_title="Drawdown (%)")
    st.plotly_chart(fig_dd, use_container_width=True)

    st.markdown("**📊 Strategy Performance Table — All Models · Buy & Hold · Random Walk**")
    st.dataframe(pd.DataFrame(perf_rows), use_container_width=True, hide_index=True)

st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
#  MONTE CARLO
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    f'<div class="sec-hdr">🎲 Monte Carlo Simulation — {mc_sims} Bootstrap Paths</div>',
    unsafe_allow_html=True)
st.caption("Bootstrap resamples historical log-returns to simulate thousands of future equity paths. "
           "Provides probability-weighted risk assessment independent of any model.")

with st.spinner(f"Running {mc_sims} Monte Carlo paths ({forecast_steps}d)..."):
    try:
        mc = monte_carlo_simulation(
            s_lr.dropna().values, n_simulations=mc_sims,
            horizon=forecast_steps, initial_capital=10_000)

        fig_mc = go.Figure()
        n_plot = min(150, mc_sims)
        for j in range(n_plot):
            fig_mc.add_trace(go.Scatter(
                y=mc["paths"][j], mode="lines",
                line=dict(color="rgba(0,212,255,0.04)" if DARK
                          else "rgba(0,85,204,0.04)", width=1),
                showlegend=False))

        pct = mc["percentiles"]
        pct_cfg = [
            (pct["p5"],  "#ff4466", "P5  (worst 5%)"),
            (pct["p25"], "#ff9500", "P25"),
            (pct["p50"], "#00d4ff", "P50 Median"),
            (pct["p75"], "#00e676", "P75"),
            (pct["p95"], "#9b4dff", "P95 (best 5%)"),
        ]
        for val, col, lbl in pct_cfg:
            fig_mc.add_hline(y=val,
                             line=dict(color=col, dash="dash", width=1.5),
                             annotation_text=f"{lbl}: ${val:,.0f}")

        fig_mc.add_hline(y=10_000,
                         line=dict(color="#888", dash="dot", width=1),
                         annotation_text="Initial $10,000")
        fig_mc.update_layout(
            **PLOT_LAYOUT, height=380,
            title=dict(text=f"Monte Carlo — {mc_sims} Bootstrap Paths · {forecast_steps}-Day Horizon",
                       font=dict(size=14)),
            yaxis_title="Portfolio Value ($)")
        st.plotly_chart(fig_mc, use_container_width=True)

        # MC stats row
        mc1, mc2, mc3, mc4, mc5, mc6 = st.columns(6)
        with mc1: st.metric("Median Final",  f"${mc['median_final']:,.0f}")
        with mc2: st.metric("Mean Final",    f"${mc['mean_final']:,.0f}")
        with mc3: st.metric("P5 (Worst 5%)", f"${pct['p5']:,.0f}")
        with mc4: st.metric("P95 (Best 5%)", f"${pct['p95']:,.0f}")
        with mc5: st.metric("Prob. of Loss", f"{mc['prob_loss']:.1f}%")
        with mc6: st.metric("95% VaR (Loss)",f"${mc['var_95']:,.0f}")

        # MC table
        mc_tbl = pd.DataFrame([{
            "Metric":    "Median Final Value",
            "Value":     f"${mc['median_final']:,.0f}",
            "Interpretation": "50% of simulated paths end above this value",
        },{
            "Metric":    "Mean Final Value",
            "Value":     f"${mc['mean_final']:,.0f}",
            "Interpretation": "Average across all simulated paths",
        },{
            "Metric":    "P5 (5th percentile)",
            "Value":     f"${pct['p5']:,.0f}",
            "Interpretation": "In 5% of worst scenarios, portfolio ends below this",
        },{
            "Metric":    "P95 (95th percentile)",
            "Value":     f"${pct['p95']:,.0f}",
            "Interpretation": "In 5% of best scenarios, portfolio exceeds this",
        },{
            "Metric":    "Probability of Loss",
            "Value":     f"{mc['prob_loss']:.1f}%",
            "Interpretation": "% of paths finishing below initial $10,000",
        },{
            "Metric":    "95% Value at Risk (VaR)",
            "Value":     f"${mc['var_95']:,.0f}",
            "Interpretation": "Loss level exceeded in worst 5% of scenarios",
        },{
            "Metric":    "95% CVaR (Expected Shortfall)",
            "Value":     f"${mc['cvar_95']:,.0f}",
            "Interpretation": "Average loss in the worst 5% of scenarios (stricter than VaR)",
        },{
            "Metric":    "Mean Max Drawdown (across sims)",
            "Value":     f"{mc['mean_max_dd']:.2f}%",
            "Interpretation": "Average worst peak-to-trough across all simulated paths",
        }])
        st.dataframe(mc_tbl, use_container_width=True, hide_index=True)

    except Exception as mce:
        st.warning(f"Monte Carlo failed: {mce}")

st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
#  MALAYSIA EDUCATION + GLOSSARY QUICK-REF
# ─────────────────────────────────────────────────────────────────────────────
with st.expander("📚 Malaysia Investor Education & Quick Glossary"):
    st.markdown("""
**Bacaan untuk Pelabur Malaysia / Malaysian Investor Reading Guide**

| Metric | Meaning (EN) | Interpretasi (BM) |
|--------|-------------|-------------------|
| **Log-Returns** | log(Pₜ/Pₜ₋₁) — stationary, additive | Pulangan log — pegun, boleh tambah |
| **ADF reject** | Series is stationary (no unit root) | Siri adalah pegun |
| **KPSS reject** | Series is NOT stationary | Siri tidak pegun |
| **Conflict** | Trend-stationary → detrend first | Pegun-tren → nyahtrend dahulu |
| **Hit Ratio >55%** | Directional edge on log-returns | Kelebihan arah >55% bermakna |
| **Sortino > Sharpe** | Only penalises downside | Hanya menghukum risiko negatif |
| **Calmar > 0.5** | Good risk-return vs max loss | Pulangan berbanding kehilangan terburuk |
| **DM p<0.05** | Statistically significant model difference | Perbezaan model signifikan secara statistik |
| **CVaR** | Average loss in worst 5% scenarios | Kerugian purata dalam 5% senario terburuk |
| **PCA** | Reduces many correlated features to fewer components | Mengurangkan ciri-ciri berkorelasi |

**Malaysian Market Notes:**
- **USDMYR** driven by BNM OPR decisions + crude oil/LNG prices + US Fed rates
- **KLCI** strongly correlated with Hang Seng HSI — monitor China macro as leading indicator
- **CPO (Palm Oil)** is a key Bursa driver — affects Felda, IOI, KL Kepong, Sime Darby
- **For Bursa stocks**: price-only models miss dividend yield (Maybank 6%+, Public Bank 4%+) — factor separately
- **Crypto** in Malaysia regulated by SC under the Capital Markets and Services Act
    """)

st.markdown("---")
st.markdown(f"""
<div class="footer">
  ⚠️ <b>Disclaimer:</b> Educational & research portfolio only. Not financial advice.
  Past performance does not guarantee future results. Bukan nasihat kewangan.<br><br>
  <b>Sammy Oliver Areh</b> · MSc Statistics · Time Series Analysis Research ·
  <a href="https://github.com/SamOliverAreh">GitHub</a> ·
  <a href="https://www.linkedin.com/in/sam-oliver-areh/">LinkedIn</a> ·
  <a href="https://sammy-quant-lab.streamlit.app">Live Dashboard</a> ·
  <a href="https://samoliverareh.github.io/sammy-quant-lab/">Portfolio Page</a>
</div>""", unsafe_allow_html=True)
