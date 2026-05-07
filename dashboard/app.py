"""
Sammy Quant Lab — Dashboard v3.0
Mobile-first · Multi-asset · 10 models + hybrids + stacked ensemble
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

from data_pipeline.ingestion   import fetch_data, get_categories, get_category
from data_pipeline.preprocessing import preprocess
from data_pipeline.features    import create_features
from models.statistical.arima  import arima_forecast
from models.statistical.garch  import garch_volatility, garch_summary
from models.statistical.ets    import ets_forecast
from models.ml.lstm            import train_lstm, forecast_lstm
from models.ml.gru             import train_gru, forecast_gru
from models.ml.transformer_model import train_transformer, forecast_transformer
from models.ml.xgboost_model   import train_xgboost, forecast_xgboost, XGB_AVAILABLE
from models.ml.random_forest   import train_rf, forecast_rf
from models.hybrid.arima_lstm  import hybrid_forecast
from models.hybrid.ets_gru     import ets_gru_forecast
from models.ensemble.stacked   import stacked_ensemble_forecast
from evaluation.metrics        import evaluate_all, sharpe_ratio, max_drawdown

# ── Optional imports ──────────────────────────────────────────────────────────
try:
    from models.statistical.prophet_model import prophet_forecast, PROPHET_AVAILABLE
except Exception:
    PROPHET_AVAILABLE = False

try:
    from models.hybrid.prophet_xgb import prophet_xgb_forecast
    PROPHET_XGB_AVAILABLE = PROPHET_AVAILABLE and XGB_AVAILABLE
except Exception:
    PROPHET_XGB_AVAILABLE = False

# ─── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sammy Quant Lab",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",   # collapsed by default → hamburger on mobile
)

# ─── CSS — mobile-first ───────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Outfit:wght@300;400;600;700;800&display=swap');

  html, body, [class*="css"] { font-family: 'Outfit', sans-serif; }
  .main  { background: #060a14; }
  .stApp { background: #060a14; }

  /* ── Sidebar ── */
  section[data-testid="stSidebar"] {
    background: #090d1c;
    border-right: 1px solid #162035;
  }
  /* hamburger button visible on all sizes */
  button[kind="header"] { display: flex !important; }

  /* ── Typography ── */
  h1,h2,h3 { font-family: 'Outfit', sans-serif; }

  .ql-title {
    font-size: clamp(1.4rem, 5vw, 2.4rem);
    font-weight: 800;
    background: linear-gradient(135deg, #00d4ff 0%, #0070f3 55%, #9b4dff 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.15;
  }
  .ql-sub {
    color: #4a6a8a;
    font-size: clamp(0.75rem, 2.5vw, 0.9rem);
    margin-top: 4px;
  }

  /* ── Metric cards ── */
  .metric-card {
    background: linear-gradient(135deg, #0b1220 0%, #0d1830 100%);
    border: 1px solid #162035;
    border-radius: 14px;
    padding: clamp(12px, 3vw, 20px);
    text-align: center;
    min-height: 90px;
  }
  .metric-val {
    font-family: 'DM Mono', monospace;
    font-size: clamp(1.1rem, 3.5vw, 1.7rem);
    font-weight: 500;
    color: #00d4ff;
  }
  .metric-lbl {
    font-size: clamp(0.65rem, 2vw, 0.75rem);
    color: #4a6a8a;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 4px;
  }
  .metric-delta { font-size: 0.82rem; margin-top: 5px; }
  .pos { color: #00e676; } .neg { color: #ff4466; }

  /* ── Section header ── */
  .sec-hdr {
    font-size: clamp(0.85rem, 2.5vw, 1rem);
    font-weight: 600;
    color: #c8d8e8;
    border-left: 3px solid #00d4ff;
    padding-left: 10px;
    margin: 22px 0 12px 0;
  }

  /* ── Badges ── */
  .badge {
    display: inline-block; padding: 3px 11px;
    border-radius: 20px; font-size: 0.72rem;
    font-family: 'DM Mono', monospace;
  }
  .badge-live  { background:#001a0f; color:#00e676; border:1px solid #00e676; }
  .badge-model { background:#001526; color:#00d4ff; border:1px solid #00d4ff; }
  .badge-cat   { background:#1a0d30; color:#9b4dff; border:1px solid #9b4dff; }

  /* ── Buttons ── */
  .stButton > button {
    background: linear-gradient(135deg, #0055cc, #0080ff);
    color: white; border: none; border-radius: 10px;
    padding: 10px 24px;
    font-family: 'Outfit', sans-serif; font-weight: 700;
    font-size: clamp(0.85rem, 2.5vw, 0.95rem);
    width: 100%; transition: opacity .2s, transform .2s;
  }
  .stButton > button:hover { opacity:.85; transform:translateY(-1px); }

  /* ── Streamlit widgets ── */
  .stSelectbox label, .stSlider label, .stCheckbox label, .stRadio label {
    color: #7a9abb !important; font-size: 0.83rem;
  }
  div[data-testid="stMetric"] {
    background: #0b1220; border: 1px solid #162035; border-radius: 10px; padding: 14px;
  }
  div[data-testid="stMetric"] label { color: #4a6a8a !important; }
  div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
    color: #00d4ff !important; font-family: 'DM Mono', monospace;
  }

  /* ── Table ── */
  .stDataFrame { border-radius: 10px; overflow: hidden; }

  /* ── Mobile: hide plotly toolbar to save space ── */
  @media (max-width: 768px) {
    .modebar { display: none !important; }
    .main .block-container { padding: 0.75rem 0.75rem 4rem !important; }
  }

  /* ── Info/warning boxes ── */
  div[data-testid="stInfo"]    { background: #001830; border: 1px solid #0040a0; border-radius: 10px; }
  div[data-testid="stWarning"] { background: #1a0c00; border: 1px solid #a06000; border-radius: 10px; }

  hr { border-color: #162035; }

  /* ── Footer ── */
  .footer {
    text-align: center; color: #2a4060; font-size: 0.72rem;
    padding: 24px 0 8px;
  }
  .footer a { color: #0070f3; text-decoration: none; }
</style>
""", unsafe_allow_html=True)

# ─── Plot defaults ────────────────────────────────────────────────────────────
PLOT_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(9,13,28,0.9)",
    font=dict(family="Outfit, sans-serif", color="#7a9abb", size=11),
    xaxis=dict(gridcolor="#0e1e35", showgrid=True, zeroline=False),
    yaxis=dict(gridcolor="#0e1e35", showgrid=True, zeroline=False),
    margin=dict(l=10, r=10, t=36, b=10),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#162035", borderwidth=1, font=dict(size=10)),
)

# ─── Model registry ───────────────────────────────────────────────────────────
MODEL_INFO = {
    # Statistical
    "ARIMA":             {"group": "Statistical", "color": "#00d4ff",  "desc": "Auto ARIMA (AIC order selection)"},
    "ETS":               {"group": "Statistical", "color": "#29b6f6",  "desc": "Exponential Smoothing (Holt damped trend)"},
    "Prophet":           {"group": "Statistical", "color": "#4fc3f7",  "desc": "Meta Prophet (trend + seasonality)", "needs": "prophet"},
    # ML
    "LSTM":              {"group": "Deep Learning","color": "#9b4dff",  "desc": "Stacked LSTM (PyTorch)"},
    "GRU":               {"group": "Deep Learning","color": "#b06fff",  "desc": "Gated Recurrent Unit (PyTorch)"},
    "Transformer":       {"group": "Deep Learning","color": "#ce93d8",  "desc": "Encoder-only Transformer (PyTorch)"},
    "XGBoost":           {"group": "Tree Models", "color": "#ff9500",  "desc": "XGBoost (recursive lag features)", "needs": "xgb"},
    "Random Forest":     {"group": "Tree Models", "color": "#ffc107",  "desc": "Random Forest (recursive lag features)"},
    # Hybrids
    "Hybrid ARIMA+LSTM": {"group": "Hybrid",      "color": "#ff4081",  "desc": "ARIMA linear + LSTM residual"},
    "Hybrid ETS+GRU":    {"group": "Hybrid",      "color": "#f48fb1",  "desc": "ETS baseline + GRU residual"},
    "Hybrid Prophet+XGB":{"group": "Hybrid",      "color": "#ff6e40",  "desc": "Prophet trend + XGBoost residual", "needs": "prophet_xgb"},
    # Ensemble
    "Stacked Ensemble":  {"group": "Ensemble",    "color": "#00e676",  "desc": "Ridge meta-learner on 6 base models"},
}

def model_available(name):
    needs = MODEL_INFO[name].get("needs", "")
    if needs == "prophet": return PROPHET_AVAILABLE
    if needs == "xgb":     return XGB_AVAILABLE
    if needs == "prophet_xgb": return PROPHET_XGB_AVAILABLE
    return True

AVAILABLE_MODELS = [m for m in MODEL_INFO if model_available(m)]
MODEL_GROUPS     = {}
for m in AVAILABLE_MODELS:
    g = MODEL_INFO[m]["group"]
    MODEL_GROUPS.setdefault(g, []).append(m)

# ─── Asset categories ─────────────────────────────────────────────────────────
CATEGORIES = get_categories()

# ─────────────────────────────────────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="ql-title">⚡ Sammy Quant</div>', unsafe_allow_html=True)
    st.markdown('<div class="ql-sub">Multi-Asset Forecasting Lab v3.0</div>', unsafe_allow_html=True)
    st.markdown("---")

    # ── Asset selection ──────────────────────────────────────────────────────
    st.markdown('<div class="sec-hdr">📊 Asset Universe</div>', unsafe_allow_html=True)
    cat_choice  = st.selectbox("Asset Class", list(CATEGORIES.keys()))
    asset_names = CATEGORIES[cat_choice]
    ticker      = st.selectbox("Instrument", asset_names)

    st.markdown('<div class="sec-hdr">📅 Data Range</div>', unsafe_allow_html=True)
    start_date     = st.date_input("Start Date", value=pd.Timestamp("2021-01-01"))
    forecast_steps = st.slider("Forecast Horizon (days)", 5, 90, 30)

    # ── Model selection ──────────────────────────────────────────────────────
    st.markdown('<div class="sec-hdr">🤖 Model</div>', unsafe_allow_html=True)
    group_choice = st.radio("Model Group", list(MODEL_GROUPS.keys()), horizontal=False)
    model_choice = st.selectbox("Select Model", MODEL_GROUPS[group_choice])

    st.markdown(f"""
    <div style="background:#050f1a;border:1px solid #162035;border-radius:8px;padding:10px 12px;font-size:0.8rem;color:#4a6a8a;margin-top:4px;">
    <b style="color:#00d4ff;">{model_choice}</b><br>
    {MODEL_INFO[model_choice]['desc']}
    </div>
    """, unsafe_allow_html=True)

    show_garch    = st.checkbox("GARCH Volatility Overlay", value=True)
    show_features = st.checkbox("Technical Indicators", value=True)

    # ── Deep-learning hyperparams (conditionally shown) ──────────────────────
    group_name = MODEL_INFO[model_choice]["group"]
    if group_name in ("Deep Learning", "Hybrid") or model_choice == "Stacked Ensemble":
        st.markdown('<div class="sec-hdr">⚙️ DL Hyperparams</div>', unsafe_allow_html=True)
        dl_epochs = st.slider("Epochs", 10, 100, 30)
        dl_hidden = st.selectbox("Hidden Units", [32, 64, 128], index=1)
        dl_window = st.slider("Look-back Window", 10, 60, 20)
    else:
        dl_epochs, dl_hidden, dl_window = 30, 64, 20

    st.markdown("---")
    run_btn = st.button("🚀 Run Analysis", use_container_width=True)

    st.markdown("""
    <div style="text-align:center;margin-top:16px;font-size:0.73rem;color:#2a4060;">
      Built by <b style="color:#00d4ff;">Sammy</b><br>
      Data Scientist · ML Engineer<br>
      <a href="https://github.com/SamOliverAreh" style="color:#0070f3;">GitHub</a> ·
      <a href="https://www.linkedin.com/in/sam-oliver-areh/" style="color:#0070f3;">LinkedIn</a><br><br>
      <span class="badge badge-live">● LIVE DATA</span>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  MAIN — header
# ─────────────────────────────────────────────────────────────────────────────
col_t, col_b = st.columns([3, 1])
with col_t:
    cat = get_category(ticker)
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;">
      <div class="ql-title">⚡ Sammy Quant Lab</div>
      <span class="badge badge-model">{model_choice}</span>
      <span class="badge badge-cat">{cat}</span>
    </div>
    <div class="ql-sub">Multi-Asset Forecasting · {ticker} · {cat} · Real-time Data</div>
    """, unsafe_allow_html=True)
with col_b:
    st.markdown('<div style="padding-top:10px;"></div>', unsafe_allow_html=True)
    st.markdown(f'<div style="text-align:right;"><span class="badge badge-live">● LIVE</span></div>',
                unsafe_allow_html=True)

st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
#  LANDING STATE (no run yet)
# ─────────────────────────────────────────────────────────────────────────────
if not run_btn:
    st.info("👈 Open the **sidebar** (☰ top-left) to configure, then tap **Run Analysis**.")

    # Asset universe overview
    st.markdown('<div class="sec-hdr">🌐 Asset Universe</div>', unsafe_allow_html=True)
    tabs = st.tabs(list(CATEGORIES.keys()))
    for tab, (cat_name, names) in zip(tabs, CATEGORIES.items()):
        with tab:
            st.dataframe(
                pd.DataFrame({"Instrument": names, "Asset Class": [cat_name] * len(names)}),
                use_container_width=True, hide_index=True
            )

    # Model overview
    st.markdown('<div class="sec-hdr">🤖 Model Catalogue</div>', unsafe_allow_html=True)
    rows = [{"Model": m, "Group": MODEL_INFO[m]["group"], "Description": MODEL_INFO[m]["desc"],
             "Available": "✅" if model_available(m) else "⚠️ pkg missing"}
            for m in MODEL_INFO]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.markdown("""
    <div class="footer">
      ⚠️ For educational & portfolio purposes only. Not financial advice.<br>
      Built by <b>Sammy</b> · <a href="https://sammy-quant-lab.streamlit.app">Live Demo</a> ·
      <a href="https://samoliverareh.github.io/sammy-quant-lab/">Portfolio</a>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
#  DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────
with st.spinner(f"📡 Fetching {ticker} data..."):
    try:
        df_raw  = fetch_data(ticker, start=str(start_date))
        df      = preprocess(df_raw.copy())
        df_feat = create_features(df.copy())
        series  = df["Close"]
        data_ok = True
    except Exception as e:
        st.error(f"Data fetch failed: {e}")
        st.stop()

if len(series) < 60:
    st.warning("⚠️ Less than 60 data points available — results may be unreliable.")

# ─────────────────────────────────────────────────────────────────────────────
#  KPI ROW
# ─────────────────────────────────────────────────────────────────────────────
cp    = float(series.iloc[-1])
pp    = float(series.iloc[-2]) if len(series) > 1 else cp
pchg  = (cp - pp) / (pp + 1e-12) * 100
vol30 = float(df["returns"].rolling(30).std().iloc[-1] * 100)
sr    = sharpe_ratio(df["returns"].dropna())
mdd   = max_drawdown(series.values) * 100

def kpi(val, lbl, delta=None):
    delta_html = ""
    if delta is not None:
        cls = "pos" if delta >= 0 else "neg"
        delta_html = f'<div class="metric-delta {cls}">{delta:+.3f}%</div>'
    return f"""
    <div class="metric-card">
      <div class="metric-val">{val}</div>
      <div class="metric-lbl">{lbl}</div>
      {delta_html}
    </div>"""

c1, c2, c3, c4, c5 = st.columns(5)
with c1: st.markdown(kpi(f"{cp:.4f}", "Price", pchg), unsafe_allow_html=True)
with c2: st.markdown(kpi(f"{vol30:.2f}%", "30d Volatility"), unsafe_allow_html=True)
with c3: st.markdown(kpi(f"{sr:.2f}", "Sharpe Ratio"), unsafe_allow_html=True)
with c4: st.markdown(kpi(f"{mdd:.2f}%", "Max Drawdown"), unsafe_allow_html=True)
with c5: st.markdown(kpi(f"{len(series):,}", "Data Points"), unsafe_allow_html=True)

st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
#  PRICE CHART
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="sec-hdr">📈 Price History & Technical Indicators</div>', unsafe_allow_html=True)
tab_price, tab_ret, tab_corr = st.tabs(["Price Chart", "Returns Distribution", "Correlation"])

with tab_price:
    has_vol = "Volume" in df_raw.columns and df_raw["Volume"].sum() > 0
    rows    = 2 if has_vol else 1
    fig     = make_subplots(rows=rows, cols=1, shared_xaxes=True,
                            row_heights=[0.72, 0.28][:rows], vertical_spacing=0.05)

    fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Close",
                             line=dict(color="#00d4ff", width=1.5)), row=1, col=1)

    if show_features and "ma_21" in df_feat.columns:
        fig.add_trace(go.Scatter(x=df_feat.index, y=df_feat["ma_21"], name="MA 21",
                                 line=dict(color="#ff9500", width=1, dash="dot")), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_feat.index, y=df_feat["ma_50"], name="MA 50",
                                 line=dict(color="#9b4dff", width=1, dash="dot")), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_feat.index, y=df_feat["bb_upper"], name="BB Upper",
                                 line=dict(color="#2a3d55", width=1), showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_feat.index, y=df_feat["bb_lower"], name="BB Lower",
                                 line=dict(color="#2a3d55", width=1),
                                 fill="tonexty", fillcolor="rgba(0,100,200,0.06)"), row=1, col=1)

    if has_vol and rows == 2:
        fig.add_trace(go.Bar(x=df_raw.index, y=df_raw["Volume"], name="Volume",
                             marker_color="rgba(0,212,255,0.18)"), row=2, col=1)

    fig.update_layout(**PLOT_BASE, height=400, title=dict(text=f"{ticker} — Price History", font=dict(size=14)))
    st.plotly_chart(fig, use_container_width=True)

with tab_ret:
    fig2 = go.Figure()
    fig2.add_trace(go.Histogram(x=df["returns"].dropna() * 100, nbinsx=60,
                                marker_color="#00d4ff", opacity=0.7, name="Returns"))
    fig2.update_layout(**PLOT_BASE, height=320,
                       xaxis_title="Daily Return (%)", yaxis_title="Frequency",
                       title=dict(text="Returns Distribution", font=dict(size=14)))
    st.plotly_chart(fig2, use_container_width=True)

with tab_corr:
    feat_cols = [c for c in ["Close","ma_21","ma_50","rsi","macd","bb_width","volatility_30","atr"]
                 if c in df_feat.columns]
    corr_df   = df_feat[feat_cols].corr()
    fig3      = go.Figure(go.Heatmap(
        z=corr_df.values, x=corr_df.columns, y=corr_df.columns,
        colorscale="RdBu", zmid=0, text=np.round(corr_df.values, 2),
        texttemplate="%{text}", showscale=True,
    ))
    fig3.update_layout(**PLOT_BASE, height=360, title=dict(text="Feature Correlation Matrix", font=dict(size=14)))
    st.plotly_chart(fig3, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
#  GARCH VOLATILITY
# ─────────────────────────────────────────────────────────────────────────────
if show_garch:
    st.markdown('<div class="sec-hdr">📉 GARCH(1,1) Volatility Forecast</div>', unsafe_allow_html=True)
    with st.spinner("Fitting GARCH..."):
        try:
            gvol = garch_volatility(df["returns"].dropna(), horizon=forecast_steps)
            gsum = garch_summary(df["returns"].dropna())
            fig_g = go.Figure()
            fig_g.add_trace(go.Scatter(
                y=gvol * 100, mode="lines+markers",
                line=dict(color="#ff9500", width=2), marker=dict(size=4),
                name="GARCH Vol Forecast"
            ))
            fig_g.update_layout(**PLOT_BASE, height=280,
                                xaxis_title=f"Days ahead (t+1 to t+{forecast_steps})",
                                yaxis_title="Volatility (%)",
                                title=dict(text="Forward Volatility Forecast", font=dict(size=14)))
            st.plotly_chart(fig_g, use_container_width=True)
            if isinstance(gsum, dict) and "aic" in gsum:
                g1, g2, g3 = st.columns(3)
                with g1: st.metric("AIC", f"{gsum['aic']:.1f}")
                with g2: st.metric("BIC", f"{gsum['bic']:.1f}")
                with g3: st.metric("Log-Likelihood", f"{gsum['loglikelihood']:.1f}")
        except Exception as ge:
            st.warning(f"GARCH failed: {ge}")

st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
#  MODEL EXECUTION
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f'<div class="sec-hdr">🤖 {model_choice} — Forecast</div>', unsafe_allow_html=True)

pred        = None
model_meta  = {}
conf_lower  = None
conf_upper  = None

with st.spinner(f"Running {model_choice}..."):
    try:
        # ── Statistical ──────────────────────────────────────────────────────
        if model_choice == "ARIMA":
            pred, order, ci = arima_forecast(series, steps=forecast_steps)
            conf_lower = ci.iloc[:, 0].values if hasattr(ci, 'iloc') else None
            conf_upper = ci.iloc[:, 1].values if hasattr(ci, 'iloc') else None
            model_meta = {"Order": str(order)}

        elif model_choice == "ETS":
            pred, params = ets_forecast(series, steps=forecast_steps)
            model_meta = params

        elif model_choice == "Prophet":
            pred, conf_lower, conf_upper, fc_df = prophet_forecast(series, steps=forecast_steps)
            model_meta = {"Uncertainty": "95% interval"}

        # ── Deep Learning ────────────────────────────────────────────────────
        elif model_choice == "LSTM":
            m, sc, losses = train_lstm(series, epochs=dl_epochs, hidden=dl_hidden, window=dl_window)
            pred = forecast_lstm(m, series, sc, steps=forecast_steps, window=dl_window)
            model_meta = {"Final Loss": f"{losses[-1]:.6f}", "Epochs": dl_epochs, "Hidden": dl_hidden}

        elif model_choice == "GRU":
            m, sc, losses = train_gru(series, epochs=dl_epochs, hidden=dl_hidden, window=dl_window)
            pred = forecast_gru(m, series, sc, steps=forecast_steps, window=dl_window)
            model_meta = {"Final Loss": f"{losses[-1]:.6f}", "Epochs": dl_epochs, "Hidden": dl_hidden}

        elif model_choice == "Transformer":
            m, sc, losses = train_transformer(series, epochs=dl_epochs, window=dl_window)
            pred = forecast_transformer(m, series, sc, steps=forecast_steps, window=dl_window)
            model_meta = {"Final Loss": f"{losses[-1]:.6f}", "Epochs": dl_epochs}

        # ── Tree Models ──────────────────────────────────────────────────────
        elif model_choice == "XGBoost":
            m, lags = train_xgboost(series)
            pred = forecast_xgboost(m, series, steps=forecast_steps, lags=lags)
            model_meta = {"Lags": lags, "n_estimators": 300}

        elif model_choice == "Random Forest":
            m, lags = train_rf(series)
            pred = forecast_rf(m, series, steps=forecast_steps, lags=lags)
            model_meta = {"Lags": lags, "n_estimators": 200}

        # ── Hybrids ──────────────────────────────────────────────────────────
        elif model_choice == "Hybrid ARIMA+LSTM":
            pred, arima_p, lstm_p, order = hybrid_forecast(series, steps=forecast_steps)
            model_meta = {"ARIMA Order": str(order), "Components": "ARIMA + LSTM residuals"}

        elif model_choice == "Hybrid ETS+GRU":
            pred, ets_p, gru_p = ets_gru_forecast(series, steps=forecast_steps)
            model_meta = {"Components": "ETS baseline + GRU residuals"}

        elif model_choice == "Hybrid Prophet+XGB":
            pred, prophet_p, xgb_p, conf_lower, conf_upper = prophet_xgb_forecast(series, steps=forecast_steps)
            model_meta = {"Components": "Prophet trend + XGBoost residuals"}

        # ── Ensemble ─────────────────────────────────────────────────────────
        elif model_choice == "Stacked Ensemble":
            with st.spinner("Training 6 base models + meta-learner (may take 1-3 min)..."):
                pred, base_preds = stacked_ensemble_forecast(series, steps=forecast_steps)
            model_meta = {"Meta-learner": "Ridge Regression", "Base models": "ARIMA, ETS, LSTM, GRU, XGB, RF"}

    except Exception as ex:
        st.error(f"Model error: {ex}")
        import traceback
        st.code(traceback.format_exc())
        st.stop()

if pred is None:
    st.error("No forecast produced.")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
#  FORECAST CHART
# ─────────────────────────────────────────────────────────────────────────────
model_color = MODEL_INFO[model_choice]["color"]
hist_show   = min(120, len(series))
hist_idx    = series.index[-hist_show:]
hist_vals   = series.values[-hist_show:]

forecast_dates = pd.bdate_range(
    start=series.index[-1] + pd.Timedelta(days=1), periods=forecast_steps
)
# For crypto/commodities (7-day markets), use calendar days if needed
if cat_choice == "Crypto":
    forecast_dates = pd.date_range(
        start=series.index[-1] + pd.Timedelta(days=1), periods=forecast_steps
    )

fig_fc = go.Figure()

# Historical
fig_fc.add_trace(go.Scatter(
    x=hist_idx, y=hist_vals,
    name="Historical", line=dict(color="#00d4ff", width=1.5)
))

# Confidence band
if conf_lower is not None and conf_upper is not None:
    n = min(len(forecast_dates), len(conf_lower), len(conf_upper))
    fig_fc.add_trace(go.Scatter(
        x=list(forecast_dates[:n]) + list(forecast_dates[:n])[::-1],
        y=list(conf_upper[:n]) + list(conf_lower[:n])[::-1],
        fill="toself", fillcolor=f"rgba{tuple(list(bytes.fromhex(model_color.lstrip('#'))) + [30])}"
              if model_color.startswith("#") else "rgba(0,200,100,0.08)",
        line=dict(width=0), showlegend=True, name="95% CI"
    ))

# Ensemble base models
if model_choice == "Stacked Ensemble":
    for bm_name, bm_pred in base_preds.items():
        n = min(len(forecast_dates), len(bm_pred))
        fig_fc.add_trace(go.Scatter(
            x=forecast_dates[:n], y=bm_pred[:n],
            name=bm_name, line=dict(width=1, dash="dot"), opacity=0.5
        ))

# Forecast line
n = min(len(forecast_dates), len(pred))
fig_fc.add_trace(go.Scatter(
    x=forecast_dates[:n], y=pred[:n],
    name=model_choice, line=dict(color=model_color, width=2.5)
))

# Connector
fig_fc.add_trace(go.Scatter(
    x=[series.index[-1], forecast_dates[0]],
    y=[float(series.iloc[-1]), float(pred[0])],
    line=dict(color=model_color, width=1.5, dash="dot"),
    showlegend=False
))

fig_fc.update_layout(**PLOT_BASE, height=420,
                     title=dict(text=f"{ticker} — {forecast_steps}-Day Forecast ({model_choice})",
                                font=dict(size=14)))
st.plotly_chart(fig_fc, use_container_width=True)

# ─── Model meta info ─────────────────────────────────────────────────────────
if model_meta:
    meta_cols = st.columns(min(len(model_meta), 4))
    for col, (k, v) in zip(meta_cols, model_meta.items()):
        with col:
            st.metric(k, str(v))

# ─────────────────────────────────────────────────────────────────────────────
#  BACKTESTING
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="sec-hdr">📐 Walk-Forward Backtest</div>', unsafe_allow_html=True)

with st.spinner("Backtesting..."):
    try:
        test_window = min(60, len(series) - forecast_steps - 10)
        if test_window < 20:
            st.warning("Not enough data for backtesting.")
        else:
            train_bt = series.iloc[-test_window - forecast_steps: -forecast_steps]
            true_bt  = series.iloc[-forecast_steps:]

            bt_pred = None
            if model_choice == "ARIMA":
                bt_pred, _, _ = arima_forecast(train_bt, steps=forecast_steps)
            elif model_choice == "ETS":
                bt_pred, _ = ets_forecast(train_bt, steps=forecast_steps)
            elif model_choice == "LSTM":
                bm, bs, _ = train_lstm(train_bt, epochs=dl_epochs, hidden=dl_hidden, window=dl_window)
                bt_pred = forecast_lstm(bm, train_bt, bs, steps=forecast_steps, window=dl_window)
            elif model_choice == "GRU":
                bm, bs, _ = train_gru(train_bt, epochs=dl_epochs, hidden=dl_hidden, window=dl_window)
                bt_pred = forecast_gru(bm, train_bt, bs, steps=forecast_steps, window=dl_window)
            elif model_choice == "Transformer":
                bm, bs, _ = train_transformer(train_bt, epochs=dl_epochs, window=dl_window)
                bt_pred = forecast_transformer(bm, train_bt, bs, steps=forecast_steps, window=dl_window)
            elif model_choice == "XGBoost":
                bm, lags = train_xgboost(train_bt)
                bt_pred  = forecast_xgboost(bm, train_bt, steps=forecast_steps, lags=lags)
            elif model_choice == "Random Forest":
                bm, lags = train_rf(train_bt)
                bt_pred  = forecast_rf(bm, train_bt, steps=forecast_steps, lags=lags)
            elif model_choice == "Hybrid ARIMA+LSTM":
                bt_pred, _, _, _ = hybrid_forecast(train_bt, steps=forecast_steps)
            elif model_choice == "Hybrid ETS+GRU":
                bt_pred, _, _ = ets_gru_forecast(train_bt, steps=forecast_steps)
            elif model_choice in ("Prophet", "Hybrid Prophet+XGB"):
                bt_pred = pred  # skip re-running prophet for speed
            elif model_choice == "Stacked Ensemble":
                bt_pred, _ = stacked_ensemble_forecast(train_bt, steps=forecast_steps)

            if bt_pred is not None:
                trim    = min(len(true_bt), len(bt_pred))
                metrics = evaluate_all(true_bt.values[:trim], bt_pred[:trim])

                m1, m2, m3, m4 = st.columns(4)
                with m1: st.metric("RMSE", f"{metrics['RMSE']:.4f}")
                with m2: st.metric("MAE",  f"{metrics['MAE']:.4f}")
                with m3: st.metric("MAPE", f"{metrics['MAPE']:.2f}%")
                with m4: st.metric("Direction Acc.", f"{metrics['Direction Accuracy (%)']:.1f}%")

                # Backtest chart
                fig_bt = go.Figure()
                fig_bt.add_trace(go.Scatter(y=true_bt.values[:trim], name="Actual",
                                            line=dict(color="#00d4ff", width=2)))
                fig_bt.add_trace(go.Scatter(y=bt_pred[:trim], name="Backtest Pred.",
                                            line=dict(color=model_color, width=2, dash="dash")))
                fig_bt.update_layout(**PLOT_BASE, height=300,
                                     title=dict(text="Backtest: Actual vs Predicted", font=dict(size=14)))
                st.plotly_chart(fig_bt, use_container_width=True)

    except Exception as be:
        st.warning(f"Backtest error: {be}")

# ─────────────────────────────────────────────────────────────────────────────
#  FORECAST TABLE
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="sec-hdr">🔮 Forecast Table</div>', unsafe_allow_html=True)

n      = min(len(forecast_dates), len(pred))
fc_tbl = pd.DataFrame({
    "Date":       [d.date() for d in forecast_dates[:n]],
    "Forecast":   [f"{p:.4f}" for p in pred[:n]],
    "Chg from Now": [f"{(p - cp) / cp * 100:+.2f}%" for p in pred[:n]],
})
if conf_lower is not None and conf_upper is not None:
    nl = min(n, len(conf_lower), len(conf_upper))
    fc_tbl["Lower 95%"] = [f"{v:.4f}" for v in conf_lower[:nl]] + ["—"] * (n - nl)
    fc_tbl["Upper 95%"] = [f"{v:.4f}" for v in conf_upper[:nl]] + ["—"] * (n - nl)

st.dataframe(fc_tbl, use_container_width=True, hide_index=True)

# ─────────────────────────────────────────────────────────────────────────────
#  RISK DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="sec-hdr">⚠️ Risk Dashboard</div>', unsafe_allow_html=True)

r1, r2 = st.columns(2)
with r1:
    # Sharpe/MDD gauge-style
    fig_risk = go.Figure()
    fig_risk.add_trace(go.Scatter(
        x=df["returns"].dropna().index, y=df["returns"].dropna().rolling(21).std() * 100 * np.sqrt(252),
        name="Annualised Vol (21d)", line=dict(color="#ff9500", width=1.5)
    ))
    fig_risk.update_layout(**PLOT_BASE, height=260,
                           yaxis_title="Ann. Vol (%)",
                           title=dict(text="Rolling Annualised Volatility", font=dict(size=13)))
    st.plotly_chart(fig_risk, use_container_width=True)

with r2:
    # Drawdown
    prices_arr = series.values
    peak_arr   = np.maximum.accumulate(prices_arr)
    dd_arr     = (prices_arr - peak_arr) / (peak_arr + 1e-12) * 100
    fig_dd = go.Figure()
    fig_dd.add_trace(go.Scatter(
        x=series.index, y=dd_arr, fill="tozeroy",
        fillcolor="rgba(255,68,102,0.15)", line=dict(color="#ff4466", width=1),
        name="Drawdown"
    ))
    fig_dd.update_layout(**PLOT_BASE, height=260,
                         yaxis_title="Drawdown (%)",
                         title=dict(text="Underwater Equity Curve", font=dict(size=13)))
    st.plotly_chart(fig_dd, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
#  MALAYSIA FINANCIAL EDUCATION PANEL
# ─────────────────────────────────────────────────────────────────────────────
with st.expander("📚 Malaysia Investor Education — Understanding the Model Output"):
    st.markdown(f"""
**Bacaan untuk Pelabur Malaysia / Reading for Malaysian Investors**

| Metric | Interpretation (EN) | Interpretasi (BM) |
|--------|---------------------|-------------------|
| **RMSE / MAE** | Average forecast error in price units | Ralat ramalan purata dalam unit harga |
| **MAPE** | % error — below 2% is good for FX; below 5% for equities | Ralat %; <2% baik untuk FX, <5% untuk ekuiti |
| **Direction Accuracy** | % of correct up/down calls — 50% = random | Ketepatan arah; 50% = rawak, >55% = bermakna |
| **Sharpe Ratio** | Risk-adjusted return; >1.0 is acceptable, >2.0 is strong | Pulangan terlaras risiko |
| **Max Drawdown** | Worst peak-to-trough loss in history | Kerugian terburuk dari puncak ke lembah |

**Key reminders / Peringatan penting:**
- ML models learn *patterns*, not *fundamentals* — macro events (BNM decisions, US Fed, oil shocks) are **not captured**.
- Models trained on past data do not guarantee future performance — **jangan bergantung sepenuhnya pada ramalan ini**.
- USDMYR is heavily influenced by oil prices, current account balance, and Bank Negara Malaysia policy.
- For Bursa stocks (KL-listed), factor in **dividend yield** and **PE ratio** which price-only models ignore.
- KLCI correlates strongly with China (Hang Seng) and global risk appetite — monitor these leading indicators.

**Sumber maklumat / Information sources:**
- [Bank Negara Malaysia](https://www.bnm.gov.my) — official monetary policy
- [Bursa Malaysia](https://www.bursamalaysia.com) — equity market data  
- [Securities Commission Malaysia](https://www.sc.com.my) — regulatory framework
""")

# ─────────────────────────────────────────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div class="footer">
  ⚠️ <b>Disclaimer:</b> For educational & portfolio purposes only. Nothing here constitutes financial advice.
  Always conduct your own research before making investment decisions.<br><br>
  Built by <b>Sammy</b> · Data Scientist & ML Engineer ·
  <a href="https://github.com/SamOliverAreh">GitHub</a> ·
  <a href="https://www.linkedin.com/in/sam-oliver-areh/">LinkedIn</a> ·
  <a href="https://sammy-quant-lab.streamlit.app">Live Dashboard</a>
</div>
""", unsafe_allow_html=True)

