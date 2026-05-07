"""
Sammy Quant Lab — Dashboard v3.1
Mobile-first · Multi-asset · IC-based automatic model selection
AIC / BIC / Log-Likelihood computed for ALL models → best auto-selected for forecast
"""
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(**file**), "..")))

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from data_pipeline.ingestion      import fetch_data, get_categories, get_category
from data_pipeline.preprocessing  import preprocess
from data_pipeline.features       import create_features
from models.statistical.arima     import arima_forecast
from models.statistical.garch     import garch_volatility, garch_summary
from models.statistical.ets       import ets_forecast
from models.ml.lstm               import train_lstm, forecast_lstm
from models.ml.gru                import train_gru, forecast_gru
from models.ml.transformer_model  import train_transformer, forecast_transformer
from models.ml.xgboost_model      import train_xgboost, forecast_xgboost, XGB_AVAILABLE
from models.ml.random_forest      import train_rf, forecast_rf
from models.hybrid.arima_lstm     import hybrid_forecast
from models.hybrid.ets_gru        import ets_gru_forecast
from models.ensemble.stacked      import stacked_ensemble_forecast
from models.model_selection       import run_model_selection
from evaluation.metrics           import evaluate_all, sharpe_ratio, max_drawdown

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
page_title=“Sammy Quant Lab”,
page_icon=“⚡”,
layout=“wide”,
initial_sidebar_state=“collapsed”,
)

# ─── CSS ─────────────────────────────────────────────────────────────────────

st.markdown("""

<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Outfit:wght@300;400;600;700;800&display=swap');
  html,body,[class*="css"]{font-family:'Outfit',sans-serif;}
  .main{background:#060a14;} .stApp{background:#060a14;}
  section[data-testid="stSidebar"]{background:#090d1c;border-right:1px solid #162035;}
  h1,h2,h3{font-family:'Outfit',sans-serif;}
  .ql-title{
    font-size:clamp(1.4rem,5vw,2.4rem);font-weight:800;
    background:linear-gradient(135deg,#00d4ff 0%,#0070f3 55%,#9b4dff 100%);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;
    line-height:1.15;
  }
  .ql-sub{color:#4a6a8a;font-size:clamp(0.75rem,2.5vw,0.9rem);margin-top:4px;}
  .metric-card{
    background:linear-gradient(135deg,#0b1220 0%,#0d1830 100%);
    border:1px solid #162035;border-radius:14px;
    padding:clamp(12px,3vw,20px);text-align:center;min-height:90px;
  }
  .metric-val{font-family:'DM Mono',monospace;font-size:clamp(1.1rem,3.5vw,1.7rem);font-weight:500;color:#00d4ff;}
  .metric-lbl{font-size:clamp(0.65rem,2vw,0.75rem);color:#4a6a8a;text-transform:uppercase;letter-spacing:.08em;margin-top:4px;}
  .metric-delta{font-size:.82rem;margin-top:5px;}
  .pos{color:#00e676;} .neg{color:#ff4466;}
  .sec-hdr{
    font-size:clamp(.85rem,2.5vw,1rem);font-weight:600;color:#c8d8e8;
    border-left:3px solid #00d4ff;padding-left:10px;margin:22px 0 12px 0;
  }
  .badge{display:inline-block;padding:3px 11px;border-radius:20px;font-size:.72rem;font-family:'DM Mono',monospace;}
  .badge-live {background:#001a0f;color:#00e676;border:1px solid #00e676;}
  .badge-model{background:#001526;color:#00d4ff;border:1px solid #00d4ff;}
  .badge-cat  {background:#1a0d30;color:#9b4dff;border:1px solid #9b4dff;}
  .badge-best {background:#001a0f;color:#00e676;border:1px solid #00e676;font-weight:700;}
  /* IC winner banner */
  .ic-winner{
    background:linear-gradient(135deg,rgba(0,230,118,.08),rgba(0,212,255,.06));
    border:1px solid rgba(0,230,118,.25);border-radius:12px;
    padding:16px 20px;margin:14px 0;
  }
  .ic-winner-title{font-size:1rem;font-weight:700;color:#00e676;margin-bottom:4px;}
  .ic-winner-sub{font-size:.82rem;color:#4a8a6a;}
  /* IC table highlight */
  .ic-best-row td{background:rgba(0,230,118,.06)!important;color:#00e676!important;}
  .stButton>button{
    background:linear-gradient(135deg,#0055cc,#0080ff);color:white;border:none;
    border-radius:10px;padding:10px 24px;font-family:'Outfit',sans-serif;font-weight:700;
    font-size:clamp(.85rem,2.5vw,.95rem);width:100%;transition:opacity .2s,transform .2s;
  }
  .stButton>button:hover{opacity:.85;transform:translateY(-1px);}
  .stSelectbox label,.stSlider label,.stCheckbox label,.stRadio label{color:#7a9abb!important;font-size:.83rem;}
  div[data-testid="stMetric"]{background:#0b1220;border:1px solid #162035;border-radius:10px;padding:14px;}
  div[data-testid="stMetric"] label{color:#4a6a8a!important;}
  div[data-testid="stMetric"] div[data-testid="stMetricValue"]{color:#00d4ff!important;font-family:'DM Mono',monospace;}
  .stDataFrame{border-radius:10px;overflow:hidden;}
  @media(max-width:768px){
    .modebar{display:none!important;}
    .main .block-container{padding:.75rem .75rem 4rem!important;}
  }
  div[data-testid="stInfo"]   {background:#001830;border:1px solid #0040a0;border-radius:10px;}
  div[data-testid="stWarning"]{background:#1a0c00;border:1px solid #a06000;border-radius:10px;}
  hr{border-color:#162035;}
  .footer{text-align:center;color:#2a4060;font-size:.72rem;padding:24px 0 8px;}
  .footer a{color:#0070f3;text-decoration:none;}
  /* IC comparison chart label */
  .ic-note{font-size:.74rem;color:#3a5a7a;margin-top:6px;padding-left:4px;}
</style>

""", unsafe_allow_html=True)

PLOT_BASE = dict(
paper_bgcolor=“rgba(0,0,0,0)”,
plot_bgcolor=“rgba(9,13,28,0.9)”,
font=dict(family=“Outfit, sans-serif”, color=”#7a9abb”, size=11),
xaxis=dict(gridcolor=”#0e1e35”, showgrid=True, zeroline=False),
yaxis=dict(gridcolor=”#0e1e35”, showgrid=True, zeroline=False),
margin=dict(l=10, r=10, t=36, b=10),
legend=dict(bgcolor=“rgba(0,0,0,0)”, bordercolor=”#162035”, borderwidth=1, font=dict(size=10)),
)

# ─── Model registry ───────────────────────────────────────────────────────────

MODEL_INFO = {
“ARIMA”:              {“group”: “Statistical”,  “color”: “#00d4ff”, “desc”: “Auto ARIMA (AIC order selection)”},
“ETS”:                {“group”: “Statistical”,  “color”: “#29b6f6”, “desc”: “Exponential Smoothing (Holt damped trend)”},
“Prophet”:            {“group”: “Statistical”,  “color”: “#4fc3f7”, “desc”: “Meta Prophet (trend + seasonality)”, “needs”: “prophet”},
“LSTM”:               {“group”: “Deep Learning”,“color”: “#9b4dff”, “desc”: “Stacked LSTM (PyTorch)”},
“GRU”:                {“group”: “Deep Learning”,“color”: “#b06fff”, “desc”: “Gated Recurrent Unit (PyTorch)”},
“Transformer”:        {“group”: “Deep Learning”,“color”: “#ce93d8”, “desc”: “Encoder-only Transformer (PyTorch)”},
“XGBoost”:            {“group”: “Tree Models”,  “color”: “#ff9500”, “desc”: “XGBoost (recursive lag features)”, “needs”: “xgb”},
“Random Forest”:      {“group”: “Tree Models”,  “color”: “#ffc107”, “desc”: “Random Forest (recursive lag features)”},
“Hybrid ARIMA+LSTM”:  {“group”: “Hybrid”,       “color”: “#ff4081”, “desc”: “ARIMA linear + LSTM residual”},
“Hybrid ETS+GRU”:     {“group”: “Hybrid”,       “color”: “#f48fb1”, “desc”: “ETS baseline + GRU residual”},
“Hybrid Prophet+XGB”: {“group”: “Hybrid”,       “color”: “#ff6e40”, “desc”: “Prophet trend + XGBoost residual”, “needs”: “prophet_xgb”},
“Stacked Ensemble”:   {“group”: “Ensemble”,     “color”: “#00e676”, “desc”: “Ridge meta-learner on 6 base models”},
}

def model_available(name):
needs = MODEL_INFO[name].get(“needs”, “”)
if needs == “prophet”:     return PROPHET_AVAILABLE
if needs == “xgb”:         return XGB_AVAILABLE
if needs == “prophet_xgb”: return PROPHET_XGB_AVAILABLE
return True

AVAILABLE_MODELS = [m for m in MODEL_INFO if model_available(m)]
MODEL_GROUPS = {}
for m in AVAILABLE_MODELS:
MODEL_GROUPS.setdefault(MODEL_INFO[m][“group”], []).append(m)

CATEGORIES = get_categories()

# ─────────────────────────────────────────────────────────────────────────────

# SIDEBAR

# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
st.markdown(’<div class="ql-title">⚡ Sammy Quant</div>’, unsafe_allow_html=True)
st.markdown(’<div class="ql-sub">Multi-Asset Forecasting Lab v3.1</div>’, unsafe_allow_html=True)
st.markdown(”—”)

```
st.markdown('<div class="sec-hdr">📊 Asset Universe</div>', unsafe_allow_html=True)
cat_choice  = st.selectbox("Asset Class", list(CATEGORIES.keys()))
asset_names = CATEGORIES[cat_choice]
ticker      = st.selectbox("Instrument", asset_names)

st.markdown('<div class="sec-hdr">📅 Data Range</div>', unsafe_allow_html=True)
start_date     = st.date_input("Start Date", value=pd.Timestamp("2021-01-01"))
forecast_steps = st.slider("Forecast Horizon (days)", 5, 90, 30)

st.markdown('<div class="sec-hdr">🤖 Model Selection Mode</div>', unsafe_allow_html=True)
selection_mode = st.radio(
    "Mode",
    ["🏆 Auto (IC-based)", "🎛 Manual"],
    help="Auto: runs AIC/BIC/LogL across selected models and picks the best. Manual: you pick the model directly.",
)

auto_mode = selection_mode.startswith("🏆")

if auto_mode:
    st.markdown('<div class="sec-hdr">📐 Models to Evaluate</div>', unsafe_allow_html=True)
    st.caption("Select which models to include in IC comparison:")
    ic_candidates = []
    for grp, mlist in MODEL_GROUPS.items():
        with st.expander(grp, expanded=(grp in ("Statistical", "Tree Models"))):
            for m in mlist:
                # Default: include Statistical + Tree; exclude slow DL from auto by default
                default = MODEL_INFO[m]["group"] in ("Statistical", "Tree Models")
                if st.checkbox(m, value=default, key=f"ic_{m}"):
                    ic_candidates.append(m)
    if not ic_candidates:
        st.warning("Select at least one model.")
else:
    group_choice = st.radio("Model Group", list(MODEL_GROUPS.keys()))
    model_choice = st.selectbox("Select Model", MODEL_GROUPS[group_choice])
    st.markdown(f"""
    <div style="background:#050f1a;border:1px solid #162035;border-radius:8px;padding:10px 12px;font-size:.8rem;color:#4a6a8a;margin-top:4px;">
    <b style="color:#00d4ff;">{model_choice}</b><br>{MODEL_INFO[model_choice]['desc']}
    </div>""", unsafe_allow_html=True)

show_garch    = st.checkbox("GARCH Volatility Overlay", value=True)
show_features = st.checkbox("Technical Indicators", value=True)

st.markdown('<div class="sec-hdr">⚙️ DL Hyperparams</div>', unsafe_allow_html=True)
dl_epochs = st.slider("Epochs", 10, 100, 30)
dl_hidden = st.selectbox("Hidden Units", [32, 64, 128], index=1)
dl_window = st.slider("Look-back Window", 10, 60, 20)

st.markdown("---")
run_btn = st.button("🚀 Run Analysis", use_container_width=True)

st.markdown("""
<div style="text-align:center;margin-top:16px;font-size:.73rem;color:#2a4060;">
  Built by <b style="color:#00d4ff;">Sammy</b><br>Data Scientist · ML Engineer<br>
  <a href="https://github.com/SamOliverAreh" style="color:#0070f3;">GitHub</a> ·
  <a href="https://www.linkedin.com/in/sam-oliver-areh/" style="color:#0070f3;">LinkedIn</a><br><br>
  <span class="badge badge-live">● LIVE DATA</span>
</div>""", unsafe_allow_html=True)
```

# ─────────────────────────────────────────────────────────────────────────────

# HEADER

# ─────────────────────────────────────────────────────────────────────────────

col_t, col_b = st.columns([3, 1])
with col_t:
cat = get_category(ticker)
mode_label = “Auto IC” if auto_mode else (model_choice if not auto_mode else “”)
st.markdown(f”””
<div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;">
<div class="ql-title">⚡ Sammy Quant Lab</div>
<span class="badge badge-model">{mode_label}</span>
<span class="badge badge-cat">{cat}</span>
</div>
<div class="ql-sub">Multi-Asset Forecasting · {ticker} · {cat} · Real-time Data · v3.1</div>
“””, unsafe_allow_html=True)
with col_b:
st.markdown(’<div style="padding-top:10px;text-align:right;"><span class="badge badge-live">● LIVE</span></div>’,
unsafe_allow_html=True)
st.markdown(”—”)

# ─────────────────────────────────────────────────────────────────────────────

# LANDING

# ─────────────────────────────────────────────────────────────────────────────

if not run_btn:
st.info(“👈 Open the **sidebar** (☰) to configure, then tap **Run Analysis**.”)
st.markdown(’<div class="sec-hdr">🌐 Asset Universe</div>’, unsafe_allow_html=True)
tabs = st.tabs(list(CATEGORIES.keys()))
for tab, (cn, names) in zip(tabs, CATEGORIES.items()):
with tab:
st.dataframe(pd.DataFrame({“Instrument”: names, “Class”: [cn]*len(names)}),
use_container_width=True, hide_index=True)
st.markdown(’<div class="sec-hdr">🤖 Model Catalogue</div>’, unsafe_allow_html=True)
rows = [{“Model”: m, “Group”: MODEL_INFO[m][“group”], “Description”: MODEL_INFO[m][“desc”],
“Available”: “✅” if model_available(m) else “⚠️ pkg missing”} for m in MODEL_INFO]
st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

```
# IC explanation
with st.expander("📐 How does IC-based Model Selection work?"):
    st.markdown("""
```

**Information Criteria (IC)** measure how well a model fits the data, penalised for complexity.

|Criterion         |Formula            |Prefer                           |
|------------------|-------------------|---------------------------------|
|**AIC** (Akaike)  |−2·LogL + 2k       |**Lowest**                       |
|**BIC** (Bayesian)|−2·LogL + k·ln(n)  |**Lowest**                       |
|**Log-Likelihood**|log P(data | model)|**Least negative** (closest to 0)|

- For **statistical models** (ARIMA, ETS, GARCH): exact likelihood from the fitted model.
- For **ML / tree models** (LSTM, GRU, XGBoost, RF): *pseudo-likelihood* from in-sample Gaussian residuals.
- For **hybrids**: average IC of the two component models.

The model with the **lowest AIC** is automatically selected and used for the forecast, backtest, and forecast table.
“””)
st.stop()

# ─────────────────────────────────────────────────────────────────────────────

# DATA

# ─────────────────────────────────────────────────────────────────────────────

with st.spinner(f”📡 Fetching {ticker} data…”):
try:
df_raw  = fetch_data(ticker, start=str(start_date))
df      = preprocess(df_raw.copy())
df_feat = create_features(df.copy())
series  = df[“Close”]
except Exception as e:
st.error(f”Data fetch failed: {e}”); st.stop()

if len(series) < 60:
st.warning(“⚠️ Less than 60 data points — results may be unreliable.”)

# ─────────────────────────────────────────────────────────────────────────────

# KPI ROW

# ─────────────────────────────────────────────────────────────────────────────

cp    = float(series.iloc[-1])
pp    = float(series.iloc[-2]) if len(series) > 1 else cp
pchg  = (cp - pp) / (pp + 1e-12) * 100
vol30 = float(df[“returns”].rolling(30).std().iloc[-1] * 100)
sr    = sharpe_ratio(df[“returns”].dropna())
mdd   = max_drawdown(series.values) * 100

def kpi(val, lbl, delta=None):
dh = “”
if delta is not None:
cls = “pos” if delta >= 0 else “neg”
dh = f’<div class="metric-delta {cls}">{delta:+.3f}%</div>’
return f’<div class="metric-card"><div class="metric-val">{val}</div><div class="metric-lbl">{lbl}</div>{dh}</div>’

c1,c2,c3,c4,c5 = st.columns(5)
with c1: st.markdown(kpi(f”{cp:.4f}”, “Price”, pchg), unsafe_allow_html=True)
with c2: st.markdown(kpi(f”{vol30:.2f}%”, “30d Volatility”), unsafe_allow_html=True)
with c3: st.markdown(kpi(f”{sr:.2f}”, “Sharpe Ratio”), unsafe_allow_html=True)
with c4: st.markdown(kpi(f”{mdd:.2f}%”, “Max Drawdown”), unsafe_allow_html=True)
with c5: st.markdown(kpi(f”{len(series):,}”, “Data Points”), unsafe_allow_html=True)
st.markdown(”—”)

# ─────────────────────────────────────────────────────────────────────────────

# PRICE CHART

# ─────────────────────────────────────────────────────────────────────────────

st.markdown(’<div class="sec-hdr">📈 Price History & Technical Indicators</div>’, unsafe_allow_html=True)
tp, tr, tc = st.tabs([“Price Chart”, “Returns Distribution”, “Correlation”])

with tp:
has_vol = “Volume” in df_raw.columns and df_raw[“Volume”].sum() > 0
nrows   = 2 if has_vol else 1
fig     = make_subplots(rows=nrows, cols=1, shared_xaxes=True,
row_heights=[0.72, 0.28][:nrows], vertical_spacing=0.05)
fig.add_trace(go.Scatter(x=df.index, y=df[“Close”], name=“Close”,
line=dict(color=”#00d4ff”, width=1.5)), row=1, col=1)
if show_features and “ma_21” in df_feat.columns:
fig.add_trace(go.Scatter(x=df_feat.index, y=df_feat[“ma_21”], name=“MA 21”,
line=dict(color=”#ff9500”, width=1, dash=“dot”)), row=1, col=1)
fig.add_trace(go.Scatter(x=df_feat.index, y=df_feat[“ma_50”], name=“MA 50”,
line=dict(color=”#9b4dff”, width=1, dash=“dot”)), row=1, col=1)
fig.add_trace(go.Scatter(x=df_feat.index, y=df_feat[“bb_upper”],
line=dict(color=”#2a3d55”, width=1), showlegend=False), row=1, col=1)
fig.add_trace(go.Scatter(x=df_feat.index, y=df_feat[“bb_lower”],
line=dict(color=”#2a3d55”, width=1),
fill=“tonexty”, fillcolor=“rgba(0,100,200,0.06)”, name=“BB Band”), row=1, col=1)
if has_vol and nrows == 2:
fig.add_trace(go.Bar(x=df_raw.index, y=df_raw[“Volume”], name=“Volume”,
marker_color=“rgba(0,212,255,0.18)”), row=2, col=1)
fig.update_layout(**PLOT_BASE, height=400,
title=dict(text=f”{ticker} — Price History”, font=dict(size=14)))
st.plotly_chart(fig, use_container_width=True)

with tr:
fig2 = go.Figure(go.Histogram(x=df[“returns”].dropna()*100, nbinsx=60,
marker_color=”#00d4ff”, opacity=0.7))
fig2.update_layout(**PLOT_BASE, height=320, xaxis_title=“Daily Return (%)”,
yaxis_title=“Frequency”, title=dict(text=“Returns Distribution”, font=dict(size=14)))
st.plotly_chart(fig2, use_container_width=True)

with tc:
fcols  = [c for c in [“Close”,“ma_21”,“ma_50”,“rsi”,“macd”,“bb_width”,“volatility_30”,“atr”] if c in df_feat.columns]
cdf    = df_feat[fcols].corr()
fig3   = go.Figure(go.Heatmap(z=cdf.values, x=cdf.columns, y=cdf.columns,
colorscale=“RdBu”, zmid=0,
text=np.round(cdf.values,2), texttemplate=”%{text}”, showscale=True))
fig3.update_layout(**PLOT_BASE, height=360, title=dict(text=“Feature Correlation Matrix”, font=dict(size=14)))
st.plotly_chart(fig3, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────

# GARCH

# ─────────────────────────────────────────────────────────────────────────────

if show_garch:
st.markdown(’<div class="sec-hdr">📉 GARCH(1,1) Volatility Forecast</div>’, unsafe_allow_html=True)
with st.spinner(“Fitting GARCH…”):
try:
gvol = garch_volatility(df[“returns”].dropna(), horizon=forecast_steps)
gsum = garch_summary(df[“returns”].dropna())
fig_g = go.Figure(go.Scatter(y=gvol*100, mode=“lines+markers”,
line=dict(color=”#ff9500”, width=2), marker=dict(size=4)))
fig_g.update_layout(**PLOT_BASE, height=260,
xaxis_title=f”Days ahead (t+1–t+{forecast_steps})”,
yaxis_title=“Volatility (%)”,
title=dict(text=“GARCH Forward Volatility Forecast”, font=dict(size=14)))
st.plotly_chart(fig_g, use_container_width=True)
except Exception as ge:
st.warning(f”GARCH failed: {ge}”)

st.markdown(”—”)

# ─────────────────────────────────────────────────────────────────────────────

# ══════════════  IC MODEL SELECTION  ══════════════

# ─────────────────────────────────────────────────────────────────────────────

st.markdown(’<div class="sec-hdr">📐 Information Criteria — Model Selection</div>’, unsafe_allow_html=True)

if auto_mode:
if not ic_candidates:
st.error(“No models selected for IC evaluation.”); st.stop()

```
with st.spinner(f"Computing AIC / BIC / Log-Likelihood for {len(ic_candidates)} models..."):
    ic_table, best_model = run_model_selection(
        series, ic_candidates, dl_epochs=dl_epochs, dl_hidden=dl_hidden, dl_window=dl_window
    )

# ── Winner banner ──────────────────────────────────────────────────────
best_row = ic_table[ic_table["Model"] == best_model].iloc[0]
st.markdown(f"""
<div class="ic-winner">
  <div class="ic-winner-title">🏆 Best Model: {best_model}</div>
  <div class="ic-winner-sub">
    AIC = <b>{best_row['AIC']}</b> &nbsp;·&nbsp;
    BIC = <b>{best_row['BIC']}</b> &nbsp;·&nbsp;
    LogL = <b>{best_row['LogL']}</b>
    &nbsp;&nbsp;→ Selected automatically for forecast, backtest & table below.
  </div>
</div>
""", unsafe_allow_html=True)

# ── IC table ───────────────────────────────────────────────────────────
st.markdown("**Full IC Comparison** — lower AIC/BIC = better · less negative LogL = better")

# Style: highlight best row green
def _style_ic(row):
    if row["Model"] == best_model:
        return ["background-color: rgba(0,230,118,0.08); color: #00e676"] * len(row)
    return [""] * len(row)

styled = ic_table.style.apply(_style_ic, axis=1).format(
    {"AIC": "{:.1f}", "BIC": "{:.1f}", "LogL": "{:.1f}"},
    na_rep="—"
)
st.dataframe(styled, use_container_width=True, hide_index=True)

# ── IC bar charts ──────────────────────────────────────────────────────
valid_ic = ic_table.dropna(subset=["AIC", "BIC", "LogL"])
if len(valid_ic) > 1:
    ic_tab1, ic_tab2, ic_tab3 = st.tabs(["AIC", "BIC", "Log-Likelihood"])

    colors_ic = ["#00e676" if m == best_model else "#1e3a5f" for m in valid_ic["Model"]]

    with ic_tab1:
        fig_aic = go.Figure(go.Bar(
            x=valid_ic["Model"], y=valid_ic["AIC"],
            marker_color=colors_ic, text=valid_ic["AIC"].round(1),
            textposition="outside"
        ))
        fig_aic.update_layout(**PLOT_BASE, height=300,
                              yaxis_title="AIC (lower = better)",
                              title=dict(text="AIC by Model", font=dict(size=13)))
        st.plotly_chart(fig_aic, use_container_width=True)
        st.markdown('<div class="ic-note">🟢 Green bar = lowest AIC = best model</div>', unsafe_allow_html=True)

    with ic_tab2:
        fig_bic = go.Figure(go.Bar(
            x=valid_ic["Model"], y=valid_ic["BIC"],
            marker_color=colors_ic, text=valid_ic["BIC"].round(1),
            textposition="outside"
        ))
        fig_bic.update_layout(**PLOT_BASE, height=300,
                              yaxis_title="BIC (lower = better)",
                              title=dict(text="BIC by Model — Penalises Complexity More Than AIC", font=dict(size=13)))
        st.plotly_chart(fig_bic, use_container_width=True)
        st.markdown('<div class="ic-note">BIC penalises extra parameters more heavily than AIC. Prefer BIC when you want a simpler model.</div>', unsafe_allow_html=True)

    with ic_tab3:
        fig_ll = go.Figure(go.Bar(
            x=valid_ic["Model"], y=valid_ic["LogL"],
            marker_color=["#00e676" if m == best_model else "#1e3a5f" for m in valid_ic["Model"]],
            text=valid_ic["LogL"].round(1), textposition="outside"
        ))
        fig_ll.update_layout(**PLOT_BASE, height=300,
                             yaxis_title="Log-Likelihood (less negative = better)",
                             title=dict(text="Log-Likelihood by Model", font=dict(size=13)))
        st.plotly_chart(fig_ll, use_container_width=True)
        st.markdown('<div class="ic-note">Less negative (closer to 0) = better fit. Note: ML models use Gaussian pseudo-likelihood.</div>', unsafe_allow_html=True)
```

else:
# Manual mode: show IC for just the chosen model
best_model = model_choice
with st.spinner(f”Computing IC for {model_choice}…”):
try:
from models.model_selection import _compute_ic
ic_single = _compute_ic(model_choice, series, dl_epochs, dl_hidden, dl_window)
c1, c2, c3 = st.columns(3)
with c1: st.metric(“AIC”,  f”{ic_single[‘AIC’]:.2f}”,  help=“Lower = better”)
with c2: st.metric(“BIC”,  f”{ic_single[‘BIC’]:.2f}”,  help=“Lower = better”)
with c3: st.metric(“Log-Likelihood”, f”{ic_single[‘LogL’]:.2f}”, help=“Less negative = better”)
st.caption(f”ℹ️ {ic_single.get(‘detail’,’’)} — IC computed for selected model only (manual mode).”)
except Exception as ice:
st.warning(f”IC computation failed for {model_choice}: {ice}”)

st.markdown(”—”)

# ─────────────────────────────────────────────────────────────────────────────

# MODEL EXECUTION  (uses best_model from IC or manual choice)

# ─────────────────────────────────────────────────────────────────────────────

st.markdown(f’<div class="sec-hdr">🤖 {best_model} — Forecast {”(IC-selected ✅)” if auto_mode else “(Manual)”}</div>’,
unsafe_allow_html=True)

pred, model_meta, conf_lower, conf_upper, base_preds = None, {}, None, None, {}

with st.spinner(f”Running {best_model}…”):
try:
if best_model == “ARIMA”:
pred, order, ci = arima_forecast(series, steps=forecast_steps)
conf_lower = ci.iloc[:,0].values if hasattr(ci,‘iloc’) else None
conf_upper = ci.iloc[:,1].values if hasattr(ci,‘iloc’) else None
model_meta = {“Order”: str(order)}

```
    elif best_model == "ETS":
        pred, params = ets_forecast(series, steps=forecast_steps)
        model_meta = params

    elif best_model == "Prophet":
        pred, conf_lower, conf_upper, _ = prophet_forecast(series, steps=forecast_steps)
        model_meta = {"Uncertainty": "95% interval"}

    elif best_model == "LSTM":
        m, sc, losses = train_lstm(series, epochs=dl_epochs, hidden=dl_hidden, window=dl_window)
        pred = forecast_lstm(m, series, sc, steps=forecast_steps, window=dl_window)
        model_meta = {"Final Loss": f"{losses[-1]:.6f}", "Epochs": dl_epochs, "Hidden": dl_hidden}

    elif best_model == "GRU":
        m, sc, losses = train_gru(series, epochs=dl_epochs, hidden=dl_hidden, window=dl_window)
        pred = forecast_gru(m, series, sc, steps=forecast_steps, window=dl_window)
        model_meta = {"Final Loss": f"{losses[-1]:.6f}", "Epochs": dl_epochs, "Hidden": dl_hidden}

    elif best_model == "Transformer":
        m, sc, losses = train_transformer(series, epochs=dl_epochs, window=dl_window)
        pred = forecast_transformer(m, series, sc, steps=forecast_steps, window=dl_window)
        model_meta = {"Final Loss": f"{losses[-1]:.6f}", "Epochs": dl_epochs}

    elif best_model == "XGBoost":
        m, lags = train_xgboost(series)
        pred = forecast_xgboost(m, series, steps=forecast_steps, lags=lags)
        model_meta = {"Lags": lags, "n_estimators": 300}

    elif best_model == "Random Forest":
        m, lags = train_rf(series)
        pred = forecast_rf(m, series, steps=forecast_steps, lags=lags)
        model_meta = {"Lags": lags, "n_estimators": 200}

    elif best_model == "Hybrid ARIMA+LSTM":
        pred, _, _, order = hybrid_forecast(series, steps=forecast_steps)
        model_meta = {"ARIMA Order": str(order), "Components": "ARIMA + LSTM residuals"}

    elif best_model == "Hybrid ETS+GRU":
        pred, _, _ = ets_gru_forecast(series, steps=forecast_steps)
        model_meta = {"Components": "ETS baseline + GRU residuals"}

    elif best_model == "Hybrid Prophet+XGB":
        pred, _, _, conf_lower, conf_upper = prophet_xgb_forecast(series, steps=forecast_steps)
        model_meta = {"Components": "Prophet trend + XGBoost residuals"}

    elif best_model == "Stacked Ensemble":
        with st.spinner("Training base models + meta-learner..."):
            pred, base_preds = stacked_ensemble_forecast(series, steps=forecast_steps)
        model_meta = {"Meta-learner": "Ridge Regression", "Base models": "ARIMA, ETS, LSTM, GRU, XGB, RF"}

except Exception as ex:
    st.error(f"Model error: {ex}")
    import traceback; st.code(traceback.format_exc()); st.stop()
```

if pred is None:
st.error(“No forecast produced.”); st.stop()

# ─────────────────────────────────────────────────────────────────────────────

# FORECAST CHART

# ─────────────────────────────────────────────────────────────────────────────

model_color    = MODEL_INFO[best_model][“color”]
hist_show      = min(120, len(series))
forecast_dates = pd.bdate_range(start=series.index[-1] + pd.Timedelta(days=1), periods=forecast_steps)
if cat_choice == “Crypto”:
forecast_dates = pd.date_range(start=series.index[-1] + pd.Timedelta(days=1), periods=forecast_steps)

fig_fc = go.Figure()
fig_fc.add_trace(go.Scatter(x=series.index[-hist_show:], y=series.values[-hist_show:],
name=“Historical”, line=dict(color=”#00d4ff”, width=1.5)))

if conf_lower is not None and conf_upper is not None:
n = min(len(forecast_dates), len(conf_lower), len(conf_upper))
try:
rgb = tuple(bytes.fromhex(model_color.lstrip(’#’)))
fill_col = f”rgba({rgb[0]},{rgb[1]},{rgb[2]},0.12)”
except Exception:
fill_col = “rgba(0,200,100,0.08)”
fig_fc.add_trace(go.Scatter(
x=list(forecast_dates[:n]) + list(forecast_dates[:n])[::-1],
y=list(conf_upper[:n]) + list(conf_lower[:n])[::-1],
fill=“toself”, fillcolor=fill_col, line=dict(width=0),
showlegend=True, name=“95% CI”
))

if base_preds:
for bm_name, bm_pred in base_preds.items():
n = min(len(forecast_dates), len(bm_pred))
fig_fc.add_trace(go.Scatter(x=forecast_dates[:n], y=bm_pred[:n],
name=bm_name, line=dict(width=1, dash=“dot”), opacity=0.45))

n = min(len(forecast_dates), len(pred))
fig_fc.add_trace(go.Scatter(x=forecast_dates[:n], y=pred[:n], name=best_model,
line=dict(color=model_color, width=2.5)))
fig_fc.add_trace(go.Scatter(x=[series.index[-1], forecast_dates[0]],
y=[float(series.iloc[-1]), float(pred[0])],
line=dict(color=model_color, width=1.5, dash=“dot”), showlegend=False))
label = “IC-Selected ✅” if auto_mode else “Manual”
fig_fc.update_layout(**PLOT_BASE, height=430,
title=dict(text=f”{ticker} — {forecast_steps}-Day Forecast · {best_model} ({label})”,
font=dict(size=14)))
st.plotly_chart(fig_fc, use_container_width=True)

if model_meta:
mcols = st.columns(min(len(model_meta), 4))
for col, (k, v) in zip(mcols, model_meta.items()):
with col: st.metric(k, str(v))

# ─────────────────────────────────────────────────────────────────────────────

# BACKTEST

# ─────────────────────────────────────────────────────────────────────────────

st.markdown(’<div class="sec-hdr">📐 Walk-Forward Backtest</div>’, unsafe_allow_html=True)

with st.spinner(“Backtesting…”):
try:
tw = min(60, len(series) - forecast_steps - 10)
if tw < 20:
st.warning(“Not enough data for backtesting.”)
else:
train_bt = series.iloc[-tw - forecast_steps: -forecast_steps]
true_bt  = series.iloc[-forecast_steps:]
bt_pred  = None

```
        if best_model == "ARIMA":
            bt_pred, _, _ = arima_forecast(train_bt, steps=forecast_steps)
        elif best_model == "ETS":
            bt_pred, _ = ets_forecast(train_bt, steps=forecast_steps)
        elif best_model == "LSTM":
            bm, bs, _ = train_lstm(train_bt, epochs=dl_epochs, hidden=dl_hidden, window=dl_window)
            bt_pred = forecast_lstm(bm, train_bt, bs, steps=forecast_steps, window=dl_window)
        elif best_model == "GRU":
            bm, bs, _ = train_gru(train_bt, epochs=dl_epochs, hidden=dl_hidden, window=dl_window)
            bt_pred = forecast_gru(bm, train_bt, bs, steps=forecast_steps, window=dl_window)
        elif best_model == "Transformer":
            bm, bs, _ = train_transformer(train_bt, epochs=dl_epochs, window=dl_window)
            bt_pred = forecast_transformer(bm, train_bt, bs, steps=forecast_steps, window=dl_window)
        elif best_model == "XGBoost":
            bm, lags = train_xgboost(train_bt)
            bt_pred = forecast_xgboost(bm, train_bt, steps=forecast_steps, lags=lags)
        elif best_model == "Random Forest":
            bm, lags = train_rf(train_bt)
            bt_pred = forecast_rf(bm, train_bt, steps=forecast_steps, lags=lags)
        elif best_model == "Hybrid ARIMA+LSTM":
            bt_pred, _, _, _ = hybrid_forecast(train_bt, steps=forecast_steps)
        elif best_model == "Hybrid ETS+GRU":
            bt_pred, _, _ = ets_gru_forecast(train_bt, steps=forecast_steps)
        elif best_model in ("Prophet", "Hybrid Prophet+XGB"):
            bt_pred = pred  # reuse — prophet too slow to re-run
        elif best_model == "Stacked Ensemble":
            bt_pred, _ = stacked_ensemble_forecast(train_bt, steps=forecast_steps)

        if bt_pred is not None:
            trim    = min(len(true_bt), len(bt_pred))
            metrics = evaluate_all(true_bt.values[:trim], bt_pred[:trim])

            m1,m2,m3,m4 = st.columns(4)
            with m1: st.metric("RMSE", f"{metrics['RMSE']:.4f}")
            with m2: st.metric("MAE",  f"{metrics['MAE']:.4f}")
            with m3: st.metric("MAPE", f"{metrics['MAPE']:.2f}%")
            with m4: st.metric("Direction Acc.", f"{metrics['Direction Accuracy (%)']:.1f}%")

            fig_bt = go.Figure()
            fig_bt.add_trace(go.Scatter(y=true_bt.values[:trim], name="Actual",
                                        line=dict(color="#00d4ff", width=2)))
            fig_bt.add_trace(go.Scatter(y=bt_pred[:trim], name="Predicted",
                                        line=dict(color=model_color, width=2, dash="dash")))
            fig_bt.update_layout(**PLOT_BASE, height=300,
                                 title=dict(text="Backtest: Actual vs Predicted", font=dict(size=14)))
            st.plotly_chart(fig_bt, use_container_width=True)
except Exception as be:
    st.warning(f"Backtest error: {be}")
```

# ─────────────────────────────────────────────────────────────────────────────

# FORECAST TABLE

# ─────────────────────────────────────────────────────────────────────────────

st.markdown(’<div class="sec-hdr">🔮 Forecast Table</div>’, unsafe_allow_html=True)
n = min(len(forecast_dates), len(pred))
fc_tbl = pd.DataFrame({
“Date”:         [d.date() for d in forecast_dates[:n]],
“Forecast”:     [f”{p:.4f}” for p in pred[:n]],
“Chg from Now”: [f”{(p-cp)/cp*100:+.2f}%” for p in pred[:n]],
})
if conf_lower is not None and conf_upper is not None:
nl = min(n, len(conf_lower), len(conf_upper))
fc_tbl[“Lower 95%”] = [f”{v:.4f}” for v in conf_lower[:nl]] + [”—”]*(n-nl)
fc_tbl[“Upper 95%”] = [f”{v:.4f}” for v in conf_upper[:nl]] + [”—”]*(n-nl)
st.dataframe(fc_tbl, use_container_width=True, hide_index=True)

# ─────────────────────────────────────────────────────────────────────────────

# RISK DASHBOARD

# ─────────────────────────────────────────────────────────────────────────────

st.markdown(’<div class="sec-hdr">⚠️ Risk Dashboard</div>’, unsafe_allow_html=True)
r1, r2 = st.columns(2)
with r1:
fig_rv = go.Figure(go.Scatter(
x=df[“returns”].dropna().index,
y=df[“returns”].dropna().rolling(21).std() * 100 * np.sqrt(252),
line=dict(color=”#ff9500”, width=1.5), name=“Ann. Vol (21d)”
))
fig_rv.update_layout(**PLOT_BASE, height=260, yaxis_title=“Ann. Vol (%)”,
title=dict(text=“Rolling Annualised Volatility”, font=dict(size=13)))
st.plotly_chart(fig_rv, use_container_width=True)

with r2:
peak_arr = np.maximum.accumulate(series.values)
dd_arr   = (series.values - peak_arr) / (peak_arr + 1e-12) * 100
fig_dd   = go.Figure(go.Scatter(x=series.index, y=dd_arr, fill=“tozeroy”,
fillcolor=“rgba(255,68,102,0.15)”,
line=dict(color=”#ff4466”, width=1), name=“Drawdown”))
fig_dd.update_layout(**PLOT_BASE, height=260, yaxis_title=“Drawdown (%)”,
title=dict(text=“Underwater Equity Curve”, font=dict(size=13)))
st.plotly_chart(fig_dd, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────

# MALAYSIA EDUCATION

# ─────────────────────────────────────────────────────────────────────────────

with st.expander(“📚 Malaysia Investor Education — Understanding the Model Output”):
st.markdown(”””
**Bacaan untuk Pelabur Malaysia / Reading for Malaysian Investors**

|Metric                |Interpretation (EN)                                |Interpretasi (BM)                      |
|----------------------|---------------------------------------------------|---------------------------------------|
|**AIC / BIC**         |Lower = better model fit penalised for complexity  |Lebih rendah = model lebih baik        |
|**Log-Likelihood**    |Less negative = better fit to data                 |Kurang negatif = padanan lebih baik    |
|**RMSE / MAE**        |Average forecast error in price units              |Ralat ramalan purata dalam unit harga  |
|**MAPE**              |% error — <2% good for FX; <5% for equities        |Ralat %; <2% baik untuk FX             |
|**Direction Accuracy**|% correct up/down — 50% = random, >55% = meaningful|Ketepatan arah                         |
|**Sharpe Ratio**      |Risk-adjusted return; >1.0 acceptable, >2.0 strong |Pulangan terlaras risiko               |
|**Max Drawdown**      |Worst peak-to-trough loss                          |Kerugian terburuk dari puncak ke lembah|

**Key reminders / Peringatan penting:**

- IC selects the best-fitting model on **historical data only** — it does not guarantee future performance.
- USDMYR is heavily influenced by Bank Negara Malaysia (BNM) policy and global oil prices.
- For Bursa stocks, factor in **dividend yield** and **PE ratio** which price-only models ignore.
- KLCI correlates strongly with Hang Seng — monitor China macro as a leading indicator.

**Resources / Sumber:** [BNM](https://www.bnm.gov.my) · [Bursa Malaysia](https://www.bursamalaysia.com) · [SC Malaysia](https://www.sc.com.my)
“””)

# ─────────────────────────────────────────────────────────────────────────────

# FOOTER

# ─────────────────────────────────────────────────────────────────────────────

st.markdown(”—”)
st.markdown(”””

<div class="footer">
  ⚠️ Educational & portfolio purposes only. Not financial advice.<br><br>
  Built by <b>Sammy</b> · Data Scientist & ML Engineer ·
  <a href="https://github.com/SamOliverAreh">GitHub</a> ·
  <a href="https://www.linkedin.com/in/sam-oliver-areh/">LinkedIn</a> ·
  <a href="https://sammy-quant-lab.streamlit.app">Live Dashboard</a>
</div>
""", unsafe_allow_html=True)