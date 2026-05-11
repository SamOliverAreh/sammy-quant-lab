"""
Sammy Quant Lab — Dashboard v4.2
All fixes + new metrics:
- Hybrid unpack errors fixed
- Prophet/Transformer/Ensemble multivariate
- Univariate vs multivariate truly different (exog passed correctly)
- Distinct model colours (full spectrum, no tone-variants)
- Underwater equity → pure line chart
- Buy & Hold + Random Walk in equity simulation table
- Light mode full coverage (tables, header, sidebar, widgets, tabs, expanders)
- New metrics: Calmar, Sterling, Ulcer Index, Max DD Duration,
  Mean/Median Daily DD, Monte Carlo simulation
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

from data_pipeline.ingestion      import fetch_data, get_categories, get_category
from data_pipeline.preprocessing  import preprocess, logret_to_price
from data_pipeline.features       import create_features
from analysis.statistical_tests   import run_all_tests, stationarity_verdict
from analysis.feature_selection   import select_features
from models.statistical.arima     import arima_forecast
from models.statistical.ets       import ets_forecast
from models.statistical.garch     import garch_volatility, garch_summary, garch_insample_vol
from models.ml.lstm               import train_lstm, forecast_lstm
from models.ml.gru                import train_gru, forecast_gru
from models.ml.xgboost_model      import train_xgboost, forecast_xgboost, XGB_AVAILABLE
from models.ml.random_forest      import train_rf, forecast_rf
from models.hybrid.arima_lstm     import hybrid_forecast
from models.hybrid.ets_gru        import ets_gru_forecast
from evaluation.metrics import (
    evaluate_all, build_comparison_table, random_walk_forecast,
    sharpe_ratio, sortino_ratio, calmar_ratio, sterling_ratio,
    information_ratio, max_drawdown, max_drawdown_duration,
    mean_daily_drawdown, median_daily_drawdown, ulcer_index,
    equity_curve_with_tc, full_equity_metrics,
    monte_carlo_simulation, diebold_mariano
)

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
    from models.ensemble.stacked import stacked_ensemble_forecast
    ENSEMBLE_AVAILABLE = True
except Exception:
    ENSEMBLE_AVAILABLE = False

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Sammy Quant Lab", page_icon="⚡",
                   layout="wide", initial_sidebar_state="collapsed")

if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True

# ── Distinct model colours — spread across full spectrum ──────────────────────
MODEL_COLORS = {
    "ARIMA":              "#00d4ff",  # cyan
    "ETS":                "#0044ff",  # deep blue
    "Prophet":            "#00ff99",  # mint
    "LSTM":               "#cc00ff",  # violet
    "GRU":                "#ff00bb",  # hot pink
    "Transformer":        "#ff6600",  # orange
    "XGBoost":            "#ffdd00",  # yellow
    "Random Forest":      "#33cc33",  # green
    "Hybrid ARIMA+LSTM":  "#ff2222",  # red
    "Hybrid ETS+GRU":     "#ff8800",  # amber
    "Stacked Ensemble":   "#00ffdd",  # aqua
    "Random Walk":        "#888888",  # grey
    "Buy & Hold":         "#bbbbbb",  # light grey
}

# ── Full CSS — dark & light ───────────────────────────────────────────────────
def get_css(dark):
    B  = "#060a14" if dark else "#f0f4f8"
    S1 = "#090d1c" if dark else "#ffffff"
    BD = "#142030" if dark else "#d0dae8"
    BD2= "#1e3050" if dark else "#b0c4da"
    TX = "#c8d8e8" if dark else "#1a2a3a"
    MU = "#4a6a8a" if dark else "#5a7090"
    AC = "#00d4ff" if dark else "#0055cc"
    A2 = "#9b4dff" if dark else "#6020cc"
    GR = "#00e676" if dark else "#007730"
    CB = "linear-gradient(135deg,#0b1220,#0d1830)" if dark else "linear-gradient(135deg,#fff,#f0f4f8)"
    IB = "#0b1220" if dark else "#ffffff"
    TB = "#090d1c" if dark else "#ffffff"
    TH = "#0c1428" if dark else "#e8edf5"
    HB = "#090d1c" if dark else "#ffffff"
    EB = "#090d1c" if dark else "#f5f8fc"
    WB = "rgba(0,230,118,.08)" if dark else "rgba(0,119,48,.06)"
    WBR= "rgba(0,230,118,.25)" if dark else "rgba(0,119,48,.25)"
    IFB= "#001830" if dark else "#e6f0ff"
    WRB= "#1a0c00" if dark else "#fff8e6"
    return f"""<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Outfit:wght@300;400;600;700;800&display=swap');
html,body,[class*="css"],.main,.stApp{{font-family:'Outfit',sans-serif!important;background:{B}!important;color:{TX}!important;}}
.main .block-container{{background:{B}!important;padding-top:1.5rem!important;}}
section[data-testid="stSidebar"]{{background:{S1}!important;border-right:1px solid {BD}!important;}}
section[data-testid="stSidebar"] *{{color:{TX}!important;}}
section[data-testid="stSidebar"] .stSelectbox>div>div{{background:{IB}!important;color:{TX}!important;border-color:{BD2}!important;}}
header[data-testid="stHeader"]{{background:{HB}!important;border-bottom:1px solid {BD}!important;}}
header[data-testid="stHeader"] *{{color:{TX}!important;background:{HB}!important;}}
.stDeployButton{{display:none;}}
h1,h2,h3,h4,h5,h6,p,li,span{{color:{TX};}}
.ql-title{{font-size:clamp(1.4rem,5vw,2.4rem);font-weight:800;
  background:linear-gradient(135deg,{AC},{A2});
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;line-height:1.15;}}
.ql-sub{{color:{MU};font-size:clamp(.75rem,2.5vw,.9rem);margin-top:4px;}}
.metric-card{{background:{CB};border:1px solid {BD};border-radius:14px;
  padding:clamp(12px,3vw,20px);text-align:center;min-height:90px;}}
.metric-val{{font-family:'DM Mono',monospace;font-size:clamp(1.1rem,3.5vw,1.7rem);font-weight:500;color:{AC};}}
.metric-lbl{{font-size:clamp(.65rem,2vw,.75rem);color:{MU};text-transform:uppercase;letter-spacing:.08em;margin-top:4px;}}
.metric-delta{{font-size:.82rem;margin-top:5px;}}
.pos{{color:{GR};}} .neg{{color:#ff4466;}}
.sec-hdr{{font-size:clamp(.85rem,2.5vw,1rem);font-weight:600;color:{TX};
  border-left:3px solid {AC};padding-left:10px;margin:22px 0 12px 0;}}
.badge{{display:inline-block;padding:3px 11px;border-radius:20px;font-size:.72rem;font-family:'DM Mono',monospace;}}
.badge-live {{background:{"#001a0f" if dark else "#e6fff0"};color:{GR};border:1px solid {GR};}}
.badge-model{{background:{"#001526" if dark else "#e6f0ff"};color:{AC};border:1px solid {AC};}}
.badge-cat  {{background:{"#1a0d30" if dark else "#f0e6ff"};color:{A2};border:1px solid {A2};}}
.test-pass{{background:{"#001a0f" if dark else "#e6fff0"};color:{GR};border:1px solid {GR};border-radius:8px;padding:8px 14px;font-size:.82rem;margin:4px 0;}}
.test-warn{{background:{"#1a0c00" if dark else "#fff8e6"};color:#ff9500;border:1px solid #ff9500;border-radius:8px;padding:8px 14px;font-size:.82rem;margin:4px 0;}}
.ic-winner{{background:{WB};border:1px solid {WBR};border-radius:12px;padding:16px 20px;margin:14px 0;}}
.ic-winner-title{{font-size:1rem;font-weight:700;color:{GR};margin-bottom:4px;}}
.ic-winner-sub{{font-size:.82rem;color:{MU};}}
.stButton>button{{background:linear-gradient(135deg,{AC},{A2})!important;color:{"#000" if not dark else "#fff"}!important;
  border:none!important;border-radius:10px!important;padding:10px 24px!important;
  font-family:'Outfit',sans-serif!important;font-weight:700!important;width:100%!important;transition:.2s!important;}}
.stButton>button:hover{{opacity:.85!important;transform:translateY(-1px)!important;}}
.stSelectbox label,.stSlider label,.stCheckbox label,.stRadio label,
.stDateInput label,.stNumberInput label{{color:{MU}!important;font-size:.83rem!important;}}
.stSelectbox>div>div,.stMultiSelect>div>div{{background:{IB}!important;color:{TX}!important;border-color:{BD2}!important;}}
div[data-baseweb="select"]>div{{background:{IB}!important;color:{TX}!important;border-color:{BD2}!important;}}
div[data-baseweb="popover"] ul{{background:{S1}!important;color:{TX}!important;}}
div[data-baseweb="popover"] li:hover{{background:{BD}!important;}}
div[data-testid="stRadio"] label,div[data-testid="stCheckbox"] label{{color:{TX}!important;}}
div[data-testid="stDateInput"] input{{background:{IB}!important;color:{TX}!important;border-color:{BD2}!important;}}
div[data-testid="stMetric"]{{background:{IB}!important;border:1px solid {BD}!important;border-radius:10px!important;padding:14px!important;}}
div[data-testid="stMetric"] label{{color:{MU}!important;}}
div[data-testid="stMetric"] div[data-testid="stMetricValue"]{{color:{AC}!important;font-family:'DM Mono',monospace!important;}}
/* DataFrames */
.stDataFrame,div[data-testid="stDataFrame"]{{background:{TB}!important;border:1px solid {BD}!important;border-radius:10px!important;overflow:hidden!important;}}
.stDataFrame table,.stDataFrame th,.stDataFrame td{{background:{TB}!important;color:{TX}!important;border-color:{BD}!important;}}
.stDataFrame thead tr th{{background:{TH}!important;color:{TX}!important;}}
.stDataFrame tbody tr:hover td{{background:{BD}!important;}}
/* Tabs */
div[data-testid="stTabs"] button{{background:{"#090d1c" if dark else "#f0f4f8"}!important;color:{MU}!important;border-bottom:2px solid transparent!important;}}
div[data-testid="stTabs"] button[aria-selected="true"]{{background:{S1}!important;color:{AC}!important;border-bottom:2px solid {AC}!important;}}
div[data-testid="stTabContent"]{{background:{B}!important;border:1px solid {BD}!important;border-top:none!important;border-radius:0 0 10px 10px!important;}}
/* Expanders */
div[data-testid="stExpander"]{{background:{EB}!important;border:1px solid {BD}!important;border-radius:10px!important;}}
div[data-testid="stExpander"] summary{{color:{TX}!important;background:{EB}!important;}}
div[data-testid="stExpander"] div{{background:{EB}!important;color:{TX}!important;}}
/* Info/warn */
div[data-testid="stInfo"]{{background:{IFB}!important;border:1px solid {AC}!important;border-radius:10px!important;color:{TX}!important;}}
div[data-testid="stWarning"]{{background:{WRB}!important;border:1px solid #ff9500!important;border-radius:10px!important;color:{TX}!important;}}
div[data-testid="stSuccess"]{{background:{"#001a0f" if dark else "#e6fff0"}!important;border:1px solid {GR}!important;border-radius:10px!important;color:{TX}!important;}}
div[data-testid="stError"]{{border-radius:10px!important;}}
div[data-testid="stProgressBar"]>div{{background:{AC}!important;}}
::-webkit-scrollbar{{width:6px;height:6px;}}
::-webkit-scrollbar-track{{background:{B};}}
::-webkit-scrollbar-thumb{{background:{BD2};border-radius:3px;}}
@media(max-width:768px){{.modebar{{display:none!important;}}.main .block-container{{padding:.75rem .75rem 4rem!important;}}}}
hr{{border-color:{BD};}}
.footer{{text-align:center;color:{MU};font-size:.72rem;padding:24px 0 8px;}}
.footer a{{color:{AC};text-decoration:none;}}
</style>"""

st.markdown(get_css(st.session_state.dark_mode), unsafe_allow_html=True)

DARK = st.session_state.dark_mode
PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(9,13,28,0.95)" if DARK else "rgba(248,250,255,0.97)",
    font=dict(family="Outfit,sans-serif", color="#7a9abb" if DARK else "#2a4060", size=11),
    xaxis=dict(gridcolor="#0e1e35" if DARK else "#d0dde8", showgrid=True, zeroline=False),
    yaxis=dict(gridcolor="#0e1e35" if DARK else "#d0dde8", showgrid=True, zeroline=False),
    margin=dict(l=10, r=10, t=40, b=10),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#1e3050" if DARK else "#c0d0e0",
                borderwidth=1, font=dict(size=10)),
)

CATEGORIES = get_categories()

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    if st.button("☀️ Light Mode" if DARK else "🌙 Dark Mode", use_container_width=True):
        st.session_state.dark_mode = not st.session_state.dark_mode
        st.rerun()

    st.markdown('<div class="ql-title">⚡ Sammy Quant</div>', unsafe_allow_html=True)
    st.markdown('<div class="ql-sub">Multi-Asset Forecasting Lab v4.2</div>', unsafe_allow_html=True)
    st.markdown("---")

    st.markdown('<div class="sec-hdr">📊 Asset</div>', unsafe_allow_html=True)
    cat_choice = st.selectbox("Asset Class", list(CATEGORIES.keys()))
    ticker     = st.selectbox("Instrument",  CATEGORIES[cat_choice])

    st.markdown('<div class="sec-hdr">📅 Range</div>', unsafe_allow_html=True)
    start_date     = st.date_input("Start Date", value=pd.Timestamp("2021-01-01"))
    forecast_steps = st.slider("Forecast Horizon (days)", 5, 90, 30)

    st.markdown('<div class="sec-hdr">🤖 Models</div>', unsafe_allow_html=True)
    st.caption("Select models — all run together and are compared.")
    mdefs = {"ARIMA":True,"ETS":True,"LSTM":False,"GRU":False,
             "XGBoost":XGB_AVAILABLE,"Random Forest":True}
    if TRANSFORMER_AVAILABLE: mdefs["Transformer"]       = False
    if PROPHET_AVAILABLE:     mdefs["Prophet"]           = False
    mdefs["Hybrid ARIMA+LSTM"] = False
    mdefs["Hybrid ETS+GRU"]    = False
    if ENSEMBLE_AVAILABLE:    mdefs["Stacked Ensemble"]  = False

    msel = {m: st.checkbox(m, value=v, key=f"m_{m}") for m, v in mdefs.items()}
    selected_models = [m for m, v in msel.items() if v]

    st.markdown('<div class="sec-hdr">📐 Feature Selection</div>', unsafe_allow_html=True)
    use_mv = st.checkbox("Enable Multivariate (Granger + Corr)", value=False)
    if use_mv:
        g_alpha = st.slider("Granger α", 0.01, 0.10, 0.05, 0.01)
        c_thr   = st.slider("Corr threshold |r|", 0.1, 0.8, 0.3, 0.05)
    else:
        g_alpha, c_thr = 0.05, 0.3

    st.markdown('<div class="sec-hdr">⚙️ DL Params</div>', unsafe_allow_html=True)
    dl_epochs = st.slider("Epochs",          10, 100, 30)
    dl_hidden = st.selectbox("Hidden Units", [32, 64, 128], index=1)
    dl_window = st.slider("Look-back",       10,  60, 20)

    st.markdown('<div class="sec-hdr">💰 Simulation</div>', unsafe_allow_html=True)
    tc_rate      = st.slider("Transaction Cost (%)", 0.0, 0.5, 0.1, 0.01) / 100
    use_garch_sz = st.checkbox("GARCH Position Sizing", value=True)
    vol_target   = st.slider("Vol Target (daily %)", 0.5, 5.0, 2.0, 0.5)/100 if use_garch_sz else 0.02
    mc_sims      = st.slider("Monte Carlo Simulations", 100, 1000, 300, 100)

    st.markdown("---")
    run_btn = st.button("🚀 Run Analysis", use_container_width=True)
    st.markdown(f"""<div style="text-align:center;margin-top:16px;font-size:.73rem;color:{'#2a4060' if DARK else '#5a7090'};">
      Built by <b style="color:{'#00d4ff' if DARK else '#0055cc'};">Sammy</b><br>
      <a href="https://github.com/SamOliverAreh" style="color:{'#0070f3' if DARK else '#0050c0'};">GitHub</a> ·
      <a href="https://www.linkedin.com/in/sam-oliver-areh/" style="color:{'#0070f3' if DARK else '#0050c0'};">LinkedIn</a>
      <br><br><span class="badge badge-live">● LIVE DATA</span></div>""", unsafe_allow_html=True)

# ── HEADER ────────────────────────────────────────────────────────────────────
ct, cb = st.columns([3,1])
with ct:
    cat   = get_category(ticker)
    mv_lbl= "Multivariate ✅" if use_mv else "Univariate"
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;">
      <div class="ql-title">⚡ Sammy Quant Lab</div>
      <span class="badge badge-model">{mv_lbl}</span>
      <span class="badge badge-cat">{cat}</span>
    </div>
    <div class="ql-sub">Multi-Asset · {ticker} · {cat} · Log-Returns Target · v4.2</div>
    """, unsafe_allow_html=True)
with cb:
    st.markdown('<div style="padding-top:10px;text-align:right;">'
                '<span class="badge badge-live">● LIVE</span></div>', unsafe_allow_html=True)
st.markdown("---")

if not run_btn:
    st.info("👈 Open the **sidebar** (☰) — select models and tap **Run Analysis**.")
    st.stop()
if not selected_models:
    st.error("Select at least one model in the sidebar."); st.stop()

# ── DATA ──────────────────────────────────────────────────────────────────────
with st.spinner(f"📡 Fetching {ticker}..."):
    try:
        df_raw  = fetch_data(ticker, start=str(start_date))
        df      = preprocess(df_raw.copy())
        df_feat = create_features(df.copy())
        s_price = df["Close"]
        s_lr    = df["log_returns"]
        last_px = float(s_price.iloc[-1])
    except Exception as e:
        st.error(f"Data fetch failed: {e}"); st.stop()

if len(s_lr) < 60:
    st.warning("⚠️ <60 data points — results may be unreliable.")

# ── KPIs ──────────────────────────────────────────────────────────────────────
cp   = float(s_price.iloc[-1])
pp   = float(s_price.iloc[-2]) if len(s_price)>1 else cp
pchg = (cp-pp)/(pp+1e-12)*100
vol30= float(s_lr.rolling(30).std().iloc[-1]*100*np.sqrt(252))
sr   = sharpe_ratio(s_lr.dropna())
mdd  = max_drawdown(s_price.values)*100
annr = float(s_lr.mean()*252*100)
ui   = ulcer_index(s_price.values)

def kpi(val,lbl,delta=None):
    dh = ""
    if delta is not None:
        cls="pos" if delta>=0 else "neg"
        dh=f'<div class="metric-delta {cls}">{delta:+.3f}%</div>'
    return f'<div class="metric-card"><div class="metric-val">{val}</div><div class="metric-lbl">{lbl}</div>{dh}</div>'

c1,c2,c3,c4,c5,c6,c7 = st.columns(7)
with c1: st.markdown(kpi(f"{cp:.4f}","Price",pchg), unsafe_allow_html=True)
with c2: st.markdown(kpi(f"{annr:.2f}%","Ann. Return"), unsafe_allow_html=True)
with c3: st.markdown(kpi(f"{vol30:.2f}%","Ann. Vol (30d)"), unsafe_allow_html=True)
with c4: st.markdown(kpi(f"{sr:.2f}","Sharpe"), unsafe_allow_html=True)
with c5: st.markdown(kpi(f"{mdd:.2f}%","Max Drawdown"), unsafe_allow_html=True)
with c6: st.markdown(kpi(f"{ui:.2f}","Ulcer Index"), unsafe_allow_html=True)
with c7: st.markdown(kpi(f"{len(s_lr):,}","Data Points"), unsafe_allow_html=True)
st.markdown("---")

# ── PRICE + LOG-RETURNS ───────────────────────────────────────────────────────
st.markdown('<div class="sec-hdr">📈 Price History & Log-Returns</div>', unsafe_allow_html=True)
tp, tlr, tcorr = st.tabs(["Price + Indicators","Log-Returns Series","Correlation"])

with tp:
    has_vol = "Volume" in df_raw.columns and df_raw["Volume"].sum()>0
    nr = 2 if has_vol else 1
    fig = make_subplots(rows=nr,cols=1,shared_xaxes=True,
                        row_heights=[0.72,0.28][:nr],vertical_spacing=0.05)
    fig.add_trace(go.Scatter(x=df.index,y=df["Close"],name="Close",
                             line=dict(color="#00d4ff",width=1.5)),row=1,col=1)
    if "ma_21" in df_feat.columns:
        fig.add_trace(go.Scatter(x=df_feat.index,y=df_feat["ma_21"],name="MA21",
                                 line=dict(color="#ff9500",width=1,dash="dot")),row=1,col=1)
        fig.add_trace(go.Scatter(x=df_feat.index,y=df_feat["ma_50"],name="MA50",
                                 line=dict(color="#cc44ff",width=1,dash="dot")),row=1,col=1)
        fig.add_trace(go.Scatter(x=df_feat.index,y=df_feat["bb_upper"],
                                 line=dict(color="#1e3a55" if DARK else "#b0c8e0",width=1),
                                 showlegend=False),row=1,col=1)
        fig.add_trace(go.Scatter(x=df_feat.index,y=df_feat["bb_lower"],
                                 line=dict(color="#1e3a55" if DARK else "#b0c8e0",width=1),
                                 fill="tonexty",fillcolor="rgba(0,100,200,0.06)",
                                 name="BB Band"),row=1,col=1)
    if has_vol and nr==2:
        fig.add_trace(go.Bar(x=df_raw.index,y=df_raw["Volume"],name="Volume",
                             marker_color="rgba(0,212,255,0.18)"),row=2,col=1)
    fig.update_layout(**PLOT_LAYOUT,height=400,
                      title=dict(text=f"{ticker} — Price History",font=dict(size=14)))
    st.plotly_chart(fig,use_container_width=True)

with tlr:
    fig_lr = make_subplots(rows=2,cols=1,shared_xaxes=True,
                           row_heights=[0.65,0.35],vertical_spacing=0.06)
    bar_colors = ["#00e676" if v>=0 else "#ff4466" for v in s_lr.values]
    fig_lr.add_trace(go.Bar(x=s_lr.index,y=s_lr.values,
                            marker_color=bar_colors,name="Log-Returns"),row=1,col=1)
    rv = s_lr.rolling(21).std()*np.sqrt(252)
    fig_lr.add_trace(go.Scatter(x=s_lr.index,y=rv,name="Rolling Vol",
                                line=dict(color="#ff9500",width=1.5)),row=2,col=1)
    fig_lr.add_hline(y=float(rv.mean()),row=2,col=1,
                     line=dict(color="#ff9500",dash="dash",width=1),
                     annotation_text=f"Mean:{rv.mean():.4f}")
    fig_lr.add_hline(y=float(rv.median()),row=2,col=1,
                     line=dict(color="#ffd600",dash="dot",width=1),
                     annotation_text=f"Median:{rv.median():.4f}")
    fig_lr.update_layout(**PLOT_LAYOUT,height=380,
                         title=dict(text="Log-Returns & Rolling Annualised Volatility",font=dict(size=14)))
    st.plotly_chart(fig_lr,use_container_width=True)

with tcorr:
    fc2 = [c for c in ["log_returns","ma_21","ma_50","rsi","macd","bb_width","volatility_30","atr"]
           if c in df_feat.columns]
    cd  = df_feat[fc2].corr()
    fig_c = go.Figure(go.Heatmap(z=cd.values,x=cd.columns,y=cd.columns,
                                  colorscale="RdBu",zmid=0,
                                  text=np.round(cd.values,2),texttemplate="%{text}",showscale=True))
    fig_c.update_layout(**PLOT_LAYOUT,height=360,
                        title=dict(text="Feature Correlation Matrix",font=dict(size=14)))
    st.plotly_chart(fig_c,use_container_width=True)

st.markdown("---")

# ── STATISTICAL TESTS ─────────────────────────────────────────────────────────
st.markdown('<div class="sec-hdr">🧪 Statistical Tests on Log-Returns</div>', unsafe_allow_html=True)
with st.spinner("Running tests..."):
    tests = run_all_tests(s_lr)

verd = stationarity_verdict(tests["ADF"],tests["KPSS"])
st.markdown(f'<div class="ic-winner"><div class="ic-winner-title">Stationarity Verdict</div>'
            f'<div class="ic-winner-sub">{verd}</div></div>', unsafe_allow_html=True)

def tbox(res):
    p    = res.get("p_value") or res.get("lm_p_value",1.0)
    stat = res.get("statistic") or res.get("lm_statistic",0.0)
    cls  = "test-pass" if not res.get("reject_H0",False) else "test-warn"
    return (f'<div class="{cls}"><b>{res["test"]}</b> &nbsp;|&nbsp; '
            f'stat={stat:.4f} p={p:.4f}<br><b>{res["conclusion"]}</b><br>'
            f'<small>{res["interpretation"]}</small></div>')

co1,co2 = st.columns(2)
with co1:
    st.markdown(tbox(tests["ADF"]),       unsafe_allow_html=True)
    st.markdown(tbox(tests["KPSS"]),      unsafe_allow_html=True)
    st.markdown(tbox(tests["LjungBox"]),  unsafe_allow_html=True)
with co2:
    st.markdown(tbox(tests["JarqueBera"]),unsafe_allow_html=True)
    ar = tests["ARCHLM"]
    st.markdown(f'<div class="{"test-pass" if ar["reject_H0"] else "test-warn"}">'
                f'<b>{ar["test"]}</b> &nbsp;|&nbsp; LM={ar["lm_statistic"]:.4f} p={ar["lm_p_value"]:.4f}<br>'
                f'<b>{ar["conclusion"]}</b><br><small>{ar["interpretation"]}</small></div>',
                unsafe_allow_html=True)
st.markdown("---")

# ── FEATURE SELECTION ─────────────────────────────────────────────────────────
s_lr_model = s_lr
exog_df    = None

if use_mv:
    st.markdown('<div class="sec-hdr">🔗 Granger Causality + Correlation</div>', unsafe_allow_html=True)
    with st.spinner("Running Granger tests..."):
        try:
            fcols = [c for c in df_feat.columns if c not in
                     ("Close","Open","High","Low","Volume","returns","log_returns")]
            df_gr = df_feat[["log_returns"]+fcols].dropna()
            sel_cols, gr_df, corr_df = select_features(
                df_gr, target_col="log_returns",
                corr_threshold=c_thr, granger_alpha=g_alpha)

            g1,g2 = st.columns(2)
            with g1:
                st.markdown("**Granger Causality**")
                st.dataframe(gr_df.head(15),use_container_width=True,hide_index=True)
            with g2:
                st.markdown("**Correlation**")
                st.dataframe(corr_df.head(15),use_container_width=True,hide_index=True)

            if sel_cols:
                st.success(f"✅ {len(sel_cols)} features selected: {', '.join(sel_cols)}")
                common     = s_lr.index.intersection(df_feat[sel_cols].dropna().index)
                s_lr_model = s_lr.loc[common]
                exog_df    = df_feat.loc[common, sel_cols]
            else:
                st.warning("No features passed — running univariate.")
        except Exception as ge:
            st.warning(f"Feature selection failed: {ge}")
    st.markdown("---")

# ── FLEXIBLE GARCH ────────────────────────────────────────────────────────────
st.markdown('<div class="sec-hdr">📉 GARCH — Flexible Order Selection</div>', unsafe_allow_html=True)
gvol = None
with st.spinner("Fitting best GARCH..."):
    try:
        gsum  = garch_summary(s_lr)
        gvol  = garch_volatility(s_lr, horizon=forecast_steps)
        gv_is = garch_insample_vol(s_lr)
        if "best_order" in gsum:
            st.markdown(f'<div class="ic-winner">'
                        f'<div class="ic-winner-title">Best GARCH{gsum["best_order"]} — {gsum["best_dist"]}</div>'
                        f'<div class="ic-winner-sub">BIC={gsum["bic"]:.2f} · LogL={gsum["loglikelihood"]:.2f}</div>'
                        f'</div>', unsafe_allow_html=True)
        fg = make_subplots(rows=1,cols=2,
                           subplot_titles=["In-Sample Conditional Vol.",
                                           f"Forward Vol ({forecast_steps}d)"])
        gva = np.array(gv_is)*100
        fg.add_trace(go.Scatter(x=s_lr.index[-len(gva):],y=gva,
                                line=dict(color="#ff9500",width=1),name="Cond.Vol"),row=1,col=1)
        fg.add_hline(y=float(np.nanmean(gva)),row=1,col=1,
                     line=dict(color="#ff9500",dash="dash",width=1),
                     annotation_text=f"Mean:{np.nanmean(gva):.2f}%")
        fg.add_hline(y=float(np.nanmedian(gva)),row=1,col=1,
                     line=dict(color="#ffd600",dash="dot",width=1),
                     annotation_text=f"Median:{np.nanmedian(gva):.2f}%")
        fg.add_trace(go.Scatter(y=gvol*100,mode="lines+markers",
                                line=dict(color="#ff9500",width=2),
                                marker=dict(size=4),name="Fwd Vol"),row=1,col=2)
        fg.update_layout(**PLOT_LAYOUT,height=280)
        st.plotly_chart(fg,use_container_width=True)
    except Exception as ge:
        st.warning(f"GARCH failed: {ge}")
st.markdown("---")

# ── MODEL EXECUTION ───────────────────────────────────────────────────────────
st.markdown(f'<div class="sec-hdr">🤖 Running {len(selected_models)} Model(s) — '
            f'{"Multivariate" if use_mv and exog_df is not None else "Univariate"}</div>',
            unsafe_allow_html=True)

mforecasts = {}   # model → forecast log-returns
minsample  = {}   # model → (fitted, actual)
mbt_preds  = {}   # model → backtest preds

tw       = min(120, len(s_lr_model) - forecast_steps - 10)
train_bt = s_lr_model.iloc[-tw-forecast_steps:-forecast_steps] if tw>=20 else None
true_bt  = s_lr_model.iloc[-forecast_steps:] if tw>=20 else None
exog_bt  = exog_df.iloc[-tw-forecast_steps:-forecast_steps] if (exog_df is not None and tw>=20) else None

prog = st.progress(0)
for i,mn in enumerate(selected_models):
    prog.progress((i+1)/len(selected_models), text=f"Training {mn}...")
    ex = exog_df  # None = univariate, DataFrame = multivariate

    try:
        if mn=="ARIMA":
            fc,order,ci,ins,act = arima_forecast(s_lr_model,steps=forecast_steps,exog=ex)
            mforecasts[mn]=fc; minsample[mn]=(ins[-250:],act[-250:])
            if train_bt is not None:
                bt,_,_,_,_ = arima_forecast(train_bt,steps=forecast_steps,exog=exog_bt)
                mbt_preds[mn]=bt

        elif mn=="ETS":
            fc,params,ins,act = ets_forecast(s_lr_model,steps=forecast_steps)
            mforecasts[mn]=fc; minsample[mn]=(ins[-250:],act[-250:])
            if train_bt is not None:
                bt,_,_,_ = ets_forecast(train_bt,steps=forecast_steps)
                mbt_preds[mn]=bt

        elif mn=="LSTM":
            m,sy,sx,_,ins,act = train_lstm(s_lr_model,window=dl_window,epochs=dl_epochs,hidden=dl_hidden,exog=ex)
            fc = forecast_lstm(m,s_lr_model,sy,steps=forecast_steps,window=dl_window,scaler_X=sx,exog=ex)
            mforecasts[mn]=fc; minsample[mn]=(ins[-250:],act[-250:])
            if train_bt is not None:
                bm,bsy,bsx,_,_,_ = train_lstm(train_bt,window=dl_window,epochs=dl_epochs,hidden=dl_hidden,exog=exog_bt)
                mbt_preds[mn] = forecast_lstm(bm,train_bt,bsy,steps=forecast_steps,window=dl_window,scaler_X=bsx,exog=exog_bt)

        elif mn=="GRU":
            m,sy,sx,_,ins,act = train_gru(s_lr_model,window=dl_window,epochs=dl_epochs,hidden=dl_hidden,exog=ex)
            fc = forecast_gru(m,s_lr_model,sy,steps=forecast_steps,window=dl_window,scaler_X=sx,exog=ex)
            mforecasts[mn]=fc; minsample[mn]=(ins[-250:],act[-250:])
            if train_bt is not None:
                bm,bsy,bsx,_,_,_ = train_gru(train_bt,window=dl_window,epochs=dl_epochs,hidden=dl_hidden,exog=exog_bt)
                mbt_preds[mn] = forecast_gru(bm,train_bt,bsy,steps=forecast_steps,window=dl_window,scaler_X=bsx,exog=exog_bt)

        elif mn=="Transformer" and TRANSFORMER_AVAILABLE:
            m,sy,sx,_,ins,act = train_transformer(s_lr_model,window=dl_window,epochs=dl_epochs,exog=ex)
            fc = forecast_transformer(m,s_lr_model,sy,steps=forecast_steps,window=dl_window,scaler_X=sx,exog=ex)
            mforecasts[mn]=fc; minsample[mn]=(ins[-250:],act[-250:])
            if train_bt is not None:
                bm,bsy,bsx,_,_,_ = train_transformer(train_bt,window=dl_window,epochs=dl_epochs,exog=exog_bt)
                mbt_preds[mn] = forecast_transformer(bm,train_bt,bsy,steps=forecast_steps,window=dl_window,scaler_X=bsx,exog=exog_bt)

        elif mn=="XGBoost" and XGB_AVAILABLE:
            m,lags,ins,act = train_xgboost(s_lr_model,exog=ex)
            fc = forecast_xgboost(m,s_lr_model,steps=forecast_steps,lags=lags,exog=ex)
            mforecasts[mn]=fc; minsample[mn]=(ins[-250:],act[-250:])
            if train_bt is not None:
                bm,bl,_,_ = train_xgboost(train_bt,exog=exog_bt)
                mbt_preds[mn] = forecast_xgboost(bm,train_bt,steps=forecast_steps,lags=bl,exog=exog_bt)

        elif mn=="Random Forest":
            m,lags,ins,act = train_rf(s_lr_model,exog=ex)
            fc = forecast_rf(m,s_lr_model,steps=forecast_steps,lags=lags,exog=ex)
            mforecasts[mn]=fc; minsample[mn]=(ins[-250:],act[-250:])
            if train_bt is not None:
                bm,bl,_,_ = train_rf(train_bt,exog=exog_bt)
                mbt_preds[mn] = forecast_rf(bm,train_bt,steps=forecast_steps,lags=bl,exog=exog_bt)

        elif mn=="Prophet" and PROPHET_AVAILABLE:
            fc,_,_,_ = prophet_forecast(s_lr_model,steps=forecast_steps,exog=ex)
            mforecasts[mn]=fc; minsample[mn]=(np.zeros(5),np.zeros(5))
            if train_bt is not None:
                bt,_,_,_ = prophet_forecast(train_bt,steps=forecast_steps,exog=exog_bt)
                mbt_preds[mn]=bt

        elif mn=="Hybrid ARIMA+LSTM":
            fc,_,_,_ = hybrid_forecast(s_lr_model,steps=forecast_steps,exog=ex)
            mforecasts[mn]=fc; minsample[mn]=(np.zeros(5),np.zeros(5))
            if train_bt is not None:
                bt,_,_,_ = hybrid_forecast(train_bt,steps=forecast_steps,exog=exog_bt)
                mbt_preds[mn]=bt

        elif mn=="Hybrid ETS+GRU":
            fc,_,_ = ets_gru_forecast(s_lr_model,steps=forecast_steps,exog=ex)
            mforecasts[mn]=fc; minsample[mn]=(np.zeros(5),np.zeros(5))
            if train_bt is not None:
                bt,_,_ = ets_gru_forecast(train_bt,steps=forecast_steps,exog=exog_bt)
                mbt_preds[mn]=bt

        elif mn=="Stacked Ensemble" and ENSEMBLE_AVAILABLE:
            with st.spinner("Stacked Ensemble..."):
                fc,_ = stacked_ensemble_forecast(s_lr_model,steps=forecast_steps,exog=ex)
            mforecasts[mn]=fc; minsample[mn]=(np.zeros(5),np.zeros(5))
            if train_bt is not None:
                bt,_ = stacked_ensemble_forecast(train_bt,steps=forecast_steps,exog=exog_bt)
                mbt_preds[mn]=bt

    except Exception as ex_:
        st.warning(f"{mn} failed: {ex_}")

prog.empty()

if not mforecasts:
    st.error("All models failed."); st.stop()

st.markdown("---")

# ── IN-SAMPLE: ACTUAL VS PREDICTED ───────────────────────────────────────────
st.markdown('<div class="sec-hdr">🔬 In-Sample: Actual vs Predicted Log-Returns</div>',
            unsafe_allow_html=True)
st.caption("A good model tracks turning points — not just a lagged copy.")
valid_ins = {k:v for k,v in minsample.items() if len(v[0])>5 and not np.all(v[0]==0)}
if valid_ins:
    fig_ins = go.Figure()
    act_ref = list(valid_ins.values())[0][1]
    fig_ins.add_trace(go.Scatter(y=act_ref,name="Actual",
                                 line=dict(color="#ffffff" if DARK else "#111111",width=2)))
    for mn,(ip,ia) in valid_ins.items():
        n = min(len(ip),len(ia))
        fig_ins.add_trace(go.Scatter(y=ip[:n],name=f"{mn} fitted",
                                     line=dict(color=MODEL_COLORS.get(mn,"#888"),width=1.2,dash="dash")))
    fig_ins.update_layout(**PLOT_LAYOUT,height=320,
                          title=dict(text="In-Sample Fitted vs Actual (last 250 obs)",font=dict(size=14)),
                          xaxis_title="Observation",yaxis_title="Log-Return")
    st.plotly_chart(fig_ins,use_container_width=True)
st.markdown("---")

# ── FORECAST CHARTS ───────────────────────────────────────────────────────────
st.markdown('<div class="sec-hdr">🔮 Forecast — Log-Returns & Reconstructed Price</div>',
            unsafe_allow_html=True)

fc_dates = pd.bdate_range(start=s_lr_model.index[-1]+pd.Timedelta(days=1),periods=forecast_steps)
if cat_choice=="Crypto":
    fc_dates = pd.date_range(start=s_lr_model.index[-1]+pd.Timedelta(days=1),periods=forecast_steps)

t1,t2 = st.tabs(["Log-Returns Forecast","Reconstructed Price Forecast"])
with t1:
    flr = go.Figure()
    hn  = min(60,len(s_lr_model))
    flr.add_trace(go.Scatter(x=s_lr_model.index[-hn:],y=s_lr_model.values[-hn:],
                              name="Historical",line=dict(color="#ffffff" if DARK else "#333",width=1),opacity=0.5))
    flr.add_hline(y=0,line=dict(color="#607080",width=1,dash="dot"))
    for mn,fc in mforecasts.items():
        n = min(len(fc_dates),len(fc))
        flr.add_trace(go.Scatter(x=fc_dates[:n],y=fc[:n],name=mn,
                                  line=dict(color=MODEL_COLORS.get(mn,"#aaa"),width=2)))
    flr.update_layout(**PLOT_LAYOUT,height=360,
                      title=dict(text=f"Forecasted Log-Returns — {forecast_steps}d",font=dict(size=14)),
                      yaxis_title="Log-Return")
    st.plotly_chart(flr,use_container_width=True)

with t2:
    fpx = go.Figure()
    hn  = min(120,len(s_price))
    fpx.add_trace(go.Scatter(x=s_price.index[-hn:],y=s_price.values[-hn:],
                              name="Historical",line=dict(color="#ffffff" if DARK else "#333",width=1.5)))
    for mn,fc in mforecasts.items():
        px_fc = logret_to_price(fc,last_px)
        n     = min(len(fc_dates),len(px_fc))
        fpx.add_trace(go.Scatter(x=fc_dates[:n],y=px_fc[:n],name=mn,
                                  line=dict(color=MODEL_COLORS.get(mn,"#aaa"),width=2)))
        fpx.add_trace(go.Scatter(x=[s_price.index[-1],fc_dates[0]],
                                  y=[last_px,float(px_fc[0])],
                                  line=dict(color=MODEL_COLORS.get(mn,"#aaa"),width=1,dash="dot"),
                                  showlegend=False))
    fpx.update_layout(**PLOT_LAYOUT,height=400,
                      title=dict(text=f"Reconstructed Price Forecast — {forecast_steps}d",font=dict(size=14)),
                      yaxis_title="Price")
    st.plotly_chart(fpx,use_container_width=True)

with st.expander("📋 Forecast Table — Reconstructed Price"):
    tbl = {"Date":[d.date() for d in fc_dates[:forecast_steps]]}
    for mn,fc in mforecasts.items():
        px = logret_to_price(fc,last_px); n=min(forecast_steps,len(px))
        tbl[mn]=[f"{v:.4f}" for v in px[:n]]+["—"]*(forecast_steps-n)
    st.dataframe(pd.DataFrame(tbl),use_container_width=True,hide_index=True)

st.markdown("---")

# ── BACKTEST + EVALUATION ─────────────────────────────────────────────────────
st.markdown('<div class="sec-hdr">📐 Backtest & Evaluation — All Models vs Random Walk</div>',
            unsafe_allow_html=True)

if train_bt is None or true_bt is None:
    st.warning("Not enough data for backtesting.")
else:
    rw_pred = random_walk_forecast(train_bt, forecast_steps)
    fig_bt  = go.Figure()
    fig_bt.add_trace(go.Scatter(y=true_bt.values,name="Actual",
                                line=dict(color="#ffffff" if DARK else "#111",width=2)))
    fig_bt.add_trace(go.Scatter(y=rw_pred,name="Random Walk",
                                line=dict(color=MODEL_COLORS["Random Walk"],width=1.5,dash="dot")))
    for mn,bt in mbt_preds.items():
        n = min(len(true_bt),len(bt))
        fig_bt.add_trace(go.Scatter(y=bt[:n],name=mn,
                                    line=dict(color=MODEL_COLORS.get(mn,"#aaa"),width=1.8,dash="dash")))
    fig_bt.update_layout(**PLOT_LAYOUT,height=320,
                         title=dict(text="Backtest: Actual vs All Models (Log-Returns)",font=dict(size=14)),
                         yaxis_title="Log-Return")
    st.plotly_chart(fig_bt,use_container_width=True)

    comp_df = build_comparison_table(y_true=true_bt.values, model_preds=dict(mbt_preds),
                                     benchmark_returns=true_bt.values, is_returns=True)
    st.markdown("**📊 Model Comparison — ★ best per metric**")
    st.dataframe(comp_df,use_container_width=True,hide_index=True)

    if len(mbt_preds)>1:
        with st.expander("🔬 Diebold-Mariano Pairwise Tests"):
            dm_rows=[]
            nms=list(mbt_preds.keys())
            for ii in range(len(nms)):
                for jj in range(ii+1,len(nms)):
                    a,b=nms[ii],nms[jj]
                    n=min(len(true_bt),len(mbt_preds[a]),len(mbt_preds[b]))
                    dm=diebold_mariano(true_bt.values[:n],mbt_preds[a][:n],mbt_preds[b][:n])
                    dm_rows.append({"Model A":a,"Model B":b,
                                    "DM stat":dm["dm_statistic"],"p-value":dm["p_value"],
                                    "Result":dm["conclusion"]})
            if dm_rows:
                st.dataframe(pd.DataFrame(dm_rows),use_container_width=True,hide_index=True)

st.markdown("---")

# ── EQUITY SIMULATION ─────────────────────────────────────────────────────────
st.markdown('<div class="sec-hdr">💰 Equity Curve — $10,000 · TC + GARCH Sizing</div>',
            unsafe_allow_html=True)
st.caption(f"TC:{tc_rate*100:.2f}%/trade · GARCH sizing:{'On' if use_garch_sz else 'Off'} · vs Buy&Hold benchmark")

if train_bt is not None and true_bt is not None and mbt_preds:

    fig_eq = go.Figure()
    bh_curve = 10_000 * np.exp(np.cumsum(true_bt.values))
    fig_eq.add_trace(go.Scatter(y=bh_curve,name="Buy & Hold",
                                line=dict(color=MODEL_COLORS["Buy & Hold"],width=2,dash="dot")))

    rw_ec = equity_curve_with_tc(true_bt.values,rw_pred[:len(true_bt)],tc_rate=tc_rate,
                                  garch_vol=gvol[:len(true_bt)] if (use_garch_sz and gvol is not None) else None,
                                  vol_target=vol_target)
    fig_eq.add_trace(go.Scatter(y=rw_ec["equity"],name="Random Walk",
                                line=dict(color=MODEL_COLORS["Random Walk"],width=1.5,dash="dash")))

    all_ec   = {}
    perf_rows = []

    # Buy & Hold row
    perf_rows.append({
        "Model":"Buy & Hold",
        "Final ($)": f"${bh_curve[-1]:,.0f}",
        "Ann. Return (%)": round(float(np.mean(true_bt.values)*252*100),2),
        "Sharpe":  round(sharpe_ratio(true_bt.values),3),
        "Sortino": round(sortino_ratio(true_bt.values),3),
        "Calmar":  round(calmar_ratio(true_bt.values),3),
        "Sterling":round(sterling_ratio(true_bt.values),3),
        "Info Ratio":"—",
        "Max DD (%)": round(max_drawdown(bh_curve)*100,2),
        "Max DD Dur":  max_drawdown_duration(bh_curve),
        "Mean DD (%)": round(mean_daily_drawdown(bh_curve)*100,3),
        "Median DD (%)":round(median_daily_drawdown(bh_curve)*100,3),
        "Ulcer Idx":   round(ulcer_index(bh_curve),3),
        "Turnover":"100%","TC Cost":"$0.00",
    })

    # Random Walk row
    rw_em = full_equity_metrics(rw_ec["equity"],rw_ec["strat_returns"],rw_ec["bench_returns"])
    perf_rows.append({
        "Model":"Random Walk",
        "Final ($)":f"${rw_em['Final ($)']:,.0f}",
        "Ann. Return (%)":rw_em["Ann. Return (%)"],
        "Sharpe":rw_em["Sharpe"],"Sortino":rw_em["Sortino"],
        "Calmar":rw_em["Calmar"],"Sterling":rw_em["Sterling"],
        "Info Ratio":round(information_ratio(rw_ec["strat_returns"],rw_ec["bench_returns"]),3),
        "Max DD (%)":rw_em["Max DD (%)"],
        "Max DD Dur":rw_em["Max DD Duration"],
        "Mean DD (%)":rw_em["Mean Daily DD (%)"],
        "Median DD (%)":rw_em["Median Daily DD (%)"],
        "Ulcer Idx":rw_em["Ulcer Index"],
        "Turnover":f"{rw_ec['turnover']*100:.1f}%","TC Cost":f"${rw_ec['total_tc_cost']:.2f}",
    })

    for mn,bt in mbt_preds.items():
        n  = min(len(true_bt),len(bt))
        gv = gvol[:n] if (use_garch_sz and gvol is not None) else None
        ec = equity_curve_with_tc(true_bt.values[:n],bt[:n],tc_rate=tc_rate,
                                   garch_vol=gv,vol_target=vol_target)
        all_ec[mn] = ec
        fig_eq.add_trace(go.Scatter(y=ec["equity"],name=mn,
                                    line=dict(color=MODEL_COLORS.get(mn,"#aaa"),width=2)))
        em = full_equity_metrics(ec["equity"],ec["strat_returns"],ec["bench_returns"])
        perf_rows.append({
            "Model":mn,
            "Final ($)":f"${em['Final ($)']:,.0f}",
            "Ann. Return (%)":em["Ann. Return (%)"],
            "Sharpe":em["Sharpe"],"Sortino":em["Sortino"],
            "Calmar":em["Calmar"],"Sterling":em["Sterling"],
            "Info Ratio":round(information_ratio(ec["strat_returns"],ec["bench_returns"]),3),
            "Max DD (%)":em["Max DD (%)"],
            "Max DD Dur":em["Max DD Duration"],
            "Mean DD (%)":em["Mean Daily DD (%)"],
            "Median DD (%)":em["Median Daily DD (%)"],
            "Ulcer Idx":em["Ulcer Index"],
            "Turnover":f"{ec['turnover']*100:.1f}%",
            "TC Cost":f"${ec['total_tc_cost']:.2f}",
        })

    fig_eq.update_layout(**PLOT_LAYOUT,height=360,
                         title=dict(text="Equity Curve — $10,000 (incl. Buy & Hold + Random Walk)",font=dict(size=14)),
                         yaxis_title="Portfolio Value ($)")
    st.plotly_chart(fig_eq,use_container_width=True)

    # ── Underwater — LINE CHART only ─────────────────────────────────────────
    fig_dd = go.Figure()
    bh_peak = np.maximum.accumulate(bh_curve)
    bh_dd   = (bh_curve-bh_peak)/(bh_peak+1e-12)*100
    fig_dd.add_trace(go.Scatter(y=bh_dd,name="Buy & Hold",mode="lines",
                                line=dict(color=MODEL_COLORS["Buy & Hold"],width=2,dash="dot")))
    # Mean/Median lines on B&H
    fig_dd.add_hline(y=float(np.mean(bh_dd)),
                     line=dict(color="#aaaaaa",dash="dash",width=1),
                     annotation_text=f"BH Mean:{np.mean(bh_dd):.2f}%")
    fig_dd.add_hline(y=float(np.median(bh_dd)),
                     line=dict(color="#888888",dash="dot",width=1),
                     annotation_text=f"BH Median:{np.median(bh_dd):.2f}%")

    rw_peak = np.maximum.accumulate(rw_ec["equity"])
    rw_dd   = (rw_ec["equity"]-rw_peak)/(rw_peak+1e-12)*100
    fig_dd.add_trace(go.Scatter(y=rw_dd,name="Random Walk",mode="lines",
                                line=dict(color=MODEL_COLORS["Random Walk"],width=1.5,dash="dash")))

    for mn,ec in all_ec.items():
        eq   = ec["equity"]
        peak = np.maximum.accumulate(eq)
        dd   = (eq-peak)/(peak+1e-12)*100
        fig_dd.add_trace(go.Scatter(y=dd,name=mn,mode="lines",
                                    line=dict(color=MODEL_COLORS.get(mn,"#aaa"),width=2)))

    fig_dd.update_layout(**PLOT_LAYOUT,height=320,
                         title=dict(text="Underwater Equity — Drawdown % (Line Chart, no fill overlap)",font=dict(size=14)),
                         yaxis_title="Drawdown (%)")
    st.plotly_chart(fig_dd,use_container_width=True)

    # ── Full performance table ────────────────────────────────────────────────
    st.markdown("**📊 Strategy Performance Table — All Models · Buy & Hold · Random Walk**")
    perf_df = pd.DataFrame(perf_rows)
    st.dataframe(perf_df, use_container_width=True, hide_index=True)

st.markdown("---")

# ── MONTE CARLO ───────────────────────────────────────────────────────────────
st.markdown('<div class="sec-hdr">🎲 Monte Carlo Simulation</div>', unsafe_allow_html=True)
st.caption(f"Bootstrap resampling of historical log-returns · {mc_sims} paths · {forecast_steps}-day horizon")

with st.spinner("Running Monte Carlo..."):
    try:
        mc = monte_carlo_simulation(s_lr.dropna().values, n_simulations=mc_sims,
                                    horizon=forecast_steps, initial_capital=10_000)

        fig_mc = go.Figure()
        # Plot subset of paths (max 100 for clarity)
        plot_n = min(100, mc_sims)
        for j in range(plot_n):
            fig_mc.add_trace(go.Scatter(
                y=mc["paths"][j], mode="lines",
                line=dict(color="rgba(0,212,255,0.06)" if DARK else "rgba(0,85,204,0.05)",width=1),
                showlegend=False))
        # Percentile lines
        pct = mc["percentiles"]
        for pval, col, lbl in [(pct["p5"],"#ff4466","P5"),
                                (pct["p25"],"#ff9500","P25"),
                                (pct["p50"],"#00d4ff","P50 (Median)"),
                                (pct["p75"],"#00e676","P75"),
                                (pct["p95"],"#9b4dff","P95")]:
            # Build path: start at 10000, end at pval, simplified as flat line at end
            fig_mc.add_hline(y=pval, line=dict(color=col, dash="dash", width=1.5),
                             annotation_text=f"{lbl}: ${pval:,.0f}")
        fig_mc.add_hline(y=10_000, line=dict(color="#888888", dash="dot", width=1),
                         annotation_text="Initial $10,000")
        fig_mc.update_layout(**PLOT_LAYOUT, height=360,
                             title=dict(text=f"Monte Carlo — {mc_sims} Bootstrap Paths ({forecast_steps}d)",
                                        font=dict(size=14)),
                             yaxis_title="Portfolio Value ($)")
        st.plotly_chart(fig_mc, use_container_width=True)

        # MC stats
        mc1,mc2,mc3,mc4,mc5,mc6 = st.columns(6)
        with mc1: st.metric("Median Final", f"${mc['median_final']:,.0f}")
        with mc2: st.metric("Mean Final",   f"${mc['mean_final']:,.0f}")
        with mc3: st.metric("P5 (Worst 5%)",f"${pct['p5']:,.0f}")
        with mc4: st.metric("P95 (Best 5%)",f"${pct['p95']:,.0f}")
        with mc5: st.metric("Prob. of Loss",f"{mc['prob_loss']:.1f}%")
        with mc6: st.metric("95% VaR",      f"${mc['var_95']:,.0f}")

        st.info(f"**CVaR (Expected Shortfall 95%):** ${mc['cvar_95']:,.0f} — "
                f"average loss in the worst 5% of scenarios. "
                f"**Mean Max Drawdown across simulations:** {mc['mean_max_dd']:.2f}%")
    except Exception as mce:
        st.warning(f"Monte Carlo failed: {mce}")

st.markdown("---")

# ── MALAYSIA EDUCATION ────────────────────────────────────────────────────────
with st.expander("📚 Malaysia Investor Education"):
    st.markdown("""
| Metric | Meaning | Interpretasi BM |
|--------|---------|-----------------|
| **Calmar Ratio** | Ann. Return / Max Drawdown — higher=better | Pulangan tahunan / DD maksimum |
| **Sterling Ratio** | Ann. Return / Avg Top-3 DD — more robust than Calmar | Lebih stabil berbanding Calmar |
| **Ulcer Index** | Depth+Duration of drawdowns combined | Gabungan kedalaman + tempoh DD |
| **Max DD Duration** | Longest time spent underwater (below peak) | Tempoh terlama di bawah puncak |
| **Mean/Median DD** | Average daily drawdown — closer to 0 is better | Purata DD harian |
| **Monte Carlo** | Simulates 1000s of return scenarios via bootstrap | Simulasi senario pulangan |
| **CVaR / ES** | Average loss in worst 5% scenarios — stricter than VaR | Kerugian purata dalam senario terburuk 5% |
| **Hit Ratio >55%** | Directional edge on log-returns | Kelebihan arah >55% = bermakna |

**Malaysia:** USDMYR driven by BNM policy + oil. KLCI follows Hang Seng. CPO impacts Bursa energy.
    """)

st.markdown("---")
st.markdown("""
<div class="footer">⚠️ Educational & portfolio purposes only. Not financial advice.<br>
Built by <b>Sam Oliver Areh</b> ·
<a href="https://github.com/SamOliverAreh">GitHub</a> ·
<a href="https://www.linkedin.com/in/sam-oliver-areh/">LinkedIn</a> ·
<a href="https://sammy-quant-lab.streamlit.app">Live Dashboard</a></div>""",
unsafe_allow_html=True)
