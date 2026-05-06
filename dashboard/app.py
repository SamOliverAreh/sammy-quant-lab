import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from data_pipeline.ingestion import fetch_data, get_available_tickers
from data_pipeline.preprocessing import preprocess
from data_pipeline.features import create_features

from models.statistical.arima import arima_forecast
from models.statistical.garch import garch_volatility, garch_summary
from models.ml.lstm import train_lstm, forecast_lstm
from models.hybrid.arima_lstm import hybrid_forecast

from evaluation.metrics import evaluate_all, sharpe_ratio, max_drawdown


# ─── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sammy Quant Lab",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

  html, body, [class*="css"] {
      font-family: 'Space Grotesk', sans-serif;
  }

  /* Dark theme overrides */
  .main { background: #0a0e1a; }
  .stApp { background: #0a0e1a; }

  /* Sidebar */
  section[data-testid="stSidebar"] {
      background: #0d1120;
      border-right: 1px solid #1e2d4a;
  }

  /* Metric cards */
  .metric-card {
      background: linear-gradient(135deg, #0d1528 0%, #111927 100%);
      border: 1px solid #1e3a5f;
      border-radius: 12px;
      padding: 20px;
      text-align: center;
  }
  .metric-value {
      font-family: 'JetBrains Mono', monospace;
      font-size: 1.8rem;
      font-weight: 600;
      color: #00d4ff;
  }
  .metric-label {
      font-size: 0.8rem;
      color: #6b8aad;
      text-transform: uppercase;
      letter-spacing: 1px;
      margin-top: 4px;
  }
  .metric-delta {
      font-size: 0.85rem;
      margin-top: 6px;
  }
  .positive { color: #00e676; }
  .negative { color: #ff4444; }

  /* Section headers */
  .section-header {
      font-size: 1.1rem;
      font-weight: 600;
      color: #c8d8e8;
      border-left: 3px solid #00d4ff;
      padding-left: 10px;
      margin: 24px 0 12px 0;
  }

  /* Title */
  .hero-title {
      font-size: 2.2rem;
      font-weight: 700;
      background: linear-gradient(135deg, #00d4ff, #0080ff, #7b2ff7);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      background-clip: text;
  }
  .hero-sub {
      color: #5577aa;
      font-size: 0.95rem;
      margin-top: -8px;
  }

  /* Status badge */
  .status-badge {
      display: inline-block;
      padding: 3px 10px;
      border-radius: 20px;
      font-size: 0.75rem;
      font-family: 'JetBrains Mono', monospace;
  }
  .badge-live { background: #003020; color: #00e676; border: 1px solid #00e676; }
  .badge-model { background: #001a33; color: #00d4ff; border: 1px solid #00d4ff; }

  /* Plotly chart container */
  .plot-container { border-radius: 12px; overflow: hidden; }

  /* Streamlit widget overrides */
  .stSelectbox label, .stSlider label, .stCheckbox label {
      color: #8aabcc !important;
      font-size: 0.85rem;
  }
  .stButton > button {
      background: linear-gradient(135deg, #0055cc, #0080ff);
      color: white;
      border: none;
      border-radius: 8px;
      padding: 8px 24px;
      font-family: 'Space Grotesk', sans-serif;
      font-weight: 600;
      width: 100%;
      transition: opacity 0.2s;
  }
  .stButton > button:hover { opacity: 0.85; }

  div[data-testid="stMetric"] {
      background: #0d1528;
      border: 1px solid #1e3a5f;
      border-radius: 10px;
      padding: 16px;
  }
  div[data-testid="stMetric"] label { color: #6b8aad !important; }
  div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
      color: #00d4ff !important;
      font-family: 'JetBrains Mono', monospace;
  }

  hr { border-color: #1e2d4a; }
</style>
""", unsafe_allow_html=True)

PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(10,14,26,0)",
    plot_bgcolor="rgba(13,17,32,0.8)",
    font=dict(family="Space Grotesk, sans-serif", color="#8aabcc"),
    xaxis=dict(gridcolor="#1a2840", showgrid=True, zeroline=False),
    yaxis=dict(gridcolor="#1a2840", showgrid=True, zeroline=False),
    margin=dict(l=20, r=20, t=40, b=20),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#1e3a5f", borderwidth=1),
)


# ─── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="hero-title">⚡ Sammy Quant</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">FX Forecasting Lab v2.0</div>', unsafe_allow_html=True)
    st.markdown("---")

    st.markdown('<div class="section-header">📡 Data Configuration</div>', unsafe_allow_html=True)
    ticker = st.selectbox("Currency Pair", get_available_tickers(), index=0)
    start_date = st.date_input("Start Date", value=pd.Timestamp("2021-01-01"))
    forecast_steps = st.slider("Forecast Horizon (days)", 5, 90, 30)

    st.markdown('<div class="section-header">🤖 Model Selection</div>', unsafe_allow_html=True)
    model_choice = st.selectbox("Forecasting Model", ["ARIMA", "LSTM", "Hybrid (ARIMA+LSTM)"])
    show_garch = st.checkbox("Show GARCH Volatility", value=True)
    show_features = st.checkbox("Show Technical Indicators", value=True)

    st.markdown('<div class="section-header">⚙️ LSTM Hyperparameters</div>', unsafe_allow_html=True)
    lstm_epochs = st.slider("Epochs", 10, 100, 30)
    lstm_hidden = st.selectbox("Hidden Units", [32, 64, 128], index=1)
    lstm_window = st.slider("Look-back Window", 10, 60, 20)

    st.markdown("---")
    run_btn = st.button("🚀 Run Analysis")

    st.markdown("---")
    st.markdown("""
    <div style="font-size:0.75rem; color:#3d5a7a; text-align:center;">
    Built by <b style="color:#00d4ff;">Sammy</b><br>
    Data Scientist · ML Engineer<br>
    Financial Time Series Analysis<br><br>
    <span class="status-badge badge-live">● LIVE DATA</span>
    </div>
    """, unsafe_allow_html=True)


# ─── Main Content ────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="display:flex; align-items:center; gap:12px; margin-bottom:8px;">
  <div class="hero-title">📊 Sammy Quant Lab</div>
  <span class="status-badge badge-model">{model_choice}</span>
</div>
<div class="hero-sub">Multi-model FX Forecasting Engine · {ticker}/USD · Real-time Market Data</div>
""", unsafe_allow_html=True)

st.markdown("---")

if not run_btn:
    st.info("👈 Configure your analysis in the sidebar and click **Run Analysis** to begin.")

    # Show a teaser with placeholder
    st.markdown('<div class="section-header">🌐 Supported Currency Pairs</div>', unsafe_allow_html=True)
    pairs = pd.DataFrame({
        "Pair": ["USDMYR", "EURUSD", "GBPUSD", "USDJPY", "AUDUSD"],
        "Description": ["US Dollar / Malaysian Ringgit", "Euro / US Dollar", "British Pound / US Dollar",
                         "US Dollar / Japanese Yen", "Australian Dollar / US Dollar"],
        "Category": ["Emerging", "Major", "Major", "Major", "Major"],
    })
    st.dataframe(pairs, use_container_width=True, hide_index=True)
    st.stop()


# ─── Load & Process Data ─────────────────────────────────────────────────────────
with st.spinner("📡 Fetching market data..."):
    try:
        df_raw = fetch_data(ticker, start=str(start_date))
        df = preprocess(df_raw.copy())
        df_feat = create_features(df.copy())
        series = df["Close"]
        data_ok = True
    except Exception as e:
        st.error(f"Data fetch failed: {e}")
        st.stop()

# ─── KPI Row ─────────────────────────────────────────────────────────────────────
col1, col2, col3, col4, col5 = st.columns(5)

current_price = float(series.iloc[-1])
prev_price = float(series.iloc[-2])
price_chg = (current_price - prev_price) / prev_price * 100
vol_30 = float(df["returns"].rolling(30).std().iloc[-1] * 100)
sharpe = sharpe_ratio(df["returns"].dropna())
mdd = max_drawdown(series.values) * 100

with col1:
    st.metric("Current Price", f"{current_price:.4f}", f"{price_chg:+.3f}%")
with col2:
    st.metric("30d Volatility", f"{vol_30:.2f}%")
with col3:
    st.metric("Sharpe Ratio", f"{sharpe:.2f}")
with col4:
    st.metric("Max Drawdown", f"{mdd:.2f}%")
with col5:
    st.metric("Data Points", f"{len(series):,}")

st.markdown("---")

# ─── Price Chart ─────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">📈 Price History & Technical Indicators</div>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["Price Chart", "Returns Distribution", "Correlation Analysis"])

with tab1:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.7, 0.3], vertical_spacing=0.05)

    fig.add_trace(go.Scatter(
        x=df.index, y=df["Close"],
        name="Close", line=dict(color="#00d4ff", width=1.5),
    ), row=1, col=1)

    if show_features and "ma_21" in df_feat.columns:
        fig.add_trace(go.Scatter(
            x=df_feat.index, y=df_feat["ma_21"],
            name="MA 21", line=dict(color="#ff9500", width=1, dash="dot"),
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df_feat.index, y=df_feat["ma_50"],
            name="MA 50", line=dict(color="#cc44ff", width=1, dash="dot"),
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df_feat.index, y=df_feat["bb_upper"],
            name="BB Upper", line=dict(color="#334455", width=1),
            showlegend=False,
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df_feat.index, y=df_feat["bb_lower"],
            name="BB Lower", line=dict(color="#334455", width=1),
            fill="tonexty", fillcolor="rgba(0,100,200,0.05)",
        ), row=1, col=1)

    # Volume bar (if available)
    if "Volume" in df_raw.columns and df_raw["Volume"].sum() > 0:
        fig.add_trace(go.Bar(
            x=df_raw.index, y=df_raw["Volume"],
            name="Volume", marker_color="rgba(0,212,255,0.2)",
        ), row=2, col=1)
    else:
        fig.add_trace(go.Bar(
            x=df.index, y=df["returns"].abs() * 1000,
            name="|Returns|", marker_color="rgba(0,212,255,0.2)",
        ), row=2, col=1)

    fig.update_layout(**PLOT_LAYOUT, height=500, title=f"{ticker} — Price History")
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    returns_clean = df["returns"].dropna()
    fig2 = go.Figure()
    fig2.add_trace(go.Histogram(
        x=returns_clean, nbinsx=60,
        marker_color="#00d4ff", opacity=0.75, name="Returns",
    ))
    # Normal overlay
    mu, sigma = returns_clean.mean(), returns_clean.std()
    x_range = np.linspace(returns_clean.min(), returns_clean.max(), 200)
    normal_pdf = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_range - mu) / sigma) ** 2)
    scale = len(returns_clean) * (returns_clean.max() - returns_clean.min()) / 60
    fig2.add_trace(go.Scatter(
        x=x_range, y=normal_pdf * scale,
        name="Normal Dist", line=dict(color="#ff9500", width=2),
    ))
    fig2.update_layout(**PLOT_LAYOUT, height=350, title="Daily Returns Distribution",
                       xaxis_title="Return", yaxis_title="Frequency")
    st.plotly_chart(fig2, use_container_width=True)

with tab3:
    if show_features:
        corr_cols = ["Close", "returns", "ma_7", "ma_21", "volatility_10", "rsi", "macd"]
        corr_cols = [c for c in corr_cols if c in df_feat.columns]
        corr = df_feat[corr_cols].corr()
        fig3 = px.imshow(
            corr, color_continuous_scale="RdBu_r", aspect="auto",
            title="Feature Correlation Matrix",
        )
        fig3.update_layout(**PLOT_LAYOUT, height=400)
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info("Enable Technical Indicators in the sidebar to see correlation analysis.")

st.markdown("---")

# ─── Forecasting ─────────────────────────────────────────────────────────────────
st.markdown(f'<div class="section-header">🔮 {model_choice} Forecast — {forecast_steps} Days</div>',
            unsafe_allow_html=True)

forecast_pred = None
arima_order = None
train_losses = None

with st.spinner(f"⚙️ Running {model_choice} model..."):
    try:
        if model_choice == "ARIMA":
            forecast_pred, arima_order, conf_int = arima_forecast(series, steps=forecast_steps)

        elif model_choice == "LSTM":
            lstm_model, scaler, train_losses = train_lstm(
                series, window=lstm_window, epochs=lstm_epochs, hidden=lstm_hidden
            )
            forecast_pred = forecast_lstm(lstm_model, series, scaler, steps=forecast_steps, window=lstm_window)
            conf_int = None

        elif model_choice == "Hybrid (ARIMA+LSTM)":
            forecast_pred, arima_part, lstm_part, arima_order = hybrid_forecast(series, steps=forecast_steps)
            conf_int = None

        model_ok = True
    except Exception as e:
        st.error(f"Model error: {e}")
        model_ok = False

if model_ok and forecast_pred is not None:
    # Build forecast index
    last_date = series.index[-1]
    forecast_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=forecast_steps)

    # ─── Forecast Plot ───────────────────────────────────────────────────────────
    fig_f = go.Figure()

    # Historical (last 90 days for context)
    hist_window = min(90, len(series))
    fig_f.add_trace(go.Scatter(
        x=series.index[-hist_window:], y=series.values[-hist_window:],
        name="Historical", line=dict(color="#00d4ff", width=1.5),
    ))

    # Confidence interval (ARIMA)
    if conf_int is not None:
        fig_f.add_trace(go.Scatter(
            x=list(forecast_dates) + list(forecast_dates[::-1]),
            y=list(conf_int.iloc[:, 1]) + list(conf_int.iloc[:, 0][::-1]),
            fill="toself", fillcolor="rgba(0,128,255,0.1)",
            line=dict(color="rgba(0,0,0,0)"), name="95% CI",
        ))

    # Hybrid components
    if model_choice == "Hybrid (ARIMA+LSTM)":
        fig_f.add_trace(go.Scatter(
            x=forecast_dates, y=arima_part,
            name="ARIMA component", line=dict(color="#ff9500", width=1, dash="dot"),
        ))

    # Main forecast
    fig_f.add_trace(go.Scatter(
        x=forecast_dates, y=forecast_pred,
        name=f"{model_choice} Forecast",
        line=dict(color="#ff4466", width=2),
        mode="lines+markers",
        marker=dict(size=4),
    ))

    # Connecting line
    fig_f.add_trace(go.Scatter(
        x=[series.index[-1], forecast_dates[0]],
        y=[series.values[-1], forecast_pred[0]],
        line=dict(color="#ff4466", width=1, dash="dot"),
        showlegend=False,
    ))

    fig_f.update_layout(**PLOT_LAYOUT, height=420,
                        title=f"{model_choice} Forecast · {ticker}")
    st.plotly_chart(fig_f, use_container_width=True)

    # ─── Forecast Stats ──────────────────────────────────────────────────────────
    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a:
        st.metric("Forecast Start", f"{forecast_pred[0]:.4f}")
    with col_b:
        st.metric("Forecast End", f"{forecast_pred[-1]:.4f}")
    with col_c:
        fchg = (forecast_pred[-1] - forecast_pred[0]) / forecast_pred[0] * 100
        st.metric("Projected Change", f"{fchg:+.2f}%")
    with col_d:
        if arima_order:
            st.metric("ARIMA Order", f"({arima_order[0]},{arima_order[1]},{arima_order[2]})")
        else:
            st.metric("Model", model_choice.split()[0])

    # ─── LSTM Training Loss ──────────────────────────────────────────────────────
    if train_losses:
        st.markdown('<div class="section-header">📉 LSTM Training Loss</div>', unsafe_allow_html=True)
        fig_loss = go.Figure()
        fig_loss.add_trace(go.Scatter(
            y=train_losses, name="MSE Loss",
            line=dict(color="#00e676", width=2),
        ))
        fig_loss.update_layout(**PLOT_LAYOUT, height=250,
                               xaxis_title="Epoch", yaxis_title="Loss",
                               title="Training Loss Curve")
        st.plotly_chart(fig_loss, use_container_width=True)

    # ─── Backtesting ─────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="section-header">📐 Walk-Forward Backtest</div>', unsafe_allow_html=True)

    backtest_window = min(60, len(series) - forecast_steps - 10)
    if backtest_window > 10:
        with st.spinner("Running backtest..."):
            try:
                test_series = series.iloc[-backtest_window - forecast_steps: -forecast_steps]
                test_true = series.iloc[-forecast_steps:]

                if model_choice == "ARIMA":
                    bt_pred, _, _ = arima_forecast(test_series, steps=forecast_steps, order=arima_order)
                elif model_choice == "LSTM":
                    bt_model, bt_scaler, _ = train_lstm(test_series, window=lstm_window,
                                                         epochs=lstm_epochs, hidden=lstm_hidden)
                    bt_pred = forecast_lstm(bt_model, test_series, bt_scaler,
                                           steps=forecast_steps, window=lstm_window)
                else:
                    bt_pred, _, _, _ = hybrid_forecast(test_series, steps=forecast_steps)

                bt_pred_trim = bt_pred[:len(test_true)]
                metrics = evaluate_all(test_true.values[:len(bt_pred_trim)], bt_pred_trim)

                mc1, mc2, mc3, mc4 = st.columns(4)
                mc1.metric("RMSE", f"{metrics['RMSE']:.5f}")
                mc2.metric("MAE", f"{metrics['MAE']:.5f}")
                mc3.metric("MAPE", f"{metrics['MAPE']:.2f}%")
                mc4.metric("Direction Acc.", f"{metrics['Direction Accuracy (%)']:.1f}%")

                # Backtest chart
                fig_bt = go.Figure()
                fig_bt.add_trace(go.Scatter(
                    x=test_true.index[:len(bt_pred_trim)], y=test_true.values[:len(bt_pred_trim)],
                    name="Actual", line=dict(color="#00d4ff", width=2),
                ))
                fig_bt.add_trace(go.Scatter(
                    x=test_true.index[:len(bt_pred_trim)], y=bt_pred_trim,
                    name="Predicted", line=dict(color="#ff4466", width=2, dash="dot"),
                ))
                fig_bt.update_layout(**PLOT_LAYOUT, height=300,
                                     title=f"Backtest: Actual vs Predicted (last {forecast_steps} days)")
                st.plotly_chart(fig_bt, use_container_width=True)

            except Exception as e:
                st.warning(f"Backtest skipped: {e}")

# ─── GARCH Volatility ────────────────────────────────────────────────────────────
if show_garch:
    st.markdown("---")
    st.markdown('<div class="section-header">🌊 GARCH Volatility Forecast</div>', unsafe_allow_html=True)

    with st.spinner("Fitting GARCH model..."):
        try:
            returns_clean = df["returns"].dropna()
            garch_vol = garch_volatility(returns_clean, horizon=forecast_steps)
            garch_dates = pd.bdate_range(
                start=series.index[-1] + pd.Timedelta(days=1), periods=len(garch_vol)
            )

            fig_g = go.Figure()
            # Historical realized vol
            hist_vol = returns_clean.rolling(10).std() * 100
            fig_g.add_trace(go.Scatter(
                x=hist_vol.index[-120:], y=hist_vol.values[-120:],
                name="Realized Volatility (10d)", line=dict(color="#00d4ff", width=1),
                fill="tozeroy", fillcolor="rgba(0,212,255,0.05)",
            ))
            # GARCH forecast
            fig_g.add_trace(go.Scatter(
                x=garch_dates, y=garch_vol * 100,
                name="GARCH(1,1) Forecast", line=dict(color="#ff9500", width=2),
                mode="lines+markers", marker=dict(size=5),
            ))

            fig_g.update_layout(**PLOT_LAYOUT, height=320,
                                title=f"Volatility Forecast · {ticker}",
                                yaxis_title="Volatility (%)")
            st.plotly_chart(fig_g, use_container_width=True)

        except Exception as e:
            st.warning(f"GARCH unavailable: {e}")

# ─── Raw Data ────────────────────────────────────────────────────────────────────
st.markdown("---")
with st.expander("📋 Raw Data & Feature Table"):
    tab_raw, tab_feat, tab_forecast = st.tabs(["Raw OHLCV", "Feature Matrix", "Forecast Values"])
    with tab_raw:
        st.dataframe(df_raw.tail(50).sort_index(ascending=False), use_container_width=True)
    with tab_feat:
        if show_features:
            st.dataframe(df_feat.tail(50).sort_index(ascending=False), use_container_width=True)
        else:
            st.info("Enable Technical Indicators to view feature matrix.")
    with tab_forecast:
        if forecast_pred is not None:
            fdf = pd.DataFrame({
                "Date": forecast_dates,
                "Forecast": forecast_pred,
                "Change": np.diff(forecast_pred, prepend=float(series.iloc[-1])),
            })
            fdf["Change %"] = fdf["Change"] / fdf["Forecast"].shift(1) * 100
            st.dataframe(fdf.set_index("Date"), use_container_width=True)

# ─── Footer ──────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#2d4a6a; font-size:0.8rem; padding:16px 0;">
  <b style="color:#00d4ff;">Sammy Quant Lab</b> · Built with Streamlit + PyTorch + Statsmodels ·
  Data via Yahoo Finance · For educational & portfolio purposes only.<br>
  <i>Not financial advice.</i>
</div>
""", unsafe_allow_html=True)
