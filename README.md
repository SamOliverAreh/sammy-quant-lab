# ⚡ Sammy Quant Lab

> **Multi-Model FX Forecasting Engine** · Statistical + Deep Learning + Hybrid Models

[![CI](https://github.com/SamOliverAreh/sammy-quant-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/SamOliverAreh/sammy-quant-lab/actions)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue)](https://python.org)
[![Streamlit](https://img.shields.io/badge/dashboard-Streamlit-FF4B4B)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A production-grade quantitative finance platform for **FX time series forecasting**, combining classical econometrics with modern deep learning. Built as a portfolio project demonstrating end-to-end data science skills in financial markets.

🌐 **[Live Overview](https://samoliverareh.github.io/sammy-quant-lab/)**
🌐 **[Live Dashboard](https://sammy-quant-lab.streamlit.app/)**

---

## 🎯 What This Project Demonstrates

| Skill Area | What's Implemented |
|---|---|
| **Data Engineering** | Real-time market data pipeline · Feature engineering · OHLCV processing |
| **Statistical Modelling** | Auto-ARIMA order selection · GARCH(1,1) volatility modelling |
| **Deep Learning** | PyTorch LSTM with dropout, LR scheduling, gradient clipping |
| **Hybrid Models** | ARIMA-LSTM: linear + nonlinear residual decomposition |
| **Evaluation** | RMSE, MAE, MAPE, Direction Accuracy, Sharpe Ratio, Max Drawdown |
| **MLOps** | CI/CD via GitHub Actions · Streamlit Cloud deployment |
| **Data Viz** | Interactive Plotly dashboards · Bollinger Bands · Correlation matrices |

---

## 📐 Architecture

```
sammy-quant-lab/
│
├── data_pipeline/
│   ├── ingestion.py       # yfinance data fetching (5 FX pairs)
│   ├── preprocessing.py   # Returns, log-returns computation
│   └── features.py        # MA, RSI, Bollinger Bands, MACD, volatility
│
├── models/
│   ├── statistical/
│   │   ├── arima.py       # Auto-ARIMA with AIC-based order selection
│   │   └── garch.py       # GARCH(1,1) volatility forecasting
│   ├── ml/
│   │   └── lstm.py        # PyTorch LSTM (multi-layer, dropout, scheduler)
│   └── hybrid/
│       └── arima_lstm.py  # Hybrid: ARIMA linear baseline + LSTM residuals
│
├── evaluation/
│   └── metrics.py         # RMSE, MAE, MAPE, Direction Acc, Sharpe, MDD
│
├── dashboard/
│   └── app.py             # Streamlit dashboard (full interactive UI)
│
├── .github/workflows/
│   └── ci.yml             # GitHub Actions CI pipeline
│
├── main.py                # CLI runner for batch analysis
└── requirements.txt
```

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/YOUR_USERNAME/sammy-quant-lab.git
cd sammy-quant-lab

python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Run the Dashboard

```bash
streamlit run dashboard/app.py
```

Open **http://localhost:8501** in your browser.

### 3. CLI Usage

```bash
# ARIMA forecast — USDMYR, 30 days
python main.py --ticker USDMYR --model arima --steps 30

# LSTM forecast — EURUSD, 14 days, 100 epochs
python main.py --ticker EURUSD --model lstm --steps 14 --epochs 100

# Hybrid forecast — GBPUSD
python main.py --ticker GBPUSD --model hybrid --steps 30
```

---

## 📊 Models

### ARIMA (Auto-Regressive Integrated Moving Average)
- **Auto order selection** via AIC minimisation across all (p,d,q) combinations up to (3,2,3)
- Returns 95% confidence intervals for forecast uncertainty
- Best for: stationary/near-stationary FX series, short-term forecasts

### LSTM (Long Short-Term Memory)
- **PyTorch** implementation · 2-layer stacked LSTM · hidden units: 32/64/128
- Gradient clipping · StepLR scheduler · configurable look-back window
- Best for: capturing nonlinear patterns, medium-term forecasts

### Hybrid ARIMA-LSTM
- Step 1: Fit ARIMA to capture linear structure
- Step 2: Extract in-sample residuals (unexplained component)
- Step 3: Train LSTM on residuals (learn nonlinear patterns)
- Step 4: Final forecast = ARIMA prediction + LSTM residual prediction
- Best for: comprehensive modelling of both linear and nonlinear dynamics

### GARCH(1,1)
- Volatility clustering modelling
- 10-day forward volatility forecast
- Used for risk assessment alongside price forecasts

---

## 📈 Supported Currency Pairs

| Ticker | Pair | Category |
|--------|------|----------|
| USDMYR | US Dollar / Malaysian Ringgit | Emerging Market |
| EURUSD | Euro / US Dollar | G10 Major |
| GBPUSD | British Pound / US Dollar | G10 Major |
| USDJPY | US Dollar / Japanese Yen | G10 Major |
| AUDUSD | Australian Dollar / US Dollar | G10 Major |

---

## 🌐 Deploy to Streamlit Cloud

1. Fork this repository
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**
3. Set:
   - **Repository**: `YOUR_USERNAME/sammy-quant-lab`
   - **Branch**: `main`
   - **Main file path**: `dashboard/app.py`
4. Click **Deploy**

That's it — Streamlit Cloud handles dependencies automatically via `requirements.txt`.

---

## 📋 Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **RMSE** | Root Mean Squared Error — penalises large errors |
| **MAE** | Mean Absolute Error — average absolute deviation |
| **MAPE** | Mean Absolute Percentage Error — relative accuracy |
| **Direction Accuracy** | % of correct up/down movement predictions |
| **Sharpe Ratio** | Risk-adjusted return (annualised, 252 trading days) |
| **Max Drawdown** | Largest peak-to-trough price decline |

---

## 🛠️ Tech Stack

- **Data**: `yfinance` · `pandas` · `numpy`
- **Statistical Models**: `statsmodels` · `arch`
- **Deep Learning**: `PyTorch` · `scikit-learn`
- **Dashboard**: `Streamlit` · `Plotly`
- **CI/CD**: GitHub Actions
- **Deployment**: Streamlit Community Cloud

---

## ⚠️ Disclaimer

This project is for **educational and portfolio purposes only**. Nothing here constitutes financial advice. Always do your own research before making investment decisions.

---

## 👤 About

**Sammy** — Data Scientist & ML Engineer specialising in:
- Financial time series analysis
- Quantitative modelling (ARIMA, GARCH, LSTM)
- Data pipeline engineering
- ML model deployment

[![GitHub](https://img.shields.io/badge/GitHub-YOUR_USERNAME-181717?logo=github)](https://github.com/SamOliverAreh)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0077B5?logo=linkedin)](https://www.linkedin.com/in/sam-oliver-areh/)

---

*Built with ⚡ and lots of coffee.*
