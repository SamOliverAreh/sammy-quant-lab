"""
Sammy Quant Lab — CLI runner for batch analysis & model evaluation.

Usage:
    python main.py --ticker USDMYR --model arima --steps 30
    python main.py --ticker EURUSD --model lstm --steps 14 --epochs 50
    python main.py --ticker GBPUSD --model hybrid --steps 30

Launch dashboard:
    streamlit run dashboard/app.py
"""

import argparse
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

from data_pipeline.ingestion import fetch_data
from data_pipeline.preprocessing import preprocess
from data_pipeline.features import create_features
from models.statistical.arima import arima_forecast
from models.statistical.garch import garch_volatility
from models.ml.lstm import train_lstm, forecast_lstm
from models.hybrid.arima_lstm import hybrid_forecast
from evaluation.metrics import evaluate_all, sharpe_ratio, max_drawdown


def run_analysis(ticker="USDMYR", model="arima", steps=30, start="2021-01-01",
                 epochs=50, hidden=64, window=20):
    print(f"\n{'='*60}")
    print(f"  Sammy Quant Lab — {ticker} · {model.upper()} · {steps}d Forecast")
    print(f"{'='*60}\n")

    print("📡 Fetching data...")
    df = fetch_data(ticker, start=start)
    df = preprocess(df)
    df = create_features(df)
    series = df["Close"]

    print(f"   ✓ {len(series)} data points loaded ({series.index[0].date()} → {series.index[-1].date()})")
    print(f"   Current price: {series.iloc[-1]:.4f}")
    print(f"   30d Vol: {df['returns'].rolling(30).std().iloc[-1]*100:.2f}%")
    print(f"   Sharpe: {sharpe_ratio(df['returns'].dropna()):.2f}")
    print(f"   Max DD: {max_drawdown(series.values)*100:.2f}%\n")

    print(f"🤖 Running {model.upper()} model...")

    if model == "arima":
        pred, order, conf_int = arima_forecast(series, steps=steps)
        print(f"   ✓ Best ARIMA order: {order}")

    elif model == "lstm":
        lstm_model, scaler, losses = train_lstm(series, epochs=epochs, hidden=hidden, window=window)
        pred = forecast_lstm(lstm_model, series, scaler, steps=steps, window=window)
        print(f"   ✓ Training complete · Final loss: {losses[-1]:.6f}")

    elif model == "hybrid":
        pred, arima_part, lstm_part, order = hybrid_forecast(series, steps=steps)
        print(f"   ✓ Hybrid complete · ARIMA{order} + LSTM residuals")

    else:
        raise ValueError(f"Unknown model: {model}. Choose arima/lstm/hybrid")

    # Backtest
    print("\n📐 Backtesting...")
    test_window = min(60, len(series) - steps - 10)
    test_series = series.iloc[-test_window - steps: -steps]
    test_true = series.iloc[-steps:]

    if model == "arima":
        bt_pred, _, _ = arima_forecast(test_series, steps=steps)
    elif model == "lstm":
        bm, bs, _ = train_lstm(test_series, epochs=epochs, hidden=hidden, window=window)
        bt_pred = forecast_lstm(bm, test_series, bs, steps=steps, window=window)
    else:
        bt_pred, _, _, _ = hybrid_forecast(test_series, steps=steps)

    trim = min(len(test_true), len(bt_pred))
    metrics = evaluate_all(test_true.values[:trim], bt_pred[:trim])

    print("\n📊 Evaluation Metrics:")
    for k, v in metrics.items():
        print(f"   {k}: {v:.4f}")

    print("\n🔮 Forecast (next {} days):".format(steps))
    forecast_dates = pd.bdate_range(
        start=series.index[-1] + pd.Timedelta(days=1), periods=steps
    )
    for d, p in zip(forecast_dates[:10], pred[:10]):
        chg = (p - float(series.iloc[-1])) / float(series.iloc[-1]) * 100
        print(f"   {d.date()}: {p:.4f}  ({chg:+.3f}%)")
    if steps > 10:
        print(f"   ... ({steps - 10} more)")

    print(f"\n{'='*60}\n")
    return pred, metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sammy Quant Lab CLI")
    parser.add_argument("--ticker", default="USDMYR")
    parser.add_argument("--model", default="arima", choices=["arima", "lstm", "hybrid"])
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--start", default="2021-01-01")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--window", type=int, default=20)

    args = parser.parse_args()
    run_analysis(
        ticker=args.ticker, model=args.model, steps=args.steps,
        start=args.start, epochs=args.epochs, hidden=args.hidden, window=args.window,
    )
