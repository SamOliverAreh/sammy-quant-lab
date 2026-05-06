import yfinance as yf
import pandas as pd


TICKERS = {
    "USDMYR": "USDMYR=X",
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "USDJPY": "USDJPY=X",
    "AUDUSD": "AUDUSD=X",
}


def fetch_data(ticker_key="USDMYR", start="2020-01-01", end=None):
    symbol = TICKERS.get(ticker_key, ticker_key)
    df = yf.download(symbol, start=start, end=end, progress=False)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.dropna(inplace=True)
    return df


def get_available_tickers():
    return list(TICKERS.keys())
