""
Data ingestion — fetches OHLCV data from yfinance.
Covers: FX, Commodities, Indices, Stocks (Global + MY), Crypto.
"""
import yfinance as yf
import pandas as pd

# ── Asset Universe ──────────────────────────────────────────────────────────────

TICKERS = {
    # ── Forex ──────────────────────────────────────────────────────────────────
    "FX": {
        "USDMYR": "USDMYR=X",
        "EURUSD": "EURUSD=X",
        "GBPUSD": "GBPUSD=X",
        "USDJPY": "USDJPY=X",
        "AUDUSD": "AUDUSD=X",
        "USDSGD": "USDSGD=X",
        "USDCNY": "USDCNY=X",
        "USDINR": "USDINR=X",
    },

    # ── Commodities ────────────────────────────────────────────────────────────
    "Commodities": {
        "Gold":        "GC=F",
        "Silver":      "SI=F",
        "Crude Oil":   "CL=F",
        "Brent Crude": "BZ=F",
        "Natural Gas": "NG=F",
        "Palm Oil":    "KE=F",   # Bursa CPO futures
        "Copper":      "HG=F",
        "Platinum":    "PL=F",
        "Rubber":      "0001.KL",  # proxy — Bursa
        "Tin":         "TIN=F",
    },

    # ── Global Indices ─────────────────────────────────────────────────────────
    "Global Indices": {
        "S&P 500":      "^GSPC",
        "Nasdaq 100":   "^NDX",
        "Dow Jones":    "^DJI",
        "FTSE 100":     "^FTSE",
        "Nikkei 225":   "^N225",
        "Hang Seng":    "^HSI",
        "Shanghai":     "000001.SS",
        "DAX":          "^GDAXI",
        "KLCI (MY)":    "^KLSE",
        "STI (SG)":     "^STI",
        "SET (TH)":     "^SET.BK",
    },

    # ── Global Stocks ──────────────────────────────────────────────────────────
    "Global Stocks": {
        "Apple":      "AAPL",
        "Microsoft":  "MSFT",
        "NVIDIA":     "NVDA",
        "Alphabet":   "GOOGL",
        "Amazon":     "AMZN",
        "Tesla":      "TSLA",
        "Meta":       "META",
        "Berkshire":  "BRK-B",
    },

    # ── Malaysia Stocks (Bursa) ────────────────────────────────────────────────
    "MY Stocks": {
        "Maybank":       "1155.KL",
        "Public Bank":   "1295.KL",
        "Tenaga":        "5347.KL",
        "CIMB":          "1023.KL",
        "IHH Healthcare":"5225.KL",
        "Petronas Gas":  "6033.KL",
        "Top Glove":     "7113.KL",
        "Axiata":        "6888.KL",
        "Press Metal":   "8869.KL",
        "Hartalega":     "5168.KL",
    },

    # ── Cryptocurrency ─────────────────────────────────────────────────────────
    "Crypto": {
        "Bitcoin":  "BTC-USD",
        "Ethereum": "ETH-USD",
        "BNB":      "BNB-USD",
        "Solana":   "SOL-USD",
        "XRP":      "XRP-USD",
        "Cardano":  "ADA-USD",
        "Dogecoin": "DOGE-USD",
        "Polkadot": "DOT-USD",
    },
}

# Flat map for quick lookup: display_name → yfinance symbol
_FLAT: dict[str, str] = {}
_CATEGORY_MAP: dict[str, str] = {}
for _cat, _pairs in TICKERS.items():
    for _name, _sym in _pairs.items():
        _FLAT[_name] = _sym
        _CATEGORY_MAP[_name] = _cat


def fetch_data(ticker_key: str, start: str = "2020-01-01", end=None) -> pd.DataFrame:
    """Download OHLCV from yfinance; ticker_key is the display name."""
    symbol = _FLAT.get(ticker_key, ticker_key)
    df = yf.download(symbol, start=start, end=end, progress=False, auto_adjust=True)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    df = df[cols].copy()

    # Volume may be 0/NaN for FX — fill with 0
    if "Volume" not in df.columns:
        df["Volume"] = 0
    df["Volume"] = df["Volume"].fillna(0)

    df.dropna(subset=["Close"], inplace=True)
    return df


def get_categories() -> dict[str, list[str]]:
    """Return {category: [display_names]} for UI dropdowns."""
    return {cat: list(pairs.keys()) for cat, pairs in TICKERS.items()}


def get_all_names() -> list[str]:
    return list(_FLAT.keys())


def get_category(name: str) -> str:
    return _CATEGORY_MAP.get(name, "Unknown")
