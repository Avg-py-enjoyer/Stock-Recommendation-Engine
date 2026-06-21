"""
data/fetcher.py — Download and cache NIFTY 50 price + fundamental data via yfinance.
"""

import os
import pickle
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import NIFTY50_TICKERS, BENCHMARK_TICKER, BACKTEST_YEARS, DATA_CACHE_PATH


def _date_range(years: int = BACKTEST_YEARS):
    end   = datetime.today()
    start = end - timedelta(days=365 * years + 30)   # a little extra buffer
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


def fetch_price_data(tickers: list[str], start: str, end: str) -> dict[str, pd.DataFrame]:
    """Download daily OHLCV for a list of tickers. Returns {ticker: DataFrame}."""
    print(f"[fetcher] Downloading price data for {len(tickers)} tickers …")
    price_data = {}
    for ticker in tickers:
        try:
            df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
            if df.empty:
                print(f"  ⚠  No data for {ticker}")
            else:
                price_data[ticker] = df
        except Exception as e:
            print(f"  ✗  {ticker}: {e}")
    print(f"[fetcher] Got data for {len(price_data)} / {len(tickers)} tickers.")
    return price_data


def fetch_fundamentals(tickers: list[str]) -> dict[str, dict]:
    """
    Pull fundamental info via yfinance .info dict.
    Gracefully handles missing keys — returns NaN for missing values.
    """
    print("[fetcher] Fetching fundamentals (this may take a moment) …")
    KEYS = [
        "trailingPE", "forwardPE", "priceToBook", "dividendYield",
        "returnOnEquity", "returnOnAssets", "debtToEquity",
        "revenueGrowth", "earningsGrowth", "currentRatio",
        "marketCap", "beta",
    ]
    fundamentals = {}
    for ticker in tickers:
        try:
            info = yf.Ticker(ticker).info
            fundamentals[ticker] = {k: info.get(k, np.nan) for k in KEYS}
        except Exception as e:
            print(f"  ✗  {ticker}: {e}")
            fundamentals[ticker] = {k: np.nan for k in KEYS}
    return fundamentals


def load_or_fetch(force_refresh: bool = False) -> dict:
    """
    Main entry point. Returns a dict:
        {
          "prices":       {ticker: DataFrame},
          "benchmark":    DataFrame,
          "fundamentals": {ticker: dict},
          "start":        str,
          "end":          str,
        }
    Uses a pickle cache so we don't re-download every run.
    """
    if not force_refresh and os.path.exists(DATA_CACHE_PATH):
        print(f"[fetcher] Loading cached data from {DATA_CACHE_PATH}")
        with open(DATA_CACHE_PATH, "rb") as f:
            return pickle.load(f)

    start, end = _date_range(BACKTEST_YEARS)
    all_tickers = NIFTY50_TICKERS + [BENCHMARK_TICKER]

    prices      = fetch_price_data(NIFTY50_TICKERS, start, end)
    benchmark   = yf.download(BENCHMARK_TICKER, start=start, end=end, progress=False, auto_adjust=True)
    fundamentals = fetch_fundamentals(NIFTY50_TICKERS)

    data = {
        "prices":       prices,
        "benchmark":    benchmark,
        "fundamentals": fundamentals,
        "start":        start,
        "end":          end,
    }

    with open(DATA_CACHE_PATH, "wb") as f:
        pickle.dump(data, f)
    print(f"[fetcher] Cached data saved to {DATA_CACHE_PATH}")
    return data


if __name__ == "__main__":
    data = load_or_fetch(force_refresh=True)
    print("Tickers loaded:", list(data["prices"].keys())[:5])
