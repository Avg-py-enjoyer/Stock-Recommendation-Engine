"""
features/engineer.py — Compute a rich feature vector for each NIFTY 50 stock.

Feature Groups
--------------
1. Return-based     : 1M / 3M / 6M / 12M momentum returns
2. Risk-based       : annualised volatility, max drawdown, Sharpe ratio
3. Market-based     : beta vs NIFTY 50, correlation with benchmark
4. Fundamental      : P/E, P/B, dividend yield, ROE, ROA, D/E ratio,
                      revenue growth, earnings growth, current ratio
5. Sector dummies   : one-hot encoding of the sector
"""

import os
import pickle
import warnings

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    NIFTY50_TICKERS, SECTOR_MAP, SECTORS,
    FEATURES_PATH, RISK_FREE_RATE
)


# ── helpers ────────────────────────────────────────────────────────────────

def _close(df: pd.DataFrame) -> pd.Series:
    """Safely extract Close series, handling MultiIndex columns from yfinance."""
    if isinstance(df.columns, pd.MultiIndex):
        return df["Close"].squeeze()
    if "Close" in df.columns:
        return df["Close"]
    return df.iloc[:, 3]   # fallback: 4th column is usually close


def _momentum(close: pd.Series, days: int) -> float:
    """Simple price return over last `days` trading days."""
    if len(close) < days + 1:
        return np.nan
    return float(close.iloc[-1] / close.iloc[-days] - 1)


def _annualised_vol(close: pd.Series) -> float:
    returns = close.pct_change().dropna()
    return float(returns.std() * np.sqrt(252))


def _max_drawdown(close: pd.Series) -> float:
    roll_max = close.cummax()
    drawdown = (close - roll_max) / roll_max
    return float(drawdown.min())


def _sharpe(close: pd.Series, rf: float = RISK_FREE_RATE) -> float:
    returns = close.pct_change().dropna()
    excess  = returns.mean() * 252 - rf
    vol     = returns.std() * np.sqrt(252)
    return float(excess / vol) if vol > 0 else np.nan


def _beta_corr(stock_close: pd.Series, bench_close: pd.Series):
    """Return (beta, correlation) against the benchmark."""
    s = stock_close.pct_change().dropna()
    b = bench_close.pct_change().dropna()
    common = s.index.intersection(b.index)
    if len(common) < 30:
        return np.nan, np.nan
    s, b = s.loc[common], b.loc[common]
    cov  = np.cov(s, b)
    beta = cov[0, 1] / cov[1, 1] if cov[1, 1] > 0 else np.nan
    corr = np.corrcoef(s, b)[0, 1]
    return float(beta), float(corr)


# ── main ───────────────────────────────────────────────────────────────────

def build_features(
    prices: dict[str, pd.DataFrame],
    fundamentals: dict[str, dict],
    benchmark_df: pd.DataFrame,
) -> tuple[pd.DataFrame, StandardScaler]:
    """
    Returns
    -------
    features_df : pd.DataFrame  (rows = tickers, cols = feature names), raw values
    scaler      : fitted StandardScaler for later use
    """
    bench_close = _close(benchmark_df)

    records = []
    for ticker in NIFTY50_TICKERS:
        if ticker not in prices:
            continue
        df    = prices[ticker]
        close = _close(df)

        beta, corr = _beta_corr(close, bench_close)
        fund = fundamentals.get(ticker, {})

        sector      = SECTOR_MAP.get(ticker, "Other")
        sector_ohe  = {f"sector_{s}": int(sector == s) for s in SECTORS}

        row = {
            "ticker": ticker,
            # momentum
            "mom_1m":  _momentum(close, 21),
            "mom_3m":  _momentum(close, 63),
            "mom_6m":  _momentum(close, 126),
            "mom_12m": _momentum(close, 252),
            # risk
            "volatility":    _annualised_vol(close),
            "max_drawdown":  _max_drawdown(close),
            "sharpe":        _sharpe(close),
            # market
            "beta": beta,
            "benchmark_corr": corr,
            # fundamentals
            "pe_ratio":        fund.get("trailingPE", np.nan),
            "pb_ratio":        fund.get("priceToBook", np.nan),
            "dividend_yield":  fund.get("dividendYield", np.nan),
            "roe":             fund.get("returnOnEquity", np.nan),
            "roa":             fund.get("returnOnAssets", np.nan),
            "debt_to_equity":  fund.get("debtToEquity", np.nan),
            "revenue_growth":  fund.get("revenueGrowth", np.nan),
            "earnings_growth": fund.get("earningsGrowth", np.nan),
            "current_ratio":   fund.get("currentRatio", np.nan),
            **sector_ohe,
        }
        records.append(row)

    df_feat = pd.DataFrame(records).set_index("ticker")

    # Clip extreme outliers (winsorise at 1st–99th percentile per feature)
    numeric_cols = df_feat.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        lo, hi = df_feat[col].quantile([0.01, 0.99])
        df_feat[col] = df_feat[col].clip(lo, hi)

    # Fill remaining NaNs with column median
    df_feat[numeric_cols] = df_feat[numeric_cols].fillna(df_feat[numeric_cols].median())

    # Standardise
    scaler  = StandardScaler()
    scaled  = scaler.fit_transform(df_feat[numeric_cols])
    df_scaled = pd.DataFrame(scaled, index=df_feat.index, columns=numeric_cols)

    return df_scaled, scaler, df_feat   # scaled, scaler, raw


def load_or_build(raw_data: dict, force_rebuild: bool = False):
    """Cache the feature matrix to disk."""
    if not force_rebuild and os.path.exists(FEATURES_PATH):
        print(f"[features] Loading cached features from {FEATURES_PATH}")
        with open(FEATURES_PATH, "rb") as f:
            return pickle.load(f)

    print("[features] Building feature matrix …")
    scaled_df, scaler, raw_df = build_features(
        raw_data["prices"],
        raw_data["fundamentals"],
        raw_data["benchmark"],
    )
    payload = {"scaled": scaled_df, "scaler": scaler, "raw": raw_df}
    with open(FEATURES_PATH, "wb") as f:
        pickle.dump(payload, f)
    print(f"[features] Features saved — shape: {scaled_df.shape}")
    return payload


if __name__ == "__main__":
    from data.fetcher import load_or_fetch
    data    = load_or_fetch()
    payload = load_or_build(data, force_rebuild=True)
    print(payload["scaled"].head())
