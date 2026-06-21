"""
portfolio/constructor.py — Build portfolio weights from recommended stocks.

Weighting schemes available:
  1. Equal Weight          — 1/N to each stock (simple baseline)
  2. Similarity Weight     — proportional to cosine similarity score
  3. Inverse Volatility    — weight inversely proportional to annualised vol
                             (risk-parity inspired, suitable for low-risk prefs)

All weights sum to 1.0.
"""

import os
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _close(df: pd.DataFrame) -> pd.Series:
    if isinstance(df.columns, pd.MultiIndex):
        return df["Close"].squeeze()
    return df["Close"] if "Close" in df.columns else df.iloc[:, 3]


def equal_weight(tickers: list[str]) -> dict[str, float]:
    n = len(tickers)
    return {t: 1.0 / n for t in tickers}


def similarity_weight(recommendations: pd.DataFrame) -> dict[str, float]:
    """Weight proportional to cosine similarity (all sims > 0 after softmax-like norm)."""
    tickers = recommendations["ticker"].tolist()
    sims    = recommendations["similarity"].values
    # Shift to ensure all positive, then normalise
    sims    = sims - sims.min() + 1e-8
    weights = sims / sims.sum()
    return dict(zip(tickers, weights))


def inverse_vol_weight(
    tickers: list[str],
    prices:  dict[str, pd.DataFrame],
    lookback_days: int = 252,
) -> dict[str, float]:
    """Risk-parity: weight inversely proportional to annualised volatility."""
    inv_vols = {}
    for ticker in tickers:
        if ticker not in prices:
            inv_vols[ticker] = 1.0
            continue
        close   = _close(prices[ticker]).iloc[-lookback_days:]
        returns = close.pct_change().dropna()
        vol     = returns.std() * np.sqrt(252)
        inv_vols[ticker] = 1.0 / max(vol, 1e-6)

    total  = sum(inv_vols.values())
    return {t: v / total for t, v in inv_vols.items()}


def build_portfolio(
    recommendations: pd.DataFrame,
    prices:          dict[str, pd.DataFrame],
    weighting:       str = "equal",   # "equal" | "similarity" | "inv_vol"
) -> dict:
    """
    Returns
    -------
    portfolio : dict with keys:
        tickers   : list[str]
        weights   : dict[str, float]
        weighting : str
    """
    tickers = recommendations["ticker"].tolist()

    if weighting == "similarity":
        weights = similarity_weight(recommendations)
    elif weighting == "inv_vol":
        weights = inverse_vol_weight(tickers, prices)
    else:
        weights = equal_weight(tickers)

    return {
        "tickers":   tickers,
        "weights":   weights,
        "weighting": weighting,
    }
