"""
backtest/engine.py — 5-year buy-and-hold backtest with annual rebalancing.

Strategy
--------
• At the start of each year, rebalance to target weights.
• Between rebalances, let weights drift with price moves.
• No transaction costs or slippage (educational simplification).
• Benchmark: NIFTY 50 index (^NSEI), buy-and-hold.

Metrics returned
----------------
  CAGR, Annualised Volatility, Sharpe Ratio, Sortino Ratio,
  Max Drawdown, Calmar Ratio, Beta, Alpha, Information Ratio,
  Win Rate (% months portfolio beats benchmark)
"""

import os
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RISK_FREE_RATE

warnings.filterwarnings("ignore")


# ── Helpers ────────────────────────────────────────────────────────────────

def _close(df: pd.DataFrame) -> pd.Series:
    if isinstance(df.columns, pd.MultiIndex):
        return df["Close"].squeeze()
    return df["Close"] if "Close" in df.columns else df.iloc[:, 3]


def _align_prices(tickers: list[str], prices: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Build a wide DataFrame of daily Close prices, forward-fill missing days."""
    series = {}
    for t in tickers:
        if t in prices:
            s = _close(prices[t])
            s.index = pd.to_datetime(s.index)
            series[t] = s
    df = pd.DataFrame(series)
    df = df.ffill().dropna(how="all")
    # Drop rows where ANY stock is missing at the very start
    df = df.dropna()
    return df


def _portfolio_returns(price_df: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    """
    Simulate annual-rebalanced buy-and-hold portfolio.
    Returns daily portfolio returns series.
    """
    tickers = [t for t in weights if t in price_df.columns]
    w = np.array([weights[t] for t in tickers])
    w = w / w.sum()

    prices_sub = price_df[tickers]
    daily_ret  = prices_sub.pct_change().fillna(0)

    # Annual rebalancing: reset weights at the start of each calendar year
    years      = sorted(prices_sub.index.year.unique())
    port_rets  = []

    for yr in years:
        yr_mask = prices_sub.index.year == yr
        yr_ret  = daily_ret.loc[yr_mask]
        # Weighted sum of daily returns within the year (constant weights — rebalanced at start)
        yr_port = yr_ret.values @ w
        port_rets.append(pd.Series(yr_port, index=yr_ret.index))

    return pd.concat(port_rets).sort_index()


def _nav(returns: pd.Series, start: float = 100.0) -> pd.Series:
    """Convert daily return series → NAV (starting at `start`)."""
    return (1 + returns).cumprod() * start


def _metrics(port_ret: pd.Series, bench_ret: pd.Series, rf: float = RISK_FREE_RATE) -> dict:
    """Compute all performance metrics."""
    n_years = len(port_ret) / 252

    # CAGR
    total_ret = (1 + port_ret).prod() - 1
    cagr      = (1 + total_ret) ** (1 / max(n_years, 0.01)) - 1

    # Volatility
    ann_vol = port_ret.std() * np.sqrt(252)

    # Sharpe
    sharpe = (port_ret.mean() * 252 - rf) / ann_vol if ann_vol > 0 else np.nan

    # Sortino (downside vol only)
    down_ret  = port_ret[port_ret < 0]
    down_vol  = down_ret.std() * np.sqrt(252) if len(down_ret) > 0 else np.nan
    sortino   = (port_ret.mean() * 252 - rf) / down_vol if down_vol else np.nan

    # Max Drawdown
    nav_series   = _nav(port_ret)
    rolling_max  = nav_series.cummax()
    drawdown_ser = (nav_series - rolling_max) / rolling_max
    max_dd       = drawdown_ser.min()

    # Calmar
    calmar = cagr / abs(max_dd) if max_dd != 0 else np.nan

    # Beta / Alpha vs benchmark
    common   = port_ret.index.intersection(bench_ret.index)
    p, b     = port_ret.loc[common], bench_ret.loc[common]
    cov_mat  = np.cov(p, b)
    beta     = cov_mat[0, 1] / cov_mat[1, 1] if cov_mat[1, 1] > 0 else np.nan
    alpha    = (port_ret.mean() - beta * bench_ret.mean()) * 252 if beta else np.nan

    # Information Ratio
    active   = p - b
    ir       = (active.mean() * 252) / (active.std() * np.sqrt(252)) if active.std() > 0 else np.nan

    # Win rate (monthly)
    pm = port_ret.resample("ME").sum()
    bm = bench_ret.resample("ME").sum()
    cm = pm.index.intersection(bm.index)
    win_rate = float((pm.loc[cm] > bm.loc[cm]).mean()) if len(cm) > 0 else np.nan

    return {
        "CAGR (%)":           round(cagr * 100, 2),
        "Annualised Vol (%)": round(ann_vol * 100, 2),
        "Sharpe Ratio":       round(sharpe, 3),
        "Sortino Ratio":      round(sortino, 3),
        "Max Drawdown (%)":   round(max_dd * 100, 2),
        "Calmar Ratio":       round(calmar, 3),
        "Beta":               round(beta, 3),
        "Alpha (%)":          round(alpha * 100, 2),
        "Info Ratio":         round(ir, 3),
        "Win Rate (%)":       round(win_rate * 100, 1),
    }


# ── Main backtest function ─────────────────────────────────────────────────

def run_backtest(
    portfolio:     dict,
    prices:        dict[str, pd.DataFrame],
    benchmark_df:  pd.DataFrame,
) -> dict:
    """
    Run the full backtest.

    Parameters
    ----------
    portfolio : dict from portfolio.constructor.build_portfolio()
    prices    : {ticker: DataFrame} from data.fetcher
    benchmark_df : DataFrame for NIFTY 50 index

    Returns
    -------
    results : dict with:
        metrics       : dict of performance metrics
        portfolio_nav : pd.Series (daily NAV)
        benchmark_nav : pd.Series (daily NAV)
        drawdown      : pd.Series (daily drawdown)
        monthly_ret   : pd.DataFrame (monthly returns for heatmap)
        figures       : dict of plotly Figure objects
    """
    tickers = portfolio["tickers"]
    weights = portfolio["weights"]

    # Align price data
    price_df = _align_prices(tickers, prices)
    if price_df.empty or len(price_df) < 252:
        raise ValueError("Insufficient price data for backtest.")

    # Benchmark
    bench_close = _close(benchmark_df)
    bench_close.index = pd.to_datetime(bench_close.index)

    # Align on common dates
    common_idx  = price_df.index.intersection(bench_close.index)
    price_df    = price_df.loc[common_idx]
    bench_close = bench_close.loc[common_idx]

    # Returns
    port_ret  = _portfolio_returns(price_df, weights)
    bench_ret = bench_close.pct_change().fillna(0).loc[port_ret.index]

    # NAV
    port_nav  = _nav(port_ret)
    bench_nav = _nav(bench_ret)

    # Drawdown
    roll_max = port_nav.cummax()
    drawdown = (port_nav - roll_max) / roll_max

    # Metrics
    metrics = _metrics(port_ret, bench_ret)

    # Monthly returns for heatmap
    monthly = port_ret.resample("ME").sum()
    monthly_df = pd.DataFrame({
        "Year":  monthly.index.year,
        "Month": monthly.index.month,
        "Return": monthly.values * 100,
    })

    figures = build_figures(port_nav, bench_nav, drawdown, port_ret, bench_ret, monthly_df, portfolio)

    return {
        "metrics":       metrics,
        "portfolio_nav": port_nav,
        "benchmark_nav": bench_nav,
        "drawdown":      drawdown,
        "port_ret":      port_ret,
        "bench_ret":     bench_ret,
        "monthly_ret":   monthly_df,
        "figures":       figures,
    }


# ── Plotly Figures ─────────────────────────────────────────────────────────

DARK_BG   = "#0e1117"
GRID_COL  = "#1e2130"
ACCENT1   = "#00d4aa"   # teal — portfolio
ACCENT2   = "#ff6b35"   # orange — benchmark
ACCENT3   = "#4c9be8"   # blue — auxiliary


def _layout(title: str, fig: go.Figure) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, font=dict(color="#ffffff", size=16)),
        paper_bgcolor=DARK_BG,
        plot_bgcolor=DARK_BG,
        font=dict(color="#aaaaaa"),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#333"),
        xaxis=dict(gridcolor=GRID_COL, zerolinecolor=GRID_COL),
        yaxis=dict(gridcolor=GRID_COL, zerolinecolor=GRID_COL),
        margin=dict(l=50, r=30, t=60, b=40),
    )
    return fig


def build_figures(
    port_nav, bench_nav, drawdown, port_ret, bench_ret, monthly_df, portfolio
) -> dict:
    figs = {}

    # ── 1. NAV Comparison ─────────────────────────────────────────────────
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=port_nav.index, y=port_nav.values,
        name="Portfolio", line=dict(color=ACCENT1, width=2.5),
        fill="tozeroy", fillcolor="rgba(0,212,170,0.07)"
    ))
    fig1.add_trace(go.Scatter(
        x=bench_nav.index, y=bench_nav.values,
        name="NIFTY 50", line=dict(color=ACCENT2, width=2, dash="dash"),
    ))
    figs["nav"] = _layout("Portfolio vs NIFTY 50 — NAV (Base = ₹100)", fig1)

    # ── 2. Drawdown ────────────────────────────────────────────────────────
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=drawdown.index, y=(drawdown.values * 100),
        name="Drawdown (%)", fill="tozeroy",
        line=dict(color="#e84393", width=1.5),
        fillcolor="rgba(232,67,147,0.15)"
    ))
    figs["drawdown"] = _layout("Portfolio Drawdown (%)", fig2)

    # ── 3. Rolling Sharpe (252-day) ────────────────────────────────────────
    roll_sharpe = (port_ret.rolling(252).mean() * 252 - RISK_FREE_RATE) / \
                  (port_ret.rolling(252).std() * np.sqrt(252))
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=roll_sharpe.index, y=roll_sharpe.values,
        name="Rolling Sharpe", line=dict(color=ACCENT3, width=2)
    ))
    fig3.add_hline(y=1.0, line_dash="dot", line_color="#666", annotation_text="Sharpe = 1")
    figs["rolling_sharpe"] = _layout("Rolling 1-Year Sharpe Ratio", fig3)

    # ── 4. Monthly Return Heatmap ──────────────────────────────────────────
    pivot = monthly_df.pivot(index="Year", columns="Month", values="Return").fillna(0)
    month_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    pivot.columns = [month_names[m-1] for m in pivot.columns]

    fig4 = go.Figure(go.Heatmap(
        z=pivot.values,
        x=pivot.columns.tolist(),
        y=[str(y) for y in pivot.index.tolist()],
        colorscale=[[0,"#c0392b"],[0.5,"#1a1a2e"],[1,"#27ae60"]],
        zmid=0,
        text=np.round(pivot.values, 1),
        texttemplate="%{text}%",
        colorbar=dict(title="Return %", tickfont=dict(color="#aaa")),
    ))
    figs["heatmap"] = _layout("Monthly Return Heatmap (%)", fig4)

    # ── 5. Return Distribution ────────────────────────────────────────────
    fig5 = go.Figure()
    fig5.add_trace(go.Histogram(
        x=port_ret.values * 100, name="Portfolio",
        nbinsx=60, marker_color=ACCENT1, opacity=0.75
    ))
    fig5.add_trace(go.Histogram(
        x=bench_ret.values * 100, name="NIFTY 50",
        nbinsx=60, marker_color=ACCENT2, opacity=0.65
    ))
    fig5.update_layout(barmode="overlay")
    figs["distribution"] = _layout("Daily Return Distribution (%)", fig5)

    # ── 6. Portfolio Weights Pie ───────────────────────────────────────────
    w = portfolio["weights"]
    labels = [t.replace(".NS", "") for t in w.keys()]
    values = list(w.values())
    fig6 = go.Figure(go.Pie(
        labels=labels, values=values,
        hole=0.45,
        marker=dict(colors=px.colors.qualitative.Dark24),
        textfont=dict(color="#ffffff"),
    ))
    fig6.update_layout(
        title=dict(text="Portfolio Weights", font=dict(color="#ffffff", size=16)),
        paper_bgcolor=DARK_BG,
        font=dict(color="#aaaaaa"),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=20, r=20, t=60, b=20),
    )
    figs["weights_pie"] = fig6

    return figs
