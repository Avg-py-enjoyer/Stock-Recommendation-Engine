"""
config.py — Global constants for the NIFTY 50 Recommender System
"""

import os

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT_DIR    = os.path.dirname(os.path.abspath(__file__))
OUTPUTS_DIR = os.path.join(ROOT_DIR, "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)

DATA_CACHE_PATH    = os.path.join(OUTPUTS_DIR, "nifty50_data.pkl")
FEATURES_PATH      = os.path.join(OUTPUTS_DIR, "features.pkl")
MODEL_PATH         = os.path.join(OUTPUTS_DIR, "autoencoder.pt")
EMBEDDINGS_PATH    = os.path.join(OUTPUTS_DIR, "embeddings.pkl")

# ── NIFTY 50 Universe (NSE tickers with .NS suffix for yfinance) ───────────
NIFTY50_TICKERS = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "BHARTIARTL.NS", "ICICIBANK.NS",
    "INFOSYS.NS", "SBIN.NS", "HINDUNILVR.NS", "ITC.NS", "LT.NS",
    "KOTAKBANK.NS", "AXISBANK.NS", "BAJFINANCE.NS", "MARUTI.NS", "SUNPHARMA.NS",
    "TITAN.NS", "ULTRACEMCO.NS", "ASIANPAINT.NS", "BAJAJFINSV.NS", "NTPC.NS",
    "POWERGRID.NS", "WIPRO.NS", "HCLTECH.NS", "TECHM.NS", "M&M.NS",
    "TATAMOTORS.NS", "TATASTEEL.NS", "ADANIENT.NS", "ADANIPORTS.NS", "COALINDIA.NS",
    "ONGC.NS", "NESTLEIND.NS", "DIVISLAB.NS", "DRREDDY.NS", "CIPLA.NS",
    "EICHERMOT.NS", "HEROMOTOCO.NS", "BRITANNIA.NS", "GRASIM.NS", "SHREECEM.NS",
    "JSWSTEEL.NS", "HINDALCO.NS", "INDUSINDBK.NS", "BPCL.NS", "HDFCLIFE.NS",
    "SBILIFE.NS", "APOLLOHOSP.NS", "BAJAJ-AUTO.NS", "TATACONSUM.NS", "UPL.NS",
]

BENCHMARK_TICKER = "^NSEI"   # NIFTY 50 index

# ── Data ───────────────────────────────────────────────────────────────────
BACKTEST_YEARS  = 5
RISK_FREE_RATE  = 0.065      # Approx Indian 10-yr G-Sec yield

# ── Model Hyperparameters ──────────────────────────────────────────────────
LATENT_DIM      = 8          # Dimension of the embedding space
HIDDEN_DIMS     = [64, 32]   # Encoder hidden layer sizes (decoder mirrors)
LEARNING_RATE   = 1e-3
EPOCHS          = 300
BATCH_SIZE      = 16         # Small — only ~50 stocks
DROPOUT         = 0.2

# ── Recommender ────────────────────────────────────────────────────────────
TOP_K           = 10         # Default number of stocks to recommend

# ── Sector mapping (approximate, for feature engineering) ─────────────────
SECTOR_MAP = {
    "RELIANCE.NS":   "Energy",
    "TCS.NS":        "IT",
    "HDFCBANK.NS":   "Banking",
    "BHARTIARTL.NS": "Telecom",
    "ICICIBANK.NS":  "Banking",
    "INFOSYS.NS":    "IT",
    "SBIN.NS":       "Banking",
    "HINDUNILVR.NS": "FMCG",
    "ITC.NS":        "FMCG",
    "LT.NS":         "Infrastructure",
    "KOTAKBANK.NS":  "Banking",
    "AXISBANK.NS":   "Banking",
    "BAJFINANCE.NS": "NBFC",
    "MARUTI.NS":     "Auto",
    "SUNPHARMA.NS":  "Pharma",
    "TITAN.NS":      "Consumer",
    "ULTRACEMCO.NS": "Cement",
    "ASIANPAINT.NS": "Consumer",
    "BAJAJFINSV.NS": "NBFC",
    "NTPC.NS":       "Power",
    "POWERGRID.NS":  "Power",
    "WIPRO.NS":      "IT",
    "HCLTECH.NS":    "IT",
    "TECHM.NS":      "IT",
    "M&M.NS":        "Auto",
    "TATAMOTORS.NS": "Auto",
    "TATASTEEL.NS":  "Metals",
    "ADANIENT.NS":   "Conglomerate",
    "ADANIPORTS.NS": "Infrastructure",
    "COALINDIA.NS":  "Energy",
    "ONGC.NS":       "Energy",
    "NESTLEIND.NS":  "FMCG",
    "DIVISLAB.NS":   "Pharma",
    "DRREDDY.NS":    "Pharma",
    "CIPLA.NS":      "Pharma",
    "EICHERMOT.NS":  "Auto",
    "HEROMOTOCO.NS": "Auto",
    "BRITANNIA.NS":  "FMCG",
    "GRASIM.NS":     "Diversified",
    "SHREECEM.NS":   "Cement",
    "JSWSTEEL.NS":   "Metals",
    "HINDALCO.NS":   "Metals",
    "INDUSINDBK.NS": "Banking",
    "BPCL.NS":       "Energy",
    "HDFCLIFE.NS":   "Insurance",
    "SBILIFE.NS":    "Insurance",
    "APOLLOHOSP.NS": "Healthcare",
    "BAJAJ-AUTO.NS": "Auto",
    "TATACONSUM.NS": "FMCG",
    "UPL.NS":        "Chemicals",
}

SECTORS = sorted(set(SECTOR_MAP.values()))
