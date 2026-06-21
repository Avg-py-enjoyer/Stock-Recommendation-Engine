"""
embeddings/encoder.py — Encode stock feature vectors and user preference
vectors through the trained autoencoder into the shared latent space.

User Preference → Feature Vector Mapping
-----------------------------------------
The user describes their investment style through a structured preference
object (built in the Streamlit UI). We translate each preference axis
into the same feature dimensions that the autoencoder was trained on,
so both stocks and user queries live in the same latent space.

This is the core trick that makes the system work without any labelled
training data: we use domain knowledge to hand-engineer a "virtual stock"
that represents the ideal portfolio the user wants, then find the closest
real stocks to that virtual embedding.
"""

import os
import pickle

import numpy as np
import pandas as pd
import torch

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import EMBEDDINGS_PATH, SECTORS, SECTOR_MAP


def encode_stocks(model, features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Pass all stock feature vectors through the encoder.

    Returns
    -------
    embeddings_df : pd.DataFrame  (tickers × latent_dim)
    """
    X   = torch.tensor(features_df.values, dtype=torch.float32)
    Z   = model.encode(X).numpy()
    return pd.DataFrame(Z, index=features_df.index,
                        columns=[f"z{i}" for i in range(Z.shape[1])])


def build_preference_vector(
    preferences: dict,
    feature_columns: list[str],
    scaler,
) -> np.ndarray:
    """
    Convert user preference sliders/toggles into a normalised feature vector
    that lives in the same space as the stock features.

    Parameters
    ----------
    preferences : dict with keys:
        momentum_tilt   : float  [-1, 1]   (-1 = contrarian, +1 = momentum)
        risk_tolerance  : float  [0, 1]    (0 = low-risk, 1 = high-risk)
        dividend_focus  : float  [0, 1]    (0 = growth, 1 = income)
        value_focus     : float  [0, 1]    (0 = growth, 1 = deep value)
        quality_focus   : float  [0, 1]    (0 = don't care, 1 = high quality)
        sectors         : list[str]        preferred sectors (empty = all)
        market_cap      : str              "large" | "mid" | "any"

    feature_columns : list of column names the scaler was fit on
    scaler          : fitted StandardScaler

    Returns
    -------
    pref_vector : np.ndarray shape (n_features,) — scaled, ready for encoder
    """
    mom_tilt       = preferences.get("momentum_tilt", 0.0)
    risk_tol       = preferences.get("risk_tolerance", 0.5)
    div_focus      = preferences.get("dividend_focus", 0.0)
    value_focus    = preferences.get("value_focus", 0.5)
    quality_focus  = preferences.get("quality_focus", 0.5)
    pref_sectors   = preferences.get("sectors", [])

    # ── Build a "raw" feature dict using the same column names ─────────────
    raw = {}

    # Momentum features  (high momentum_tilt → strong positive recent returns)
    raw["mom_1m"]  = mom_tilt * 0.05    # ±5% monthly return range
    raw["mom_3m"]  = mom_tilt * 0.12
    raw["mom_6m"]  = mom_tilt * 0.20
    raw["mom_12m"] = mom_tilt * 0.35

    # Risk features
    # High risk_tol → accept higher volatility, deeper drawdowns
    raw["volatility"]   =  0.15 + risk_tol * 0.25        # 15%–40% annual vol
    raw["max_drawdown"] = -0.15 - risk_tol * 0.35         # -15% to -50%
    raw["sharpe"]       =  1.5 - risk_tol * 0.8          # prefer higher Sharpe for low risk

    # Market features
    raw["beta"]             = 0.7 + risk_tol * 0.6        # 0.7–1.3
    raw["benchmark_corr"]   = 0.5 + risk_tol * 0.3

    # Fundamental features
    # Value: low P/E, low P/B → high value_focus
    raw["pe_ratio"]  = 30 - value_focus * 20              # 10–30x
    raw["pb_ratio"]  = 4  - value_focus * 3               # 1–4x

    # Income: high dividend yield → high dividend_focus
    raw["dividend_yield"] = div_focus * 0.04              # 0%–4%

    # Quality: high ROE, low D/E → high quality_focus
    raw["roe"]            =  0.10 + quality_focus * 0.25   # 10%–35%
    raw["roa"]            =  0.04 + quality_focus * 0.12   # 4%–16%
    raw["debt_to_equity"] =  2.0  - quality_focus * 1.5    # 0.5–2.0
    raw["current_ratio"]  =  1.0  + quality_focus * 1.5    # 1.0–2.5

    # Growth
    raw["revenue_growth"]  = 0.05 + (1 - value_focus) * 0.20   # 5%–25%
    raw["earnings_growth"] = 0.05 + (1 - value_focus) * 0.30

    # ── Sector one-hot ─────────────────────────────────────────────────────
    for s in SECTORS:
        col = f"sector_{s}"
        if pref_sectors:
            raw[col] = 1.0 if s in pref_sectors else 0.0
        else:
            raw[col] = 1.0 / len(SECTORS)   # uniform if no preference

    # ── Assemble into array in the correct column order ────────────────────
    pref_array = np.array([raw.get(col, 0.0) for col in feature_columns],
                          dtype=np.float32).reshape(1, -1)

    # Apply the same scaler that was fitted on stock features
    pref_scaled = scaler.transform(pref_array)[0]
    return pref_scaled


def encode_preference(model, pref_vector: np.ndarray) -> np.ndarray:
    """Run the scaled preference vector through the encoder."""
    x = torch.tensor(pref_vector, dtype=torch.float32).unsqueeze(0)
    z = model.encode(x).squeeze(0).numpy()
    return z


def load_or_build_embeddings(model, features_payload: dict, force_rebuild: bool = False) -> dict:
    """Build and cache stock embeddings."""
    if not force_rebuild and os.path.exists(EMBEDDINGS_PATH):
        print(f"[embeddings] Loading cached embeddings from {EMBEDDINGS_PATH}")
        with open(EMBEDDINGS_PATH, "rb") as f:
            return pickle.load(f)

    print("[embeddings] Encoding stocks into latent space …")
    stock_embeddings = encode_stocks(model, features_payload["scaled"])
    payload = {
        "embeddings": stock_embeddings,
        "feature_columns": list(features_payload["scaled"].columns),
    }
    with open(EMBEDDINGS_PATH, "wb") as f:
        pickle.dump(payload, f)
    print(f"[embeddings] Embeddings saved — shape: {stock_embeddings.shape}")
    return payload
