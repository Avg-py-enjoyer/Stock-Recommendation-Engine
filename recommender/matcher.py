"""
recommender/matcher.py — Find the top-K stocks closest to the user's
preference embedding using cosine similarity in latent space.

Why cosine similarity?
  L2 distance cares about magnitude; cosine similarity cares about
  direction. Since we want "what kind of stock is this" (its profile
  shape) rather than "how large is the number", cosine is a better fit.
"""

import os
import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TOP_K, SECTOR_MAP


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity in [−1, 1]."""
    return 1.0 - cosine(a, b)


def recommend(
    user_embedding:   np.ndarray,
    stock_embeddings: pd.DataFrame,
    top_k:            int = TOP_K,
    exclude_sectors:  list[str] | None = None,
) -> pd.DataFrame:
    """
    Rank all NIFTY 50 stocks by cosine similarity to the user's embedding.

    Parameters
    ----------
    user_embedding   : np.ndarray (latent_dim,)
    stock_embeddings : pd.DataFrame  (tickers × latent_dim)
    top_k            : int
    exclude_sectors  : optional list of sectors to filter out

    Returns
    -------
    recommendations : pd.DataFrame with columns:
        ticker, similarity, rank, sector
    """
    similarities = {}
    for ticker, row in stock_embeddings.iterrows():
        sim = cosine_similarity(user_embedding, row.values)
        similarities[ticker] = sim

    results = pd.DataFrame(
        list(similarities.items()), columns=["ticker", "similarity"]
    ).sort_values("similarity", ascending=False).reset_index(drop=True)

    results["rank"]   = range(1, len(results) + 1)
    results["sector"] = results["ticker"].map(SECTOR_MAP)

    # Optionally exclude sectors
    if exclude_sectors:
        results = results[~results["sector"].isin(exclude_sectors)]

    # Return top-K
    return results.head(top_k).reset_index(drop=True)


def diversified_recommend(
    user_embedding:    np.ndarray,
    stock_embeddings:  pd.DataFrame,
    top_k:             int = TOP_K,
    max_per_sector:    int = 3,
    exclude_sectors:   list[str] | None = None,
) -> pd.DataFrame:
    """
    Like recommend() but enforces sector diversification:
    no more than max_per_sector stocks from the same sector.

    Greedy selection: take stocks in similarity order, skip if sector is full.
    """
    all_ranked = recommend(
        user_embedding, stock_embeddings, top_k=len(stock_embeddings),
        exclude_sectors=exclude_sectors,
    )

    sector_count: dict[str, int] = {}
    selected = []

    for _, row in all_ranked.iterrows():
        sector = row["sector"]
        if sector_count.get(sector, 0) < max_per_sector:
            selected.append(row)
            sector_count[sector] = sector_count.get(sector, 0) + 1
        if len(selected) >= top_k:
            break

    result = pd.DataFrame(selected).reset_index(drop=True)
    result["rank"] = range(1, len(result) + 1)
    return result
