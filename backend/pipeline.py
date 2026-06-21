"""
backend/pipeline.py — Loads and holds the full pipeline in memory.
Called once at server startup; results cached for fast API responses.
"""

import os, sys, warnings
warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from data.fetcher       import load_or_fetch
from features.engineer  import load_or_build
from model.trainer      import train_autoencoder
from embeddings.encoder import load_or_build_embeddings

_cache = {}

def get_pipeline(force_refresh: bool = False):
    if _cache and not force_refresh:
        return _cache

    print("[pipeline] Loading data …")
    raw_data = load_or_fetch(force_refresh=force_refresh)

    print("[pipeline] Building features …")
    feat_payload = load_or_build(raw_data, force_rebuild=force_refresh)

    print("[pipeline] Training autoencoder …")
    model = train_autoencoder(feat_payload["scaled"], force_retrain=force_refresh)

    print("[pipeline] Building embeddings …")
    emb_payload = load_or_build_embeddings(model, feat_payload, force_rebuild=force_refresh)

    _cache.update({
        "raw_data":     raw_data,
        "feat_payload": feat_payload,
        "model":        model,
        "emb_payload":  emb_payload,
    })
    print("[pipeline] Ready.")
    return _cache