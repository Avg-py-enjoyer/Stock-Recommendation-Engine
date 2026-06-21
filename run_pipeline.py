"""
run_pipeline.py — Run the full pipeline from the command line (no UI).
Useful for debugging and verifying each module works before launching the app.

Usage:
    python run_pipeline.py
    python run_pipeline.py --refresh   # force re-download + retrain
"""

import argparse
import os
import sys
import warnings
warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)


def main(refresh: bool = False):
    print("=" * 60)
    print("  NIFTY 50 AI Recommender — Pipeline Test Run")
    print("=" * 60)

    # ── Step 1: Data ─────────────────────────────────────────────────────
    print("\n[1/6] Loading data …")
    from data.fetcher import load_or_fetch
    raw_data = load_or_fetch(force_refresh=refresh)
    print(f"      Stocks loaded: {len(raw_data['prices'])}")
    print(f"      Date range:    {raw_data['start']} → {raw_data['end']}")

    # ── Step 2: Features ─────────────────────────────────────────────────
    print("\n[2/6] Building features …")
    from features.engineer import load_or_build
    feat_payload = load_or_build(raw_data, force_rebuild=refresh)
    print(f"      Feature matrix: {feat_payload['scaled'].shape}")
    print(f"      Columns: {list(feat_payload['scaled'].columns[:6])} …")

    # ── Step 3: Train autoencoder ─────────────────────────────────────────
    print("\n[3/6] Training autoencoder …")
    from model.trainer import train_autoencoder
    model = train_autoencoder(feat_payload["scaled"], force_retrain=refresh)
    print(f"      Model: {model}")

    # ── Step 4: Embeddings ────────────────────────────────────────────────
    print("\n[4/6] Building embeddings …")
    from embeddings.encoder import load_or_build_embeddings
    emb_payload = load_or_build_embeddings(model, feat_payload, force_rebuild=refresh)
    print(f"      Embeddings: {emb_payload['embeddings'].shape}")

    # ── Step 5: Recommend ─────────────────────────────────────────────────
    print("\n[5/6] Running recommender …")
    from embeddings.encoder import build_preference_vector, encode_preference
    from recommender.matcher import diversified_recommend

    preferences = {
        "momentum_tilt":  0.3,
        "risk_tolerance": 0.4,
        "dividend_focus": 0.2,
        "value_focus":    0.6,
        "quality_focus":  0.7,
        "sectors":        [],
    }
    pref_vec = build_preference_vector(
        preferences,
        list(feat_payload["scaled"].columns),
        feat_payload["scaler"],
    )
    pref_emb = encode_preference(model, pref_vec)
    recs     = diversified_recommend(pref_emb, emb_payload["embeddings"], top_k=8)

    print("\n  ── Recommendations ──────────────────────────────────")
    for _, row in recs.iterrows():
        print(f"  #{row['rank']:2d}  {row['ticker']:<18s}  sim={row['similarity']:.4f}  {row['sector']}")

    # ── Step 6: Backtest ──────────────────────────────────────────────────
    print("\n[6/6] Running backtest …")
    from portfolio.constructor import build_portfolio
    from backtest.engine import run_backtest

    portfolio = build_portfolio(recs, raw_data["prices"], weighting="equal")
    results   = run_backtest(portfolio, raw_data["prices"], raw_data["benchmark"])

    print("\n  ── Performance Metrics ──────────────────────────────")
    for k, v in results["metrics"].items():
        print(f"  {k:<25s} {v}")

    print("\n✅ Pipeline complete. Launch the UI with:  streamlit run app.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true", help="Force re-download and retrain")
    args = parser.parse_args()
    main(refresh=args.refresh)
