"""
backend/main.py — FastAPI server.

Endpoints
---------
GET  /api/status          → pipeline health + metadata
GET  /api/sectors         → list of available sectors
POST /api/recommend       → run recommender + backtest
GET  /api/refresh         → force data re-download + model retrain
GET  /                    → serve frontend/index.html
GET  /{path}              → serve static frontend files
"""

import os, sys, json, warnings
warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import Optional

from config import SECTORS, SECTOR_MAP, NIFTY50_TICKERS
from backend.pipeline import get_pipeline
from embeddings.encoder   import build_preference_vector, encode_preference
from recommender.matcher  import diversified_recommend
from portfolio.constructor import build_portfolio
from backtest.engine       import run_backtest
from sklearn.decomposition import PCA

app = FastAPI(title="NIFTY 50 AI Recommender", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

FRONTEND_DIR = os.path.join(ROOT, "frontend")


# ── Serve frontend ────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def serve_index():
    path = os.path.join(FRONTEND_DIR, "index.html")
    with open(path) as f:
        return HTMLResponse(content=f.read())


# ── API: status ────────────────────────────────────────────────────────────
@app.get("/api/status")
async def status():
    try:
        pipe = get_pipeline()
        raw  = pipe["raw_data"]
        feat = pipe["feat_payload"]
        return {
            "ready":        True,
            "stocks_loaded": len(raw["prices"]),
            "feature_dim":  feat["scaled"].shape[1],
            "date_start":   raw["start"],
            "date_end":     raw["end"],
            "sectors":      SECTORS,
        }
    except Exception as e:
        return {"ready": False, "error": str(e)}


# ── API: sectors ───────────────────────────────────────────────────────────
@app.get("/api/sectors")
async def sectors():
    return {"sectors": SECTORS}


# ── Request model ──────────────────────────────────────────────────────────
class PreferenceRequest(BaseModel):
    momentum_tilt:  float = Field(0.2,  ge=-1.0, le=1.0)
    risk_tolerance: float = Field(0.4,  ge=0.0,  le=1.0)
    dividend_focus: float = Field(0.3,  ge=0.0,  le=1.0)
    value_focus:    float = Field(0.5,  ge=0.0,  le=1.0)
    quality_focus:  float = Field(0.6,  ge=0.0,  le=1.0)
    sectors:        list[str] = []
    top_k:          int = Field(8,  ge=3, le=20)
    max_per_sector: int = Field(3,  ge=1, le=5)
    weighting:      str = Field("equal")  # equal | similarity | inv_vol


# ── API: recommend + backtest ──────────────────────────────────────────────
@app.post("/api/recommend")
async def recommend(req: PreferenceRequest):
    try:
        pipe         = get_pipeline()
        raw_data     = pipe["raw_data"]
        feat_payload = pipe["feat_payload"]
        model        = pipe["model"]
        emb_payload  = pipe["emb_payload"]

        feat_cols = list(feat_payload["scaled"].columns)

        # Build preference vector and encode
        pref_vec = build_preference_vector(
            req.model_dump(), feat_cols, feat_payload["scaler"]
        )
        pref_emb = encode_preference(model, pref_vec)

        # Recommend
        recs = diversified_recommend(
            user_embedding   = pref_emb,
            stock_embeddings = emb_payload["embeddings"],
            top_k            = req.top_k,
            max_per_sector   = req.max_per_sector,
        )

        # Portfolio + Backtest
        portfolio = build_portfolio(recs, raw_data["prices"], req.weighting)
        results   = run_backtest(portfolio, raw_data["prices"], raw_data["benchmark"])

        # ── PCA for latent space viz ──────────────────────────────────────
        stock_embs = emb_payload["embeddings"]
        all_vecs   = np.vstack([stock_embs.values, pref_emb.reshape(1, -1)])
        pca        = PCA(n_components=2)
        coords     = pca.fit_transform(all_vecs)
        stock_xy   = coords[:-1]
        pref_xy    = coords[-1]

        rec_tickers = set(recs["ticker"].tolist())
        latent_points = []
        for i, ticker in enumerate(stock_embs.index):
            latent_points.append({
                "ticker":    ticker.replace(".NS",""),
                "sector":    SECTOR_MAP.get(ticker, "Other"),
                "x":         float(stock_xy[i, 0]),
                "y":         float(stock_xy[i, 1]),
                "recommended": ticker in rec_tickers,
            })

        # ── Serialise chart data ──────────────────────────────────────────
        def s(series):
            return {
                "dates":  [str(d)[:10] for d in series.index.tolist()],
                "values": [round(float(v), 4) for v in series.values],
            }

        port_nav_s  = s(results["portfolio_nav"])
        bench_nav_s = s(results["benchmark_nav"])
        drawdown_s  = s(results["drawdown"] * 100)
        roll_sharpe = (
            results["port_ret"].rolling(252).mean() * 252 - 0.065
        ) / (results["port_ret"].rolling(252).std() * np.sqrt(252))
        roll_sharpe_s = s(roll_sharpe.dropna())

        # Monthly heatmap
        monthly_df = results["monthly_ret"]
        pivot      = monthly_df.pivot(index="Year", columns="Month", values="Return").fillna(0)
        month_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

        heatmap = {
            "years":  [int(y) for y in pivot.index.tolist()],
            "months": [month_names[m-1] for m in pivot.columns.tolist()],
            "values": [[round(float(v), 2) for v in row] for row in pivot.values],
        }

        # Return distribution
        port_daily  = [round(float(v)*100, 4) for v in results["port_ret"].values]
        bench_daily = [round(float(v)*100, 4) for v in results["bench_ret"].values]

        # Recommendations list
        raw_feats = feat_payload["raw"]
        recs_out  = []
        for _, row in recs.iterrows():
            t = row["ticker"]
            recs_out.append({
                "ticker":     t.replace(".NS",""),
                "sector":     row["sector"],
                "similarity": round(float(row["similarity"]), 4),
                "rank":       int(row["rank"]),
                "weight":     round(portfolio["weights"].get(t, 0) * 100, 2),
                "sharpe":     round(float(raw_feats.loc[t, "sharpe"]), 3) if t in raw_feats.index else None,
                "volatility": round(float(raw_feats.loc[t, "volatility"]) * 100, 2) if t in raw_feats.index else None,
                "mom_12m":    round(float(raw_feats.loc[t, "mom_12m"]) * 100, 2) if t in raw_feats.index else None,
                "pe_ratio":   round(float(raw_feats.loc[t, "pe_ratio"]), 1) if t in raw_feats.index else None,
            })

        return JSONResponse({
            "recommendations": recs_out,
            "metrics":         results["metrics"],
            "charts": {
                "portfolio_nav":  port_nav_s,
                "benchmark_nav":  bench_nav_s,
                "drawdown":       drawdown_s,
                "rolling_sharpe": roll_sharpe_s,
                "heatmap":        heatmap,
                "port_daily":     port_daily,
                "bench_daily":    bench_daily,
            },
            "latent_space": {
                "points":     latent_points,
                "user":       {"x": float(pref_xy[0]), "y": float(pref_xy[1])},
                "variance":   [round(float(v)*100, 1) for v in pca.explained_variance_ratio_],
            },
            "portfolio": {
                "tickers":   [t.replace(".NS","") for t in portfolio["tickers"]],
                "weights":   {t.replace(".NS",""): round(w*100,2) for t,w in portfolio["weights"].items()},
                "weighting": portfolio["weighting"],
            },
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ── API: force refresh ─────────────────────────────────────────────────────
@app.get("/api/refresh")
async def refresh():
    try:
        get_pipeline(force_refresh=True)
        return {"status": "refreshed"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=False)