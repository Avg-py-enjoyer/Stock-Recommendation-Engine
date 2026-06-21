# NIFTY 50 Stock Recommender — Vector Embedding System

An educational project that uses an **autoencoder neural network** to learn dense vector embeddings of NIFTY 50 stocks, then matches them against a user's natural-language preferences to build and backtest a portfolio.

---

## How It Works

```
User Preferences (text)
        │
        ▼
  Feature Vector (manual encoding of preferences)
        │
        ▼
  Autoencoder Encoder  ◄── trained on NIFTY 50 stock features
        │
        ▼
  Latent Embedding Space
        │
  Cosine Similarity Search
        │
        ▼
  Top-K Recommended Stocks
        │
        ▼
  Portfolio Construction + 5-Year Backtest vs NIFTY 50
```

### Pipeline Steps

1. **Data Fetching** (`data/fetcher.py`) — Downloads 5 years of OHLCV + fundamentals for all NIFTY 50 stocks via `yfinance`.
2. **Feature Engineering** (`features/engineer.py`) — Computes ~20 features per stock: momentum, volatility, Sharpe, P/E, beta, sector dummies, etc.
3. **Autoencoder** (`model/autoencoder.py`) — A symmetric encoder-decoder that compresses the feature vector into a low-dimensional latent space.
4. **Embedding** (`embeddings/encoder.py`) — Encodes each stock into its latent vector. Also encodes user preference vectors.
5. **Recommender** (`recommender/matcher.py`) — Finds top-K stocks by cosine similarity to the user's preference embedding.
6. **Portfolio** (`portfolio/constructor.py`) — Builds an equal-weight (or volatility-weighted) portfolio from recommendations.
7. **Backtest** (`backtest/engine.py`) — Simulates 5-year performance, benchmarks against NIFTY 50, computes Sharpe, CAGR, max drawdown, etc.
8. **App** (`app.py`) — Streamlit UI with sliders, plots, and metric cards.

---

## Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## Project Structure

```
nifty_recommender/
├── README.md
├── requirements.txt
├── app.py                      # Streamlit UI
├── config.py                   # Global constants
├── data/
│   ├── __init__.py
│   └── fetcher.py              # yfinance data download + caching
├── features/
│   ├── __init__.py
│   └── engineer.py             # Feature engineering pipeline
├── model/
│   ├── __init__.py
│   ├── autoencoder.py          # PyTorch autoencoder
│   └── trainer.py              # Training loop
├── embeddings/
│   ├── __init__.py
│   └── encoder.py              # Encode stocks + user prefs
├── recommender/
│   ├── __init__.py
│   └── matcher.py              # Cosine similarity matching
├── portfolio/
│   ├── __init__.py
│   └── constructor.py          # Portfolio weight construction
├── backtest/
│   ├── __init__.py
│   └── engine.py               # Backtest + metrics + plots
└── outputs/                    # Saved models, embeddings, plots
```

---

## Educational Notes

- The autoencoder is trained **unsupervised** — it learns to reconstruct stock features, forcing the latent space to capture the most salient patterns.
- User preferences are **mapped to the same feature space** using a preference vector builder, then encoded through the same encoder.
- This is a **content-based filtering** system — no historical user data needed.
- The backtest uses a **simple buy-and-hold** strategy with annual rebalancing, which is appropriate for an educational project.
