"""
model/trainer.py — Training loop for the StockAutoencoder.

Even with just ~50 stocks the autoencoder can learn a meaningful latent
representation by treating each feature vector as a training sample and
minimising reconstruction error.

Training tricks used here:
  • CosineAnnealingLR — smooth lr decay
  • Early stopping   — prevent over-fitting on small dataset
  • Data augmentation — add Gaussian noise during training (denoising AE variant)
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    LATENT_DIM, HIDDEN_DIMS, LEARNING_RATE,
    EPOCHS, BATCH_SIZE, DROPOUT, MODEL_PATH
)
from model.autoencoder import StockAutoencoder


def add_noise(x: torch.Tensor, noise_factor: float = 0.05) -> torch.Tensor:
    """Add Gaussian noise for denoising autoencoder training."""
    noise = torch.randn_like(x) * noise_factor
    return x + noise


def train_autoencoder(
    features_df,
    force_retrain: bool = False,
    verbose:       bool = True,
) -> StockAutoencoder:
    """
    Train (or load) the autoencoder.

    Parameters
    ----------
    features_df : pd.DataFrame  — scaled feature matrix (tickers × features)
    force_retrain : bool        — ignore cached model
    verbose   : bool            — print training progress

    Returns
    -------
    model : StockAutoencoder (eval mode)
    """
    if not force_retrain and os.path.exists(MODEL_PATH):
        if verbose:
            print(f"[trainer] Loading saved model from {MODEL_PATH}")
        n_features = features_df.shape[1]
        model = StockAutoencoder(n_features, HIDDEN_DIMS, LATENT_DIM, DROPOUT)
        model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
        model.eval()
        return model

    # ── Build dataset ─────────────────────────────────────────────────────
    X = torch.tensor(features_df.values, dtype=torch.float32)
    dataset = TensorDataset(X)
    loader  = DataLoader(dataset, batch_size=min(BATCH_SIZE, len(X)), shuffle=True)

    n_features = X.shape[1]
    model      = StockAutoencoder(n_features, HIDDEN_DIMS, LATENT_DIM, DROPOUT)
    optimiser  = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=EPOCHS)
    criterion  = nn.MSELoss()

    best_loss  = float("inf")
    patience   = 40
    wait       = 0
    best_state = None

    if verbose:
        print(f"[trainer] Training autoencoder — {n_features} features → {LATENT_DIM}D latent")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        for (batch,) in loader:
            noisy_batch = add_noise(batch)
            x_hat, _    = model(noisy_batch)
            loss = criterion(x_hat, batch)   # reconstruct CLEAN from NOISY input
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            epoch_loss += loss.item() * len(batch)
        scheduler.step()

        avg_loss = epoch_loss / len(X)

        if avg_loss < best_loss - 1e-6:
            best_loss  = avg_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1

        if verbose and epoch % 50 == 0:
            print(f"  Epoch {epoch:4d}/{EPOCHS}  loss={avg_loss:.6f}  best={best_loss:.6f}  lr={scheduler.get_last_lr()[0]:.2e}")

        if wait >= patience:
            if verbose:
                print(f"  Early stopping at epoch {epoch}.")
            break

    # Restore best weights
    model.load_state_dict(best_state)
    model.eval()
    torch.save(best_state, MODEL_PATH)
    if verbose:
        print(f"[trainer] Model saved → {MODEL_PATH}  (best loss={best_loss:.6f})")

    return model


if __name__ == "__main__":
    import pickle
    from config import FEATURES_PATH
    with open(FEATURES_PATH, "rb") as f:
        payload = pickle.load(f)
    train_autoencoder(payload["scaled"], force_retrain=True)
