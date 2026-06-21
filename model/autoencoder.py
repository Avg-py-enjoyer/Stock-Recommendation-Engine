"""
model/autoencoder.py — Symmetric Autoencoder for stock feature embedding.

Architecture
------------
Input (n_features)
  → Linear(n_features, H1) + BatchNorm + LeakyReLU + Dropout
  → Linear(H1, H2)         + BatchNorm + LeakyReLU + Dropout
  → Linear(H2, latent_dim)                              ← LATENT SPACE (embedding)
  → Linear(latent_dim, H2) + BatchNorm + LeakyReLU + Dropout
  → Linear(H2, H1)         + BatchNorm + LeakyReLU + Dropout
  → Linear(H1, n_features)                              ← Reconstruction

Loss: MSE (reconstruction) — no KL divergence, this is a plain AE not VAE.

Why autoencoder?
  The encoder learns to compress each stock's 20+ features into a dense,
  information-rich vector of size `latent_dim`. Stocks with similar
  fundamental + price characteristics cluster together in latent space.
  We exploit this for nearest-neighbour recommendation.
"""

import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, n_features: int, hidden_dims: list[int], latent_dim: int, dropout: float):
        super().__init__()
        layers = []
        in_dim = n_features
        for h in hidden_dims:
            layers += [
                nn.Linear(in_dim, h),
                nn.BatchNorm1d(h),
                nn.LeakyReLU(0.1),
                nn.Dropout(dropout),
            ]
            in_dim = h
        layers.append(nn.Linear(in_dim, latent_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, latent_dim: int, hidden_dims: list[int], n_features: int, dropout: float):
        super().__init__()
        layers = []
        in_dim = latent_dim
        for h in reversed(hidden_dims):
            layers += [
                nn.Linear(in_dim, h),
                nn.BatchNorm1d(h),
                nn.LeakyReLU(0.1),
                nn.Dropout(dropout),
            ]
            in_dim = h
        layers.append(nn.Linear(in_dim, n_features))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class StockAutoencoder(nn.Module):
    """Full autoencoder: encoder + decoder."""

    def __init__(
        self,
        n_features:  int,
        hidden_dims: list[int],
        latent_dim:  int,
        dropout:     float = 0.2,
    ):
        super().__init__()
        self.encoder = Encoder(n_features, hidden_dims, latent_dim, dropout)
        self.decoder = Decoder(latent_dim, hidden_dims, n_features, dropout)
        self.n_features = n_features
        self.latent_dim = latent_dim

    def forward(self, x: torch.Tensor):
        z    = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return only the latent embedding (inference)."""
        self.eval()
        with torch.no_grad():
            return self.encoder(x)

    def extra_repr(self) -> str:
        return f"n_features={self.n_features}, latent_dim={self.latent_dim}"
