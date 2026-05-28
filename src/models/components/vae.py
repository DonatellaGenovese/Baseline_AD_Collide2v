from typing import List, Tuple

import torch
import torch.nn as nn

from .autoencoder import Decoder


class VAEEncoder(nn.Module):
    """Shared-trunk encoder with separate mu and log_var heads."""

    def __init__(self, in_dim: int, hidden_dims: List[int], latent_dim: int):
        super().__init__()
        dims = [in_dim] + list(hidden_dims)
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())
        self.trunk = nn.Sequential(*layers)
        last_dim = hidden_dims[-1] if hidden_dims else in_dim
        self.mu_head = nn.Linear(last_dim, latent_dim)
        self.log_var_head = nn.Linear(last_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(x)
        return self.mu_head(h), self.log_var_head(h)


class VariationalAutoEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dims: List[int], latent_dim: int):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = VAEEncoder(in_dim, hidden_dims, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dims, in_dim)

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        if self.training:
            std = (0.5 * log_var).exp()
            return mu + std * torch.randn_like(std)
        return mu  # at inference, use the mean directly

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (x_hat, mu, log_var)."""
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        return self.decoder(z), mu, log_var

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.encoder(x)
