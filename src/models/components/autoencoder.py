from typing import List

import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dims: List[int], latent_dim: int):
        super().__init__()
        dims = [in_dim] + list(hidden_dims) + [latent_dim]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, latent_dim: int, hidden_dims: List[int], out_dim: int):
        super().__init__()
        dims = [latent_dim] + list(reversed(hidden_dims)) + [out_dim]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class AutoEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dims: List[int], latent_dim: int):
        super().__init__()
        self.encoder = Encoder(in_dim, hidden_dims, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dims, in_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)
