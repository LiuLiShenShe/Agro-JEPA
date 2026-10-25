"""S4A conditioning encoders: normalized GDD scalars and a masked weather GRU.

Both encoders output an ``action_emb_dim`` vector, so the downstream
``ProprioceptiveEmbedding`` + ``num_action_repeat`` geometry of the upstream
DINO-WM ``concat_dim=1`` contract is unchanged from V0.

Normalisation statistics are always fitted on official **train** rows only and
stored in the checkpoint.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class VectorNormStats:
    """Per-feature mean/std fitted on the train split only."""

    mean: tuple[float, ...]
    std: tuple[float, ...]
    n_samples: int = 0

    @classmethod
    def fit(cls, x: np.ndarray) -> "VectorNormStats":
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x[:, None]
        if x.shape[0] == 0:
            raise ValueError("cannot fit VectorNormStats on an empty array")
        mean = np.nanmean(x, axis=0)
        std = np.nanstd(x, axis=0)
        std = np.where(std > 1e-8, std, 1.0)
        return cls(
            mean=tuple(float(v) for v in mean),
            std=tuple(float(v) for v in std),
            n_samples=int(x.shape[0]),
        )

    def normalise(self, x: torch.Tensor) -> torch.Tensor:
        mean = torch.as_tensor(self.mean, dtype=x.dtype, device=x.device)
        std = torch.as_tensor(self.std, dtype=x.dtype, device=x.device)
        return (x - mean) / std

    def as_dict(self) -> dict:
        return {
            "mean": list(self.mean),
            "std": list(self.std),
            "n_samples": self.n_samples,
            "rule": "fitted on official-train rows only; reused for validation",
        }


@dataclass
class GDDEncoderConfig:
    n_frequencies: int = 8
    base_frequency: float = 2.0
    hidden_dim: int = 128
    output_dim: int = 10
    dropout: float = 0.0
    use_learned_scale: bool = False


class GDDEncoder(nn.Module):
    """Standardised GDD scalar(s) -> sinusoidal features -> MLP -> embedding."""

    def __init__(
        self,
        n_scalars: int,
        config: GDDEncoderConfig | None = None,
        norm_stats: VectorNormStats | None = None,
    ) -> None:
        super().__init__()
        if n_scalars < 1:
            raise ValueError("GDDEncoder needs at least one scalar")
        self.config = config or GDDEncoderConfig()
        self.n_scalars = int(n_scalars)
        cfg = self.config
        self.norm_stats = norm_stats or VectorNormStats(
            mean=tuple([0.0] * n_scalars), std=tuple([1.0] * n_scalars), n_samples=0
        )
        if len(self.norm_stats.mean) != self.n_scalars:
            raise ValueError("GDD norm stats do not match the scalar count")

        freqs = cfg.base_frequency ** (
            torch.arange(cfg.n_frequencies, dtype=torch.float32)
            / max(cfg.n_frequencies - 1, 1)
        )
        self.register_buffer("frequencies", freqs, persistent=True)
        self.log_scale = nn.Parameter(torch.zeros(cfg.n_frequencies * 2))

        in_dim = self.n_scalars * cfg.n_frequencies * 2
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, cfg.output_dim),
        )

    def forward(self, gdd: torch.Tensor) -> torch.Tensor:
        if gdd.dim() != 2 or gdd.shape[1] != self.n_scalars:
            raise ValueError(
                f"gdd must be [B,{self.n_scalars}], got {tuple(gdd.shape)}"
            )
        z = self.norm_stats.normalise(gdd.to(torch.float32))
        angles = z[..., None] * self.frequencies                     # (B, S, F)
        feats = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        if self.config.use_learned_scale:
            feats = feats * (1.0 + torch.tanh(self.log_scale))
        return self.mlp(feats.flatten(1))


@dataclass
class WeatherGRUConfig:
    hidden_dim: int = 64
    output_dim: int = 10
    dropout: float = 0.0
    pooling: str = "masked_mean"
    n_channels: int = 5


class WeatherGRUEncoder(nn.Module):
    """Masked GRU over a variable-length daily weather sequence.

    Padding is never treated as real weather: padded timesteps are zeroed
    before the GRU and excluded from the masked mean.  A zero-length mask would
    be a programming error and raises.
    """

    def __init__(
        self,
        n_channels: int,
        config: WeatherGRUConfig | None = None,
        norm_stats: VectorNormStats | None = None,
    ) -> None:
        super().__init__()
        self.config = config or WeatherGRUConfig()
        cfg = self.config
        self.n_channels = int(n_channels)
        self.norm_stats = norm_stats or VectorNormStats(
            mean=tuple([0.0] * n_channels), std=tuple([1.0] * n_channels), n_samples=0
        )
        if len(self.norm_stats.mean) != self.n_channels:
            raise ValueError("weather norm stats do not match the channel count")
        if cfg.pooling != "masked_mean":
            raise ValueError(f"unsupported weather pooling {cfg.pooling!r}")

        self.gru = nn.GRU(
            input_size=self.n_channels,
            hidden_size=cfg.hidden_dim,
            batch_first=True,
        )
        self.mlp = nn.Sequential(
            nn.Linear(2 * cfg.hidden_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, cfg.output_dim),
        )

    def forward(self, weather: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if weather.dim() != 3 or weather.shape[-1] != self.n_channels:
            raise ValueError(
                f"weather must be [B,L,{self.n_channels}], got {tuple(weather.shape)}"
            )
        if mask.shape != weather.shape[:2]:
            raise ValueError(f"mask {tuple(mask.shape)} != weather {tuple(weather.shape)}")
        mask = mask.to(torch.bool)
        counts = mask.sum(dim=1)
        if bool((counts == 0).any()):
            raise ValueError("a weather sequence has no valid day")

        x = self.norm_stats.normalise(weather.to(torch.float32))
        x = torch.where(mask.unsqueeze(-1), x, torch.zeros_like(x))
        h, _ = self.gru(x)                                           # (B, L, H)
        h = torch.where(mask.unsqueeze(-1), h, torch.zeros_like(h))
        mean = h.sum(dim=1) / counts.unsqueeze(-1).to(h.dtype)       # masked mean
        # last valid hidden state, as a companion feature
        last_idx = (mask.cumsum(dim=1) == counts.unsqueeze(1)).float().argmax(dim=1)
        last = h[torch.arange(h.shape[0], device=h.device), last_idx]
        return self.mlp(torch.cat([mean, last], dim=-1))
