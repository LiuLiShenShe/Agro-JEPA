"""S3: irregular time-interval conditioning for the FIP predictor.

DINO-WM's world model is trained on fixed-frequency robot sequences, where the
action channel already encodes the transition.  FIP observations are irregular:
``delta_days`` for a pair of images of the same plot ranges from 1 to 120 d
(``reports/s2_pairs.md`` §4.5).  This module turns that scalar interval into the
action-channel embedding, so the upstream predictor interface is unchanged.

Design
------
``delta_days`` is first normalised with statistics that are computed **on the
train split only** (never on validation/test), then embedded with a sinusoidal
encoder at several frequencies and projected to the predictor's embedding
width.  Sinusoidal encoding is used rather than a learned per-day embedding
because the interval is continuous and the daily counts are heavy-tailed
(1 d common, 90 d rare) -- a per-day table would waste capacity on unseen days.

The projection output is the *action* vector fed to ``action_encoder``.  It has
``action_dim`` entries and is repeated ``num_action_repeat`` times per patch, so
the effective channel width matches the upstream ``concat_dim=1`` geometry.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import torch
from torch import nn


@dataclass(frozen=True)
class DeltaNormStats:
    """Train-split statistics used to normalise ``delta_days``.

    Fitted on the train split only and stored inside the checkpoint, so the
    validation/test intervals are always normalised with the train parameters
    (S3 protocol §3.7).
    """

    mean: float
    std: float
    min: float = 0.0
    max: float = 120.0
    n_samples: int = 0

    @classmethod
    def fit(cls, delta_days: Sequence[float] | torch.Tensor) -> "DeltaNormStats":
        if isinstance(delta_days, torch.Tensor):
            x = delta_days.detach().cpu().to(torch.float64).numpy()
        else:
            x = torch.tensor(list(delta_days), dtype=torch.float64).numpy()
        if x.size == 0:
            raise ValueError("cannot fit DeltaNormStats on an empty sequence")
        return cls(
            mean=float(x.mean()),
            std=float(x.std()) if float(x.std()) > 1e-8 else 1.0,
            min=float(x.min()),
            max=float(x.max()),
            n_samples=int(x.size),
        )

    def normalise(self, delta_days: torch.Tensor) -> torch.Tensor:
        """Standardised interval; returned in roughly [-1, 1] for typical values."""
        d = delta_days.to(torch.float32)
        return (d - self.mean) / self.std

    def as_dict(self) -> dict:
        return {
            "mean": self.mean,
            "std": self.std,
            "min": self.min,
            "max": self.max,
            "n_samples": self.n_samples,
            "rule": "fitted on train split delta_days only; reused for validation/test",
        }


@dataclass
class DeltaEmbeddingConfig:
    """Configuration of the time-conditioning encoder."""

    n_frequencies: int = 8
    base_frequency: float = 2.0
    hidden_dim: int = 256
    output_dim: int = 10          # == upstream action_emb_dim (repeated to action_dim)
    dropout: float = 0.0
    use_learned_scale: bool = True

    def as_dict(self) -> dict:
        return {
            "n_frequencies": self.n_frequencies,
            "base_frequency": self.base_frequency,
            "hidden_dim": self.hidden_dim,
            "output_dim": self.output_dim,
            "dropout": self.dropout,
            "use_learned_scale": self.use_learned_scale,
            "note": (
                "sinusoidal encoding of the normalised interval, then an MLP to "
                "action_dim; matches the upstream action-channel width"
            ),
        }


class DeltaTimeEncoder(nn.Module):
    """Map ``delta_days`` -> an action embedding compatible with DINO-WM.

    Parameters
    ----------
    config:
        Frequency count, MLP width and output dimensionality.
    norm_stats:
        Train-split normalisation statistics.  If ``None`` the encoder falls
        back to ``(d - mean) / std`` with ``mean=0, std=1`` and the caller must
        attach the real statistics before training.
    """

    def __init__(
        self,
        config: DeltaEmbeddingConfig | None = None,
        norm_stats: DeltaNormStats | None = None,
    ) -> None:
        super().__init__()
        self.config = config or DeltaEmbeddingConfig()
        self.norm_stats = norm_stats or DeltaNormStats(mean=0.0, std=1.0, n_samples=0)
        cfg = self.config

        # fixed geometric ladder of frequencies (not learned): this keeps the
        # representation monotone and lets it generalise to unseen intervals
        freqs = cfg.base_frequency ** (
            torch.arange(cfg.n_frequencies, dtype=torch.float32)
            / max(cfg.n_frequencies - 1, 1)
        )
        self.register_buffer("frequencies", freqs, persistent=True)
        self.log_scale = nn.Parameter(torch.zeros(cfg.n_frequencies * 2))

        in_dim = cfg.n_frequencies * 2
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, cfg.output_dim),
        )
        self.output_dim = cfg.output_dim

    # -- normalisation ------------------------------------------------------
    def set_norm_stats(self, stats: DeltaNormStats) -> None:
        self.norm_stats = stats

    def encode_normalised(self, delta_norm: torch.Tensor) -> torch.Tensor:
        """Sinusoidal features of an already-normalised interval."""
        # (B, F)
        angles = delta_norm[..., None].to(torch.float32) * self.frequencies
        feats = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        if self.config.use_learned_scale:
            feats = feats * (1.0 + torch.tanh(self.log_scale))
        return feats

    def forward(self, delta_days: torch.Tensor) -> torch.Tensor:
        """``delta_days`` (B,) or (B, T) -> (B, output_dim) or (B, T, output_dim)."""
        if delta_days.dim() == 1:
            single = True
            d = delta_days[:, None]                      # (B, 1)
        elif delta_days.dim() == 2:
            single = False
            d = delta_days
        else:
            raise ValueError(f"delta_days must be (B,) or (B,T); got {delta_days.shape}")

        norm = self.norm_stats.normalise(d)
        feats = self.encode_normalised(norm)
        emb = self.mlp(feats)
        return emb[..., 0, :] if single else emb

    # -- checkpointing ------------------------------------------------------
    def state_dict_with_stats(self) -> dict:
        sd = {k: v for k, v in self.state_dict().items()}
        sd["norm_stats"] = torch.tensor(
            [
                self.norm_stats.mean,
                self.norm_stats.std,
                self.norm_stats.min,
                self.norm_stats.max,
                float(self.norm_stats.n_samples),
            ]
        )
        return sd

    @classmethod
    def from_state_dict_with_stats(
        cls, sd: dict, config: DeltaEmbeddingConfig | None = None
    ) -> "DeltaTimeEncoder":
        stats = DeltaNormStats(
            mean=float(sd["norm_stats"][0]),
            std=float(sd["norm_stats"][1]),
            min=float(sd["norm_stats"][2]),
            max=float(sd["norm_stats"][3]),
            n_samples=int(sd["norm_stats"][4]),
        )
        model = cls(config=config, norm_stats=stats)
        sd = {k: v for k, v in sd.items() if k != "norm_stats"}
        model.load_state_dict(sd)
        return model
