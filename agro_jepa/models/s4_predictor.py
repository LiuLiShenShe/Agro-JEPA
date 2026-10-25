"""S4A shared latent predictor: one architecture, four possible input sets.

The visual trunk is the upstream DINO-WM ``ViTPredictor`` (pinned SHA in
``README.md``), exactly as in V0.  Only the *conditioning* channel changes:

    z = cat([z_visual (256x768), proprio_placeholder (256x160), condition (256x160)])
    Z_hat_b = predictor(z)[..., :768]

When a condition has no informative input at all (``b2_no_time``) the
conditioning channel is **removed from the network**, not zeroed at inference:
the predictor input width becomes 768 + 160 = 928 instead of 1088.  That is the
honest "no time input" ablation; the parameter difference is reported.

The ``proprio`` channel stays the constant-zero placeholder of V0 and never
carries a phenotype or a weather value.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn

from ..datasets.s4_dataset import ConditioningSpec
from .fip_dino_wm import _load_upstream, move_attention_masks  # noqa: F401
from .s4_conditioning import (
    GDDEncoder,
    GDDEncoderConfig,
    VectorNormStats,
    WeatherGRUConfig,
    WeatherGRUEncoder,
)
from .time_embedding import DeltaEmbeddingConfig, DeltaNormStats, DeltaTimeEncoder


def _sub_dataclass(cls, data: dict | None):
    data = data or {}
    return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class S4ModelConfig:
    num_patches: int = 256
    visual_dim: int = 768
    depth: int = 6
    heads: int = 16
    mlp_dim: int = 2048
    dropout: float = 0.1
    emb_dropout: float = 0.0
    pool: str = "mean"
    action_emb_dim: int = 10
    num_action_repeat: int = 16
    proprio_emb_dim: int = 10
    num_proprio_repeat: int = 16
    delta: DeltaEmbeddingConfig = field(default_factory=DeltaEmbeddingConfig)
    gdd: GDDEncoderConfig = field(default_factory=GDDEncoderConfig)
    weather: WeatherGRUConfig = field(default_factory=WeatherGRUConfig)
    fusion_hidden: int = 64

    @classmethod
    def from_dict(cls, d: dict) -> "S4ModelConfig":
        d = dict(d or {})
        cfg = cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
        cfg.delta = _sub_dataclass(DeltaEmbeddingConfig, d.get("delta"))
        cfg.gdd = _sub_dataclass(GDDEncoderConfig, d.get("gdd"))
        cfg.weather = _sub_dataclass(WeatherGRUConfig, d.get("weather"))
        return cfg

    def condition_dim(self) -> int:
        return self.action_emb_dim * self.num_action_repeat

    def proprio_dim(self) -> int:
        return self.proprio_emb_dim * self.num_proprio_repeat

    def predictor_dim(self, has_condition: bool) -> int:
        return self.visual_dim + self.proprio_dim() + (
            self.condition_dim() if has_condition else 0
        )

    def as_dict(self) -> dict:
        return {
            "num_patches": self.num_patches,
            "visual_dim": self.visual_dim,
            "depth": self.depth,
            "heads": self.heads,
            "mlp_dim": self.mlp_dim,
            "dropout": self.dropout,
            "emb_dropout": self.emb_dropout,
            "pool": self.pool,
            "action_emb_dim": self.action_emb_dim,
            "num_action_repeat": self.num_action_repeat,
            "proprio_emb_dim": self.proprio_emb_dim,
            "num_proprio_repeat": self.num_proprio_repeat,
            "condition_dim": self.condition_dim(),
            "proprio_dim": self.proprio_dim(),
            "fusion_hidden": self.fusion_hidden,
            "delta": self.delta.as_dict(),
            "gdd": vars(self.gdd),
            "weather": vars(self.weather),
            "proprio_placeholder": "constant zeros (no future phenotype input)",
        }


class S4LatentPredictor(nn.Module):
    """Configurable conditional latent predictor on frozen DINOv2 tokens."""

    def __init__(
        self,
        config: S4ModelConfig,
        spec: ConditioningSpec,
        delta_norm: DeltaNormStats | None = None,
        gdd_norm: VectorNormStats | None = None,
        weather_norm: VectorNormStats | None = None,
    ) -> None:
        super().__init__()
        ViTPredictor, ProprioceptiveEmbedding = _load_upstream()
        self.config = config
        self.spec = spec
        cfg = config

        self.parts: list[str] = []
        if spec.use_delta:
            self.parts.append("delta")
        if spec.gdd_mode != "none":
            self.parts.append("gdd")
        if spec.weather_mode != "none":
            self.parts.append("weather")
        self.has_condition = bool(self.parts)

        dim = cfg.predictor_dim(self.has_condition)
        self.predictor = ViTPredictor(
            num_patches=cfg.num_patches,
            num_frames=1,
            dim=dim,
            depth=cfg.depth,
            heads=cfg.heads,
            mlp_dim=cfg.mlp_dim,
            pool=cfg.pool,
            dropout=cfg.dropout,
            emb_dropout=cfg.emb_dropout,
        )

        # delta channel (identical to V0 when it is the only part)
        self.delta_encoder = None
        if spec.use_delta:
            if cfg.delta.output_dim != cfg.action_emb_dim:
                raise ValueError("delta.output_dim must equal action_emb_dim")
            self.delta_encoder = DeltaTimeEncoder(config=cfg.delta, norm_stats=delta_norm)

        self.gdd_encoder = None
        if spec.gdd_mode != "none":
            if cfg.gdd.output_dim != cfg.action_emb_dim:
                raise ValueError("gdd.output_dim must equal action_emb_dim")
            self.gdd_encoder = GDDEncoder(
                n_scalars=len(spec.gdd_keys()), config=cfg.gdd, norm_stats=gdd_norm
            )

        self.weather_encoder = None
        if spec.weather_mode != "none":
            if cfg.weather.output_dim != cfg.action_emb_dim:
                raise ValueError("weather.output_dim must equal action_emb_dim")
            self.weather_encoder = WeatherGRUEncoder(
                n_channels=cfg.weather.n_channels,
                config=cfg.weather,
                norm_stats=weather_norm,
            )

        self.fusion = None
        if len(self.parts) > 1:
            self.fusion = nn.Sequential(
                nn.Linear(len(self.parts) * cfg.action_emb_dim, cfg.fusion_hidden),
                nn.GELU(),
                nn.Linear(cfg.fusion_hidden, cfg.action_emb_dim),
            )

        self.action_encoder = None
        if self.has_condition:
            self.action_encoder = ProprioceptiveEmbedding(
                num_frames=1, tubelet_size=1, use_3d_pos=False,
                in_chans=cfg.action_emb_dim, emb_dim=cfg.action_emb_dim,
            )
        self.proprio_encoder = ProprioceptiveEmbedding(
            num_frames=1, tubelet_size=1, use_3d_pos=False,
            in_chans=1, emb_dim=cfg.proprio_emb_dim,
        )
        self.register_buffer("proprio_placeholder", torch.zeros(1, 1, 1))

        self.concat_dim = 1
        self.num_action_repeat = cfg.num_action_repeat
        self.num_proprio_repeat = cfg.num_proprio_repeat
        self.emb_dim = cfg.visual_dim

    # -- channels -----------------------------------------------------------
    def encode_proprio(self, batch_size: int, device: torch.device) -> torch.Tensor:
        raw = self.proprio_placeholder.to(device).expand(batch_size, -1, -1)
        emb = self.proprio_encoder(raw)
        return emb.repeat(1, 1, self.num_proprio_repeat)

    def condition_vector(
        self,
        delta_days: torch.Tensor | None,
        gdd: torch.Tensor | None,
        weather: torch.Tensor | None,
        weather_mask: torch.Tensor | None,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor | None:
        """Raw conditioning vector before the upstream action encoder."""
        if not self.has_condition:
            return None
        out = []
        if "delta" in self.parts:
            if delta_days is None:
                raise ValueError("condition needs delta_days but none was passed")
            out.append(self.delta_encoder(delta_days.to(device).reshape(batch_size)))
        if "gdd" in self.parts:
            if gdd is None:
                raise ValueError("condition needs gdd but none was passed")
            out.append(self.gdd_encoder(gdd.to(device)))
        if "weather" in self.parts:
            if weather is None or weather_mask is None:
                raise ValueError("condition needs weather and weather_mask")
            out.append(self.weather_encoder(weather.to(device), weather_mask.to(device)))
        if len(out) == 1:
            return out[0]
        return self.fusion(torch.cat(out, dim=-1))

    def _condition_channel(self, cond: torch.Tensor, batch_size: int) -> torch.Tensor:
        emb = self.action_encoder(cond[:, None, :])
        return emb.repeat(1, 1, self.num_action_repeat)

    # -- forward ------------------------------------------------------------
    def forward(
        self,
        z_a: torch.Tensor,
        delta_days: torch.Tensor | None = None,
        gdd: torch.Tensor | None = None,
        weather: torch.Tensor | None = None,
        weather_mask: torch.Tensor | None = None,
        return_components: bool = False,
    ):
        if z_a.dim() != 3 or z_a.shape[1:] != (
            self.config.num_patches, self.config.visual_dim
        ):
            raise ValueError(
                f"z_a must be [B,{self.config.num_patches},{self.config.visual_dim}], "
                f"got {tuple(z_a.shape)}"
            )
        b = z_a.shape[0]
        proprio = self.encode_proprio(b, z_a.device)
        proprio = proprio.expand(b, self.config.num_patches, self.config.proprio_dim())
        pieces = [z_a, proprio]
        cond_raw = self.condition_vector(
            delta_days, gdd, weather, weather_mask, b, z_a.device
        )
        if cond_raw is not None:
            pieces.append(self._condition_channel(cond_raw, b).expand(
                b, self.config.num_patches, self.config.condition_dim()
            ))
        z = torch.cat(pieces, dim=-1)
        z_all = self.predictor(z)
        z_visual = z_all[..., : self.config.visual_dim]
        if return_components:
            return z_visual, {
                "condition": cond_raw,
                "predictor_input_dim": int(z.shape[-1]),
                "parts": list(self.parts),
            }
        return z_visual

    # -- introspection ------------------------------------------------------
    def n_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def n_condition_parameters(self) -> int:
        mods = [m for m in (self.delta_encoder, self.gdd_encoder,
                            self.weather_encoder, self.fusion) if m is not None]
        return sum(p.numel() for m in mods for p in m.parameters())

    @torch.no_grad()
    def predict(self, z_a: torch.Tensor, **kwargs) -> torch.Tensor:
        was_training = self.training
        self.eval()
        out = self.forward(z_a, **kwargs)
        self.train(was_training)
        return out


def conditioning_spec(condition: dict) -> ConditioningSpec:
    """Build the spec from one entry of ``config["conditions"]``."""
    return ConditioningSpec(
        use_delta=bool(condition.get("use_delta", False)),
        gdd_mode=str(condition.get("gdd_mode", "none")),
        gdd_base=float(condition.get("gdd_base", 0.0)),
        weather_mode=str(condition.get("weather_mode", "none")),
    )


def build_s4_predictor(
    config: S4ModelConfig,
    spec: ConditioningSpec,
    delta_norm: DeltaNormStats | None = None,
    gdd_norm: VectorNormStats | None = None,
    weather_norm: VectorNormStats | None = None,
) -> S4LatentPredictor:
    """Instantiate the subclass that matches the conditioning spec."""
    if spec.gdd_mode != "none":
        from .gdd_conditioned_predictor import GDDConditionedPredictor

        return GDDConditionedPredictor(
            config=config, spec=spec, delta_norm=delta_norm, gdd_norm=gdd_norm
        )
    if spec.weather_mode != "none":
        from .weather_gru_predictor import WeatherGRUPredictor

        return WeatherGRUPredictor(
            config=config, spec=spec, delta_norm=delta_norm, weather_norm=weather_norm
        )
    return S4LatentPredictor(config=config, spec=spec, delta_norm=delta_norm)
