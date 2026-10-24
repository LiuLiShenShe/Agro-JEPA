"""S3 V0: FIP-DINO-WM latent growth predictor.

What this module is
-------------------
A frozen DINOv2 ViT-B/14 encodes a real FIP inner-plot image at ``224x224`` into
``256`` patch tokens of width ``768`` (the ``mainline224`` protocol of S1/S2).
The trainable part is **only** the predictor: it receives the current image's
patch tokens and the real ``delta_days`` to the next observation and predicts
the next image's patch tokens::

    Z_hat_b = Predictor(Z_a, delta_days)          # [B, 256, 768]
    L       = MSE(Z_hat_b, stopgrad(Z_b))

This is *not* the I-JEPA / V-JEPA EMA teacher.  The target encoder is a frozen
DINOv2 checkpoint; there is no momentum update, no masked-token objective and no
pixel decoder.  The report says so explicitly.

Reuse of upstream DINO-WM
-------------------------
The transformer is the upstream ``models.vit.ViTPredictor`` (vendored clone at
the SHA pinned in ``README.md``), and the time/placeholder channels use the
upstream ``models.proprio.ProprioceptiveEmbedding``.  The injection follows the
upstream ``concat_dim=1`` contract:

    z = cat([z_visual (256 x 768),
             proprio_repeated (256 x proprio_dim),
             action_repeated  (256 x action_dim)], dim=-1)
    z_hat = predictor(z)[..., :768]

The **action channel is the time conditioning**: a trainable
``DeltaTimeEncoder`` maps ``delta_days`` to ``action_emb_dim`` values, which the
upstream action encoder repeats ``num_action_repeat`` times.  The **proprio
channel is a fixed constant placeholder** (zeros) so the upstream interface is
kept; it soaks up no supervision and is never produced from future
CC/PH/weather.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import nn

from .time_embedding import DeltaEmbeddingConfig, DeltaNormStats, DeltaTimeEncoder

# --------------------------------------------------------------------------
# upstream import
# --------------------------------------------------------------------------
def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _ensure_upstream_on_path() -> None:
    """Make the vendored, git-ignored ``dino_wm`` clone importable.

    The clone is reconstructed from the pinned SHA (README "Reconstruct the
    vendored upstream").  A missing clone is a hard error: silently falling
    back to a local re-implementation would make the "built on DINO-WM"
    claim untraceable.
    """
    root = _repo_root()
    if (root / "dino_wm" / "models" / "vit.py").exists():
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        return
    raise ImportError(
        "The vendored dino_wm clone is missing. Reconstruct it from the pinned "
        "SHA (README.md -> 'Reconstruct the vendored upstream') before running S3."
    )


def _load_upstream():
    _ensure_upstream_on_path()
    from dino_wm.models.proprio import ProprioceptiveEmbedding  # noqa: PLC0415
    from dino_wm.models.vit import ViTPredictor  # noqa: PLC0415

    return ViTPredictor, ProprioceptiveEmbedding


def upstream_available() -> bool:
    try:
        _load_upstream()
        return True
    except ImportError:
        return False


def move_attention_masks(predictor: nn.Module, device: torch.device | str) -> None:
    """Move upstream ``bias`` tensors that are plain attributes, not buffers.

    ``models/vit.py`` builds ``self.bias = generate_mask_matrix(...).to('cuda')``
    inside every ``Attention``.  Because it is not a registered buffer,
    ``predictor.to(device)`` does not move it.  This helper makes the device
    explicit and also supports CPU smoke tests.
    """
    for module in predictor.modules():
        if hasattr(module, "bias") and isinstance(module.bias, torch.Tensor):
            module.bias = module.bias.to(device)


# --------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------
@dataclass
class FIPDinoWMConfig:
    """All architectural constants of V0.

    Defaults are the upstream ViT predictor width/depth plus the
    ``mainline224`` geometry (256 tokens x 768).
    """

    # geometry -- pinned by S1/S2, not free parameters
    num_patches: int = 256
    visual_dim: int = 768

    # upstream ViTPredictor
    depth: int = 6
    heads: int = 16
    mlp_dim: int = 2048
    dropout: float = 0.1
    emb_dropout: float = 0.0
    pool: str = "mean"

    # channel bookkeeping (upstream concat_dim=1 semantics)
    action_emb_dim: int = 10
    num_action_repeat: int = 16
    proprio_emb_dim: int = 10
    num_proprio_repeat: int = 16

    # time conditioning
    delta: DeltaEmbeddingConfig = field(default_factory=DeltaEmbeddingConfig)

    def predictor_dim(self) -> int:
        return (
            self.visual_dim
            + self.proprio_emb_dim * self.num_proprio_repeat
            + self.action_emb_dim * self.num_action_repeat
        )

    def action_dim(self) -> int:
        return self.action_emb_dim * self.num_action_repeat

    def proprio_dim(self) -> int:
        return self.proprio_emb_dim * self.num_proprio_repeat

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
            "predictor_dim": self.predictor_dim(),
            "action_dim": self.action_dim(),
            "proprio_dim": self.proprio_dim(),
            "time_conditioning": self.delta.as_dict(),
            "proprio_placeholder": "constant zeros (no future phenotype input)",
        }

    @classmethod
    def from_dict(cls, d: dict) -> "FIPDinoWMConfig":
        d = dict(d)
        delta = d.pop("delta", None)
        cfg = cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
        if delta:
            cfg.delta = DeltaEmbeddingConfig(**{
                k: v for k, v in delta.items() if k in DeltaEmbeddingConfig.__dataclass_fields__
            })
        return cfg


# --------------------------------------------------------------------------
# frozen encoder (for live-image verification / eval; training uses the cache)
# --------------------------------------------------------------------------
class FrozenDinoV2Encoder(nn.Module):
    """DINOv2 ViT-B/14 loaded from the pinned local clone, fully frozen."""

    def __init__(
        self,
        repo_path: str | Path = "external_data/third_party/dinov2",
        model_name: str = "dinov2_vitb14",
        expected_commit: str | None = "7764ea0f912e53c92e82eb78a2a1631e92725fc8",
        device: str | torch.device = "cuda",
    ) -> None:
        super().__init__()
        repo = Path(repo_path)
        if not (repo / "hubconf.py").exists():
            raise FileNotFoundError(
                f"{repo} is not the pinned dinov2 clone (no hubconf.py)"
            )
        if expected_commit:
            import subprocess

            got = subprocess.check_output(
                ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
            ).strip()
            if got != expected_commit:
                raise RuntimeError(
                    f"dinov2 clone is at {got}, expected the pinned {expected_commit}"
                )
        base = torch.hub.load(
            str(repo.resolve()), model_name, source="local", pretrained=True,
            verbose=False,
        )
        self.base_model = base
        self.name = model_name
        self.emb_dim = int(base.num_features)
        self.patch_size = int(base.patch_size)
        self.latent_ndim = 2
        for p in self.base_model.parameters():
            p.requires_grad_(False)
        self.base_model.eval().to(device)
        self._device = torch.device(device)

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """``images`` [B,3,224,224] -> patch tokens [B,256,768]."""
        if not torch.is_grad_enabled():
            feats = self.base_model.forward_features(images)
            return feats["x_norm_patchtokens"]
        # even if a caller enables grad, the encoder must stay frozen
        with torch.no_grad():
            feats = self.base_model.forward_features(images)
        return feats["x_norm_patchtokens"]


# --------------------------------------------------------------------------
# V0 predictor
# --------------------------------------------------------------------------
class FIPDinoWM(nn.Module):
    """Time-conditioned latent predictor on frozen DINOv2 patch tokens.

    ``forward(z_a, delta_days)`` returns ``[B, 256, 768]``.  The only loss the
    trainer applies is ``MSE(pred, stopgrad(z_b))`` on those visual tokens.
    """

    def __init__(
        self,
        config: FIPDinoWMConfig | None = None,
        norm_stats: DeltaNormStats | None = None,
    ) -> None:
        super().__init__()
        ViTPredictor, ProprioceptiveEmbedding = _load_upstream()
        self.config = config or FIPDinoWMConfig()
        cfg = self.config

        self.predictor = ViTPredictor(
            num_patches=cfg.num_patches,
            num_frames=1,
            dim=cfg.predictor_dim(),
            depth=cfg.depth,
            heads=cfg.heads,
            mlp_dim=cfg.mlp_dim,
            pool=cfg.pool,
            dropout=cfg.dropout,
            emb_dropout=cfg.emb_dropout,
        )

        if cfg.delta.output_dim != cfg.action_emb_dim:
            raise ValueError(
                "DeltaEmbeddingConfig.output_dim must equal action_emb_dim "
                f"({cfg.delta.output_dim} != {cfg.action_emb_dim})"
            )
        self.delta_encoder = DeltaTimeEncoder(config=cfg.delta, norm_stats=norm_stats)
        self.action_encoder = ProprioceptiveEmbedding(
            num_frames=1, tubelet_size=1, use_3d_pos=False,
            in_chans=cfg.action_emb_dim, emb_dim=cfg.action_emb_dim,
        )
        self.proprio_encoder = ProprioceptiveEmbedding(
            num_frames=1, tubelet_size=1, use_3d_pos=False,
            in_chans=1, emb_dim=cfg.proprio_emb_dim,
        )
        # constant placeholder input; its value carries no information
        self.register_buffer("proprio_placeholder", torch.zeros(1, 1, 1))

        # keep the upstream interface visible
        self.concat_dim = 1
        self.num_action_repeat = cfg.num_action_repeat
        self.num_proprio_repeat = cfg.num_proprio_repeat
        self.action_dim = cfg.action_dim()
        self.proprio_dim = cfg.proprio_dim()
        self.emb_dim = cfg.visual_dim

    # -- time channel -------------------------------------------------------
    def encode_delta(self, delta_days: torch.Tensor) -> torch.Tensor:
        """``delta_days`` (B,) -> repeated action channel (B, 1, action_dim)."""
        raw = self.delta_encoder(delta_days)                       # (B, action_emb_dim)
        emb = self.action_encoder(raw[:, None, :])                 # (B, 1, action_emb_dim)
        return emb.repeat(1, 1, self.num_action_repeat)            # (B, 1, action_dim)

    def encode_proprio(self, batch_size: int, device: torch.device) -> torch.Tensor:
        raw = self.proprio_placeholder.to(device).expand(batch_size, -1, -1)
        emb = self.proprio_encoder(raw)                            # (B, 1, proprio_emb_dim)
        return emb.repeat(1, 1, self.num_proprio_repeat)           # (B, 1, proprio_dim)

    # -- forward ------------------------------------------------------------
    def forward(
        self,
        z_a: torch.Tensor,
        delta_days: torch.Tensor,
        return_components: bool = False,
    ):
        if z_a.dim() != 3 or z_a.shape[1:] != (self.config.num_patches, self.config.visual_dim):
            raise ValueError(
                f"z_a must be [B,{self.config.num_patches},{self.config.visual_dim}], "
                f"got {tuple(z_a.shape)}"
            )
        b = z_a.shape[0]
        delta = delta_days.to(z_a.device).reshape(b)
        act = self.encode_delta(delta)                             # (B,1,action_dim)
        proprio = self.encode_proprio(b, z_a.device)               # (B,1,proprio_dim)
        proprio = proprio.expand(b, self.config.num_patches, self.proprio_dim)
        act = act.expand(b, self.config.num_patches, self.action_dim)
        z = torch.cat([z_a, proprio, act], dim=-1)                 # (B,256,predictor_dim)
        z_all = self.predictor(z)
        z_visual = z_all[..., : self.config.visual_dim]
        if return_components:
            return z_visual, {
                "action": act[:, 0, :],
                "proprio": proprio[:, 0, :],
                "predictor_dim": z_all.shape[-1],
            }
        return z_visual

    # -- convenience --------------------------------------------------------
    @torch.no_grad()
    def predict(self, z_a: torch.Tensor, delta_days: torch.Tensor) -> torch.Tensor:
        was_training = self.training
        self.eval()
        out = self.forward(z_a, delta_days)
        self.train(was_training)
        return out

    @torch.no_grad()
    def predict_from_images(
        self,
        image_a: torch.Tensor,
        delta_days: torch.Tensor,
        encoder: "FrozenDinoV2Encoder",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Convenience wrapper for the literal ``predict(image_a, delta)`` API.

        The frozen encoder maps a real ``[B,3,224,224]`` image to ``z_a``; the
        trainable predictor then maps ``(z_a, delta_days)`` to the future patch
        tokens ``[B,256,768]``.  Training uses the S1/S2 cache directly and does
        not call this method.
        """
        z_a = encoder(image_a.to(next(encoder.parameters()).device))
        return self.predict(z_a, delta_days), z_a

    def set_delta_norm_stats(self, stats: DeltaNormStats) -> None:
        self.delta_encoder.set_norm_stats(stats)

    def n_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def trainable_parameter_names(self) -> list[str]:
        return [n for n, p in self.named_parameters() if p.requires_grad]


# --------------------------------------------------------------------------
# loss / metrics
# --------------------------------------------------------------------------
def latent_prediction_loss(
    z_pred: torch.Tensor,
    z_target: torch.Tensor,
    visual_dim: int = 768,
) -> torch.Tensor:
    """MSE on the visual patch tokens only.

    ``z_target`` is detached by the caller (or here) so the frozen encoder and
    the future image can never receive gradient.  Placeholder/proprio channels
    are not part of the tensors this function sees.
    """
    if z_pred.shape != z_target.shape:
        raise ValueError(f"shape mismatch {z_pred.shape} vs {z_target.shape}")
    return nn.functional.mse_loss(z_pred, z_target.detach(), reduction="mean")


@torch.no_grad()
def latent_metrics(z_pred: torch.Tensor, z_target: torch.Tensor) -> dict:
    """Per-batch MSE / cosine and the persistence baseline (Z_a is the caller's)."""
    mse = nn.functional.mse_loss(z_pred, z_target, reduction="mean").item()
    cos = nn.functional.cosine_similarity(
        z_pred.flatten(1), z_target.flatten(1), dim=-1
    ).mean().item()
    return {"mse": mse, "cosine": cos}
