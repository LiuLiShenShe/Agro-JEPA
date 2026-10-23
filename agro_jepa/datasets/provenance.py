"""Frozen provenance for the Agro-JEPA S2 data path.

Every constant here is copied from an S1 artifact and must not be inferred.
If any of them changes, the derived pair/feature caches are invalid.

Sources
-------
* DINOv2 code commit / weights digest: ``reports/s1_feature_cache_spec.md``
  and ``reports/s1_feature_cache_smoke.json``.
* Dataset revision: ``data/manifests/manifest_spec.json``.
* Preprocessing: ``tools/s1_dino_cache.py::build_transform`` (S1 protocol
  ``mainline224``): ToTensor -> Resize(224) -> CenterCrop(224) ->
  Normalize(0.5) -> DINOv2 ViT-B/14.  DINOv2 does no internal
  normalization, so this is the only normalization applied.
"""
from __future__ import annotations

import hashlib

import torchvision.transforms as T

# --- dataset ---------------------------------------------------------------
DATASET_NAME = "mikeboss/FIP1"
DATASET_REVISION = "11cdd84bc0ba8d5c6641ea7b53550b1658527e1c"
DATASET_LICENSE = "CC-BY-4.0"

# --- frozen vision encoder (S1) -------------------------------------------
DINOV2_MODEL = "dinov2_vitb14"
DINOV2_REPO_GIT_COMMIT = "7764ea0f912e53c92e82eb78a2a1631e92725fc8"
DINOV2_WEIGHTS_SHA256 = (
    "0b8b82f85de91b424aded121c7e1dcc2b7bc6d0adeea651bf73a13307fad8c73"
)
DINOV2_EMBED_DIM = 768
DINOV2_PATCH_SIZE = 14

# --- S2 cache protocol ----------------------------------------------------
CACHE_PROTOCOL = "mainline224"
CACHE_PROTOCOL_VERSION = "s1-v1"
MAINLINE_IMAGE_SIZE = 224          # encoder input
MAINLINE_N_TOKENS = 256            # (224 // 14) ** 2, 16x16 grid
UPSTREAM_IMAGE_SIZE = 196          # upstream DINO-WM path (224 // 16) * 14
UPSTREAM_N_TOKENS = 196            # (196 // 14) ** 2, 14x14 grid

# --- manifests ------------------------------------------------------------
S2_MANIFEST_ID = "fip1-s2-v1"
S1_MANIFEST_ID = "fip1-s1-v1"


def build_mainline_transform(image_size: int = MAINLINE_IMAGE_SIZE) -> T.Compose:
    """Return the exact S1 ``mainline224`` preprocessing pipeline.

    ``Resize`` before ``CenterCrop`` preserves aspect ratio and is what the
    upstream ``datasets.img_transforms.default_transform(224)`` does.  The
    identical chain is used by ``tools/s1_dino_cache.py``; a mismatch is
    detectable by the cache round-trip check in ``tools/s2_inspect_batch.py``.
    """
    if image_size != MAINLINE_IMAGE_SIZE:
        raise ValueError(
            f"S2 mainline protocol is pinned to {MAINLINE_IMAGE_SIZE}px; "
            f"got {image_size}.  A different size requires a new cache "
            "protocol id and must not reuse the existing .npy files."
        )
    return T.Compose([
        T.ToTensor(),                                   # uint8 HWC -> float CHW/255
        T.Resize(image_size),                           # upstream default_transform
        T.CenterCrop(image_size),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])


def cache_key_for_image(
    source_sha256: str,
    protocol: str = CACHE_PROTOCOL,
    protocol_version: str = CACHE_PROTOCOL_VERSION,
    weights_sha256: str = DINOV2_WEIGHTS_SHA256,
) -> str:
    """Cache key identical to ``tools/s1_dino_cache.py``.

    key = sha256("<image sha256>|<protocol>|<protocol-version>|<weights sha256>")
    """
    return hashlib.sha256(
        f"{source_sha256}|{protocol}|{protocol_version}|{weights_sha256}".encode()
    ).hexdigest()
