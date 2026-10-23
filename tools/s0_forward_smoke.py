"""Stage-0 interface smoke test for the UNMODIFIED upstream DINO-WM model stack.

What this verifies (no upstream file is touched):
  1. the hydra configs in conf/{encoder,predictor,decoder} instantiate as-is;
  2. DINOv2 ViT-S/14 and ViT-B/14 patch-token shapes at the 196px encoder input;
  3. VWorldModel forward/backward for concat_dim=1 and concat_dim=0;
  4. has_decoder=False (latent-only) forward/backward;
  5. the num_pred=1 constraint;
  6. rollout() shapes.

Run from the repo root that contains the dino_wm clone:
  .venv-s0/bin/python tools/s0_forward_smoke.py --repo dino_wm --devices cuda:0
"""

import argparse
import gc
import sys

import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf


def load_cfg(repo, rel, **overrides):
    cfg = OmegaConf.load(f"{repo}/conf/{rel}")
    for k, v in overrides.items():
        OmegaConf.update(cfg, k, v, force_add=True)
    return cfg


def build(repo, enc_name, concat_dim, has_decoder, num_hist, num_pred, frameskip,
          proprio_dim, action_dim, img_size, proprio_emb_dim=10, action_emb_dim=10):
    # --- exactly the instantiation order of train.py:init_models ---
    encoder = instantiate(load_cfg(repo, "encoder/dino.yaml", name=enc_name))
    # train.py:216-218 freezes the encoder when cfg.model.train_encoder is False
    for param in encoder.parameters():
        param.requires_grad = False
    proprio_encoder = instantiate(
        load_cfg(repo, "proprio_encoder/proprio.yaml"),
        in_chans=proprio_dim, emb_dim=proprio_emb_dim,
    )
    action_encoder = instantiate(
        load_cfg(repo, "action_encoder/proprio.yaml"),
        in_chans=action_dim * frameskip, emb_dim=action_emb_dim,
    )

    if encoder.latent_ndim == 1:
        num_patches = 1
    else:
        num_patches = (img_size // 16) ** 2  # train.py:247-249, hard-coded 16
    if concat_dim == 0:
        num_patches += 2  # train.py:251-252

    predictor = instantiate(
        load_cfg(repo, "predictor/vit.yaml"),
        num_patches=num_patches,
        num_frames=num_hist,
        dim=encoder.emb_dim + (proprio_emb_dim + action_emb_dim) * concat_dim,
    )
    decoder = None
    if has_decoder:
        decoder = instantiate(
            load_cfg(repo, "decoder/vqvae.yaml"), emb_dim=encoder.emb_dim
        )

    model_cfg = OmegaConf.create({
        "_target_": "models.visual_world_model.VWorldModel",
        "image_size": img_size,
        "num_hist": num_hist,
        "num_pred": num_pred,
        "train_encoder": False,
        "train_predictor": True,
        "train_decoder": True,
    })
    model = instantiate(
        model_cfg,
        encoder=encoder,
        proprio_encoder=proprio_encoder,
        action_encoder=action_encoder,
        predictor=predictor,
        decoder=decoder,
        proprio_dim=proprio_emb_dim,
        action_dim=action_emb_dim,
        concat_dim=concat_dim,
        num_action_repeat=1,
        num_proprio_repeat=1,
    )
    return model, num_patches


def probe_encoder(repo, enc_name, device):
    enc = instantiate(load_cfg(repo, "encoder/dino.yaml", name=enc_name)).to(device)
    x = torch.randn(2, 3, 196, 196, device=device)
    with torch.no_grad():
        out = enc(x)  # DinoV2Encoder.forward -> x_norm_patchtokens
    n_params = sum(p.numel() for p in enc.parameters()) / 1e6
    print(f"[encoder] {enc_name}: emb_dim={enc.emb_dim} patch={enc.patch_size} "
          f"params={n_params:.1f}M tokens={tuple(out.shape)}")
    del enc, x, out
    gc.collect()
    torch.cuda.empty_cache()


def one_step(model, b, num_hist, num_pred, frameskip, proprio_dim, action_dim,
             img_size, device, do_backward=True):
    t = num_hist + num_pred
    obs = {
        "visual": torch.rand(b, t, 3, img_size, img_size, device=device) * 2 - 1,
        "proprio": torch.randn(b, t, proprio_dim, device=device),
    }
    act = torch.randn(b, t, action_dim * frameskip, device=device)
    z_pred, visual_pred, visual_rec, loss, comps = model(obs, act)
    shapes = {
        "z_pred": None if z_pred is None else tuple(z_pred.shape),
        "visual_pred": None if visual_pred is None else tuple(visual_pred.shape),
        "visual_rec": None if visual_rec is None else tuple(visual_rec.shape),
        "loss": float(loss.detach()),
        "components": {k: round(float(v.detach()), 6) for k, v in comps.items()},
    }
    if do_backward:
        loss.backward()
        shapes["grad_predictor_pos"] = model.predictor.pos_embedding.grad is not None
        shapes["grad_decoder"] = (
            None if model.decoder is None
            else next(model.decoder.parameters()).grad is not None
        )
        shapes["grad_encoder"] = next(model.encoder.parameters()).grad is not None
    return shapes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="dino_wm")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    sys.path.insert(0, args.repo)

    device = torch.device(args.device)
    torch.manual_seed(0)

    # 1) encoder token shapes -------------------------------------------------
    for name in ("dinov2_vits14", "dinov2_vitb14"):
        probe_encoder(args.repo, name, device)

    common = dict(num_hist=3, num_pred=1, frameskip=5, proprio_dim=4,
                  action_dim=2, img_size=224)

    # 2) ViT-S, concat_dim=1, decoder on -------------------------------------
    model, npatch = build(args.repo, "dinov2_vits14", 1, True, **common)
    model = model.to(device)
    print(f"[model vit-s] num_patches={npatch} predictor_pos={tuple(model.predictor.pos_embedding.shape)} "
          f"dec_in={model.decoder.dec.blocks[0].in_channels} "
          f"dec_up_in={model.decoder.upsample_b.blocks[0].in_channels}")
    print("[forward vit-s dec=on]", one_step(model, 2, 3, 1, 5, 4, 2, 224, device))
    del model
    gc.collect(); torch.cuda.empty_cache()

    # 3) ViT-B, concat_dim=1, decoder on -------------------------------------
    model, npatch = build(args.repo, "dinov2_vitb14", 1, True, **common)
    model = model.to(device)
    print(f"[model vit-b] num_patches={npatch} predictor_pos={tuple(model.predictor.pos_embedding.shape)} "
          f"dec_in={model.decoder.dec.blocks[0].in_channels} "
          f"dec_up_in={model.decoder.upsample_b.blocks[0].in_channels}")
    print("[forward vit-b dec=on]", one_step(model, 2, 3, 1, 5, 4, 2, 224, device))
    with torch.no_grad():
        obs_0 = {"visual": torch.rand(1, 3, 3, 224, 224, device=device),
                 "proprio": torch.randn(1, 3, 4, device=device)}
        acts = torch.randn(1, 8, 2 * 5, device=device)
        z_obs, z = model.rollout(obs_0, acts)
    print(f"[rollout vit-b] z_obs.visual={tuple(z_obs['visual'].shape)} z={tuple(z.shape)}")
    del model, z_obs, z
    gc.collect(); torch.cuda.empty_cache()

    # 4) ViT-S, concat_dim=1, decoder OFF (has_decoder=False) ----------------
    model, npatch = build(args.repo, "dinov2_vits14", 1, False, **common)
    model = model.to(device)
    print("[forward vit-s dec=off]", one_step(model, 2, 3, 1, 5, 4, 2, 224, device))
    del model
    gc.collect(); torch.cuda.empty_cache()

    # 5) concat_dim=0 (extra action/proprio tokens) --------------------------
    # 5a. with the shipped emb dims (10/10) the cat() cannot work: visualized below
    try:
        model, npatch = build(args.repo, "dinov2_vits14", 0, False, **common)
        model.to(device)
        one_step(model, 2, 3, 1, 5, 4, 2, 224, device)
        print("[concat0 default emb dims] UNEXPECTED: forward succeeded")
    except RuntimeError as exc:
        print(f"[concat0 default emb dims] fails as documented: {exc}")
    gc.collect(); torch.cuda.empty_cache()

    # 5b. concat_dim=0 only works when proprio/action emb dims == encoder.emb_dim
    model, npatch = build(args.repo, "dinov2_vits14", 0, False, **common,
                          proprio_emb_dim=384, action_emb_dim=384)
    model = model.to(device)
    print(f"[model concat0/proj] num_patches={npatch} predictor_pos={tuple(model.predictor.pos_embedding.shape)}")
    print("[forward concat0/proj dec=off]", one_step(model, 2, 3, 1, 5, 4, 2, 224, device))
    del model
    gc.collect(); torch.cuda.empty_cache()

    # 6) num_pred > 1: the config comment says "only supports 1", but the
    # slicing in forward() is shape-consistent for any num_pred as long as the
    # dataset window is T = num_hist + num_pred.  What breaks is semantics:
    # predict() re-predicts the whole input window, so the loss compares frame
    # i (pred) against frame i + num_pred (target).  rollout() ignores num_pred.
    model, _ = build(args.repo, "dinov2_vits14", 1, False,
                     num_hist=3, num_pred=2, frameskip=5, proprio_dim=4,
                     action_dim=2, img_size=224)
    model = model.to(device)
    print("[num_pred=2]", one_step(model, 2, 3, 2, 5, 4, 2, 224, device))


if __name__ == "__main__":
    main()
