# S0 — Upstream DINO-WM Audit (baseline landing)

Stage: S0 (baseline + code reading only; **no Agro-JEPA module is implemented here**).
Date: 2026-10-23.

## 0. TL;DR — pinned versions

| Item | Value |
|---|---|
| Upstream repo | `https://github.com/gaoyuezhou/dino_wm` |
| Upstream branch | `main` |
| **Upstream HEAD SHA** | **`0a9492fa12044b852ae9e001cc74604b79c8bb0c`** (`0a9492f`) |
| Upstream HEAD date / subject | 2025-03-24 15:46:34 -0400 / `add checkpoints` |
| Upstream history | 6 commits, no tags (`5851471` → `0a9492f`) |
| Local clone | `/fj/Agro-JEPA/dino_wm` (vendored; excluded from the Agro-JEPA repo by `.gitignore`) |
| Local working branch | `agro-s0-upstream` |
| Local commit on top of upstream | `c7417ce` — *only* `models/dino.py`, torch.hub explicit `:main` ref (see §6.1) |
| Upstream diff on working branch | 1 file, +3/−1 (`models/dino.py`) |
| DINOv2 code used at runtime | `torch.hub` cache `~/.cache/torch/hub/facebookresearch_dinov2_main` (repo zip, **no git SHA available** — see §6.2) |
| DINOv2 weights used | `dinov2_vits14_pretrain.pth` (84.2 MB), `dinov2_vitb14_pretrain.pth` (330 MB), downloaded from `dl.fbaipublicfiles.com` |
| License | MIT (see §2) |

Reproduce this provenance:

```bash
git -C /fj/Agro-JEPA/dino_wm log -1 --format='%H %ad %s'   # 0a9492fa... (main)
git -C /fj/Agro-JEPA/dino_wm log --oneline agro-s0-upstream  # c7417ce on top of 0a9492f
git -C /fj/Agro-JEPA/dino_wm diff main --stat                # models/dino.py | 4 +++-
```

## 1. Repository shape

```
train.py                     entry point for training          (Hydra config_path="conf", config_name="train")
plan.py                      entry point for planning           (Hydra config_name="plan")
conf/                        Hydra configs (env/, encoder/, predictor/, decoder/, action_encoder/, proprio_encoder/, planner/)
models/
  dino.py                    DinoV2Encoder  (frozen DINOv2 patch-token extractor)
  visual_world_model.py      VWorldModel    (encode → predict → latent loss → decode)
  vit.py                     ViTPredictor   (causal block-causal ViT over per-frame tokens)
  proprio.py                 ProprioceptiveEmbedding (Conv1d k=1, used for BOTH proprio and action)
  dummy.py                   DummyModel / DummyRepeatActionEncoder
  vqvae.py                   VQVAE decoder (quantize=False in the shipped config)
  decoder/transposed_conv.py alternative decoder (not used by train.yaml)
datasets/
  traj_dset.py               TrajDataset / TrajSubset / TrajSlicerDataset
  point_maze_dset.py, pusht_dset.py, wall_dset.py, deformable_env_dset.py
preprocessor.py              normalization helper, used by plan.py only
utils.py                     slicing / seeding / cfg helpers
metrics/image_metrics.py     L1/L2/SSIM/PSNR/LPIPS, called only when has_decoder
env/, planning/              simulator + planners; NOT imported by a training run (see §6.4)
```

`models/__init__.py` and `datasets/__init__.py` are empty: everything is resolved by Hydra
`_target_` strings, not by a registry.

## 2. License

- Root `LICENSE`: **MIT**, `Copyright (c) 2025 gaoyuezhou`.
- Nested asset license: `env/deformable_env/src/sim/assets/xarm/LICENSE` = UFACTORY Inc.
  BSD-3-clause-style license (2018). Only relevant to the deformable-environment simulator,
  which Agro-JEPA does not use.
- `metrics/lpipsPyTorch/modules/lpips.py` header states the code is
  "free for non-commercial, research and evaluation use under the terms of the LICENSE.md file",
  **but no `LICENSE.md` exists in the repo**. The vendored LPIPS is only executed when
  `has_decoder=True` (image metrics) and is not needed for latent-only training.
  Treat LPIPS as research-only / non-commercial.

## 3. Dependencies

`environment.yaml` pins a conda env (`dino_wm`) with **Python 3.9.19, torch 2.3.0,
torchvision 0.18.0, CUDA 12.1 wheels**. That file is the only supported reproduction target.

Split by actual import path:

### 3.1 Required to import and run `train.py`

| Package | env.yaml pin | Used by |
|---|---|---|
| torch / torchvision | 2.3.0 / 0.18.0 | everything, LPIPS/VGG metrics |
| numpy | 1.26.4 | datasets, metrics, utils |
| einops | 0.4.1 | all model reshapes |
| hydra-core | 1.2.0 | entry point, `hydra.utils.instantiate/call` |
| omegaconf | 2.3.0 | config objects |
| hydra-submitit-launcher | 1.2.0 | **config composition only** — `conf/train.yaml` has `override hydra/launcher: submitit_slurm` and `hydra.launcher.submitit_folder`, so the plugin must be importable even for a single local run |
| accelerate | 0.26.1 | `Accelerator`, `prepare`, `backward`, `gather_for_metrics` |
| wandb | 0.13.1 | imported unconditionally and `wandb.init` is called unconditionally; offline/disabled mode still needs the package |
| decord | 0.6.0 | imported at module import by **every** dataset file (even the `.pth`-based point_maze) |
| tqdm, psutil | 4.66.4 / 5.9.8 | progress bars, `utils` |
| submitit | 1.5.1 | launcher plugin dependency |

`metrics/image_metrics.py` is imported at `train.py:23` unconditionally, which imports the
vendored LPIPS. That is a hard import but a soft runtime path (only `eval_images()` is called
when `has_decoder=True`).

### 3.2 Required only for `plan.py` / simulation (NOT for training)

`mujoco-py==2.1.2.14`, `mujoco==3.2.7`, `gym==0.23.1`, `dm-control`, `robosuite`, `pyflex`,
`pybullet`, `d4rl`, `tensorflow`, `jax`, `flax`, etc. All of these come in through
`plan.py → env.venv → env/__init__.py` and the simulator wrappers.
`train.py` imports `plan` lazily and only when `cfg.plan_settings.plan_cfg_path is not None`
(`train.py:382-406`). The shipped default is `null` (`conf/train.yaml:77`), so **a training run
never imports mujoco/gym**. Do not install them for FIP.

### 3.3 External model code / weights downloaded at runtime

`models/dino.py:10` calls `torch.hub.load("facebookresearch/dinov2", name)`. Consequences:

- the DINOv2 *code* is the current `main` of `facebookresearch/dinov2`, downloaded as a zip
  into `~/.cache/torch/hub/facebookresearch_dinov2_main` (no commit hash recorded);
- the *weights* are fetched from `dl.fbaipublicfiles.com` into
  `~/.cache/torch/hub/checkpoints/`;
- this makes the encoder code/weights the least pinned part of the stack. Mitigation for S1:
  archive `~/.cache/torch/hub/facebookresearch_dinov2_main` + the two `.pth` files, or
  pin by switching to a git clone with `source="local"`.

## 4. Entry points and commands

| Entry | Command | Config |
|---|---|---|
| Train | `python train.py --config-name train.yaml env=point_maze frameskip=5 num_hist=3` | `conf/train.yaml` |
| Plan | `python plan.py --config-name plan_point_maze.yaml model_name=point_maze` | `conf/plan_point_maze.yaml` |

Training requires `DATASET_DIR` (dataset loader configs use `${oc.env:DATASET_DIR}`) and a
writable `ckpt_base_path` (default `./`). Hydra changes the process CWD to
`${ckpt_base_path}/outputs/<date>/<time>/`; `train.py:33` sets `saved_folder = os.getcwd()`.

`train.py` does not accept a `--seed` flag; the seed is `training.seed` (default 0).

## 5. What was actually executed on this host (evidence, not claims)

Host: Linux 6.17, 64 CPU, 2× RTX A6000 (GPU0 was busy with another job; smoke runs used it
anyway), Python 3.12.3, **torch 2.10.0+cu128 / torchvision 0.25.0** from the user site
(not the pinned 2.3.0), accelerate 1.15.0 (not 0.26.1).

Clean venv used for the runs: `/fj/Agro-JEPA/.venv-s0`
(`python3 -m venv --system-site-packages`; installed `hydra-core==1.3.2`,
`hydra-submitit-launcher==1.2.0`, `decord==0.6.0`, `wandb==0.31.0`). No conda on the host, so
the exact `environment.yaml` stack was **not** recreated.

Executed and passed:

1. **Backbone probe** (`tools/s0_forward_smoke.py`, step 1)
   - `dinov2_vits14`: 22.1 M params, `emb_dim=384`, `patch=14`, input 196×196 →
     patch tokens `(2, 196, 384)`.
   - `dinov2_vitb14`: 86.6 M params, `emb_dim=768`, `patch=14`, input 196×196 →
     patch tokens `(2, 196, 768)`.
2. **Model forward/backward** (`tools/s0_forward_smoke.py`), b=2, num_hist=3, num_pred=1,
   frameskip=5, img_size=224, proprio_dim=4, action_dim=2:
   - ViT-S + decoder: `z_pred (2,3,196,404)`, `visual_pred (2,3,3,224,224)`,
     `visual_rec (2,4,3,224,224)`, all losses finite, grads reach predictor/decoder but not
     encoder.
   - ViT-B + decoder: `z_pred (2,3,196,788)`, visual outputs unchanged in shape.
   - ViT-B rollout, 3 context frames + 5 actions: `z (1,9,196,788)`,
     `z_obs.visual (1,9,196,768)`.
   - `has_decoder=False`: `visual_pred=None`, `visual_rec=None`, loss = `z_loss` only.
   - `concat_dim=0` with shipped emb dims 10/10: **RuntimeError** (see §6.6).
   - `num_pred=2` with a window `T=num_hist+num_pred`: runs (see §6.7).
3. **End-to-end `train.py`** on a synthetic on-disk `point_maze` dataset (random pixels;
   metrics are meaningless).
   - `has_decoder=True` (the shipped default), 1 epoch, batch 4: **completed**.
     Train loss 2.7168, validation loss 2.6180; wrote
     `checkpoints/model_1.pth` and `model_latest.pth`, plus train/val rollout plots.
     Transcript: `reports/s0_run_logs/train_has_decoder_true.log`.
   - `has_decoder=False`, 1 epoch: **crashes in `save_ckpt`** with
     `KeyError: 'decoder_optimizer'` (see §6.13). It did complete training and validation
     (train loss 2.5662, val 2.4679) before the checkpoint step.
     Transcript: `reports/s0_run_logs/train_has_decoder_false.log`.
   This exercises: Hydra composition → `PointMazeDataset` → `TrajSlicerDataset` →
   `DataLoader` → `VWorldModel.forward` → `accelerator.backward` → optimizer steps →
   `openloop_rollout` → `save_ckpt`.

Artifacts of the runs (not part of the repo): `/tmp/s0_toy`, `/tmp/s0_run`,
`/tmp/s0_forward_smoke.log`, `/tmp/s0_train_e2e.log`.

## 6. Reproducibility findings and blockers (ordered by impact on Agro-JEPA)

### 6.1 torch.hub hangs on the default-branch probe (patched, branch-only)
`torch.hub.load("facebookresearch/dinov2", name)` calls `_parse_repo_info`, which does
`urlopen("https://github.com/facebookresearch/dinov2/tree/main/")` **on every construction**,
even when the repo is already cached. On this network the GitHub HTML endpoint stalls
indefinitely (the process sat on a socket for >5 min with no bytes), while git clone and
`codeload.github.com` work. `models/dino.py` now passes `"facebookresearch/dinov2:main"`,
which skips that probe and resolves the same ref and the same cache dir. Semantics unchanged.
Commit `c7417ce`. If the GitHub HTML endpoint is reachable in your environment, upstream
`main` works unmodified; keep this patch either way for startup latency.

### 6.2 DINOv2 code and weights are not pinned
`torch.hub` resolves the repo `main` branch and ignores a commit. Record and archive the
hub directory and `.pth` files before any FIP run, or switch `DinoV2Encoder` to
`torch.hub.load(local_path, name)` with a pinned local clone.

### 6.3 `models/vit.py:58` hard-codes CUDA device 0
```python
self.bias = generate_mask_matrix(NUM_PATCHES, NUM_FRAMES).to('cuda')
```
Consequences: (a) CPU-only execution is impossible; (b) on multi-GPU, the mask always lands on
`cuda:0` even if the model is on `cuda:1`; (c) it is a plain tensor, not a registered buffer, so
`.to(device)` does not move it; (d) `NUM_FRAMES`/`NUM_PATCHES` are **module globals** that are
overwritten by every `ViTPredictor` construction (`vit.py:106-108`), and the mask is baked at
construction. Two predictors with different `num_frames` in one process are not independent.
Any agro model that changes `num_hist` or the token layout must rebuild/replace this mask.

### 6.4 Mujoco / planning are not training dependencies
Confirmed by code path: `plan` is imported lazily at `train.py:386` only when
`plan_settings.plan_cfg_path != null`. Keep it `null` for FIP training. The planning stack also
requires a simulator to execute actions, so it is not reusable for a field-crop setting
without a new environment.

### 6.5 Checkpoints are full Python objects; `torch.load` breaks on torch ≥ 2.6
`save_ckpt` (`train.py:176-195`) pickles `nn.Module` and optimizer objects, not `state_dict`s.
`train.py:198`, `train.py:278/282`, `plan.py:355/383/387` call `torch.load(...)` without
`weights_only=False`. On torch ≥ 2.6 (host: 2.10) the default `weights_only=True` will refuse
to unpickle them. A fresh run works; **resume and planning will fail**. Minimal fix (when
needed): add `weights_only=False` at those call sites, or switch to `state_dict` checkpoints.
Do not do this in S0 without a compatibility test — it changes the checkpoint contract.

### 6.6 `concat_dim=0` is broken with the shipped embedding dims
`VWorldModel.encode` (`visual_world_model.py:98-101`) concatenates
`[visual (…,384), proprio (…,10), act (…,10)]` along the token dimension. Feature dims must
match; with `proprio_emb_dim=action_emb_dim=10` it raises
`RuntimeError: Sizes of tensors must match except in dimension 2. Expected size 384 but got
size 10`. Verified. The default config uses `concat_dim=1`, so the shipped baseline is fine.
If `concat_dim=0` is ever needed, set `proprio_emb_dim = action_emb_dim = encoder.emb_dim`
(verified to work: `num_patches=198`, `z (2,3,198,384)`).

### 6.7 `num_pred` "only supports 1" is a comment, not an invariant
`conf/train.yaml:59` says `num_pred: 1 # only supports 1`, but `forward()` slices
`z_tgt = z[:, num_pred:]` against a window `T = num_hist + num_pred`, so shapes align for any
`num_pred` (verified `num_pred=2`). What changes is semantics: the loss at window position `i`
targets frame `i + num_pred`. `openloop_rollout` ignores `num_pred` and advances one frame per
step. Treat the comment as the intended baseline; document any departure.

### 6.8 `openloop_rollout` can hang forever on short trajectories
`Trainer.openloop_rollout` (`train.py:632-720`) has a `while not valid_traj` loop with no
iteration cap. It needs `max_horizon > min_horizon + num_hist`, i.e. roughly
`T > (num_hist + min_horizon + 1) * frameskip`. With `num_hist=3, frameskip=5, min_horizon=2`,
a 30-frame trajectory gives `max_horizon == 5 == min_horizon` for every start, and validation
**never returns** (observed: the E2E run stalled after the train epoch; no error, no timeout).
A 60-frame trajectory completes. For FIP, either use long windows or cap/disable
`openloop_rollout`.

### 6.9 Short-trajectory filter uses the wrong threshold
`TrajSlicerDataset.__init__` (`traj_dset.py:64-73`) prints "Ignored short sequence" when
`T - num_frames < 0`, but generates slices only when `T - num_frames*frameskip >= 0`.
A trajectory with `num_frames <= T < num_frames*frameskip` produces zero slices and **no
warning**. With `num_hist+num_pred=4, frameskip=5`, a valid trajectory needs ≥ 20 raw frames.

### 6.10 Image-size assumptions
- `default_transform(224)` (`datasets/img_transforms.py`) is `Resize(224) → CenterCrop(224) →
  Normalize(0.5,0.5,0.5)`. `Resize(int)` on a non-square image preserves aspect ratio, then
  `CenterCrop` makes it square. Fine for square inputs.
- `VWorldModel` computes `encoder_image_size = (image_size // 16) * encoder.patch_size = 196`
  and applies `transforms.Resize(196)` (`visual_world_model.py:55-61`). `Resize(int)` on a
  **non-square** tensor scales the short side to 196 and the long side proportionally, which
  will not be divisible by 14 and will fail DINO's patch assertion. For square 224 inputs it
  is exact. Keep images square, or change to `Resize((196, 196))`.
- DINOv2 does **not** normalize inputs internally (the hub `PatchEmbed` is a bare conv; no
  mean/std). The only normalization is the dataset's `Normalize(0.5)`. Switching to ImageNet
  mean/std would change the frozen-feature distribution and break comparability with upstream
  results.

### 6.11 Config composition is coupled to the submitit launcher
`conf/train.yaml:9` has `override hydra/launcher: submitit_slurm` and
`hydra.launcher.submitit_folder`. A single local run composes fine **if**
`hydra-submitit-launcher` is installed (verified). Overriding `hydra/launcher=basic` fails at
composition with `Key 'submitit_folder' not in 'BasicLauncherConf'` (verified); deleting the key
from the CLI does not help. To run without the plugin you need a separate config file, which
S0 deliberately does not add.

### 6.12 Concept check (not a bug, but central to the project claim)
The encoder is frozen (`train_encoder: False`, `conf/train.yaml:68`) and the "predictor" is a
small causal ViT over **DINOv2 patch tokens**. There is no EMA teacher/student, no mask-token
prediction, and no I-JEPA-style latent target encoder. Calling this "I-JEPA" in Agro-JEPA
papers would be inaccurate; the honest description is "DINO-WM-style world model on frozen
DINOv2 features".

### 6.13 `has_decoder=False` cannot save a checkpoint (blocks the latent-only path)
Verified by execution. `_keys_to_save` (`train.py:166-168`) is built from
`self.train_decoder` (a *model* flag) but the decoder optimizer exists only when
`cfg.has_decoder` (`train.py:334-338`). With `train_decoder=True, has_decoder=False`,
`_keys_to_save` contains `"decoder_optimizer"` while `Trainer.__dict__` has no such
attribute, so `save_ckpt` raises `KeyError: 'decoder_optimizer'`. The train and validation
epochs complete; only the checkpoint write fails.

Practical impact: the latent-only configuration that Agro-JEPA needs first must either set
`model.train_decoder=False` (then `_keys_to_save` skips `decoder`/`decoder_optimizer`) or patch
the two flags to be keyed off `has_decoder`. Do not patch before recording the failure; the
minimal workaround is `model.train_decoder=False`.

### 6.14 `torch.load` needs `weights_only=False` on modern torch (verified)
`save_ckpt` stores full `nn.Module`/optimizer objects. Loading the produced checkpoint on this
host's torch 2.10:
- `torch.load(p, map_location="cpu")` → `UnpicklingError` ("Weights only load failed");
- `torch.load(p, map_location="cpu", weights_only=False)` → works, and unpickling requires
  `models/` to be importable (needs `sys.path` on the repo root).

Resuming a run and `plan.py`'s `load_ckpt` therefore need `weights_only=False` on torch ≥ 2.6.
Not changed in S0.

### 6.15 Checkpoint content and shapes (from a real save)
`checkpoints/model_1.pth` keys for `has_decoder=True, has_predictor=True, train_encoder=False`:
`epoch, predictor, predictor_optimizer, decoder, decoder_optimizer, action_encoder,
proprio_encoder` (the encoder is **not** saved, consistent with `_keys_to_save`). Observed
tensor shapes with ViT-S/14:
`predictor.pos_embedding (1, 588, 404)`, `predictor.to_qkv.weight (3072, 404)`,
`decoder.dec[0].weight (384,384,3,3)`, `decoder.upsample_b[0].weight (384,384,3,3)`.
For ViT-B/14 the corresponding predictor/decoder dims become 788/768.

### 6.16 `wandb` API compatibility
`plan.py:436` calls `wandb.run.name = ...`. With wandb 0.31 in disabled mode the run object is
`NoopRun`; setting the attribute works, but with `WANDB_MODE=disabled` a non-`wandb.init` run
object may be `RunDisabled` (the `isinstance` check at `plan.py:425` still passes). `train.py`
requires a *real* wandb run object: `wandb.init(..., mode="disabled")` is the tested working
combination. Do not rely on `RunDisabled`.

## 7. Minimal execution command (verified)

### 7.1 Upstream path with the real datasets (reproduction reference)

```bash
cd /fj/Agro-JEPA/dino_wm
# 1) dataset layout: $DATASET_DIR/point_maze/{states.pth,actions.pth,seq_lengths.pth,obses/}
export DATASET_DIR=/path/to/data
# 2) wandb is imported unconditionally; use offline/disabled to avoid a login
export WANDB_MODE=offline
# 3) the pinned conda env from environment.yaml (python 3.9.19, torch 2.3.0) is the reference
python train.py --config-name train.yaml env=point_maze frameskip=5 num_hist=3
```

Notes:
- No Mujoco/gym install is needed: `plan_settings.plan_cfg_path` is `null`.
- `hydra-submitit-launcher` must be importable for config composition (it is, in the pinned env).
- Output: `${ckpt_base_path}/outputs/<date>/<time>_point_maze_f5_h3_p1/checkpoints/model_*.pth`.

### 7.2 Smoke command actually executed here (synthetic data, no science)

```bash
cd /fj/Agro-JEPA
.venv-s0/bin/python tools/make_toy_pointmaze.py --out /tmp/s0_toy \
  --n-traj 4 --traj-len 60 --img 64

cd /fj/Agro-JEPA/dino_wm
DATASET_DIR=/tmp/s0_toy WANDB_MODE=disabled HYDRA_FULL_ERROR=1 \
/fj/Agro-JEPA/.venv-s0/bin/python train.py --config-name train.yaml \
  env=point_maze frameskip=5 num_hist=3 num_pred=1 img_size=224 \
  has_decoder=True has_predictor=True concat_dim=1 \
  training.epochs=1 training.batch_size=4 training.num_reconstruct_samples=2 \
  env.num_workers=0 ckpt_base_path=/tmp/s0_run_dec
```

This is the command whose transcript is in `reports/s0_run_logs/train_has_decoder_true.log`;
it is a pipeline check only. The synthetic pixels/states must never be reported as an
experiment. For the latent-only variant (no decoder, no image metrics), add
`has_decoder=False model.train_decoder=False` — with the upstream default
`model.train_decoder=True` the run trains and validates but dies in `save_ckpt`
(§6.13); that failure transcript is
`reports/s0_run_logs/train_has_decoder_false.log`.

## 8. Scope discipline

- No Agro-JEPA module was added in S0.
- The only change to the upstream clone is `models/dino.py` (commit `c7417ce`), documented above.
- No file in this repo outside `dino_wm/`, `tools/`, `reports/` was modified.
