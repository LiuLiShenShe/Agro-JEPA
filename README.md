# Agro-JEPA

Environment-driven latent growth prediction on FIP 1.0 winter-wheat imagery, built as an
incremental evolution of the official DINO-WM training code.

**Stage 3 status: FIP-DINO-WM V0 is trained and evaluated on real FIP 1.0
winter-wheat imagery — a frozen DINOv2 ViT-B/14 feature predictor conditioned on
the real `delta_days`, using the upstream DINO-WM `ViTPredictor`. This is a
Pilot, not a full-FIP baseline (1,000 / 2,930 train plots; all 380 validation
plots).**

Real measured result on the official validation split (15,175 pairs, 380 plots):
future-latent MSE **1.13656** vs persistence **2.19515** (48.22% reduction),
mean cosine **0.7829** vs **0.6279**, and the predicted features beat the
current feature on both CC and PH linear probes. See `reports/s3_v0_results.md`.

## Layout

```
Agro-JEPA/
├── agro_jepa/                   # project code (added on top of upstream)
│   ├── datasets/
│   │   ├── fip_pairs.py         # consecutive + fixed-span pair builder
│   │   ├── fip_weather.py       # daily weather store, (t_a, t_b] windows
│   │   ├── fip_dataset.py       # torch Dataset (images or cached features)
│   │   ├── s3_pairs.py          # S3 frozen pair index + cached-feature Dataset
│   │   ├── collate.py           # variable-length padding + masks
│   │   └── provenance.py        # pinned model/dataset/cache identifiers
│   ├── models/
│   │   ├── time_embedding.py    # train-only delta normalisation + time encoder
│   │   └── fip_dino_wm.py       # frozen DINOv2 encoder + upstream ViTPredictor
│   ├── train/train_v0.py        # V0 training, AMP, checkpoint/resume
│   ├── eval/eval_v0.py          # E1 future-latent + E2 CC/PH probes
│   ├── config/s2_dataset.yaml   # one config for the whole data path
│   ├── config/s3_v0.yaml        # V0 (span7/span14)
│   └── config/s3_v0_consecutive.yaml  # V0 + irregular consecutive intervals
├── data/manifests/              # frozen S1 manifests (committed)
├── reports/
│   ├── s0_upstream_audit.md     # provenance, license, deps, entry points, blockers, commands
│   ├── s0_api_map.md            # real call chain, tensor shapes, change-interface list,
│   │                            #   module -> upstream class/function map
│   ├── s1_*.md / s1_*.json      # FIP audit, splits, label timing, cache protocol
│   ├── s2_pairs.md              # S2 pair construction and real counts
│   ├── s2_real_batch.md         # S2 real batch + encoder geometry verification
│   ├── s3_image_acquisition.md  # real image download, cache, S3 index
│   ├── s3_training.md           # V0 model, optimisation, curve, checkpoint
│   ├── s3_v0_results.md         # E1 future-latent + E2 CC/PH probes
│   └── s*_run_logs/             # transcripts of the actual runs
├── tools/
│   ├── make_toy_pointmaze.py    # synthetic dataset in point_maze on-disk format
│   ├── s0_forward_smoke.py      # shape/forward/backward/rollout probe
│   ├── s1_*.py                  # FIP download, audit, cache, figures
│   ├── s2_build_pairs.py        # build and report the S2 pair table
│   ├── s2_inspect_batch.py      # real batch + frozen-encoder verification
│   ├── s2_verify.py             # S2 acceptance checks (74 checks)
│   ├── s3_prepare_images.py     # resumable real-image download (train/validation only)
│   ├── s3_build_features.py     # frozen DINOv2 feature cache
│   ├── s3_build_index.py        # frozen S3 pair index + train-only time stats
│   ├── s3_verify.py             # S3 acceptance checks (45 checks)
│   ├── s3_make_figures.py       # training-curve figure
│   ├── s3_write_reports.py      # reports from the run artifacts
│   ├── s3_1_diagnostics.py      # S3.1 provenance + global Δt shuffle + bootstrap
│   ├── s3_1_probe_predictions.py  # per-sample probe predictions for bootstrap
│   └── s3_1_write_reports.py    # S3.1 reports
├── patches/
│   └── 0001-...patch            # the only upstream edit (torch.hub ':main' ref)
└── dino_wm/                     # vendored clone, git-ignored (not a submodule)
```

`dino_wm/` is excluded from this repository's history on purpose: the upstream code is pinned
by SHA and reconstructed with the commands below, so upstream history is never duplicated.

## Pinned versions

| Item | Value |
|---|---|
| Upstream | `https://github.com/gaoyuezhou/dino_wm` |
| Upstream branch | `main` |
| **Upstream HEAD** | **`0a9492fa12044b852ae9e001cc74604b79c8bb0c`** |
| Upstream license | MIT (2025 gaoyuezhou) |
| Upstream working branch | `agro-s0-upstream` |
| Upstream change | `c7417ce` — `models/dino.py` only; `torch.hub.load(..., "…:main")` |
| This repo branch | `agro-s0-upstream` |

## Reconstruct the vendored upstream

```bash
cd /fj/Agro-JEPA
git clone https://github.com/gaoyuezhou/dino_wm.git dino_wm
git -C dino_wm checkout 0a9492fa12044b852ae9e001cc74604b79c8bb0c
git -C dino_wm checkout -b agro-s0-upstream
git -C dino_wm am ../patches/0001-fix-hub-pass-explicit-main-ref-to-torch.hub.load-for.patch
git -C dino_wm log --oneline -2     # patch on top of 0a9492f (SHA may differ from c7417ce)
git -C dino_wm diff main --stat     # must be exactly models/dino.py | 4 +++-
```

## Run the S0 checks

```bash
python3 -m venv --system-site-packages .venv-s0
.venv-s0/bin/pip install hydra-core==1.3.2 hydra-submitit-launcher==1.2.0 decord wandb

.venv-s0/bin/python tools/make_toy_pointmaze.py --out /tmp/s0_toy --n-traj 4 --traj-len 60
.venv-s0/bin/python tools/s0_forward_smoke.py --repo dino_wm --device cuda:0

cd dino_wm
DATASET_DIR=/tmp/s0_toy WANDB_MODE=disabled HYDRA_FULL_ERROR=1 \
  ../.venv-s0/bin/python train.py --config-name train.yaml \
  env=point_maze frameskip=5 num_hist=3 num_pred=1 img_size=224 \
  has_decoder=True has_predictor=True concat_dim=1 \
  training.epochs=1 training.batch_size=4 env.num_workers=0 \
  ckpt_base_path=/tmp/s0_run_dec
```

These are pipeline checks on **synthetic random data**; the numbers they print are not
results. The paper-grade stack is the `dino_wm` conda environment in `dino_wm/environment.yaml`
(Python 3.9.19, torch 2.3.0).

## What S0 established

- Exact upstream SHA, branch, MIT license, dependency split (training vs planning), entry points.
- Verified DINOv2 ViT-S/14 (196×384) and ViT-B/14 (196×768) behavior at the encoder input
  `196×196`, and the corresponding predictor/decoder feature dims (404 vs 788 / 384 vs 768).
- Verified `VWorldModel` forward/backward for decoder on/off, `concat_dim=0/1`, `num_pred=1/2`,
  and rollout shapes.
- End-to-end `train.py` run on a synthetic dataset, including checkpoint write.
- 16 reproducible findings (see the audit), including four that directly constrain Agro-JEPA:
  `concat_dim=0` is broken with the shipped embedding dims, `has_decoder=False` crashes at
  checkpoint save unless `model.train_decoder=False`, `openloop_rollout` can hang on short
  trajectories, and the DINOv2 code/weights are not SHA-pinned.
- Mujoco / gym / planning are **not** required for training (`plan_settings.plan_cfg_path=null`
  by default; `plan` is imported lazily).

## Next stage entry point

Read `reports/s0_api_map.md` §6–§7: every planned Agro-JEPA module is mapped to the upstream
class/function it attaches to (dataset adapter, weather forcing via the action channel,
phenotype anchoring via proprio, predictor swap, rollout evaluator, loss masking).

## Reproduce the S1 + S2 data path

```bash
.venv-s0/bin/python tools/s2_build_pairs.py \
    --config agro_jepa/config/s2_dataset.yaml --mode both \
    --out external_data/pairs_cache/fip1_pairs_s2.parquet \
    --report reports/s2_pairs_stats.json --sample-csv reports/s2_pairs_sample.csv

.venv-s0/bin/python tools/s2_inspect_batch.py \
    --config agro_jepa/config/s2_dataset.yaml --batch-size 8 \
    --report reports/s2_real_batch.json

.venv-s0/bin/python tools/s2_verify.py     # 74/74 acceptance checks
```

Real numbers: 569,821 pairs (147,092 consecutive + 4 fixed-span families),
569,821/569,821 weather windows fully covered, `[8, 256, 768]` patch tokens from
`224×224` inputs. Full detail in `reports/s2_pairs.md` and
`reports/s2_real_batch.md`.

## Reproduce the S3 data + V0 model

```bash
# 1) real images (resumable; only official train/validation plots)
python3 tools/s3_prepare_images.py --train-plots-per-year 200 --workers 40
# 2) frozen DINOv2 features, mainline224_s1-v1 protocol
CUDA_VISIBLE_DEVICES=1 .venv-s0/bin/python tools/s3_build_features.py
# 3) frozen S3 pair index (span7/span14; time stats fitted on train only)
.venv-s0/bin/python tools/s3_build_index.py --config agro_jepa/config/s3_v0.yaml
# 4) train V0 (frozen encoder, upstream ViTPredictor, AMP, resume-capable)
CUDA_VISIBLE_DEVICES=1 .venv-s0/bin/python agro_jepa/train/train_v0.py \
    --config agro_jepa/config/s3_v0.yaml --report reports/s3_training.json
# 5) E1 + E2 evaluation and acceptance
CUDA_VISIBLE_DEVICES=1 .venv-s0/bin/python agro_jepa/eval/eval_v0.py \
    --config agro_jepa/config/s3_v0.yaml \
    --checkpoint external_data/s3_runs/v0/checkpoints/best.pt \
    --report reports/s3_v0_results.json \
    --predictions reports/s3_v0_val_predictions.parquet
CUDA_VISIBLE_DEVICES=1 .venv-s0/bin/python tools/s3_verify.py   # 45/45
python3 tools/s3_make_figures.py --config agro_jepa/config/s3_v0.yaml
python3 tools/s3_write_reports.py --config agro_jepa/config/s3_v0.yaml
```

The irregular-interval expansion uses `agro_jepa/config/s3_v0_consecutive.yaml`:
span7 + span14 + true consecutive pairs, 103,368 pairs (81,751 train / 21,617
validation), realised `delta_days` 1–84.

## What S3 established

- Real imagery: 49,058 PNGs / 53.35 GB / 1,380 plots / 5 harvest years, 0 failed
  downloads. Validation is complete at plot level (380/380); train is a Pilot
  (1,000/2,930 plots).
- Frozen DINOv2 ViT-B/14 `224×224 → [256, 768]` tokens, cached once; cache vs
  live forward cos ≈ 0.999999.
- V0 = upstream `ViTPredictor` (6 layers, 16 heads, width 1088) + trainable
  sinusoidal time projection from the real `delta_days`; 53.88 M trainable
  parameters; the future image is only a stop-gradient target.
- Validation E1: MSE **1.13656** vs persistence **2.19515** (−48.22%) and vs
  train-mean future **1.83567** (−38.08%); cosine **0.7829** vs **0.6279**.
- E2 linear probes (same readout, 3 seeds): predicted future features beat the
  current feature on CC (R² 0.8458 vs 0.7978) and PH (0.9541 vs 0.9470).
- 45/45 S3 acceptance checks pass. Full detail in `reports/s3_v0_results.md`.

### S3.1 diagnostics (corrections, not new training)

- The S3 Δt-shuffle diagnostic was **batch-local**; its 0.41% figure is kept as
  history only. A true global permutation over all 15,175 validation pairs
  (20 seeds) raises MSE by **11.46%**, essentially equal to replacing every
  interval with the train mean (**11.44%**): the time channel matters, but it
  behaves as a broad growth-level adjustment rather than precise interval
  resolution.
- Plot-cluster bootstrap (2,000 resamples over 380 plots): MSE reduction vs
  persistence **48.23% [47.97%, 48.49%]**, vs train-mean **38.10% [37.71%, 38.48%]**.
- E2 probes were re-run to save per-sample predictions; the B−A gains remain
  positive with CIs excluding zero (CC ΔR² +0.046 [0.040, 0.053]; PH +0.0075
  [0.0058, 0.0092]).
- Provenance: the checkpoint records `74efcf8` + an uncommitted S3 tree; the
  committed `c1fdb88` code reproduces the stored predictions exactly
  (max per-sample diff 0.0). See `reports/s3_1_diagnostic_audit.md` and
  `reports/s3_1_provenance.md`.

## Next stage entry point (S4)

S4 holds the strong-baseline comparisons (weather encoder / GDD forcing,
physiological phenotype anchoring in the proprio channel, EMA-style target,
LoRA adaptation, multi-step rollout). V0 deliberately contains none of them.
The upstream VQVAE decoder is **incompatible** with 256 tokens, so all latent
stages stay decoder-free (`has_decoder=False`, `model.train_decoder=False`).
