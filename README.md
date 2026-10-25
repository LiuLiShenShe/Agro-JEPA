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
│   ├── s3_2_delta_diagnostics.py  # S3.2 A0–A5 controlled Δt interventions
│   ├── s3_2_probe_eval.py        # S3.2 corrected E2 probe protocol (C1–C6)
│   ├── s3_2_verify.py            # S3.2 leakage/integrity checks (D1–D10)
│   ├── s3_2_write_reports.py    # S3.2 reports
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

### Reproduce the S3.2 diagnostics (frozen V0, no retraining)

```bash
# 1) A0–A5 controlled Δt interventions + plot-level bootstrap (~22 min)
CUDA_VISIBLE_DEVICES=1 python3 tools/s3_2_delta_diagnostics.py \
    --config agro_jepa/config/s3_2_diagnostics.yaml
# 2) corrected E2 probe protocol: inner split, selection, refit, bootstrap (~38 min)
CUDA_VISIBLE_DEVICES=1 python3 tools/s3_2_probe_eval.py \
    --config agro_jepa/config/s3_2_diagnostics.yaml
# 3) leakage / integrity checks (10 checks, all recomputed from the artifacts)
python3 tools/s3_2_verify.py --config agro_jepa/config/s3_2_diagnostics.yaml
# 4) markdown reports from the JSON artifacts
python3 tools/s3_2_write_reports.py
```

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

### S3.2 controlled time diagnostics + corrected E2 protocol (no retraining)

- **Same-pair_type permutation** (span7/span14 kept separate, 20 seeds) raises
  MSE by **+6.91% [6.71, 7.13]**, versus **+5.61%** for the per-pair_type train
  mean and **+6.65%** for the pair_type × harvest_year permutation. The global
  permutation remains **+11.46%**. So roughly **6 points of the global 11.5
  points** survive when the short/long class is preserved: the model does use
  within-span interval information, but most of the time-channel effect comes
  from the coarse span/growth level, and the precise interval is a modest
  refinement.
- Per-interval breakdown: the model is most sensitive at Δt=5 (A2 +13.5%,
  A3 +17.3%) and least at Δt=7 (A3 +0.1%), i.e. the sensitivity is not uniform
  across the realised interval.
- **E2 protocol fixed**: the S3 `fit_probe` selected its best epoch on official
  validation. S3.2 selects on an inner 200-plot dev set carved from the official
  train plots (80/20 by plot, stratified by year), then refits on all 1,000
  train plots and evaluates once on validation. B still beats A everywhere:
  overall CC ΔR² **+0.047 [+0.037, +0.049]**, PH ΔR² **+0.010 [+0.0066, +0.010]**;
  the shared-epoch control gives the same ordering. Absolute R² drops slightly
  (CC A 0.7978 → 0.7930, B 0.8458 → 0.8400), consistent with removing
  selection bias.
- 10/10 leakage/integrity checks pass (`tools/s3_2_verify.py`).
- Reports: `reports/s3_2_delta_diagnostics.md`, `reports/s3_2_probe_protocol.md`,
  `reports/s3_2_probe_results.md`, `reports/s3_2_verify.json`.

## S4A: time / GDD / weather baselines (pilot)

S4A holds the controlled comparisons. Every trainable condition is trained
under one fixed protocol (official-train plots only, 80/20 inner split,
inner-dev early stopping, seeds {0,1}, 16-epoch candidate budget, identical
visual target). Official validation is read only for the final Pilot
comparison; official test is never read.

| Condition | Input | Protocol |
|---|---|---|
| `b0_persistence` | `Z_a` | — |
| `b1_train_mean` | inner-train mean `Z_b` | — |
| `b2_no_time` | `z_a` only (time input removed from the network) | — |
| `b3_delta` | `z_a + delta_days` (V0 match, retrained) | — |
| `b4_gdd_R` | `z_a + interval GDD` | R |
| `b4_delta_gdd_R` | `z_a + delta + interval GDD` | R |
| `b4_delta_gdd_R_tbase5` | same, `T_base = 5 °C` | R |
| `b4_delta_gdd_P` | `z_a + delta + past 7/14/28 d GDD` | P |
| `b5_delta_weather_P` | `z_a + delta + GRU(past 28 d weather)` | P |
| `b5_delta_weather_R` | `z_a + delta + GRU((t_a, t_b] weather)` | R |

Protocol P is deployable (weather strictly before `t_a`); Protocol R is
retrospective/oracle interval conditioning and is labelled as such everywhere.

Pilot results (official validation, 15,175 pairs / 380 plots, never used for
selection; 2,000-resample plot bootstrap):

- Future-latent MSE: persistence 2.19515, train-mean 1.83561, B2 1.17733
  (46.4% below persistence), B3 1.13245 (48.4%), best B5 weather-R 1.12412
  (48.8%).
- Δt beyond the visual state: B3 − B2 = **−0.04486 [−0.04642, −0.04330]**
  (3.8% of B2); permuting Δt at inference raises MSE by 9.95%.
- GDD beyond Δt: past-GDD P **−0.00641 [−0.00750, −0.00532]**, interval-GDD R
  −0.00399 [−0.00513, −0.00287]; `T_base = 5` is 0.00144 better than 0.
- Weather beyond Δt: past-only P (deployable) **−0.00215 [−0.00308,
  −0.00116]**, interval R (retrospective) −0.00833 [−0.00957, −0.00710].
- The P→R gap is not uniform: past GDD beats interval GDD by 0.00242, while
  interval weather beats past weather by 0.00619 — consistent with past GDD
  acting as a phenology/stage proxy and interval weather acting as oracle
  information about the target window.
- CC readout (S3.2 protocol, lag ≤ 7 d): A 0.79297 → B2 0.82102 → B3 0.84654 →
  weather-R 0.85370; PH: 0.94419 → 0.95130 → 0.95542 → 0.95777. Weather-P −
  B3 is +0.00095 [−0.00175, +0.00358] on CC (not significant) but +0.00179
  [+0.00081, +0.00276] on PH; weather-R − B3 is +0.00697 [+0.00365, +0.01028]
  on CC.
- Conditioning permutation at inference (10 permutations, plot bootstrap) gives
  MSE increases of +9.95% (Δt), +7.04% (past-GDD P), +14.78% (interval-GDD R),
  +14.60% (past-weather P), +18.01% (interval-weather R) when the input is
  shuffled within a harvest year; shuffling across years gives
  +9.98 / +7.49 / +16.66 / +16.14 / +23.29% (Δt nearly unchanged, since it
  carries no year identity). So the conditioning is genuinely consumed and not
  just year identity, but the 0.45–5.3 pp gap between the two permutations
  (largest for interval weather) shows that part of its apparent value is
  year-level environment; the small gains over B3 mean most of the consumed
  information is redundant with Δt and the visual state.
- 44 of 183 validation genotypes (6,987 pairs) have no training plot; the Δt
  and weather/GDD gains reproduce there (weather-R − B3 = −0.00836 unseen vs
  −0.00831 seen; GDD-P − B3 = −0.00717 vs −0.00576).
- 15/15 data/leakage and model checks pass (`tools/s4_verify.py`).

```bash
# 1) derive per-pair GDD + weather tensors (Protocol P and R side by side)
python3 tools/s4_prepare_weather.py --config agro_jepa/config/s4_pilot.yaml
# 2) train every condition / seed under the shared budget (~10 h on one A6000)
CUDA_VISIBLE_DEVICES=1 python3 tools/s4_train_pilot.py \
    --config agro_jepa/config/s4_pilot.yaml
# 3) E1 future-latent baselines + E2 CC/PH probes + plot bootstrap
CUDA_VISIBLE_DEVICES=1 python3 agro_jepa/eval/eval_s4.py --mode all
# 4) inference-only conditioning permutation diagnostics
CUDA_VISIBLE_DEVICES=1 python3 tools/s4_conditioning_diagnostics.py
# 5) controls and derived analyses (CPU only)
CUDA_VISIBLE_DEVICES=1 python3 tools/s4_e2_shared_epoch.py
python3 tools/s4_e2_contrasts.py --config agro_jepa/config/s4_pilot.yaml
python3 tools/s4_generalization_split.py --config agro_jepa/config/s4_pilot.yaml
# 6) acceptance checks and reports
python3 tools/s4_verify.py --config agro_jepa/config/s4_pilot.yaml
python3 tools/s4_write_reports.py --config agro_jepa/config/s4_pilot.yaml
```

Results are in `reports/s4_pilot_results.md` / `.json`; the protocol card is
`reports/s4_data_protocol.md`, and the eight stop-question answers (with the
explicit "cannot claim" list) are in `reports/s4_stop_questions.md`.

Added for S4A:

```
agro_jepa/datasets/s4_dataset.py        # conditioning spec + aligned dataset
agro_jepa/models/s4_conditioning.py     # GDD and masked-weather encoders
agro_jepa/models/s4_predictor.py        # shared conditional predictor + factory
agro_jepa/models/gdd_conditioned_predictor.py
agro_jepa/models/weather_gru_predictor.py
agro_jepa/train/train_s4.py
agro_jepa/eval/s4_common.py
agro_jepa/eval/eval_s4.py
agro_jepa/config/s4_pilot.yaml
tools/s4_prepare_weather.py
tools/s4_train_pilot.py
tools/s4_conditioning_diagnostics.py
tools/s4_e2_contrasts.py
tools/s4_e2_shared_epoch.py
tools/s4_generalization_split.py
tools/s4_verify.py
tools/s4_write_reports.py
```

## Next stage entry point (S5)

S4A is a controlled pilot. S4B (full 2,930-plot training + official test
splits) is not started and must not be started without an explicit go-ahead.
The upstream VQVAE decoder is **incompatible** with 256 tokens, so all latent
stages stay decoder-free (`has_decoder=False`, `model.train_decoder=False`).
