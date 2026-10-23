# Agro-JEPA

Environment-driven latent growth prediction on FIP 1.0 winter-wheat imagery, built as an
incremental evolution of the official DINO-WM training code.

**Stage 0 status: baseline landing + upstream code audit only. No Agro-JEPA model module
exists yet.**

## Layout

```
Agro-JEPA/
├── reports/
│   ├── s0_upstream_audit.md     # provenance, license, deps, entry points, blockers, commands
│   ├── s0_api_map.md            # real call chain, tensor shapes, change-interface list,
│   │                            #   module -> upstream class/function map
│   └── s0_run_logs/             # transcripts of the actual S0 runs
├── tools/
│   ├── make_toy_pointmaze.py    # synthetic dataset in point_maze on-disk format
│   └── s0_forward_smoke.py      # shape/forward/backward/rollout probe
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
