# S3.2 — corrected E2 probe results (CC / PH)

Stage: **S3.2**. Generated from `reports/s3_2_probe_results.json` and `reports/s3_2_probe_protocol.json`.

A = current DINOv2 feature `mean(Z_a)`; B = V0 predicted future feature `mean(Z_hat_b)`. Both use the same linear readout, pooling, optimiser, budget, plot split and seeds. Selection uses inner dev only; the reported numbers are the final refits on all official-train plots evaluated once on official validation.

## 1. Headline — primary protocol (|lag| ≤ 7 d, per-condition epochs)

| trait | scope | readout | MAE | RMSE | R² | n val |
|---|---|---|---|---|---|---|
| cc | overall | A current | 0.08343 | 0.10876 | 0.79297 | 8496 |
| cc | overall | B predicted | 0.07100 | 0.09562 | 0.83999 | 8496 |
| cc | overall | **B − A** | -0.01242 | -0.01314 | +0.04702 | — |
| cc | span7 | A current | 0.08083 | 0.10510 | 0.81135 | 6542 |
| cc | span7 | B predicted | 0.06865 | 0.09289 | 0.85267 | 6542 |
| cc | span7 | **B − A** | -0.01217 | -0.01221 | +0.04132 | — |
| cc | span14 | A current | 0.08336 | 0.10819 | 0.80553 | 6322 |
| cc | span14 | B predicted | 0.07015 | 0.09598 | 0.84694 | 6322 |
| cc | span14 | **B − A** | -0.01321 | -0.01221 | +0.04141 | — |
| ph | overall | A current | 0.06183 | 0.08373 | 0.94419 | 6302 |
| ph | overall | B predicted | 0.05597 | 0.07611 | 0.95389 | 6302 |
| ph | overall | **B − A** | -0.00585 | -0.00762 | +0.00969 | — |
| ph | span7 | A current | 0.06041 | 0.08101 | 0.94571 | 5046 |
| ph | span7 | B predicted | 0.05369 | 0.07364 | 0.95514 | 5046 |
| ph | span7 | **B − A** | -0.00672 | -0.00738 | +0.00944 | — |
| ph | span14 | A current | 0.06192 | 0.08355 | 0.94403 | 4972 |
| ph | span14 | B predicted | 0.05664 | 0.07746 | 0.95189 | 4972 |
| ph | span14 | **B − A** | -0.00528 | -0.00609 | +0.00786 | — |

## 2. New protocol vs the S3/S3.1 protocol (overall scope)

The S3 columns are the old protocol re-run from the same pooled features (validation early stopping); the S3.2 columns use inner-dev selection and a full-train refit.

| trait | readout | MAE old → S3.2 | R² old → S3.2 | Δ(B − A) MAE old → S3.2 | Δ(B − A) R² old → S3.2 |
|---|---|---|---|---|---|
| cc | A current | 0.08206 → 0.08343 | 0.79780 → 0.79297 | — | — |
| cc | B predicted | 0.06923 → 0.07100 | 0.84583 → 0.83999 | — | — |
| cc | **Δ (B − A)** | — | — | +0.01283 → +0.01242 | +0.04802 → +0.04702 |
| ph | A current | 0.06030 → 0.06183 | 0.94696 → 0.94419 | — | — |
| ph | B predicted | 0.05576 → 0.05597 | 0.95408 → 0.95389 | — | — |
| ph | **Δ (B − A)** | — | — | +0.00455 → +0.00585 | +0.00712 → +0.00969 |

The old columns reproduce the S3/S3.1 numbers from the same pooled features; the S3.2 columns come from the corrected protocol.

## 3. Plot-level cluster bootstrap (Δ = B − A)

2,000 resamples of `plot_uid`, fixed seed; negative ΔMAE/ΔRMSE and positive ΔR² favour the predicted future feature.  Before resampling, the per-sample predictions are averaged over the 3 seeds, which slightly reduces the point errors; that is why the bootstrap mean Δ can differ in the third decimal from the per-seed mean Δ in §1.

| variant | trait | scope | n | plots | ΔMAE [95% CI] | ΔRMSE [95% CI] | ΔR² [95% CI] |
|---|---|---|---|---|---|---|---|
| primary_lag7 | cc | overall | 8496 | 380 | -0.01136 [-0.01266, -0.01007] | -0.01210 [-0.01397, -0.01025] | +0.04284 [+0.03664, +0.04894] |
| primary_lag7 | cc | span7 | 6542 | 380 | -0.01018 [-0.01158, -0.00879] | -0.00997 [-0.01181, -0.00804] | +0.03331 [+0.02698, +0.03909] |
| primary_lag7 | cc | span14 | 6322 | 380 | -0.01280 [-0.01431, -0.01141] | -0.01169 [-0.01380, -0.00960] | +0.03940 [+0.03254, +0.04631] |
| primary_lag7 | ph | overall | 6302 | 380 | -0.00522 [-0.00643, -0.00397] | -0.00675 [-0.00823, -0.00522] | +0.00849 [+0.00657, +0.01034] |
| primary_lag7 | ph | span7 | 5046 | 380 | -0.00591 [-0.00708, -0.00473] | -0.00665 [-0.00812, -0.00521] | +0.00843 [+0.00659, +0.01027] |
| primary_lag7 | ph | span14 | 4972 | 380 | -0.00544 [-0.00678, -0.00412] | -0.00618 [-0.00783, -0.00451] | +0.00790 [+0.00576, +0.01010] |
| strict_lag3 | cc | overall | 8112 | 380 | -0.01249 [-0.01386, -0.01116] | -0.01350 [-0.01547, -0.01166] | +0.04606 [+0.04037, +0.05205] |
| strict_lag3 | ph | overall | 5401 | 380 | -0.00548 [-0.00680, -0.00412] | -0.00757 [-0.00920, -0.00593] | +0.00968 [+0.00751, +0.01179] |
| same_day_lag0 | cc | overall | 7890 | 380 | -0.01204 [-0.01333, -0.01078] | -0.01291 [-0.01479, -0.01106] | +0.04359 [+0.03799, +0.04947] |
| same_day_lag0 | ph | overall | 2908 | 374 | -0.00281 [-0.00436, -0.00126] | -0.00299 [-0.00474, -0.00129] | +0.00427 [+0.00185, +0.00688] |

## 4. Shared-epoch sensitivity

Both conditions use the same epoch (median of the six inner-dev runs), so the B − A difference cannot be attributed to B being trained longer.

| trait | scope | epochs | MAE A | MAE B | ΔMAE | R² A | R² B | ΔR² |
|---|---|---|---|---|---|---|---|---|
| cc | overall | 39 | 0.08371 | 0.07205 | -0.01166 | 0.78895 | 0.83533 | +0.04638 |
| cc | span7 | 74 | 0.08002 | 0.06969 | -0.01033 | 0.81433 | 0.84846 | +0.03414 |
| cc | span14 | 78 | 0.08287 | 0.07030 | -0.01258 | 0.80666 | 0.84624 | +0.03958 |
| ph | overall | 68 | 0.06098 | 0.05578 | -0.00519 | 0.94560 | 0.95404 | +0.00844 |
| ph | span7 | 56 | 0.06042 | 0.05556 | -0.00486 | 0.94533 | 0.95360 | +0.00827 |
| ph | span14 | 64 | 0.06203 | 0.05692 | -0.00511 | 0.94342 | 0.95144 | +0.00802 |

## 5. Label-timing sensitivity

Same protocol, re-selected and refitted on each label subset (overall scope).

| label set | trait | n val | plots | MAE A | MAE B | ΔMAE | R² A | R² B | ΔR² |
|---|---|---|---|---|---|---|---|---|---|
| abs(lag) ≤ 7 d | cc | 8496 | 380 | 0.08343 | 0.07100 | -0.01242 | 0.79297 | 0.83999 | +0.04702 |
| abs(lag) ≤ 7 d | ph | 6302 | 380 | 0.06183 | 0.05597 | -0.00585 | 0.94419 | 0.95389 | +0.00969 |
| abs(lag) ≤ 3 d | cc | 8112 | 380 | 0.08064 | 0.06819 | -0.01244 | 0.80635 | 0.85313 | +0.04678 |
| abs(lag) ≤ 3 d | ph | 5401 | 380 | 0.06056 | 0.05385 | -0.00671 | 0.94418 | 0.95579 | +0.01161 |
| lag = 0 | cc | 7890 | 380 | 0.07972 | 0.06729 | -0.01243 | 0.81061 | 0.85661 | +0.04600 |
| lag = 0 | ph | 2908 | 374 | 0.05788 | 0.05336 | -0.00452 | 0.94388 | 0.95088 | +0.00701 |

PH negatives, zeros and quality flags are carried through unchanged; no clipping is applied anywhere in this stage.

## 6. Interpretation

Read the per-condition Δ together with its bootstrap interval.  A positive ΔR² and a negative ΔMAE/ΔRMSE whose CI excludes zero means the predicted future feature still carries more future-phenotype information than the current feature under a protocol whose model selection never touched official validation.

- **CC**: ΔR² +0.04284 [+0.03664, +0.04894], ΔMAE -0.01136 [-0.01266, -0.01007] on 8,496 validation pairs / 380 plots — the predicted feature is better and the interval excludes zero.
- **PH**: ΔR² +0.00849 [+0.00657, +0.01034], ΔMAE -0.00522 [-0.00643, -0.00397] on 6,302 validation pairs / 380 plots — the predicted feature is better and the interval excludes zero.

Caveats that still apply: the V0 checkpoint was selected on this validation split in S3; the probe is a linear readout on mean-pooled tokens; and the official test split has not been used.

## 7. Command

```bash
CUDA_VISIBLE_DEVICES=1 python3 tools/s3_2_probe_eval.py \
    --config agro_jepa/config/s3_2_diagnostics.yaml
python3 tools/s3_2_write_reports.py
```
