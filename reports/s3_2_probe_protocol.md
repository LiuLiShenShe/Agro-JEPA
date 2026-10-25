# S3.2 — corrected E2 probe protocol

Stage: **S3.2**. Generated from `reports/s3_2_probe_protocol.json`.

## 1. C1 — what was wrong with the S3 early stopping

File: `agro_jepa/eval/eval_v0.py`; functions `fit_probe / _run_scope / e2_probes`.

fit_probe computes the validation MSE after every epoch and keeps the parameters from the epoch that minimise it; _run_scope passes the official validation split as x_val/y_val, and e2_probes then reports metrics on that very split.  The reported S3/S3.1 E2 numbers are therefore optimistic under model selection.

Evidence (line numbers in the committed file):

| code fact | lines | code |
|---|---|---|
| validation_mse_each_epoch | 437 | `val = float(torch.nn.functional.mse_loss(pred, yv_t))` |
| model_selection_on_validation | 438 | `if val < best_val - 1e-9:` |
| best_state_from_validation | 419, 441 | `best_state = copy.deepcopy(model.state_dict())` / `best_state = copy.deepcopy(model.state_dict())` |

Additional split usage: train_rows at lines [496, 582]; validation_rows at lines [497, 583]; probe_fit_called_with_validation at lines [519].

Correction: selection uses an inner dev set of official-train plots only; the final readout is refit on all official train plots with a fixed epoch rule and evaluated once on official validation.

## 2. C2 — inner split of the official train plots

| Item | Value |
|---|---|
| rule | official-train plots only; 80/20 by plot_uid, stratified by harvest_year; the same split is reused by every trait, scope, variant and condition |
| seed | 20261025 |
| inner-train fraction | 0.8 |
| strata | harvest_year |
| plots total / inner-train / inner-dev | 1000 / 800 / 200 |
| inner-train ∩ inner-dev | False |
| official validation plots in the split | 0 |
| split file | `reports/s3_2_probe_plot_split.csv` |

| harvest year | plots | inner-train | inner-dev |
|---|---|---|---|
| 2016 | 200 | 160 | 40 |
| 2017 | 200 | 160 | 40 |
| 2018 | 200 | 160 | 40 |
| 2021 | 200 | 160 | 40 |
| 2022 | 200 | 160 | 40 |

## 3. Sample counts per condition

| variant | trait | scope | train pairs | train plots | inner-train pairs | inner-train plots | inner-dev pairs | inner-dev plots | val pairs | val plots | abs(lag) ≤ d |
|---|---|---|---|---|---|---|---|---|---|---|---|
| primary_lag7 | cc | overall | 31146 | 1000 | 24931 | 800 | 6215 | 200 | 8496 | 380 | 7 |
| primary_lag7 | cc | span7 | 25128 | 1000 | 20121 | 800 | 5007 | 200 | 6542 | 380 | 7 |
| primary_lag7 | cc | span14 | 24675 | 1000 | 19742 | 800 | 4933 | 200 | 6322 | 380 | 7 |
| primary_lag7 | ph | overall | 25372 | 1000 | 20316 | 800 | 5056 | 200 | 6302 | 380 | 7 |
| primary_lag7 | ph | span7 | 21143 | 1000 | 16936 | 800 | 4207 | 200 | 5046 | 380 | 7 |
| primary_lag7 | ph | span14 | 21230 | 1000 | 16994 | 800 | 4236 | 200 | 4972 | 380 | 7 |
| strict_lag3 | cc | overall | 29948 | 1000 | 23973 | 800 | 5975 | 200 | 8112 | 380 | 3 |
| strict_lag3 | ph | overall | 23851 | 1000 | 19093 | 800 | 4758 | 200 | 5401 | 380 | 3 |
| same_day_lag0 | cc | overall | 28751 | 1000 | 23012 | 800 | 5739 | 200 | 7890 | 380 | 0 |
| same_day_lag0 | ph | overall | 16423 | 999 | 13143 | 800 | 3280 | 199 | 2908 | 374 | 0 |

## 4. C3 — inner model selection

Rule: minimum inner-dev MSE over the candidate budget (150 epochs, patience 20), the same rule for A and B.

| variant | trait | scope | A best epochs | A selected | B best epochs | B selected | shared |
|---|---|---|---|---|---|---|---|
| primary_lag7 | cc | overall | 38, 43, 123 | 43 | 38, 40, 33 | 38 | 39 |
| primary_lag7 | cc | span7 | 45, 29, 37 | 37 | 103, 113, 110 | 110 | 74 |
| primary_lag7 | cc | span14 | 44, 52, 89 | 52 | 68, 134, 97 | 97 | 78 |
| primary_lag7 | ph | overall | 39, 129, 51 | 51 | 38, 94, 85 | 85 | 68 |
| primary_lag7 | ph | span7 | 59, 19, 35 | 35 | 54, 125, 70 | 70 | 56 |
| primary_lag7 | ph | span14 | 101, 44, 39 | 44 | 81, 123, 47 | 81 | 64 |
| strict_lag3 | cc | overall | 67, 40, 66 | 66 | 62, 134, 48 | 62 | 64 |
| strict_lag3 | ph | overall | 135, 54, 29 | 54 | 61, 107, 76 | 76 | 68 |
| same_day_lag0 | cc | overall | 42, 32, 30 | 32 | 64, 103, 67 | 67 | 53 |
| same_day_lag0 | ph | overall | 48, 69, 57 | 57 | 19, 127, 110 | 110 | 63 |

Across all 60 inner-dev runs the best epoch ranges 19–135.  The dev loss surface is therefore flat and the epoch choice is only weakly identified; the shared-epoch control below is reported for that reason, and no conclusion in this stage rests on a small epoch difference.

## 5. C4 — final refit

After selection the epoch rule is frozen and the readout is retrained on **all** official-train plots for that trait/scope.  The cosine schedule is anchored to the same candidate budget as the inner runs, so a final run stopped at the selected epoch has exactly the parameters a full-budget run would have at that epoch.  No official-validation target is read during training or selection.  A shared-epoch variant (same epoch for A and B) is reported as a sensitivity check.

Caveat: the V0 checkpoint itself was selected on this validation split in S3, so this correction removes the *probe-level* early-stopping bias only. The split is not an untouched test set, and the formal external claim still requires the official test split.

### Inputs

| Item | Value |
|---|---|
| probe_samples | reports/s3_probe_samples.parquet |
| probe_samples_sha256 | 6e5bac5e326a5fe8a8366e159764865125c8f28214a96cd4e09bcec5ef5a2c45 |
| pooled_features | reports/s3_probe_pooled.npz |
| pooled_features_sha256 | 0c1444bfc4688450e91c8df02bcfc34bb3d7d4b87b5950002243ebfe588a7898 |
| pooled_shape | [64827, 768] |
| index_sha256 | f6ef1871655277549aba2d883b6f76a5831b7976af90fc74d80206591d884933 |
| checkpoint_sha256 | c337fca506e6a7e31fb90fca1f9197033992c2f8054287d76ec69b4c37426414 |
| pooling | mean over the 256 patch tokens (identical for A and B) |
| architecture | single linear layer (PooledReadout, 768 -> 1) |

### Alignment check against the S3 aggregates (old protocol re-run)

| condition | recomputed MAE / RMSE / R² | S3 reference | within tolerance |
|---|---|---|---|
| cc|A_current | 0.08206 / 0.10749 / 0.79780 | 0.08206 / 0.10750 / 0.79780 | True |
| cc|B_predicted | 0.06923 / 0.09386 / 0.84583 | 0.06923 / 0.09390 / 0.84580 | True |
| ph|A_current | 0.06030 / 0.08163 / 0.94696 | 0.06030 / 0.08160 / 0.94700 | True |
| ph|B_predicted | 0.05576 / 0.07595 / 0.95408 | 0.05576 / 0.07600 / 0.95410 | True |

## 6. Command

```bash
CUDA_VISIBLE_DEVICES=1 python3 tools/s3_2_probe_eval.py \
    --config agro_jepa/config/s3_2_diagnostics.yaml
python3 tools/s3_2_write_reports.py
```

Runtime 2305 s. Per-sample predictions: `reports/s3_2_probe_val_predictions.parquet` (371946 rows).
