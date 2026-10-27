# S4B.2 Full-Train Training Results

- Generated: 2026-10-28T03:31:08+0800
- Config: `agro_jepa/config/s4b2_full_train.yaml` (sha `45e641f02150040b...`)
- Runs present: **9 / 9**
- Frozen epoch budget: B2 (no time) = 14, B2-C (fixed 9.0 d) = 13, B3 (real delta) = 12
- Batch 128, lr 0.0003, wd 0.05, warmup 200, amp bf16, grad-clip 1.0
- Train pairs/epoch: 179,823 -> steps/epoch 1,404
- No early stopping, no epoch re-selection, official validation not used for selection, official test never read.

## Per-run

| condition | seed | epochs | global step | first loss | last loss | wall (s) | samples/s | peak MiB | params | final sha (16) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| b2_no_time | 0 | 14 | 19,656 | 1.32363 | 1.11483 | 6788 | 370.7 | 13185 | 45,898,196 | `08c9dafc450a70a2` |
| b2_no_time | 1 | 14 | 19,656 | 1.32355 | 1.11366 | 6684 | 376.4 | 13185 | 45,898,196 | `7b1b3827c5e6a6d8` |
| b2_no_time | 2 | 14 | 19,656 | 1.32356 | 1.11320 | 6696 | 375.8 | 13185 | 45,898,196 | `540bbb96dbb350ff` |
| b2_no_time_constant_delta | 0 | 13 | 18,252 | 1.31388 | 1.11260 | 6710 | 348.2 | 13760 | 53,882,396 | `d8cebf42d2f97ebe` |
| b2_no_time_constant_delta | 1 | 13 | 18,252 | 1.31463 | 1.11199 | 6722 | 347.5 | 13760 | 53,882,396 | `61a0d962309d3361` |
| b2_no_time_constant_delta | 2 | 13 | 18,252 | 1.31444 | 1.11334 | 6736 | 346.8 | 13760 | 53,882,396 | `9391faa54f26c676` |
| b3_delta | 0 | 12 | 16,848 | 1.29577 | 1.03880 | 6194 | 348.2 | 13760 | 53,882,396 | `f433164646886c69` |
| b3_delta | 1 | 12 | 16,848 | 1.29647 | 1.03836 | 6175 | 349.2 | 13760 | 53,882,396 | `64ee70f8e4574ba7` |
| b3_delta | 2 | 12 | 16,848 | 1.29671 | 1.03744 | 6121 | 352.3 | 13760 | 53,882,396 | `37ca926d05f7ac95` |

## Per-condition aggregate (mean over seeds)

| condition | epochs | step | params (trainable/cond/dim) | last loss mean±std | wall mean (s) | samples/s |
|---|---:|---:|---|---:|---:|---:|
| b2_no_time | 14 | 19,656 | 45,898,196 / 0 / 928 | 1.11389 ± 0.00084 | 6722 | 374.3 |
| b2_no_time_constant_delta | 13 | 18,252 | 53,882,396 / 72,730 / 1088 | 1.11265 ± 0.00067 | 6723 | 347.5 |
| b3_delta | 12 | 16,848 | 53,882,396 / 72,730 / 1088 | 1.03820 ± 0.00070 | 6163 | 349.9 |

## Artifacts

- `external_data/s4b2_runs/b2_no_time/seed0/checkpoints/final.pt` (sha `08c9dafc450a70a2c24a47216f30c0350c600b858e5eb5ef8847c6dbb3f34b04`)
- `external_data/s4b2_runs/b2_no_time/seed1/checkpoints/final.pt` (sha `7b1b3827c5e6a6d84322dda1b397b0cde9ea4a6ce239f6494a8bf615a5b31503`)
- `external_data/s4b2_runs/b2_no_time/seed2/checkpoints/final.pt` (sha `540bbb96dbb350ffb80cb0fab715e33ce2f36550843b04b14f11cab5e50a57df`)
- `external_data/s4b2_runs/b2_no_time_constant_delta/seed0/checkpoints/final.pt` (sha `d8cebf42d2f97ebe27dbf2cfeb231669cbf4f6ff5e60e330649399a3cf292757`)
- `external_data/s4b2_runs/b2_no_time_constant_delta/seed1/checkpoints/final.pt` (sha `61a0d962309d3361a5bc1291e13af6b3908f40caf6201c7b9f543d1e823d9d20`)
- `external_data/s4b2_runs/b2_no_time_constant_delta/seed2/checkpoints/final.pt` (sha `9391faa54f26c6760a11d86601bed7e94225b7a5e2d3c4b35cd1c593a1316230`)
- `external_data/s4b2_runs/b3_delta/seed0/checkpoints/final.pt` (sha `f433164646886c691eb2a5d795924167c0f114c0b95f5d261722df565e8a0b5a`)
- `external_data/s4b2_runs/b3_delta/seed1/checkpoints/final.pt` (sha `64ee70f8e4574ba7bff47b2206e6d19a1fed03c34a0e9b59fcc63463afe9ab7f`)
- `external_data/s4b2_runs/b3_delta/seed2/checkpoints/final.pt` (sha `37ca926d05f7ac959710027efc57997173fb4275e2ed67ae5d277e2c55d33e21`)

Run logs: `reports/s4b2_run_logs/`.
