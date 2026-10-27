# S4B.2 Provenance

- Generated: 2026-10-28T03:31:08+0800
- Implementation commit (`git_commit` recorded by the trainer): `8d616150fdc675a77d9b1fa6c3313f7ba8e4dc35`
- Current HEAD: `fd2a993236ce776fc0f2f67afeddc3e2a4448884`
- Branch: `agro-s4b2-full-train`

## Frozen upstream / model

- Upstream DINO-WM commit: `0a9492fa12044b852ae9e001cc74604b79c8bb0c`
- DINOv2 visual encoder commit: `7764ea0f912e53c92e82eb78a2a1631e92725fc8`
- Visual weights sha256: `0b8b82f85de91b424aded121c7e1dcc2b7bc6d0adeea651bf73a13307fad8c73`
- Feature cache version: `mainline224_s1-v1` (shape [256, 768], float16 on disk)

## Data

- Full official-train pair table: `external_data/s4b_index/full_train_pairs.parquet` (sha `a49b546f57d688818db8f7c5b94504498572dfc39749c33e0e6608886d26e3c6`)
- Feature index: `external_data/s4b_features/index.parquet` (sha `3c0196f8592fe2b403d62af5e9094d41985cbe85e8b3837735b017a0c0615a0d`)
- Derived training index: `external_data/s4b2/train_index.parquet`
- S3 validation index: `external_data/s3_index/fip1_s3_v0_index.parquet` (sha `f6ef1871655277549aba2d883b6f76a5831b7976af90fc74d80206591d884933`)

Data audit:

- pairs **179,823**, plots **2,930**, split ['train'], train-only = True
- labels stripped (34 columns): cc_a, cc_b, ph_a, ph_b, whc_a, whc_b, sen_a, sen_b, cc_a_lag, cc_b_lag, ph_a_lag, ph_b_lag, whc_a_lag, whc_b_lag, sen_a_lag, sen_b_lag, cc_a_date, cc_b_date, ph_a_date, ph_b_date, whc_a_date, whc_b_date, sen_a_date, sen_b_date, cc_mask_a, cc_mask_b, ph_mask_a, ph_mask_b, whc_mask_a, whc_mask_b, sen_mask_a, sen_mask_b, ph_negative_a, ph_negative_b
- index sha256: `cfd170201b93be701594fec81664e03e2384fe5b2a258ca8bf9f0865132a07d8`
- cache coverage: 114,791 files, 0 missing, 0 bad shape

## Frozen epoch policy

- Rule: median over seeds 0,1,2 of (best_epoch + 1); best_epoch is 0-based
- Source roots: external_data/s4a1_runs, external_data/s4_runs

| condition | best_epoch (0-based) seeds 0/1/2 | +1 values | fixed epochs |
|---|---|---|---:|
| b2_no_time | [13, 13, 15] | [14, 14, 16] | 14 |
| b2_no_time_constant_delta | [12, 12, 12] | [13, 13, 13] | 13 |
| b3_delta | [11, 11, 11] | [12, 12, 12] | 12 |

Source checkpoints (frozen, must not change):

- `external_data/s4_runs/b2_no_time/seed0/checkpoints/best.pt` -> `9b9ae4ee557cc46c1edf75b67244fe939851eb97c52dddc76a2e034c87987006`
- `external_data/s4_runs/b2_no_time/seed1/checkpoints/best.pt` -> `2354a7b72a9574a30995b76b878fe9646d5c1a5b732ce788e732b546f3e49d40`
- `external_data/s4a1_runs/b2_no_time/seed2/checkpoints/best.pt` -> `108fb502662dae9f912af9fba6b8310db73831e17f3a4129773f769f4ebe5b7f`
- `external_data/s4a1_runs/b2_no_time_constant_delta/seed0/checkpoints/best.pt` -> `135bad41d4e463ce0e67b2f72084259278c603f8c55a1acaeb12ab6af7a70e36`
- `external_data/s4a1_runs/b2_no_time_constant_delta/seed1/checkpoints/best.pt` -> `fbae6262a1626be2f36f6be5c349208f86b1ddba51da5ee79e594a55a69f8b71`
- `external_data/s4a1_runs/b2_no_time_constant_delta/seed2/checkpoints/best.pt` -> `bbccf990afa4cf19885fcc88ef307446ec07b4d28f77f7466b2c4634b5b4c422`
- `external_data/s4_runs/b3_delta/seed0/checkpoints/best.pt` -> `e068b6f8653c424feec6ba2af33dcf39e4112743af574b23389cecab2e6ad9e0`
- `external_data/s4_runs/b3_delta/seed1/checkpoints/best.pt` -> `aa264c8aebc0df1e1557c715387cf7be9e4ce6f6a9d6d8c700317e271a4c52e6`
- `external_data/s4a1_runs/b3_delta/seed2/checkpoints/best.pt` -> `4d248312333d006b1b5fa4fc8584c318f205dac3bd3005a3a7a425d6885af082`

## Verification

- all_pass = **True** (core=True, reload=True, fresh-init=True, frozen-assets=True)
- steps/epoch = 1404, total optimizer steps over all 9 runs = 164,268
- warning: CLAUDE.md has a pre-existing worktree modification; it was not staged or committed by this stage

## E1 validation diagnostics (diagnostic only)

- 15,175 validation pairs / 380 plots
- bootstrap: plot-clustered, 2000 resamples
- b2_no_time: MSE 1.129680, cos 0.7842, vs persistence 48.54%
- b2_no_time_constant_delta: MSE 1.129654, cos 0.7842, vs persistence 48.54%
- b3_delta: MSE 1.075503, cos 0.7957, vs persistence 51.01%
