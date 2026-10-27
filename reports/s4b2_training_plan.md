# S4B.2 Frozen Training Plan

- Generated: 2026-10-27T10:00:14+0800
- Rule: median over seeds 0,1,2 of (best_epoch + 1); best_epoch is 0-based
- Source roots: external_data/s4a1_runs, external_data/s4_runs

| condition | seed0 best | seed1 best | seed2 best | +1 values | fixed epochs | matches config |
|---|---:|---:|---:|---|---:|---|
| b2_no_time | 13 | 13 | 15 | [14, 14, 16] | 14 | True |
| b2_no_time_constant_delta | 12 | 12 | 12 | [13, 13, 13] | 13 | True |
| b3_delta | 11 | 11 | 11 | [12, 12, 12] | 12 | True |

Source checkpoint SHA256:

- `external_data/s4_runs/b2_no_time/seed0/checkpoints/best.pt` -> `9b9ae4ee557cc46c1edf75b67244fe939851eb97c52dddc76a2e034c87987006`
- `external_data/s4_runs/b2_no_time/seed1/checkpoints/best.pt` -> `2354a7b72a9574a30995b76b878fe9646d5c1a5b732ce788e732b546f3e49d40`
- `external_data/s4a1_runs/b2_no_time/seed2/checkpoints/best.pt` -> `108fb502662dae9f912af9fba6b8310db73831e17f3a4129773f769f4ebe5b7f`
- `external_data/s4a1_runs/b2_no_time_constant_delta/seed0/checkpoints/best.pt` -> `135bad41d4e463ce0e67b2f72084259278c603f8c55a1acaeb12ab6af7a70e36`
- `external_data/s4a1_runs/b2_no_time_constant_delta/seed1/checkpoints/best.pt` -> `fbae6262a1626be2f36f6be5c349208f86b1ddba51da5ee79e594a55a69f8b71`
- `external_data/s4a1_runs/b2_no_time_constant_delta/seed2/checkpoints/best.pt` -> `bbccf990afa4cf19885fcc88ef307446ec07b4d28f77f7466b2c4634b5b4c422`
- `external_data/s4_runs/b3_delta/seed0/checkpoints/best.pt` -> `e068b6f8653c424feec6ba2af33dcf39e4112743af574b23389cecab2e6ad9e0`
- `external_data/s4_runs/b3_delta/seed1/checkpoints/best.pt` -> `aa264c8aebc0df1e1557c715387cf7be9e4ce6f6a9d6d8c700317e271a4c52e6`
- `external_data/s4a1_runs/b3_delta/seed2/checkpoints/best.pt` -> `4d248312333d006b1b5fa4fc8584c318f205dac3bd3005a3a7a425d6885af082`

**all_complete = True**
