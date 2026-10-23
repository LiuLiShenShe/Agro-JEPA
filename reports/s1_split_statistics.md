# S1 — Split and sample statistics (real data)

Companion to `reports/s1_fip_audit.md`. Every number below is computed by
`tools/s1_dataset_stats.py` and `tools/s1_build_custom_split.py` from the real
parquet shards (revision `11cdd84bc0ba8d5c6641ea7b53550b1658527e1c`).
Machine-readable output: `reports/s1_split_statistics.json`,
`reports/s1_custom_split_statistics.json`.

```bash
python3 tools/s1_dataset_stats.py \
    --manifests data/manifests --report reports/s1_split_statistics.json
python3 tools/s1_build_custom_split.py \
    --plots data/manifests/fip1_plots.csv \
    --out data/manifests/custom_crossyear_splits.csv \
    --report reports/s1_custom_split_statistics.json
```

## 1. Official splits (verbatim from the HF shard layout)

The split is the parquet shard a row came from, not a column. There is no split
column in the data.

| Split | Plots | Share |
|---|---|---|
| `train` | 2,930 | 72.2 % |
| `validation` | 380 | 9.4 % |
| `test_plot` | 250 | 6.2 % |
| `test_genotype` | 246 | 6.1 % |
| `test_environment` | 190 | 4.7 % |
| `test_genotype_environment` | 62 | 1.5 % |
| **Total** | **4,058** | 100 % |

### Split × harvest year

| Split | 2016 | 2017 | 2018 | 2019 | 2021 | 2022 |
|---|---|---|---|---|---|---|
| train | 452 | 492 | 520 | 0 | 742 | 724 |
| validation | 119 | 126 | 102 | 0 | 11 | 22 |
| test_plot | 79 | 78 | 74 | 0 | 11 | 8 |
| test_genotype | 60 | 60 | 60 | 0 | 28 | 38 |
| test_environment | 0 | 0 | 0 | **190** | 0 | 0 |
| test_genotype_environment | 0 | 0 | 0 | **62** | 0 | 0 |

Confirmed consequences:

* `test_environment` and `test_genotype_environment` are **the whole of 2019**
  (190 + 62 = 252 = all 2019 plots). 2019 is a held-out *environment*;
* `train`, `validation`, `test_plot`, `test_genotype` contain no 2019 data;
* the official protocol therefore cannot be trained on 2019.

### Official split semantics (from the paper, verified against the data)

| Split | Definition | Verified |
|---|---|---|
| `test_plot` (P) | unseen plots, seen genotypes, seen environments | 164 genotypes shared with `train`; 0 shared `plot_uid` |
| `test_genotype` (G) | unseen genotypes, seen environments | **0** genotypes shared with `train`/`validation`/`test_plot` |
| `test_environment` (E) | 2019 only, seen genotypes | 43 genotypes shared with `train`; 0 shared plots |
| `test_genotype_environment` (G&E) | 2019 only, unseen genotypes | 30 genotypes shared with `test_genotype`; 0 shared with `train` |

## 2. Cross-split overlap (the leakage audit)

| Pair | Shared `plot_uid` | Shared `genotype_id` | Shared `yearsite_uid` |
|---|---|---|---|
| train ↔ validation | 0 | 162 | 5 |
| train ↔ test_plot | 0 | 164 | 5 |
| train ↔ test_genotype | 0 | **0** | 5 |
| train ↔ test_environment | 0 | 43 | 0 |
| train ↔ test_genotype_environment | 0 | **0** | 0 |
| validation ↔ test_plot | 0 | 95 | 5 |
| validation ↔ test_genotype | 0 | **0** | 5 |
| validation ↔ test_environment | 0 | 8 | 0 |
| test_plot ↔ test_genotype | 0 | **0** | 5 |
| test_plot ↔ test_environment | 0 | 9 | 0 |
| test_genotype ↔ test_genotype_environment | 0 | **30** | 0 |

Unique genotypes per split: train 849, validation 183, test_plot 164,
test_genotype 30, test_environment 48, test_genotype_environment 30.

Interpretation:

* **No `plot_uid` is ever shared** between splits — the image-level entity is
  cleanly partitioned;
* the genotype-level leakage pattern matches the intended design: G and G&E are
  the only genotype-unseen tests; `test_environment` reuses training genotypes
  (E is an environment shift, not a genotype shift);
* `test_genotype_environment` reuses the same 30 unseen genotypes as
  `test_genotype`, evaluated in 2019 — it is *not* an independent genotype set;
* `yearsite_uid` is shared across splits **inside the same year** (e.g. FPWW012
  appears in 2016 train/validation/test_plot), which is expected because
  `yearsite_uid` identifies a year-location, not a plot. It must not be used as
  an independent group key.

## 3. Project-defined cross-year split (separate manifest)

Frozen in `data/manifests/custom_crossyear_splits.csv`, rule
`custom-crossyear-v1`, seed 42. This manifest is additive; the official split
column is preserved in `official_split`.

Rule: for 2016, 2017, 2019, 2021, sort the year's genotypes, permute with
`numpy.random.RandomState(42)`, assign 80 % / 10 % / 10 % to
`train` / `val` / `id_test`; 2018 and 2022 are held out entirely as
`cross_year_test`.

| Custom split | Plots | Genotypes |
|---|---|---|
| train | 1,978 | 574 |
| val | 298 | 106 |
| id_test | 234 | 99 |
| cross_year_test | 1,548 | 645 |

Per year:

| Year | train | val | id_test | cross_year_test |
|---|---|---|---|---|
| 2016 | 536 | 108 | 66 | — |
| 2017 | 616 | 70 | 70 | — |
| 2018 | — | — | — | 756 |
| 2019 | 202 | 24 | 26 | — |
| 2021 | 624 | 96 | 72 | — |
| 2022 | — | — | — | 792 |

**Known property (not a bug):** genotypes are split *within* each year, so the
same genotype can appear in `train` in 2016 and `id_test` in 2017. Cross-year
leakage at the genotype level is therefore present by construction. The
`cross_year_test` years (2018, 2022) are held out as *years*, not as genotypes,
so 370 of their genotypes also occur in `train`. If a genotype-unseen
cross-year test is required later, use the official `test_genotype` split or
redefine the custom rule; do not silently reinterpret this manifest.

## 4. Valid sample counts by span and split

Consecutive same-plot pairs: **148,964** (Δt 0–93 d, median 4 d, p90 12 d).
Δt histogram: 0–3 d 31,724; 4–7 d 77,270; 8–14 d 29,287; 15–21 d 5,536;
22–28 d 1,267; 29–35 d 1,497; 36–45 d 697; 46–60 d 802; 60+ d 884.

Pairs with a label valid at **both** ends within |lag| ≤ 7 d:

| Trait | All splits | train | validation | test_plot | test_genotype | test_environment | test_G&E |
|---|---|---|---|---|---|---|---|
| CC | 141,751 | 111,634 | 10,342 | 6,771 | 7,804 | 3,931 | 1,269 |
| PH | 111,868 | 88,207 | 7,147 | 4,624 | 5,798 | 4,496 | 1,596 |
| WHC | 67,060 | 56,264 | 4,095 | 2,689 | 3,508 | 361 | 143 |
| Sen | 9,145 | 5,948 | 1,492 | 935 | 770 | 0 | 0 |

Target-span candidates (nearest later image within ±2 d of the target span):

| Span | Pairs | CC both valid | PH both valid |
|---|---|---|---|
| 7 d | 117,403 | 112,882 | 95,647 |
| 14 d | 110,892 | 105,373 | 88,778 |
| 21 d | 102,183 | 97,238 | 81,377 |
| 28 d | 97,713 | 91,637 | 77,592 |
| 35 d | 88,869 | 83,563 | 71,538 |
| 42 d | 84,115 | 80,139 | 66,126 |

`test_environment` / `test_G&E` have very small WHC/Sen support (Sen is
2016–2018 only, and those splits are 2019-only). Any claim about WHC or Sen on
the environment shift will be underpowered — report the effective n.

## 5. What is frozen

`data/manifests/manifest_spec.json` records the manifest id (`fip1-s1-v1`), the
dataset revision, the official/custom split rules, the sample definition, the
weather-window convention `(t_a, t_b]`, the label-matching rule, and the SHA256
of every manifest artifact. Changing any rule requires a new manifest id.
