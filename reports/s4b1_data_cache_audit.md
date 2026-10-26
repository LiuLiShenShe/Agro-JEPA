# S4B.1 Data & Feature Cache Audit

- Generated: 2026-10-26T21:04:53+0800
- Config: `agro_jepa/config/s4b1_data_cache.yaml` (sha256 `4a113c0d517f8fc2…`)
- Frozen commit: `4e01ec0f054238e2951912923ff52f40d46eb45a` (agro-s4a1-matched-control)
- DINOv2 code commit: `7764ea0f912e53c92e82eb78a2a1631e92725fc8`
- Weights sha256: `0b8b82f85de91b42…`
- Cache protocol: `mainline224_s1-v1`

## Acceptance

1. Full official-train covers **2,930 plots**: **True** (2930 plots in index)
2. Pairing rule consistent with S3 (source `fip1-s2-v1`, dedupe physical pairs): **True**
3. Unique official-train pair endpoints required: **114,791**
4. Reused vs new: S3 features reused = **36,631**, S4B cache recovered = **60,186**, S4B newly computed = **38,933**; images reused from disk = **36,663**, newly downloaded = **99,087**
5. Train pair endpoint feature coverage (100%): **True**
6. All four test splits' pair endpoints ready: **True**
7. Provenance + SHA256 complete (cache integrity ok): **True**
8. Any test phenotype label read? **False**
9. Frozen S1/S2/S3/S4A/S4A.1 assets unchanged: **True** (0 modified)
10. Ready for full-training + one-shot test eval: **True**

## Full train index

- pairs: 179,823, span7 = 92,136, span14 = 87,687
- unique endpoints: 114,791
- index sha256: `643865330cb4840432a0e88f0618a6bafed555bcd1b947460c9451cf30f92726`

## Coverage by split

| split | pairs | endpoints | fully covered |
|---|---|---|---|
| train (full) | 179,823 | 114,791 | True |
| test_plot | 9,851 | 7,050 | True |
| test_genotype | 12,059 | 8,149 | True |
| test_environment | 6,300 | 4,281 | True |
| test_genotype_environment | 2,147 | 1,479 | True |
