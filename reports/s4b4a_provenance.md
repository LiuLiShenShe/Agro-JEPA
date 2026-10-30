# S4B.4A Provenance

- Branch: `agro-s4b4a-year-heldout`
- HEAD: `e626bc65c80eaf44254fcda58caeaf1b3d99fb86`
- Baseline (S4B.3): `27504922095366ac246c8760a08702b3b762d620`
- Config: `agro_jepa/config/s4b4a_year_heldout.yaml`
- Protocol: `reports/s4b4a_protocol.json`
  - data: 179,823 pairs / 2,930 plots / years [2016, 2017, 2018, 2021, 2022]
- Audit inputs: full_train_pairs sha `a49b546f57d68881`, conditioning sha `daf732cd723a1321`
- GDD/weather conditioning (full train): `external_data/s4b4a_conditioning/full_train_conditioning.npz`
- Folds: `external_data/s4b4a_folds/`
- Runs: `external_data/s4b4a_runs/`

## Reproduce

```bash
python3 tools/s4b4a_data_audit.py --config agro_jepa/config/s4b4a_year_heldout.yaml
python3 tools/s4b4a_protocol.py  --config agro_jepa/config/s4b4a_year_heldout.yaml
for f in 2016 2017 2018 2021 2022; do
  for c in b2_no_time b2_no_time_constant_delta b3_delta gdd_p weather_p; do
    CUDA_VISIBLE_DEVICES=1 python3 tools/s4b4a_train.py \
      --config agro_jepa/config/s4b4a_year_heldout.yaml \
      --fold $f --condition $c --seed 0
  done
done
CUDA_VISIBLE_DEVICES=1 python3 tools/s4b4a_eval_e1.py --config agro_jepa/config/s4b4a_year_heldout.yaml
CUDA_VISIBLE_DEVICES=1 python3 tools/s4b4a_eval_e2.py --config agro_jepa/config/s4b4a_year_heldout.yaml
python3 tools/s4b4a_write_reports.py --config agro_jepa/config/s4b4a_year_heldout.yaml
python3 tools/s4b4a_verify.py --config agro_jepa/config/s4b4a_year_heldout.yaml
```

