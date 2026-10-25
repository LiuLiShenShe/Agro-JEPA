#!/usr/bin/env python3
"""S4A: build ``reports/s4_pilot_results.md`` from the JSON artifacts.

Every number in the markdown is read from a produced artifact; nothing is
hard-coded.  If an artifact is missing the corresponding table is marked as
unavailable instead of being invented.

Usage
-----
    python3 tools/s4_write_reports.py --config agro_jepa/config/s4_pilot.yaml
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load(path: Path):
    if not path.exists():
        return None
    with open(path) as fh:
        return json.load(fh)


def pct(x, nd=2):
    return "—" if x is None else f"{100.0 * x:.{nd}f}%"


def num(x, nd=5):
    return "—" if x is None else f"{x:.{nd}f}"


def ci(d, key="relative_increase_mean"):
    if not d:
        return "—"
    return f"{d[key]*100:+.2f}% [{d['ci95_low']*100:+.2f}, {d['ci95_high']*100:+.2f}]"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="agro_jepa/config/s4_pilot.yaml")
    ap.add_argument("--out", default="reports/s4_pilot_results.md")
    args = ap.parse_args()
    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)

    e1 = load(ROOT / cfg["artifacts"]["results_json"])
    boot = load(ROOT / cfg["artifacts"]["e1_bootstrap"])
    e2 = load(ROOT / cfg["artifacts"]["e2_metrics"])
    diag = load(ROOT / "reports/s4_conditioning_diagnostics.json")
    shared = load(ROOT / "reports/s4_e2_shared_epoch.json")
    gen = load(ROOT / "reports/s4_generalization.json")
    verify = load(ROOT / "reports/s4_verify.json")
    cond_summary = load(ROOT / "reports/s4_conditioning_summary.json")

    L: list[str] = []
    A = L.append
    A("# S4A Pilot 结果：时间、积温与天气驱动的未来潜空间预测基线")
    A("")
    A("> 本文件由 `tools/s4_write_reports.py` 从真实运行产物生成。"
      "所有数字均来自 `reports/*.json`，未手工填写。")
    A("")

    # ---------------- 1. run inventory ------------------------------------
    A("## 1. 运行清单")
    A("")
    A("| 条件 | 协议 | 可训练参数 | 条件参数 | 预测器输入维 | seeds | 候选 epoch | 实际 epoch (seed0) |")
    A("|---|---|---|---|---|---|---|---|")
    run_rows = []
    for name, c in cfg["conditions"].items():
        if c.get("mode") != "trained":
            continue
        ck = None
        if e1 and name in e1.get("E1", {}).get("checkpoints", {}):
            ck = e1["E1"]["checkpoints"][name]
        summary = load(ROOT / cfg["training"]["output_root"] / name / "seed0" / "run_summary.json")
        run_rows.append((name, c, ck, summary))
        A(f"| `{name}` | {c.get('protocol')} | "
          f"{ck['n_trainable_parameters'] if ck else '—'} | "
          f"{ck['n_condition_parameters'] if ck else '—'} | "
          f"{ck['predictor_input_dim'] if ck else '—'} | "
          f"{cfg['training']['seeds']} | "
          f"{summary['epochs_requested'] if summary else '—'} | "
          f"{summary['epochs_completed'] if summary else '—'} |")
    A("")

    # ---------------- 2. E1 ------------------------------------------------
    A("## 2. E1：官方 validation 未来潜空间预测")
    A("")
    if not e1 or "E1" not in e1:
        A("_E1 产物尚不存在。_")
    else:
        block = e1["E1"]
        A(f"评价集：{block['n_val_pairs']} 对 / {block['n_val_plots']} 个 plot / "
          f"{block['n_val_images']} 张图像；年份 {block['val_years']}。")
        A("")
        A("相对降幅的 95% CI 来自 2,000 次 plot 级 cluster bootstrap。")
        A("")
        A("| 条件 | MSE | vs Persistence | vs train-mean | cosine | n |")
        A("|---|---|---|---|---|---|")
        for name in ["b0_persistence", "b1_train_mean"] + [
            n for n in cfg["conditions"] if cfg["conditions"][n].get("mode") == "trained"
        ]:
            c = block["conditions"].get(name)
            if not c:
                continue
            b = (boot or {}).get("per_condition", {}).get(name, {})
            ci_p = b.get("rel_improvement_vs_persist", {})
            ci_m = b.get("rel_improvement_vs_train_mean", {})
            ci_ps = (f"{pct(c.get('relative_improvement_vs_persist'))} "
                     f"[{pct(ci_p.get('ci95_low'))}, {pct(ci_p.get('ci95_high'))}]"
                     if ci_p else pct(c.get("relative_improvement_vs_persist")))
            ci_ms = (f"{pct(c.get('relative_improvement_vs_train_mean'))} "
                     f"[{pct(ci_m.get('ci95_low'))}, {pct(ci_m.get('ci95_high'))}]"
                     if ci_m else pct(c.get("relative_improvement_vs_train_mean")))
            A(f"| `{name}` | {num(c.get('mse_pred'))} | {ci_ps} | {ci_ms} | "
              f"{num(c.get('cosine_pred'), 4)} | {c.get('n_samples')} |")
        A("")

        A("### 2.1 关键对照（MSE 差值，95% CI）")
        A("")
        A("| 对照 | ΔMSE | 95% CI |")
        A("|---|---|---|")
        for key, d in (boot or {}).get("contrasts", {}).items():
            x = d["mse_delta"]
            A(f"| `{key}` | {x['mean']:+.5f} | "
              f"[{x['ci95_low']:+.5f}, {x['ci95_high']:+.5f}] |")
        A("")

        A("### 2.2 按 pair_type")
        A("")
        A("| 条件 | span7 MSE | span14 MSE | span7 cos | span14 cos |")
        A("|---|---|---|---|---|")
        for name in cfg["conditions"]:
            c = block["conditions"].get(name)
            if not c or "by_span" not in c:
                continue
            s7 = c["by_span"].get("span7", {})
            s14 = c["by_span"].get("span14", {})
            A(f"| `{name}` | {num(s7.get('mse_pred'))} | {num(s14.get('mse_pred'))} | "
              f"{num(s7.get('cosine_pred'), 4)} | {num(s14.get('cosine_pred'), 4)} |")
        A("")

        A("### 2.3 按 harvest_year（相对 persistence 的 MSE 降幅）")
        A("")
        A("| 条件 | " + " | ".join(sorted(
            str(y) for y in next(iter(
                [c for c in block["conditions"].values() if "by_year" in c]
            )).get("by_year", {}))) + " |")
        A("|---|" + "---|" * len(next(iter(
            [c for c in block["conditions"].values() if "by_year" in c]
        )).get("by_year", {})))
        for name in cfg["conditions"]:
            c = block["conditions"].get(name)
            if not c or "by_year" not in c:
                continue
            cells = [pct(v.get("relative_improvement_vs_persist"))
                     for _, v in sorted(c["by_year"].items())]
            A(f"| `{name}` | " + " | ".join(cells) + " |")
        A("")

        # per-year contrasts
        if boot and boot.get("by_year"):
            A("### 2.4 年内对照（年份身份不变的天气/GDD 效应）")
            A("")
            A("| 年份 | 对照 | ΔMSE | 95% CI |")
            A("|---|---|---|---|")
            for year, yb in sorted(boot["by_year"].items()):
                for key, x in yb.get("contrasts", {}).items():
                    A(f"| {year} | `{key}` | {x['mean']:+.5f} | "
                      f"[{x['ci95_low']:+.5f}, {x['ci95_high']:+.5f}] |")
            A("")

        A("### 2.5 基因型泛化切分（官方 validation）")
        A("")
        if not gen:
            A("_泛化切分产物尚不存在。_")
        else:
            A(f"官方 split 只留出 plot、不留出年份：每个 validation 年份在训练集中都出现过。"
              f"但 {gen['n_val_genotypes']} 个 validation 基因型中有 "
              f"{gen['n_val_genotypes_unseen']} 个从未出现在任何训练 plot 上"
              f"（{gen['n_pairs_unseen_genotype']} 对样本）。按该轴切分已有 E1 预测"
              f"（`tools/s4_generalization_split.py`，不新增前向）。")
            A("")
            A("| 组 | 条件 | MSE | vs persistence [95% CI] |")
            A("|---|---|---|---|")
            for grp, blk in gen["groups"].items():
                for name, s in blk["conditions"].items():
                    ci = gen["contrasts"][grp]["improvement_vs_persist"][name]
                    A(f"| {grp} | `{name}` | {num(s['mse_pred'])} | "
                      f"{pct(s['relative_improvement_vs_persist'])} "
                      f"[{pct(ci['ci95_low'])}, {pct(ci['ci95_high'])}] |")
            A("")
            A("| 组 | 对照 | ΔMSE | 95% CI |")
            A("|---|---|---|---|")
            for grp, blk in gen["contrasts"].items():
                for key, x in blk["contrasts"].items():
                    A(f"| {grp} | `{key}` | {x['mean']:+.5f} | "
                      f"[{x['ci95_low']:+.5f}, {x['ci95_high']:+.5f}] |")
            A("")

    # ---------------- 3. diagnostics --------------------------------------
    A("## 3. 条件输入置换诊断（冻结模型，仅推理）")
    A("")
    if not diag:
        A("_诊断产物尚不存在。_")
    else:
        A("`within_year`：只在与同一 harvest_year 的样本间置换输入；年份身份保留，"
          "只破坏日期对齐。`global`：跨年份置换。")
        A("")
        A("| 条件 | 输入 | real MSE | within-year 增幅 (95% CI) | global 增幅 (95% CI) |")
        A("|---|---|---|---|---|")
        for name, b in diag["conditions"].items():
            wy = b.get("within_year", {})
            gl = b.get("global", {})
            wyb = b.get("within_year_bootstrap", {})
            glb = b.get("global_bootstrap", {})
            def cell(d, bootc):
                if not d:
                    return "—"
                base = f"{d.get('relative_increase_mean', float('nan'))*100:+.2f}%"
                if bootc:
                    base += (f" [{bootc['ci95_low']*100:+.2f}, "
                             f"{bootc['ci95_high']*100:+.2f}]")
                return base
            A(f"| `{name}` | {b['mode']} | {b['real']['mse']:.5f} | "
              f"{cell(wy, wyb)} | {cell(gl, glb)} |")
        A("")
        A("within-year 增幅的 CI 来自单次 within-year 置换的 plot 级 bootstrap；"
          "两种置换的均值与标准差见 `reports/s4_conditioning_diagnostics.json`。")
        A("")

    # ---------------- 4. E2 ----------------------------------------------
    A("## 4. E2：未来 CC / PH 表型读出")
    A("")
    if not e2:
        A("_E2 产物尚不存在。_")
    else:
        A("协议：S3.2 修正协议（inner-dev 选模 → 全 train 重拟合 → validation 单次评价，"
          "2,000 次 plot 级 bootstrap）。A = 当前特征；B = 预测未来特征；"
          "C = GDD / 天气基线的预测未来特征（`C_environment_predicted`）。")
        A("")
        a_vals = []
        for trait in ("cc", "ph"):
            blk = e2.get("rows", {}).get("primary_lag7", {}).get(trait, {})
            m = blk.get("overall", {}).get("conditions", {}).get("A_current")
            if m:
                a_vals.append(f"{trait.upper()} {m['r2']['mean']:.5f}")
        if a_vals:
            A("`A_current` 总体 R²：" + " / ".join(a_vals) +
              "（由本次运行的 pooled 特征重算，可与 S3.2 的 A 条件对照）。")
        A("")
        for variant in e2.get("rows", {}):
            A(f"### {variant}")
            A("")
            for trait in ("cc", "ph"):
                tb = e2["rows"][variant].get(trait, {})
                for scope, block in tb.items():
                    if block.get("skipped"):
                        A(f"- `{trait}/{scope}` 跳过（样本不足：{block}）")
                        continue
                    A(f"**{trait.upper()} / {scope}** — 有效样本 {block['counts']['n_val']}，"
                      f"独立 plot {block['counts']['n_val_plots']}")
                    A("")
                    A("| 条件 | 角色 | MAE | RMSE | R² | 选中 epoch |")
                    A("|---|---|---|---|---|---|")
                    for label, m in block["conditions"].items():
                        A(f"| `{label}` | {m.get('role', '—')} | "
                          f"{m['mae']['mean']:.5f} ± {m['mae']['std']:.5f} | "
                          f"{m['rmse']['mean']:.5f} ± {m['rmse']['std']:.5f} | "
                          f"{m['r2']['mean']:.5f} ± {m['r2']['std']:.5f} | "
                          f"{m.get('selected_epoch_median', '—')} |")
                    A("")
                    if "B_minus_A" in block:
                        A("| B − A | ΔMAE | ΔRMSE | ΔR² |")
                        A("|---|---|---|---|")
                        for label, m in block["B_minus_A"].items():
                            A(f"| `{label}` | {m['mae']['mean_delta']:+.5f} | "
                              f"{m['rmse']['mean_delta']:+.5f} | "
                              f"{m['r2']['mean_delta']:+.5f} |")
                        A("")
        if e2.get("bootstrap"):
            A("### 4.1 B − A 的 plot 级 bootstrap（primary_lag7 / overall）")
            A("")
            A("| trait | 条件 | ΔMAE [95% CI] | ΔRMSE [95% CI] | ΔR² [95% CI] |")
            A("|---|---|---|---|---|")
            for trait, blk in e2["bootstrap"]["scopes"].items():
                for label, d in blk.items():
                    if label in ("n_samples", "n_plots"):
                        continue
                    def f(x):
                        return f"{x['mean']:+.5f} [{x['ci95_low']:+.5f}, {x['ci95_high']:+.5f}]"
                    A(f"| {trait} | `{label}` | {f(d['delta_mae'])} | "
                      f"{f(d['delta_rmse'])} | {f(d['delta_r2'])} |")
            A("")

    # ---------------- 4b. pairwise contrasts ------------------------------
    A("### 4.2 条件间对照（E2 预测特征，plot 级 bootstrap）")
    A("")
    e2c = load(ROOT / "reports/s4_e2_contrasts.json")
    if not e2c:
        A("_E2 条件间对照产物尚不存在。_")
    else:
        A("与 4.1 的 B−A 不同，这里比较的是两个**预测特征**条件；"
          "每个条件的 seed 预测先取平均，再计算 R² 并对 plot 做 bootstrap。"
          "因此绝对 R² 与 4.1 的“逐 seed 再取均值”略有差异。")
        A("")
        for trait, blk in e2c["traits"].items():
            A(f"**{trait.upper()}** — n={blk['n_pairs']}，独立 plot {blk['n_plots']}")
            A("")
            A("| 对照 | ΔR² [95% CI] | ΔMAE [95% CI] |")
            A("|---|---|---|")
            for key, m in blk["contrasts"].items():
                r2 = m["delta_r2"]
                mae = m["delta_mae"]
                A(f"| `{key}` | {r2['mean']:+.5f} "
                  f"[{r2['ci95_low']:+.5f}, {r2['ci95_high']:+.5f}] | "
                  f"{mae['mean']:+.5f} "
                  f"[{mae['ci95_low']:+.5f}, {mae['ci95_high']:+.5f}] |")
            A("")

    # ---------------- 4c. shared-epoch control ---------------------------
    A("### 4.3 共享 epoch 对照（S3.2 第二种读出 regime）")
    A("")
    if not shared:
        A("_共享 epoch 对照产物尚不存在。_")
    else:
        A("每个条件都使用同一个 epoch（所有 A/B/C 条件与 probe seed 的 inner-dev 选中 epoch 中位数），"
          "以排除“B 只是训练更久”的解释。")
        A("")
        for trait, blk in shared["traits"].items():
            A(f"**{trait.upper()} / overall** — shared epoch {blk['shared_epoch']}")
            A("")
            A("| 条件 | R² | ΔR² vs A [95% CI] |")
            A("|---|---|---|")
            for label, m in blk["conditions"].items():
                b = blk["bootstrap_B_minus_A"].get(label)
                d = (f"{b['delta_r2']['mean']:+.5f} "
                     f"[{b['delta_r2']['ci95_low']:+.5f}, {b['delta_r2']['ci95_high']:+.5f}]"
                     if b else "—")
                A(f"| `{label}` | {m['r2']['mean']:.5f} | {d} |")
            A("")

    # ---------------- 5. verify ------------------------------------------
    A("## 5. 验收检查")
    A("")
    if not verify:
        A("_验收产物尚不存在。_")
    else:
        A(f"`{verify['n_pass']}/{verify['n_checks']}` 通过。")
        A("")
        A("| 检查 | 结果 | 说明 |")
        A("|---|---|---|")
        for c in verify["checks"]:
            A(f"| {c['id']} | {'PASS' if c['pass'] else 'FAIL'} | {c['detail']} |")
        A("")

    # ---------------- 6. protocol card ------------------------------------
    A("## 6. 天气 / GDD 定义（摘要）")
    A("")
    if cond_summary:
        A(f"- 天气通道：{cond_summary['weather_vector']['channels']}")
        A(f"- 单位：{cond_summary['weather_vector']['units']}")
        A(f"- 排除：{cond_summary['weather_vector']['excluded']}")
        A(f"- Protocol P：{cond_summary['protocol_P']['definition']}")
        A(f"- Protocol R：{cond_summary['protocol_R']['definition']}")
        A(f"- GDD：{cond_summary['gdd']['definition']}")
        A(f"- 结构性限制：{cond_summary['structural_limit']}")
        A("")

    A("## 7. 产物路径")
    A("")
    for k, v in cfg["artifacts"].items():
        A(f"- `{k}`: `{v}`")
    A("")
    A("完整协议定义见 `reports/s4_data_protocol.md`。")

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
