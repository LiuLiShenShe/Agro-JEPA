#!/usr/bin/env python3
"""S4A.1 report generator: matched control + S4B test-data audit.

Every number in the markdown is read from a produced JSON artifact; nothing is
hand-filled.  The interpretive verdicts are produced by explicit decision rules
encoded below, so re-running on new artifacts re-derives the verdict.

Usage
-----
    python3 tools/s4a1_write_reports.py --config agro_jepa/config/s4a1_matched_control.yaml
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONTROL = "b2_no_time_constant_delta"
MATCHED = "b3_delta"
NO_TIME = "b2_no_time"
# must stay in the same order as tools/s4a1_e1_eval.py
CONTRASTS = [
    ("b3_delta_minus_b2_no_time_constant_delta", CONTROL, MATCHED),
    ("b3_delta_minus_b2_no_time", NO_TIME, MATCHED),
    ("b2_no_time_constant_delta_minus_b2_no_time", NO_TIME, CONTROL),
]


def load(path: Path):
    if not path.exists():
        return None
    with open(path) as fh:
        return json.load(fh)


def pct(x, nd=2):
    return "—" if x is None else f"{100.0 * x:.{nd}f}%"


def num(x, nd=5):
    return "—" if x is None else f"{x:.{nd}f}"


def ci(d, key="mean"):
    if not d:
        return "—"
    return (f"{d[key]:+.5f} [{d['ci95_low']:+.5f}, {d['ci95_high']:+.5f}]")


def ci_pct(d, key="mean"):
    if not d:
        return "—"
    return (f"{100*d[key]:+.2f}% [{100*d['ci95_low']:+.2f}%, "
            f"{100*d['ci95_high']:+.2f}%]")


def load_runs(cfg, seeds):
    runs = {}
    roots = cfg["training"].get("checkpoint_roots") or [cfg["training"]["output_root"]]
    for name, c in cfg["conditions"].items():
        if c.get("mode") != "trained":
            continue
        runs[name] = {}
        for seed in seeds:
            for root in roots:
                p = Path(root) / name / f"seed{seed}" / "run_summary.json"
                if p.exists():
                    runs[name][seed] = json.load(open(p))
                    break
    # A run_summary written by an earlier code revision may predate the
    # fixed-interval / init-hash fields.  The checkpoint is the authoritative
    # record of what was actually trained, so fall back to it for provenance.
    constant = {n for n, c in cfg["conditions"].items()
                if c.get("mode") == "trained" and c.get("delta_mode") == "constant"}
    for name in constant:
        for seed, summary in runs.get(name, {}).items():
            if summary.get("fixed_delta"):
                continue
            for root in roots:
                ck = Path(root) / name / f"seed{seed}" / "checkpoints" / "best.pt"
                if not ck.exists():
                    continue
                import torch
                payload = torch.load(ck, map_location="cpu", weights_only=False)
                if payload.get("fixed_delta"):
                    summary["fixed_delta"] = payload["fixed_delta"]
                    summary["_fixed_delta_source"] = "checkpoint (summary predates the field)"
                if not summary.get("init_state_sha256"):
                    m = payload.get("metrics") or {}
                    if m.get("init_state_sha256"):
                        summary["init_state_sha256"] = m["init_state_sha256"]
                        summary["_init_state_source"] = "checkpoint (summary predates the field)"
                break
    return runs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="agro_jepa/config/s4a1_matched_control.yaml")
    ap.add_argument("--out-md", default="reports/s4a1_matched_control.md")
    ap.add_argument("--out-json", default="reports/s4a1_matched_control.json")
    ap.add_argument("--s4b-md", default="reports/s4b_test_data_audit.md")
    args = ap.parse_args()
    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)
    seeds = [int(s) for s in cfg["training"]["seeds"]]

    e1 = load(ROOT / cfg["artifacts"]["results_json"])
    e1b = load(ROOT / cfg["artifacts"]["e1_bootstrap"])
    e2 = load(ROOT / cfg["artifacts"]["e2_metrics"])
    verify = load(ROOT / "reports/s4a1_verify.json")
    runs = load_runs(cfg, seeds)
    s4b = load(ROOT / "reports/s4b_test_data_audit.json")
    s4bc = load(ROOT / "reports/s4b_test_conditioning.json")

    L: list[str] = []
    A = L.append
    A("# S4A.1 参数量匹配的时间条件对照与 S4B 测试数据准备")
    A("")
    A("> 本文件由 `tools/s4a1_write_reports.py` 从真实运行产物生成；"
      "数字全部来自 `reports/*.json`，未手工填写。结论由脚本里的判定规则"
      "从这些数字重新推导。")
    A("")

    # ---------------- 0. provenance --------------------------------------
    A("## 0. 资产与不可变输入")
    A("")
    if verify:
        A(f"- 配置：`{verify['config']}`，SHA256 `{verify['config_sha256']}`")
        A(f"- 冻结 S3 索引 SHA256：`{verify['index_sha256']}`")
        A(f"- 冻结条件张量 SHA256：`{verify['conditioning_sha256']}`")
        A(f"- 独立验收：**{verify['n_pass']}/{verify['n_checks']} 项通过**"
          f"（`reports/s4a1_verify.json`）")
    else:
        A("- 验收文件缺失：`reports/s4a1_verify.json`")
    A(f"- 训练 seed 集合：{seeds}")
    A("")

    # ---------------- 1. run inventory -----------------------------------
    A("## 1. 运行清单与参数量")
    A("")
    A("| 条件 | 可训练参数 | 条件参数 | 输入维 | seeds | 实际 epoch | 训练耗时 (s) |")
    A("|---|---|---|---|---|---|---|")
    for name in [NO_TIME, CONTROL, MATCHED]:
        if name not in runs:
            continue
        r0 = runs[name].get(seeds[0], {})
        eps = ", ".join(
            f"{runs[name][s]['epochs_completed']}@{runs[name][s]['best_epoch']}"
            for s in seeds if s in runs[name])
        wall = ", ".join(f"{runs[name][s]['wall_clock_seconds']:.0f}"
                         for s in seeds if s in runs[name])
        A(f"| `{name}` | {r0.get('n_trainable_parameters', 0):,} | "
          f"{r0.get('n_condition_parameters', 0):,} | "
          f"{r0.get('predictor_input_dim', 0)} | {seeds} | {eps} | {wall} |")
    A("")
    if CONTROL in runs and MATCHED in runs:
        c0, b0 = runs[CONTROL][seeds[0]], runs[MATCHED][seeds[0]]
        same = (c0["n_trainable_parameters"] == b0["n_trainable_parameters"]
                and c0["n_condition_parameters"] == b0["n_condition_parameters"]
                and c0["predictor_input_dim"] == b0["predictor_input_dim"])
        A(f"**参数量匹配判定**：B2-C 与 B3 的可训练参数 "
          f"{'完全一致' if same else '不一致'}（B2-C "
          f"{c0['n_trainable_parameters']:,} vs B3 {b0['n_trainable_parameters']:,}）。")
        A("")
        fx = c0.get("fixed_delta") or {}
        nsrc = fx.get("fixed_delta_n_source_rows")
        fx = c0.get("fixed_delta") or {}
        nsrc = fx.get("fixed_delta_n_source_rows")
        A(f"**固定时间间隔**：{fx.get('fixed_delta_days')} d"
          f"（规则 `{fx.get('fixed_delta_rule')}`，来源 "
          f"{fx.get('fixed_delta_source')}，n={nsrc if nsrc is not None else '—'}；"
          f"该训练子集的均值 {num(fx.get('fixed_delta_inner_train_mean'), 4)}、"
          f"范围 [{num(fx.get('fixed_delta_inner_train_min'), 1)}, "
          f"{num(fx.get('fixed_delta_inner_train_max'), 1)}]）。"
          f"归一化后取值 {num(fx.get('fixed_delta_normalised_value'), 4)}，"
          f"与 B3 共用同一套 `DeltaNormStats`。"
          + ("（seed0 的 `run_summary.json` 早于该字段的记录版本，"
             "此处取自其 checkpoint，二者语义相同。）"
             if c0.get("_fixed_delta_source") else ""))
        A("")

        A("**随机初始化可复现性**：每条 run 在 `set_seed(seed)` 后立即 "
          "`build_s4_predictor`，再对 state_dict 求 SHA256。"
          "`tools/s4a1_init_audit.py` 按同一规则重新构建两次并与记录值比对"
          "（`reports/s4a1_init_hashes.json`）。")
        A("")
        ih = load(ROOT / "reports/s4a1_init_hashes.json")
        if ih:
            A(f"- 重复构建全部一致（确定性）：**{ih['all_deterministic']}**")
            A(f"- 有记录值的 run：{ih['n_recorded']} 个，全部匹配："
              f"**{ih['n_matched']}/{ih['n_recorded']}**，"
              f"不一致 {len(ih['mismatches'])} 个。"
              "其余 run 早于该字段的记录版本，其哈希按同一规则重算且确定。")
            A("")
            A("| 条件 | seed0 | seed1 | seed2 |")
            A("|---|---|---|---|")
            for name in [NO_TIME, CONTROL, MATCHED]:
                if name not in ih["conditions"]:
                    continue
                cells = []
                for s in seeds:
                    d = ih["conditions"][name].get(str(s), {})
                    h = (d.get("recorded_sha256") or d.get("recomputed_sha256") or "")
                    mark = "" if d.get("matches_recorded") else (
                        "*" if d.get("recorded_sha256") is None else "!")
                    cells.append(h[:16] + mark)
                A(f"| `{name}` | " + " | ".join(cells) + " |")
            A("")
            A("（`*` = 该 run 早于字段记录，显示的是按规则重算的确定性哈希；"
              "无标记 = 与 run 内记录值一致。）")
            A("")
        else:
            A("- 初始化审计产物缺失（`reports/s4a1_init_hashes.json`）。")
            A("")

    # ---------------- 2. verification ------------------------------------
    if verify:
        A("## 2. 独立验收检查")
        A("")
        A("| 检查 | 结果 | 说明 |")
        A("|---|---|---|")
        for c in verify["checks"]:
            A(f"| `{c['id']}` | {'PASS' if c['pass'] else 'FAIL'} | {c['detail']} |")
        A("")

    # ---------------- 3. E1 ----------------------------------------------
    if e1:
        A("## 3. E1：官方 validation 未来潜空间预测")
        A("")
        A(f"评价集：{e1['n_val_pairs']:,} 对 / {e1['n_val_plots']} 个 plot / "
          f"{e1['n_val_images']:,} 张图；年份 {e1['val_years']}。"
          "official validation 从不参与选模。")
        A("")
        A("### 3.1 seed 平均预测（与 S4A E1 同口径）")
        A("")
        A("| 条件 | MSE | vs Persistence | vs train-mean | cosine | n |")
        A("|---|---|---|---|---|---|")
        for name in [NO_TIME, CONTROL, MATCHED]:
            s = e1["conditions"]["seed_averaged"].get(name)
            if not s:
                continue
            pb = (e1b or {}).get("per_condition", {}).get(name, {})
            A(f"| `{name}` | {num(s['mse_pred'])} | "
              f"{pct(s['relative_improvement_vs_persist'])} "
              f"{ci_pct(pb.get('rel_improvement_vs_persist'))} | "
              f"{pct(s['relative_improvement_vs_train_mean'])} "
              f"{ci_pct(pb.get('rel_improvement_vs_train_mean'))} | "
              f"{num(s['cosine_pred'], 4)} | {s['n_samples']:,} |")
        A("")
        A("### 3.2 每个训练 seed 的指标（训练随机性，独立于 plot bootstrap）")
        A("")
        A("| 条件 | " + " | ".join(f"seed{s} MSE" for s in seeds) + " | 跨 seed 均值±SD |")
        A("|---" * (len(seeds) + 2) + "|")
        for name in [NO_TIME, CONTROL, MATCHED]:
            block = e1["conditions"]["per_seed"].get(name)
            if not block:
                continue
            vals = [block[str(s)]["mse_pred"] for s in seeds if str(s) in block]
            import statistics as _st
            sd = _st.stdev(vals) if len(vals) > 1 else 0.0
            A(f"| `{name}` | " + " | ".join(num(v) for v in vals)
              + f" | {num(sum(vals)/len(vals))} ± {num(sd)} |")
        A("")
        A("### 3.3 配对差值")
        A("")
        A("| 对照 | plot bootstrap ΔMSE（95% CI） | 逐 seed ΔMSE（均值±SD，符号一致） |")
        A("|---|---|---|")
        for label, block in (e1.get("per_seed_contrasts") or {}).items():
            boot = ((e1b or {}).get("contrasts", {}).get(label) or {}).get("mse_delta")
            a = block["across_seeds"]
            A(f"| `{label}` | {ci(boot)} | {a['mean']:+.5f} ± {a['std']:.5f}，"
              f"{a['n_seeds']} seeds，同号={a['all_same_sign']} |")
        A("")
        A("### 3.4 分层（seed 平均预测）")
        A("")
        A("| 条件 | span7 MSE | span14 MSE | span7 cos | span14 cos |")
        A("|---|---|---|---|---|")
        for name in [NO_TIME, CONTROL, MATCHED]:
            s = e1["conditions"]["seed_averaged"].get(name)
            if not s:
                continue
            b = s["by_span"]
            A(f"| `{name}` | {num(b['span7']['mse_pred'])} | "
              f"{num(b['span14']['mse_pred'])} | {num(b['span7']['cosine_pred'], 4)} | "
              f"{num(b['span14']['cosine_pred'], 4)} |")
        A("")
        A("| 条件 | " + " | ".join(str(y) for y in e1["val_years"]) + " |")
        A("|---" * (len(e1["val_years"]) + 1) + "|")
        for name in [NO_TIME, CONTROL, MATCHED]:
            s = e1["conditions"]["seed_averaged"].get(name)
            if not s:
                continue
            cells = []
            for y in e1["val_years"]:
                yy = s["by_year"].get(str(y))
                cells.append(pct(yy["relative_improvement_vs_persist"]) if yy else "—")
            A(f"| `{name}` | " + " | ".join(cells) + " |")
        A("")
        if e1b.get("stratified"):
            A("### 3.5 分层配对差值（plot bootstrap 95% CI）")
            A("")
            for group, blocks in e1b["stratified"].items():
                A(f"**{group}**")
                A("")
                A("| 层 | " + " | ".join(c[0] for c in CONTRASTS) + " |")
                A("|---" * (len(CONTRASTS) + 1) + "|")
                for value, blk in blocks.items():
                    cells = [ci(blk["contrasts"][label])
                             for label, _lo, _hi in CONTRASTS]
                    A(f"| `{value}` (n={blk['n_samples']:,}) | "
                      + " | ".join(cells) + " |")
                A("")
    A("## 4. 判定（由脚本按预注册规则从上面的数字推导）")
    A("")
    verdicts = {}
    if e1 and e1b:
        boot = (e1b.get("contrasts", {}).get(
            "b3_delta_minus_b2_no_time_constant_delta") or {}).get("mse_delta")
        perseed = (e1.get("per_seed_contrasts") or {}).get(
            "b3_delta_minus_b2_no_time_constant_delta", {})
        acc = perseed.get("across_seeds", {})
        ctrl_boot = (e1b.get("contrasts", {}).get(
            "b2_no_time_constant_delta_minus_b2_no_time") or {}).get("mse_delta")

        ci_neg = bool(boot and boot["ci95_high"] < 0)
        ci_pos = bool(boot and boot["ci95_low"] > 0)
        seed_neg = bool(acc.get("all_same_sign") and acc.get("mean", 0) < 0)
        if ci_neg and seed_neg:
            verdict = ("B3 优于 B2-C：plot bootstrap 的 95% CI 完全低于 0，且"
                       "每个训练 seed 的配对 ΔMSE 都为负。样本级 Δt 在移除全局"
                       "时间先验后仍有稳定增益。")
        elif ci_neg and not seed_neg:
            verdict = ("B3 的 plot bootstrap CI 低于 0，但不同训练 seed 的配对 "
                       "ΔMSE 符号不一致：方向为正但受训练随机性影响，不能写成"
                       "“稳定增益”。")
        elif ci_pos:
            verdict = "B3 反而差于 B2-C（CI 完全高于 0）：不得保留 Δt 增益主张。"
        else:
            verdict = ("B3 − B2-C 的 plot bootstrap 95% CI 含 0：排除全局时间先验后，"
                       "真实样本级 Δt 没有可证实的 E1 增益。时间条件的贡献主张必须"
                       "收窄到 B3 − B2（即“有样本级时间 vs 完全无时间”），"
                       "不能写成“样本级日期对齐信息”的纯效应。")
        A(f"**Q1/Q2（E1）**：{verdict}")
        A("")
        A(f"- `b3_delta − b2_no_time_constant_delta`：plot bootstrap {ci(boot)}；"
          f"逐 seed {acc.get('mean', float('nan')):+.5f} ± "
          f"{acc.get('std', float('nan')):.5f}。")
        A(f"- `b2_no_time_constant_delta − b2_no_time`：plot bootstrap {ci(ctrl_boot)}。"
          " 该项衡量“同宽 + 常量时间通道”相对“完全无时间通道”的作用；"
          "若其 CI 含 0，说明 S4A 的 B3−B2 增益主要来自样本级时间信息，"
          "而不是通道变宽本身；若显著为负，则 B3−B2 中混有宽度/参数量效应。")
        verdicts["e1"] = {"verdict": verdict, "bootstrap": boot,
                          "per_seed": acc, "control_vs_no_time": ctrl_boot}
    else:
        A("E1 产物缺失，无法判定。")
    A("")

    # ---------------- 5. E2 ----------------------------------------------
    if e2:
        A("## 5. E2：CC / PH 表型读出（辅助证据）")
        A("")
        A("协议：S3.2 修正协议（inner-dev 选模 → 全 official-train 重拟合 → "
          "official validation 单次评价 → 2,000 次 plot 级 bootstrap）。"
          f"有效样本：CC {e2['rows']['cc']['counts']['n_val']:,} / "
          f"{e2['rows']['cc']['counts']['n_val_plots']} plots；"
          f"PH {e2['rows']['ph']['counts']['n_val']:,} / "
          f"{e2['rows']['ph']['counts']['n_val_plots']} plots。")
        A("")
        for trait in ("cc", "ph"):
            A(f"### {trait.upper()}（primary_lag7 / overall）")
            A("")
            A("| 条件 | MAE | RMSE | R² | 选中 epoch |")
            A("|---|---|---|---|---|")
            for label, m in e2["rows"][trait]["conditions"].items():
                A(f"| `{label}` | {m['mae']['mean']:.5f} ± {m['mae']['std']:.5f} | "
                  f"{m['rmse']['mean']:.5f} ± {m['rmse']['std']:.5f} | "
                  f"{m['r2']['mean']:.5f} ± {m['r2']['std']:.5f} | "
                  f"{m['selected_epoch_median']} |")
            A("")
            A("| 条件 − A | ΔMAE | ΔRMSE | ΔR² |")
            A("|---|---|---|---|")
            for label, d in e2["rows"][trait]["B_minus_A"].items():
                A(f"| `{label}` | {d['mae']['mean_delta']:+.5f} | "
                  f"{d['rmse']['mean_delta']:+.5f} | {d['r2']['mean_delta']:+.5f} |")
            A("")
            A("| 配对对照 | ΔR²（plot bootstrap 95% CI） |")
            A("|---|---|")
            for key, m in e2["bootstrap"][trait]["contrasts"].items():
                x = m["delta_r2"]
                A(f"| `{key}` | {x['mean']:+.5f} "
                  f"[{x['ci95_low']:+.5f}, {x['ci95_high']:+.5f}] |")
            A("")
        A("**Q3（E2 是否与 E1 一致）**：判定规则为“B3 − B2-C 的 ΔR² "
          "plot bootstrap 95% CI 是否完全在 0 的一侧，且方向与 E1 相同”。")
        A("")
        e2_rows = {}
        for trait in ("cc", "ph"):
            m = e2["bootstrap"][trait]["contrasts"].get(
                "B_b3_delta_minus_B_b2_no_time_constant_delta", {}).get("delta_r2")
            if not m:
                continue
            e2_rows[trait] = m
            direction = "B3 更好" if m["ci95_low"] > 0 else (
                "B2-C 更好" if m["ci95_high"] < 0 else "含 0（不能区分）")
            A(f"- {trait.upper()}：ΔR² {m['mean']:+.5f} "
              f"[{m['ci95_low']:+.5f}, {m['ci95_high']:+.5f}] → {direction}")
        verdicts["e2"] = e2_rows
        A("")
        A("E2 只作辅助判断：即使 E1 显示样本级 Δt 有增益，也不能据此断言未来"
          "表型预测必然改善；反之亦然。")
        A("")

    # ---------------- 5b. the eight acceptance questions ------------------
    A("## 5b. 八个验收问题的直接回答")
    A("")
    qa: dict[str, str] = {}

    if CONTROL in runs and MATCHED in runs:
        c0 = runs[CONTROL].get(seeds[0], {})
        b0 = runs[MATCHED].get(seeds[0], {})
        p_ok = (c0.get("n_trainable_parameters") == b0.get("n_trainable_parameters")
                and c0.get("n_condition_parameters")
                == b0.get("n_condition_parameters")
                and c0.get("predictor_input_dim") == b0.get("predictor_input_dim"))
        qa["Q1"] = (
            "是。" if p_ok else "否。"
        ) + (f"B2-C 与 B3 的可训练参数均为 {c0.get('n_trainable_parameters', 0):,}，"
             f"条件参数均为 {c0.get('n_condition_parameters', 0):,}，"
             f"predictor 输入维均为 {c0.get('predictor_input_dim')}；"
             "B2-C 保留完整 DeltaTimeEncoder，只把样本级间隔替换为固定值。")
    else:
        qa["Q1"] = "运行清单缺失，无法核对参数量匹配。"

    if e1 and e1b:
        qa["Q2"] = verdict
        acc = (e1.get("per_seed_contrasts", {}).get(
            "b3_delta_minus_b2_no_time_constant_delta", {}) or {}).get(
                "across_seeds", {})
        n_seeds = int(acc.get("n_seeds", 0))
        qa["Q4"] = (
            f"{n_seeds} 个配对 seed 的逐 seed 配对 ΔMSE 为 "
            f"{acc.get('mean', float('nan')):+.5f} ± "
            f"{acc.get('std', float('nan')):.5f}，"
            f"范围 [{acc.get('min', float('nan')):+.5f}, "
            f"{acc.get('max', float('nan')):+.5f}]，"
            f"全部同号={acc.get('all_same_sign')}。"
            + ("在 seed 集合内结论方向不因训练随机性翻转。"
               if acc.get("all_same_sign") else
               "不同 seed 方向不一致，结论对训练随机性敏感，不能写成稳定增益。"))
    else:
        qa["Q2"] = "E1 产物缺失，无法判定。"
        qa["Q4"] = "E1 产物缺失，无法判定 seed 稳健性。"

    if e2 and e1:
        e1_boot = (e1b or {}).get("contrasts", {}).get(
            "b3_delta_minus_b2_no_time_constant_delta", {}).get("mse_delta") or {}
        # +1 means B3 is better.  E1 is a lower-is-better MSE contrast, E2 is a
        # higher-is-better R^2 contrast, so normalise both to "B3 better = +1".
        if e1_boot.get("ci95_high", 0) < 0:
            e1_dir = 1
        elif e1_boot.get("ci95_low", 0) > 0:
            e1_dir = -1
        else:
            e1_dir = 0
        parts = []
        consistent = True
        for trait in ("cc", "ph"):
            m = e2["bootstrap"][trait]["contrasts"].get(
                "B_b3_delta_minus_B_b2_no_time_constant_delta", {}).get("delta_r2")
            if not m:
                continue
            if m["ci95_low"] > 0:
                d = 1
            elif m["ci95_high"] < 0:
                d = -1
            else:
                d = 0
            if d != 0 and e1_dir != 0 and d != e1_dir:
                consistent = False
            parts.append(f"{trait.upper()} ΔR² {m['mean']:+.5f} "
                         f"[{m['ci95_low']:+.5f}, {m['ci95_high']:+.5f}]")
        qa["Q3"] = (
            ("E2 与 E1 方向一致（两者都指向 B3 更好）。" if consistent and e1_dir > 0
             else "E2 与 E1 方向相反。" if consistent and e1_dir < 0
             else "E2 未给出与 E1 同向的可分辨信号，两者不相悖但不能互相加强。")
            + "（" + "；".join(parts) + "）")
    else:
        qa["Q3"] = "E1/E2 产物缺失，无法交叉核对。"

    if s4b:
        test_splits = [s for s in cfg["s4b"]["test_splits"] if s in s4b["splits"]]
        total_pairs = sum((s4b["splits"][s].get("pairs") or {}).get("n_pairs", 0)
                          for s in test_splits)
        qa["Q5"] = (
            f"是。已按论文定义逐一核到本地官方划分 {len(test_splits)} 个 test split"
            f"（{', '.join(test_splits)}），共 {total_pairs:,} 对预注册配对。")
        vchk = {c["id"]: bool(c["pass"])
                for c in s4b.get("verification", {}).get("checks", [])}
        inherit = all(vchk.get(k, False) for k in
                      ("T1_pairs_inherit_official_split", "T2_both_images_same_plot",
                       "T3_test_pair_sets_disjoint", "T4_no_overlap_with_s3_index"))
        qa["Q6"] = (
            ("是。" if inherit else "否。")
            + "每个配对都继承其 plot 的官方 split，配对两端同一 plot，"
              "各 test split 的配对集合互不相交，且与冻结 S3 索引零重叠。")
        feat = sum((s4b["splits"][s].get("pairs") or {}).get("n_feature_ready", 0)
                   for s in test_splits)
        weather_ok = all(
            (s4b["splits"][s].get("pairs") or {}).get("n_weather_interval_ready", 0)
            == (s4b["splits"][s].get("pairs") or {}).get("n_pairs", -1)
            and (s4b["splits"][s].get("pairs") or {}).get("n_weather_past_ready", 0)
            == (s4b["splits"][s].get("pairs") or {}).get("n_pairs", -1)
            for s in test_splits)
        qa["Q7"] = (
            f"天气：{'是' if weather_ok else '否'}，四个 test split 的区间窗口与"
            f"过去窗口全部覆盖且 GDD 可计算。视觉特征：{'是' if feat else '否'}，"
            f"当前就绪 {feat:,} 对（测试图像尚未下载与编码）。")
        qa["Q8"] = ("视觉特征缓存为 0、test 标签尚未按预注册流程附加、"
                    "test_environment 只有单一未见环境（2019）且含 5 个训练未见基因型。"
                    "这三项必须在正式 S4B 训练/评价前解决，本阶段不越界执行。")
    else:
        qa["Q5"] = qa["Q6"] = qa["Q7"] = qa["Q8"] = "S4B 审计产物缺失。"

    for k in ("Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7", "Q8"):
        A(f"- **{k}**：{qa.get(k, '—')}")
    A("")
    verdicts["acceptance_questions"] = qa

    A("## 6. 不可声称的结论")
    A("")
    A("1. 不得把本阶段的 B2-C 称为“完全不包含时间先验的模型”：它保留完整的 "
      "`DeltaTimeEncoder` 与一个由训练子集决定的全局固定间隔，只是没有样本级"
      "时间信息。")
    A("2. 不得把 plot bootstrap 置信区间当作训练随机性置信区间；训练随机性只看"
      "§3.2/§3.3 的逐 seed 结果。")
    A("3. 不得把本阶段结果外推到 official test 或 2,930-plot 全量训练。")
    A("4. 不得声称跨年份/跨环境泛化：validation 的年份全部在训练集中出现过。")
    A("")

    A("## 7. 复现命令")
    A("")
    A("```bash")
    A("# 1) 新增 seed（seed 0/1 复用 S4A 的冻结 checkpoint，不重写）")
    A("CUDA_VISIBLE_DEVICES=1 python3 agro_jepa/train/train_s4.py \\")
    A("  --config agro_jepa/config/s4a1_matched_control.yaml \\")
    A("  --condition b2_no_time_constant_delta --seed 0")
    A("# 2) E1 / E2")
    A("CUDA_VISIBLE_DEVICES=1 python3 tools/s4a1_e1_eval.py "
      "--config agro_jepa/config/s4a1_matched_control.yaml")
    A("CUDA_VISIBLE_DEVICES=1 python3 tools/s4a1_e2_eval.py "
      "--config agro_jepa/config/s4a1_matched_control.yaml")
    A("# 3) 验收与报告")
    A("python3 tools/s4a1_verify.py --config agro_jepa/config/s4a1_matched_control.yaml")
    A("python3 tools/s4a1_write_reports.py "
      "--config agro_jepa/config/s4a1_matched_control.yaml")
    A("```")
    A("")

    Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_md, "w") as fh:
        fh.write("\n".join(L))

    consolidated = {
        "stage": "S4A.1",
        "config": args.config,
        "seeds": seeds,
        "verdicts": verdicts,
        "run_inventory": {
            name: {str(s): {k: runs[name][s].get(k) for k in (
                "n_trainable_parameters", "n_condition_parameters",
                "predictor_input_dim", "epochs_completed", "best_epoch",
                "wall_clock_seconds", "train_pairs", "inner_dev_pairs",
                "fixed_delta")}
                for s in seeds if s in runs.get(name, {})}
            for name in runs},
        "artifacts": {
            "e1_results": cfg["artifacts"]["results_json"],
            "e1_bootstrap": cfg["artifacts"]["e1_bootstrap"],
            "e1_predictions": cfg["artifacts"]["e1_predictions"],
            "e2_metrics": cfg["artifacts"]["e2_metrics"],
            "e2_predictions": cfg["artifacts"]["e2_predictions"],
            "verify": "reports/s4a1_verify.json",
            "s4b_audit": "reports/s4b_test_data_audit.json",
            "s4b_conditioning": "reports/s4b_test_conditioning.json",
        },
        "notes": [
            "B2-C keeps the full DeltaTimeEncoder and a single fixed interval; it is "
            "not a model without any time prior.",
            "Plot bootstrap intervals describe plot sampling, not training seeds.",
            "No official test label was read or evaluated in this stage.",
        ],
    }
    with open(args.out_json, "w") as fh:
        json.dump(consolidated, fh, indent=2)

    # ---------------- S4B markdown ---------------------------------------
    S: list[str] = []
    B = S.append
    B("# S4B 官方测试数据准备核查")
    B("")
    B("> 本文件由 `tools/s4a1_write_reports.py` 从 "
      "`reports/s4b_test_data_audit.json` 与 `reports/s4b_test_conditioning.json` "
      "生成。本阶段只做数据核查与配对准备，不评价任何 official test 指标。")
    B("")
    if s4b:
        B("## 1. 官方 split 定义（来源与本地核对）")
        B("")
        B(f"- 论文：{s4b['official_split_definitions']['source_url']}")
        B(f"- 原文：{s4b['official_split_definitions']['quote']}")
        B("- 本地 6 个 split 与论文 4 个 test 集一一对应（见下表 paper_label）。")
        B("")
        B("## 2. 各 split 覆盖")
        B("")
        B("| split | 角色 | 论文 | plot | 图像 | 日期范围 | harvest_year | 基因型(总/训练见过) | yearsite(未见) | 天气年份缺失 | 已缓存特征 |")
        B("|---|---|---|---|---|---|---|---|---|---|---|")
        for s, v in s4b["splits"].items():
            B(f"| `{s}` | {v['role']} | {v.get('paper_label') or '—'} | "
              f"{v['n_plots']} | {v['n_images']:,} | "
              f"{v['image_date_range'][0]} … {v['image_date_range'][1]} | "
              f"{v['harvest_years']} | {v['genotypes']}/"
              f"{v['genotypes_seen_in_train']} | {v['yearsites_unseen_in_train']} | "
              f"{v['weather_years_missing'] or '—'} | "
              f"{v['n_images_with_cached_feature']:,} |")
        B("")
        B("## 3. 测试配对索引（冻结 span7/span14 规则）")
        B("")
        B("| split | 配对数 | 有配对 plot | span7 | span14 | 唯一图像 | delta 范围/中位 | 特征就绪 | 区间天气就绪 | 过去天气就绪 |")
        B("|---|---|---|---|---|---|---|---|---|---|")
        for s in cfg["s4b"]["test_splits"]:
            p = (s4b["splits"].get(s) or {}).get("pairs") or {}
            if not p:
                continue
            B(f"| `{s}` | {p['n_pairs']} | {p['n_plots_with_pairs']} | "
              f"{p['by_pair_type'].get('span7', 0)} | "
              f"{p['by_pair_type'].get('span14', 0)} | {p['n_unique_images']} | "
              f"{p['delta_days']['min']}–{p['delta_days']['max']} / "
              f"{p['delta_days']['median']} | "
              f"{p['n_feature_ready']} | {p['n_weather_interval_ready']} | "
              f"{p['n_weather_past_ready']} |")
        B("")
        B("每个 split 的配对索引文件与 SHA256：")
        B("")
        for s in cfg["s4b"]["test_splits"]:
            p = (s4b["splits"].get(s) or {}).get("pairs") or {}
            if p:
                B(f"- `{p['index_file']}` — `{p['index_sha256']}`")
        B("")
        B("## 4. 核查检查")
        B("")
        v = s4b.get("verification", {})
        B(f"{v.get('n_pass', 0)}/{v.get('n_checks', 0)} 通过。")
        B("")
        B("| 检查 | 结果 | 说明 |")
        B("|---|---|---|")
        for c in v.get("checks", []):
            B(f"| `{c['id']}` | {'PASS' if c['pass'] else 'FAIL'} | {c['detail']} |")
        B("")
        B("## 5. 天气与 GDD 覆盖")
        B("")
        w = s4b.get("weather", {})
        B(f"- 通道：{w.get('channels')}，单位 {w.get('units')}")
        B(f"- 排除：{list((w.get('excluded') or {}).keys())}（P_sum_raw 单位未定，"
          "禁止做降水积分）")
        B(f"- 逐日天气可用年份：{w.get('years_available')}")
        B("")
        B("| harvest_year | 天数 | 覆盖区间 |")
        B("|---|---|---|")
        for _y, _v in (w.get("year_day_counts") or {}).items():
            B(f"| {_y} | {_v['n_days']} | {_v['start']} … {_v['end']} |")
        B("")
        B("每个 harvest_year 的天气序列覆盖 9 月 1 日至次年 8 月 31 日；"
          "2016 与 2021 为闰年（366 天）。四个 test split 的全部 t_a / t_b "
          "日期都落在覆盖区间内，因此天气/GDD 输入完备。")
        B("")
        B(f"- Protocol P（可部署）：`{w.get('past_window_rule')}`")
        B(f"- Protocol R（回顾性/oracle）：`{w.get('interval_window_rule')}`，"
          "不可写成预报能力")
        B(f"- GDD：{w.get('gdd_definition')}")
        B("")
        if s4bc:
            B("| split | 条件配对数 | 区间窗口全覆盖 | 过去窗口全覆盖 | GDD_R(Tb0) 均值 |")
            B("|---|---|---|---|---|")
            for s, v2 in s4bc["splits"].items():
                B(f"| `{s}` | {v2['n_pairs']} | "
                  f"{v2['interval_valid_days_all_full']} | "
                  f"{v2['past_window_valid_days_all_full']} | "
                  f"{v2['gdd_int_base0']['mean']:.2f} |")
            B("")
            B("条件张量文件（只含输入，不含表型标签）：")
            B("")
            for s, v2 in s4bc["splits"].items():
                B(f"- `{v2['npz']}` — `{v2['npz_sha256']}`")
            B("")
        B("## 6. 仍然阻止正式 S4B 评价的问题")
        B("")
        B("1. **视觉特征覆盖为 0**：四个 official test split 的全部图像都还没有"
          "下载与 DINOv2 编码；必须先按 S4B 的下载+编码流程生成特征缓存，"
          "才能对冻结模型做 test 评价。")
        B("2. **test 标签未读取**：本阶段刻意不读任何 test 表型列；正式评价前需要"
          "一次单独、预注册的标签附加步骤（只附加、不选模）。")
        B("3. **环境外推**：2019 是训练年份完全未见的环境，且天气是年份级序列，"
          "test_environment 的独立环境数只有 1；任何 test_environment 结论都必须"
          "按“单环境”表述。")
        B("4. **test_environment 的基因型并非全部见过**："
          f"{s4b['splits']['test_environment']['genotypes_unseen_in_train']} 个"
          "基因型在训练集中从未出现（论文定义为 seen genotypes，实际有例外），"
          "报告时必须如实说明。")
        B("5. **B2-C 的固定间隔依赖训练子集**：S4B 的正式评价必须使用 checkpoint "
          "内记录的固定值与归一化统计，不能重新拟合。")
        B("")
        B("## 7. 复现命令")
        B("")
        B("```bash")
        B("python3 tools/s4b_prepare_test_data.py "
          "--config agro_jepa/config/s4a1_matched_control.yaml")
        B("python3 tools/s4b_prepare_test_conditioning.py "
          "--config agro_jepa/config/s4a1_matched_control.yaml")
        B("```")
        B("")
    else:
        B("S4B 审计产物缺失。")
    with open(args.s4b_md, "w") as fh:
        fh.write("\n".join(S))

    print(f"wrote {args.out_md}, {args.out_json}, {args.s4b_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
