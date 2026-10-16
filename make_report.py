"""Generate Chinese delivery note and scientific plots from completed runs."""
import csv
import os
import json
from pathlib import Path
import numpy as np
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".runtime_cache" / "matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from trajectory_encoder.common import ROOT, read_json, write_json


def main():
    sel = read_json(ROOT / "reports" / "selection.json")
    results = read_json(ROOT / "reports" / "test_results.json")
    initial = read_json(ROOT / "reports" / "untrained_test_results.json")
    targets = read_json(ROOT / "config.json")["evaluation"]["targets"]
    chosen = sel["selected_run"]["run_id"]
    labels = {"path_m": "Path length", "curvature_rad": "Accumulated turn", "sinuosity": "Sinuosity", "duration_s": "Duration"}
    rows = []
    for job in sel["grid"]:
        name = job["run_id"]
        for pooling in ("sum", "mean", "gem2"):
            result = results[name][pooling]
            for target in targets:
                rows.append({"run_id": name, "dimension": job["dimension"], "seed": job["seed"], "pooling": pooling,
                    "target": target, "validation_R2": result["probes"]["global_validation_R2"][target],
                    "test_R2": result["probes"]["global_test_R2"][target],
                    "within_STAR_test_macro_R2": result["probes"]["within_STAR_test_macro_R2"][target],
                    "untrained_test_R2": initial[name][pooling]["global_test_R2"][target],
                    "untrained_within_STAR_test_macro_R2": initial[name][pooling]["within_STAR_test_macro_R2"][target]})
    with (ROOT / "reports" / "probe_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    summary = {}
    for dimension in (4, 8, 16):
        for pooling in ("sum", "mean", "gem2"):
            values = [r for r in rows if r["dimension"] == dimension and r["pooling"] == pooling]
            entry = {}
            for metric in ("test_R2", "within_STAR_test_macro_R2", "untrained_test_R2", "untrained_within_STAR_test_macro_R2"):
                entry[metric] = {t: {"mean": float(np.mean([r[metric] for r in values if r["target"] == t])),
                                     "std_ddof1": float(np.std([r[metric] for r in values if r["target"] == t], ddof=1))} for t in targets}
            summary[f"d{dimension}_{pooling}"] = entry
    write_json(ROOT / "reports" / "three_seed_summary.json", summary)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 140})
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    for j, target in enumerate(targets):
        ax = axes.flat[j]
        dims = np.array([4, 8, 16])
        for pooling, color, offset in (("sum", "#1769aa", -.7), ("mean", "#e98b29", 0), ("gem2", "#39886b", .7)):
            values = [summary[f"d{d}_{pooling}"]["within_STAR_test_macro_R2"][target] for d in dims]
            ax.errorbar(dims+offset, [v["mean"] for v in values], yerr=[v["std_ddof1"] for v in values], color=color, marker="o", capsize=3, label=pooling)
        ax.set(title=labels[target], xlabel="Embedding dimension", ylabel="Within-STAR TEST R²", xticks=dims)
        ax.axhline(0, color="gray", lw=.6); ax.grid(alpha=.2)
    axes.flat[0].legend(ncol=3)
    fig.suptitle("Held-out geometric proxies: mean ± SD over 3 seeds (choices fixed on validation)")
    fig.savefig(ROOT / "reports" / "dimension_pooling.png"); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for job in sel["grid"]:
        history = read_json(ROOT / "runs" / job["run_id"] / "history.json")
        for ax, metric in zip(axes, ("train_loss", "validation_loss")):
            ax.plot([r["epoch"] for r in history], [r[metric] for r in history], label=f"D{job['dimension']} seed{job['seed']}", alpha=.75, lw=1)
            ax.set(xlabel="Epoch", ylabel="Flat-SNN loss", title=metric); ax.grid(alpha=.2)
    axes[1].legend(fontsize=7, ncol=2)
    fig.savefig(ROOT / "reports" / "training_curves.png"); plt.close(fig)
    confounding = read_json(ROOT / "reports" / "confounding_and_route.json")
    length_baseline = confounding["length_only_cubic_ridge"]["within_STAR_test_macro_R2"]
    trained_proxy = results[chosen]["sum"]["probes"]["within_STAR_test_macro_R2"]
    random_proxy = initial[chosen]["sum"]["within_STAR_test_macro_R2"]
    fig, ax = plt.subplots(figsize=(10, 4.5), constrained_layout=True)
    positions = np.arange(len(targets))
    for offset, name, values, color in [(-.25, "Trained Encoder", trained_proxy, "#1769aa"),
                                       (0., "Random Encoder", random_proxy, "#e98b29"),
                                       (.25, "Length-only ridge", length_baseline, "#7d7d7d")]:
        ax.bar(positions + offset, [values[t] for t in targets], width=.24, label=name, color=color)
    ax.set(xticks=positions, xticklabels=[labels[t] for t in targets], ylabel="Within-STAR TEST macro R²",
           title="Selected frozen Encoder vs initialization and length baselines")
    ax.axhline(0, color="black", lw=.6); ax.grid(axis="y", alpha=.2); ax.legend(ncol=3, fontsize=9)
    fig.savefig(ROOT / "reports" / "baseline_comparison.png"); plt.close(fig)
    histories = [read_json(ROOT / "runs" / j["run_id"] / "history.json") for j in sel["grid"]]
    epoch_count = sum(len(h) for h in histories)
    selected = results[chosen]["sum"]
    main_rows = [r for r in rows if r["run_id"] == chosen and r["pooling"] == "sum"]
    header = "| 几何代理指标 | TEST R² | STAR 内 TEST 宏平均 R² | 随机初始化 STAR 内 R² | 长度基线 STAR 内 R² |\n|---|---:|---:|---:|---:|\n"
    table = header + "\n".join(f"| {r['target']} | {r['test_R2']:.4f} | {r['within_STAR_test_macro_R2']:.4f} | {r['untrained_within_STAR_test_macro_R2']:.4f} | {length_baseline[r['target']]:.4f} |" for r in main_rows)
    stronger = [t for t in targets if trained_proxy[t] > max(random_proxy[t], length_baseline[t])]
    baseline_interpretation = ("所选模型在这些代理上数值高于两种基线：" + ", ".join(stronger) + "；其余代理尚未显示超过较强基线。") if stronger else "所选模型在四个代理上均未超过两种基线中的较强者，因此这些测试尚未证明自监督训练带来了额外的几何代理预测能力。"
    write_json(ROOT / "reports" / "quality_summary.json", {"training_complete": True, "selected_run": chosen,
               "within_STAR_trained": trained_proxy, "within_STAR_untrained": random_proxy,
               "within_STAR_length_only": length_baseline, "proxies_numerically_above_both_baselines": stronger,
               "complexity_application_validated": False, "interpretation": baseline_interpretation})
    note = f"""# Encoder 训练结果

已完成 PLAN.md 所要求的 Encoder 训练与当前数据支持的表示评估。主模型由验证集选择为 **{sel['selected_dimension']} 维、sum pooling、seed {sel['selected_run']['seed']}**。冻结权重在 `encoder/encoder.pt`，对应 `{chosen}`；三维数×三种子的九份最佳验证 checkpoint 在 `checkpoints/`。

## 实际完成

- 超参数筛选：8维 seed42 的 ε=120/1200m × τ=.01/.05，4次独立30轮训练。
- 维数/稳定性：固定所选 ε={sel['selected_run']['epsilon']}m、τ={sel['selected_run']['temperature']}，4/8/16维各3种子，9次30轮；复用筛选中的一次，总计12个独立模型、360轮。
- 批大小16，bf16混合精度；初期两进程并行，一个进程不推进后按checkpoint改为串行续训，没有梯度累积。验证选择 checkpoint 后才最终评估 TEST。
- 训练/验证/测试：6877/1214/8092条，使用完整轨迹按5秒采样并保留终点。原环境和原始数据未修改。

## 代表权重的独立测试

{table}

{baseline_interpretation}

所有探针只在训练集拟合，在验证集选择 ridge alpha。STAR 内宏平均用于减少仅区分航路的影响，三种子结果和 mean/GeM 敏感性见 `three_seed_summary.json` 与 `probe_summary.csv`。这些目标是几何代理量，不是复杂度真值。随机初始化基线用于观察训练带来的变化，不能把初始化已可预测的信号都归因于自监督训练。

## 表示诊断

代表模型 TEST 有效秩={selected['diagnostics']['effective_rank']:.4f} / {sel['selected_dimension']}；阈值1e-8下低方差维度数={selected['diagnostics']['collapsed_dimensions']}；sum向量范数与样本长度相关系数={selected['diagnostics']['norm_length_pearson']:.4f}。有效秩低于输出维数说明信息可能集中于少数方向；逐维方差、距离分布、长度分层 probe 均保存在 `test_results.json`，不能仅凭向量非零就宣称没有表示退化。sum与长度的混杂应结合mean/GeM和长度分层结果使用。

## 交付与边界

权重、预处理参数、配置、原始文件SHA256、每条轨迹的split清单、seed、逐轮日志、验证选择记录、测试结果、随机初始化基线、可调用冻结接口和真实样例均已保存。`export_verification.json` 检验加载、冻结和重建特征一致性；`checkpoint_manifest.json` 给出九份权重校验值。

数据缺绝对时间及真实航班ID，所以真实航班/日期级隔离和小时点云/相关维数稳定区间没有验证。本交付是训练完成的 ATSCC/ITSC-inspired Encoder，未宣称完整小时复杂度系统或原论文严格复现。运行中一次PowerShell遥测拦截已记录在 PROTOCOL.md，随后通过禁止外连的进程从checkpoint续训。
"""
    (ROOT / "reports" / "RESULTS_ZH.md").write_text(note, encoding="utf-8")
    print("Report written: reports/RESULTS_ZH.md")


if __name__ == "__main__":
    main()
