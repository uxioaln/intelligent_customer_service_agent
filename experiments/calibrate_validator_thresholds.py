# -*- coding: utf-8 -*-
"""
experiments.calibrate_validator_thresholds - 验证器阈值校准工具

背景：
- 当前 validator.py 的 PASS_THRESHOLD=0.6 和 CLAIM_HALLUCINATION_THRESHOLD=0.3
  是人工设定，未经过阈值校准。不同业务场景可能需要不同的阈值。
- 本工具基于人工标注数据，用 Cohen's Kappa 系数评估人工标注与模型判定的一致性，
  在不同阈值组合下搜索最优 operating point。

使用方法：

Step 1（首次运行）：从已有的 accuracy_eval_report.json 导出待标注样本
    python experiments/calibrate_validator_thresholds.py --export-to-label

    输出 -> data/stats/to_label.jsonl
    每条记录包含 query、answer 片段、模型原始判定结果（overall / claim_rate）。
    人工阅读后在该文件中补充 "human_label" 字段：
        "correct"    — 答案正确，无幻觉
        "hallucination" — 答案包含幻觉
        "incorrect"  — 答案错误（非幻觉，如答非所问、要点遗漏）
    标注完成后将文件重命名为 data/stats/human_labels.jsonl。

Step 2：运行网格搜索校准阈值
    python experiments/calibrate_validator_thresholds.py

    脚本读取 human_labels.jsonl，对 PASS_THRESHOLD 和 CLAIM_HALLUCINATION_THRESHOLD
    做网格搜索，计算 Cohen's Kappa，输出最优阈值组合及对应的一致率。

技术说明：
- Cohen's Kappa：用于衡量两名标注者（人工 vs 模型）之间的一致性，排除随机因素。
  κ = (P_o - P_e) / (1 - P_e)，范围 [-1, 1]。
  - κ > 0.8：几乎完全一致
  - 0.6 < κ <= 0.8：实质性一致
  - 0.4 < κ <= 0.6：中等一致
- 本脚本将 "correct" 视为非幻觉（0），"hallucination" 视为幻觉（1）。
  "incorrect" 不参与幻觉判定校准，仅做统计。
"""
import argparse
import json
import logging
import os
import sys
from itertools import product
from typing import Any, Dict, List

# 把工程根加入 path，使 app.* 包可被导入
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from app.config import STATS_DIR

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ========== 文件路径 ==========
ACCURACY_EVAL_REPORT = os.path.join(STATS_DIR, "accuracy_eval_report.json")
TO_LABEL_FILE = os.path.join(STATS_DIR, "to_label.jsonl")
HUMAN_LABEL_FILE = os.path.join(STATS_DIR, "human_labels.jsonl")
CALIBRATION_REPORT = os.path.join(STATS_DIR, "threshold_calibration_report.json")

# ========== 网格搜索参数 ==========
# PASS_THRESHOLD 候选（维度打分 overall 阈值，0.0~1.0，步长 0.05）
PASS_THRESHOLD_GRID = [round(x * 0.05, 2) for x in range(3, 17)]  # 0.15 ~ 0.80
# CLAIM_HALLUCINATION_THRESHOLD 候选（claim 幻觉率阈值，0.0~1.0，步长 0.05）
CLAIM_THRESHOLD_GRID = [round(x * 0.05, 2) for x in range(1, 11)]  # 0.05 ~ 0.50


def _cohens_kappa(y_true: List[int], y_pred: List[int]) -> float:
    """计算 Cohen's Kappa（二分类）

    参数：
        y_true: 人工标注（0=非幻觉，1=幻觉）
        y_pred: 模型判定（0=非幻觉，1=幻觉）

    返回：
        κ 值，范围 [-1, 1]
    """
    if not y_true or not y_pred or len(y_true) != len(y_pred):
        return 0.0
    n = len(y_true)
    # 构建混淆矩阵
    tp = fp = fn = tn = 0
    for t, p in zip(y_true, y_pred):
        if t == 1 and p == 1:
            tp += 1
        elif t == 0 and p == 1:
            fp += 1
        elif t == 1 and p == 0:
            fn += 1
        else:
            tn += 1

    # 观察一致率 Po
    po = (tp + tn) / n if n else 0.0

    # 随机一致率 Pe
    p_true_1 = (tp + fn) / n if n else 0.0
    p_pred_1 = (tp + fp) / n if n else 0.0
    p_true_0 = 1 - p_true_1
    p_pred_0 = 1 - p_pred_1
    pe = p_true_1 * p_pred_1 + p_true_0 * p_pred_0

    if abs(1 - pe) < 1e-9:
        return 1.0 if abs(po - pe) < 1e-9 else 0.0

    kappa = (po - pe) / (1 - pe)
    return round(kappa, 4)


def _model_decision(row: Dict[str, Any], pass_threshold: float, claim_threshold: float) -> int:
    """用给定阈值判定模型是否判定为幻觉

    判定逻辑与 agent_accuracy_eval.py 保持一致：
    - claim_hallucination_rate 存在时优先使用 claim 级判定
    - 否则退回维度打分 overall
    """
    claim_rate = row.get("claim_hallucination_rate")
    if claim_rate is not None and row.get("claim_count", 0) > 0:
        return 1 if claim_rate > claim_threshold else 0
    overall = row.get("validation_overall", 0.0)
    try:
        overall = float(overall)
    except (TypeError, ValueError):
        overall = 0.0
    return 1 if overall < pass_threshold else 0


def export_to_label() -> None:
    """从 accuracy_eval_report.json 导出待标注样本"""
    if not os.path.exists(ACCURACY_EVAL_REPORT):
        logger.error("准确率评测报告不存在: %s，请先跑 agent_accuracy_eval.py", ACCURACY_EVAL_REPORT)
        sys.exit(1)

    with open(ACCURACY_EVAL_REPORT, "r", encoding="utf-8") as f:
        report = json.load(f)

    # 同时取基线和 Agent 的 per_query，分别标注
    records = []
    for label, key in (("baseline", "baseline"), ("agent", "agent")):
        per_query = report.get(key, {}).get("per_query", [])
        for item in per_query:
            records.append({
                "system": label,
                "query_id": item.get("id", ""),
                "query": item.get("query", ""),
                "expected_answer": item.get("expected_answer", ""),
                "actual_answer": item.get("actual_answer", ""),
                "validation_overall": item.get("validation_overall"),
                "claim_count": item.get("claim_count", 0),
                "claim_hallucination_rate": item.get("claim_hallucination_rate", 0.0),
                # 以下是人工标注时填写的字段（请阅读答案后判断）
                "human_label": None,  # correct / hallucination / incorrect / 不标注
                "human_note": "",     # 标注理由（可选）
            })

    os.makedirs(STATS_DIR, exist_ok=True)
    with open(TO_LABEL_FILE, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    logger.info("已导出 %s 条待标注样本 -> %s", len(records), TO_LABEL_FILE)
    print("""
标注说明：
1. 打开 %s
2. 对每条记录补充 "human_label" 字段：
   - "correct"       : 答案正确，无幻觉
   - "hallucination" : 答案包含无法由检索原文支撑的事实
   - "incorrect"     : 答案错误但不属于幻觉（如答非所问、要点遗漏）
   - 不确定可留 None
3. 标注完成后将文件重命名为 %s
4. 运行：python experiments/calibrate_validator_thresholds.py
""" % (TO_LABEL_FILE, HUMAN_LABEL_FILE))


def load_human_labels() -> List[Dict[str, Any]]:
    """加载人工标注结果"""
    if not os.path.exists(HUMAN_LABEL_FILE):
        logger.error("人工标注文件不存在: %s", HUMAN_LABEL_FILE)
        # 如果 to_label 存在但 human_labels 不存在，提示用户先标注
        if os.path.exists(TO_LABEL_FILE):
            print("请先完成标注：编辑 %s 并补充 human_label，然后重命名为 %s" % (TO_LABEL_FILE, HUMAN_LABEL_FILE))
        else:
            print("首次运行请先导出待标注样本：\n  python experiments/calibrate_validator_thresholds.py --export-to-label")
        sys.exit(1)

    records = []
    with open(HUMAN_LABEL_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            label = r.get("human_label")
            if label not in ("correct", "hallucination", "incorrect", None):
                continue
            records.append(r)
    return records


def run_calibration(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """网格搜索最优阈值组合"""
    # 仅使用有明确人工标注的样本
    labeled = [r for r in records if r.get("human_label") in ("correct", "hallucination")]
    logger.info("有效人工标注样本: %s 条（共 %s 条）", len(labeled), len(records))
    if len(labeled) < 10:
        logger.warning("人工标注样本不足 10 条，Kappa 系数置信度较低，建议至少标注 50 条")

    # 人工标注转 0/1：hallucination=1，correct=0
    y_true = [1 if r["human_label"] == "hallucination" else 0 for r in labeled]

    results = []
    best_kappa = -1.0
    best = None

    for pass_t, claim_t in product(PASS_THRESHOLD_GRID, CLAIM_THRESHOLD_GRID):
        y_pred = [_model_decision(r, pass_t, claim_t) for r in labeled]
        kappa = _cohens_kappa(y_true, y_pred)
        # 同时计算 accuracy 作为参考
        acc = sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true) if y_true else 0.0
        results.append({
            "pass_threshold": pass_t,
            "claim_threshold": claim_t,
            "cohens_kappa": kappa,
            "accuracy": round(acc, 4),
            "sample_count": len(labeled),
        })
        if kappa > best_kappa:
            best_kappa = kappa
            best = {"pass_threshold": pass_t, "claim_threshold": claim_t, "cohens_kappa": kappa, "accuracy": round(acc, 4)}

    # 按 Kappa 降序排
    results.sort(key=lambda x: x["cohens_kappa"], reverse=True)

    # 生成报告
    report = {
        "metadata": {
            "total_labeled": len(records),
            "valid_labeled": len(labeled),
            "pass_threshold_grid": PASS_THRESHOLD_GRID,
            "claim_threshold_grid": CLAIM_THRESHOLD_GRID,
            "current_pass_threshold": 0.6,
            "current_claim_threshold": 0.3,
        },
        "top_10": results[:10],
        "best": best,
        "all_results": results,
    }
    return report


def print_report(report: Dict[str, Any]) -> None:
    """打印校准结果到控制台"""
    meta = report["metadata"]
    print("\n" + "=" * 70)
    print("  验证器阈值校准报告（Cohen's Kappa）")
    print("=" * 70)
    print("  有效人工标注样本: %s 条（总共 %s 条）" % (meta["valid_labeled"], meta["total_labeled"]))
    print("  当前 PASS_THRESHOLD:        %.2f" % meta["current_pass_threshold"])
    print("  当前 CLAIM_HALLUCINATION_THRESHOLD: %.2f" % meta["current_claim_threshold"])
    print()
    best = report.get("best")
    if best:
        print("  ★ 最优阈值组合：")
        print("    PASS_THRESHOLD = %.2f" % best["pass_threshold"])
        print("    CLAIM_HALLUCINATION_THRESHOLD = %.2f" % best["claim_threshold"])
        print("    Cohen's Kappa = %.4f" % best["cohens_kappa"])
        print("    Accuracy      = %.2f%%" % (best["accuracy"] * 100))
    print()
    print("  前 10 名阈值组合：")
    print("  %-12s %-12s %-12s %-12s" % ("PASS阈值", "CLAIM阈值", "Kappa", "Accuracy"))
    print("  " + "-" * 48)
    for r in report["top_10"]:
        print("  %-12.2f %-12.2f %-12.4f %-12.2f%%" % (
            r["pass_threshold"], r["claim_threshold"], r["cohens_kappa"], r["accuracy"] * 100))
    print("=" * 70)
    print("  报告已保存: %s" % CALIBRATION_REPORT)
    print()
    print("  阈值解读参考：")
    print("    κ > 0.80 : 几乎完全一致")
    print("    0.60 < κ ≤ 0.80 : 实质性一致（推荐）")
    print("    0.40 < κ ≤ 0.60 : 中等一致")
    print("    κ ≤ 0.40 : 一致度较低，需扩大人工标注量或重新设计判定规则")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="验证器阈值校准工具")
    parser.add_argument("--export-to-label", action="store_true", help="导出待标注样本，不跑校准")
    args = parser.parse_args()

    if args.export_to_label:
        export_to_label()
        return

    records = load_human_labels()
    report = run_calibration(records)

    os.makedirs(STATS_DIR, exist_ok=True)
    with open(CALIBRATION_REPORT, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print_report(report)


if __name__ == "__main__":
    main()
