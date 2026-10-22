# -*- coding: utf-8 -*-
"""
experiments.badcase_diagnose - Badcase 自动定位与归因诊断脚本

从已生成的评测报告中筛选 badcase，按"五段管线决策树"自动归因到具体环节，
输出可读的 JSON + Markdown 诊断报告，供人工排查根因。

输入报告（由 run_eval.py 产出）：
- data/stats/accuracy_eval_report.json   首答准确率 + 幻觉率报告（含 baseline/agent per_query）
- data/stats/cross_modal_recall_report.json  跨模态 Recall@5 报告（含 per_query）

归因决策树：
1. accuracy 报告 badcase（judge_correct=False）
   - error 字段存在                 -> 系统异常（调用失败/限流）
   - coverage = fail
     - references_count == 0        -> 召回完全失败（查询改写跑偏/相关度不足触发拒答）
     - references_count > 0         -> 召回不足或排序错位（rerank 挤出真证据/生成漏答）
   - coverage = pass & faithfulness=fail -> 生成幻觉（LLM 自由发挥/温度过高/截断）
   - claim_hallucination_rate > 0.3 -> claim 级幻觉（无支撑 claim 占比高）
   - abstained = True               -> 拒答（验证失败后拒答/召回不足）
   - coverage=pass & faithfulness=pass 但 judge 仍判错 -> judge 误判（expected_answer 不完整/口径错位）

2. cross_modal 报告 badcase（hit=False 或 precise_hit=False）
   - hit = False（模态集合口径未命中）    -> 跨模态召回完全失败（目标模态未进入 top-k）
   - hit = True & precise_hit = False    -> 模态配额保送（真实排序差，目标媒体向量排名低）
   - expected_modalities 含 video        -> 额外建议：视频语义描述需多角度增强

输出文件：
- data/stats/badcase_diagnose_report.json  （机器可读结构化诊断结果）
- data/stats/badcase_diagnose_report.md    （人类可读归因报告）

使用方法：
    python experiments/badcase_diagnose.py
"""
import json
import logging
import os
import sys
from collections import Counter
from datetime import datetime
from typing import Any, Dict, List, Optional

# 把工程根加入 path，使 app.* 包与 experiments.* 包可被导入
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from app.config import STATS_DIR

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ========== 输入/输出文件 ==========
ACCURACY_REPORT_FILE = os.path.join(STATS_DIR, "accuracy_eval_report.json")
CROSS_MODAL_REPORT_FILE = os.path.join(STATS_DIR, "cross_modal_recall_report.json")
DIAGNOSE_REPORT_FILE = os.path.join(STATS_DIR, "badcase_diagnose_report.json")
DIAGNOSE_MD_FILE = os.path.join(STATS_DIR, "badcase_diagnose_report.md")

# ========== 归因阈值（与项目 config / validator 对齐） ==========
# claim 级幻觉率阈值：超过此比例无支撑 claim 即判幻觉（与 validator.py CLAIM_HALLUCINATION_THRESHOLD 一致）
CLAIM_HALLUCINATION_THRESHOLD = 0.3
# 验证器综合分通过阈值（与 validator.py PASS_THRESHOLD 一致）
VALIDATOR_PASS_THRESHOLD = 0.6


def _load_json(path: str) -> Optional[Dict[str, Any]]:
    """安全加载 JSON 报告文件，不存在或解析失败返回 None"""
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("加载报告失败 %s: %s", path, e)
        return None


# ========== accuracy 报告归因 ==========

def diagnose_accuracy_badcase(item: Dict[str, Any], side: str) -> Dict[str, Any]:
    """对单条 accuracy badcase（judge_correct=False）按决策树归因

    Args:
        item: per_query 单条记录（baseline 或 agent 侧）
        side: "baseline" 或 "agent"，用于区分链路并补充特有信号

    Returns:
        {"category": str, "signals": dict, "root_causes": list, "suggestion": str}
    """
    coverage = item.get("judge_coverage", "unknown")
    faithfulness = item.get("judge_faithfulness", "unknown")
    references_count = item.get("references_count", 0)
    validation_overall = item.get("validation_overall")
    claim_count = item.get("claim_count", 0)
    claim_hallu_rate = item.get("claim_hallucination_rate", 0.0)
    abstained = item.get("abstained", False)
    retried = item.get("retried", False)

    # 采集关键信号，供报告中直接查阅
    signals: Dict[str, Any] = {
        "coverage": coverage,
        "faithfulness": faithfulness,
        "references_count": references_count,
        "validation_overall": validation_overall,
        "is_hallucination": item.get("is_hallucination"),
        "judge_reason": item.get("judge_reason", ""),
    }
    if side == "agent":
        signals.update({
            "claim_count": claim_count,
            "unsupported_count": item.get("unsupported_count", 0),
            "claim_hallucination_rate": claim_hallu_rate,
            "retried": retried,
            "abstained": abstained,
            "steps_count": item.get("steps_count", 0),
        })

    root_causes: List[str] = []
    suggestion = ""

    # 1) 系统异常（重试3次仍失败 / judge JSON 解析失败）
    if item.get("error"):
        category = "系统异常"
        root_causes = [
            "基线/Agent 调用异常（重试3次仍失败或 API 返回空响应）",
            "可能原因：API 限流、QPS 超限、网络瞬时故障",
        ]
        suggestion = "检查 AGICTO_API_KEY 配额与 QPS 限制，必要时增大 LLM_CALL_INTERVAL。"
        return _pack(category, signals, root_causes, suggestion)

    # 2) 召回环节问题：coverage = fail（遗漏多数要点）或 marginal（遗漏1-2要点）
    if coverage in ("fail", "marginal"):
        if references_count == 0:
            category = "召回完全失败"
            root_causes = [
                "查询改写跑偏：HyDE 改写未覆盖原查询实体或语义",
                "向量检索 + BM25 关键词检索均未命中目标 chunk",
                "检索相关度低于 MIN_RELEVANCE_SCORE(0.3) 或条数不足触发拒答",
            ]
            suggestion = "排查查询改写输出，检查该 query 的向量/BM25 候选池是否为空，确认知识库覆盖该知识点。"
        else:
            category = "召回不足或排序错位"
            root_causes = [
                "rerank 精排将真正证据挤出 top-k（候选池有但未被选入）",
                "查询改写未覆盖查询中的关键实体（多实体/跨条件遗漏）",
                "生成阶段漏答：召回到正确 chunk 但 LLM 未提取要点",
            ]
            suggestion = "调高检索 top-k 扩大候选池，复查 rerank 输入顺序，检查生成 prompt 是否要求逐点回答。"
        return _pack(category, signals, root_causes, suggestion)

    # 3) 生成环节问题：coverage=pass 但 faithfulness=fail（含原文不存在的事实）
    if faithfulness == "fail":
        category = "生成幻觉"
        root_causes = [
            "LLM 自由发挥，输出了检索原文中不存在的事实",
            "生成 temperature 过高（建议降至 0.3）",
            "MAX_TOKENS 截断导致答案后半段无原文支撑",
        ]
        suggestion = "降低生成 temperature 至 0.3，在 system prompt 中强化「严格忠于检索原文，不得编造」的要求，检查 MAX_TOKENS 是否截断要点。"
        return _pack(category, signals, root_causes, suggestion)

    # 4) claim 级幻觉率高（agent 侧特有信号）
    if side == "agent" and claim_count > 0 and claim_hallu_rate > CLAIM_HALLUCINATION_THRESHOLD:
        category = "claim 级幻觉"
        root_causes = [
            "无支撑 claim 占比超过 %.0f%%（%d 个 claim 中 %d 个无原文支撑）" % (
                CLAIM_HALLUCINATION_THRESHOLD * 100, claim_count, item.get("unsupported_count", 0)),
            "引用编号 [n] 与原文 chunk 不对应（程序化校验缺失）",
        ]
        suggestion = "启用引用编号程序化校验（编号存在且引用句与被引 chunk 文本重叠度达标），强化 claim 级验证拒答。"
        return _pack(category, signals, root_causes, suggestion)

    # 5) 验证器综合分低于通过阈值（agent 侧）
    if side == "agent" and validation_overall is not None and float(validation_overall) < VALIDATOR_PASS_THRESHOLD:
        category = "验证器判幻觉"
        root_causes = [
            "验证器三维综合分 %.2f 低于通过阈值 %.1f" % (float(validation_overall), VALIDATOR_PASS_THRESHOLD),
            "可能为准确性/引用/推理某一维度 fail，需查 validator 原始打分",
        ]
        suggestion = "查看 validator 详细打分（accuracy/citation/reasoning），定位具体维度 fail 原因。"
        return _pack(category, signals, root_causes, suggestion)

    # 6) Agent 拒答（验证失败后拒答）
    if side == "agent" and abstained:
        category = "拒答"
        root_causes = [
            "验证失败且重检索后仍不通过，触发拒答流程",
            "上游召回不足导致无法给出有支撑的答案",
        ]
        suggestion = "回溯该 query 的召回结果，确认 references 是否充足；检查 MIN_RELEVANCE_COUNT 拒答阈值是否过严。"
        return _pack(category, signals, root_causes, suggestion)

    # 7) 验证器判幻觉但 judge 判正确 -> 验证器三维打分口径偏严
    is_hallu = item.get("is_hallucination")
    if is_hallu is True:
        category = "验证器口径偏严"
        root_causes = [
            "验证器三维综合分或 claim 级判定将正确答案误判为幻觉（coverage/faithfulness 均为 pass）",
            "验证器幻觉阈值 %.1f / claim 阈值 %.1f 可能偏严，导致 PASS 答案被误杀" % (
                VALIDATOR_PASS_THRESHOLD, CLAIM_HALLUCINATION_THRESHOLD),
        ]
        suggestion = "核查该条 validator 详细打分（accuracy/citation/reasoning），若 judge 与检索原文均支持答案，可考虑放宽幻觉判定阈值。"
        return _pack(category, signals, root_causes, suggestion)

    # 8) coverage=pass & faithfulness=pass 且非幻觉，但 judge 仍判错 -> judge 误判或口径错位
    category = "judge 误判或口径错位"
    root_causes = [
        "expected_answer 不完整：评测集标注遗漏了知识库有效要点（需人工校准）",
        "judge 模型与被测模型判定口径错位（judge 未结合检索原文充分判定）",
        "学生答案覆盖参考答案要点且忠于原文，但 judge 因措辞差异误判",
    ]
    suggestion = "人工复核 expected_answer 是否完整，检查 judge prompt 是否注入检索原文作为事实依据，必要时校准该条 expected_answer。"
    return _pack(category, signals, root_causes, suggestion)


def _pack(category: str, signals: Dict[str, Any], root_causes: List[str], suggestion: str) -> Dict[str, Any]:
    """组装单条 badcase 的归因结果"""
    return {
        "category": category,
        "signals": signals,
        "root_causes": root_causes,
        "suggestion": suggestion,
    }


def diagnose_accuracy_side(report: Dict[str, Any], side: str) -> List[Dict[str, Any]]:
    """筛 accuracy 报告某一侧（baseline/agent）的所有 badcase 并归因

    Args:
        report: accuracy_eval_report.json 解析出的 dict
        side: "baseline" 或 "agent"

    Returns:
        badcase 诊断列表，每条含 source/id/query/expected_answer/actual_answer + 归因字段
    """
    side_data = report.get(side, {})
    per_query: List[Dict[str, Any]] = side_data.get("per_query", [])
    results: List[Dict[str, Any]] = []
    for item in per_query:
        # badcase 判定：judge 判错 或 标记为幻觉（含 skipped 视为无法判定但保留）
        is_hallu = item.get("is_hallucination")
        if item.get("judge_correct") is True and is_hallu is False:
            continue
        diagnosis = diagnose_accuracy_badcase(item, side)
        results.append({
            "source": "accuracy_%s" % side,
            "id": item.get("id", ""),
            "query": item.get("query", ""),
            "expected_answer": item.get("expected_answer", ""),
            "actual_answer": item.get("actual_answer", ""),
            "error": item.get("error", ""),
            **diagnosis,
        })
    return results


# ========== cross_modal 报告归因 ==========

def diagnose_cross_modal_badcase(item: Dict[str, Any]) -> Dict[str, Any]:
    """对单条 cross_modal badcase（hit=False 或 precise_hit=False）按决策树归因

    Args:
        item: cross_modal.per_query 单条记录

    Returns:
        {"category": str, "signals": dict, "root_causes": list, "suggestion": str}
    """
    cm = item.get("cross_modal", {})
    retrieved_types = cm.get("retrieved_types", [])
    loose_hit = cm.get("hit", False)
    precise_hit = cm.get("precise_hit", False)
    expected_modalities = item.get("expected_modalities", [])
    expected_media_ref = item.get("expected_media_ref", "")

    signals: Dict[str, Any] = {
        "expected_modalities": expected_modalities,
        "expected_media_ref": expected_media_ref,
        "retrieved_types": retrieved_types,
        "loose_hit": loose_hit,
        "precise_hit": precise_hit,
        "modality_quota": item.get("modality_quota"),
    }

    # 检索结果中实际命中的模态集合
    retrieved_modal_set = set(retrieved_types)
    expected_modal_set = set(expected_modalities)
    missing_modalities = expected_modal_set - retrieved_modal_set

    root_causes: List[str] = []
    suggestion = ""

    if not loose_hit:
        # 1) 宽松口径都未命中：目标模态根本没进入 top-k
        category = "跨模态召回完全失败"
        root_causes = [
            "Top-K 模态集合未覆盖期望模态：%s（缺失模态：%s）" % (
                retrieved_modal_set, missing_modalities),
            "目标媒体的多模态 embedding 与查询向量空间未对齐",
            "BM25 关键词检索无法匹配图片/视频（媒体无纯文本关键词命中）",
        ]
        if "image" in missing_modalities:
            root_causes.append("图片语义描述未生成或多样化问题数量不足")
        if "video" in missing_modalities:
            root_causes.append("视频语义描述需用 qwen-vl-plus 生成多角度描述（场景+画面文字OCR+关键实体+适用问题），避免硬编码简短文本")
        suggestion = "为目标媒体补充多角度语义描述，扩充 diverse_question 数量，确认多模态 embedding 模型对齐。"
    else:
        # 2) 宽松口径命中但精确口径未命中：模态配额保送，真实排序差
        category = "模态配额保送（真实排序差）"
        root_causes = [
            "配额机制将目标模态条目塞进 top-k，但目标媒体条目本身向量排名低",
            "precise_hit=False：Top-K 中无 path/url == expected_media_ref 的条目",
            "目标媒体语义描述不够强，导致向量相似度低于其他同模态条目",
        ]
        if "video" in expected_modalities:
            root_causes.append("视频描述未做增强（仅硬编码简短文本）会导致向量排名过低，无法进入 RRF 候选池")
            suggestion = "用 qwen-vl-plus 为视频生成多角度语义描述（场景+OCR+关键实体+适用问题），扩充 diverse_question。"
        elif "image" in expected_modalities:
            suggestion = "为该图片补充更精准的语义描述与多样化问题，提升其在向量空间的区分度。"
        else:
            suggestion = "增强目标媒体的语义描述与多样化问题数量，提升其向量相似度排名。"

    return {
        "category": category,
        "signals": signals,
        "root_causes": root_causes,
        "suggestion": suggestion,
    }


def diagnose_cross_modal(report: Dict[str, Any]) -> List[Dict[str, Any]]:
    """筛 cross_modal 报告的所有 badcase（precise_hit=False 或 hit=False）并归因

    Args:
        report: cross_modal_recall_report.json 解析出的 dict

    Returns:
        badcase 诊断列表
    """
    cm_data = report.get("cross_modal", {})
    per_query: List[Dict[str, Any]] = cm_data.get("per_query", [])
    results: List[Dict[str, Any]] = []
    for item in per_query:
        cm = item.get("cross_modal", {})
        loose_hit = cm.get("hit", False)
        precise_hit = cm.get("precise_hit", False)
        # badcase 判定：精确口径未命中 或 宽松口径未命中
        if precise_hit and loose_hit:
            continue
        diagnosis = diagnose_cross_modal_badcase(item)
        results.append({
            "source": "cross_modal",
            "id": item.get("id", ""),
            "query": item.get("query", ""),
            **diagnosis,
        })
    return results


# ========== 汇总与报告生成 ==========

def build_summary(
    accuracy_report: Optional[Dict],
    cross_modal_report: Optional[Dict],
    baseline_badcases: List[Dict],
    agent_badcases: List[Dict],
    cross_modal_badcases: List[Dict],
) -> Dict[str, Any]:
    """汇总所有 badcase 诊断结果为总报告"""
    all_badcases = baseline_badcases + agent_badcases + cross_modal_badcases
    by_category = dict(Counter(b["category"] for b in all_badcases))
    by_source = dict(Counter(b["source"] for b in all_badcases))

    # accuracy 报告整体指标摘要
    accuracy_summary = {}
    if accuracy_report:
        baseline = accuracy_report.get("baseline", {})
        agent = accuracy_report.get("agent", {})
        accuracy_summary = {
            "baseline_accuracy": baseline.get("first_answer_accuracy"),
            "agent_accuracy": agent.get("first_answer_accuracy"),
            "baseline_hallucination_rate": baseline.get("hallucination_rate"),
            "agent_hallucination_rate": agent.get("hallucination_rate"),
            "total_queries": accuracy_report.get("metadata", {}).get("total_queries"),
        }

    # cross_modal 报告整体指标摘要
    cross_modal_summary = {}
    if cross_modal_report:
        cm = cross_modal_report.get("cross_modal", {})
        cross_modal_summary = {
            "recall@5_loose": cm.get("recall@5_loose"),
            "recall@5_strict": cm.get("recall@5_strict"),
            "total_queries": cross_modal_report.get("metadata", {}).get("total_queries"),
        }

    return {
        "metadata": {
            "diagnosed_at": datetime.now().isoformat(),
            "accuracy_report_path": ACCURACY_REPORT_FILE,
            "cross_modal_report_path": CROSS_MODAL_REPORT_FILE,
            "accuracy_report_found": accuracy_report is not None,
            "cross_modal_report_found": cross_modal_report is not None,
            "total_badcases": len(all_badcases),
            "by_category": by_category,
            "by_source": by_source,
        },
        "accuracy_summary": accuracy_summary,
        "cross_modal_summary": cross_modal_summary,
        "badcases": all_badcases,
    }


def _build_markdown(summary: Dict[str, Any]) -> str:
    """生成人类可读的 Markdown 诊断报告"""
    meta = summary.get("metadata", {})
    lines: List[str] = []

    # ===== 文档头 =====
    lines.append("# Badcase 定位与归因诊断报告")
    lines.append("")
    lines.append("| 项目 | 值 |")
    lines.append("|------|-----|")
    lines.append("| 诊断时间 | %s |" % meta.get("diagnosed_at", "N/A"))
    lines.append("| Badcase 总数 | %s 条 |" % meta.get("total_badcases", 0))
    lines.append("| accuracy 报告 | %s |" % ("已加载" if meta.get("accuracy_report_found") else "未找到"))
    lines.append("| cross_modal 报告 | %s |" % ("已加载" if meta.get("cross_modal_report_found") else "未找到"))
    lines.append("")

    # 整体指标摘要
    acc = summary.get("accuracy_summary", {})
    if acc:
        lines.append("## 整体指标摘要")
        lines.append("")
        lines.append("| 指标 | 基线 RAG | Agent |")
        lines.append("|------|---------|-------|")
        lines.append("| 首答准确率 | %.2f%% | %.2f%% |" % (
            (acc.get("baseline_accuracy") or 0) * 100,
            (acc.get("agent_accuracy") or 0) * 100))
        lines.append("| 幻觉率 | %.2f%% | %.2f%% |" % (
            (acc.get("baseline_hallucination_rate") or 0) * 100,
            (acc.get("agent_hallucination_rate") or 0) * 100))
        lines.append("")

    # 按类别统计
    by_cat = meta.get("by_category", {})
    if by_cat:
        lines.append("## Badcase 类别分布")
        lines.append("")
        lines.append("| 归因类别 | 数量 |")
        lines.append("|---------|------|")
        for cat, cnt in sorted(by_cat.items(), key=lambda x: -x[1]):
            lines.append("| %s | %s |" % (cat, cnt))
        lines.append("")

    # 按来源统计
    by_src = meta.get("by_source", {})
    if by_src:
        lines.append("## Badcase 来源分布")
        lines.append("")
        lines.append("| 来源 | 数量 |")
        lines.append("|------|------|")
        for src, cnt in sorted(by_src.items(), key=lambda x: -x[1]):
            lines.append("| %s | %s |" % (src, cnt))
        lines.append("")

    # 按 category 分组列出 badcase 详情
    badcases = summary.get("badcases", [])
    if not badcases:
        lines.append("## 详细诊断")
        lines.append("")
        lines.append("> 未发现 badcase，所有评测样本均通过。")
        lines.append("")
    else:
        # 按 category 分组
        cat_groups: Dict[str, List[Dict[str, Any]]] = {}
        for b in badcases:
            cat_groups.setdefault(b["category"], []).append(b)

        lines.append("## 详细诊断（按归因类别分组）")
        lines.append("")
        for cat, items in cat_groups.items():
            lines.append("### %s（%s 条）" % (cat, len(items)))
            lines.append("")
            for b in items:
                lines.append("#### [%s] %s" % (b.get("source", ""), b.get("id", "")))
                lines.append("")
                lines.append("- **Query**: %s" % b.get("query", "")[:120])
                if b.get("error"):
                    lines.append("- **错误**: %s" % b.get("error", ""))
                lines.append("- **关键信号**:")
                signals = b.get("signals", {})
                for k, v in signals.items():
                    lines.append("  - `%s`: %s" % (k, v))
                lines.append("- **根因候选**:")
                for rc in b.get("root_causes", []):
                    lines.append("  - %s" % rc)
                lines.append("- **修复建议**: %s" % b.get("suggestion", ""))
                lines.append("")

    lines.append("---")
    lines.append("*本报告由 experiments/badcase_diagnose.py 自动生成*")
    lines.append("")
    return "\n".join(lines)


def _print_console_summary(summary: Dict[str, Any]) -> None:
    """打印可读的汇总到控制台"""
    meta = summary.get("metadata", {})
    print("\n" + "=" * 70)
    print("  Badcase 定位与归因诊断完成")
    print("=" * 70)
    print("  诊断时间:        %s" % meta.get("diagnosed_at"))
    print("  Badcase 总数:    %s 条" % meta.get("total_badcases", 0))
    print("  accuracy 报告:   %s" % ("已加载" if meta.get("accuracy_report_found") else "未找到"))
    print("  cross_modal 报告:%s" % ("已加载" if meta.get("cross_modal_report_found") else "未找到"))

    by_cat = meta.get("by_category", {})
    if by_cat:
        print("\n  [按类别分布]")
        for cat, cnt in sorted(by_cat.items(), key=lambda x: -x[1]):
            print("    - %s: %s 条" % (cat, cnt))

    by_src = meta.get("by_source", {})
    if by_src:
        print("\n  [按来源分布]")
        for src, cnt in sorted(by_src.items(), key=lambda x: -x[1]):
            print("    - %s: %s 条" % (src, cnt))

    print("\n  诊断报告(JSON): %s" % DIAGNOSE_REPORT_FILE)
    print("  诊断报告(MD):   %s" % DIAGNOSE_MD_FILE)
    print("=" * 70)


def main():
    print("=" * 70)
    print("  Badcase 定位与归因诊断")
    print("=" * 70)

    os.makedirs(STATS_DIR, exist_ok=True)

    # 加载已生成的评测报告
    accuracy_report = _load_json(ACCURACY_REPORT_FILE)
    cross_modal_report = _load_json(CROSS_MODAL_REPORT_FILE)

    if accuracy_report is None and cross_modal_report is None:
        print("  错误：未找到任何评测报告，请先运行 python experiments/run_eval.py")
        print("    期望路径：")
        print("      %s" % ACCURACY_REPORT_FILE)
        print("      %s" % CROSS_MODAL_REPORT_FILE)
        return

    # accuracy 报告归因
    baseline_badcases: List[Dict[str, Any]] = []
    agent_badcases: List[Dict[str, Any]] = []
    if accuracy_report:
        baseline_badcases = diagnose_accuracy_side(accuracy_report, "baseline")
        agent_badcases = diagnose_accuracy_side(accuracy_report, "agent")
        logger.info("accuracy 报告归因完成：baseline badcase %s 条，agent badcase %s 条",
                    len(baseline_badcases), len(agent_badcases))
    else:
        logger.warning("accuracy 报告未找到，跳过准确率归因")

    # cross_modal 报告归因
    cross_modal_badcases: List[Dict[str, Any]] = []
    if cross_modal_report:
        cross_modal_badcases = diagnose_cross_modal(cross_modal_report)
        logger.info("cross_modal 报告归因完成：badcase %s 条", len(cross_modal_badcases))
    else:
        logger.warning("cross_modal 报告未找到，跳过跨模态归因")

    # 汇总并保存
    summary = build_summary(
        accuracy_report, cross_modal_report,
        baseline_badcases, agent_badcases, cross_modal_badcases,
    )

    with open(DIAGNOSE_REPORT_FILE, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logger.info("已保存诊断报告(JSON) -> %s", DIAGNOSE_REPORT_FILE)

    markdown_content = _build_markdown(summary)
    with open(DIAGNOSE_MD_FILE, "w", encoding="utf-8") as f:
        f.write(markdown_content)
    logger.info("已保存诊断报告(Markdown) -> %s", DIAGNOSE_MD_FILE)

    _print_console_summary(summary)


if __name__ == "__main__":
    main()
