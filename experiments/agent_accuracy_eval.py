# -*- coding: utf-8 -*-
"""
experiments.agent_accuracy_eval - 首答准确率 + 幻觉率评测脚本

评测口径（与 README 一致）：
- 首答准确率：对 50 条复杂查询，用 LLM-as-judge 判定答案要点是否与 expected_answer 一致
  - 基线：rag_ask_api（固定检索链 RAG，无规划/无验证）
  - Agent：agent_chat（ReAct 规划 + 5 工具 + 三维验证自纠错）
- 幻觉率：验证器综合分 overall < 0.8 判定为幻觉（与 loop.py 的 VALIDATION_THRESHOLD 一致）
  - Agent 侧：直接用 agent_chat 返回的 validation.overall
  - 基线侧：对基线 answer + references 跑 validate_answer（同口径可比）

输出文件：data/stats/accuracy_eval_report.json

使用方法：
    export AGICTO_API_KEY=xxx
    export DASHSCOPE_API_KEY=xxx
    python experiments/build_eval_dataset.py        # 先生成评测集
    python experiments/agent_accuracy_eval.py       # 跑准确率 + 幻觉率评测
"""
import json
import logging
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

# 把工程根加入 path，使 app.* 包可被导入
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from app.config import JUDGE_MODEL, STATS_DIR
from app.core import query
from app.core.cost_tracker import CostTracker, tracked_chat_completion
from app.core.rag_service import rag_ask_api
from app.agent.loop import agent_chat
from app.agent.validator import validate_answer
from app.state import get_metadata, get_qdrant_client, load_resources

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ========== 评测参数 ==========
# 幻觉判定阈值：与 validator.py 的 PASS_THRESHOLD 对齐（0.6）
# 注：loop.py 的 VALIDATION_THRESHOLD=0.8 是触发重检索的阈值，不是判定幻觉的阈值
HALLUCINATION_THRESHOLD = 0.6
LLM_CALL_INTERVAL = 2.0        # LLM 调用间隔（秒），防限流（gpt-4o 调用密集时需更大间隔）
JUDGE_TEMPERATURE = 0.1        # judge 低温度提升稳定性
PROGRESS_LOG_EVERY = 5         # 每 N 条打印一次进度

# ========== 输入/输出文件 ==========
EVAL_DATASET_FILE = os.path.join(STATS_DIR, "eval_dataset.json")
REPORT_FILE = os.path.join(STATS_DIR, "accuracy_eval_report.json")
# 断点续跑中间结果文件：每完成一条查询立即持久化，崩溃后重启可跳过已完成项
PARTIAL_FILE = os.path.join(STATS_DIR, "accuracy_eval_report.partial.json")


def _load_partial() -> Dict[str, Any]:
    """加载断点续跑中间结果，不存在或损坏时返回空结构"""
    if not os.path.exists(PARTIAL_FILE):
        return {"baseline_results": [], "agent_results": [], "completed_ids": []}
    try:
        with open(PARTIAL_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        data.setdefault("baseline_results", [])
        data.setdefault("agent_results", [])
        data.setdefault("completed_ids", [])
        return data
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("partial 文件损坏，从头开始: %s", e)
        return {"baseline_results": [], "agent_results": [], "completed_ids": []}


def _save_partial(partial: Dict[str, Any]) -> None:
    """原子写中间结果（先写临时文件再 rename，避免崩溃时截断 JSON）"""
    os.makedirs(STATS_DIR, exist_ok=True)
    tmp_file = PARTIAL_FILE + ".tmp"
    with open(tmp_file, "w", encoding="utf-8") as f:
        json.dump(partial, f, ensure_ascii=False, indent=2)
    os.replace(tmp_file, PARTIAL_FILE)


# ========== LLM-as-judge ==========

JUDGE_PROMPT = """你是一位严格的阅卷老师。请判断 [学生答案] 是否正确回答了 [用户问题]。

你会收到四个部分：
- [用户问题]：用户提出的问题
- [参考答案]：标准答案，包含核心要点
- [检索原文]：知识库中实际检索到的原文片段，是事实依据
- [学生答案]：被测系统生成的答案

## 分维度打分

### 维度1：要点覆盖（coverage）
学生答案是否覆盖了 [参考答案] 中的核心要点？
- pass：覆盖了参考答案的绝大多数核心要点（允许措辞不同）
- marginal：覆盖了部分核心要点，但遗漏了 1-2 个重要信息
- fail：遗漏了多数核心要点，或答非所问

### 维度2：事实正确性（faithfulness）
学生答案中的事实是否忠于 [检索原文]？
- pass：答案中的事实均可在检索原文中找到支撑，无编造
- marginal：大部分事实有支撑，但有少量不够严谨的表述
- fail：包含与检索原文矛盾的信息，或编造了检索原文中不存在的事实

## 综合判定

correct = coverage_pass AND faithfulness_pass
（两个维度均为 pass 才判正确）

## 判定规则

1. 学生答案只要覆盖参考答案的核心要点即可，多余的补充信息不影响判定
2. 如果学生答案补充的信息在 [检索原文] 中有支撑，即使参考答案未提及，也不应判错
3. 如果学生答案与参考答案矛盾，但忠于检索原文，以检索原文为准
4. 如果学生答案遗漏了参考答案的关键要点，coverage 应为 marginal 或 fail

请返回 JSON 格式（仅返回 JSON，不要包裹 markdown 代码块）：
{
    "coverage": {"grade": "pass", "reason": "一句话说明"},
    "faithfulness": {"grade": "pass", "reason": "一句话说明"},
    "correct": true,
    "reason": "综合判定理由，一句话"
}
"""


def judge_answer(query: str, expected_answer: str, actual_answer: str, tracker: CostTracker,
                 references: List[Dict[str, Any]] = None) -> Dict[str, Any]:
    """用 LLM-as-judge 判定单条答案是否首答正确（分维度打分 + 检索原文注入）

    Args:
        query: 用户问题
        expected_answer: 评测集标注的期望答案
        actual_answer: 被测系统实际生成的答案
        tracker: 成本追踪器
        references: 检索到的原文片段列表，注入 judge prompt 作为事实依据

    Returns:
        {"correct": bool, "coverage": str, "faithfulness": str, "reason": str, "raw": str}
    """
    # 构建检索原文文本
    context_str = ""
    if references:
        for i, ref in enumerate(references, 1):
            source = ref.get("source", "未知来源")
            content = ref.get("content", "")
            context_str += f"原文{i} (来源: {source}):\n{content}\n\n"
    if not context_str.strip():
        context_str = "(无检索原文)"

    user_content = (
        f"[用户问题]\n{query}\n\n"
        f"[参考答案]\n{expected_answer}\n\n"
        f"[检索原文]\n{context_str}\n\n"
        f"[学生答案]\n{actual_answer}"
    )
    response = tracked_chat_completion(
        client=query_module_client(),
        model=JUDGE_MODEL,
        messages=[
            {"role": "system", "content": JUDGE_PROMPT},
            {"role": "user", "content": user_content},
        ],
        tracker=tracker,
        source="准确率评测-judge",
        temperature=JUDGE_TEMPERATURE,
    )
    raw = response.choices[0].message.content.strip()
    # 兼容 markdown 代码块包裹的 JSON
    parsed = _parse_json_safely(raw)
    if parsed is None:
        logger.warning("judge JSON 解析失败，默认判错。原始输出: %s", raw[:200])
        return {"correct": False, "reason": "judge JSON 解析失败", "raw": raw}

    correct = bool(parsed.get("correct", False))
    coverage = parsed.get("coverage", {}).get("grade", "unknown")
    faithfulness = parsed.get("faithfulness", {}).get("grade", "unknown")
    reason = parsed.get("reason", "")
    return {"correct": correct, "coverage": coverage, "faithfulness": faithfulness, "reason": reason, "raw": raw}


def query_module_client():
    """返回 app.core.query 中已初始化的 AGICTO OpenAI client"""
    return query.client


def _parse_json_safely(raw: str):
    """安全解析 JSON，兼容 markdown 代码块"""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    import re
    code_block = re.search(r"```(?:json)?\s*(.*?)\s*```", raw, re.DOTALL)
    if code_block:
        try:
            return json.loads(code_block.group(1))
        except json.JSONDecodeError:
            pass
    first_brace = raw.find("{")
    last_brace = raw.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        try:
            return json.loads(raw[first_brace:last_brace + 1])
        except json.JSONDecodeError:
            pass
    return None


# ========== 被测系统调用 ==========

def run_baseline(query_str: str) -> Dict[str, Any]:
    """基线 RAG：固定检索链，无规划/无验证"""
    qdrant_client = get_qdrant_client()
    metadata = get_metadata()
    result = rag_ask_api(query_str, qdrant_client, metadata)
    return {
        "answer": result.get("answer", ""),
        "references": result.get("references", []),
        "image_path": result.get("image_path"),
        "video_url": result.get("video_url"),
    }


def run_agent(query_str: str) -> Dict[str, Any]:
    """Agent：ReAct 规划 + 5 工具 + 三维验证自纠错"""
    result = agent_chat(query_str, session_id="", visitor_id="eval")
    return {
        "answer": result.get("answer", ""),
        "references": result.get("references", []),
        "validation": result.get("validation"),
        "retried": result.get("retried", False),
        "steps": result.get("steps", []),
        "cost": result.get("cost", {}),
    }


# ========== 幻觉判定 ==========

def is_hallucination_for_agent(agent_result: Dict[str, Any]) -> Optional[bool]:
    """Agent 侧幻觉判定：基于 claim 级幻觉率判定

    优先使用验证器的 claim_level_hallucination 布尔值（无支撑 claim 比例 > 30% 即幻觉）。
    当 validation 为 None 时（references 为空，跳过验证），返回 None 表示无法判定。
    """
    validation = agent_result.get("validation")
    if validation is None:
        return None
    # claim 级判定优先
    if validation.get("claim_count", 0) > 0:
        return validation.get("claim_level_hallucination", False)
    # claim 级判定失败时退回维度打分
    overall = validation.get("overall", 0.0)
    try:
        overall = float(overall)
    except (TypeError, ValueError):
        overall = 0.0
    return overall < HALLUCINATION_THRESHOLD


def is_hallucination_for_baseline(baseline_result: Dict[str, Any], tracker: CostTracker) -> Dict[str, Any]:
    """基线侧幻觉判定：对基线 answer + references 跑一遍 validate_answer（与 Agent 同口径）

    返回 {"is_hallucination": bool, "validation": dict}
    """
    references = baseline_result.get("references", [])
    answer = baseline_result.get("answer", "")
    validation = validate_answer(references, answer, tracker=tracker)
    # claim 级判定优先
    if validation.get("claim_count", 0) > 0:
        is_hallu = validation.get("claim_level_hallucination", False)
    else:
        overall = float(validation.get("overall", 0.0))
        is_hallu = overall < HALLUCINATION_THRESHOLD
    return {
        "is_hallucination": is_hallu,
        "validation": validation,
    }


# ========== 主流程 ==========

def load_complex_queries() -> List[Dict[str, Any]]:
    """加载评测集中的复杂查询"""
    if not os.path.exists(EVAL_DATASET_FILE):
        raise FileNotFoundError(
            "评测集不存在: %s，请先运行 python experiments/build_eval_dataset.py" % EVAL_DATASET_FILE
        )
    with open(EVAL_DATASET_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("complex_queries", [])


def _recount_from_results(baseline_judge_results: List[Dict[str, Any]],
                          agent_judge_results: List[Dict[str, Any]]) -> Dict[str, int]:
    """从已累加的结果列表中重算所有计数器（断点恢复时用于初始化）"""
    counts = {
        "baseline_correct_count": 0,
        "baseline_hallucination_count": 0,
        "agent_correct_count": 0,
        "agent_hallucination_count": 0,
        "agent_skip_count": 0,
        "agent_abstain_count": 0,
        "agent_retry_count": 0,
    }
    for r in baseline_judge_results:
        if r.get("judge_correct"):
            counts["baseline_correct_count"] += 1
        if r.get("is_hallucination") is True:
            counts["baseline_hallucination_count"] += 1
    for r in agent_judge_results:
        if r.get("judge_correct"):
            counts["agent_correct_count"] += 1
        hallu = r.get("is_hallucination")
        if hallu == "skipped" or hallu is None:
            counts["agent_skip_count"] += 1
        elif hallu is True:
            counts["agent_hallucination_count"] += 1
        if r.get("retried"):
            counts["agent_retry_count"] += 1
        if r.get("abstained"):
            counts["agent_abstain_count"] += 1
    return counts


def main():
    print("=" * 60)
    print("  首答准确率 + 幻觉率评测")
    print("=" * 60)

    # 初始化 Qdrant 客户端与内存索引（Agent/基线都依赖）
    logger.info("初始化 Qdrant 客户端与内存索引...")
    load_resources()

    queries = load_complex_queries()
    logger.info("加载复杂查询 %s 条", len(queries))
    if not queries:
        print("  评测集为空，退出")
        return

    # ===== 断点续跑：加载已完成的中间结果 =====
    partial = _load_partial()
    completed_ids = set(partial.get("completed_ids", []))
    baseline_judge_results: List[Dict[str, Any]] = partial.get("baseline_results", [])
    agent_judge_results: List[Dict[str, Any]] = partial.get("agent_results", [])
    if completed_ids:
        logger.info("检测到断点续跑文件，已完成 %s 条，将跳过", len(completed_ids))

    # 从 partial 结果中恢复计数器
    counts = _recount_from_results(baseline_judge_results, agent_judge_results)
    baseline_correct_count = counts["baseline_correct_count"]
    baseline_hallucination_count = counts["baseline_hallucination_count"]
    agent_correct_count = counts["agent_correct_count"]
    agent_hallucination_count = counts["agent_hallucination_count"]
    agent_skip_count = counts["agent_skip_count"]
    agent_abstain_count = counts["agent_abstain_count"]
    agent_retry_count = counts["agent_retry_count"]

    # judge 调用走独立 tracker（基线/Agent 各自内部也有成本统计）
    judge_tracker = CostTracker()
    judge_tracker.begin()
    baseline_validation_tracker = CostTracker()
    baseline_validation_tracker.begin()

    for i, q_item in enumerate(queries, 1):
        q_id = q_item.get("id", "")
        q_text = q_item["query"]
        expected = q_item.get("expected_answer", "")

        # 断点续跑：已完成的 query 直接跳过
        if q_id in completed_ids:
            logger.info("[%s/%s] 跳过已完成 id=%s", i, len(queries), q_id)
            continue

        logger.info("[%s/%s] query=%s", i, len(queries), q_text[:50])

        # ----- 基线 RAG -----
        # LLM 调用已在 tracked_chat_completion 内建重试，此处仅捕获最终失败兜底
        try:
            baseline_result = run_baseline(q_text)
            baseline_judge = judge_answer(q_text, expected, baseline_result["answer"], judge_tracker,
                                          references=baseline_result.get("references", []))
            baseline_hallu = is_hallucination_for_baseline(baseline_result, baseline_validation_tracker)
            if baseline_hallu["is_hallucination"]:
                baseline_hallucination_count += 1
            if baseline_judge["correct"]:
                baseline_correct_count += 1
            baseline_judge_results.append({
                "id": q_id,
                "query": q_text,
                "expected_answer": expected,
                "actual_answer": baseline_result["answer"][:300],
                "judge_correct": baseline_judge["correct"],
                "judge_coverage": baseline_judge.get("coverage", "unknown"),
                "judge_faithfulness": baseline_judge.get("faithfulness", "unknown"),
                "judge_reason": baseline_judge["reason"],
                "is_hallucination": baseline_hallu["is_hallucination"],
                "validation_overall": baseline_hallu["validation"].get("overall", 0.0),
                "references_count": len(baseline_result.get("references", [])),
            })
        except Exception as e:
            logger.exception("基线评测失败: %s", q_text)
            baseline_judge_results.append({
                "id": q_id,
                "query": q_text,
                "error": str(e),
                "judge_correct": False,
                "is_hallucination": True,
            })
        time.sleep(LLM_CALL_INTERVAL)

        # ----- Agent -----
        try:
            agent_result = run_agent(q_text)
            agent_judge = judge_answer(q_text, expected, agent_result["answer"], judge_tracker,
                                       references=agent_result.get("references", []))
            agent_hallu = is_hallucination_for_agent(agent_result)
            if agent_hallu is None:
                agent_skip_count += 1
            elif agent_hallu:
                agent_hallucination_count += 1
            if agent_result.get("retried"):
                agent_retry_count += 1
            if agent_result.get("abstained"):
                agent_abstain_count += 1
            if agent_judge["correct"]:
                agent_correct_count += 1
            # 提取 claim 级指标
            val = agent_result.get("validation") or {}
            agent_judge_results.append({
                "id": q_id,
                "query": q_text,
                "expected_answer": expected,
                "actual_answer": agent_result["answer"][:300],
                "judge_correct": agent_judge["correct"],
                "judge_coverage": agent_judge.get("coverage", "unknown"),
                "judge_faithfulness": agent_judge.get("faithfulness", "unknown"),
                "judge_reason": agent_judge["reason"],
                "is_hallucination": agent_hallu if agent_hallu is not None else "skipped",
                "validation_overall": val.get("overall"),
                "claim_count": val.get("claim_count", 0),
                "unsupported_count": val.get("unsupported_count", 0),
                "claim_hallucination_rate": val.get("claim_hallucination_rate", 0.0),
                "retried": agent_result.get("retried", False),
                "abstained": agent_result.get("abstained", False),
                "steps_count": len(agent_result.get("steps", [])),
                "agent_cost": agent_result.get("cost", {}),
            })
        except Exception as e:
            logger.exception("Agent 评测失败: %s", q_text)
            agent_judge_results.append({
                "id": q_id,
                "query": q_text,
                "error": str(e),
                "judge_correct": False,
                "is_hallucination": True,
            })
        time.sleep(LLM_CALL_INTERVAL)

        # ===== 断点续跑：每条完成后立即持久化 =====
        completed_ids.add(q_id)
        _save_partial({
            "baseline_results": baseline_judge_results,
            "agent_results": agent_judge_results,
            "completed_ids": sorted(completed_ids),
        })

        if i % PROGRESS_LOG_EVERY == 0 or i == len(queries):
            logger.info("  进度 %s/%s | 基线准确 %s/幻觉 %s | Agent 准确 %s/幻觉 %s/跳过 %s/拒答 %s/重试 %s",
                        i, len(queries),
                        baseline_correct_count, baseline_hallucination_count,
                        agent_correct_count, agent_hallucination_count,
                        agent_skip_count, agent_abstain_count, agent_retry_count)

    n = len(queries)
    baseline_accuracy = round(baseline_correct_count / n, 4) if n else 0.0
    agent_accuracy = round(agent_correct_count / n, 4) if n else 0.0
    baseline_hallu_rate = round(baseline_hallucination_count / n, 4) if n else 0.0
    # 幻觉率分母排除跳过验证的样本（references 为空，无法判定幻觉）
    agent_evaluated = n - agent_skip_count
    agent_hallu_rate = round(agent_hallucination_count / agent_evaluated, 4) if agent_evaluated else 0.0

    judge_tracker.finish()
    baseline_validation_tracker.finish()
    judge_cost = judge_tracker.get_summary()
    baseline_validation_cost = baseline_validation_tracker.get_summary()

    report = {
        "metadata": {
            "total_queries": n,
            "timestamp": datetime.now().isoformat(),
            "judge_model": JUDGE_MODEL,
            "validator_model": "gpt-4o",
            "hallucination_threshold": HALLUCINATION_THRESHOLD,
            "hallucination_criteria": "claim 级判定：无支撑 claim 比例 > 0.3 即幻觉",
            "judge_criteria": "LLM-as-judge (gpt-4o) 判定答案要点是否与 expected_answer 一致",
        },
        "baseline": {
            "first_answer_accuracy": baseline_accuracy,
            "correct_count": baseline_correct_count,
            "hallucination_rate": baseline_hallu_rate,
            "hallucination_count": baseline_hallucination_count,
            "per_query": baseline_judge_results,
            "validation_cost": {
                "call_count": baseline_validation_cost["call_count"],
                "total_tokens": baseline_validation_cost["total_tokens"],
                "estimated_cost": baseline_validation_cost["estimated_cost"],
            },
        },
        "agent": {
            "first_answer_accuracy": agent_accuracy,
            "correct_count": agent_correct_count,
            "hallucination_rate": agent_hallu_rate,
            "hallucination_count": agent_hallucination_count,
            "skipped_validation_count": agent_skip_count,
            "evaluated_for_hallucination": n - agent_skip_count,
            "abstained_count": agent_abstain_count,
            "retry_triggered_count": agent_retry_count,
            "per_query": agent_judge_results,
        },
        "judge_cost": {
            "call_count": judge_cost["call_count"],
            "total_tokens": judge_cost["total_tokens"],
            "estimated_cost": judge_cost["estimated_cost"],
            "duration_seconds": judge_cost["duration_seconds"],
        },
    }

    os.makedirs(STATS_DIR, exist_ok=True)
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    logger.info("已保存准确率评测报告 -> %s", REPORT_FILE)

    # 评测完整跑通后删除断点续跑 partial 文件，避免下次启动误恢复旧进度
    if os.path.exists(PARTIAL_FILE):
        try:
            os.remove(PARTIAL_FILE)
            logger.info("已清理断点续跑 partial 文件: %s", PARTIAL_FILE)
        except OSError as e:
            logger.warning("清理 partial 文件失败: %s", e)

    print("\n" + "=" * 60)
    print("  首答准确率 + 幻觉率评测完成")
    print("=" * 60)
    print("  复杂查询数量: %s" % n)
    print("  基线 RAG:")
    print("    首答准确率: %.2f%%（%s/%s）" % (baseline_accuracy * 100, baseline_correct_count, n))
    print("    幻觉率:     %.2f%%（%s/%s）" % (baseline_hallu_rate * 100, baseline_hallucination_count, n))
    print("  Agent (ReAct + 验证自纠错):")
    print("    首答准确率: %.2f%%（%s/%s）" % (agent_accuracy * 100, agent_correct_count, n))
    print("    幻觉率:     %.2f%%（%s/%s）" % (agent_hallu_rate * 100, agent_hallucination_count, n - agent_skip_count))
    print("    拒答:       %s 次" % agent_abstain_count)
    print("    触发重检索: %s 次" % agent_retry_count)
    print("  judge 成本: %s 元，%s 次调用，%s tokens" % (
        judge_cost["estimated_cost"], judge_cost["call_count"], judge_cost["total_tokens"]))
    print("  报告文件: %s" % REPORT_FILE)
    print("=" * 60)


if __name__ == "__main__":
    main()
