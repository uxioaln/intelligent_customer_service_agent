# -*- coding: utf-8 -*-
"""
app.agent.validator - 答案验证器（离散分档+多次采样投票 + claim级幻觉判定）

在 Agent 生成答案后，用 LLM 扮演"阅卷老师"，对答案进行质量评分，检测幻觉和引用造假。

验证体系（双重判定）：

1. 维度打分（离散分档 + 多次采样投票）
   - 三个维度：准确性(accuracy)、引用质量(citation)、推理链(reasoning)
   - 每个维度离散分档：pass(2) / marginal(1) / fail(0)
   - 采样 N 次（temperature=0.3），取众数作为最终判定
   - overall = 三维得分均值（0.0~2.0 归一化为 0.0~1.0）
   - passed = overall >= PASS_THRESHOLD

2. claim 级幻觉判定（FActScore 式）
   - 将答案拆成原子事实（claim）
   - 逐条与检索原文做蕴含判断（supported / unsupported）
   - 幻觉率 = 无支撑 claim 数 / 总 claim 数

输出格式：
{
  "accuracy": {"grade": "pass", "score": 1.0, "reason": "..."},
  "citation": {"grade": "pass", "score": 1.0, "reason": "..."},
  "reasoning": {"grade": "pass", "score": 1.0, "reason": "..."},
  "overall": 0.0,
  "passed": true/false,
  "vote_samples": 3,
  "claims": [{"text": "...", "supported": true, "reason": "..."}, ...],
  "claim_count": 5,
  "unsupported_count": 1,
  "claim_hallucination_rate": 0.2,
  "claim_level_hallucination": false
}

设计原则：
- 离散分档比连续打分更稳定（G-Eval, EMNLP 2023）
- 多次采样投票降低单次 LLM 判定方差
- claim 级判定提供可解释的逐条幻觉分析
- LLM 调用走 cost_tracker，纳入成本统计
- 解析失败时返回默认零分，不阻断 Agent 主流程
"""
import json
import logging
import re
from collections import Counter
from typing import Any, Dict, List, Optional

from app.config import VALIDATOR_MODEL, VALIDATOR_SAMPLE_COUNT, VALIDATOR_SAMPLE_TEMPERATURE
from app.core import query
from app.core.cost_tracker import CostTracker, tracked_chat_completion

logger = logging.getLogger(__name__)

# 通过阈值：三维归一化平均分 >= 0.6 视为通过
PASS_THRESHOLD = 0.6
# claim 级幻觉率阈值：超过此比例的 claim 无支撑则判定为幻觉
CLAIM_HALLUCINATION_THRESHOLD = 0.3

# 分层判定开关：维度打分全部为 pass 时跳过 claim 级判定以降低成本
# 依据：v5 评测中 Agent 首答准确率 100%，大多数答案本身质量高，无需再做 claim 级细检
# 设为 False 可回到旧行为（每条都跑 claim 级判定）
ENABLE_TIERED_VALIDATION = True
# 跳过 claim 判定的最小采样数：vote_samples 必须达到此值，否则仍跑 claim 级判定
TIERED_SKIP_MIN_VOTE_SAMPLES = 2

# 离散分档到数值的映射
GRADE_TO_SCORE = {"pass": 1.0, "marginal": 0.5, "fail": 0.0}
SCORE_TO_GRADE = {1.0: "pass", 0.5: "marginal", 0.0: "fail"}


# ========== 维度打分 Prompt（离散分档） ==========

VALIDATOR_SYSTEM_PROMPT = """你是一位严格的阅卷老师。请根据检索到的原文，对生成的答案进行三维离散打分。

## 输入说明

你会收到两个部分：
- [检索原文]：知识库中检索到的原始文本片段，是事实依据。
- [生成答案]：AI 基于检索原文生成的回答，是需要被打分评估的对象。

## 评分维度（每维离散三档：pass / marginal / fail）

1. 准确性 (accuracy)：答案的内容是否忠实于检索原文？
   - pass：完全忠于原文，无任何编造或歪曲。
   - marginal：部分准确，存在少量不严谨的表述但无重大错误。
   - fail：答案与原文矛盾，或包含原文中不存在的信息（幻觉）。

2. 引用质量 (citation)：答案中引用的来源、数据、规则是否真实存在于原文中？
   - pass：所有引用均可在原文中找到对应内容。
   - marginal：部分引用可以找到，部分无法核实。
   - fail：引用了原文中完全不存在的内容（伪造引用）。

3. 推理链 (reasoning)：答案的每一步推理是否有原文证据支撑？
   - pass：每个结论都有对应的原文证据，逻辑链完整。
   - marginal：部分推理有支撑，部分跳跃或缺乏依据。
   - fail：推理过程与原文无关，或存在明显逻辑错误。

## 输出格式（必须是合法 JSON，不要输出其他内容）

{
  "accuracy": {"grade": "pass", "reason": "打分理由，一句话"},
  "citation": {"grade": "pass", "reason": "打分理由，一句话"},
  "reasoning": {"grade": "pass", "reason": "打分理由，一句话"}
}

## 规则

1. 严格基于[检索原文]评分，不要使用原文以外的知识。
2. 如果答案中出现了原文没有的事实、数据或规则，准确性应为 fail。
3. 如果答案声称"根据xxx规定"但原文中找不到该规定，引用质量应为 fail。
4. 输出必须是合法 JSON，不要包裹在 markdown 代码块中。

## 示例

[检索原文]
背景知识1: 门票退款需在购票后7天内申请，需提供购票凭证。特殊票种（年卡）不支持退款。

[生成答案]
门票可以在30天内随时退款，无需任何凭证。所有票种均支持退款。

输出:
{"accuracy": {"grade": "fail", "reason": "答案称30天内可退款且无需凭证，与原文7天内需凭证矛盾"}, "citation": {"grade": "fail", "reason": "答案声称所有票种支持退款，但原文明确年卡不支持"}, "reasoning": {"grade": "fail", "reason": "推理与原文事实完全相悖，无证据支撑"}}

[检索原文]
背景知识1: 上海迪士尼乐园酒店提供免费班车接送，每30分钟一班，运营时间为早7点至晚10点。

[生成答案]
上海迪士尼乐园酒店提供免费班车服务，每30分钟一班，运营时间为早7点至晚10点。

输出:
{"accuracy": {"grade": "pass", "reason": "答案完全忠实于原文，未添加任何额外信息"}, "citation": {"grade": "pass", "reason": "班车频率、运营时间均在原文中可找到"}, "reasoning": {"grade": "pass", "reason": "陈述直接来自原文，无推理跳跃"}}
"""


# ========== claim 级幻觉判定 Prompt ==========

CLAIM_EXTRACTION_AND_ENTAILMENT_PROMPT = """你是一位事实核查专家。请将[生成答案]拆解为原子事实（claim），并逐条判断每个 claim 是否被[检索原文]所支撑。

## 定义

- 原子事实（claim）：答案中不可再分的单一事实陈述，例如"门票退款需在7天内申请"是一个 claim。
- 支撑（supported）：该 claim 的内容可以在检索原文中找到直接或间接的证据。
- 无支撑（unsupported）：该 claim 在检索原文中找不到任何证据，或与原文矛盾。

## 输出格式（必须是合法 JSON）

{
  "claims": [
    {"text": "claim 文本", "supported": true, "reason": "一句话说明判断依据"},
    {"text": "claim 文本", "supported": false, "reason": "一句话说明判断依据"}
  ]
}

## 规则

1. 把答案拆成尽可能多的原子事实，每个 claim 只包含一个事实点。
2. 严格基于[检索原文]判断，不要使用原文以外的知识。
3. 如果 claim 中的数据、规则、事实在原文中找不到，标记为 unsupported。
4. 输出必须是合法 JSON，不要包裹在 markdown 代码块中。

## 示例

[检索原文]
背景知识1: 门票退款需在购票后7天内申请，需提供购票凭证。特殊票种（年卡）不支持退款。

[生成答案]
门票可以在30天内随时退款，无需任何凭证。所有票种均支持退款。

输出:
{"claims": [{"text": "门票可以在30天内退款", "supported": false, "reason": "原文说7天内，答案说30天"}, {"text": "退款无需凭证", "supported": false, "reason": "原文明确要求需提供购票凭证"}, {"text": "所有票种均支持退款", "supported": false, "reason": "原文明确年卡不支持退款"}]}
"""


# ========== 核心函数 ==========

def validate_answer(
    references: List[Dict[str, Any]],
    answer: str,
    tracker: Optional[CostTracker] = None,
) -> Dict[str, Any]:
    """对 Agent 生成的答案进行双重验证（维度打分+claim级判定）

    分层判定（ENABLE_TIERED_VALIDATION=True 时生效）：
    - 第一层：离散分档 + 多次采样投票（3 次调用）
    - 第二层：claim 级蕴含判定（1 次调用）
    - 若第一层三维众数全部为 pass 且采样数充足，跳过第二层以降低成本
    - 第二层失败（解析异常等）不影响第一层结果，claim_count=0 时上层按维度打分判定

    Args:
        references: 检索到的原文片段列表，每项应含 content/source 字段
        answer: Agent 生成的答案文本
        tracker: 可选的 CostTracker 实例

    Returns:
        验证结果 dict，格式见模块文档。
        解析失败时返回默认零分结果。
        claim_skipped=True 表示第二层被跳过（高置信度路径）。
    """
    # 构建检索原文文本
    context_str = _build_context_str(references)

    # ===== 1. 维度打分：离散分档 + 多次采样投票 =====
    dimension_result = _vote_on_dimensions(context_str, answer, tracker)

    # ===== 2. claim 级幻觉判定（分层判定：高置信度跳过） =====
    if ENABLE_TIERED_VALIDATION and _should_skip_claim_check(dimension_result):
        logger.info(
            "分层判定：三维全 pass（accuracy=%s, citation=%s, reasoning=%s, overall=%.2f, vote_samples=%s），"
            "跳过 claim 级判定以降成本",
            dimension_result["accuracy"]["grade"],
            dimension_result["citation"]["grade"],
            dimension_result["reasoning"]["grade"],
            dimension_result["overall"],
            dimension_result.get("vote_samples", 0),
        )
        claim_result = _skipped_claim_result("三维全 pass 高置信度路径，跳过 claim 级判定")
    else:
        claim_result = _evaluate_claims(context_str, answer, tracker)

    # ===== 合并结果 =====
    result = {**dimension_result, **claim_result}
    return result


def _should_skip_claim_check(dimension_result: Dict[str, Any]) -> bool:
    """判断是否可跳过 claim 级判定

    跳过条件（同时满足）：
    - 三个维度众数 grade 均为 "pass"（无 marginal / fail）
    - vote_samples 达到 TIERED_SKIP_MIN_VOTE_SAMPLES（采样数可信）
    """
    if dimension_result.get("vote_samples", 0) < TIERED_SKIP_MIN_VOTE_SAMPLES:
        return False
    for dim in ("accuracy", "citation", "reasoning"):
        grade = dimension_result.get(dim, {}).get("grade")
        if grade != "pass":
            return False
    return True


def _skipped_claim_result(reason: str) -> Dict[str, Any]:
    """分层判定跳过 claim 级判定时的占位结果

    与 _default_claim_failed_result 的差别：
    - claim_skipped=True 显式标记跳过（上层可据此统计节省的调用次数）
    - 不算错误（没有 claim_error 字段）
    """
    return {
        "claims": [],
        "claim_count": 0,
        "unsupported_count": 0,
        "claim_hallucination_rate": 0.0,
        "claim_level_hallucination": False,
        "claim_skipped": True,
        "claim_skipped_reason": reason,
    }


def _build_context_str(references: List[Dict[str, Any]]) -> str:
    """从 references 列表构建检索原文文本"""
    context_str = ""
    for i, ref in enumerate(references, 1):
        source = ref.get("source", "未知来源")
        content = ref.get("content", "")
        context_str += f"背景知识{i} (来源: {source}):\n{content}\n\n"
    if not context_str.strip():
        context_str = "(无检索结果)"
    return context_str


def _vote_on_dimensions(
    context_str: str,
    answer: str,
    tracker: Optional[CostTracker],
) -> Dict[str, Any]:
    """多次采样维度打分并取众数投票

    Args:
        context_str: 检索原文文本
        answer: 答案文本
        tracker: 成本追踪器

    Returns:
        {"accuracy": {"grade":..., "score":..., "reason":...},
         "citation": {...}, "reasoning": {...},
         "overall": 0.0, "passed": bool, "vote_samples": N}
    """
    user_content = f"[检索原文]\n{context_str}\n[生成答案]\n{answer}"
    messages = [
        {"role": "system", "content": VALIDATOR_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    # 收集 N 次采样结果
    samples: List[Dict[str, Any]] = []
    for sample_idx in range(VALIDATOR_SAMPLE_COUNT):
        try:
            response = tracked_chat_completion(
                client=query.client,
                model=VALIDATOR_MODEL,
                messages=messages,
                tracker=tracker,
                source="答案验证-维度打分",
                temperature=VALIDATOR_SAMPLE_TEMPERATURE,
            )
            raw = response.choices[0].message.content.strip()
            parsed = _parse_json_safely(raw)
            if parsed is not None:
                samples.append(parsed)
            else:
                logger.warning("维度打分第 %s 次采样 JSON 解析失败", sample_idx + 1)
        except Exception:
            logger.warning("维度打分第 %s 次采样异常", sample_idx + 1, exc_info=True)

    if not samples:
        logger.warning("维度打分全部采样失败，返回默认零分")
        return _default_dimension_failed_result("全部采样解析失败")

    # 对每个维度取众数
    result: Dict[str, Any] = {"vote_samples": len(samples)}
    for dim in ("accuracy", "citation", "reasoning"):
        grades = []
        reasons = []
        for s in samples:
            dim_data = s.get(dim, {})
            grade = dim_data.get("grade", "fail")
            if grade not in GRADE_TO_SCORE:
                grade = "fail"
            grades.append(grade)
            reasons.append(dim_data.get("reason", "无说明"))

        # 取众数
        grade_counter = Counter(grades)
        majority_grade = grade_counter.most_common(1)[0][0]
        score = GRADE_TO_SCORE[majority_grade]
        # 取与众数 grade 对应的第一个 reason
        majority_reason = reasons[grades.index(majority_grade)]

        result[dim] = {"grade": majority_grade, "score": score, "reason": majority_reason}

    # 计算 overall
    scores = [result[d]["score"] for d in ("accuracy", "citation", "reasoning")]
    overall = round(sum(scores) / len(scores), 2)
    result["overall"] = overall
    result["passed"] = overall >= PASS_THRESHOLD

    logger.info("维度打分投票结果(%s次采样): accuracy=%s, citation=%s, reasoning=%s, overall=%.2f",
                len(samples),
                result["accuracy"]["grade"],
                result["citation"]["grade"],
                result["reasoning"]["grade"],
                overall)

    return result


def _evaluate_claims(
    context_str: str,
    answer: str,
    tracker: Optional[CostTracker],
) -> Dict[str, Any]:
    """claim 级幻觉判定：拆解答案为原子事实，逐条与原文做蕴含判断

    Args:
        context_str: 检索原文文本
        answer: 答案文本
        tracker: 成本追踪器

    Returns:
        {"claims": [...], "claim_count": N, "unsupported_count": M,
         "claim_hallucination_rate": 0.0, "claim_level_hallucination": bool}
    """
    user_content = f"[检索原文]\n{context_str}\n[生成答案]\n{answer}"
    messages = [
        {"role": "system", "content": CLAIM_EXTRACTION_AND_ENTAILMENT_PROMPT},
        {"role": "user", "content": user_content},
    ]

    try:
        response = tracked_chat_completion(
            client=query.client,
            model=VALIDATOR_MODEL,
            messages=messages,
            tracker=tracker,
            source="答案验证-claim级判定",
            temperature=0.1,
        )
        raw = response.choices[0].message.content.strip()
        logger.info("claim 级判定 LLM 原始输出:\n%s", raw)
    except Exception:
        logger.warning("claim 级判定调用异常", exc_info=True)
        return _default_claim_failed_result("LLM 调用异常")

    parsed = _parse_json_safely(raw)
    if parsed is None:
        logger.warning("claim 级判定 JSON 解析失败。原始输出: %s", raw[:200])
        return _default_claim_failed_result("JSON 解析失败")

    claims_raw = parsed.get("claims", [])
    if not isinstance(claims_raw, list) or not claims_raw:
        logger.warning("claim 级判定: claims 为空或格式异常")
        return _default_claim_failed_result("claims 为空")

    # 规范化 claim 列表
    claims = []
    for c in claims_raw:
        if not isinstance(c, dict):
            continue
        text = c.get("text", "")
        supported = c.get("supported", False)
        reason = c.get("reason", "")
        claims.append({"text": text, "supported": bool(supported), "reason": reason})

    claim_count = len(claims)
    unsupported_count = sum(1 for c in claims if not c["supported"])
    hallu_rate = round(unsupported_count / claim_count, 4) if claim_count else 0.0
    claim_hallucination = hallu_rate > CLAIM_HALLUCINATION_THRESHOLD

    logger.info("claim 级判定: %s 个 claim，%s 个无支撑，幻觉率=%.2f",
                claim_count, unsupported_count, hallu_rate)

    return {
        "claims": claims,
        "claim_count": claim_count,
        "unsupported_count": unsupported_count,
        "claim_hallucination_rate": hallu_rate,
        "claim_level_hallucination": claim_hallucination,
    }


def _default_dimension_failed_result(reason: str) -> Dict[str, Any]:
    """维度打分全部失败时的默认零分结果"""
    return {
        "accuracy": {"grade": "fail", "score": 0.0, "reason": f"验证失败: {reason}"},
        "citation": {"grade": "fail", "score": 0.0, "reason": f"验证失败: {reason}"},
        "reasoning": {"grade": "fail", "score": 0.0, "reason": f"验证失败: {reason}"},
        "overall": 0.0,
        "passed": False,
        "vote_samples": 0,
    }


def _default_claim_failed_result(reason: str) -> Dict[str, Any]:
    """claim 级判定失败时的默认结果"""
    return {
        "claims": [],
        "claim_count": 0,
        "unsupported_count": 0,
        "claim_hallucination_rate": 0.0,
        "claim_level_hallucination": False,
        "claim_error": reason,
    }


def _parse_json_safely(raw: str) -> Any:
    """安全解析 JSON，兼容 LLM 可能包裹 markdown 代码块的情况

    与 planner.py 中的 _parse_json_safely 逻辑一致，避免跨模块依赖。
    """
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

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


def format_validation_summary(result: Dict[str, Any]) -> str:
    """把验证结果格式化为可读字符串（用于日志和前端展示）"""
    lines = [
        f"准确性: {result['accuracy']['grade']} - {result['accuracy']['reason']}",
        f"引用质量: {result['citation']['grade']} - {result['citation']['reason']}",
        f"推理链: {result['reasoning']['grade']} - {result['reasoning']['reason']}",
        f"综合: {result['overall']} | {'通过' if result['passed'] else '不通过'} | 采样{result.get('vote_samples', 0)}次",
    ]
    # 分层判定：claim 被跳过时显式标注
    if result.get("claim_skipped"):
        lines.append(f"Claim 级: 已跳过（{result.get('claim_skipped_reason', '高置信度路径')}）")
        return "\n".join(lines)
    claim_count = result.get("claim_count", 0)
    if claim_count:
        lines.append(
            f"Claim 级: {claim_count} 个 claim，{result.get('unsupported_count', 0)} 个无支撑，"
            f"幻觉率={result.get('claim_hallucination_rate', 0.0):.2f} | "
            f"{'幻觉' if result.get('claim_level_hallucination') else '通过'}"
        )
    return "\n".join(lines)
