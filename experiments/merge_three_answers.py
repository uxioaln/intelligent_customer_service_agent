"""
对 text_subset_with_answers.json 中 50 条 × 3 份答案做 LLM 仲裁，输出 merged_answer。

流程：
  1. 加载 A2 产物：50 条 × 3 模型答案 + 检索原文
  2. 对每条用 gpt-4o 做"仲裁者"：传入 3 份答案 + 检索原文，让它输出：
       - merged_answer: 原子要点列表（去重 + 补全 + 标注来源模型）
       - agreement_score: 3 模型一致度 (0-1)
       - conflict_points: 冲突要点列表
       - needs_human_review: bool（high 分歧/低一致度 时为 True）
  3. 输出 text_subset_merged.json
  4. 控制台汇总 needs_human_review 的条目清单

缓存机制：
  - partial.json 支持断点续跑
  - 每条独立 LLM 调用，失败不影响其他条
"""
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from app.config import JUDGE_MODEL, AGICTO_API_KEY, CHAT_BASE_URL
from app.core.cost_tracker import CostTracker, tracked_chat_completion
from openai import OpenAI

# ===== 配置 =====
INPUT_FILE = PROJECT_ROOT / "data/stats/text_subset_with_answers.json"
OUTPUT_FILE = PROJECT_ROOT / "data/stats/text_subset_merged.json"
PARTIAL_FILE = PROJECT_ROOT / "data/stats/text_subset_merged.partial.json"

LLM_CALL_INTERVAL = 0.3          # LLM 调用间隔
RANDOM_SEED = 42
LOW_AGREEMENT_THRESHOLD = 0.6   # agreement_score < 此值标记 needs_human_review

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def make_client() -> OpenAI:
    return OpenAI(api_key=AGICTO_API_KEY, base_url=CHAT_BASE_URL)


ARBITRATION_PROMPT = """你是一名严格的评测集仲裁助手。请对以下 3 个模型生成的答案做仲裁，输出最终 expected_answer。

任务：
  1. 对比 3 份答案，提炼出全部有效要点（去重、合并相似表述）
  2. 每个要点必须能在检索原文中找到支撑（不可编造）
  3. 标注 3 个模型对该要点的覆盖情况（"3模型一致" / "仅X模型提及" 等）
  4. 标记冲突点：3 模型在同一要点上给出冲突答案的，列出来
  5. 输出 agreement_score：3 模型答案要点集的 Jaccard 相似度（0-1）
  6. 若 agreement_score < 0.6 或存在冲突点，标记 needs_human_review=true

[检索原文]
{context}

[3 份模型答案]
--- gpt-4o ---
{answer_gpt4o}

--- deepseek-v4-flash ---
{answer_flash}

--- deepseek-v4-pro ---
{answer_pro}

[用户问题]
{query}

输出 JSON（严格格式）：
{{
    "merged_answer": "原子要点列表（每个要点一行，最多 8 个要点）",
    "agreement_score": 0.0,
    "conflict_points": ["冲突点1", "冲突点2"],
    "needs_human_review": true,
    "coverage_notes": "3模型覆盖情况简要说明"
}}
"""


def call_arbitration(client: OpenAI, prompt: str, tracker: CostTracker) -> str:
    """调用 gpt-4o 仲裁"""
    time.sleep(LLM_CALL_INTERVAL)
    response = tracked_chat_completion(
        client=client,
        model=JUDGE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        tracker=tracker,
        source="评测集扩充-A2仲裁",
        temperature=0.0,
    )
    return response.choices[0].message.content


def parse_arbitration_response(raw: str) -> Dict:
    """解析仲裁响应（容错处理 markdown 包裹/单引号/缺逗号等）"""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.rsplit("```", 1)[0]
    raw = raw.strip()

    # 尝试直接解析
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # 容错：修复常见格式问题
    # 1) 单引号转双引号
    fixed = raw.replace("'", '"')
    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass

    # 2) 缺逗号（键值对之间）- 简化修复
    import re
    fixed2 = re.sub(r'("\w+")\s*:\s*("(?:[^"\\]|\\.)*"(?:\s*,\s*"(?:[^"\\]|\\.)*")*)\s*(?="[a-z_]+":)', r'\1: \2,', raw)
    try:
        return json.loads(fixed2)
    except json.JSONDecodeError:
        pass

    return {"parse_error": True, "raw_response": raw[:500]}


def load_partial() -> Dict:
    if PARTIAL_FILE.exists():
        with open(PARTIAL_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"completed": {}}


def save_partial(data: Dict) -> None:
    with open(PARTIAL_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    # 1. 加载 A2 产物
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        a2_data = json.load(f)
    samples = a2_data["samples"]
    logger.info("加载 A2 产物: %d 条", len(samples))

    # 2. 断点续跑
    partial = load_partial()
    completed = partial.get("completed", {})
    logger.info("断点续跑: 已完成 %d / %d", len(completed), len(samples))

    # 3. 初始化
    client = make_client()
    tracker = CostTracker()

    # 4. 逐条仲裁
    t0 = time.time()
    results = []
    needs_review_list = []

    for i, s in enumerate(samples):
        qid = s["id"]
        if qid in completed:
            logger.info("[%d/%d] %s 命中缓存", i + 1, len(samples), qid)
            results.append(completed[qid])
            if completed[qid].get("arbitration", {}).get("needs_human_review", False):
                needs_review_list.append(completed[qid])
            continue

        logger.info("[%d/%d] %s 仲裁中: %s",
                    i + 1, len(samples), qid, s["query"][:50])

        # 构造 prompt
        answers = s.get("answers", {})
        prompt = ARBITRATION_PROMPT.format(
            context=s.get("context", ""),
            answer_gpt4o=answers.get("gpt-4o", ""),
            answer_flash=answers.get("deepseek-v4-flash", ""),
            answer_pro=answers.get("deepseek-v4-pro", ""),
            query=s["query"],
        )

        # 调用仲裁
        try:
            raw_resp = call_arbitration(client, prompt, tracker)
            arbitration = parse_arbitration_response(raw_resp)
            if arbitration.get("parse_error"):
                logger.warning("  解析失败: %s", arbitration.get("raw_response", "")[:100])
                arbitration = {
                    "merged_answer": answers.get("gpt-4o", ""),
                    "agreement_score": 0.0,
                    "conflict_points": ["仲裁响应解析失败"],
                    "needs_human_review": True,
                    "coverage_notes": "parse_error",
                }
        except Exception as e:
            logger.error("  仲裁失败: %s", str(e)[:120])
            arbitration = {
                "merged_answer": answers.get("gpt-4o", ""),
                "agreement_score": 0.0,
                "conflict_points": [f"仲裁调用失败: {str(e)[:80]}"],
                "needs_human_review": True,
                "coverage_notes": "call_error",
            }

        # 强制 agreement_score < 阈值的标记 needs_human_review
        try:
            score = float(arbitration.get("agreement_score", 0))
        except (TypeError, ValueError):
            score = 0.0
            arbitration["needs_human_review"] = True
        if score < LOW_AGREEMENT_THRESHOLD:
            arbitration["needs_human_review"] = True

        record = {
            "id": qid,
            "query": s["query"],
            "topic": s.get("topic", ""),
            "original_disagreement": s.get("disagreement", {}),
            "arbitration": arbitration,
        }
        results.append(record)
        completed[qid] = record

        if arbitration.get("needs_human_review", False):
            needs_review_list.append(record)

        logger.info("  agreement=%.2f review=%s merged=%s",
                    score,
                    arbitration.get("needs_human_review", False),
                    str(arbitration.get("merged_answer", ""))[:80])

        # 每 5 条落盘
        if (i + 1) % 5 == 0:
            save_partial({"completed": completed})
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (len(samples) - i - 1)
            logger.info("[进度] %d/%d 已用 %.1fs 剩余预计 %.1fs",
                        i + 1, len(samples), elapsed, eta)

    # 5. 最终输出
    save_partial({"completed": completed})

    output = {
        "fingerprint": {
            "source": str(INPUT_FILE.name),
            "count": len(results),
            "needs_human_review_count": len(needs_review_list),
            "low_agreement_threshold": LOW_AGREEMENT_THRESHOLD,
            "arbitration_model": JUDGE_MODEL,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
        "cost_summary": tracker.summary() if hasattr(tracker, "summary") else {},
        "samples": results,
        "needs_human_review_list": [
            {
                "id": r["id"],
                "query": r["query"],
                "topic": r.get("topic", ""),
                "agreement_score": r["arbitration"].get("agreement_score", 0),
                "conflict_points": r["arbitration"].get("conflict_points", []),
            }
            for r in needs_review_list
        ],
    }
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    if PARTIAL_FILE.exists():
        os.remove(PARTIAL_FILE)
        logger.info("断点续跑缓存已清理")

    logger.info("输出: %s", OUTPUT_FILE)

    # 汇总
    print(f"\n=== 仲裁汇总 ===")
    print(f"总条数: {len(results)}")
    print(f"需人工复核: {len(needs_review_list)} ({len(needs_review_list)*100//len(results)}%)")
    print(f"耗时: {time.time()-t0:.1f}s")
    print()
    print("=== agreement_score 分布 ===")
    scores = [r["arbitration"].get("agreement_score", 0) for r in results]
    bins = {"0.0-0.4": 0, "0.4-0.6": 0, "0.6-0.8": 0, "0.8-1.0": 0}
    for s in scores:
        if s < 0.4:
            bins["0.0-0.4"] += 1
        elif s < 0.6:
            bins["0.4-0.6"] += 1
        elif s < 0.8:
            bins["0.6-0.8"] += 1
        else:
            bins["0.8-1.0"] += 1
    for k, v in bins.items():
        print(f"  {k}: {v}")
    print()
    print(f"=== 需人工复核清单（{len(needs_review_list)} 条）===")
    for r in needs_review_list[:10]:
        print(f"  [{r['id']}] score={r['arbitration'].get('agreement_score', 0):.2f} {r['query']}")
        for cp in r["arbitration"].get("conflict_points", [])[:2]:
            print(f"    冲突: {cp[:80]}")
    if len(needs_review_list) > 10:
        print(f"  ... 共 {len(needs_review_list)} 条，详见输出文件")


if __name__ == "__main__":
    main()
