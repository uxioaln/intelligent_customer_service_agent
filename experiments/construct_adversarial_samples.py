"""
对抗样本构造：30 条，基于已有 121 条（A2 文本 50 + B 跨模态 41 + C 负例 30）做扰动。

4 种对抗类型：
  - 实体替换(8 条)：把"上海迪士尼"换成"北京迪士尼/东京迪士尼"等
  - 伪事实注入(8 条)：在合理问题中混入错误前提（"听说老人票免费"）
  - 指代歧义(7 条)：代词指代不明确（"那个怎么样"）
  - 数字篡改(7 条)：故意写错价格/数字（"老人票50元"）

每条带 expected_behavior：
  - "拒答-实体不在范围"
  - "纠错-指出伪事实"
  - "拒答-指代不明确"
  - "纠错-数字错误"

策略：
  1. 加载 A2/B/C 已有 121 条作为种子
  2. 用 gpt-4o 按对抗类型批量生成变体
  3. 每条校验：跑一次 Qdrant 检索，记录 top-1 similarity
  4. 输出 adversarial_samples.json
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
os.environ.setdefault("QDRANT_URL", "http://localhost:6333")

from app.config import JUDGE_MODEL, AGICTO_API_KEY, CHAT_BASE_URL
from app.core.cost_tracker import CostTracker, tracked_chat_completion
from app.core.query import distance_to_similarity, get_text_embedding, search_vectors
from app.index.builder import get_qdrant_client, collection_exists
from openai import OpenAI

# ===== 配置 =====
TEXT_FILE = PROJECT_ROOT / "data/stats/text_subset_with_answers.json"
CROSS_FILE = PROJECT_ROOT / "data/stats/cross_modal_samples_expanded.json"
NEG_FILE = PROJECT_ROOT / "data/stats/negative_samples_constructed.json"
OUTPUT_FILE = PROJECT_ROOT / "data/stats/adversarial_samples.json"
PARTIAL_FILE = PROJECT_ROOT / "data/stats/adversarial_samples.partial.json"

TARGET_COUNT = 30
LLM_CALL_INTERVAL = 0.3
RANDOM_SEED = 42

# 4 种对抗类型 + 目标条数
ADVERSARIAL_TYPES = [
    {
        "type": "entity_substitution",
        "name": "实体替换",
        "target_count": 8,
        "instruction": "把查询中的'上海迪士尼'替换为'北京迪士尼'或'东京迪士尼'等其他乐园。expected_behavior='拒答-实体不在范围'",
    },
    {
        "type": "false_fact_injection",
        "name": "伪事实注入",
        "target_count": 8,
        "instruction": "在合理问题中混入明显错误的前提（如'听说老人票免费'、'据说门票10元'等）。expected_behavior='纠错-指出伪事实'",
    },
    {
        "type": "ambiguous_reference",
        "name": "指代歧义",
        "target_count": 7,
        "instruction": "用代词或省略指代不明确（如'那个怎么样'、'怎么办理'、'能告诉我吗'）。expected_behavior='拒答-指代不明确'",
    },
    {
        "type": "number_tampering",
        "name": "数字篡改",
        "target_count": 7,
        "instruction": "故意写错价格或数字（如'老人票50元'、'65岁老人票100元'等），诱导模型给出错误答案。expected_behavior='纠错-数字错误'",
    },
]

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def make_client() -> OpenAI:
    return OpenAI(api_key=AGICTO_API_KEY, base_url=CHAT_BASE_URL)


GEN_PROMPT = """你是一名对抗样本构造助手。请基于以下种子查询，生成 {n} 条对抗样本。

对抗类型：{adv_type} - {adv_name}
指令：{instruction}

要求：
1. 每条对抗样本必须基于种子查询改写，保留原问题的合理外壳但注入对抗元素
2. 风格自然，像真实用户问法（口语化、含省略/指代均可）
3. 长度控制在 15~50 字
4. 不可与种子查询完全重复
5. 必须能从对抗元素本身判断 expected_behavior（拒答或纠错）

种子查询（从中选取并改写）：
{seeds}

输出 JSON（固定格式，每个字段都必填）：
{{
    "queries": [
        {{
            "query": "对抗样本查询文本",
            "source_query_id": "原查询的 id（如 text_001/cm_001/neg_001）",
            "expected_behavior": "{expected_behavior}",
            "tampered_element": "具体被篡改的元素（如'上海迪士尼→北京迪士尼'）"
        }}
    ]
}}
"""


def call_llm(client: OpenAI, prompt: str, tracker: CostTracker) -> str:
    time.sleep(LLM_CALL_INTERVAL)
    response = tracked_chat_completion(
        client=client,
        model=JUDGE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        tracker=tracker,
        source="评测集扩充-对抗样本",
        temperature=0.7,
    )
    return response.choices[0].message.content


def verify_query(query_str: str, qdrant_client) -> Dict:
    """记录检索 top-1 相似度，用于判断对抗样本是否会触发检索"""
    try:
        q_vec = get_text_embedding(query_str)
        raw_results = search_vectors(q_vec, qdrant_client)
        if not raw_results:
            return {"top_similarity": 0.0, "top_source": ""}
        top_dist, _, m = raw_results[0]
        return {
            "top_similarity": round(float(distance_to_similarity(top_dist)), 4),
            "top_source": m.get("source", ""),
        }
    except Exception as e:
        return {"top_similarity": 0.0, "top_source": "", "error": str(e)[:80]}


def load_seeds() -> list:
    """加载 A2/B/C 三类已有样本作为种子"""
    seeds = []
    # A2 文本
    with open(TEXT_FILE) as f:
        d = json.load(f)
    for s in d["samples"]:
        seeds.append({
            "id": s["id"],
            "query": s["query"],
            "type": "text",
            "topic": s.get("topic", ""),
        })
    # B 跨模态
    with open(CROSS_FILE) as f:
        d = json.load(f)
    for s in d["samples"]:
        seeds.append({
            "id": s["id"],
            "query": s["query"],
            "type": "cross_modal",
            "asset_name": s.get("asset_name", ""),
        })
    # C 负例
    with open(NEG_FILE) as f:
        d = json.load(f)
    for s in d["samples"]:
        seeds.append({
            "id": s["id"],
            "query": s["query"],
            "type": "negative",
            "topic": s.get("topic", ""),
        })
    return seeds


def load_partial() -> Dict:
    if PARTIAL_FILE.exists():
        with open(PARTIAL_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"completed": [], "rejected": []}


def save_partial(data: Dict) -> None:
    with open(PARTIAL_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    # 1. 加载种子
    seeds = load_seeds()
    logger.info("加载种子 %d 条 (text=%d, cross=%d, neg=%d)",
                len(seeds),
                sum(1 for s in seeds if s["type"] == "text"),
                sum(1 for s in seeds if s["type"] == "cross_modal"),
                sum(1 for s in seeds if s["type"] == "negative"))

    # 2. 断点续跑
    partial = load_partial()
    completed = partial.get("completed", [])
    rejected = partial.get("rejected", [])
    logger.info("断点续跑: 已完成 %d, 已拒绝 %d", len(completed), len(rejected))

    # 3. 初始化
    client = make_client()
    qdrant_client = get_qdrant_client()
    if not collection_exists(qdrant_client):
        logger.error("Qdrant collection 不存在")
        return
    tracker = CostTracker()

    # 4. 按对抗类型生成
    t0 = time.time()
    rounds = 0
    max_rounds = 10

    # 把种子按类型分组（文本种子用于实体替换/数字篡改/伪事实注入；跨模态用于指代歧义）
    text_seeds = [s for s in seeds if s["type"] == "text"]
    cross_seeds = [s for s in seeds if s["type"] == "cross_modal"]
    neg_seeds = [s for s in seeds if s["type"] == "negative"]

    while len(completed) < TARGET_COUNT and rounds < max_rounds:
        rounds += 1
        for adv in ADVERSARIAL_TYPES:
            current_count = sum(1 for c in completed if c.get("adv_type") == adv["type"])
            need = adv["target_count"] - current_count
            if need <= 0:
                continue
            batch_target = min(need + 3, 8)
            logger.info("[第 %d 轮][%s] 还需 %d 条, 本轮生成 %d 条候选",
                        rounds, adv["name"], need, batch_target)

            # 选择种子源
            if adv["type"] == "ambiguous_reference":
                source_seeds = cross_seeds + text_seeds
            elif adv["type"] == "entity_substitution":
                source_seeds = text_seeds
            elif adv["type"] == "false_fact_injection":
                source_seeds = text_seeds + neg_seeds
            else:  # number_tampering
                source_seeds = text_seeds

            # 随机抽 10 条种子
            import random
            random.seed(RANDOM_SEED + rounds)
            sampled_seeds = random.sample(source_seeds, min(10, len(source_seeds)))
            seed_str = "\n".join(f"  - [{s['id']}] {s['query']}" for s in sampled_seeds)

            prompt = GEN_PROMPT.format(
                n=batch_target,
                adv_type=adv["type"],
                adv_name=adv["name"],
                instruction=adv["instruction"],
                expected_behavior=adv["instruction"].split("expected_behavior=")[1].strip("'"),
                seeds=seed_str,
            )
            try:
                raw = call_llm(client, prompt, tracker)
            except Exception as e:
                logger.error("LLM 调用失败: %s", str(e)[:120])
                continue

            # 去 markdown
            raw = raw.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.rsplit("```", 1)[0]
            raw = raw.strip()

            try:
                data = json.loads(raw)
                queries = data.get("queries", [])
            except json.JSONDecodeError as e:
                logger.warning("JSON 解析失败: %s", str(e)[:80])
                continue

            for q_data in queries:
                q = (q_data.get("query") or "").strip()
                if not q:
                    continue
                if q in [c["query"] for c in completed]:
                    continue
                verification = verify_query(q, qdrant_client)
                record = {
                    "id": f"adv_{len(completed)+1:03d}",
                    "query": q,
                    "adv_type": adv["type"],
                    "adv_name": adv["name"],
                    "source_query_id": q_data.get("source_query_id", ""),
                    "expected_behavior": q_data.get("expected_behavior", ""),
                    "tampered_element": q_data.get("tampered_element", ""),
                    "verification": verification,
                }
                completed.append(record)
                logger.info("  [✓ %d] %s sim=%s %s",
                            len(completed), record["id"],
                            verification["top_similarity"], q[:50])
                if len(completed) >= TARGET_COUNT:
                    break
            save_partial({"completed": completed, "rejected": rejected})

        elapsed = time.time() - t0
        logger.info("[进度] 已完成 %d/%d, 已用 %.1fs", len(completed), TARGET_COUNT, elapsed)

    # 5. 最终输出
    type_stats = {}
    for adv in ADVERSARIAL_TYPES:
        records = [c for c in completed if c.get("adv_type") == adv["type"]]
        type_stats[adv["type"]] = {
            "name": adv["name"],
            "count": len(records),
            "target": adv["target_count"],
        }

    output = {
        "fingerprint": {
            "count": len(completed),
            "target_count": TARGET_COUNT,
            "types": type_stats,
            "rounds": rounds,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
        "samples": completed,
    }
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    if PARTIAL_FILE.exists():
        os.remove(PARTIAL_FILE)

    logger.info("输出: %s", OUTPUT_FILE)
    print(f"\n=== 汇总 ===")
    print(f"目标: {TARGET_COUNT}, 完成: {len(completed)}, 耗时: {time.time()-t0:.1f}s")
    print(f"\n=== 各对抗类型分布 ===")
    for t, s in type_stats.items():
        print(f"  {s['name']} ({t}): {s['count']}/{s['target']}")
    print(f"\n=== 样本预览（前 10 条）===")
    for s in completed[:10]:
        print(f"  [{s['id']}] [{s['adv_name']}] sim={s['verification']['top_similarity']:.4f} {s['query']}")
        print(f"    expected: {s['expected_behavior']}")
        print(f"    tampered: {s['tampered_element']}")


if __name__ == "__main__":
    main()
