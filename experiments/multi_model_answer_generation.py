"""
对 text_subset_candidates.json 中的 50 条候选查询做"多模型并行 expected_answer 生成 + 交叉校准"。

流程：
  1. 加载 50 条候选清单
  2. 对每条 query，从 Qdrant 检索 top-5 文本 chunk 作为事实依据
  3. 用 3 个模型并行生成 expected_answer：
       - gpt-4o        (AGICTO 平台，最强第三方裁判)
       - deepseek-v4-flash (AGICTO 平台，主被测模型家族)
       - deepseek-v4-pro  (AGICTO 平台，主被测模型家族升级版)
  4. 输出交叉对照清单：每条 query 带 3 份答案 + 检索原文 + 模型间分歧标记
  5. 提供"合并要点"字段（待人工校准时使用）

缓存机制：
  - 检索结果按 query 哈希缓存，避免重复消耗 embedding 额度
  - 每条 query 的 3 份答案独立落盘 partial JSON，支持断点续跑
  - 全部跑通后汇总输出 final JSON
"""
import json
import logging
import os
import sys
import time
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from app.config import (
    AGICTO_API_KEY,
    CHAT_BASE_URL,
    CHAT_MODEL,
    DIVERSE_REWRITE_MODEL,
    JUDGE_MODEL,
)
from app.core.cost_tracker import CostTracker, tracked_chat_completion
from app.core.query import distance_to_similarity, get_text_embedding, search_vectors
from app.index.builder import get_qdrant_client, collection_exists
from openai import OpenAI

# ===== 配置 =====
INPUT_FILE = PROJECT_ROOT / "data/stats/text_subset_candidates.json"
RETR_CACHE = PROJECT_ROOT / "data/stats/text_subset_retrieval.json"
PARTIAL_FILE = PROJECT_ROOT / "data/stats/text_subset_answers.partial.json"
OUTPUT_FILE = PROJECT_ROOT / "data/stats/text_subset_with_answers.json"

# 3 个并行生成模型
MODELS = [
    {"name": "gpt-4o", "model_id": JUDGE_MODEL, "temperature": 0.0, "family": "openai"},
    {"name": "deepseek-v4-flash", "model_id": CHAT_MODEL, "temperature": 0.0, "family": "deepseek"},
    {"name": "deepseek-v4-pro", "model_id": DIVERSE_REWRITE_MODEL, "temperature": 0.0, "family": "deepseek"},
]

LLM_CALL_INTERVAL = 0.2          # LLM 调用间隔（秒）
LLM_MAX_WORKERS = 3              # 并发线程数（每条 query 内 3 模型并行）
TOP_K_CONTEXT = 5                # 每条 query 取 top-5 文本 chunk 作为事实依据
RANDOM_SEED = 42

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def make_client() -> OpenAI:
    """创建 AGICTO OpenAI 兼容客户端"""
    return OpenAI(api_key=AGICTO_API_KEY, base_url=CHAT_BASE_URL)


# ========== Step 1: 检索 ==========

def retrieve_context(query_str: str, qdrant_client, retrie_cache: Dict) -> List[Dict]:
    """对单条 query 检索 top-k 文本 chunk，按 query 哈希缓存"""
    q_hash = hashlib.md5(query_str.encode()).hexdigest()[:8]
    if q_hash in retrie_cache:
        return retrie_cache[q_hash]

    q_vec = get_text_embedding(query_str)
    raw_results = search_vectors(q_vec, qdrant_client)
    text_results = []
    for dist, _business_id, m in raw_results:
        if m.get("deleted", False):
            continue
        if m.get("type") != "text":
            continue
        sim = distance_to_similarity(dist)
        text_results.append({
            "source": m.get("source", ""),
            "content": m.get("content", "")[:600],
            "similarity": round(float(sim), 4),
            "distance": float(dist),
        })
        if len(text_results) >= TOP_K_CONTEXT:
            break
    retrie_cache[q_hash] = text_results
    return text_results


def build_context_str(results: List[Dict]) -> str:
    """把检索结果拼接成 LLM 输入的 context 字符串"""
    if not results:
        return "[无检索结果]"
    parts = []
    for i, r in enumerate(results, 1):
        parts.append(f"[背景知识 {i}] (来源: {r['source']}, 相似度: {r['similarity']:.4f})\n{r['content']}")
    return "\n\n".join(parts)


# ========== Step 2: 多模型并行 expected_answer 生成 ==========

GEN_PROMPT_TEMPLATE = """你是一名严格的客服知识评测集构造助手。请基于以下检索到的知识库原文，为用户问题生成 expected_answer。

要求：
1. 答案必须严格基于检索到的原文，不允许编造
2. 答案为"原子事实清单"形式：每个原子事实一句话，按要点列出，每点不超过 40 字
3. 答案覆盖用户问题的所有可回答要点（不可遗漏关键信息）
4. 如果检索原文中无相关信息，返回 "[未找到相关信息]"
5. 不要使用 Markdown 格式，直接输出要点列表

[检索原文]
{context}

[用户问题]
{query}

[输出格式]
- 要点 1
- 要点 2
- ...
"""


def call_one_model(client: OpenAI, model_cfg: Dict, prompt: str, tracker: CostTracker) -> str:
    """调用单个模型生成 expected_answer"""
    time.sleep(LLM_CALL_INTERVAL)
    response = tracked_chat_completion(
        client=client,
        model=model_cfg["model_id"],
        messages=[{"role": "user", "content": prompt}],
        tracker=tracker,
        source=f"评测集扩充-{model_cfg['name']}",
        temperature=model_cfg["temperature"],
    )
    return response.choices[0].message.content.strip()


def generate_three_answers(query_str: str, context_str: str, tracker: CostTracker) -> Dict[str, str]:
    """对单条 query 用 3 个模型并行生成答案"""
    prompt = GEN_PROMPT_TEMPLATE.format(context=context_str, query=query_str)
    client = make_client()
    answers = {}
    with ThreadPoolExecutor(max_workers=LLM_MAX_WORKERS) as pool:
        futures = {
            pool.submit(call_one_model, client, m, prompt, tracker): m["name"]
            for m in MODELS
        }
        for fut in as_completed(futures):
            model_name = futures[fut]
            try:
                answers[model_name] = fut.result()
            except Exception as e:
                logger.warning("模型 %s 生成失败: %s", model_name, str(e)[:120])
                answers[model_name] = f"[ERROR: {str(e)[:80]}]"
    return answers


# ========== Step 3: 分歧检测 ==========

def detect_disagreement(answers: Dict[str, str]) -> Dict:
    """检测 3 份答案间的分歧程度"""
    keys = list(answers.keys())
    if any(answers[k].startswith("[ERROR") or answers[k].startswith("[未找到") for k in keys):
        return {"level": "high", "reason": "有模型生成失败或检索无结果"}

    # 简化分歧度量：用每两个模型答案的字符集 Jaccard 相似度均值
    def jaccard(a: str, b: str) -> float:
        sa, sb = set(a), set(b)
        if not sa and not sb:
            return 1.0
        return len(sa & sb) / max(1, len(sa | sb))

    sims = []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            sims.append(jaccard(answers[keys[i]], answers[keys[j]]))
    avg_sim = sum(sims) / max(1, len(sims))

    if avg_sim >= 0.85:
        level = "low"
    elif avg_sim >= 0.6:
        level = "medium"
    else:
        level = "high"
    return {
        "level": level,
        "avg_similarity": round(avg_sim, 4),
    }


# ========== Step 4: 断点续跑 ==========

def load_partial() -> Dict:
    """加载断点续跑缓存"""
    if PARTIAL_FILE.exists():
        with open(PARTIAL_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"completed": {}}


def save_partial(data: Dict) -> None:
    """保存断点续跑缓存"""
    with open(PARTIAL_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ========== 主流程 ==========

def main():
    # 1. 加载 50 条候选
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        candidates = json.load(f)
    samples = candidates["samples"]
    logger.info("加载 %d 条候选 query", len(samples))

    # 2. 加载检索缓存
    retrie_cache = {}
    if RETR_CACHE.exists():
        with open(RETR_CACHE, "r", encoding="utf-8") as f:
            retrie_cache = json.load(f)
        logger.info("检索缓存命中: %d 条", len(retrie_cache))

    # 3. 加载断点续跑缓存
    partial = load_partial()
    completed = partial.get("completed", {})
    logger.info("断点续跑: 已完成 %d / %d", len(completed), len(samples))

    # 4. 初始化 Qdrant + CostTracker
    qdrant_client = get_qdrant_client()
    if not collection_exists(qdrant_client):
        logger.error("Qdrant collection 不存在，请先构建索引")
        return
    tracker = CostTracker()

    # 5. 逐条处理
    t0 = time.time()
    results = []
    for i, s in enumerate(samples):
        qid = s["id"]
        query_str = s["query"]

        if qid in completed:
            logger.info("[%d/%d] %s 命中缓存，跳过", i + 1, len(samples), qid)
            results.append(completed[qid])
            continue

        logger.info("[%d/%d] %s 处理中: %s", i + 1, len(samples), qid, query_str[:40])

        # 5.1 检索 context
        try:
            ctx_results = retrieve_context(query_str, qdrant_client, retrie_cache)
            context_str = build_context_str(ctx_results)
        except Exception as e:
            logger.error("  检索失败: %s", str(e)[:120])
            context_str = "[检索失败]"

        # 5.2 三模型并行生成
        try:
            answers = generate_three_answers(query_str, context_str, tracker)
        except Exception as e:
            logger.error("  三模型生成失败: %s", str(e)[:120])
            answers = {m["name"]: f"[ERROR: {str(e)[:80]}]" for m in MODELS}

        # 5.3 分歧检测
        disagreement = detect_disagreement(answers)

        # 5.4 组装记录
        record = {
            "id": qid,
            "query": query_str,
            "topic": s.get("topic", ""),
            "length": s.get("length", len(query_str)),
            "context": context_str,
            "retrieval": ctx_results if isinstance(ctx_results, list) else [],
            "answers": answers,
            "disagreement": disagreement,
            "merged_answer": "",  # 待人工校准时填充
        }
        results.append(record)
        completed[qid] = record

        # 每 5 条落盘一次断点续跑
        if (i + 1) % 5 == 0:
            save_partial({"completed": completed})
            with open(RETR_CACHE, "w", encoding="utf-8") as f:
                json.dump(retrie_cache, f, ensure_ascii=False, indent=2)
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (len(samples) - i - 1)
            logger.info("  [进度] %d/%d 完成, 已用 %.1fs, 预计剩余 %.1fs",
                        i + 1, len(samples), elapsed, eta)

    # 6. 最终落盘
    save_partial({"completed": completed})
    with open(RETR_CACHE, "w", encoding="utf-8") as f:
        json.dump(retrie_cache, f, ensure_ascii=False, indent=2)

    output = {
        "fingerprint": {
            "source": str(INPUT_FILE.name),
            "count": len(results),
            "models": [m["name"] for m in MODELS],
            "context_top_k": TOP_K_CONTEXT,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
        "cost_summary": tracker.summary() if hasattr(tracker, "summary") else {},
        "samples": results,
    }
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    # 跑通后清理 partial
    if PARTIAL_FILE.exists():
        os.remove(PARTIAL_FILE)
        logger.info("断点续跑缓存已清理")

    logger.info("输出: %s", OUTPUT_FILE)

    # 控制台汇总
    print("\n=== 汇总 ===")
    print(f"总条数: {len(results)}")
    level_counter = {"low": 0, "medium": 0, "high": 0}
    for r in results:
        level_counter[r["disagreement"]["level"]] = level_counter.get(r["disagreement"]["level"], 0) + 1
    print(f"分歧分布: low={level_counter['low']}, medium={level_counter['medium']}, high={level_counter['high']}")
    print(f"\n=== 预览 (前 3 条) ===")
    for r in results[:3]:
        print(f"\n[{r['id']}] {r['query']}")
        for m_name, ans in r["answers"].items():
            print(f"  --- {m_name} ---")
            print(f"  {ans[:120]}...")
        print(f"  分歧: {r['disagreement']}")


if __name__ == "__main__":
    main()
