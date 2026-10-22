# -*- coding: utf-8 -*-
"""
experiments/build_bench_dataset.py - 构建 HNSW 基准数据集

目的：为 Qdrant HNSW ef 调优生成足够规模的向量数据集（~5100 条），
使服务端模式下 HNSW 图真实建出（indexing_threshold=500），ef 参数生效。

流程：
1. 收集种子文本（docx chunk + 沉淀知识 + Qdrant payload content）—— 不调 API
2. LLM 扩写：每条种子生成 15-20 条语义变体（temperature=0.9）
3. 余弦去重：相似度 >= 0.95 的丢弃，防近重复导致 recall 虚高
4. embedding + upsert 到 disney_knowledge_bench（indexing_threshold=500）
5. 生成 500 条基准查询（LLM 从种子改写 + Qdrant diverse_questions 直取）

使用方法：
    # dry-run：只统计种子，不调任何 API
    export QDRANT_URL=http://localhost:6333
    python experiments/build_bench_dataset.py --dry-run

    # 完整执行（会产生 API 调用费用）
    export QDRANT_URL=http://localhost:6333
    python experiments/build_bench_dataset.py
"""
import argparse
import hashlib
import json
import logging
import os
import random
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
from concurrent.futures import ThreadPoolExecutor

# 把工程根加入 path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

from app.config import (
    CHAT_MODEL,
    DOCS_DIR,
    MULTIMODAL_EMBEDDING_MODEL,
    QDRANT_VECTOR_SIZE,
    STATS_DIR,
)
from app.core.query import client as agicto_client, get_text_embedding
from app.core.cost_tracker import tracked_chat_completion
from app.index.builder import (
    get_qdrant_client,
    parse_docx,
    preprocess_json_response,
    split_text,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ========== 参数 ==========
BENCH_COLLECTION_NAME = os.getenv("BENCH_COLLECTION", "disney_knowledge_bench")
PROD_COLLECTION_NAME = "disney_knowledge"

TARGET_VECTOR_COUNT = 3000     # 目标向量数（> indexing_threshold=500 即可触发 HNSW 建图）
TARGET_QUERY_COUNT = 500       # 目标基准查询数
COSINE_DEDUP_THRESHOLD = 0.95  # 余弦去重阈值（>= 此值视为近重复）
LLM_VARIANTS_PER_SEED = 60     # 每条种子 LLM 扩写变体数（60 变体/条，53 种子 1 轮出 3180 条）
EMBEDDING_INTERVAL = 0.1       # embedding 调用间隔（秒）
DEDUP_EMBED_WORKERS = 5      # 去重阶段并发 embedding 线程数（决策仍串行，保证结果确定性）
LLM_INTERVAL = 0.2             # LLM 调用间隔（秒）
RANDOM_SEED = 42

CHUNK_MIN_LEN = 200
CHUNK_MAX_LEN = 500

# ========== 输出文件 ==========
RAW_CORPUS_FILE = os.path.join(STATS_DIR, "bench_corpus_raw.json")
DEDUP_CORPUS_FILE = os.path.join(STATS_DIR, "bench_corpus_dedup.json")
BENCH_QUERIES_FILE = os.path.join(STATS_DIR, "bench_queries_500.json")
COST_LOG_FILE = os.path.join(STATS_DIR, "bench_dataset_cost.json")


# ========== 工具函数 ==========

def _list_hash(items: List[str]) -> str:
    """列表内容 hash，用于缓存指纹"""
    return hashlib.sha256("\n".join(items).encode("utf-8")).hexdigest()[:16]


def _load_env_seeds() -> List[str]:
    """从沉淀知识文件提取种子文本"""
    distilled_file = os.path.join(STATS_DIR, "disney_distilled.json")
    if not os.path.exists(distilled_file):
        return []
    with open(distilled_file, "r", encoding="utf-8") as f:
        payload = json.load(f)
    merged = payload.get("merged_knowledge", []) if isinstance(payload, dict) else []
    texts = []
    for item in merged:
        if isinstance(item, dict):
            content = item.get("content", "").strip()
            if content and len(content) >= 20:
                texts.append(content)
    return texts


def _load_docx_seeds() -> List[str]:
    """从 docx 文件切分 chunk 作为种子"""
    chunks = []
    for filename in sorted(os.listdir(DOCS_DIR)):
        if not filename.endswith(".docx"):
            continue
        file_path = os.path.join(DOCS_DIR, filename)
        if os.path.isdir(file_path):
            continue
        full_text = parse_docx(file_path)
        file_chunks = split_text(full_text)
        for ch in file_chunks:
            if CHUNK_MIN_LEN <= len(ch) <= CHUNK_MAX_LEN:
                chunks.append(ch)
    return chunks


def _load_qdrant_seeds(qdrant_client) -> Tuple[List[str], List[str]]:
    """从生产 collection 读取 content（种子）和 diverse_questions（直接做查询）

    返回 (content_seeds, direct_queries)
    """
    content_seeds = []
    direct_queries = []
    offset = None
    while True:
        points, next_offset = qdrant_client.scroll(
            collection_name=PROD_COLLECTION_NAME,
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for point in points:
            pl = point.payload or {}
            # content 字段作为种子文本
            content = (pl.get("content") or "").strip()
            if content and len(content) >= 20:
                content_seeds.append(content)
            # diverse_questions 直接作为基准查询
            dq = pl.get("diverse_questions") or []
            for q in dq:
                if isinstance(q, dict):
                    question = (q.get("question") or "").strip()
                    if question:
                        direct_queries.append(question)
                elif isinstance(q, str) and q.strip():
                    direct_queries.append(q.strip())
        if next_offset is None:
            break
        offset = next_offset
    return content_seeds, direct_queries


def collect_seeds() -> Dict:
    """收集全部种子源，返回统计信息（不调 LLM/embedding API）"""
    qdrant_client = get_qdrant_client()

    docx_seeds = _load_docx_seeds()
    distilled_seeds = _load_env_seeds()
    qdrant_content_seeds, qdrant_direct_queries = _load_qdrant_seeds(qdrant_client)

    # 合并种子文本（docx + 沉淀知识 + Qdrant content），去重
    all_seeds = docx_seeds + distilled_seeds + qdrant_content_seeds
    seen = set()
    unique_seeds = []
    for s in all_seeds:
        key = s.strip()[:100].lower()
        if key and key not in seen:
            seen.add(key)
            unique_seeds.append(s)

    return {
        "docx_chunks": len(docx_seeds),
        "distilled_knowledge": len(distilled_seeds),
        "qdrant_content": len(qdrant_content_seeds),
        "qdrant_direct_queries": len(qdrant_direct_queries),
        "unique_seeds": len(unique_seeds),
        "seeds": unique_seeds,
        "direct_queries": qdrant_direct_queries,
    }


# ========== LLM 扩写 ==========

def llm_rewrite_seeds(seeds: List[str]) -> List[str]:
    """对每条种子调用 LLM 生成语义变体，返回原始语料列表

    每条种子生成 LLM_VARIANTS_PER_SEED 条变体（改写/换问法/相关延伸）。
    带缓存到 bench_corpus_raw.json，按 seeds hash 指纹校验。
    """
    # 缓存校验
    seeds_fp = _list_hash(seeds)
    if os.path.exists(RAW_CORPUS_FILE):
        with open(RAW_CORPUS_FILE, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, dict) and payload.get("fingerprint", {}).get("seeds_hash") == seeds_fp:
            corpus = payload.get("corpus", [])
            logger.info("已加载缓存原始语料: %s 条（指纹匹配）", len(corpus))
            return corpus

    logger.info("开始 LLM 扩写: %s 条种子 x %s 变体/条", len(seeds), LLM_VARIANTS_PER_SEED)
    corpus = []
    for i, seed in enumerate(seeds, 1):
        prompt = f"""请基于以下知识内容，生成 {LLM_VARIANTS_PER_SEED} 条语义变体文本。

要求：
1. 变体方式：同义改写、换问法、相关延伸问题、口语化表达、正式表述
2. 每条变体必须是独立完整的文本（20-200 字）
3. 内容必须与原文语义相关，可从原文推导
4. 多样化，不要重复

返回 JSON 格式：
{{"variants": ["变体1", "变体2", ...]}}

知识内容：
{seed}"""
        try:
            response = tracked_chat_completion(
                client=agicto_client,
                model=CHAT_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.9,
                source="bench_dataset-扩写",
            )
            raw = response.choices[0].message.content
            raw = preprocess_json_response(raw)
            result = json.loads(raw)
            variants = result.get("variants", [])
            corpus.extend([v.strip() for v in variants if v and v.strip()])
        except Exception as e:
            logger.warning("LLM 扩写失败 (seed %s): %s", i, e)

        if i % 20 == 0 or i == len(seeds):
            logger.info("  扩写进度: %s/%s seeds, 累计 %s 条变体", i, len(seeds), len(corpus))
        time.sleep(LLM_INTERVAL)

    # 缓存
    os.makedirs(STATS_DIR, exist_ok=True)
    payload = {
        "fingerprint": {
            "seeds_hash": seeds_fp,
            "llm_model": CHAT_MODEL,
            "variants_per_seed": LLM_VARIANTS_PER_SEED,
            "created_at": datetime.now().isoformat(),
        },
        "corpus": corpus,
    }
    with open(RAW_CORPUS_FILE, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    logger.info("LLM 扩写完成: %s 条原始语料 -> %s", len(corpus), RAW_CORPUS_FILE)
    return corpus


# ========== 余弦去重 ==========

def dedup_by_cosine(texts: List[str]) -> List[str]:
    """对文本列表做余弦相似度去重，返回去重后的文本列表

    逐条计算 embedding，与已保留文本的向量做余弦相似度，
    >= COSINE_DEDUP_THRESHOLD 的视为近重复，跳过。
    """
    # 缓存校验
    texts_fp = _list_hash(texts)
    if os.path.exists(DEDUP_CORPUS_FILE):
        with open(DEDUP_CORPUS_FILE, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, dict) and payload.get("fingerprint", {}).get("texts_hash") == texts_fp:
            deduped = payload.get("deduped", [])
            logger.info("已加载缓存去重语料: %s 条（指纹匹配）", len(deduped))
            return deduped

    # 断点恢复：检查是否有 checkpoint
    checkpoint_file = os.path.join(STATS_DIR, "bench_dedup_checkpoint.json")
    start_idx = 0
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r", encoding="utf-8") as f:
            cp = json.load(f)
        if cp.get("texts_fp") == texts_fp:
            kept_texts = cp.get("kept_texts", [])
            kept_vectors = cp.get("kept_vectors", [])
            start_idx = cp.get("next_idx", 0)
            logger.info("从断点恢复: 跳过前 %s 条, 已保留 %s 条", start_idx, len(kept_texts))
        else:
            kept_texts = []
            kept_vectors = []
    else:
        kept_texts = []
        kept_vectors = []

    logger.info("开始余弦去重: %s 条原始语料, 阈值 %.2f, 从 %s 开始（块内并发 embedding，决策串行保持确定性）",
                len(texts), COSINE_DEDUP_THRESHOLD, start_idx)
    chunk_size = 200  # 与断点间隔对齐：每块并发 embedding，块末按原顺序串行判重并保存断点
    for chunk_start in range(start_idx, len(texts), chunk_size):
        chunk_end = min(chunk_start + chunk_size, len(texts))
        chunk_texts = texts[chunk_start:chunk_end]

        # 阶段 A：并发 embedding（仅加速向量计算，不改变任何判重顺序）
        with ThreadPoolExecutor(max_workers=DEDUP_EMBED_WORKERS) as pool:
            chunk_vecs = list(pool.map(
                lambda t: np.array(get_text_embedding(t), dtype=np.float32),
                chunk_texts,
            ))

        # 阶段 B：按原始顺序串行判重（与串行版结果完全一致）
        for text, vec in zip(chunk_texts, chunk_vecs):
            is_dup = False
            if kept_vectors:
                vecs_np = np.array(kept_vectors, dtype=np.float32)
                dots = vecs_np @ vec
                norms = np.linalg.norm(vecs_np, axis=1) * np.linalg.norm(vec)
                sims = dots / (norms + 1e-8)
                if np.max(sims) >= COSINE_DEDUP_THRESHOLD:
                    is_dup = True
            if not is_dup:
                kept_texts.append(text)
                kept_vectors.append(vec.tolist())

        logger.info("  去重进度: %s/%s, 保留 %s 条", chunk_end, len(texts), len(kept_texts))
        # 保存断点（每 200 条，格式与原版一致）
        cp_payload = {
            "texts_fp": texts_fp,
            "next_idx": chunk_end,
            "kept_texts": kept_texts,
            "kept_vectors": kept_vectors,
        }
        with open(checkpoint_file, "w", encoding="utf-8") as f:
            json.dump(cp_payload, f, ensure_ascii=False)

    # 清理断点文件
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)

    # 缓存
    payload = {
        "fingerprint": {
            "texts_hash": texts_fp,
            "threshold": COSINE_DEDUP_THRESHOLD,
            "embedding_model": MULTIMODAL_EMBEDDING_MODEL,
            "created_at": datetime.now().isoformat(),
        },
        "deduped": kept_texts,
    }
    with open(DEDUP_CORPUS_FILE, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    logger.info("余弦去重完成: %s -> %s 条 -> %s", len(texts), len(kept_texts), DEDUP_CORPUS_FILE)
    return kept_texts


# ========== Qdrant collection 创建 ==========

def create_bench_collection(qdrant_client):
    """创建基准 collection（indexing_threshold=500, dim=1152, Euclid）"""
    from qdrant_client import models

    # 如果已存在先删除（重建）
    collections = qdrant_client.get_collections()
    existing = [c.name for c in collections.collections]
    if BENCH_COLLECTION_NAME in existing:
        logger.info("collection '%s' 已存在，删除重建", BENCH_COLLECTION_NAME)
        qdrant_client.delete_collection(BENCH_COLLECTION_NAME)

    qdrant_client.create_collection(
        collection_name=BENCH_COLLECTION_NAME,
        vectors_config=models.VectorParams(
            size=QDRANT_VECTOR_SIZE,
            distance=models.Distance.EUCLID,
        ),
        optimizers_config=models.OptimizersConfigDiff(
            indexing_threshold=500,
        ),
        hnsw_config=models.HnswConfigDiff(
            m=16,
            ef_construct=100,
        ),
    )
    logger.info("已创建 collection '%s' (dim=%s, distance=Euclid, indexing_threshold=500)",
                BENCH_COLLECTION_NAME, QDRANT_VECTOR_SIZE)


# ========== embedding + upsert ==========

def embed_and_upsert(texts: List[str], qdrant_client) -> Dict:
    """批量 embedding 并 upsert 到 bench collection"""
    from qdrant_client import models

    logger.info("开始 embedding + upsert: %s 条文本", len(texts))
    batch_size = 64
    total_upserted = 0
    point_id = 0
    t0 = time.time()

    for batch_start in range(0, len(texts), batch_size):
        batch_texts = texts[batch_start:batch_start + batch_size]
        # 批内并发 embedding（提速，不改变向量与文本的对应顺序）
        with ThreadPoolExecutor(max_workers=DEDUP_EMBED_WORKERS) as pool:
            vectors = list(pool.map(get_text_embedding, batch_texts))

        # 构建 points
        points = []
        for j, (text, vec) in enumerate(zip(batch_texts, vectors)):
            points.append(models.PointStruct(
                id=point_id,
                vector=vec,
                payload={
                    "text": text,
                    "business_id": "bench_%06d" % point_id,
                    "source": "llm_rewrite",
                    "created_at": datetime.now().isoformat(),
                },
            ))
            point_id += 1

        qdrant_client.upsert(
            collection_name=BENCH_COLLECTION_NAME,
            points=points,
        )
        total_upserted += len(points)
        elapsed = time.time() - t0
        if total_upserted % 320 == 0 or total_upserted >= len(texts):
            logger.info("  upsert 进度: %s/%s (%.1fs)", total_upserted, len(texts), elapsed)

    elapsed = time.time() - t0
    logger.info("embedding + upsert 完成: %s 条, 耗时 %.1fs", total_upserted, elapsed)
    return {"upserted": total_upserted, "elapsed_sec": round(elapsed, 1)}


# ========== 基准查询生成 ==========

def generate_benchmark_queries(seeds: List[str], direct_queries: List[str]) -> List[str]:
    """生成 500 条基准查询（Qdrant diverse_questions 直取 + LLM 从种子改写）"""
    # 缓存校验
    all_input = seeds + direct_queries
    input_fp = _list_hash(all_input)
    if os.path.exists(BENCH_QUERIES_FILE):
        with open(BENCH_QUERIES_FILE, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, dict) and payload.get("fingerprint", {}).get("input_hash") == input_fp:
            queries = payload.get("queries", [])
            logger.info("已加载缓存基准查询: %s 条（指纹匹配）", len(queries))
            return queries

    logger.info("生成基准查询: 直取 %s 条 + LLM 补充", len(direct_queries))
    queries = list(direct_queries)
    needed = TARGET_QUERY_COUNT - len(queries)
    if needed > 0:
        # 每条种子生成 2 条查询
        per_seed = max(1, needed // len(seeds) + 1)
        for i, seed in enumerate(seeds):
            if len(queries) >= TARGET_QUERY_COUNT:
                break
            prompt = f"""基于以下知识内容，生成 {per_seed} 条用户可能提出的提问。

要求：多样化（直接问/口语化/省略主语），能从知识中找到答案。
返回 JSON: {{"questions": ["问1", "问2"]}}

知识内容：
{seed}"""
            try:
                response = tracked_chat_completion(
                    client=agicto_client,
                    model=CHAT_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,
                    source="bench_dataset-查询生成",
                )
                raw = response.choices[0].message.content
                raw = preprocess_json_response(raw)
                result = json.loads(raw)
                qs = result.get("questions", [])
                queries.extend([q.strip() for q in qs if q and q.strip()])
            except Exception as e:
                logger.warning("查询生成失败 (seed %s): %s", i, e)
            if i % 10 == 0:
                logger.info("  查询进度: %s/%s seeds, 累计 %s 条", i, len(seeds), len(queries))
            time.sleep(LLM_INTERVAL)

    # 去重 + 截断到 500
    seen = set()
    unique = []
    for q in queries:
        key = q.strip().lower()
        if key and key not in seen:
            seen.add(key)
            unique.append(q.strip())
    unique = unique[:TARGET_QUERY_COUNT]

    # 缓存
    payload = {
        "fingerprint": {"input_hash": input_fp, "created_at": datetime.now().isoformat()},
        "queries": unique,
    }
    with open(BENCH_QUERIES_FILE, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    logger.info("基准查询生成完成: %s 条 -> %s", len(unique), BENCH_QUERIES_FILE)
    return unique


# ========== 主流程 ==========

def main():
    parser = argparse.ArgumentParser(description="构建 HNSW 基准数据集")
    parser.add_argument("--dry-run", action="store_true", help="只统计种子，不调 API")
    args = parser.parse_args()

    print("=" * 60)
    print("  构建 HNSW 基准数据集")
    print("=" * 60)

    # Step 1: 收集种子（不调 API）
    print("\n[Step 1] 收集种子文本...")
    stats = collect_seeds()
    seeds = stats["seeds"]
    direct_queries = stats["direct_queries"]

    print(f"\n  种子统计:")
    print(f"    docx chunk:        {stats['docx_chunks']}")
    print(f"    沉淀知识:          {stats['distilled_knowledge']}")
    print(f"    Qdrant content:    {stats['qdrant_content']}")
    print(f"    Qdrant 直取查询:   {stats['qdrant_direct_queries']}")
    print(f"    去重后种子总数:    {stats['unique_seeds']}")

    if args.dry_run:
        # 估算
        # LLM 扩写：种子 * 18 变体 = 原始语料
        raw_corpus_est = len(seeds) * LLM_VARIANTS_PER_SEED
        # 去重保留率约 70%（经验值）
        deduped_est = int(raw_corpus_est * 0.7)
        # 如果去重后不足 5100，需要多轮扩写
        rounds = max(1, (TARGET_VECTOR_COUNT + deduped_est - 1) // max(deduped_est, 1))
        # LLM 调用：扩写 + 查询生成
        llm_rewrite_calls = len(seeds) * rounds
        llm_query_calls = max(0, (TARGET_QUERY_COUNT - len(direct_queries) + 1) // 2 + 1)  # 每种子 2 条
        llm_total = llm_rewrite_calls + llm_query_calls
        # embedding 调用：去重（原始语料）+ 最终入库（去重后）+ 查询
        emb_dedup = raw_corpus_est * rounds
        emb_upsert = min(deduped_est * rounds, TARGET_VECTOR_COUNT)
        emb_queries = TARGET_QUERY_COUNT
        emb_total = emb_dedup + emb_upsert + emb_queries
        # 成本估算（DashScope embedding ~0.0007元/千token, LLM ~0.001元/千token）
        llm_cost = llm_total * 0.002
        emb_cost = emb_total * 0.0007
        total_cost = llm_cost + emb_cost

        print(f"\n  预估:")
        print(f"    原始语料(1轮):     {raw_corpus_est} 条")
        print(f"    去重后(1轮):       {deduped_est} 条")
        print(f"    需要扩写轮数:      {rounds}")
        print(f"    LLM 调用:          {llm_total} 次 (扩写 {llm_rewrite_calls} + 查询 {llm_query_calls})")
        print(f"    embedding 调用:    {emb_total} 次 (去重 {emb_dedup} + 入库 {emb_upsert} + 查询 {emb_queries})")
        print(f"    预估成本:          {total_cost:.2f} 元 (LLM {llm_cost:.2f} + embedding {emb_cost:.2f})")
        print(f"    预估耗时:          {emb_total * EMBEDDING_INTERVAL / 60:.1f} 分钟 (embedding 间隔 {EMBEDDING_INTERVAL}s)")
        print(f"\n  [dry-run] 未调用任何 API")
        return

    # Step 2: LLM 扩写
    print("\n[Step 2] LLM 扩写...")
    raw_corpus = llm_rewrite_seeds(seeds)
    print(f"  原始语料: {len(raw_corpus)} 条")

    # 如果不足目标，多轮扩写（换 temperature/prompt 变体）
    rounds = 1
    while len(raw_corpus) < TARGET_VECTOR_COUNT and rounds < 5:
        rounds += 1
        logger.info("第 %s 轮扩写（当前 %s 条，目标 %s+）", rounds, len(raw_corpus), TARGET_VECTOR_COUNT)
        more = llm_rewrite_seeds(seeds)
        raw_corpus.extend(more)
        logger.info("累计原始语料: %s 条", len(raw_corpus))

    # Step 3: 余弦去重
    print("\n[Step 3] 余弦去重...")
    deduped_corpus = dedup_by_cosine(raw_corpus)
    print(f"  去重后语料: {len(deduped_corpus)} 条")

    if len(deduped_corpus) < TARGET_VECTOR_COUNT:
        logger.warning("去重后仅 %s 条 < 目标 %s，将全部入库", len(deduped_corpus), TARGET_VECTOR_COUNT)
    else:
        deduped_corpus = deduped_corpus[:TARGET_VECTOR_COUNT]

    # Step 4: 创建 bench collection
    print("\n[Step 4] 创建 bench collection...")
    qdrant_client = get_qdrant_client()
    create_bench_collection(qdrant_client)

    # Step 5: embedding + upsert
    print("\n[Step 5] embedding + upsert...")
    upsert_stats = embed_and_upsert(deduped_corpus, qdrant_client)

    # Step 6: 生成基准查询
    print("\n[Step 6] 生成基准查询...")
    queries = generate_benchmark_queries(seeds, direct_queries)

    # Step 7: 验证 HNSW 建图
    print("\n[Step 7] 验证 HNSW 建图...")
    import time as _time
    _time.sleep(5)  # 等待 optimizer 处理
    col_info = qdrant_client.get_collection(BENCH_COLLECTION_NAME)
    indexed = col_info.indexed_vectors_count
    total = col_info.points_count
    print(f"  indexed_vectors_count: {indexed}")
    print(f"  points_count:          {total}")

    if indexed > 0:
        print("  HNSW 索引已建出, ef 参数将生效")
    else:
        print("  [警告] indexed_vectors_count=0, HNSW 未建图, ef 不会生效")
        print("  可能原因: 向量数未超 indexing_threshold(2000), 或 optimizer 未完成")

    # 成本记录
    cost_log = {
        "seeds_count": len(seeds),
        "raw_corpus": len(raw_corpus),
        "deduped_corpus": len(deduped_corpus),
        "upserted": upsert_stats["upserted"],
        "queries": len(queries),
        "indexed_vectors": indexed,
        "points_count": total,
        "rounds": rounds,
        "timestamp": datetime.now().isoformat(),
    }
    with open(COST_LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(cost_log, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print("  基准数据集构建完成")
    print("=" * 60)
    print(f"  种子数:        {len(seeds)}")
    print(f"  原始语料:      {len(raw_corpus)}")
    print(f"  去重后语料:    {len(deduped_corpus)}")
    print(f"  入库向量:      {upsert_stats['upserted']}")
    print(f"  基准查询:      {len(queries)}")
    print(f"  HNSW 索引:     {indexed}/{total}")
    print(f"  成本记录:      {COST_LOG_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    main()
