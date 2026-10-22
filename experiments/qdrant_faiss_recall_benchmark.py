# -*- coding: utf-8 -*-
"""
experiments.5-qdrant_faiss_recall_benchmark - Qdrant vs FAISS 召回质量离线对比实验

功能：
1. 从知识库 docx 与 SQLite 历史对话生成约 300~400 条测试 query
2. 批量计算 query embedding 并缓存到 .npy，避免重复调用 API
3. 以 Qdrant 全量向量构建 FAISS IndexFlatL2 暴力基准，生成 Ground Truth (Top-20)
4. 双路召回对比：
   - FAISS 侧：加载 backup 索引（IVF 则 nprobe=1），检索 Top-10
   - Qdrant 侧：测试 hnsw_ef = [64, 96, 128, 192, 256, 384, 512]，检索 Top-10
5. 计算 Recall@1/3/5/10、MRR、与 FAISS 的 Jaccard 重叠度、P50/P95/P99 延迟
6. 输出结构化报告 data/stats/recall_benchmark_report.json

使用方法：
    export DASHSCOPE_API_KEY=xxx
    export AGICTO_API_KEY=xxx
    # HNSW/ef 生效前提：服务端模式（本地持久化模式退化为精确搜索，ef 不生效）
    export QDRANT_URL=http://localhost:6333
    python experiments/qdrant_faiss_recall_benchmark.py

依赖（FAISS 仅实验脚本临时使用）：
    pip install faiss-cpu==1.7.4 -i https://pypi.tuna.tsinghua.edu.cn/simple
"""
import glob
import hashlib
import json
import logging
import os
import random
import sqlite3
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np

# 把工程根加入 path，使 app.* 包可被导入
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from app.config import (
    CHAT_MODEL,
    DB_FILE,
    DOCS_DIR,
    INDEX_DIR,
    MULTIMODAL_EMBEDDING_MODEL,
    QDRANT_SEARCH_EF,
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

# ========== 实验参数 ==========
RANDOM_SEED = 42
CHUNK_MIN_LEN = 200          # chunk 最小长度（字）
CHUNK_MAX_LEN = 500          # chunk 最大长度（字）
TARGET_CHUNK_COUNT = 20      # 从 docx 抽取的 chunk 数量上限
QUESTIONS_PER_CHUNK = 3      # 每个 chunk 生成的问题数
TARGET_HISTORY_QUERY_COUNT = 20  # 从历史对话抽取的 query 数量上限
GT_TOP_K = 20                # Ground Truth Top-K（暴力搜索精确 Top-20）
SEARCH_TOP_K = 10            # 检索 Top-K
HNSW_EF_LIST = [8, 16, 32, 64, 96, 128, 192, 256, 384, 512]  # 10 档覆盖从低召回到满召回的完整权衡区间
FAISS_NPROBE = 1             # FAISS IVF nprobe（与旧生产配置一致）
EMBEDDING_CALL_INTERVAL = 0.1  # embedding 调用间隔（秒），避免触发限流
LLM_CALL_INTERVAL = 0.2        # LLM 调用间隔（秒）
WARMUP_ROUNDS = 3              # 预热搜索轮次（不计入延迟统计）

# ========== 输出文件 ==========
QUERIES_FILE = os.path.join(STATS_DIR, "benchmark_queries.json")
GT_FILE = os.path.join(STATS_DIR, "benchmark_ground_truth.json")
EMBEDDINGS_FILE = os.path.join(STATS_DIR, "benchmark_query_embeddings.npy")
REPORT_FILE = os.path.join(STATS_DIR, "recall_benchmark_report.json")

# 基准专用 collection（独立于生产 disney_knowledge，可用环境变量 BENCH_COLLECTION 覆盖）
BENCH_COLLECTION_NAME = os.getenv("BENCH_COLLECTION", "disney_knowledge_bench")
# embedding 缓存指纹 sidecar：模型或查询变更后禁止复用旧缓存（防止跨模型世代向量混算）
EMBEDDINGS_META_FILE = os.path.join(STATS_DIR, "benchmark_query_embeddings.meta.json")
# 沉淀知识文件（LLM 从历史对话提炼的结构化知识，作为 query 种子源补充）
DISNEY_DISTILLED_FILE = os.path.join(STATS_DIR, "disney_distilled.json")

# FAISS backup 目录
FAISS_BACKUP_DIR = os.path.join(INDEX_DIR, "backup")


# ========== 工具函数 ==========

def _docs_signature() -> str:
    """DOCS_DIR 下 docx 文件签名（文件名+大小+mtime），用于查询缓存指纹溯源"""
    parts = []
    for filename in sorted(os.listdir(DOCS_DIR)):
        if not filename.endswith(".docx"):
            continue
        file_path = os.path.join(DOCS_DIR, filename)
        if os.path.isfile(file_path):
            st = os.stat(file_path)
            parts.append("%s:%s:%s" % (filename, st.st_size, int(st.st_mtime)))
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:16]


def _queries_hash(queries: List[str]) -> str:
    """查询列表内容 hash，embedding 缓存指纹的核心字段"""
    return hashlib.sha256("\n".join(queries).encode("utf-8")).hexdigest()[:16]


def _import_faiss():
    """延迟导入 faiss，提供安装提示"""
    try:
        import faiss
        return faiss
    except ImportError:
        print("\n错误: 无法导入 faiss 模块。请先安装 faiss-cpu：")
        print("  pip install faiss-cpu==1.7.4 -i https://pypi.tuna.tsinghua.edu.cn/simple")
        sys.exit(1)


def chat_completion(prompt: str, model: str = CHAT_MODEL) -> str:
    """调用 AGICTO chat LLM 生成文本（复用 app.core.query 的 client）

    项目约定：所有 chat LLM 调用使用 AGICTO 平台。
    走 tracked_chat_completion 以获得空响应防护与重试。
    """
    response = tracked_chat_completion(
        client=agicto_client,
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        source="qdrant_benchmark-查询生成",
    )
    return response.choices[0].message.content


# ========== Step 1: 测试 Query 生成 ==========

def extract_docx_chunks() -> List[str]:
    """从 data/knowledge_base/ 下所有 docx 文件提取文本 chunk（长度 200~500 字）

    使用 app.index.builder 的 parse_docx / split_text 保持与索引构建一致的切分逻辑。
    """
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


def generate_questions_for_chunk(chunk: str) -> List[str]:
    """为单个文本 chunk 生成多样化用户提问（3 条）

    多样化要求：直接问、口语化、省略主语、同义改写。
    返回 JSON 格式解析后的问题列表。
    """
    prompt = f"""请基于以下知识内容，生成 {QUESTIONS_PER_CHUNK} 条用户可能提出的真实提问。

要求：
1. 提问方式多样化：直接提问、口语化表达、省略主语、同义改写
2. 问题必须能从给定知识内容中找到答案
3. 问题应自然，符合真实用户表达习惯

请返回 JSON 格式：
{{
    "questions": ["问题1", "问题2", "问题3"]
}}

知识内容：
{chunk}
"""
    response = chat_completion(prompt)
    response = preprocess_json_response(response)
    try:
        result = json.loads(response)
        questions = result.get("questions", [])
        return [q.strip() for q in questions if q and q.strip()]
    except json.JSONDecodeError:
        logger.warning("问题生成 JSON 解析失败，跳过该 chunk")
        return []


def load_history_queries() -> List[str]:
    """从 SQLite conversations 表抽取历史用户 query（去重）

    仅取 role='user' 的消息内容，DISTINCT 去重。
    """
    if not os.path.exists(DB_FILE):
        logger.warning("数据库文件不存在: %s", DB_FILE)
        return []

    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute(
        "SELECT DISTINCT content FROM conversations "
        "WHERE role = 'user' AND content IS NOT NULL AND content != ''"
    )
    rows = cur.fetchall()
    conn.close()

    queries = [row[0].strip() for row in rows if row[0] and row[0].strip()]
    return queries


def generate_test_queries() -> List[str]:
    """生成测试 query 列表（docx 改写 + 历史对话，合并去重）

    流程：
    1. 从 docx 抽取 chunk -> LLM 改写为提问
    2. 从 conversations 表抽取历史 user query
    3. 合并去重，保存为 benchmark_queries.json
    """
    # 若已有缓存则直接加载（兼容旧格式裸列表与新格式含指纹）
    if os.path.exists(QUERIES_FILE):
        with open(QUERIES_FILE, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, dict) and "queries" in payload:
            logger.info("查询缓存指纹: %s", payload.get("fingerprint"))
            queries = payload["queries"]
        else:
            queries = payload
        logger.info("已加载缓存测试 query: %s 条 -> %s", len(queries), QUERIES_FILE)
        return queries

    random.seed(RANDOM_SEED)

    # 1. 从 docx 抽取 chunk
    all_chunks = extract_docx_chunks()
    logger.info("从 docx 提取到 %s 个 chunk（长度 %s~%s 字）",
                len(all_chunks), CHUNK_MIN_LEN, CHUNK_MAX_LEN)
    if len(all_chunks) > TARGET_CHUNK_COUNT:
        sampled_chunks = random.sample(all_chunks, TARGET_CHUNK_COUNT)
    else:
        sampled_chunks = all_chunks
    logger.info("抽样 %s 个 chunk 用于生成提问", len(sampled_chunks))

    # 2. 为每个 chunk 生成 3 条提问
    generated_queries = []
    for i, chunk in enumerate(sampled_chunks, 1):
        questions = generate_questions_for_chunk(chunk)
        generated_queries.extend(questions)
        if i % 10 == 0 or i == len(sampled_chunks):
            logger.info("  已处理 %s/%s 个 chunk，累计生成 %s 条提问",
                        i, len(sampled_chunks), len(generated_queries))
        time.sleep(LLM_CALL_INTERVAL)
    logger.info("从 chunk 生成提问完成: %s 条", len(generated_queries))

    # 3. 从 SQLite 抽取历史 query（表不存在时容错返回空列表）
    history_queries = load_history_queries()
    logger.info("从 conversations 表抽取历史 query: %s 条", len(history_queries))
    if len(history_queries) > TARGET_HISTORY_QUERY_COUNT:
        history_queries = random.sample(history_queries, TARGET_HISTORY_QUERY_COUNT)

    # 3.5 从沉淀知识抽取种子文本并改写为提问（弥补 conversations 表缺失时的种子量）
    distilled_texts = load_distilled_knowledge_texts()
    distilled_queries = []
    for i, text in enumerate(distilled_texts, 1):
        questions = generate_questions_for_chunk(text)
        distilled_queries.extend(questions)
        if i % 5 == 0 or i == len(distilled_texts):
            logger.info("  沉淀知识已处理 %s/%s 条，累计生成 %s 条提问",
                        i, len(distilled_texts), len(distilled_queries))
        time.sleep(LLM_CALL_INTERVAL)
    logger.info("从沉淀知识生成提问完成: %s 条", len(distilled_queries))

    # 4. 合并去重（大小写不敏感去重，保留原始写法）
    all_queries = generated_queries + distilled_queries + history_queries
    seen = set()
    unique_queries = []
    for q in all_queries:
        key = q.strip().lower()
        if key and key not in seen:
            seen.add(key)
            unique_queries.append(q.strip())
    logger.info("合并去重后共 %s 条测试 query", len(unique_queries))

    # 5. 保存（含指纹，便于溯源；硬一致性由 embedding meta 的 query_hash 保证）
    os.makedirs(STATS_DIR, exist_ok=True)
    fingerprint = {
        "source": "docx+history",
        "docs_signature": _docs_signature(),
        "chat_model": CHAT_MODEL,
        "seed": RANDOM_SEED,
        "created_at": datetime.now().isoformat(),
    }
    with open(QUERIES_FILE, "w", encoding="utf-8") as f:
        json.dump({"fingerprint": fingerprint, "queries": unique_queries}, f, ensure_ascii=False, indent=2)
    logger.info("已保存测试 query -> %s", QUERIES_FILE)

    return unique_queries


# ========== Step 2: 批量计算 Query Embedding ==========

def compute_query_embeddings(queries: List[str]) -> np.ndarray:
    """批量计算 query embedding 并缓存到 .npy，避免重复调用 API

    所有检索（Ground Truth / FAISS / Qdrant）共用同一份 embedding，
    确保 query 侧变量唯一可控。
    """
    # 缓存硬校验：npy + meta sidecar 双文件，模型名与 query_hash 必须同时匹配，
    # 否则视为跨模型世代/跨查询集的陈旧缓存，强制重算（旧缓存曾导致 GT 与检索侧向量空间不一致）
    if os.path.exists(EMBEDDINGS_FILE) and os.path.exists(EMBEDDINGS_META_FILE):
        with open(EMBEDDINGS_META_FILE, "r", encoding="utf-8") as f:
            meta = json.load(f)
        cached = np.load(EMBEDDINGS_FILE)
        if (cached.shape[0] == len(queries)
                and meta.get("query_hash") == _queries_hash(queries)
                and meta.get("embedding_model") == MULTIMODAL_EMBEDDING_MODEL):
            logger.info("已加载缓存 embedding: %s 条（指纹匹配） -> %s", cached.shape[0], EMBEDDINGS_FILE)
            return cached
        logger.warning("embedding 缓存指纹不匹配（模型或查询集已变更），重新计算")
    elif os.path.exists(EMBEDDINGS_FILE):
        logger.warning("embedding 缓存缺少指纹元文件（旧时代缓存），重新计算")

    logger.info("开始批量计算 %s 条 query embedding...", len(queries))
    embeddings = []
    for i, query in enumerate(queries, 1):
        vec = get_text_embedding(query)
        embeddings.append(vec)
        if i % 50 == 0 or i == len(queries):
            logger.info("  embedding 进度: %s/%s", i, len(queries))
        time.sleep(EMBEDDING_CALL_INTERVAL)

    embeddings_np = np.array(embeddings, dtype=np.float32)
    os.makedirs(STATS_DIR, exist_ok=True)
    np.save(EMBEDDINGS_FILE, embeddings_np)
    # 写入指纹 meta，后续运行仅当模型与查询集均未变更时才复用缓存
    meta = {
        "embedding_model": MULTIMODAL_EMBEDDING_MODEL,
        "query_hash": _queries_hash(queries),
        "count": int(embeddings_np.shape[0]),
        "dim": int(embeddings_np.shape[1]),
        "created_at": datetime.now().isoformat(),
    }
    with open(EMBEDDINGS_META_FILE, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    logger.info("已缓存 embedding(%s) -> %s", embeddings_np.shape, EMBEDDINGS_FILE)
    return embeddings_np


# ========== Step 3: Ground Truth 生成 ==========

def scroll_qdrant_all_points(qdrant_client) -> Tuple[List[str], np.ndarray]:
    """从 Qdrant 滚动读取所有 point（含向量），返回 (business_id 列表, 向量矩阵)

    business_id 取自 payload，作为跨系统（Qdrant / FAISS）比对的统一标识。
    """
    all_business_ids = []
    all_vectors = []
    offset = None
    while True:
        points, next_offset = qdrant_client.scroll(
            collection_name=BENCH_COLLECTION_NAME,
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=True,
        )
        for point in points:
            bid = str((point.payload or {}).get("business_id", point.id))
            all_business_ids.append(bid)
            all_vectors.append(point.vector)
        if next_offset is None:
            break
        offset = next_offset

    vectors_np = np.array(all_vectors, dtype=np.float32) if all_vectors else np.zeros((0, QDRANT_VECTOR_SIZE), dtype=np.float32)
    dim = vectors_np.shape[1] if vectors_np.size > 0 else 0
    logger.info("从 Qdrant 导出 %s 个 point，向量维度 %s", len(all_business_ids), dim)
    return all_business_ids, vectors_np


def build_ground_truth(qdrant_client, queries: List[str], query_embeddings: np.ndarray) -> Dict:
    """构建 Ground Truth：以 FAISS IndexFlatL2 暴力搜索精确 Top-20

    流程：
    1. 从 Qdrant 导出全部向量与 business_id
    2. 构建 IndexFlatL2 灌入向量
    3. 对每条 query embedding 搜索 Top-20，作为 Ground Truth
    4. 保存为 benchmark_ground_truth.json
    """
    faiss = _import_faiss()

    # 导出 Qdrant 全量向量
    business_ids, vectors_np = scroll_qdrant_all_points(qdrant_client)
    if len(business_ids) == 0:
        raise ValueError("Qdrant collection 为空，无法构建 Ground Truth")

    # 构建 IndexFlatL2 暴力基准
    dim = vectors_np.shape[1]
    flat_index = faiss.IndexFlatL2(dim)
    flat_index.add(vectors_np)
    logger.info("已构建 IndexFlatL2 暴力基准: %s 条向量，维度 %s", flat_index.ntotal, dim)

    # 批量搜索 Top-20
    distances, indices = flat_index.search(query_embeddings, GT_TOP_K)

    # 构建 GT 字典：{query_id: {"query": "...", "gt_ids": [...]}}
    gt = {}
    for i, query in enumerate(queries):
        gt_ids = []
        for idx in indices[i]:
            if idx >= 0:
                gt_ids.append(business_ids[int(idx)])
        gt["q%d" % i] = {"query": query, "gt_ids": gt_ids}

    # 保存（GT 每次运行重建不缓存：暴力搜索成本为秒级，从根上避免陈旧 GT）
    os.makedirs(STATS_DIR, exist_ok=True)
    gt_payload = {
        "fingerprint": {
            "collection": BENCH_COLLECTION_NAME,
            "vector_count": len(business_ids),
            "query_hash": _queries_hash(queries),
            "embedding_model": MULTIMODAL_EMBEDDING_MODEL,
            "created_at": datetime.now().isoformat(),
        },
        "gt": gt,
    }
    with open(GT_FILE, "w", encoding="utf-8") as f:
        json.dump(gt_payload, f, ensure_ascii=False, indent=2)
    logger.info("已保存 Ground Truth(%s 条) -> %s", len(gt), GT_FILE)

    return gt


# ========== Step 4: 双路召回 ==========

def load_faiss_backup() -> Optional[Tuple]:
    """加载 FAISS backup 索引和元数据

    返回 (index, metadata_list, index_type) 或 None（backup 不存在时）。
    若索引为 IVF 类型，自动设置 nprobe=1（与旧生产配置一致）。
    """
    faiss = _import_faiss()

    index_files = sorted(glob.glob(os.path.join(FAISS_BACKUP_DIR, "*_disney_index.faiss")))
    metadata_files = sorted(glob.glob(os.path.join(FAISS_BACKUP_DIR, "*_disney_metadata.json")))
    if not index_files or not metadata_files:
        logger.warning("未找到 FAISS backup 索引/元数据文件: %s", FAISS_BACKUP_DIR)
        return None

    index_path = index_files[-1]
    metadata_path = metadata_files[-1]
    logger.info("加载 FAISS backup 索引: %s", os.path.basename(index_path))

    index = faiss.read_index(index_path)
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata_list = json.load(f)

    # 判断是否为 IVF 索引并设置 nprobe
    index_type = "Flat"
    try:
        ivf = faiss.extract_index_ivf(index)
        ivf.nprobe = FAISS_NPROBE
        index_type = "IVF(nprobe=%s)" % FAISS_NPROBE
        logger.info("FAISS 索引为 IVF 类型，已设置 nprobe=%s", FAISS_NPROBE)
    except RuntimeError:
        logger.info("FAISS 索引为 Flat 类型，无需设置 nprobe")

    logger.info("FAISS backup: %s 条向量，维度 %s", index.ntotal, index.d)
    return index, metadata_list, index_type


def search_faiss_single(index, metadata_list: List[Dict], query_vec_np: np.ndarray, top_k: int) -> List[str]:
    """使用 FAISS 索引检索单条 query，返回 business_id 列表

    FAISS 返回的 index 可能是位置（Flat）或存储 ID（IDMap），
    统一映射为 metadata 中的 business_id（即 str(metadata['id])）。
    """
    distances, indices = index.search(query_vec_np, top_k)
    result_ids = []
    for idx in indices[0]:
        if idx == -1:
            continue
        idx_int = int(idx)
        # 优先按位置映射（Flat 场景），兜底直接用 id 值
        if 0 <= idx_int < len(metadata_list):
            result_ids.append(str(metadata_list[idx_int].get("id", idx_int)))
        else:
            result_ids.append(str(idx_int))
    return result_ids


def search_qdrant_single(qdrant_client, query_vec: List[float], top_k: int, hnsw_ef: int) -> List[str]:
    """使用 Qdrant 检索单条 query，返回 business_id 列表

    通过 search_params 设置 hnsw_ef，exact=False 强制使用 HNSW 近似搜索。
    """
    from qdrant_client import models
    response = qdrant_client.query_points(
        collection_name=BENCH_COLLECTION_NAME,
        query=query_vec,
        limit=top_k,
        with_payload=True,
        with_vectors=False,
        search_params=models.SearchParams(hnsw_ef=hnsw_ef, exact=False),
    )
    result_ids = []
    for point in response.points:
        bid = str((point.payload or {}).get("business_id", point.id))
        result_ids.append(bid)
    return result_ids


def run_faiss_search(query_embeddings: np.ndarray, faiss_backup: Tuple) -> Tuple[List[List[str]], List[float]]:
    """执行 FAISS 全量检索，返回 (每条 query 的 business_id 列表, 延迟ms 列表)"""
    faiss_index, faiss_metadata, _ = faiss_backup

    # 预热（不计入延迟统计）
    for i in range(min(WARMUP_ROUNDS, len(query_embeddings))):
        faiss_index.search(query_embeddings[i:i + 1], SEARCH_TOP_K)

    all_results = []
    all_latencies = []
    for i in range(len(query_embeddings)):
        query_np = query_embeddings[i:i + 1]
        t0 = time.perf_counter()
        results = search_faiss_single(faiss_index, faiss_metadata, query_np, SEARCH_TOP_K)
        t1 = time.perf_counter()
        all_results.append(results)
        all_latencies.append((t1 - t0) * 1000)

    logger.info("FAISS 检索完成: %s 条 query，平均延迟 %.3f ms",
                len(all_results), float(np.mean(all_latencies)))
    return all_results, all_latencies


def run_qdrant_search(qdrant_client, query_embeddings: np.ndarray, hnsw_ef: int) -> Tuple[List[List[str]], List[float]]:
    """执行 Qdrant 全量检索（指定 hnsw_ef），返回 (每条 query 的 business_id 列表, 延迟ms 列表)"""
    # 预热
    for i in range(min(WARMUP_ROUNDS, len(query_embeddings))):
        search_qdrant_single(qdrant_client, query_embeddings[i].tolist(), SEARCH_TOP_K, hnsw_ef)

    all_results = []
    all_latencies = []
    for i in range(len(query_embeddings)):
        query_vec = query_embeddings[i].tolist()
        t0 = time.perf_counter()
        results = search_qdrant_single(qdrant_client, query_vec, SEARCH_TOP_K, hnsw_ef)
        t1 = time.perf_counter()
        all_results.append(results)
        all_latencies.append((t1 - t0) * 1000)

    return all_results, all_latencies


# ========== Step 5: 评估指标计算 ==========

def compute_recall_at_k(retrieved_ids: List[str], gt_ids: List[str], k: int) -> float:
    """Recall@K = |retrieved_top_K 与 GT_top_K 的交集| / K

    ANN 基准标准口径：衡量近似检索相对暴力精确检索（GT）的保留率。
    GT 为 Flat Index 暴力搜索的 Top-20，此处与检索侧取相同截断的前 K 条，
    精确搜索场景下各 K 值 recall 应恒为 1.0（可作为 ef 是否生效的判据）。
    """
    retrieved_set = set(retrieved_ids[:k])
    gt_set = set(gt_ids[:k])
    if not retrieved_set:
        return 0.0
    return len(retrieved_set & gt_set) / k


def compute_mrr(retrieved_ids: List[str], gt_ids: List[str]) -> float:
    """MRR: 第一个命中 GT top-K（K=SEARCH_TOP_K）的结果的倒数排名（1/rank）"""
    gt_set = set(gt_ids[:SEARCH_TOP_K])
    for i, rid in enumerate(retrieved_ids):
        if rid in gt_set:
            return 1.0 / (i + 1)
    return 0.0


def compute_jaccard(list_a: List[str], list_b: List[str], k: int) -> float:
    """Jaccard 重叠度（Top-K）：|A 交 B| / |A 并 B|"""
    set_a = set(list_a[:k])
    set_b = set(list_b[:k])
    union = set_a | set_b
    if not union:
        return 0.0
    return len(set_a & set_b) / len(union)


def compute_latency_stats(latencies_ms: List[float]) -> Dict:
    """计算延迟统计：P50 / P95 / P99 / 均值（毫秒）"""
    arr = np.array(latencies_ms)
    return {
        "p50_ms": round(float(np.percentile(arr, 50)), 3),
        "p95_ms": round(float(np.percentile(arr, 95)), 3),
        "p99_ms": round(float(np.percentile(arr, 99)), 3),
        "mean_ms": round(float(np.mean(arr)), 3),
    }


def compute_metrics_for_system(
    retrieved_per_query: List[List[str]],
    latencies_ms: List[float],
    gt: Dict,
    faiss_results: Optional[List[List[str]]] = None,
) -> Dict:
    """为某个系统（FAISS 或 Qdrant 某 ef 档位）计算全部指标

    指标包括：Recall@1/3/5/10、MRR、与 FAISS 的 Jaccard（Top-5/10）、延迟统计。
    """
    query_keys = sorted(gt.keys(), key=lambda x: int(x[1:]) if x[1:].isdigit() else 0)
    n = len(query_keys)

    recalls = {1: [], 3: [], 5: [], 10: []}
    mrrs = []
    jaccard_5_list = []
    jaccard_10_list = []

    for i, qkey in enumerate(query_keys):
        gt_ids = gt[qkey]["gt_ids"]
        retrieved = retrieved_per_query[i] if i < len(retrieved_per_query) else []

        for k in recalls:
            recalls[k].append(compute_recall_at_k(retrieved, gt_ids, k))
        mrrs.append(compute_mrr(retrieved, gt_ids))

        if faiss_results is not None:
            faiss_ret = faiss_results[i] if i < len(faiss_results) else []
            jaccard_5_list.append(compute_jaccard(retrieved, faiss_ret, 5))
            jaccard_10_list.append(compute_jaccard(retrieved, faiss_ret, 10))

    metrics = {
        "recall@1": round(float(np.mean(recalls[1])), 4),
        "recall@3": round(float(np.mean(recalls[3])), 4),
        "recall@5": round(float(np.mean(recalls[5])), 4),
        "recall@10": round(float(np.mean(recalls[10])), 4),
        "mrr": round(float(np.mean(mrrs)), 4),
    }

    # Jaccard 仅对 Qdrant（与 FAISS 对比）计算
    if faiss_results is not None:
        metrics["jaccard_top5_vs_faiss"] = round(float(np.mean(jaccard_5_list)), 4)
        metrics["jaccard_top10_vs_faiss"] = round(float(np.mean(jaccard_10_list)), 4)

    metrics["latency"] = compute_latency_stats(latencies_ms)
    return metrics


# ========== Step 6: 推荐 ef 选择 ==========

def recommend_best_ef(qdrant_metrics_by_ef: Dict) -> Dict:
    """推荐最优 hnsw_ef：在 recall@10 达到最高值 99% 阈值的前提下选最小 ef

    策略：先找所有 ef 中 recall@10 的最大值，然后找达到该值 99% 的最小 ef，
    兼顾召回质量与搜索效率。
    """
    if not qdrant_metrics_by_ef:
        return {"best_ef": None, "reason": "无 Qdrant 指标数据"}

    max_recall = max(m["recall@10"] for m in qdrant_metrics_by_ef.values())
    threshold = max_recall * 0.99

    candidates = [
        ef for ef, m in sorted(qdrant_metrics_by_ef.items())
        if m["recall@10"] >= threshold
    ]
    best_ef = candidates[0] if candidates else None

    reason = (
        "最高 recall@10=%.4f，99%% 阈值=%.4f，"
        "满足阈值的最小 ef=%s" % (max_recall, threshold, best_ef)
    )
    return {"best_ef": best_ef, "reason": reason}


def validate_experiment_scale(vector_count: int, query_count: int) -> Dict:
    """校验实验规模是否足以支撑 ef 调优的统计意义

    本地持久化模式（未设 QDRANT_URL）任何规模都退化为精确暴力搜索，ef 不生效；
    服务端模式需向量数超过 HNSW 建图阈值（bench collection 建库时 indexing_threshold=2000，
    按 1.5 倍余量取 3000）ef 才会真实作用在 HNSW 图上；
    样本过少时 recall 指标随机波动大，无法稳定排序 ef 档位。

    Args:
        vector_count: Qdrant collection 当前向量总数
        query_count:  本次测试 query 数量

    Returns:
        dict: {valid: bool, warning: str}
    """
    warnings = []
    server_mode = bool(os.getenv("QDRANT_URL"))
    if not server_mode:
        warnings.append(
            "未设置 QDRANT_URL（本地持久化模式），Qdrant 退化为精确暴力搜索，"
            "ef 参数不生效，recall 各档位无统计差异"
        )
    elif vector_count < 750:
        warnings.append(
            f"向量数 {vector_count} < 750，低于 bench collection HNSW 建图阈值（indexing_threshold=500）"
            f"的 1.5 倍安全余量，ef 调优结果无统计意义"
        )
    if query_count < 30:
        warnings.append(
            f"测试 query 数 {query_count} < 30，样本过少，recall 指标随机波动大，"
            f"无法稳定排序 ef 档位"
        )
    if not warnings:
        return {"valid": True, "warning": ""}
    return {"valid": False, "warning": "；".join(warnings)}


# ========== 主流程 ==========

def main():
    print("=" * 60)
    print("  Qdrant vs FAISS 召回质量离线对比实验")
    print("=" * 60)

    faiss = _import_faiss()
    os.makedirs(STATS_DIR, exist_ok=True)

    # Step 1: 生成测试 query
    print("\n[Step 1] 生成测试 query...")
    queries = generate_test_queries()

    # Step 2: 批量计算 embedding（全量缓存，后续所有检索共用）
    print("\n[Step 2] 批量计算 query embedding...")
    query_embeddings = compute_query_embeddings(queries)

    # Step 3: 构建 Ground Truth（FAISS IndexFlatL2 暴力基准）
    print("\n[Step 3] 构建 Ground Truth (FAISS IndexFlatL2 暴力基准)...")
    qdrant_client = get_qdrant_client()
    total_vectors = qdrant_client.count(collection_name=BENCH_COLLECTION_NAME).count
    logger.info("Qdrant collection '%s' 当前共 %s 条记录", BENCH_COLLECTION_NAME, total_vectors)

    # 规模校验：小规模下 ef 调优无统计意义，仅打印警告不中断
    scale_check = validate_experiment_scale(total_vectors, len(queries))
    if not scale_check["valid"]:
        logger.warning("=" * 60)
        logger.warning("实验规模警告：%s", scale_check["warning"])
        logger.warning("将跳过 ef 网格差异详情，直接推荐 ef=%s（待复测）", QDRANT_SEARCH_EF)
        logger.warning("=" * 60)

    gt = build_ground_truth(qdrant_client, queries, query_embeddings)

    # Step 4: 加载 FAISS backup 索引
    print("\n[Step 4] 加载 FAISS backup 索引...")
    faiss_backup = load_faiss_backup()

    # Step 5: FAISS 检索
    faiss_results = None
    faiss_latencies = []
    faiss_index_type = None
    if faiss_backup is not None:
        print("\n[Step 5] FAISS 检索 (Top-%s)..." % SEARCH_TOP_K)
        faiss_index_type = faiss_backup[2]
        faiss_results, faiss_latencies = run_faiss_search(query_embeddings, faiss_backup)
        # ID 命名空间校验：旧 backup metadata 的 id 与当前 collection 的 business_id
        # 可能不是同一套（历史遗留），零重叠时对照无意义，跳过避免输出全零噪声
        gt_id_sample = set()
        for qkey in sorted(gt.keys())[:20]:
            gt_id_sample.update(gt[qkey]["gt_ids"])
        faiss_id_sample = set()
        for res in faiss_results[:20]:
            faiss_id_sample.update(res)
        if gt_id_sample and not (gt_id_sample & faiss_id_sample):
            logger.warning("FAISS backup 与 collection '%s' 的 ID 命名空间零重叠，跳过 FAISS 对照",
                           BENCH_COLLECTION_NAME)
            faiss_results = None
    else:
        logger.warning("FAISS backup 不存在，跳过 FAISS 检索（Jaccard 指标也将缺失）")

    # Step 6: Qdrant 检索（多 hnsw_ef 档位）
    print("\n[Step 6] Qdrant 检索（测试 %s 个 hnsw_ef 档位）..." % len(HNSW_EF_LIST))
    qdrant_results_by_ef = {}
    qdrant_latencies_by_ef = {}
    for ef in HNSW_EF_LIST:
        logger.info("  正在检索 hnsw_ef=%s ...", ef)
        results_list, latencies = run_qdrant_search(qdrant_client, query_embeddings, ef)
        qdrant_results_by_ef[ef] = results_list
        qdrant_latencies_by_ef[ef] = latencies
        logger.info("  hnsw_ef=%s 完成，平均延迟 %.3f ms", ef, float(np.mean(latencies)))

    # Step 7: 计算评估指标
    print("\n[Step 7] 计算评估指标...")
    report = {
        "metadata": {
            "total_queries": len(queries),
            "total_vectors": total_vectors,
            "collection": BENCH_COLLECTION_NAME,
            "vector_dim": QDRANT_VECTOR_SIZE,
            "gt_top_k": GT_TOP_K,
            "search_top_k": SEARCH_TOP_K,
            "faiss_nprobe": FAISS_NPROBE,
            "hnsw_ef_list": HNSW_EF_LIST,
            "scale_valid": scale_check["valid"],
            "scale_warning": scale_check["warning"],
            "timestamp": datetime.now().isoformat(),
        },
    }

    # FAISS 指标
    if faiss_results is not None:
        faiss_metrics = compute_metrics_for_system(
            faiss_results, faiss_latencies, gt, faiss_results=None
        )
        faiss_metrics["index_type"] = faiss_index_type
        report["faiss"] = faiss_metrics
        logger.info("FAISS 指标: recall@5=%.4f, recall@10=%.4f, mrr=%.4f, p99=%.3f ms",
                     faiss_metrics["recall@5"], faiss_metrics["recall@10"],
                     faiss_metrics["mrr"], faiss_metrics["latency"]["p99_ms"])

    # Qdrant 指标（每个 ef 档位）
    qdrant_section = {}
    qdrant_metrics_by_ef = {}
    for ef in HNSW_EF_LIST:
        results = qdrant_results_by_ef[ef]
        latencies = qdrant_latencies_by_ef[ef]
        metrics = compute_metrics_for_system(
            results, latencies, gt,
            faiss_results=faiss_results,  # 传入 FAISS 结果用于 Jaccard 计算
        )
        ef_key = "ef_%s" % ef
        qdrant_section[ef_key] = metrics
        qdrant_metrics_by_ef[ef_key] = metrics
        logger.info("Qdrant ef=%s: recall@5=%.4f, recall@10=%.4f, mrr=%.4f, "
                     "jaccard5=%.4f, p99=%.3f ms",
                     ef, metrics["recall@5"], metrics["recall@10"], metrics["mrr"],
                     metrics.get("jaccard_top5_vs_faiss", 0), metrics["latency"]["p99_ms"])

    report["qdrant"] = qdrant_section

    # 推荐 ef
    report["recommendation"] = recommend_best_ef(qdrant_metrics_by_ef)
    # 规模不足时各 ef 结果无差异，强制采用默认 ef 并标注待复测
    if not scale_check["valid"]:
        report["recommendation"] = {
            "best_ef": "ef_%s" % QDRANT_SEARCH_EF,
            "reason": "规模不足（%s），各 ef 结果一致，采用默认 ef=%s（待复测）" % (
                scale_check["warning"], QDRANT_SEARCH_EF),
        }
    # 规模与推荐信息回填 metadata，便于离线报告溯源
    report["metadata"]["recommendation"] = report["recommendation"]
    logger.info("推荐: %s", report["recommendation"])

    # 保存报告
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    logger.info("已保存召回对比报告 -> %s", REPORT_FILE)

    # 打印摘要
    print("\n" + "=" * 60)
    print("  实验完成 - 结果摘要")
    print("=" * 60)
    print("  测试 query 数量: %s" % len(queries))
    print("  向量总数: %s" % total_vectors)
    print("  Ground Truth Top-K: %s" % GT_TOP_K)
    print("  检索 Top-K: %s" % SEARCH_TOP_K)
    if not scale_check["valid"]:
        # 规模不足：跳过 ef 网格差异详情，直接给出简化结论
        print("  [规模警告] %s" % scale_check["warning"])
        print("  当前规模下所有 ef 档位 recall 一致，推荐 ef=%s（待复测）" % QDRANT_SEARCH_EF)
    else:
        if "faiss" in report:
            fm = report["faiss"]
            print("  FAISS(%s): recall@5=%.4f, recall@10=%.4f, mrr=%.4f, p99=%.3f ms" % (
                fm.get("index_type", "?"), fm["recall@5"], fm["recall@10"],
                fm["mrr"], fm["latency"]["p99_ms"]))
        for ef in HNSW_EF_LIST:
            m = qdrant_section["ef_%s" % ef]
            print("  Qdrant ef=%-3s: recall@5=%.4f, recall@10=%.4f, mrr=%.4f, p99=%.3f ms" % (
                ef, m["recall@5"], m["recall@10"], m["mrr"], m["latency"]["p99_ms"]))
    print("  推荐 ef: %s" % report["recommendation"]["best_ef"])
    print("  报告文件: %s" % REPORT_FILE)
    print("=" * 60)


if __name__ == "__main__":
    main()
