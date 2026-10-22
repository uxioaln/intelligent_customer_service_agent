# -*- coding: utf-8 -*-
"""
app.core.retrieval - 跨模态混合检索管线

三段式检索改造：
1. 查询改写：LLM 将口语化查询改写为检索友好形式
2. 混合检索：向量检索 + BM25 关键词检索，RRF 融合取 top-20
3. 重排序：AGICTO cross-encoder rerank 模型精排，取 top-5

设计要点：
- 纯函数化，QdrantClient 通过参数注入，BM25 索引通过全局单例访问
- 复用 app.core.query 的 embedding / search_vectors / distance_to_similarity
- 复用 app.core.cost_tracker 进行成本统计
- rerank 走 AGICTO /v1/rerank 接口（Cohere 风格，非 OpenAI 兼容），用 requests 直调
"""
import logging
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import requests

from app.config import (
    AGICTO_API_KEY,
    CHAT_BASE_URL,
    CHAT_MODEL,
    RERANK_MODEL,
)
from app.core import query
from app.core.cost_tracker import get_global_stats, tracked_chat_completion

logger = logging.getLogger(__name__)

# ========== 可调参数 ==========

# RRF 融合参数 k（标准值 60，越大对排名差异越平滑）
RRF_K = 60

# 混合检索召回数量（融合后取 top-N 送入 rerank）
HYBRID_RECALL_TOP = 20

# rerank 后最终保留数量
RERANK_TOP = 5

# AGICTO rerank 接口地址
RERANK_URL = CHAT_BASE_URL.rstrip("/") + "/rerank"

# 混合意图查询的默认模态配额：保证 top-k 中目标模态至少占 N 条
# 图片意图默认至少 2 条图片，视频意图默认至少 1 条视频
DEFAULT_IMAGE_QUOTA = 2
DEFAULT_VIDEO_QUOTA = 1

# P2-1 媒体定向检索通道：存在模态配额时，每个目标模态从全库向量结果中
# 至少取 top-N 附加到融合候选池，保证配额补入有候选可用
DIRECT_RECALL_TOP = 3


# ========== 查询改写 ==========

def rewrite_query(query_str: str) -> str:
    """用 LLM 将口语化查询改写为检索友好形式

    保留原意，补充关键词，去除口语表达，提升向量与 BM25 召回率。
    改写失败时回退到原查询（外部 API 边界容错，非架构 fallback）。

    Args:
        query_str: 原始用户查询

    Returns:
        改写后的检索式；失败时返回原查询
    """
    prompt = f"""请将以下用户口语化问题改写为适合知识库检索的形式。
要求：
1. 保留原意，不要回答问题
2. 补充关键词，去除口语表达（如"我想了解一下"、"麻烦问下"等）
3. 输出简洁的检索式，不超过 50 字
4. 直接输出改写结果，不要加任何前缀说明

用户问题：{query_str}
改写结果："""

    try:
        completion = tracked_chat_completion(
            client=query.client,
            model=CHAT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            source="查询改写",
        )
        rewritten = completion.choices[0].message.content.strip()
        if not rewritten:
            return query_str
        logger.info("查询改写: '%s' -> '%s'", query_str, rewritten)
        return rewritten
    except Exception as e:
        logger.warning("查询改写失败，回退原查询: %s", e)
        return query_str


# ========== BM25 关键词检索 ==========

class BM25Index:
    """基于 rank_bm25 的内存 BM25 索引

    从 metadata 列表构建，使用 jieba 中文分词。
    索引内容为 metadata 的 content 字段，覆盖文本/图片/视频/多样化问题全部条目。
    """

    def __init__(self):
        self._bm25 = None
        self._docs: List[Dict] = []
        self._tokenized: List[List[str]] = []

    def build(self, metadata_list: List[Dict]) -> None:
        """从 metadata 列表构建 BM25 索引

        Args:
            metadata_list: 内部 metadata 结构列表（由 scroll_all_metadata 产出）
        """
        import jieba
        from rank_bm25 import BM25Okapi

        self._docs = []
        self._tokenized = []
        for m in metadata_list:
            if m.get("deleted", False):
                continue
            content = m.get("content", "")
            if not content or not content.strip():
                continue
            self._docs.append(m)
            # jieba.cut 返回生成器，转 list 得到分词结果
            self._tokenized.append(list(jieba.cut(content)))

        if self._tokenized:
            self._bm25 = BM25Okapi(self._tokenized)
        else:
            self._bm25 = None
        logger.info("BM25 索引已构建: %s 篇文档", len(self._docs))

    def search(self, query_str: str, top_k: int = 20) -> List[Tuple[float, Dict]]:
        """BM25 检索

        Args:
            query_str: 查询文本
            top_k: 返回前 top_k 条结果

        Returns:
            [(bm25_score, metadata), ...] 按 bm25_score 降序
        """
        import jieba

        if self._bm25 is None or not self._docs:
            return []

        tokens = list(jieba.cut(query_str))
        scores = self._bm25.get_scores(tokens)

        # 按分数降序取 top-k
        import numpy as np
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] <= 0:
                continue
            results.append((float(scores[idx]), self._docs[idx]))
        return results


# 全局 BM25 索引单例（由 state.load_resources 初始化）
_bm25_index: Optional[BM25Index] = None


def get_bm25_index() -> Optional[BM25Index]:
    """获取全局 BM25 索引实例"""
    return _bm25_index


def set_bm25_index(index: BM25Index) -> None:
    """设置全局 BM25 索引实例（state.load_resources 调用）"""
    global _bm25_index
    _bm25_index = index


def build_bm25_from_metadata(metadata_list: List[Dict]) -> BM25Index:
    """从 metadata 列表构建并设置全局 BM25 索引

    Args:
        metadata_list: 内部 metadata 结构列表

    Returns:
        构建完成的 BM25Index 实例
    """
    index = BM25Index()
    index.build(metadata_list)
    set_bm25_index(index)
    return index


# ========== RRF 融合 ==========

def rrf_fusion(
    vector_results: List[Tuple[float, str, Dict]],
    bm25_results: List[Tuple[float, Dict]],
    k: int = RRF_K,
) -> List[Tuple[float, Dict]]:
    """RRF (Reciprocal Rank Fusion) 融合向量与 BM25 排序

    公式：rrf_score = sum(1 / (k + rank)) 对每个检索器的排名累加

    Args:
        vector_results: [(distance, business_id, metadata), ...]
                       向量检索结果，按 distance 升序（distance 越小越相似）
        bm25_results: [(score, metadata), ...]
                      BM25 检索结果，按 score 降序（score 越大越相关）
        k: RRF 平滑参数，默认 60

    Returns:
        [(rrf_score, metadata), ...] 按 rrf_score 降序
    """
    # 按 point id 聚合 RRF 分数，同时记录对应的 metadata 与向量距离
    scores: Dict[Any, float] = {}
    metadata_map: Dict[Any, Dict] = {}
    distance_map: Dict[Any, Optional[float]] = {}

    # 向量结果排名：distance 越小排名越靠前，rank 从 1 开始
    for rank, (dist, _, m) in enumerate(vector_results, 1):
        pid = m.get("id")
        if pid is None:
            continue
        scores[pid] = scores.get(pid, 0.0) + 1.0 / (k + rank)
        metadata_map[pid] = m
        distance_map[pid] = float(dist)

    # BM25 结果排名：score 越大排名越靠前，rank 从 1 开始
    for rank, (_, m) in enumerate(bm25_results, 1):
        pid = m.get("id")
        if pid is None:
            continue
        scores[pid] = scores.get(pid, 0.0) + 1.0 / (k + rank)
        if pid not in metadata_map:
            metadata_map[pid] = m
            # BM25-only 命中无向量距离，保持 None
            distance_map[pid] = None

    # 按 rrf_score 降序输出，将 rrf_score 与 vector_distance 附加到 metadata 副本
    sorted_pids = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    results = []
    for pid, score in sorted_pids:
        m = metadata_map[pid].copy()
        m["rrf_score"] = score
        m["vector_distance"] = distance_map.get(pid)
        results.append((score, m))
    return results


# ========== Rerank 精排 ==========

def rerank(query_str: str, candidates: List[Dict], top_n: int = RERANK_TOP) -> List[Dict]:
    """使用 AGICTO cross-encoder rerank 模型精排

    调用 AGICTO /v1/rerank 接口（Cohere 风格），对候选文档按查询相关性重排。

    Args:
        query_str: 查询文本
        candidates: 候选 metadata 列表
        top_n: 精排后保留前 top_n 条

    Returns:
        重排后的 metadata 列表（附加 rerank_score 字段），长度 <= top_n
    """
    if not candidates:
        return []

    documents = [m.get("content", "") for m in candidates]
    # 去除空文档，避免接口报错
    if not any(doc.strip() for doc in documents):
        return candidates[:top_n]

    payload = {
        "model": RERANK_MODEL,
        "query": query_str,
        "top_n": min(top_n, len(candidates)),
        "documents": documents,
    }
    headers = {
        "Authorization": f"Bearer {AGICTO_API_KEY}",
        "Content-Type": "application/json",
    }

    start_ts = time.time()
    try:
        resp = requests.post(RERANK_URL, json=payload, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.warning("rerank 调用失败，回退融合排序: %s", e)
        return candidates[:top_n]

    # API 可能返回 200 OK 但 body 为 null
    if data is None:
        logger.warning("rerank 返回空响应体，回退融合排序")
        return candidates[:top_n]

    duration = round(time.time() - start_ts, 3)

    # 解析 results: [{index, relevance_score}, ...]
    rerank_results = data.get("results", []) if isinstance(data, dict) else []
    if not rerank_results:
        logger.warning("rerank 返回空结果，回退融合排序")
        return candidates[:top_n]

    reranked = []
    for item in rerank_results:
        idx = item.get("index")
        score = item.get("relevance_score", 0.0)
        if idx is None or idx < 0 or idx >= len(candidates):
            continue
        m = candidates[idx].copy()
        m["rerank_score"] = float(score)
        reranked.append(m)

    # 记录成本（按 total_tokens 统计，rerank 通常无 completion_tokens）
    usage = data.get("usage", {}) or {}
    total_tokens = usage.get("total_tokens", 0)
    estimated_cost = _estimate_rerank_cost(total_tokens)
    try:
        get_global_stats().record_call(
            source="重排序",
            duration_seconds=duration,
            prompt_tokens=total_tokens,
            completion_tokens=0,
            estimated_cost=estimated_cost,
        )
    except Exception:
        logger.debug("rerank 成本记录失败", exc_info=True)

    logger.info("rerank 完成: %s -> %s 条 (耗时 %ss)", len(candidates), len(reranked), duration)
    return reranked


def _estimate_rerank_cost(total_tokens: int) -> float:
    """估算 rerank 调用成本（单位：元）

    rerank 模型按 token 计费，暂按默认单价估算。
    """
    # 按 0.0001 元/1K tokens 估算，后续如有实际单价可调整
    return round(total_tokens * 0.0001 / 1000, 6)


# ========== 模态配额 ==========

def _apply_modality_quota(
    reranked: List[Dict],
    fused_candidates: List[Dict],
    modality_quota: Dict[str, int],
    k: int,
) -> List[Dict]:
    """在 rerank 精排后应用模态配额，保证 top-k 中目标模态至少占 N 条

    解决问题：向量库中文本条目占多数（如 57 条中文本占大多数），混合意图查询时
    图片/视频条目会被多数文本条目挤出 top-k，导致目标模态召回缺失。

    策略：
    1. 统计当前 reranked 结果中各模态的条目数
    2. 若某模态条目数 < 配额 N，从 fused_candidates 池中按 rrf_score 降序
       补入该模态条目，替换 reranked 末尾非配额模态的条目
    3. 已在结果中的条目不重复补入

    Args:
        reranked: rerank 精排后的结果列表（top-k）
        fused_candidates: RRF 融合后的候选池（含 rrf_score，供配额补入使用）
        modality_quota: {modality: min_n}，如 {"image": 2} 表示图片至少 2 条
        k: 最终结果长度上限

    Returns:
        应用配额后的结果列表，长度 <= k
    """
    if not modality_quota:
        return reranked

    result = list(reranked[:k])
    existing_ids = {m.get("id") for m in result}

    for modality, min_n in modality_quota.items():
        current_count = sum(1 for m in result if m.get("type") == modality)
        if current_count >= min_n:
            continue

        needed = min_n - current_count
        # 从融合候选池中挑选该模态、且未在结果中的条目，按 rrf_score 降序
        pool = sorted(
            [m for m in fused_candidates
             if m.get("type") == modality
             and m.get("id") not in existing_ids
             and not m.get("deleted", False)],
            key=lambda m: m.get("rrf_score", 0.0),
            reverse=True,
        )

        for cand in pool[:needed]:
            # 替换结果末尾第一个非目标模态的条目，为配额条目腾出位置
            replaced_idx = None
            for i in range(len(result) - 1, -1, -1):
                if result[i].get("type") != modality:
                    replaced_idx = i
                    break
            if replaced_idx is None:
                # 末尾没有可替换的非目标模态条目，直接追加（可能超出 k，后续截断）
                result.append(cand)
            else:
                replaced = result.pop(replaced_idx)
                existing_ids.discard(replaced.get("id"))
                # 在替换位置插入配额条目，保持整体顺序
                result.insert(replaced_idx, cand)
            existing_ids.add(cand.get("id"))

    # 截断到 k
    return result[:k]


def hybrid_retrieve(
    query_str: str,
    qdrant_client: Any,
    k: int = RERANK_TOP,
    allowed_types: Optional[Set[str]] = None,
    enable_rewrite: bool = True,
    enable_rerank: bool = True,
    modality_quota: Optional[Dict[str, int]] = None,
) -> List[Dict]:
    """跨模态混合检索完整管线

    流程：查询改写 -> 向量检索(top-20) + BM25检索(top-20) -> 类型过滤 -> RRF融合 -> top-20 -> rerank精排 -> 模态配额 -> top-k

    Args:
        query_str: 原始用户查询
        qdrant_client: QdrantClient 实例
        k: 最终返回条数，默认 5
        allowed_types: 候选类型过滤集合；None 表示不过滤
                       文本检索用 {"text","diverse_question"}，图片 {"image"}，视频 {"video"}
        enable_rewrite: 是否启用查询改写，默认 True
        enable_rerank: 是否启用 rerank 精排，默认 True
        modality_quota: 模态配额，如 {"image": 2} 表示 top-k 中图片至少 2 条；
                       None 表示不配额；仅在 allowed_types 包含多模态时生效

    Returns:
        排序后的 metadata 列表，长度 <= k
    """
    # 1. 查询改写
    if enable_rewrite:
        search_query = rewrite_query(query_str)
    else:
        search_query = query_str

    # 2. 向量检索（复用 query.search_vectors，返回 (distance, business_id, metadata)）
    query_vec = query.get_text_embedding(search_query)
    # P2-1：存在模态配额时向量召回深度扩至全库，保证语义距离较远的目标模态条目
    # （如文本意图查询下的视频）也能进入候选池；无配额时维持 top-40 深度不变
    vector_top_k = None if modality_quota else HYBRID_RECALL_TOP * 2
    vector_results_all = query.search_vectors(query_vec, qdrant_client, top_k=vector_top_k)

    # 类型过滤：仅保留 allowed_types 内的条目
    if allowed_types is not None:
        vector_results_all = [
            (dist, bid, m) for dist, bid, m in vector_results_all
            if m.get("type") in allowed_types and not m.get("deleted", False)
        ]
    else:
        vector_results_all = [
            (dist, bid, m) for dist, bid, m in vector_results_all
            if not m.get("deleted", False)
        ]

    # 主通道保持 top-40 深度参与 RRF 融合，避免全库低分条目稀释融合排序
    vector_results = vector_results_all[:HYBRID_RECALL_TOP * 2]

    # 3. BM25 检索
    bm25_results: List[Tuple[float, Dict]] = []
    bm25 = get_bm25_index()
    if bm25 is not None:
        bm25_results = bm25.search(search_query, top_k=HYBRID_RECALL_TOP * 2)
        # 类型过滤
        if allowed_types is not None:
            bm25_results = [
                (score, m) for score, m in bm25_results
                if m.get("type") in allowed_types and not m.get("deleted", False)
            ]
        else:
            bm25_results = [
                (score, m) for score, m in bm25_results
                if not m.get("deleted", False)
            ]

    # 4. RRF 融合
    fused = rrf_fusion(vector_results, bm25_results, k=RRF_K)

    # P2-1 媒体定向检索通道：对配额模态从全量向量结果中取 top-N 附加到融合列表尾部，
    # 保证 _apply_modality_quota 有候选可补。定向条目 rrf_score=0 排在自然召回条目之后，
    # 仅在自然召回中该模态条目不足时才会被配额补入；vector_distance 保留真实检索距离，
    # 下游距离阈值过滤（如 video_search 的 MEDIA_DISTANCE_THRESHOLD）行为不受影响
    if modality_quota:
        fused_ids = {m.get("id") for _, m in fused}
        for modality, min_n in modality_quota.items():
            direct_limit = max(min_n, DIRECT_RECALL_TOP)
            direct_added = 0
            for dist, _, m in vector_results_all:
                if m.get("type") != modality:
                    continue
                pid = m.get("id")
                if pid is None or pid in fused_ids:
                    continue
                mc = m.copy()
                mc["rrf_score"] = 0.0
                mc["vector_distance"] = float(dist)
                fused.append((0.0, mc))
                fused_ids.add(pid)
                direct_added += 1
                if direct_added >= direct_limit:
                    break
            if direct_added:
                logger.info("定向通道补入 %s 模态 %s 条", modality, direct_added)

    # 融合后取 top-20 送入 rerank
    candidates = [m for _, m in fused[:HYBRID_RECALL_TOP]]

    # 5. rerank 精排
    if enable_rerank and len(candidates) > 1:
        reranked = rerank(search_query, candidates, top_n=k)
    else:
        # 未启用 rerank 时按 RRF 分数顺序取 top-k
        reranked = candidates[:k]

    # 6. 模态配额：保证 top-k 中目标模态至少占 N 条，防止图片/视频被多数文本条目挤出
    # 候选池扩展为全量融合列表，确保低排名的视频/图片向量也能进入配额补入流程
    if modality_quota:
        all_fused = [m for _, m in fused]
        reranked = _apply_modality_quota(reranked, all_fused, modality_quota, k)

    logger.info(
        "混合检索完成: query='%s' (改写='%s') -> 向量 %s 条 + BM25 %s 条 -> 融合 %s 条 -> %s 条结果 (配额=%s)",
        query_str, search_query, len(vector_results), len(bm25_results),
        len(candidates), len(reranked), modality_quota or "无",
    )
    return reranked
