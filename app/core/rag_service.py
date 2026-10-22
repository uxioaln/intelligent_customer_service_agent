# -*- coding: utf-8 -*-
"""
app.core.rag_service - RAG 问答业务逻辑（纯函数化）

设计要点：
- 不读取任何 module-level globals
- qdrant_client / metadata 作为参数注入，方便单测和未来切换数据源
- 复用 app.core.query 中的 embedding / 检索辅助 / LLM client
"""
from typing import Any, Dict, List

from app.config import MEDIA_DISTANCE_THRESHOLD
from app.core import query
from app.core.cost_tracker import tracked_chat_completion


# 类型别名：Qdrant 客户端 + 元数据列表，作为 RAG 检索的输入
RagClient = Any
RagMetadata = List[Dict[str, Any]]


def rag_ask_api(
    query_str: str,
    qdrant_client: RagClient,
    metadata: RagMetadata,
    k: int = 3,
) -> Dict[str, Any]:
    """
    RAG 问答接口实现，复用 app.core.query 的检索/embedding/LLM 逻辑
    返回结构化 dict：{answer, image_path, video_url, references}

    Args:
        query_str: 用户问题文本
        qdrant_client: QdrantClient 实例
        metadata: 元数据列表（保留参数兼容，检索结果已含 metadata）
        k: top-k 文本检索数量

    Returns:
        dict: {
            "answer": str,
            "image_path": Optional[str],
            "video_url": Optional[str],
            "references": List[Dict[str, Any]],
        }
    """
    # 1. 获取 query 向量并执行 Qdrant 全量检索
    query_vec = query.get_text_embedding(query_str)
    raw_results = query.search_vectors(query_vec, qdrant_client)

    # 2. 转换为内部统一的结果格式
    results = []
    for dist, business_id, m in raw_results:
        if m.get("deleted", False):
            continue
        sim = query.distance_to_similarity(dist)
        results.append({
            "idx": m.get("id"),
            "distance": float(dist),
            "similarity": float(sim),
            "metadata": m,
        })

    # 3. 媒体意图检测
    want_image, want_video = query.detect_media_intent(query_str)

    # 4. 取 top-k 文本结果
    top_results = [r for r in results if r["metadata"]["type"] == "text"][:k]

    # 5. 匹配图片（距离<threshold 中距离最小Top1）
    matched_image = None
    if want_image:
        image_results = [r for r in results if r["metadata"]["type"] == "image" and r["distance"] < MEDIA_DISTANCE_THRESHOLD]
        if image_results:
            image_results.sort(key=lambda x: x["distance"])
            matched_image = image_results[0]

    # 6. 匹配视频
    matched_video = None
    if want_video:
        video_results = [r for r in results if r["metadata"]["type"] == "video" and r["distance"] < MEDIA_DISTANCE_THRESHOLD]
        if video_results:
            video_results.sort(key=lambda x: x["distance"])
            matched_video = video_results[0]

    # 7. 构建 context 并调用 LLM（复用 app.core.query 的 client）
    context_str = ""
    for i, r in enumerate(top_results):
        m = r["metadata"]
        context_str += f"背景知识 {i+1} (来源: {m['source']}, 相似度: {r['similarity']:.4f}):\n{m['content']}\n\n"

    prompt = f"""你是一个迪士尼客服助手。请根据以下背景知识回答用户问题。

[背景知识]
{context_str}
[用户问题]
{query_str}
"""
    # 走 tracked_chat_completion：自带空响应防护 + 指数退避重试，避免网关瞬时故障导致服务崩溃
    completion = tracked_chat_completion(
        client=query.client,
        model=query.CHAT_MODEL,
        messages=[
            {"role": "system", "content": "你是一个迪士尼客服助手。"},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        source="rag_ask_api",
    )
    answer = completion.choices[0].message.content

    return {
        "answer": answer,
        "image_path": matched_image["metadata"]["path"] if matched_image else None,
        "video_url": matched_video["metadata"]["url"] if matched_video else None,
        "references": [
            {
                "source": r["metadata"]["source"],
                "similarity": round(r["similarity"], 4),
                "content": r["metadata"]["content"]
            } for r in top_results
        ]
    }
