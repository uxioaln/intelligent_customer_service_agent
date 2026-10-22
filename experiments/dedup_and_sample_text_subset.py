"""
对 bench_queries_500.json 中的 498 条查询做语义去重 + 主题配额分层抽样，
输出文本复杂查询子集候选清单（目标 50 条）。

流程：
  1. 加载 .env，并发调用 DashScope MultiModalEmbedding 为 498 条生成向量
  2. 计算余弦相似度矩阵，> DEDUP_THRESHOLD 视为重复，贪心去重保留更长的一条
  3. 按主题配额分层抽样 50 条（保证主题均衡）
  4. 输出候选清单到 data/stats/text_subset_candidates.json

缓存机制：embedding 生成后落盘到 data/stats/bench_queries_500_embeddings.npy，
二次运行直接读缓存，避免重复消耗 API 额度。
"""
import sys
import json
import time
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

import dashscope
from app.config import MULTIMODAL_EMBEDDING_MODEL

# ===== 配置 =====
INPUT_FILE = PROJECT_ROOT / "data/stats/bench_queries_500.json"
EMB_CACHE = PROJECT_ROOT / "data/stats/bench_queries_500_embeddings.npy"
META_CACHE = PROJECT_ROOT / "data/stats/bench_queries_500_embeddings.meta.json"
OUTPUT_FILE = PROJECT_ROOT / "data/stats/text_subset_candidates.json"

DEDUP_THRESHOLD = 0.92       # 余弦相似度阈值，超过视为语义重复
EMBED_WORKERS = 2             # 并发 embedding 线程数（已降速避免 DashScope 限流）
EMBED_INTERVAL = 0.5          # 单线程调用间隔（秒）
EMBED_MAX_RETRY = 5           # 单条最大重试次数
EMBED_RETRY_BASE = 1.5        # 退避基数（秒）
TARGET_SAMPLES = 50           # 抽样目标条数
RANDOM_SEED = 42

# 主题关键词（与内容分析阶段一致）
TOPIC_KEYWORDS = {
    "门票票价": ["门票", "一日票", "两日票", "票价", "价格"],
    "预约购买": ["预约", "提前", "抢", "购买", "发售"],
    "老人儿童优惠": ["老人", "儿童", "优惠", "免票", "半价", "残障", "残疾"],
    "尊享快速通行": ["尊享", "快速通行", "快速"],
    "退款改签": ["退款", "退票", "退", "改签", "修改", "取消"],
    "地图位置": ["地图", "在哪", "位置", "怎么走", "路线"],
    "节庆活动": ["万圣节", "圣诞", "春节", "活动", "节日", "巡游", "烟花", "花车"],
    "生日庆典": ["生日", "庆典", "生日卡", "奇妙"],
    "高峰平日": ["平日", "高峰日", "特别高峰"],
    "年卡年票": ["年卡", "年票", "年费"],
    "停车交通": ["停车", "地铁", "交通", "剐蹭", "剐", "蹭"],
    "童车储物": ["童车", "储物柜", "行李", "寄存", "存包", "大件"],
    "酒店住宿": ["酒店", "住宿", "入住", "旅馆"],
    "餐饮": ["餐饮", "餐厅", "吃饭", "美食", "餐", "吃"],
    "失物招领": ["丢失", "遗失", "丢", "失物", "招领", "捡到"],
}

# 主题配额（保证主题均衡，合计 50 条）
TOPIC_QUOTA = {
    "门票票价": 6,
    "预约购买": 6,
    "老人儿童优惠": 6,
    "尊享快速通行": 5,
    "退款改签": 5,
    "地图位置": 4,
    "节庆活动": 4,
    "生日庆典": 3,
    "高峰平日": 3,
    "年卡年票": 2,
    "停车交通": 2,
    "童车储物": 2,
    "酒店住宿": 1,
    "餐饮": 1,
    "失物招领": 0,  # 498 条中仅 1 条，从 ai_generate/qa.json 补
}
# 兜底：剩余配额分给"多实体复合"（>=2 主题命中）
MULTI_ENTITY_QUOTA = 0  # 实际抽样时按需补足到 50


def get_text_embedding(text: str):
    """调用 DashScope MultiModalEmbedding 单条 embedding，带指数退避重试"""
    import time as _time
    last_err = None
    for attempt in range(EMBED_MAX_RETRY):
        try:
            _time.sleep(EMBED_INTERVAL)
            resp = dashscope.MultiModalEmbedding.call(
                model=MULTIMODAL_EMBEDDING_MODEL,
                input=[{"text": text}],
            )
            if resp.status_code == 200:
                return resp.output["embeddings"][0]["embedding"]
            # 限流/5xx 重试
            if resp.status_code in (429, 500, 502, 503, 504) or "rate limit" in str(resp.message).lower():
                last_err = f"status={resp.status_code} msg={resp.message}"
                wait = EMBED_RETRY_BASE ** attempt
                print(f"  [重试 {attempt+1}/{EMBED_MAX_RETRY}] {last_err} -> sleep {wait:.1f}s")
                _time.sleep(wait)
                continue
            raise RuntimeError(f"Embedding 不可重试失败: status={resp.status_code} msg={resp.message}")
        except RuntimeError:
            raise
        except Exception as e:
            last_err = str(e)
            if "rate limit" in last_err.lower() or "timeout" in last_err.lower():
                wait = EMBED_RETRY_BASE ** attempt
                print(f"  [重试 {attempt+1}/{EMBED_MAX_RETRY}] {last_err} -> sleep {wait:.1f}s")
                _time.sleep(wait)
                continue
            raise
    raise RuntimeError(f"Embedding 重试 {EMBED_MAX_RETRY} 次仍失败: {last_err}")


def embed_all(queries, force=False):
    """并发 embedding 全部查询，带缓存落盘"""
    if EMB_CACHE.exists() and META_CACHE.exists() and not force:
        with open(META_CACHE, "r", encoding="utf-8") as f:
            meta = json.load(f)
        # 指纹校验：查询列表哈希 + 模型名 + 条数
        q_hash = hashlib.md5(json.dumps(queries, ensure_ascii=False).encode()).hexdigest()[:16]
        if (meta.get("query_hash") == q_hash
            and meta.get("embedding_model") == MULTIMODAL_EMBEDDING_MODEL
            and meta.get("count") == len(queries)):
            print(f"[缓存] 命中 embedding 缓存: {EMB_CACHE}")
            return np.load(EMB_CACHE)

    print(f"[Embedding] 开始为 {len(queries)} 条查询生成向量 (model={MULTIMODAL_EMBEDDING_MODEL})")
    t0 = time.time()
    vectors = [None] * len(queries)
    with ThreadPoolExecutor(max_workers=EMBED_WORKERS) as pool:
        futures = {pool.submit(get_text_embedding, q): i for i, q in enumerate(queries)}
        done = 0
        for fut in as_completed(futures):
            idx = futures[fut]
            vectors[idx] = fut.result()
            done += 1
            if done % 50 == 0:
                print(f"  进度 {done}/{len(queries)} ({done*100//len(queries)}%)")
    elapsed = time.time() - t0
    print(f"[Embedding] 完成，耗时 {elapsed:.1f}s")

    arr = np.array(vectors, dtype=np.float32)
    np.save(EMB_CACHE, arr)
    q_hash = hashlib.md5(json.dumps(queries, ensure_ascii=False).encode()).hexdigest()[:16]
    meta = {
        "embedding_model": MULTIMODAL_EMBEDDING_MODEL,
        "query_hash": q_hash,
        "count": len(queries),
        "dim": int(arr.shape[1]),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(META_CACHE, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[缓存] embedding 已落盘: {EMB_CACHE}")
    return arr


def cosine_sim_matrix(vectors):
    """计算余弦相似度矩阵"""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    normed = vectors / (norms + 1e-9)
    return normed @ normed.T


def dedup(queries, vectors, threshold=DEDUP_THRESHOLD):
    """贪心去重：相似度 > threshold 视为重复，保留更长的一条"""
    print(f"[去重] 阈值={threshold}, 共 {len(queries)} 条")
    sim = cosine_sim_matrix(vectors)
    # 按长度降序排，长的优先保留
    order = sorted(range(len(queries)), key=lambda i: -len(queries[i]))
    kept = []
    dropped = set()
    for i in order:
        if i in dropped:
            continue
        kept.append(i)
        # 标记后续与 i 相似度 > threshold 的为重复
        for j in range(i + 1, len(queries)):
            if j in dropped:
                continue
            if sim[i][j] > threshold:
                dropped.add(j)
    print(f"[去重] 完成: 保留 {len(kept)} 条, 去除 {len(dropped)} 条 (压缩比 {len(dropped)*100//len(queries)}%)")
    return kept


def classify_topic(query):
    """返回 query 命中的主题列表"""
    hits = []
    for topic, kws in TOPIC_KEYWORDS.items():
        if any(k in query for k in kws):
            hits.append(topic)
    return hits


def sample_by_quota(queries, kept_indices):
    """按主题配额分层抽样"""
    import random
    random.seed(RANDOM_SEED)

    # 每个主题归集候选
    topic_pool = {t: [] for t in TOPIC_KEYWORDS}
    multi_entity_pool = []  # 命中 >=2 主题的复杂查询
    for i in kept_indices:
        q = queries[i]
        hits = classify_topic(q)
        if len(hits) >= 2:
            multi_entity_pool.append((i, q, hits))
        for t in hits:
            topic_pool[t].append((i, q))

    selected = []  # (idx, query, topic)
    selected_idx = set()

    # 按配额抽取
    for topic, quota in TOPIC_QUOTA.items():
        pool = [(i, q) for i, q in topic_pool[topic] if i not in selected_idx]
        random.shuffle(pool)
        taken = 0
        for i, q in pool:
            if taken >= quota:
                break
            selected.append((i, q, topic))
            selected_idx.add(i)
            taken += 1
        if taken < quota:
            print(f"  [警告] 主题 {topic} 仅命中 {taken}/{quota} 条")

    # 补足到 TARGET_SAMPLES：从多实体复合池抽取
    remaining = TARGET_SAMPLES - len(selected)
    if remaining > 0:
        random.shuffle(multi_entity_pool)
        for i, q, hits in multi_entity_pool:
            if i in selected_idx:
                continue
            selected.append((i, q, "multi_entity"))
            selected_idx.add(i)
            remaining -= 1
            if remaining <= 0:
                break

    print(f"[抽样] 完成: 选中 {len(selected)} 条")
    return selected


def main():
    # 1. 加载 498 条
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    queries = data["queries"]
    print(f"[加载] 共 {len(queries)} 条查询")

    # 2. embedding
    vectors = embed_all(queries)

    # 3. 语义去重
    kept = dedup(queries, vectors)

    # 4. 主题配额抽样
    selected = sample_by_quota(queries, kept)

    # 5. 输出
    output = {
        "fingerprint": {
            "source": str(INPUT_FILE.name),
            "source_count": len(queries),
            "dedup_threshold": DEDUP_THRESHOLD,
            "deduped_count": len(kept),
            "target_samples": TARGET_SAMPLES,
            "embedding_model": MULTIMODAL_EMBEDDING_MODEL,
            "random_seed": RANDOM_SEED,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
        "samples": [
            {
                "id": f"text_{n:03d}",
                "query": q,
                "topic": topic,
                "length": len(q),
                "source_idx": source_idx,
            }
            for n, (source_idx, q, topic) in enumerate(selected, start=1)
        ],
    }
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"[输出] 候选清单已写入: {OUTPUT_FILE}")

    # 控制台预览
    print("\n=== 抽样预览 ===")
    for s in output["samples"][:10]:
        print(f"  [{s['topic'][:10]:10s}] [{s['length']:2d}字] {s['query']}")
    print(f"  ... 共 {len(output['samples'])} 条")


if __name__ == "__main__":
    main()
