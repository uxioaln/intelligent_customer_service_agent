"""
跨模态查询扩展：从现有 10 条扩到 40 条。

策略：
  1. 媒体素材 3 个：万圣节图、生日图、汽车剐蹭视频
  2. 每个素材生成不同意图维度的变体查询：
     - 纯媒体意图（"想看图/视频"）
     - 媒体+文本混合（"看图+问票价"等）
     - 对比类（"和 XX 对比"）
     - 条件分支（"如果...的话"）
     - 列举类（"有哪些"）
  3. 每条带 expected_modalities 和 expected_media_ref（与现有 10 条对齐）
  4. 用 gpt-4o 基于种子查询扩展，去重
  5. 每条校验：跑一次 Qdrant 检索，记录 top-5 是否含目标媒体条目（用于后续召回评测口径）

输出文件：data/stats/cross_modal_samples_expanded.json
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

from app.config import JUDGE_MODEL
from app.core.cost_tracker import CostTracker, tracked_chat_completion
from app.core.query import distance_to_similarity, get_text_embedding, search_vectors
from app.index.builder import get_qdrant_client, collection_exists
from openai import OpenAI
from app.config import AGICTO_API_KEY, CHAT_BASE_URL

# ===== 配置 =====
INPUT_FILE = PROJECT_ROOT / "data/stats/eval_dataset.json"
OUTPUT_FILE = PROJECT_ROOT / "data/stats/cross_modal_samples_expanded.json"
PARTIAL_FILE = PROJECT_ROOT / "data/stats/cross_modal_samples.partial.json"

TARGET_COUNT = 40                  # 目标条数
LLM_CALL_INTERVAL = 0.3            # LLM 调用间隔
TOP_K = 5                          # 校验时取 top-5
RANDOM_SEED = 42

# 媒体素材（与现有 eval_dataset 对齐）
MEDIA_ASSETS = [
    {
        "name": "万圣节图",
        "source": "图片: 2-万圣节.jpeg",
        "media_type": "image",
        "expected_media_ref": "/Users/xiaolin/Desktop/项目实战：多模态数据处理RAG_new/data/knowledge_base/images/2-万圣节.jpeg",
        "content_brief": "上海迪士尼万圣节主题布置装饰，南瓜灯、幽灵、万圣节氛围，园区 Halloween 装点",
        "target_count": 13,  # 现有 4 + 新增 9
    },
    {
        "name": "生日图",
        "source": "图片: 1-聚在一起说奇妙.jpg",
        "media_type": "image",
        "expected_media_ref": "/Users/xiaolin/Desktop/项目实战：多模态数据处理RAG_new/data/knowledge_base/images/1-聚在一起说奇妙.jpg",
        "content_brief": "上海迪士尼奇妙讲述会场景，卡通角色聚在一起说奇妙，生日庆典互动",
        "target_count": 13,  # 现有 4 + 新增 9
    },
    {
        "name": "汽车剐蹭视频",
        "source": "视频: 汽车剐蹭视频",
        "media_type": "video",
        "expected_media_ref": "https://dataset-1255932437.cos.ap-nanjing.myqcloud.com/mp4/car.mp4",
        "content_brief": "汽车剐蹭视频，上海迪士尼停车场监控场景",
        "target_count": 14,  # 现有 2 + 新增 12（视频是召回难点，多生成）
    },
]

INTENT_DIMENSIONS = [
    "纯媒体意图（只看图/视频，无文本意图）",
    "媒体+文本混合（看图+问票价/规则/老人票等）",
    "对比类（万圣节和生日装饰对比/和北京迪士尼对比）",
    "条件分支（如果带老人去/如果年卡用户的话）",
    "列举类（园区有哪些万圣节装饰/有哪些生日庆典活动）",
]

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def make_client() -> OpenAI:
    return OpenAI(api_key=AGICTO_API_KEY, base_url=CHAT_BASE_URL)


GEN_PROMPT = """你是一名跨模态评测集构造助手。请基于以下媒体素材，生成 {n} 条跨模态用户查询。

要求：
1. 每条查询必须明确包含对该媒体的访问意图（如"想看图"、"想看视频"、"图片"、"照片"等关键词）
2. 查询需覆盖以下意图维度，每个维度至少 1 条：
   {intent_dims}
3. 风格多样化：口语化、含指代、含条件、含对比、含列举
4. 长度控制在 30~80 字（比纯文本查询长一些，因为要包含媒体意图+文本意图）
5. 不可与种子查询重复（仅做风格变体）

媒体素材：
- 名称：{media_name}
- 内容简述：{content_brief}
- 期望模态：{expected_modalities}

种子查询（参考风格，但不可重复）：
{seed_queries}

知识库已覆盖的文本话题（用于混合意图的组合）：
- 上海迪士尼门票（一日票/两日票/票价）
- 老人票（年满65周岁/证件核验）
- 年卡（宝石卡/珍珠卡/翡翠卡）
- 尊享卡/礼宾服务
- 退票/改期/手续费
- 平日/高峰日/特别高峰日价格
- 万圣节活动
- 生日庆典/生日卡
- 酒店/住宿
- 停车场
- 储物柜/童车
- 地图/APP
- 烟花/巡游
- 残障/无障碍
- 失物招领
- 餐厅/餐饮
- 闭园/台风

输出 JSON 格式：
{{
    "queries": [
        {{
            "query": "...",
            "expected_modalities": ["image"] 或 ["video"] 或 ["image","text"] 或 ["video","text"],
            "intent_dim": "纯媒体意图|媒体+文本混合|对比类|条件分支|列举类"
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
        source="评测集扩充-跨模态扩展",
        temperature=0.7,
    )
    return response.choices[0].message.content


def verify_cross_modal(query_str: str, qdrant_client, expected_media_ref: str) -> Dict:
    """校验：跑一次 Qdrant 检索，记录 top-5 是否含目标媒体条目"""
    try:
        q_vec = get_text_embedding(query_str)
        raw_results = search_vectors(q_vec, qdrant_client)
        top5 = []
        precise_hit = False
        modal_set = set()
        for dist, _, m in raw_results[:TOP_K]:
            source = m.get("source", "")
            top5.append({
                "source": source,
                "similarity": round(float(distance_to_similarity(dist)), 4),
                "type": m.get("type", ""),
                "path": m.get("path", ""),
                "url": m.get("url", ""),
            })
            # 模态集合口径
            if "图片" in source:
                modal_set.add("image")
            elif "视频" in source:
                modal_set.add("video")
            else:
                modal_set.add("text")
            # 精确匹配口径：通过 source 或 path/url 是否含 expected_media_ref 的 basename
            ref_basename = expected_media_ref.split("/")[-1].split("?")[0]
            if (ref_basename in source
                or ref_basename in m.get("path", "")
                or ref_basename in m.get("url", "")):
                precise_hit = True
        return {
            "top5": top5,
            "modal_set": sorted(modal_set),
            "precise_hit": precise_hit,
        }
    except Exception as e:
        return {"top5": [], "modal_set": [], "precise_hit": False, "error": str(e)[:80]}


def load_partial() -> Dict:
    if PARTIAL_FILE.exists():
        with open(PARTIAL_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"completed": [], "rejected": []}


def save_partial(data: Dict) -> None:
    with open(PARTIAL_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    # 1. 加载现有 10 条跨模态查询作为种子
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        existing = json.load(f)
    existing_cross = existing.get("cross_modal_queries", [])
    logger.info("现有跨模态种子: %d 条", len(existing_cross))

    # 按 media_ref 分组
    seeds_by_ref = {}
    for s in existing_cross:
        ref = s.get("expected_media_ref", "")
        seeds_by_ref.setdefault(ref, []).append(s)

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

    # 4. 先把现有 10 条加入 completed（统一格式）
    if not completed:
        for s in existing_cross:
            ref = s.get("expected_media_ref", "")
            # 找对应 asset
            asset = next((a for a in MEDIA_ASSETS if a["expected_media_ref"] == ref), None)
            if not asset:
                continue
            verification = verify_cross_modal(s["query"], qdrant_client, ref)
            record = {
                "id": f"cm_existing_{len(completed)+1:03d}",
                "query": s["query"],
                "expected_modalities": s.get("expected_modalities", []),
                "expected_media_ref": ref,
                "expected_media_type": s.get("expected_media_type", asset["media_type"]),
                "asset_name": asset["name"],
                "intent_dim": "existing",
                "verification": verification,
            }
            completed.append(record)
            logger.info("  [种子] %s sim_top1=%s %s",
                        record["id"],
                        verification["top5"][0]["similarity"] if verification["top5"] else "N/A",
                        s["query"][:50])
        save_partial({"completed": completed, "rejected": rejected})

    # 5. 按素材扩展
    t0 = time.time()
    rounds = 0
    max_rounds = 12

    while len(completed) < TARGET_COUNT and rounds < max_rounds:
        rounds += 1
        # 找出每个素材还差多少
        for asset in MEDIA_ASSETS:
            current_count = sum(1 for c in completed if c["asset_name"] == asset["name"])
            need = asset["target_count"] - current_count
            if need <= 0:
                continue
            batch_target = min(need + 3, 8)  # 多生成 3 条容错
            logger.info("[第 %d 轮][%s] 还需 %d 条, 本轮生成 %d 条候选",
                        rounds, asset["name"], need, batch_target)

            seed_queries = "\n".join(
                f"  {i+1}. {s['query'][:60]}"
                for i, s in enumerate(seeds_by_ref.get(asset["expected_media_ref"], []))
            ) or "  [无种子]"

            prompt = GEN_PROMPT.format(
                n=batch_target,
                intent_dims="\n   ".join(f"- {d}" for d in INTENT_DIMENSIONS),
                media_name=asset["name"],
                content_brief=asset["content_brief"],
                expected_modalities=f'["{asset["media_type"]}"] 或 ["{asset["media_type"]}","text"]',
                seed_queries=seed_queries,
            )
            try:
                raw = call_llm(client, prompt, tracker)
            except Exception as e:
                logger.error("LLM 调用失败: %s", str(e)[:120])
                continue

            # 去 markdown 包裹
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

            # 校验每条
            for q_data in queries:
                q = (q_data.get("query") or "").strip()
                if not q:
                    continue
                if q in [c["query"] for c in completed]:
                    continue
                expected_modalities = q_data.get("expected_modalities") or [asset["media_type"]]
                intent_dim = q_data.get("intent_dim", "")

                verification = verify_cross_modal(q, qdrant_client, asset["expected_media_ref"])
                record = {
                    "id": f"cm_{len(completed)+1:03d}",
                    "query": q,
                    "expected_modalities": expected_modalities,
                    "expected_media_ref": asset["expected_media_ref"],
                    "expected_media_type": asset["media_type"],
                    "asset_name": asset["name"],
                    "intent_dim": intent_dim,
                    "verification": verification,
                }
                completed.append(record)
                logger.info("  [✓ %d] %s intent=%s precise_hit=%s %s",
                            len(completed), record["id"], intent_dim,
                            verification.get("precise_hit", False), q[:50])

                if len(completed) >= TARGET_COUNT:
                    break
            save_partial({"completed": completed, "rejected": rejected})

        elapsed = time.time() - t0
        logger.info("[进度] 已完成 %d/%d, 已用 %.1fs", len(completed), TARGET_COUNT, elapsed)

    # 6. 最终输出
    # 按 asset 统计精确命中率
    stats = {}
    for asset in MEDIA_ASSETS:
        records = [c for c in completed if c["asset_name"] == asset["name"]]
        hits = sum(1 for r in records if r["verification"].get("precise_hit", False))
        stats[asset["name"]] = {
            "count": len(records),
            "precise_hits": hits,
            "hit_rate": round(hits / max(1, len(records)), 4),
        }

    output = {
        "fingerprint": {
            "count": len(completed),
            "target_count": TARGET_COUNT,
            "media_assets": [{"name": a["name"], "ref": a["expected_media_ref"]} for a in MEDIA_ASSETS],
            "intent_dimensions": INTENT_DIMENSIONS,
            "stats": stats,
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
    print(f"\n=== 各素材精确命中率 ===")
    for asset_name, s in stats.items():
        print(f"  {asset_name}: {s['precise_hits']}/{s['count']} ({s['hit_rate']*100:.1f}%)")


if __name__ == "__main__":
    main()
