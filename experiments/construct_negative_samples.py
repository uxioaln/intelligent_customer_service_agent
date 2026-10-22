"""
构造负例/拒答子集：30 条查询，知识库明确未覆盖，Agent 应拒答。

策略：
  1. 定义知识库未覆盖话题清单（外部乐园/其他城市迪士尼/不相关业务/不存在服务/超范围询问）
  2. 用 gpt-4o 基于话题清单批量生成自然问法（每个话题 3-5 条）
  3. 自动校验：每条都跑一次 Qdrant 检索，确认 top-1 similarity < 0.5（确实无相关结果）
  4. 输出 negative_samples_constructed.json

知识库覆盖范围（不可作为负例来源）：
  上海迪士尼门票/老人票/年卡/尊享卡/礼宾/退改/平日高峰日/万圣节/生日/酒店/停车/
  储物柜童车/地图APP/烟花巡游/残障无障碍/失物招领/餐饮/闭园台风/团队票
"""
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")
os.environ.setdefault("QDRANT_URL", "http://localhost:6333")

from app.config import CHAT_MODEL, JUDGE_MODEL
from app.core.cost_tracker import CostTracker, tracked_chat_completion
from app.core.query import distance_to_similarity, get_text_embedding, search_vectors
from app.index.builder import get_qdrant_client, collection_exists
from openai import OpenAI

# ===== 配置 =====
OUTPUT_FILE = PROJECT_ROOT / "data/stats/negative_samples_constructed.json"
PARTIAL_FILE = PROJECT_ROOT / "data/stats/negative_samples.partial.json"

TARGET_COUNT = 30                  # 目标条数
LLM_CALL_INTERVAL = 0.3            # 调用间隔
NEGATIVE_SIM_THRESHOLD = 0.5       # 检索最高相似度低于此值才认为是真负例
RANDOM_SEED = 42

# 知识库已覆盖话题（负例必须避开）
COVERED_TOPICS = [
    "上海迪士尼门票（一日票/两日票/票价）",
    "老人票（年满65周岁）",
    "儿童票（1.4米以下/3岁以下）",
    "年卡",
    "尊享卡/礼宾服务/快速通行",
    "退票/改期/手续费",
    "平日/高峰日/特别高峰日价格",
    "万圣节活动",
    "生日庆典/奇妙生日卡",
    "上海迪士尼酒店/住宿",
    "停车场/剐蹭视频",
    "储物柜/童车/寄存",
    "地图/APP/位置",
    "烟花/巡游/花车",
    "残障人士/无障碍/轮椅",
    "失物招领",
    "餐厅/餐饮",
    "闭园/台风/暴雨改期",
    "团队票/团购8折",
]

# 知识库未覆盖话题（负例来源）
UNCOVERED_TOPICS = [
    # 1. 其他城市/其他乐园
    {"topic": "东京迪士尼", "examples": ["东京迪士尼门票多少钱", "东京迪士尼怎么去", "东京迪士尼海洋乐园值得去吗"]},
    {"topic": "香港迪士尼", "examples": ["香港迪士尼门票价格", "香港迪士尼怎么走", "香港迪士尼和上海迪士尼哪个好玩"]},
    {"topic": "环球影城", "examples": ["环球影城门票多少钱", "环球影城有哪些项目", "北京环球影城和上海迪士尼哪个好"]},
    {"topic": "长隆乐园", "examples": ["长隆欢乐世界门票", "长隆野生动物园", "广州长隆怎么去"]},
    # 2. 不相关业务
    {"topic": "电影票", "examples": ["电影票怎么退", "最近有什么好电影", "电影院会员卡怎么办"]},
    {"topic": "机票/航班", "examples": ["机票怎么改签", "航班延误怎么办", "东航会员有什么权益"]},
    {"topic": "外卖退款", "examples": ["外卖送错了怎么退款", "美团退款多久到账", "饿了么客服电话"]},
    {"topic": "网购退款", "examples": ["淘宝退款多久到账", "京东退货流程", "拼多多客服怎么联系"]},
    # 3. 不存在的服务
    {"topic": "宠物寄存", "examples": ["可以带宠物进乐园吗", "乐园有宠物寄存服务吗", "宠物推车能租吗"]},
    {"topic": "抽烟区", "examples": ["乐园里有抽烟区吗", "哪里可以抽烟", "电子烟能带进乐园吗"]},
    {"topic": "儿童托管", "examples": ["有儿童托管服务吗", "可以托管3岁小孩吗", "儿童看护怎么收费"]},
    {"topic": "导游服务", "examples": ["可以请导游吗", "导游服务怎么收费", "有中文导游吗"]},
    # 4. 超范围询问
    {"topic": "未来票价", "examples": ["明年门票会涨吗", "2026年有什么新活动", "明年会开新园区吗"]},
    {"topic": "员工政策", "examples": ["迪士尼员工工资多少", "怎么应聘迪士尼", "员工有什么福利"]},
    {"topic": "内部经营", "examples": ["迪士尼一年营收多少", "乐园股东是谁", "上海迪士尼投资多少亿"]},
    # 5. 边缘场景（看似合理但知识库无）
    {"topic": "其他乐园门票对比", "examples": ["迪士尼门票比环球影城贵吗", "迪士尼和长隆哪个便宜"]},
    {"topic": "迪士尼历史", "examples": ["迪士尼是什么时候成立的", "华特迪士尼是谁", "上海迪士尼是哪年开的"]},
    {"topic": "周边商品购买", "examples": ["哪里能买正版周边", "乐园外有周边店吗", "淘宝有迪士尼官方店吗"]},
    {"topic": "乐园外餐饮", "examples": ["乐园周边有什么好吃的", "乐园外有迪士尼主题餐厅吗"]},
]

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def make_client() -> OpenAI:
    from app.config import AGICTO_API_KEY, CHAT_BASE_URL
    return OpenAI(api_key=AGICTO_API_KEY, base_url=CHAT_BASE_URL)


GEN_PROMPT = """你是一名客服评测集构造助手。请基于以下"知识库未覆盖话题"，生成 {n} 条真实的用户提问。

要求：
1. 每条查询必须是用户在客服场景下自然问法（口语化、含省略/指代均可）
2. 查询必须围绕给定话题，不可偏离
3. 查询必须与上海迪士尼知识库已覆盖话题不重叠（避免误伤）
4. 风格多样：疑问句、陈述句求确认、对比类、列举类等
5. 每条不超过 30 字

知识库已覆盖话题（必须避开）：
{covered}

知识库未覆盖话题（从这里生成）：
{uncovered}

请输出 JSON：
{{
    "queries": [
        {{"query": "...", "topic": "...", "expected_behavior": "拒答-建议咨询人工客服"}}
    ]
}}
"""


def call_llm(client: OpenAI, prompt: str, tracker: CostTracker) -> str:
    time.sleep(LLM_CALL_INTERVAL)
    response = tracked_chat_completion(
        client=client,
        model=JUDGE_MODEL,  # 用 gpt-4o 生成，质量更高
        messages=[{"role": "user", "content": prompt}],
        tracker=tracker,
        source="评测集扩充-负例构造",
        temperature=0.7,
    )
    return response.choices[0].message.content


def verify_negative(query_str: str, qdrant_client) -> Dict:
    """校验：检索 top-1 similarity 必须低于阈值，才认为是真负例"""
    try:
        q_vec = get_text_embedding(query_str)
        raw_results = search_vectors(q_vec, qdrant_client)
        if not raw_results:
            return {"is_negative": True, "top_similarity": 0.0, "reason": "无检索结果"}
        top_dist, _, m = raw_results[0]
        top_sim = distance_to_similarity(top_dist)
        is_neg = top_sim < NEGATIVE_SIM_THRESHOLD
        return {
            "is_negative": is_neg,
            "top_similarity": round(float(top_sim), 4),
            "top_source": m.get("source", ""),
            "reason": f"top1 sim={top_sim:.4f} {'<' if is_neg else '>='} {NEGATIVE_SIM_THRESHOLD}"
        }
    except Exception as e:
        return {"is_negative": False, "top_similarity": 0.0, "reason": f"检索失败: {str(e)[:80]}"}


def load_partial() -> Dict:
    if PARTIAL_FILE.exists():
        with open(PARTIAL_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"completed": [], "rejected": []}


def save_partial(data: Dict) -> None:
    with open(PARTIAL_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    # 1. 准备 prompt
    covered_str = "\n".join(f"- {t}" for t in COVERED_TOPICS)
    uncovered_str = "\n".join(
        f"- {t['topic']}（示例: {', '.join(t['examples'][:2])}）"
        for t in UNCOVERED_TOPICS
    )

    # 2. 加载断点续跑
    partial = load_partial()
    completed = partial.get("completed", [])
    rejected = partial.get("rejected", [])
    logger.info("断点续跑: 已完成 %d 条, 已拒绝 %d 条", len(completed), len(rejected))

    # 3. 初始化
    client = make_client()
    qdrant_client = get_qdrant_client()
    if not collection_exists(qdrant_client):
        logger.error("Qdrant collection 不存在")
        return
    tracker = CostTracker()

    # 4. 多轮生成（每轮 10 条），直到凑够 30 条
    t0 = time.time()
    rounds = 0
    max_rounds = 8

    while len(completed) < TARGET_COUNT and rounds < max_rounds:
        rounds += 1
        need = TARGET_COUNT - len(completed)
        batch_target = max(10, need + 5)  # 多生成一些容错
        logger.info("[第 %d 轮] 还需 %d 条, 本轮生成 %d 条候选", rounds, need, batch_target)

        prompt = GEN_PROMPT.format(
            n=batch_target,
            covered=covered_str,
            uncovered=uncovered_str,
        )
        try:
            raw = call_llm(client, prompt, tracker)
        except Exception as e:
            logger.error("LLM 调用失败: %s", str(e)[:120])
            continue

        # 去除可能的 markdown 包裹
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
            logger.warning("JSON 解析失败: %s, 原始响应前200字符: %s", str(e), raw[:200])
            continue

        logger.info("本轮生成 %d 条候选", len(queries))

        # 5. 逐条校验
        for q_data in queries:
            q = (q_data.get("query") or "").strip()
            topic = q_data.get("topic", "")
            if not q:
                continue
            if q in [c["query"] for c in completed]:
                continue  # 去重

            verification = verify_negative(q, qdrant_client)
            record = {
                "id": f"neg_{len(completed)+1:03d}",
                "query": q,
                "topic": topic,
                "expected_behavior": q_data.get("expected_behavior", "拒答-建议咨询人工客服"),
                "verification": verification,
            }
            if verification["is_negative"]:
                completed.append(record)
                logger.info("  [✓ %d] %s (%s)", len(completed), q, verification["reason"])
            else:
                rejected.append(record)
                logger.info("  [✗ 拒绝] %s (%s)", q, verification["reason"])

            if len(completed) >= TARGET_COUNT:
                break

        # 落盘断点
        save_partial({"completed": completed, "rejected": rejected})
        elapsed = time.time() - t0
        logger.info("[进度] 已完成 %d/%d, 已拒绝 %d, 已用 %.1fs",
                    len(completed), TARGET_COUNT, len(rejected), elapsed)

    # 6. 最终输出
    output = {
        "fingerprint": {
            "source": "LLM 生成 + Qdrant 检索校验",
            "count": len(completed),
            "rejected_count": len(rejected),
            "negative_sim_threshold": NEGATIVE_SIM_THRESHOLD,
            "covered_topics": COVERED_TOPICS,
            "uncovered_topics": [t["topic"] for t in UNCOVERED_TOPICS],
            "rounds": rounds,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
        "samples": completed,
        "rejected_samples": rejected,
    }
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    # 清理 partial
    if PARTIAL_FILE.exists():
        os.remove(PARTIAL_FILE)

    logger.info("输出: %s", OUTPUT_FILE)
    print(f"\n=== 汇总 ===")
    print(f"目标: {TARGET_COUNT}, 完成: {len(completed)}, 拒绝: {len(rejected)}, 耗时: {time.time()-t0:.1f}s")
    print(f"\n=== 负例清单（前 10 条）===")
    for s in completed[:10]:
        print(f"  [{s['id']}] sim={s['verification']['top_similarity']:.4f} [{s['topic']}] {s['query']}")


if __name__ == "__main__":
    main()
