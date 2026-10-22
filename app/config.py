# -*- coding: utf-8 -*-
"""
app.config - 全局常量与文件路径

约束：本模块是叶子模块，不导入任何项目内其他模块
"""
import os

# ========== 项目根目录与数据目录 ==========
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

INDEX_DIR = os.path.join(DATA_DIR, "indexes")
DB_DIR = os.path.join(DATA_DIR, "db")
STATS_DIR = os.path.join(DATA_DIR, "stats")
KB_DIR = os.path.join(DATA_DIR, "knowledge_base")

# ========== 路径常量 ==========
DOCS_DIR = KB_DIR
IMG_DIR = os.path.join(KB_DIR, "images")

# doc_hashes 仍用独立文件存储（增量检测），与 Qdrant 解耦
HASHES_FILE = os.path.join(INDEX_DIR, "disney_doc_hashes.json")
DB_FILE = os.path.join(DB_DIR, "users.db")
COST_STATS_FILE = os.path.join(STATS_DIR, "cost_stats.json")

# ========== Qdrant 配置 ==========
QDRANT_PATH = os.path.join(DATA_DIR, "qdrant")
QDRANT_COLLECTION_NAME = "disney_knowledge"

# Qdrant 向量维度：优先环境变量，其次从旧版 disney_vectors.npy 读取，兜底 1024
QDRANT_VECTOR_SIZE = int(os.getenv("QDRANT_VECTOR_SIZE", "0"))
if not QDRANT_VECTOR_SIZE:
    _npy_file = os.path.join(INDEX_DIR, "disney_vectors.npy")
    try:
        import numpy as _np
        if os.path.exists(_npy_file):
            QDRANT_VECTOR_SIZE = int(_np.load(_npy_file).shape[1])
    except Exception:
        pass
if not QDRANT_VECTOR_SIZE:
    # tongyi-embedding-vision-plus 实际输出维度为 1024
    QDRANT_VECTOR_SIZE = 1024

# HNSW 图搜索时扩展的邻居数 ef，控制召回精度与延迟的折中
# 基于 2026-09 小规模实验（63 vectors）推荐，知识库增长至 >1000 后需复测调优
QDRANT_SEARCH_EF = 128

# HNSW 图连接数 m；builder 会根据向量规模自动建议（<500 建议 8），可覆盖
QDRANT_HNSW_M = 16

# ========== 上传文件类型 ==========
UPLOAD_DOC_EXT = {".docx"}
UPLOAD_IMG_EXT = {".png", ".jpg", ".jpeg", ".gif", ".bmp"}

# ========== 媒体匹配阈值 ==========
MEDIA_DISTANCE_THRESHOLD = 3.0

# ========== 文本切分参数 ==========
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# ========== 模型与服务 ==========
# 多模态 embedding（DashScope 平台，AGICTO OpenAI 兼容接口不支持该模型）
MULTIMODAL_EMBEDDING_MODEL = "tongyi-embedding-vision-plus"

# Chat LLM（AGICTO 平台，基于中文训练的性价比模型）
CHAT_BASE_URL = "https://api.agicto.cn/v1"
CHAT_MODEL = "deepseek-v4-flash"

# 评测 judge 模型（AGICTO 平台，gpt-4o 作为更强第三方裁判，减少自我偏好偏差）
JUDGE_MODEL = "gpt-4o"

# 验证器模型（与 judge 一致，用于离散分档打分和 claim 级蕴含判断）
VALIDATOR_MODEL = "gpt-4o"

# 多样化改写专用模型（要求较高，索引构建时调用）
DIVERSE_REWRITE_MODEL = "deepseek-v4-pro"

# 视觉理解模型（AGICTO 平台，qwen-vl-plus 用于图片多角度语义描述生成）
VISION_MODEL = "qwen-vl-plus"

# Rerank 精排模型（AGICTO 平台 cross-encoder，走 /v1/rerank 接口）
# 注：原 rerank-v3.5 上游 Cohere 欠费停用，切换为 gte-rerank-v2（中文区分度更好）
RERANK_MODEL = "gte-rerank-v2"

# 图片/视频多样化问题生成数量（每条素材扩充的问题数，影响跨模态召回覆盖度）
MEDIA_DIVERSE_QUESTION_NUM = 8

# ========== 媒体意图关键词 ==========
IMAGE_KEYWORDS = ["图片", "海报", "照片", "看看", "长什么样", "图"]
VIDEO_KEYWORDS = ["视频", "录像", "影片", "看一下", "播放"]

# ========== Agent 链路护栏参数 ==========
# 检索结果相关性阈值：knowledge_search 返回的最高 similarity 低于此值时触发拒答
MIN_RELEVANCE_SCORE = 0.3
# 检索结果最少条数：低于此条数时触发拒答引导
MIN_RELEVANCE_COUNT = 1

# ========== 验证器参数 ==========
# 多次采样投票次数（离散分档打分时每个维度采样 N 次取众数）
VALIDATOR_SAMPLE_COUNT = 3
# 采样温度（高温增加多样性，配合投票提升稳定性）
VALIDATOR_SAMPLE_TEMPERATURE = 0.3

# ========== 环境变量读取（fail-fast） ==========
# DashScope API Key 已改为可选（embedding 和 OCR 均已迁移至 AGICTO 平台）
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")

AGICTO_API_KEY = os.getenv("AGICTO_API_KEY")
if not AGICTO_API_KEY:
    raise ValueError("错误：请设置 'AGICTO_API_KEY' 环境变量（用于 chat LLM）。")
