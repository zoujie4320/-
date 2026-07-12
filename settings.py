"""
=============================================================================
RAG 数据预处理系统 — 部署配置文件
=============================================================================

所有可替换的部署信息集中在此文件：
- 文件路径 (知识库目录、输出目录)
- 嵌入模型选择
- OCR 参数
- 数据库连接信息 (预留)
- 大模型 API 密钥 (预留)

使用方式：修改此文件中的变量值即可，无需改动任何业务代码。
其他模块通过 `from settings import ...` 导入所需变量。

=============================================================================
"""

import os
from pathlib import Path

# ============================================================================
# 项目根目录
# ============================================================================
PROJECT_ROOT = Path(__file__).parent.absolute()

# ============================================================================
# 路径配置
# ============================================================================

# 知识库源文件目录 — 将需要处理的文档放在此文件夹中
# main.py 启动时会自动扫描此目录
KNOWLEDGE_BASE_DIR = os.path.join(PROJECT_ROOT, "knowledge_base")

# 向量化结果输出目录
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")

# 输出文件格式: "csv" 或 "excel"
OUTPUT_FORMAT = "csv"

# ============================================================================
# 嵌入模型配置
# ============================================================================

# True  = TF-IDF 离线模式 (无需联网，使用 sklearn)
# False = Sentence-Transformers 在线模式 (需联网下载模型，语义理解更强)
USE_TFIDF = True

# Sentence-Transformers 模型名称 (仅 USE_TFIDF=False 时生效)
# 推荐选项:
#   "paraphrase-multilingual-MiniLM-L12-v2"  — 多语言，轻量，384维
#   "BAAI/bge-small-zh-v1.5"                 — 中文优化，512维
#   "intfloat/multilingual-e5-base"          — 多语言，768维
EMBEDDING_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

# TF-IDF 特征维度 (仅 USE_TFIDF=True 时生效，建议 256~1024)
TFIDF_MAX_FEATURES = 512

# 嵌入编码的批量大小 (GPU 用户可适当调大)
EMBEDDING_BATCH_SIZE = 32

# ============================================================================
# OCR 配置 (PDF & 图片处理器)
# ============================================================================

# Tesseract OCR 语言包
# "chi_sim+eng" = 中文简体 + 英文
# "chi_sim"     = 仅中文简体
# "eng"         = 仅英文
# "chi_sim+eng+jpn" = 中英日
OCR_LANG = "chi_sim+eng"

# PDF 转图片时的分辨率 (DPI)，越高越清晰但越慢
PDF_DPI = 300

# 是否对图片进行预处理 (灰度化+对比度增强)，通常能提升 OCR 准确率
IMAGE_PREPROCESS = True

# ============================================================================
# 数据库连接 (预留 — 接入向量数据库时填写)
# ============================================================================

# --- Milvus ---
MILVUS_HOST = "localhost"
MILVUS_PORT = 19530
MILVUS_USER = ""
MILVUS_PASSWORD = ""
MILVUS_COLLECTION_NAME = "rag_knowledge_base"

# --- Chroma ---
CHROMA_PERSIST_DIR = os.path.join(PROJECT_ROOT, "chroma_db")

# 当前使用的向量数据库类型: "auto" | "chroma" | "milvus" | "file"
# "auto" = 按顺序自动检测 chroma → milvus → 回退文件
VECTOR_DB_TYPE = "auto"

# ============================================================================
# Elasticsearch 连接 (预留 — 3层召回检索引擎)
# ============================================================================

# Elasticsearch 连接地址
ES_HOST = "localhost"
ES_PORT = 9200
ES_USER = ""
ES_PASSWORD = ""
ES_INDEX_NAME = "rag_knowledge_base"
# 是否使用 ES (False=使用本地 FAISS+BM25 方案)
USE_ELASTICSEARCH = False

# ============================================================================
# 检索配置
# ============================================================================

# 默认返回数量
DEFAULT_TOP_K = 5
# Layer 1 各路径召回数量
LAYER1_RECALL_K = 20
# Layer 2 各路径召回数量
LAYER2_RECALL_K = 20
# BM25 权重 (关键词匹配)
BM25_WEIGHT_LAYER1 = 0.3
# 向量权重 (语义匹配)
VECTOR_WEIGHT_LAYER1 = 0.7

# Rerank 模型 (CrossEncoder)
# 中文推荐: BAAI/bge-reranker-base
# 英文推荐: cross-encoder/ms-marco-MiniLM-L-6-v2
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# ============================================================================
# 大模型 API 连接 (预留 — 后续检索/生成模块使用)
# ============================================================================

# --- OpenAI 兼容接口 ---
LLM_API_KEY = os.environ.get("OPENAI_API_KEY", "your-api-key-here")
LLM_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
LLM_MODEL = "gpt-4o"

# --- 本地模型 (如 Ollama) ---
# LLM_BASE_URL = "http://localhost:11434/v1"
# LLM_MODEL = "qwen2.5:7b"

# ============================================================================
# 日志配置
# ============================================================================

LOG_LEVEL = "INFO"  # DEBUG | INFO | WARNING | ERROR

# ============================================================================
# 处理策略配置
# ============================================================================

# 增量处理模式: True=跳过已处理的文件（基于文件哈希），False=全量重新处理
INCREMENTAL_MODE = True

# 文本清洗: True=对提取的文本做规范化处理（去除噪声、修复OCR常见错误）
ENABLE_TEXT_CLEANING = True

# Chunk 去重: True=自动删除高度相似的重复 chunk（由 config.ChunkConfig.deduplicate 控制）

# 并行处理线程数（1=单线程，0=自动检测CPU核心数）
NUM_WORKERS = 1

# ============================================================================
# 辅助函数
# ============================================================================

def ensure_directories():
    """确保所有必要的目录存在"""
    dirs_to_create = [
        KNOWLEDGE_BASE_DIR,
        OUTPUT_DIR,
    ]
    # 如果使用 Chroma，也创建其持久化目录
    if VECTOR_DB_TYPE == "chroma":
        dirs_to_create.append(CHROMA_PERSIST_DIR)

    for d in dirs_to_create:
        os.makedirs(d, exist_ok=True)


def show_settings():
    """打印当前配置（隐藏敏感信息）"""
    print("=" * 60)
    print("当前 RAG 系统配置")
    print("=" * 60)
    print(f"  知识库目录:     {KNOWLEDGE_BASE_DIR}")
    print(f"  输出目录:       {OUTPUT_DIR}")
    print(f"  输出格式:       {OUTPUT_FORMAT}")
    print(f"  嵌入模式:       {'TF-IDF 离线' if USE_TFIDF else f'在线模型 ({EMBEDDING_MODEL_NAME})'}")
    if USE_TFIDF:
        print(f"  TF-IDF 维度:    {TFIDF_MAX_FEATURES}")
    print(f"  OCR 语言:       {OCR_LANG}")
    print(f"  PDF DPI:        {PDF_DPI}")
    print(f"  向量存储方式:   {VECTOR_DB_TYPE}")
    print(f"  检索引擎:       {'Elasticsearch' if USE_ELASTICSEARCH else 'FAISS+BM25 (本地)'}")
    print(f"  默认返回数:     {DEFAULT_TOP_K}")
    print(f"  Rerank 模型:    {RERANK_MODEL}")
    print(f"  LLM 模型:       {LLM_MODEL}")
    print(f"  API Key 已配置: {'是' if LLM_API_KEY != 'your-api-key-here' else '否'}")
    print("=" * 60)
