"""
向量数据库自动检测工厂

启动时依次尝试连接已配置的向量数据库:
  1. ChromaDB  — 轻量嵌入式 (推荐，无需服务端)
  2. Milvus    — 高性能分布式 (需独立服务)
  3. FileStorage — 本地文件 (最终回退)

使用方式:
    storage = auto_detect_storage()
    if storage.is_vectordb:
        storage.insert(vectors, texts, metadatas)
    else:
        print("使用本地文件存储")
"""

import logging
from typing import Optional

import settings
from .vector_db_interface import VectorDBInterface, ChromaStorage, MilvusStorage
from .file_storage import FileStorage
from embedder import Embedder

logger = logging.getLogger(__name__)


class FileStorageAdapter(VectorDBInterface):
    """
    将 FileStorage 包装为 VectorDBInterface，统一接口。
    实际仍写 CSV 文件，但对外暴露与向量数据库相同的方法。
    """

    def __init__(self, file_storage: FileStorage):
        self._fs = file_storage
        self._texts = []
        self._vectors = []
        self._metadatas = []
        self._ids = []

    def connect(self, **kwargs) -> bool:
        return True  # 文件存储始终可用

    def insert(self, vectors, texts, metadatas=None, ids=None):
        """暂存数据，调用 save_file() 时写入文件"""
        self._texts.extend(texts)
        self._vectors.append(vectors)
        self._metadatas.extend(metadatas or [{}] * len(texts))
        return ids or [f"local_{i}" for i in range(len(texts))]

    def search(self, query_vector, top_k=5, filter_expr=None):
        """本地文件不支持检索 — 需先导入向量数据库"""
        logger.warning("本地文件存储不支持检索。请使用 import_to_vectordb.py 导入向量数据库。")
        return []

    def delete(self, ids):
        return 0

    def count(self):
        return len(self._texts)

    def disconnect(self):
        pass

    @property
    def is_vectordb(self) -> bool:
        return False


class AutoDetectResult:
    """自动检测结果包装"""

    def __init__(self, storage, db_type: str, is_vectordb: bool, message: str):
        self.storage = storage       # VectorDBInterface 实例
        self.db_type = db_type        # "chroma" | "milvus" | "file"
        self.is_vectordb = is_vectordb
        self.message = message

    def __repr__(self):
        return f"<AutoDetectResult type={self.db_type} vectordb={self.is_vectordb}>"


def auto_detect_storage(
    chroma_dir: str = None,
    milvus_host: str = None,
    milvus_port: int = None,
    collection_name: str = None,
    embedder: Embedder = None,
) -> AutoDetectResult:
    """
    自动检测可用的向量数据库并返回连接。

    检测顺序:
      1. ChromaDB  (如果 settings.VECTOR_DB_TYPE 包含 "chroma")
      2. Milvus    (如果 settings.VECTOR_DB_TYPE 包含 "milvus")
      3. FileStorage (最终回退)

    Args:
        chroma_dir: Chroma 持久化目录
        milvus_host: Milvus 主机地址
        milvus_port: Milvus 端口
        collection_name: 集合名称
        embedder: Embedder 实例 (用于获取向量维度)

    Returns:
        AutoDetectResult 对象
    """
    db_type = settings.VECTOR_DB_TYPE.lower()
    chroma_dir = chroma_dir or settings.CHROMA_PERSIST_DIR
    milvus_host = milvus_host or settings.MILVUS_HOST
    milvus_port = milvus_port or settings.MILVUS_PORT
    collection_name = collection_name or settings.MILVUS_COLLECTION_NAME

    # "auto" = 尝试所有可用数据库
    try_all = (db_type == "auto")

    # --- 尝试 ChromaDB ---
    if try_all or "chroma" in db_type:
        logger.info("检测 ChromaDB 连接...")
        chroma = ChromaStorage(
            persist_dir=chroma_dir,
            collection_name=collection_name,
        )
        if chroma.connect():
            logger.info(f"✓ ChromaDB 连接成功 (目录: {chroma_dir})")
            return AutoDetectResult(
                storage=chroma,
                db_type="chroma",
                is_vectordb=True,
                message=f"ChromaDB 已连接 — {chroma_dir}",
            )
        else:
            logger.info("✗ ChromaDB 不可用 (未安装 chromadb 或连接失败)")

    # --- 尝试 Milvus ---
    if try_all or "milvus" in db_type:
        logger.info(f"检测 Milvus 连接 ({milvus_host}:{milvus_port})...")
        dim = embedder.dim if embedder else settings.TFIDF_MAX_FEATURES
        milvus = MilvusStorage(
            host=milvus_host,
            port=milvus_port,
            user=settings.MILVUS_USER,
            password=settings.MILVUS_PASSWORD,
            collection_name=collection_name,
            dim=dim,
        )
        if milvus.connect():
            logger.info(f"✓ Milvus 连接成功 ({milvus_host}:{milvus_port})")
            return AutoDetectResult(
                storage=milvus,
                db_type="milvus",
                is_vectordb=True,
                message=f"Milvus 已连接 — {milvus_host}:{milvus_port}",
            )
        else:
            logger.info("✗ Milvus 不可用 (未安装 pymilvus 或服务未启动)")

    # --- 回退到文件存储 ---
    logger.info("未检测到可用向量数据库，回退到本地文件存储")
    file_storage = FileStorage(
        output_dir=settings.OUTPUT_DIR,
        format=settings.OUTPUT_FORMAT,
    )
    adapter = FileStorageAdapter(file_storage)

    return AutoDetectResult(
        storage=adapter,
        db_type="file",
        is_vectordb=False,
        message="文件存储模式 — 向量数据保存到 CSV/Excel",
    )


def test_connection(db_type: str = "chroma", **kwargs) -> dict:
    """
    单独测试指定向量数据库的连接。

    Args:
        db_type: "chroma" | "milvus"
        **kwargs: 传递给对应 Storage 构造函数的参数

    Returns:
        {"ok": True/False, "message": "...", "count": int}
    """
    if db_type == "chroma":
        storage = ChromaStorage(**kwargs)
    elif db_type == "milvus":
        storage = MilvusStorage(**kwargs)
    else:
        return {"ok": False, "message": f"不支持的数据库类型: {db_type}", "count": 0}

    if storage.connect():
        cnt = storage.count()
        storage.disconnect()
        return {"ok": True, "message": f"{db_type} 连接成功", "count": cnt}
    else:
        return {"ok": False, "message": f"{db_type} 连接失败 — 服务不可用或未安装依赖", "count": 0}
