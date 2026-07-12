"""
向量数据库接口

提供统一的向量数据库操作接口，支持多种后端:
- ChromaDB  — 轻量嵌入式，无需独立服务
- Milvus    — 高性能分布式
- File      — 本地文件存储 (默认回退)

所有实现遵循相同的 VectorDBInterface 抽象接口。
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import numpy as np


class VectorDBInterface(ABC):
    """
    向量数据库统一接口。

    所有向量数据库实现必须实现此接口。
    """

    @abstractmethod
    def connect(self, **kwargs) -> bool:
        """连接到向量数据库，返回 True 表示连接成功"""
        ...

    @abstractmethod
    def insert(
        self,
        vectors: np.ndarray,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        """
        批量插入向量数据。

        Args:
            vectors: (n, dim) 嵌入向量矩阵
            texts: 对应的文本列表
            metadatas: 每条数据的元数据
            ids: 指定的 ID 列表 (可选，不指定则自动生成)

        Returns:
            插入记录的 ID 列表
        """
        ...

    @abstractmethod
    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
        filter_expr: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        向量相似度搜索。

        Args:
            query_vector: (dim,) 查询向量
            top_k: 返回数量
            filter_expr: 元数据过滤条件

        Returns:
            [{"id": ..., "text": ..., "score": ..., "metadata": {...}}, ...]
        """
        ...

    @abstractmethod
    def delete(self, ids: List[str]) -> int:
        """删除指定 ID 的记录，返回删除数量"""
        ...

    def delete_by_source(self, source_file: str) -> int:
        """
        按源文件名删除所有相关向量（用于文档更新场景）。

        默认实现：子类可覆盖以提供更高效的实现。
        """
        return 0

    @abstractmethod
    def count(self) -> int:
        """返回集合中的记录总数"""
        ...

    @abstractmethod
    def disconnect(self):
        """断开数据库连接"""
        ...


# ============================================================================
# ChromaDB 实现
# ============================================================================

class ChromaStorage(VectorDBInterface):
    """
    ChromaDB 向量存储实现。

    Chroma 是一个轻量级嵌入式向量数据库，无需独立服务进程。
    数据持久化在本地磁盘。

    安装: pip install chromadb
    """

    def __init__(
        self,
        persist_dir: str = "./chroma_db",
        collection_name: str = "rag_knowledge_base",
    ):
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self._client = None
        self._collection = None

    # --- 接口实现 ---

    def connect(self, **kwargs) -> bool:
        """连接（或创建）Chroma 持久化数据库"""
        try:
            import chromadb

            # 兼容 ChromaDB 0.4.x 和 0.5.x+ 的 Settings API
            try:
                # ChromaDB >= 0.5.x: 直接传参
                self._client = chromadb.PersistentClient(
                    path=self.persist_dir,
                )
                # 新版通过环境变量关闭遥测: CHROMA_TELEMETRY=false
                import os as _os
                _os.environ.setdefault("CHROMA_TELEMETRY", "false")
            except TypeError:
                # ChromaDB < 0.5.x: 通过 Settings 对象
                from chromadb.config import Settings as ChromaSettings
                self._client = chromadb.PersistentClient(
                    path=self.persist_dir,
                    settings=ChromaSettings(anonymized_telemetry=False),
                )

            # 获取或创建集合
            try:
                self._collection = self._client.get_collection(
                    name=self.collection_name
                )
            except Exception:
                self._collection = self._client.create_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": "cosine"},  # 余弦相似度
                )

            return True

        except ImportError:
            return False
        except Exception as e:
            print(f"[Chroma] 连接失败: {e}")
            return False

    def insert(
        self,
        vectors: np.ndarray,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        if self._collection is None:
            raise RuntimeError("Chroma 未连接，请先调用 connect()")

        # 准备数据
        if ids is None:
            # 用时间戳 + 序号生成 ID
            import time
            ts = str(int(time.time() * 1000))
            ids = [f"{ts}_{i:06d}" for i in range(len(texts))]

        # 向量转 list
        vec_list = vectors.tolist() if isinstance(vectors, np.ndarray) else vectors

        # 元数据必须都是基本类型（str, int, float, bool）
        clean_metadatas = None
        if metadatas:
            clean_metadatas = []
            for m in metadatas:
                clean = {}
                for k, v in m.items():
                    if isinstance(v, (str, int, float, bool)):
                        clean[k] = v
                    else:
                        clean[k] = str(v)  # 非基本类型转字符串
                clean_metadatas.append(clean)

        # 分批插入 (Chroma 单次插入限制)
        batch_size = 500
        for i in range(0, len(texts), batch_size):
            end = min(i + batch_size, len(texts))
            self._collection.add(
                ids=ids[i:end],
                documents=texts[i:end],
                embeddings=vec_list[i:end],
                metadatas=clean_metadatas[i:end] if clean_metadatas else None,
            )

        return ids

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
        filter_expr: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        if self._collection is None:
            return []

        vec = query_vector.tolist() if isinstance(query_vector, np.ndarray) else query_vector

        results = self._collection.query(
            query_embeddings=[vec],
            n_results=top_k,
            where=filter_expr,
            include=["documents", "metadatas", "distances"],
        )

        # 转换为统一格式
        output = []
        if results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                output.append({
                    "id": doc_id,
                    "text": results["documents"][0][i] if results["documents"] else "",
                    "score": 1.0 - results["distances"][0][i],  # distance → similarity
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                })

        return output

    def delete(self, ids: List[str]) -> int:
        if self._collection is None:
            return 0
        try:
            self._collection.delete(ids=ids)
            return len(ids)
        except Exception:
            return 0

    def delete_by_source(self, source_file: str) -> int:
        """
        按源文件名删除所有相关向量（用于文档更新场景）。

        通过 ChromaDB 元数据过滤查找并删除。
        """
        if self._collection is None:
            return 0
        try:
            # 查询该源文件的所有记录 ID
            result = self._collection.get(
                where={"source_file": source_file},
                include=["metadatas"],
            )
            if result["ids"]:
                self._collection.delete(ids=result["ids"])
                return len(result["ids"])
            return 0
        except Exception:
            return 0

    def count(self) -> int:
        if self._collection is None:
            return 0
        return self._collection.count()

    def disconnect(self):
        self._client = None
        self._collection = None


# ============================================================================
# Milvus 实现 (骨架)
# ============================================================================

class MilvusStorage(VectorDBInterface):
    """
    Milvus 向量数据库实现。

    Milvus 是高性能分布式向量数据库，适合大规模生产环境。

    安装: pip install pymilvus
    需要独立运行的 Milvus 服务 (Docker/Local)
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 19530,
        user: str = "",
        password: str = "",
        collection_name: str = "rag_knowledge_base",
        dim: int = 512,
    ):
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.collection_name = collection_name
        self.dim = dim
        self._connected = False
        self._collection = None

    def connect(self, **kwargs) -> bool:
        """连接到 Milvus 服务"""
        try:
            from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility

            # 连接
            if self.user and self.password:
                connections.connect(
                    alias="default",
                    host=self.host,
                    port=self.port,
                    user=self.user,
                    password=self.password,
                )
            else:
                connections.connect(
                    alias="default",
                    host=self.host,
                    port=self.port,
                )

            # 检查集合是否存在
            if utility.has_collection(self.collection_name):
                self._collection = Collection(self.collection_name)
            else:
                # 创建集合
                fields = [
                    FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=255),
                    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
                    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dim),
                ]
                schema = CollectionSchema(fields, description="RAG Knowledge Base")
                self._collection = Collection(self.collection_name, schema)

                # 创建索引
                index_params = {
                    "metric_type": "COSINE",
                    "index_type": "IVF_FLAT",
                    "params": {"nlist": 128},
                }
                self._collection.create_index("embedding", index_params)

            self._collection.load()
            self._connected = True
            return True

        except ImportError:
            return False
        except Exception as e:
            print(f"[Milvus] 连接失败: {e}")
            return False

    def insert(
        self,
        vectors: np.ndarray,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        if not self._connected or self._collection is None:
            raise RuntimeError("Milvus 未连接")

        import time
        ts = str(int(time.time() * 1000))
        if ids is None:
            ids = [f"{ts}_{i:06d}" for i in range(len(texts))]

        vec_list = vectors.tolist() if isinstance(vectors, np.ndarray) else vectors

        entities = [ids, texts, vec_list]
        self._collection.insert(entities)
        self._collection.flush()

        return ids

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
        filter_expr: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        if not self._connected or self._collection is None:
            return []

        vec = query_vector.tolist() if isinstance(query_vector, np.ndarray) else query_vector

        search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
        results = self._collection.search(
            data=[vec],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            output_fields=["text"],
        )

        output = []
        if results and results[0]:
            for hit in results[0]:
                output.append({
                    "id": hit.id,
                    "text": hit.entity.get("text", ""),
                    "score": hit.score,
                    "metadata": {},
                })

        return output

    def delete(self, ids: List[str]) -> int:
        if not self._connected or self._collection is None:
            return 0
        try:
            expr = f"id in {ids}"
            self._collection.delete(expr)
            return len(ids)
        except Exception:
            return 0

    def count(self) -> int:
        if not self._connected or self._collection is None:
            return 0
        return self._collection.num_entities

    def disconnect(self):
        try:
            from pymilvus import connections
            connections.disconnect("default")
        except Exception:
            pass
        self._connected = False
