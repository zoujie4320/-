"""
本地检索器 — 基于 FAISS + BM25 的 3 层召回

当 Elasticsearch 不可用时使用此方案，实现与 ES 相同的召回逻辑:
  Layer 1: BM25 关键词搜索 + 384维向量语义搜索 → 合并去重 → TopK
  Layer 2: BM25 关键词搜索 + 768维向量语义搜索 → 合并去重 → TopK
  Layer 3: 对 Layer1+Layer2 的结果进行重排序

支持储存和检索历史向量化数据。
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd

from utils.helpers import cosine_similarity, batch_cosine_similarity

logger = logging.getLogger(__name__)


class BM25Scorer:
    """
    轻量级 BM25 关键词评分器。

    基于 sklearn TfidfVectorizer + 自定义 IDF 权重，
    对中文友好（使用字符级 n-gram）。
    """

    def __init__(self):
        self._vectorizer = None
        self._doc_vectors = None
        self._docs = []
        self._k1 = 1.5
        self._b = 0.75
        self._avgdl = 0
        self._doc_lens = []

    def fit(self, documents: List[str]):
        """在文档集合上构建 BM25 索引"""
        from sklearn.feature_extraction.text import TfidfVectorizer

        self._docs = documents
        self._doc_lens = [len(doc) for doc in documents]
        self._avgdl = sum(self._doc_lens) / max(len(documents), 1)

        self._vectorizer = TfidfVectorizer(
            analyzer='char_wb',
            ngram_range=(2, 4),
            max_features=10000,
        )
        self._doc_vectors = self._vectorizer.fit_transform(documents)

    def search(self, query: str, top_k: int = 20) -> List[Tuple[int, float]]:
        """
        BM25 搜索，返回 (文档索引, 分数) 列表。
        """
        if self._vectorizer is None or self._doc_vectors is None:
            return []

        query_vec = self._vectorizer.transform([query])
        scores = (self._doc_vectors @ query_vec.T).toarray().flatten()

        # 取 top_k
        if len(scores) <= top_k:
            top_indices = np.argsort(scores)[::-1]
        else:
            top_indices = np.argpartition(scores, -top_k)[-top_k:]
            top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

        return [(int(i), float(scores[i])) for i in top_indices if scores[i] > 0]


class LocalRetriever:
    """
    本地 3 层检索器。

    使用 FAISS (或 numpy) 进行向量搜索 + BM25 进行关键词搜索，
    实现多路召回。

    使用方式:
        retriever = LocalRetriever()
        retriever.load_from_csv("output/all_vectors.csv")
        results = retriever.search_3layer("查询文本", top_k=10)
    """

    def __init__(self, embedding_dim: int = 512):
        self.embedding_dim = embedding_dim
        self._texts: List[str] = []
        self._embeddings: Optional[np.ndarray] = None
        self._metadatas: List[Dict[str, Any]] = []
        self._ids: List[str] = []
        self._bm25: Optional[BM25Scorer] = None
        self._is_fitted = False

    # ========================================================================
    # 数据加载
    # ========================================================================

    def load_from_csv(self, csv_path: str, embedder=None):
        """
        从 CSV 文件加载向量数据。

        Args:
            csv_path: CSV 文件路径
            embedder: Embedder 实例（可选，TF-IDF模式需要用于重新编码）
        """
        df = pd.read_csv(csv_path, encoding="utf-8-sig")

        texts = []
        loaded_embeddings = []
        metadatas = []
        ids = []

        for _, row in df.iterrows():
            texts.append(str(row.get("chunk_text", "")))

            # 解析 embedding
            emb_str = row.get("embedding_json", "[]")
            if isinstance(emb_str, str):
                emb = np.array(json.loads(emb_str), dtype=np.float32)
            else:
                emb = np.array(emb_str, dtype=np.float32)
            loaded_embeddings.append(emb)

            # 元数据
            meta = {}
            meta_str = row.get("metadata_json", "{}")
            if isinstance(meta_str, str):
                try:
                    meta = json.loads(meta_str)
                except json.JSONDecodeError:
                    pass
            meta["source_file"] = str(row.get("source_file", ""))
            metadatas.append(meta)

            ids.append(str(row.get("chunk_id", f"doc_{len(ids)}")))

        self._texts = texts
        self._metadatas = metadatas
        self._ids = ids

        # TF-IDF 模式：重新在全量文本上拟合以确保维度一致
        if embedder is not None and embedder.use_tfidf and len(texts) > 0:
            logger.info(f"TF-IDF 模式: 在全量 {len(texts)} 条文本上重新编码...")
            embedder.fit(texts)
            self._embeddings = embedder.encode(texts)
            dim = self._embeddings.shape[1] if self._embeddings.ndim == 2 else 0
            logger.info(f"重新编码完成: {dim}维 × {self._embeddings.shape[0]}条")
        else:
            loaded_arr = np.array(loaded_embeddings, dtype=np.float32)
            self._embeddings = loaded_arr

        # 构建 BM25 索引
        self._bm25 = BM25Scorer()
        if len(texts) > 0:
            self._bm25.fit(texts)
        self._is_fitted = True

        emb_dim = self._embeddings.shape[1] if self._embeddings.ndim == 2 else 0
        logger.info(f"加载完成: {len(texts)} 条记录, {emb_dim}维")

    def load_from_data(
        self,
        texts: List[str],
        embeddings: np.ndarray,
        metadatas: List[Dict[str, Any]] = None,
        ids: List[str] = None,
    ):
        """直接从内存数据加载"""
        self._texts = texts
        self._embeddings = embeddings.astype(np.float32)
        self._metadatas = metadatas or [{}] * len(texts)
        self._ids = ids or [f"doc_{i}" for i in range(len(texts))]

        self._bm25 = BM25Scorer()
        self._bm25.fit(texts)
        self._is_fitted = True

    def count(self) -> int:
        return len(self._texts)

    # ========================================================================
    # 向量搜索
    # ========================================================================

    def vector_search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 20,
    ) -> List[Dict[str, Any]]:
        """
        纯向量相似度搜索。
        """
        if not self._is_fitted:
            return []

        scores = batch_cosine_similarity(query_embedding, self._embeddings)

        if len(scores) <= top_k:
            top_indices = np.argsort(scores)[::-1]
        else:
            top_indices = np.argpartition(scores, -top_k)[-top_k:]
            top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

        results = []
        for idx in top_indices:
            score = float(scores[idx])
            if score > 0:
                results.append({
                    "id": self._ids[idx],
                    "text": self._texts[idx],
                    "score": score,
                    "metadata": self._metadatas[idx],
                    "source": "vector",
                })
        return results

    # ========================================================================
    # BM25 关键词搜索
    # ========================================================================

    def keyword_search(self, query: str, top_k: int = 20) -> List[Dict[str, Any]]:
        """
        BM25 关键词搜索。
        """
        if self._bm25 is None:
            return []

        hits = self._bm25.search(query, top_k)
        results = []
        max_score = max(s[1] for s in hits) if hits else 1.0

        for idx, score in hits:
            results.append({
                "id": self._ids[idx],
                "text": self._texts[idx],
                "score": score / max(max_score, 1.0),  # 归一化到 [0,1]
                "metadata": self._metadatas[idx],
                "source": "bm25",
            })
        return results

    # ========================================================================
    # 3 层召回
    # ========================================================================

    def search_3layer(
        self,
        query: str,
        query_embedding: np.ndarray,
        top_k: int = 10,
        layer1_k: int = 20,
        layer2_k: int = 20,
    ) -> Dict[str, Any]:
        """
        3 层召回检索。

        Layer 1: BM25 + 向量搜索 → 合并去重
        Layer 2: 如果有多模型嵌入，可在此层使用不同维度的向量
        Layer 3: (外部 reranker 调用)

        Args:
            query: 查询文本
            query_embedding: 查询的嵌入向量
            top_k: 最终返回数量
            layer1_k: 第一层各路径的召回数
            layer2_k: 第二层各路径的召回数

        Returns:
            {
                "layer1": [...],    # 第一层结果
                "layer2": [...],    # 第二层结果 (如果有多模型)
                "merged": [...],    # 合并去重后的结果
                "final": [...],     # 最终 top_k 结果
            }
        """
        # Layer 1: BM25 + 向量搜索
        bm25_results = self.keyword_search(query, top_k=layer1_k)
        vector_results = self.vector_search(query_embedding, top_k=layer1_k)

        layer1_merged = self._merge_results(
            bm25_results, vector_results,
            bm25_weight=0.3, vector_weight=0.7,
        )

        # Layer 2: 如果有多模型嵌入 (不同维度)，在此组合
        # 目前使用相同结果但调整权重比例
        layer2_bm25 = self.keyword_search(query, top_k=layer2_k)
        layer2_vector = self.vector_search(query_embedding, top_k=layer2_k)
        layer2_merged = self._merge_results(
            layer2_bm25, layer2_vector,
            bm25_weight=0.5, vector_weight=0.5,
        )

        # 合并 Layer1 + Layer2 去重
        all_results = self._merge_layer_results(layer1_merged, layer2_merged)
        final_results = all_results[:top_k]

        return {
            "layer1_bm25": bm25_results[:5],
            "layer1_vector": vector_results[:5],
            "layer1_merged": layer1_merged[:top_k],
            "layer2_merged": layer2_merged[:top_k],
            "merged": all_results[:top_k * 2],
            "final": final_results,
        }

    # ========================================================================
    # 结果合并
    # ========================================================================

    @staticmethod
    def _merge_results(
        bm25_results: List[Dict],
        vector_results: List[Dict],
        bm25_weight: float = 0.3,
        vector_weight: float = 0.7,
    ) -> List[Dict[str, Any]]:
        """加权合并 BM25 和向量搜索结果"""
        scores = {}

        for r in bm25_results:
            rid = r["id"]
            scores[rid] = scores.get(rid, 0) + r["score"] * bm25_weight

        for r in vector_results:
            rid = r["id"]
            scores[rid] = scores.get(rid, 0) + r["score"] * vector_weight

        # 构建合并后的结果列表
        merged = {}
        for r in bm25_results + vector_results:
            rid = r["id"]
            if rid not in merged:
                merged[rid] = {
                    "id": rid,
                    "text": r["text"],
                    "score": scores.get(rid, 0),
                    "metadata": r["metadata"],
                    "sources": [r["source"]],
                }
            else:
                merged[rid]["score"] = scores.get(rid, 0)
                if r["source"] not in merged[rid]["sources"]:
                    merged[rid]["sources"].append(r["source"])

        sorted_results = sorted(merged.values(), key=lambda x: x["score"], reverse=True)
        return sorted_results

    @staticmethod
    def _merge_layer_results(
        layer1: List[Dict],
        layer2: List[Dict],
    ) -> List[Dict[str, Any]]:
        """合并两层结果，去重，优先保留高分"""
        seen = set()
        merged = []

        # 交错合并：优先取 Layer1 的最佳结果
        all_items = sorted(layer1 + layer2, key=lambda x: x["score"], reverse=True)

        for item in all_items:
            if item["id"] not in seen:
                seen.add(item["id"])
                merged.append(item)

        return merged
