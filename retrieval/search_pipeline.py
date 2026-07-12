"""
多路召回 + LLM 合成 — 完整检索流水线

流程:
  1. 用户查询 → 向量化
  2. Layer 1: BM25 + 384维向量搜索 → 合并
  3. Layer 2: BM25 + 768维向量搜索 → 合并 (如果多模型)
  4. Layer 3: Reranker 精排
  5. 多路召回结果合成 + LLM 生成答案

使用方式:
    pipeline = SearchPipeline(retriever, embedder, reranker)
    answer = pipeline.search("查询内容")
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """单条检索结果"""
    id: str
    text: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    source: str = ""           # "bm25" | "vector" | "bm25+vector"
    rerank_score: float = 0.0


@dataclass
class SearchResponse:
    """完整检索响应"""
    query: str
    results: List[SearchResult] = field(default_factory=list)
    answer: str = ""            # LLM 合成的答案
    layer_stats: Dict[str, int] = field(default_factory=dict)  # 各层召回统计
    elapsed_ms: float = 0.0


class SearchPipeline:
    """
    多路召回检索流水线。

    支持:
    - 3 层召回 (BM25 + 向量 × 2 + Rerank)
    - LLM 答案合成
    - 多路结果融合

    使用方式:
        from retrieval import SearchPipeline, LocalRetriever, CrossEncoderReranker
        from embedder import Embedder

        retriever = LocalRetriever()
        retriever.load_from_csv("output/all_vectors.csv")

        embedder = Embedder(use_tfidf=True)
        reranker = CrossEncoderReranker()

        pipeline = SearchPipeline(retriever, embedder, reranker)
        response = pipeline.search("Python开发经验")
        print(response.answer)
    """

    # LLM 答案合成 Prompt
    SYNTHESIS_PROMPT = """你是一个专业的知识检索助手。请根据以下参考资料回答用户问题。

## 规则
- 仅使用提供的参考资料回答问题
- 如果参考资料不足以回答问题，请明确说明
- 引用资料时标明来源
- 回答应结构清晰、准确、简洁

## 用户问题
{query}

## 参考资料
{context}

## 回答"""

    def __init__(
        self,
        retriever,      # LocalRetriever 实例
        embedder,       # Embedder 实例
        reranker=None,  # Reranker 实例 (可选)
        llm_config: dict = None,
    ):
        """
        Args:
            retriever: 检索器 (LocalRetriever)
            embedder: 嵌入器 (Embedder)
            reranker: 重排序器 (Reranker, 可选)
            llm_config: LLM 配置 {"api_key": ..., "base_url": ..., "model": ...}
        """
        self.retriever = retriever
        self.embedder = embedder
        self.reranker = reranker
        self.llm_config = llm_config or {}

    def search(
        self,
        query: str,
        top_k: int = 5,
        use_rerank: bool = True,
        use_llm: bool = True,
    ) -> SearchResponse:
        """
        执行完整检索流程。

        Args:
            query: 用户查询
            top_k: 最终返回数量
            use_rerank: 是否启用重排序
            use_llm: 是否启用 LLM 答案合成

        Returns:
            SearchResponse 对象
        """
        import time
        start_time = time.time()

        # 1. 查询向量化
        query_embedding = self.embedder.encode(query)

        # 2. 3 层召回
        layer_result = self.retriever.search_3layer(
            query=query,
            query_embedding=query_embedding,
            top_k=top_k,
            layer1_k=20,
            layer2_k=20,
        )

        # 3. 合并多路结果
        candidates = layer_result["merged"]

        # 4. Rerank 精排
        if use_rerank and self.reranker and len(candidates) > top_k:
            candidates = self.reranker.rerank(query, candidates, top_k=top_k * 2)
        else:
            candidates = candidates[:top_k * 2]

        # 5. 取最终结果
        final_results = candidates[:top_k]
        search_results = [
            SearchResult(
                id=r.get("id", ""),
                text=r.get("text", ""),
                score=r.get("score", 0.0),
                metadata=r.get("metadata", {}),
                source="+".join(r.get("sources", [])),
                rerank_score=r.get("rerank_score", 0.0),
            )
            for r in final_results
        ]

        # 6. LLM 合成答案
        answer = ""
        if use_llm:
            answer = self._synthesize_answer(query, search_results)

        elapsed_ms = (time.time() - start_time) * 1000

        return SearchResponse(
            query=query,
            results=search_results,
            answer=answer,
            layer_stats={
                "layer1_bm25": len(layer_result.get("layer1_bm25", [])),
                "layer1_vector": len(layer_result.get("layer1_vector", [])),
                "layer1_merged": len(layer_result.get("layer1_merged", [])),
                "layer2_merged": len(layer_result.get("layer2_merged", [])),
                "final": len(search_results),
            },
            elapsed_ms=elapsed_ms,
        )

    def _synthesize_answer(self, query: str, results: List[SearchResult]) -> str:
        """
        使用 LLM 合成多路召回结果，生成最终答案。
        """
        api_key = self.llm_config.get("api_key", "")
        if not api_key or api_key == "your-api-key-here":
            # 无 LLM 时，返回拼接的上下文字符串
            return self._format_context(results)

        # 构建上下文
        context_parts = []
        for i, r in enumerate(results, 1):
            source = r.metadata.get("source_file", "未知来源")
            context_parts.append(f"[{i}] (来源: {source})\n{r.text}")
        context = "\n\n".join(context_parts)

        prompt = self.SYNTHESIS_PROMPT.format(query=query, context=context)

        try:
            return self._call_llm(prompt)
        except Exception as e:
            logger.warning(f"LLM 合成失败: {e}，使用原始上下文")
            return self._format_context(results)

    def _format_context(self, results: List[SearchResult]) -> str:
        """将检索结果格式化为上下文字符串"""
        if not results:
            return "未找到相关资料。"

        parts = []
        for i, r in enumerate(results, 1):
            source = r.metadata.get("source_file", "未知来源")
            score = r.score
            parts.append(f"--- [{i}] 相关度: {score:.2f} | 来源: {source} ---\n{r.text}")
        return "\n\n".join(parts)

    def _call_llm(self, prompt: str) -> str:
        """调用 LLM API（URL 从 settings 统一读取）"""
        import urllib.request
        import json
        import settings as _settings

        payload = {
            "model": self.llm_config.get("model", _settings.LLM_MODEL),
            "messages": [
                {"role": "system", "content": "你是一个专业的知识检索助手，基于参考资料回答问题。"},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.3,
            "max_tokens": 1024,
        }

        base = self.llm_config.get('base_url', _settings.LLM_BASE_URL).rstrip('/')
        req = urllib.request.Request(
            f"{base}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.llm_config.get('api_key', _settings.LLM_API_KEY)}",
            },
        )

        with urllib.request.urlopen(req, timeout=60) as resp:
            raw = resp.read()
            data = json.loads(raw)
            # 处理 API 错误响应
            if "error" in data:
                raise RuntimeError(f"LLM API 错误: {data['error'].get('message', str(data['error']))}")
            if "choices" not in data:
                raise RuntimeError(f"LLM API 返回异常: {json.dumps(data, ensure_ascii=False)[:500]}")
            return data["choices"][0]["message"]["content"]
