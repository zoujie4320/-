"""
重排序器 (Reranker)

第 3 层：对召回结果进行精细排序。

支持两种模式:
1. CrossEncoder 模型重排 — 使用 sentence-transformers CrossEncoder
2. LLM 重排 — 使用配置的大模型逐条打分

使用方式:
    reranker = CrossEncoderReranker()
    reranked = reranker.rerank(query, [result1, result2, ...], top_k=5)
"""

import logging
from typing import List, Dict, Any, Optional
import numpy as np

logger = logging.getLogger(__name__)


class Reranker:
    """重排序器基类"""

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        raise NotImplementedError


class CrossEncoderReranker(Reranker):
    """
    使用 CrossEncoder 模型进行精排。

    CrossEncoder 同时编码 query 和 document，比双塔模型更精确，
    但速度较慢，适合对召回结果的 top-N 进行重排序。

    默认模型: cross-encoder/ms-marco-MiniLM-L-6-v2 (英文)
    中文推荐: BAAI/bge-reranker-base
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self._model = None

    @property
    def model(self):
        if self._model is None:
            try:
                self._load_model()
            except Exception as e:
                logger.warning(f"CrossEncoder 模型加载失败: {e}")
                self._model = False  # 标记为失败，不再重试
        return self._model if self._model is not False else None

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        if not candidates:
            return []

        try:
            pairs = [(query, c["text"]) for c in candidates]
            scores = self.model.predict(pairs, show_progress_bar=False)

            # 归一化到 [0, 1]
            scores = np.array(scores)
            scores_min, scores_max = scores.min(), scores.max()
            if scores_max > scores_min:
                scores = (scores - scores_min) / (scores_max - scores_min)

            # 排序
            for i, c in enumerate(candidates):
                c["rerank_score"] = float(scores[i])
                c["final_score"] = float(scores[i])

            ranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
            return ranked[:top_k]

        except Exception as e:
            logger.warning(f"CrossEncoder 重排失败: {e}，回退到原始顺序")
            return candidates[:top_k]

    def _load_model(self):
        try:
            from sentence_transformers import CrossEncoder
            logger.info(f"加载 CrossEncoder 模型: {self.model_name}")
            self._model = CrossEncoder(self.model_name)
        except ImportError:
            raise ImportError("需要 sentence-transformers 库: pip install sentence-transformers")
        except Exception as e:
            logger.warning(f"模型加载失败: {e}，将使用原始顺序")
            self._model = None


class LLMReranker(Reranker):
    """
    使用 LLM 对搜索结果进行重排序。

    向 LLM 发送 query + 候选文档列表，让 LLM 按相关性排序。
    适合对语义理解要求极高的场景。

    使用方式:
        reranker = LLMReranker(api_key="...", base_url="...", model="gpt-4o")
        results = reranker.rerank(query, candidates, top_k=5)
    """

    PROMPT_TEMPLATE = """你是一个专业的文档检索排序助手。请根据用户查询，对以下文档片段按相关性从高到低排序。

## 用户查询
{query}

## 候选文档片段
{candidates}

## 任务
分析每个文档与查询的相关性，返回排序后的文档编号列表（从最相关到最不相关）。

返回格式（仅返回编号列表，每行一个，不要其他文字）:
3
1
5
2
4
"""

    def __init__(
        self,
        api_key: str = None,
        base_url: str = None,
        model: str = None,
    ):
        import settings as _settings
        self.api_key = api_key or _settings.LLM_API_KEY
        self.base_url = base_url or _settings.LLM_BASE_URL
        self.model = model or _settings.LLM_MODEL

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        if not candidates:
            return []

        if not self.api_key or self.api_key == "your-api-key-here":
            logger.warning("LLM API Key 未配置，回退到原始顺序。请在 settings.py 中设置 LLM_API_KEY 或设置环境变量 OPENAI_API_KEY")
            return candidates[:top_k]

        try:
            # 构建候选列表
            candidate_text = ""
            for i, c in enumerate(candidates, 1):
                text_preview = c["text"][:300].replace("\n", " ")
                candidate_text += f"[{i}] {text_preview}\n"

            prompt = self.PROMPT_TEMPLATE.format(
                query=query,
                candidates=candidate_text,
            )

            # 调用 LLM
            ranked_indices = self._call_llm(prompt, len(candidates))

            # 按 LLM 排序重建结果
            ranked = []
            for idx in ranked_indices:
                if 1 <= idx <= len(candidates):
                    c = candidates[idx - 1].copy()
                    c["rerank_score"] = 1.0 - (ranked_indices.index(idx) / len(ranked_indices))
                    c["final_score"] = c["rerank_score"]
                    ranked.append(c)

            return ranked[:top_k]

        except Exception as e:
            logger.warning(f"LLM 重排失败: {e}，回退到原始顺序")
            return candidates[:top_k]

    def _call_llm(self, prompt: str, num_candidates: int) -> List[int]:
        """调用 LLM API"""
        import urllib.request
        import json as json_mod

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "你是一个精确的文档排序助手。只返回排序后的编号，每行一个。"},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0,
            "max_tokens": 200,
        }

        base = self.base_url.rstrip('/')
        req = urllib.request.Request(
            f"{base}/chat/completions",
            data=json_mod.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )

        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json_mod.loads(resp.read())
            content = data["choices"][0]["message"]["content"]

        # 解析返回的编号
        import re
        indices = []
        for line in content.strip().split("\n"):
            match = re.search(r'\b(\d+)\b', line)
            if match:
                idx = int(match.group(1))
                if 1 <= idx <= num_candidates and idx not in indices:
                    indices.append(idx)

        # 补充缺失的编号
        for i in range(1, num_candidates + 1):
            if i not in indices:
                indices.append(i)

        return indices
