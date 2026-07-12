"""
语义分块器

核心算法:
1. 初始分割: 按段落→句子的层级将文本切分为基本单元
2. 语义合并: 使用嵌入模型计算相邻文本块的余弦相似度
3. 贪婪合并: 相似度 ≥ 阈值的相邻块合并，遇语义断点则切分
4. 重叠窗口: 相邻 chunk 之间保留部分重叠文本

这样确保切割位置在"语义变向"处，不会在知识点中间切断。
"""

import re
from typing import List, Dict, Any, Optional
import numpy as np

from utils.helpers import cosine_similarity, normalize_text


class SemanticChunker:
    """
    基于语义相似度的自适应文本分块器。

    使用方法:
        embedder = Embedder()
        chunker = SemanticChunker(embedder)
        chunks = chunker.chunk("这是一段很长的文本...")
    """

    def __init__(
        self,
        embedder,  # Embedder 实例，用于计算语义相似度
        similarity_threshold: float = 0.6,
        max_chunk_chars: int = 1000,
        min_chunk_chars: int = 80,
        overlap_ratio: float = 0.15,
    ):
        """
        Args:
            embedder: Embedder 实例
            similarity_threshold: 合并相似度阈值 (0~1)，低于此值视为语义断点
            max_chunk_chars: 单个 chunk 最大字符数
            min_chunk_chars: 单个 chunk 最小字符数（低于此值尝试与相邻块合并）
            overlap_ratio: chunk 间重叠比例 (0~0.3)
        """
        self.embedder = embedder
        self.similarity_threshold = similarity_threshold
        self.max_chunk_chars = max_chunk_chars
        self.min_chunk_chars = min_chunk_chars
        self.overlap_ratio = overlap_ratio

    def chunk(self, text: str) -> List[Dict[str, Any]]:
        """
        对文本进行语义分块。

        Args:
            text: 输入文本

        Returns:
            chunk 列表，每个 chunk 包含:
                - text: chunk文本
                - char_count: 字符数
                - start_pos: 原文起始位置
                - end_pos: 原文结束位置
        """
        if not text or not text.strip():
            return []

        text = normalize_text(text)

        # Step 1: 初始分割 —— 按自然边界切分成基本单元
        segments = self._initial_split(text)
        if len(segments) <= 1:
            return [{"text": text, "char_count": len(text), "start_pos": 0, "end_pos": len(text)}]

        # Step 2: 计算所有 segment 的嵌入向量（批量）
        segment_embeddings = self.embedder.encode(segments)

        # Step 3: 基于语义相似度的贪婪合并
        merged_chunks = self._greedy_merge(segments, segment_embeddings)

        # Step 4: 处理过小的 chunk（合并到相邻块）
        merged_chunks = self._handle_small_chunks(merged_chunks)

        # Step 5: 添加重叠窗口
        final_chunks = self._add_overlap(merged_chunks)

        return final_chunks

    # --- 内部方法 ---

    def _initial_split(self, text: str) -> List[str]:
        """
        将文本按自然边界切分为基本单元。
        优先级: 段落 > 长句分割 > 句子
        """
        # 1. 按段落切分
        paragraphs = re.split(r'\n\s*\n', text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        # 2. 对过长的段落进一步按句子切分
        segments = []
        for para in paragraphs:
            if len(para) > self.max_chunk_chars:
                # 按句子边界细分
                sentences = self._split_by_sentences(para)
                # 合并过短的句子到相邻句
                sentences = self._merge_short_sentences(sentences)
                segments.extend(sentences)
            else:
                segments.append(para)

        return segments

    @staticmethod
    def _split_by_sentences(text: str) -> List[str]:
        """
        按句子边界切分文本。
        支持中英文混合场景：
        - 中文句号/问号/感叹号/分号
        - 英文句号后跟空格和大写字母
        - 换行
        """
        # 在标点符号后添加切分标记
        # 中英文句子结束标点: 。！？!? — 换行
        pattern = r'(?<=[。！？!?\n])\s*'
        parts = re.split(pattern, text)

        # 对英文句号进一步处理（不在数字或缩写中的句号）
        refined = []
        for part in parts:
            if not part.strip():
                continue
            # 尝试在 ". Xxx" 模式处切分 (英文句号+空格+大写)
            sub_parts = re.split(r'(?<=[a-z])\.\s+(?=[A-Z])', part)
            refined.extend([p.strip() for p in sub_parts if p.strip()])

        return refined

    def _merge_short_sentences(self, sentences: List[str]) -> List[str]:
        """合并过短的句子到其相邻句子，避免碎片化"""
        if len(sentences) <= 1:
            return sentences

        merged = []
        buffer = ""
        for sent in sentences:
            if buffer and len(buffer) < self.min_chunk_chars:
                buffer += " " + sent
            elif len(sent) < self.min_chunk_chars // 2:
                buffer += " " + sent if buffer else sent
            else:
                if buffer:
                    merged.append(buffer)
                buffer = sent

        if buffer:
            # 最后一个 buffer 如果太短，合并到上一个
            if merged and len(buffer) < self.min_chunk_chars // 2:
                merged[-1] += " " + buffer
            else:
                merged.append(buffer)

        return merged

    def _greedy_merge(
        self,
        segments: List[str],
        embeddings: np.ndarray,
    ) -> List[Dict[str, Any]]:
        """
        基于嵌入相似度的贪婪合并算法。

        遍历 segments，若当前 chunk 与下一个 segment 语义相似度 ≥ 阈值，
        则合并；否则完成当前 chunk 并开始新 chunk。
        """
        if len(segments) == 0:
            return []
        if len(segments) == 1:
            return [{"text": segments[0], "char_count": len(segments[0]), "segments": [segments[0]]}]

        chunks = []
        current_text = segments[0]
        current_embedding = embeddings[0].copy()
        current_segments = [segments[0]]
        segment_count = 1

        for i in range(1, len(segments)):
            next_seg = segments[i]
            next_emb = embeddings[i]

            # 计算当前 chunk（均值向量）与下一个 segment 的相似度
            similarity = cosine_similarity(current_embedding, next_emb)

            combined_length = len(current_text) + len(next_seg)

            # 判断是否合并
            should_merge = (
                similarity >= self.similarity_threshold
                and combined_length <= self.max_chunk_chars * 1.2  # 允许略微超出
            )

            if should_merge:
                # 合并
                current_text += "\n\n" + next_seg
                # 更新均值向量
                current_embedding = (current_embedding * segment_count + next_emb) / (segment_count + 1)
                current_segments.append(next_seg)
                segment_count += 1
            else:
                # 当前 chunk 完成
                if current_text.strip():
                    chunks.append({
                        "text": current_text.strip(),
                        "char_count": len(current_text.strip()),
                        "segments": current_segments,
                    })
                # 开始新 chunk
                current_text = next_seg
                current_embedding = next_emb.copy()
                current_segments = [next_seg]
                segment_count = 1

        # 最后一个 chunk
        if current_text.strip():
            chunks.append({
                "text": current_text.strip(),
                "char_count": len(current_text.strip()),
                "segments": current_segments,
            })

        return chunks

    def _handle_small_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        处理过小的 chunk：尝试合并到相邻（相似度更高的）chunk。
        """
        if len(chunks) <= 1:
            return chunks

        # 找出过小的 chunk
        result = []
        i = 0
        while i < len(chunks):
            chunk = chunks[i]
            if chunk["char_count"] >= self.min_chunk_chars or len(result) == 0:
                result.append(chunk)
                i += 1
            else:
                # 太小了，合并到前一个
                prev = result[-1]
                prev["text"] += "\n\n" + chunk["text"]
                prev["char_count"] = len(prev["text"])
                prev["segments"].extend(chunk["segments"])
                i += 1

        return result

    def _add_overlap(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        在相邻 chunk 之间添加文本重叠。
        从前一个 chunk 的末尾取部分文本加到后一个 chunk 的开头。
        """
        if len(chunks) <= 1 or self.overlap_ratio <= 0:
            return chunks

        result = [chunks[0].copy()]

        for i in range(1, len(chunks)):
            prev_original = chunks[i - 1]     # 取原始前一个chunk，避免重叠累积
            current = chunks[i].copy()

            # 从前一个 chunk 的原始末尾取 overlap 文本
            overlap_chars = int(len(current["text"]) * self.overlap_ratio)
            overlap_chars = min(overlap_chars, len(prev_original["text"]) // 3)

            if overlap_chars > 0:
                # 尝试在句子边界处截断
                overlap_text = prev_original["text"][-overlap_chars:]
                # 找到第一个完整句子的开始位置
                boundary = self._find_sentence_start(overlap_text)
                if boundary > 0:
                    overlap_text = overlap_text[boundary:]

                if overlap_text.strip():
                    current["text"] = overlap_text.strip() + "\n\n" + current["text"]
                    current["char_count"] = len(current["text"])
                    current["overlap_from_prev"] = len(overlap_text)

            result.append(current)

        return result

    @staticmethod
    def _find_sentence_start(text: str) -> int:
        """
        在文本中寻找第一个句子边界的起始位置。
        用于确保 overlap 从完整的句子开始。
        """
        for i, ch in enumerate(text):
            if ch in '。！？!?\n' and i + 1 < len(text):
                return i + 1
        return 0
