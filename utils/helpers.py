"""
通用工具函数
"""

import os
import re
import time
import hashlib
import numpy as np
from pathlib import Path
from typing import List, Set, Tuple


# ============================================================================
# Token 估算
# ============================================================================

# 更完整的 CJK 字符范围
_CJK_PATTERN = re.compile(
    r'[一-鿿㐀-䶿豈-﫿⺀-⻿　-〿㇀-㇯︐-︟︰-﹏]'
)


def estimate_tokens(text: str) -> int:
    """
    估算文本的 token 数量。
    中文约每字 1.5 token，英文约每词 1.3 token。
    """
    chinese_chars = len(_CJK_PATTERN.findall(text))
    english_words = len(re.findall(r'[a-zA-Z]+', text))
    other_chars = max(0, len(text) - chinese_chars - len(re.findall(r'[a-zA-Z\s]', text)))
    return int(chinese_chars * 1.5 + english_words * 1.3 + other_chars * 0.5)


# ============================================================================
# 文件工具
# ============================================================================

def get_file_extension(file_path: str) -> str:
    """获取文件扩展名（小写）"""
    return Path(file_path).suffix.lower()


def ensure_output_dir(output_dir: str) -> str:
    """确保输出目录存在，返回绝对路径"""
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return str(path.absolute())


def compute_file_hash(file_path: str, algorithm: str = "md5") -> str:
    """计算文件的哈希值（用于增量检测）"""
    h = hashlib.new(algorithm)
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


# ============================================================================
# 时间格式化
# ============================================================================

def format_duration(seconds: float) -> str:
    """格式化时间间隔"""
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    else:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.0f}s"


# ============================================================================
# 向量运算
# ============================================================================

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    计算两个向量（已L2归一化）的余弦相似度。
    如果向量未归一化则自动归一化。
    """
    # 检查是否已归一化（范数≈1）
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    if abs(a_norm - 1.0) > 1e-6:
        a = a / (a_norm + 1e-10)
    if abs(b_norm - 1.0) > 1e-6:
        b = b / (b_norm + 1e-10)
    return float(np.dot(a, b))


def batch_cosine_similarity(query: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    """
    计算一个查询向量与一组候选向量的余弦相似度。
    query: (dim,) 或 (1, dim)
    candidates: (n, dim)
    返回: (n,) 相似度数组
    """
    query = query.reshape(1, -1)
    # 归一化
    query_norm = query / (np.linalg.norm(query, axis=1, keepdims=True) + 1e-10)
    cand_norms = candidates / (np.linalg.norm(candidates, axis=1, keepdims=True) + 1e-10)
    return (query_norm @ cand_norms.T).flatten()


# ============================================================================
# 文本清洗
# ============================================================================

# 常见 OCR 错误修正映射
_OCR_FIXES = {
    '‘': "'", '’': "'",  # 弯引号 → 直引号
    '“': '"', '”': '"',
    '–': '-', '—': '--',  # 短破折号
    ' ': ' ',                   # 不间断空格
    '　': ' ',                   # 全角空格
}


def clean_text(text: str) -> str:
    """
    清洗文本：规范化空白、修复常见 OCR 错误、去除噪声。
    """
    if not text:
        return ""

    # 1. 统一 Unicode 字符
    for old, new in _OCR_FIXES.items():
        text = text.replace(old, new)

    # 2. 去除行内多余空格
    text = re.sub(r' {2,}', ' ', text)

    # 3. 合并过多换行（保留最多2个连续换行 = 段落分隔）
    text = re.sub(r'\n{3,}', '\n\n', text)

    # 4. 去除每行首尾空白
    lines = [line.strip() for line in text.split('\n')]
    text = '\n'.join(lines)

    # 5. 去除全空白行组成的连续段落分隔
    text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)

    # 6. OCR 常见噪声：单字符行（可能是识别碎片）
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        # 保留有意义的行（长度>1 或 是有效单字/CJK字符）
        if len(stripped) > 1 or (len(stripped) == 1 and stripped.isalnum()):
            cleaned_lines.append(line)
    text = '\n'.join(cleaned_lines)

    return text.strip()


def normalize_text(text: str) -> str:
    """
    规范化文本：合并多余空白、统一换行。
    （轻量版，用于分块前处理）
    """
    text = re.sub(r' {2,}', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    lines = [line.strip() for line in text.split('\n')]
    return '\n'.join(lines)


# ============================================================================
# Chunk 去重
# ============================================================================

def compute_text_hash(text: str) -> str:
    """计算文本的规范化哈希值（用于去重检测）"""
    # 规范化：小写 + 去空白
    normalized = re.sub(r'\s+', '', text.lower())
    return hashlib.md5(normalized.encode('utf-8')).hexdigest()


def deduplicate_chunks(
    chunks: List[dict],
    embeddings: np.ndarray,
    threshold: float = 0.95,
) -> Tuple[List[dict], np.ndarray]:
    """
    基于语义相似度去除重复/高度相似的 chunk。

    Args:
        chunks: chunk 字典列表 (含 'text' 字段)
        embeddings: 对应的嵌入向量 (n, dim)
        threshold: 相似度阈值，高于此值视为重复

    Returns:
        (去重后的 chunks, 去重后的 embeddings)
    """
    if len(chunks) <= 1:
        return chunks, embeddings

    keep_indices = [0]  # 保留第一个
    kept_embeddings = [embeddings[0]]

    for i in range(1, len(embeddings)):
        # 快速检查：精确哈希去重
        current_hash = compute_text_hash(chunks[i]["text"])
        is_dup = False
        for j in keep_indices:
            if compute_text_hash(chunks[j]["text"]) == current_hash:
                is_dup = True
                break

        if is_dup:
            continue

        # 语义去重：与已保留的最近 chunk 比较
        sims = batch_cosine_similarity(embeddings[i], np.array(kept_embeddings))
        if np.max(sims) >= threshold:
            continue  # 高度相似，跳过

        keep_indices.append(i)
        kept_embeddings.append(embeddings[i])

    filtered_chunks = [chunks[i] for i in keep_indices]
    filtered_embeddings = np.array(kept_embeddings)

    return filtered_chunks, filtered_embeddings


# ============================================================================
# 文本分割
# ============================================================================

def split_sentences(text: str) -> List[str]:
    """
    将文本按句子边界切分。
    支持中英文混合：中文句号/问号/感叹号、英文句号后跟大写字母、换行。
    """
    # 中英文句子结束标点
    pattern = r'(?<=[。！？!?\n])\s*'
    parts = re.split(pattern, text)

    refined = []
    for part in parts:
        if not part.strip():
            continue
        # 英文句号后跟空格+大写字母 → 句子边界
        sub_parts = re.split(r'(?<=[a-z])\.\s+(?=[A-Z])', part)
        refined.extend([p.strip() for p in sub_parts if p.strip()])

    return refined


def split_paragraphs(text: str) -> List[str]:
    """按段落（双换行或更多）切分文本"""
    paragraphs = re.split(r'\n\s*\n', text)
    return [p.strip() for p in paragraphs if p.strip()]
