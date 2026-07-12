"""
RAG 数据预处理系统 — 算法参数配置

此文件存放算法/策略级的可调参数（分块策略、相似度阈值等）。
部署级配置（路径、密钥、连接信息）请修改 settings.py。
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class ChunkConfig:
    """语义分块参数"""
    # 相邻块语义相似度低于此值视为断点 [0, 1]
    similarity_threshold: float = 0.6
    # 单个 chunk 最大字符数
    max_chunk_chars: int = 1000
    # 单个 chunk 最小字符数（避免碎片化）
    min_chunk_chars: int = 80
    # 相邻 chunk 之间的重叠比例 (0~0.3)
    overlap_ratio: float = 0.15
    # 是否启用 chunk 去重（基于文本哈希）
    deduplicate: bool = True
    # 去重相似度阈值 (0~1)，高于此值视为重复
    dedup_threshold: float = 0.95


@dataclass
class FileTypeConfig:
    """支持的文件类型扩展名"""
    word_extensions: List[str] = field(default_factory=lambda: [".docx", ".doc"])
    pdf_extensions: List[str] = field(default_factory=lambda: [".pdf"])
    excel_extensions: List[str] = field(default_factory=lambda: [".xlsx", ".xls"])
    image_extensions: List[str] = field(
        default_factory=lambda: [".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"]
    )
    text_extensions: List[str] = field(
        default_factory=lambda: [".txt", ".md", ".csv", ".json", ".html", ".htm"]
    )
    pptx_extensions: List[str] = field(default_factory=lambda: [".pptx"])


