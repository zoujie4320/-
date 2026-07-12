"""
处理器注册中心

新增处理器步骤:
1. 创建处理器文件，继承 BaseProcessor
2. 在此处导入并添加到 get_all_processors() 的列表中
3. 无需修改其他任何代码

处理器实例化时会自动从 settings.py 读取配置（OCR语言、DPI等）。
"""

import settings
from .base_processor import BaseProcessor
from .word_processor import WordProcessor
from .pdf_processor import PDFProcessor
from .excel_processor import ExcelProcessor
from .image_processor import ImageProcessor
from .text_processor import TextProcessor
from .pptx_processor import PptxProcessor
from .doc_processor import DocProcessor


def get_all_processors() -> list:
    """
    返回所有已注册的处理器实例。

    新增处理器时，在此列表中添加即可自动生效。
    处理器参数从 settings.py 自动注入。
    """
    return [
        WordProcessor(),
        PDFProcessor(
            dpi=settings.PDF_DPI,
            lang=settings.OCR_LANG,
        ),
        ExcelProcessor(),
        ImageProcessor(
            lang=settings.OCR_LANG,
            preprocess=settings.IMAGE_PREPROCESS,
        ),
        TextProcessor(),
        PptxProcessor(),
        DocProcessor(),
    ]


def find_processor(file_path: str) -> BaseProcessor:
    """
    根据文件路径自动匹配适合的处理器。

    Args:
        file_path: 文件路径

    Returns:
        匹配的处理器实例

    Raises:
        ValueError: 没有找到能处理此文件的处理器
    """
    for processor in get_all_processors():
        if processor.can_handle(file_path):
            return processor
    raise ValueError(f"不支持的文件类型: {file_path}")


__all__ = [
    "BaseProcessor",
    "WordProcessor",
    "PDFProcessor",
    "ExcelProcessor",
    "ImageProcessor",
    "TextProcessor",
    "PptxProcessor",
    "DocProcessor",
    "get_all_processors",
    "find_processor",
]
