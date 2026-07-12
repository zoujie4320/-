"""
PDF 文档处理器

双路径策略:
1. 优先使用 PyMuPDF (fitz) 直接提取文本 — 适合文字型 PDF
2. 若提取文本为空（扫描件），回退到 OCR (pytesseract + pdf2image)

前置依赖:
- PyMuPDF:  pip install PyMuPDF
- Tesseract OCR: https://github.com/UB-Mannheim/tesseract/wiki
- Poppler: http://blog.alivate.com.au/poppler-windows.html (仅 OCR 回退时需要)
"""

import os
from pathlib import Path
from typing import Dict, Any, List

from .base_processor import BaseProcessor

# --- 依赖检查 ---
try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False

try:
    from pdf2image import convert_from_path
    HAS_PDF2IMAGE = True
except ImportError:
    HAS_PDF2IMAGE = False

try:
    import pytesseract
    HAS_TESSERACT = True
except ImportError:
    HAS_TESSERACT = False


class PDFProcessor(BaseProcessor):
    processor_name = "pdf"
    SUPPORTED_EXTENSIONS = {".pdf"}

    def __init__(self, dpi: int = 300, lang: str = "chi_sim+eng"):
        """
        Args:
            dpi: OCR 时的分辨率（仅回退 OCR 时有效）
            lang: OCR 语言
        """
        self.dpi = dpi
        self.lang = lang

    def can_handle(self, file_path: str) -> bool:
        return Path(file_path).suffix.lower() == ".pdf"

    def extract_text(self, file_path: str) -> str:
        # 路径1: 尝试 PyMuPDF 直接提取
        if HAS_PYMUPDF:
            text = self._extract_with_pymupdf(file_path)
            if text.strip():
                return text.strip()
            # 文本为空 → 扫描件，回退到 OCR

        # 路径2: OCR 回退
        if not HAS_PYMUPDF:
            # 没有 PyMuPDF，直接走 OCR
            pass

        return self._extract_with_ocr(file_path)

    def extract_metadata(self, file_path: str) -> Dict[str, Any]:
        metadata = super().extract_metadata(file_path)
        try:
            if HAS_PYMUPDF:
                doc = fitz.open(file_path)
                metadata["page_count"] = len(doc)
                metadata["extraction_method"] = "pymupdf"
                # 提取 PDF 信息
                pdf_meta = doc.metadata or {}
                metadata["pdf_title"] = pdf_meta.get("title", "")
                metadata["pdf_author"] = pdf_meta.get("author", "")
                metadata["pdf_subject"] = pdf_meta.get("subject", "")
                doc.close()
            else:
                metadata["extraction_method"] = "ocr"
                try:
                    from pdf2image.pdf2image import pdfinfo_from_path
                    info = pdfinfo_from_path(file_path)
                    metadata["page_count"] = info.get("Pages", 0)
                except Exception:
                    metadata["page_count"] = 0
        except Exception:
            metadata["page_count"] = 0
            metadata["extraction_method"] = "unknown"

        metadata["ocr_dpi"] = self.dpi
        metadata["ocr_lang"] = self.lang
        return metadata

    # --- 内部方法 ---

    def _extract_with_pymupdf(self, file_path: str) -> str:
        """使用 PyMuPDF 逐页提取文本（内存友好）"""
        doc = fitz.open(file_path)
        text_parts: List[str] = []

        for page_idx in range(len(doc)):
            page = doc[page_idx]
            page_text = page.get_text("text")  # 纯文本模式
            if page_text.strip():
                text_parts.append(f"[第 {page_idx + 1} 页]\n{page_text.strip()}")

            # 提取表格（如果存在）
            try:
                tables = page.find_tables()
                if tables:
                    for table in tables:
                        table_text = self._format_table(table.extract())
                        if table_text:
                            text_parts.append(f"[表格 - 第 {page_idx + 1} 页]\n{table_text}")
            except Exception:
                pass  # 表格提取是可选的

        doc.close()
        return "\n\n".join(text_parts)

    def _extract_with_ocr(self, file_path: str) -> str:
        """使用 OCR 识别 PDF（扫描件回退方案）"""
        self._check_ocr_dependencies()

        # 逐批转换，避免一次性加载所有页面到内存
        # 先获取总页数
        from pdf2image.pdf2image import pdfinfo_from_path
        info = pdfinfo_from_path(file_path)
        total_pages = info["Pages"]

        text_parts = []
        # 每次处理 10 页，避免内存溢出
        batch_size = 10

        for start_page in range(1, total_pages + 1, batch_size):
            end_page = min(start_page + batch_size - 1, total_pages)
            images = convert_from_path(
                file_path,
                dpi=self.dpi,
                first_page=start_page,
                last_page=end_page,
                grayscale=True,
            )

            for i, image in enumerate(images):
                page_num = start_page + i
                processed = self._preprocess_image(image)
                page_text = pytesseract.image_to_string(processed, lang=self.lang)

                if page_text.strip():
                    text_parts.append(f"[第 {page_num} 页]\n{page_text.strip()}")

        return "\n\n".join(text_parts)

    def _check_ocr_dependencies(self):
        """检查 OCR 必要的依赖是否可用"""
        if not HAS_PDF2IMAGE:
            raise ImportError(
                "PDF OCR 需要 pdf2image 库。请运行: pip install pdf2image\n"
                "同时需要安装 Poppler:\n"
                "  Windows: https://github.com/oschwartz10612/poppler-windows/releases/\n"
                "  macOS: brew install poppler\n"
                "  Linux: sudo apt install poppler-utils"
            )
        if not HAS_TESSERACT:
            raise ImportError(
                "PDF OCR 需要 pytesseract 库。请运行: pip install pytesseract\n"
                "同时需要安装 Tesseract OCR:\n"
                "  https://github.com/UB-Mannheim/tesseract/wiki"
            )
        # 验证 Poppler 二进制是否可用（pdf2image 可能已安装但 poppler 未安装）
        try:
            from pdf2image.pdf2image import pdfinfo_from_path
        except Exception:
            raise ImportError(
                "pdf2image 已安装但 Poppler 二进制文件未找到。\n"
                "请安装 Poppler 并确保其在 PATH 中:\n"
                "  Windows: https://github.com/oschwartz10612/poppler-windows/releases/\n"
                "  macOS: brew install poppler\n"
                "  Linux: sudo apt install poppler-utils"
            )

    def _preprocess_image(self, image):
        """图片预处理：转灰度 → 二值化，提升 OCR 准确率"""
        img = image.convert('L')  # 灰度化
        try:
            import numpy as np
            from PIL import ImageFilter
            img = img.filter(ImageFilter.SHARPEN)
            np_img = np.array(img)
            threshold = self._otsu_threshold(np_img)
            img = img.point(lambda x: 0 if x < threshold else 255, '1')
        except Exception:
            pass
        return img

    @staticmethod
    def _otsu_threshold(np_img) -> int:
        """大津法计算二值化阈值"""
        try:
            import numpy as np
            pixel_values = np_img.ravel()
            total = len(pixel_values)
            if total == 0:
                return 128
            hist, _ = np.histogram(pixel_values, bins=256, range=(0, 256))
            sum_all = np.dot(np.arange(256), hist)
            weight_bg, sum_bg, max_variance, best_threshold = 0, 0, 0, 128
            for t in range(256):
                weight_bg += hist[t]
                if weight_bg == 0:
                    continue
                weight_fg = total - weight_bg
                if weight_fg == 0:
                    break
                sum_bg += t * hist[t]
                mean_bg = sum_bg / weight_bg
                mean_fg = (sum_all - sum_bg) / weight_fg
                variance = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
                if variance > max_variance:
                    max_variance = variance
                    best_threshold = t
            return best_threshold
        except Exception:
            return 128

    @staticmethod
    def _format_table(table_data) -> str:
        """将表格数据格式化为可读文本"""
        if not table_data:
            return ""
        rows = []
        for row in table_data:
            cells = [str(cell).strip() if cell else "" for cell in row]
            if any(cells):
                rows.append(" | ".join(cells))
        return "\n".join(rows) if rows else ""
