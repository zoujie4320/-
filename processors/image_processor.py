"""
图片 OCR 处理器

使用 pytesseract 直接对图片进行 OCR 文字识别。
支持常见图片格式：PNG, JPG, JPEG, BMP, TIFF

前置依赖: Tesseract OCR（同 PDF 处理器）
"""

from pathlib import Path
from typing import Dict, Any

from .base_processor import BaseProcessor

try:
    import pytesseract
    HAS_TESSERACT = True
except ImportError:
    HAS_TESSERACT = False

try:
    from PIL import Image, ImageFilter
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


class ImageProcessor(BaseProcessor):
    processor_name = "image_ocr"

    SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}

    def __init__(self, lang: str = "chi_sim+eng", preprocess: bool = True):
        """
        Args:
            lang: OCR语言，默认中英文
            preprocess: 是否进行图片预处理
        """
        self.lang = lang
        self.preprocess_enabled = preprocess

    def can_handle(self, file_path: str) -> bool:
        return Path(file_path).suffix.lower() in self.SUPPORTED_EXTENSIONS

    def extract_text(self, file_path: str) -> str:
        if not HAS_TESSERACT:
            raise ImportError(
                "缺少 pytesseract 库。请运行: pip install pytesseract\n"
                "同时需要安装 Tesseract OCR: https://github.com/UB-Mannheim/tesseract/wiki"
            )
        if not HAS_PIL:
            raise ImportError("缺少 Pillow 库。请运行: pip install Pillow")

        image = Image.open(file_path)

        if self.preprocess_enabled:
            image = self._preprocess_image(image)

        # OCR 识别
        text = pytesseract.image_to_string(image, lang=self.lang)

        # 同时获取带位置信息的识别结果（可用于后续布局分析）
        # data = pytesseract.image_to_data(image, lang=self.lang, output_type=pytesseract.Output.DICT)

        return text.strip()

    def extract_metadata(self, file_path: str) -> Dict[str, Any]:
        metadata = super().extract_metadata(file_path)
        try:
            with Image.open(file_path) as img:
                metadata["image_width"] = img.width
                metadata["image_height"] = img.height
                metadata["image_mode"] = img.mode
                metadata["image_format"] = img.format
        except Exception:
            pass
        metadata["ocr_lang"] = self.lang
        return metadata

    # --- 内部方法 ---

    def _preprocess_image(self, image: Image.Image) -> Image.Image:
        """
        图片预处理管线：灰度化 → 去噪 → 对比度增强 → 二值化
        """
        # 1. 转灰度
        if image.mode != 'L':
            image = image.convert('L')

        # 2. 去噪（中值滤波）
        try:
            image = image.filter(ImageFilter.MedianFilter(size=3))
        except Exception:
            pass

        # 3. 对比度拉伸
        try:
            import numpy as np
            np_img = np.array(image)
            p2, p98 = np.percentile(np_img, (2, 98))
            if p98 > p2:
                np_img = np.clip((np_img - p2) * 255.0 / (p98 - p2), 0, 255).astype(np.uint8)
            image = Image.fromarray(np_img)
        except Exception:
            pass

        return image
