"""
纯文本 / 标记文档处理器

支持格式:
- .txt   — 纯文本
- .md    — Markdown 文档
- .csv   — CSV 数据 (转为可读文本)
- .json  — JSON 数据 (展平为可读文本)
- .html/.htm — HTML 网页 (提取纯文本)
"""

import re
import json
import csv
from pathlib import Path
from typing import Dict, Any
from html.parser import HTMLParser

from .base_processor import BaseProcessor


class TextProcessor(BaseProcessor):
    processor_name = "text"
    SUPPORTED_EXTENSIONS = {".txt", ".md", ".csv", ".json", ".html", ".htm"}

    def can_handle(self, file_path: str) -> bool:
        return Path(file_path).suffix.lower() in self.SUPPORTED_EXTENSIONS

    def extract_text(self, file_path: str) -> str:
        ext = Path(file_path).suffix.lower()

        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            raw = f.read()

        if ext == ".json":
            return self._extract_json(raw)
        elif ext == ".csv":
            return self._extract_csv(raw)
        elif ext in (".html", ".htm"):
            return self._extract_html(raw)
        else:
            # .txt, .md 直接返回原文
            return raw.strip()

    def extract_metadata(self, file_path: str) -> Dict[str, Any]:
        metadata = super().extract_metadata(file_path)
        ext = Path(file_path).suffix.lower()
        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
            metadata["line_count"] = content.count('\n') + 1
            metadata["char_count"] = len(content)
            metadata["file_type"] = ext.lstrip('.')
        except Exception:
            pass
        return metadata

    # --- 内部方法 ---

    @staticmethod
    def _extract_json(raw: str) -> str:
        """将 JSON 数据展平为可读文本"""
        try:
            data = json.loads(raw)
            return TextProcessor._flatten_json(data)
        except json.JSONDecodeError:
            return raw.strip()

    @staticmethod
    def _flatten_json(data, prefix: str = "", depth: int = 0) -> str:
        """递归展平 JSON 结构（限制深度防止栈溢出）"""
        MAX_DEPTH = 50
        if depth > MAX_DEPTH:
            return f"{prefix}: [达到最大嵌套深度 {MAX_DEPTH}]"

        lines = []
        if isinstance(data, dict):
            for key, value in data.items():
                full_key = f"{prefix}.{key}" if prefix else key
                if isinstance(value, (dict, list)):
                    lines.append(f"{full_key}:")
                    lines.append(TextProcessor._flatten_json(value, full_key, depth + 1))
                else:
                    lines.append(f"{full_key}: {value}")
        elif isinstance(data, list):
            for i, item in enumerate(data):
                full_key = f"{prefix}[{i}]"
                if isinstance(item, (dict, list)):
                    lines.append(f"{full_key}:")
                    lines.append(TextProcessor._flatten_json(item, full_key, depth + 1))
                else:
                    lines.append(f"{full_key}: {item}")
        else:
            lines.append(str(data))
        return "\n".join(lines)

    @staticmethod
    def _extract_csv(raw: str) -> str:
        """将 CSV 转为可读文本表格"""
        lines = []
        try:
            reader = csv.reader(raw.splitlines())
            for row_idx, row in enumerate(reader):
                if row_idx == 0:
                    lines.append(" | ".join(row))
                    lines.append("-" * 40)
                else:
                    lines.append(" | ".join(row))
        except Exception:
            return raw.strip()
        return "\n".join(lines)

    @staticmethod
    def _extract_html(raw: str) -> str:
        """从 HTML 中提取纯文本"""

        class TextExtractor(HTMLParser):
            def __init__(self):
                super().__init__()
                self.texts = []
                self.skip_tags = {"script", "style", "noscript", "iframe"}
                self.current_skip = None

            def handle_starttag(self, tag, attrs):
                if tag in self.skip_tags:
                    self.current_skip = tag
                elif tag in ("p", "br", "div", "li", "h1", "h2", "h3", "h4", "h5", "h6", "tr"):
                    self.texts.append("\n")

            def handle_endtag(self, tag):
                if tag == self.current_skip:
                    self.current_skip = None
                if tag in ("p", "div", "li", "h1", "h2", "h3", "h4", "h5", "h6"):
                    self.texts.append("\n")

            def handle_data(self, data):
                if self.current_skip:
                    return
                text = data.strip()
                if text:
                    self.texts.append(text)

        extractor = TextExtractor()
        extractor.feed(raw)
        text = " ".join(extractor.texts)
        # 清理多余空白
        text = re.sub(r' {2,}', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()
