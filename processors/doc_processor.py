"""
旧版 Word (.doc) 二进制文档处理器

支持 Microsoft Word 97-2003 二进制格式 (.doc)。

与 .docx 不同，.doc 是 OLE2 复合文档格式，python-docx 无法处理。
本处理器使用多级回退策略提取文本:

策略优先级（自动降级）:
  1. LibreOffice 无头模式 — 跨平台首选 (Windows/macOS/Linux)
  2. antiword 命令行工具 — Linux/macOS 轻量方案
  3. python-pptx/textract 包装库 — 如果有 Word 环境

前置依赖 (至少安装一种):
  - LibreOffice: https://www.libreoffice.org/download/
    安装后将 soffice.exe 所在目录加入 PATH
  - antiword (Linux): sudo apt install antiword
  - antiword (macOS): brew install antiword
"""

import os
import re
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, Optional

from .base_processor import BaseProcessor


class DocProcessor(BaseProcessor):
    processor_name = "doc"
    SUPPORTED_EXTENSIONS = {".doc"}

    # 搜索 LibreOffice 的常见路径
    _LIBREOFFICE_PATHS = [
        # Windows
        "soffice.exe",
        "C:\\Program Files\\LibreOffice\\program\\soffice.exe",
        "C:\\Program Files (x86)\\LibreOffice\\program\\soffice.exe",
        # macOS
        "/Applications/LibreOffice.app/Contents/MacOS/soffice",
        # Linux
        "soffice",
        "/usr/bin/soffice",
        "/usr/lib/libreoffice/program/soffice",
    ]

    def can_handle(self, file_path: str) -> bool:
        return Path(file_path).suffix.lower() == ".doc"

    def extract_text(self, file_path: str) -> str:
        # 策略 1: LibreOffice
        text = self._try_libreoffice(file_path)
        if text:
            return text

        # 策略 2: antiword
        text = self._try_antiword(file_path)
        if text:
            return text

        # 策略 3: textract (Python 包装)
        text = self._try_textract(file_path)
        if text:
            return text

        # 全部失败
        raise RuntimeError(
            "无法处理 .doc 文件，所有提取方式均失败。\n\n"
            "请安装以下任一工具:\n"
            "  1. LibreOffice (推荐): https://www.libreoffice.org/download/\n"
            "     安装后确保 soffice 在 PATH 中\n"
            "  2. antiword:\n"
            "     - Linux: sudo apt install antiword\n"
            "     - macOS: brew install antiword\n"
            "  3. textract (Python): pip install textract\n"
        )

    def extract_metadata(self, file_path: str) -> Dict[str, Any]:
        metadata = super().extract_metadata(file_path)
        metadata["file_format"] = "Microsoft Word 97-2003 (.doc)"
        return metadata

    # ========================================================================
    # 策略 1: LibreOffice 无头模式
    # ========================================================================

    def _try_libreoffice(self, file_path: str) -> Optional[str]:
        """使用 LibreOffice 无头模式将 .doc 转为纯文本"""
        soffice = self._find_libreoffice()
        if not soffice:
            return None

        # 创建临时目录存放转换结果
        tmp_dir = tempfile.mkdtemp(prefix="rag_doc_")
        try:
            cmd = [
                soffice,
                "--headless",
                "--convert-to", "txt:Text",
                "--outdir", tmp_dir,
                file_path,
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120,  # 2 分钟超时，大文件需更长时间
            )

            # 查找生成的 .txt 文件
            input_stem = Path(file_path).stem
            for f in Path(tmp_dir).glob("*.txt"):
                text = f.read_text(encoding="utf-8", errors="replace")
                if text.strip():
                    return self._clean_doc_text(text)

            # 有时文件名不完全匹配，尝试所有txt
            for f in Path(tmp_dir).glob("*.txt"):
                text = f.read_text(encoding="utf-8", errors="replace")
                if text.strip():
                    return self._clean_doc_text(text)

        except FileNotFoundError:
            return None
        except subprocess.TimeoutExpired:
            # LibreOffice 超时，可能文件过大
            return None
        except Exception:
            return None
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

        return None

    def _find_libreoffice(self) -> Optional[str]:
        """查找 LibreOffice 可执行文件"""
        for path in self._LIBREOFFICE_PATHS:
            if os.path.exists(path):
                return path

        # 尝试通过 shutil.which 在 PATH 中查找
        which = shutil.which("soffice")
        if which:
            return which

        return None

    # ========================================================================
    # 策略 2: antiword (Linux/macOS 轻量工具)
    # ========================================================================

    def _try_antiword(self, file_path: str) -> Optional[str]:
        """使用 antiword 提取 .doc 文本"""
        antiword = shutil.which("antiword")
        if not antiword:
            return None

        try:
            result = subprocess.run(
                [antiword, file_path],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0 and result.stdout.strip():
                text = self._clean_doc_text(result.stdout)
                # 过滤 antiword 的 banner 信息
                text = re.sub(r'^.*antiword.*\n?', '', text, flags=re.IGNORECASE)
                return text.strip() or None
        except Exception:
            pass

        return None

    # ========================================================================
    # 策略 3: textract (Python 包装库)
    # ========================================================================

    def _try_textract(self, file_path: str) -> Optional[str]:
        """使用 textract 提取 .doc 文本"""
        try:
            import textract
            raw = textract.process(file_path)
            if isinstance(raw, bytes):
                text = raw.decode("utf-8", errors="replace")
            else:
                text = str(raw)

            if text.strip():
                return self._clean_doc_text(text)
        except ImportError:
            pass
        except Exception:
            pass

        return None

    # ========================================================================
    # 文本清理
    # ========================================================================

    @staticmethod
    def _clean_doc_text(text: str) -> str:
        """
        清理从 .doc 提取的文本：
        - 移除过多的空行
        - 合并多余空格
        - 修复常见转码问题
        """
        # 移除控制字符（保留换行和制表符）
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)

        # 三行以上空行 → 双行
        text = re.sub(r'\n{3,}', '\n\n', text)

        # 多余空格
        text = re.sub(r' {2,}', ' ', text)

        # 去除每行首尾空白
        lines = [line.strip() for line in text.split('\n')]
        text = '\n'.join(lines)

        return text.strip()
