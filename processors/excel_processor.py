"""
Excel 表格处理器

支持 .xlsx 和 .xls 格式。
遍历所有工作表，提取非空单元格内容，保留表格结构信息。
"""

from pathlib import Path
from typing import Dict, Any, List
import pandas as pd

from .base_processor import BaseProcessor


class ExcelProcessor(BaseProcessor):
    processor_name = "excel"
    SUPPORTED_EXTENSIONS = {".xlsx", ".xls"}

    # 每行最大展示列数（避免超宽表格输出过长文本）
    MAX_COLS_PER_ROW = 50

    def can_handle(self, file_path: str) -> bool:
        ext = Path(file_path).suffix.lower()
        return ext in (".xlsx", ".xls")

    def extract_text(self, file_path: str) -> str:
        ext = Path(file_path).suffix.lower()
        engine = "openpyxl" if ext == ".xlsx" else "xlrd"

        # 读取所有 sheet
        try:
            xl_file = pd.ExcelFile(file_path, engine=engine)
        except Exception:
            # xlrd 2.0+ 不再支持 .xls，尝试用 openpyxl
            if ext == ".xls":
                xl_file = pd.ExcelFile(file_path, engine="openpyxl")
            else:
                raise

        text_parts: List[str] = []

        for sheet_name in xl_file.sheet_names:
            # 读取 sheet，保留所有数据
            df = pd.read_excel(
                xl_file,
                sheet_name=sheet_name,
                header=None,  # 不自动使用第一行作为列名
                dtype=str,    # 全部按字符串读取
            )

            # 去除全空行和全空列
            df = df.dropna(how='all', axis=0)
            df = df.dropna(how='all', axis=1)

            if df.empty:
                continue

            sheet_text = self._dataframe_to_text(df, sheet_name)
            text_parts.append(sheet_text)

        return "\n\n".join(text_parts)

    def extract_metadata(self, file_path: str) -> Dict[str, Any]:
        metadata = super().extract_metadata(file_path)
        try:
            ext = Path(file_path).suffix.lower()
            engine = "openpyxl" if ext == ".xlsx" else "xlrd"
            xl_file = pd.ExcelFile(file_path, engine=engine)
            metadata["sheet_names"] = xl_file.sheet_names
            metadata["sheet_count"] = len(xl_file.sheet_names)
        except Exception:
            pass
        return metadata

    # --- 内部方法 ---

    def _dataframe_to_text(self, df: pd.DataFrame, sheet_name: str) -> str:
        """将 DataFrame 转为可读文本，保留表格结构"""
        lines = [f"[工作表: {sheet_name}]"]

        # 限制列数
        if df.shape[1] > self.MAX_COLS_PER_ROW:
            df = df.iloc[:, :self.MAX_COLS_PER_ROW]

        # 逐行输出
        for row_idx, (_, row) in enumerate(df.iterrows()):
            # 获取非空单元格
            cells = []
            for col_idx, value in enumerate(row):
                if pd.notna(value) and str(value).strip():
                    cells.append(f"{self._col_letter(col_idx)}: {str(value).strip()}")

            if cells:
                lines.append(f"  行{row_idx + 1}: " + " | ".join(cells))
            else:
                lines.append(f"  行{row_idx + 1}: [空行]")

        return "\n".join(lines)

    @staticmethod
    def _col_letter(col_idx: int) -> str:
        """将列索引转为 Excel 列字母 (0→A, 1→B, ...)"""
        if col_idx < 0:
            return "?"
        col_idx += 1
        result = ""
        while col_idx > 0:
            col_idx, remainder = divmod(col_idx - 1, 26)
            result = chr(65 + remainder) + result
        return result
