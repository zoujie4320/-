"""
文件存储层

将向量化后的 chunk 数据存储为 CSV 或 Excel 文件。
embedding 以 JSON 字符串形式存储，方便人工查看和程序读取。

输出文件结构:
- chunks.csv: chunk_id, source_file, chunk_text, embedding_json, char_count, metadata_json, created_at
"""

import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np

from utils.helpers import ensure_output_dir


class FileStorage:
    """
    向量化数据的文件存储器。

    使用方法:
        storage = FileStorage(output_dir="./output")
        storage.save(chunks, embeddings, source_file="doc.pdf")
    """

    def __init__(self, output_dir: str = "./output", format: str = "csv"):
        """
        Args:
            output_dir: 输出目录
            format: 输出格式 ("csv" 或 "excel")
        """
        self.output_dir = ensure_output_dir(output_dir)
        self.format = format.lower()
        if self.format not in ("csv", "excel"):
            raise ValueError(f"不支持的输出格式: {format}，请使用 'csv' 或 'excel'")

    def save(
        self,
        chunks: List[Dict[str, Any]],
        embeddings: np.ndarray,
        source_file: str,
        source_metadata: Optional[Dict[str, Any]] = None,
        chunk_metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        保存向量化的 chunk 数据到文件。

        Args:
            chunks: chunk 字典列表 (来自 SemanticChunker.chunk())
            embeddings: 对应的嵌入向量矩阵 (n_chunks x dim)
            source_file: 源文件名
            source_metadata: 源文件的元数据
            chunk_metadatas: 每个 chunk 的额外元数据

        Returns:
            输出文件的路径
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"chunks 数量 ({len(chunks)}) 与 embeddings 数量 ({len(embeddings)}) 不一致"
            )

        source_name = Path(source_file).stem
        output_path = Path(self.output_dir) / f"{source_name}_vectors.{self.format}"

        # 构建数据行
        rows = []
        created_at = time.strftime("%Y-%m-%d %H:%M:%S")

        for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            row = {
                "chunk_id": f"{source_name}_{i:04d}",
                "source_file": Path(source_file).name,
                "chunk_text": chunk["text"],
                "embedding_json": json.dumps(emb.tolist(), ensure_ascii=False),
                "char_count": chunk.get("char_count", len(chunk["text"])),
                "created_at": created_at,
            }

            # 合并元数据
            meta = {}
            if source_metadata:
                meta.update(source_metadata)
            if chunk_metadatas and i < len(chunk_metadatas):
                meta.update(chunk_metadatas[i])
            if chunk.get("overlap_from_prev"):
                meta["overlap_from_prev"] = chunk["overlap_from_prev"]
            row["metadata_json"] = json.dumps(meta, ensure_ascii=False)

            rows.append(row)

        df = pd.DataFrame(rows)

        # 写入文件
        if self.format == "csv":
            df.to_csv(output_path, index=False, encoding="utf-8-sig")
        else:  # excel
            df.to_excel(output_path, index=False, engine="openpyxl")

        return str(output_path)

    def load(self, file_path: str) -> pd.DataFrame:
        """
        从文件中加载向量化数据。

        Args:
            file_path: CSV 或 Excel 文件路径

        Returns:
            包含所有列的 DataFrame
        """
        ext = Path(file_path).suffix.lower()
        if ext == ".csv":
            df = pd.read_csv(file_path, encoding="utf-8-sig")
        elif ext in (".xlsx", ".xls"):
            df = pd.read_excel(file_path, engine="openpyxl")
        else:
            raise ValueError(f"不支持的文件格式: {ext}")

        # 将 JSON 字符串的 embedding 转为 numpy 数组
        if "embedding_json" in df.columns:
            df["embedding"] = df["embedding_json"].apply(
                lambda x: np.array(json.loads(x)) if isinstance(x, str) else None
            )

        return df

    def merge_to_single_file(self, output_name: str = "all_vectors") -> str:
        """
        将输出目录中的所有向量文件合并为一个文件。

        Args:
            output_name: 合并后的文件名（不含扩展名）

        Returns:
            合并文件的路径
        """
        all_dfs = []
        output_dir = Path(self.output_dir)

        pattern = f"*_vectors.{self.format}"
        for file in sorted(output_dir.glob(pattern)):
            # 跳过同名合并文件，避免重复读取自身
            if file.stem == output_name:
                continue
            df = self.load(str(file))
            all_dfs.append(df)

        if not all_dfs:
            raise FileNotFoundError(f"在 {self.output_dir} 中未找到任何向量文件")

        merged = pd.concat(all_dfs, ignore_index=True)
        output_path = output_dir / f"{output_name}.{self.format}"

        if self.format == "csv":
            merged.to_csv(output_path, index=False, encoding="utf-8-sig")
        else:
            merged.to_excel(output_path, index=False, engine="openpyxl")

        return str(output_path)
