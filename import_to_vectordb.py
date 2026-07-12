#!/usr/bin/env python3
"""
将本地向量文件导入向量数据库

读取 output/ 目录中的 CSV/Excel 向量文件，导入到向量数据库。

使用方式:
    # 自动检测可用向量数据库并导入
    python import_to_vectordb.py

    # 指定源目录和目标数据库类型
    python import_to_vectordb.py --input ./output --db chroma

    # 测试数据库连接
    python import_to_vectordb.py --test-connection

    # 查看数据库状态
    python import_to_vectordb.py --status
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pandas as pd

import settings
from storage.vectordb_factory import auto_detect_storage, test_connection
from storage.file_storage import FileStorage
from utils.helpers import format_duration

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("Import")


# ============================================================================
# 导入器
# ============================================================================

class VectorImporter:
    """
    将本地 CSV/Excel 向量文件导入到向量数据库。
    """

    def __init__(self, input_dir: str = None, db_type: str = None):
        self.input_dir = input_dir or settings.OUTPUT_DIR
        self.db_type = db_type or settings.VECTOR_DB_TYPE
        self.fs = FileStorage(output_dir=self.input_dir)

    def run(self) -> dict:
        """
        执行导入流程:
        1. 扫描本地向量文件
        2. 连接向量数据库
        3. 批量导入
        4. 汇总报告
        """
        # 1. 扫描文件
        vector_files = self._find_vector_files()
        if not vector_files:
            logger.error(f"在 {self.input_dir} 中未找到向量文件 (*_vectors.csv/xlsx)")
            return {"status": "error", "message": "未找到向量文件"}

        logger.info(f"找到 {len(vector_files)} 个向量文件:")
        for vf in vector_files:
            logger.info(f"  - {vf.name}")

        # 2. 加载所有向量数据
        all_data = []
        total_rows = 0
        for vf in vector_files:
            df = self.fs.load(str(vf))
            all_data.append(df)
            total_rows += len(df)
        logger.info(f"共 {total_rows} 条向量记录")

        # 3. 连接向量数据库
        logger.info(f"\n连接向量数据库 (类型: {self.db_type})...")
        result = auto_detect_storage()

        if not result.is_vectordb:
            logger.error(f"无法连接向量数据库: {result.message}")
            logger.info("请确保已安装向量数据库依赖:")
            logger.info("  Chroma:  pip install chromadb")
            logger.info("  Milvus:  pip install pymilvus (需要独立服务)")
            return {"status": "error", "message": result.message}

        logger.info(f"✓ {result.message}")

        # 4. 逐文件导入
        storage = result.storage
        total_inserted = 0
        errors = 0
        start_time = time.time()

        # 合并数据
        merged = pd.concat(all_data, ignore_index=True)

        # 解析 embedding
        logger.info("解析嵌入向量...")
        embeddings_list = []
        texts = []
        metadatas = []
        ids = []

        for _, row in merged.iterrows():
            try:
                # 解析 embedding JSON
                if "embedding" in row and row["embedding"] is not None:
                    emb = row["embedding"]
                    if isinstance(emb, np.ndarray):
                        vec = emb
                    else:
                        vec = np.array(emb) if not isinstance(emb, np.ndarray) else emb
                elif "embedding_json" in row and isinstance(row["embedding_json"], str):
                    vec = np.array(json.loads(row["embedding_json"]))
                else:
                    continue

                embeddings_list.append(vec)
                texts.append(str(row.get("chunk_text", "")))
                ids.append(str(row.get("chunk_id", "")))

                # 元数据
                meta = {}
                if "metadata_json" in row and isinstance(row["metadata_json"], str):
                    try:
                        meta = json.loads(row["metadata_json"])
                    except json.JSONDecodeError:
                        pass
                if "source_file" in row:
                    meta["source_file"] = str(row["source_file"])
                if "char_count" in row:
                    meta["char_count"] = int(row["char_count"])
                metadatas.append(meta)

            except Exception as e:
                errors += 1
                continue

        if not embeddings_list:
            logger.error("未能解析任何嵌入向量")
            return {"status": "error", "message": "嵌入向量解析失败"}

        embeddings = np.array(embeddings_list, dtype=np.float32)
        logger.info(f"解析完成: {embeddings.shape[1]}维 × {embeddings.shape[0]}条")

        # 批量导入
        logger.info(f"\n开始导入 {len(texts)} 条记录...")
        inserted_ids = storage.insert(
            vectors=embeddings,
            texts=texts,
            metadatas=metadatas,
            ids=ids,
        )
        total_inserted = len(inserted_ids)

        storage.disconnect()

        elapsed = time.time() - start_time

        # 5. 汇总
        logger.info("\n" + "=" * 50)
        logger.info("导入完成")
        logger.info(f"  数据库类型: {result.db_type}")
        logger.info(f"  成功导入: {total_inserted} 条")
        if errors:
            logger.info(f"  解析失败: {errors} 条")
        logger.info(f"  耗时: {format_duration(elapsed)}")

        # 验证
        logger.info("\n验证导入结果...")
        conn_result = test_connection(result.db_type,
                                      persist_dir=settings.CHROMA_PERSIST_DIR,
                                      collection_name=settings.MILVUS_COLLECTION_NAME)
        if conn_result["ok"]:
            logger.info(f"  数据库中记录数: {conn_result['count']}")

        return {
            "status": "success",
            "db_type": result.db_type,
            "inserted": total_inserted,
            "errors": errors,
            "time": elapsed,
        }

    def _find_vector_files(self) -> List[Path]:
        """查找所有向量文件"""
        input_path = Path(self.input_dir)
        if not input_path.exists():
            return []

        files = []
        for pattern in ["*_vectors.csv", "*_vectors.xlsx"]:
            files.extend(input_path.glob(pattern))
        return sorted(files)


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="本地向量文件 → 向量数据库 导入工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python import_to_vectordb.py                     # 自动检测数据库并导入
  python import_to_vectordb.py --db chroma         # 指定 ChromaDB
  python import_to_vectordb.py --test-connection   # 测试数据库连接
  python import_to_vectordb.py --status            # 查看数据库状态
        """,
    )

    parser.add_argument(
        "--input", "-i", type=str, default=None,
        help=f"向量文件目录 (默认: {settings.OUTPUT_DIR})",
    )
    parser.add_argument(
        "--db", type=str, default=None,
        choices=["chroma", "milvus"],
        help="目标向量数据库类型",
    )
    parser.add_argument(
        "--test-connection", action="store_true",
        help="测试向量数据库连接",
    )
    parser.add_argument(
        "--status", action="store_true",
        help="查看数据库状态（记录数等）",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="详细日志",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # 测试连接
    if args.test_connection:
        print("\n" + "=" * 50)
        print("向量数据库连接测试")
        print("=" * 50)
        for db in ["chroma", "milvus"]:
            print(f"\n--- {db.upper()} ---")
            r = test_connection(db,
                                persist_dir=settings.CHROMA_PERSIST_DIR,
                                collection_name=settings.MILVUS_COLLECTION_NAME,
                                host=settings.MILVUS_HOST,
                                port=settings.MILVUS_PORT)
            status = "✓ 可用" if r["ok"] else "✗ 不可用"
            print(f"  状态: {status}")
            print(f"  信息: {r['message']}")
            if r["ok"]:
                print(f"  记录数: {r['count']}")
        print()
        return

    # 查看状态
    if args.status:
        print("\n" + "=" * 50)
        print("向量数据库状态")
        print("=" * 50)
        for db in ["chroma", "milvus"]:
            r = test_connection(db,
                                persist_dir=settings.CHROMA_PERSIST_DIR,
                                collection_name=settings.MILVUS_COLLECTION_NAME,
                                host=settings.MILVUS_HOST,
                                port=settings.MILVUS_PORT)
            if r["ok"]:
                print(f"\n[{db.upper()}] ✓ 已连接 — {r['count']} 条记录")
            else:
                print(f"\n[{db.upper()}] ✗ 不可用 — {r['message']}")
        print()
        return

    # 运行导入
    importer = VectorImporter(
        input_dir=args.input,
        db_type=args.db,
    )
    result = importer.run()

    if result["status"] == "error":
        sys.exit(1)


if __name__ == "__main__":
    main()
