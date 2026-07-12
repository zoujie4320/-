"""
RAG 数据预处理系统 — 主入口

将知识库文件夹中的文档经过 解析→清洗→语义分块→向量化→去重→存储 流水线处理。

使用方式:
    python main.py                         # 默认模式，扫描 knowledge_base/ 文件夹
    python main.py --input ./my_docs/      # 指定输入目录
    python main.py --show-settings         # 查看当前配置
    python main.py --list-formats          # 查看支持的文件格式
    python main.py --force                 # 强制重新处理所有文件（忽略增量记录）
    python main.py --dry-run               # 预览将要处理的文件，不实际处理
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np

import settings
from config import ChunkConfig, FileTypeConfig
from processors import find_processor, get_all_processors
from chunker import SemanticChunker
from embedder import Embedder
from storage import FileStorage
from utils.helpers import (
    format_duration,
    compute_file_hash,
    clean_text,
    deduplicate_chunks,
)

# 配置日志
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("RAG")


class RAGPipeline:
    """
    RAG 预处理流水线。

    改进后的流程:
    文档扫描 → 增量过滤 → 文本提取 → 文本清洗 → 语义分块 →
    TF-IDF 全局拟合 → 向量化 → 去重 → 存储

    所有部署级配置从 settings.py 读取，算法级配置从 config.py 读取。
    """

    def __init__(
        self,
        input_dir: str = None,
        output_dir: str = None,
        output_format: str = None,
        chunk_config: ChunkConfig = None,
        file_types: FileTypeConfig = None,
        force_reprocess: bool = False,
        dry_run: bool = False,
    ):
        self.input_dir = input_dir or settings.KNOWLEDGE_BASE_DIR
        self.output_dir = output_dir or settings.OUTPUT_DIR
        self.output_format = output_format or settings.OUTPUT_FORMAT
        self.chunk_config = chunk_config or ChunkConfig()
        self.file_types = file_types or FileTypeConfig()
        self.force_reprocess = force_reprocess
        self.dry_run = dry_run

        # 增量处理的 manifest 文件
        self._manifest_path = Path(self.output_dir) / ".processing_manifest.json"

        # 延迟初始化
        self._embedder: Optional[Embedder] = None
        self._chunker: Optional[SemanticChunker] = None
        self._storage: Optional[FileStorage] = None

    @property
    def embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = Embedder(
                use_tfidf=settings.USE_TFIDF,
                model_name=settings.EMBEDDING_MODEL_NAME,
                batch_size=settings.EMBEDDING_BATCH_SIZE,
                tfidf_max_features=settings.TFIDF_MAX_FEATURES,
            )
        return self._embedder

    @property
    def chunker(self) -> SemanticChunker:
        if self._chunker is None:
            self._chunker = SemanticChunker(
                embedder=self.embedder,
                similarity_threshold=self.chunk_config.similarity_threshold,
                max_chunk_chars=self.chunk_config.max_chunk_chars,
                min_chunk_chars=self.chunk_config.min_chunk_chars,
                overlap_ratio=self.chunk_config.overlap_ratio,
            )
        return self._chunker

    @property
    def storage(self) -> FileStorage:
        if self._storage is None:
            self._storage = FileStorage(
                output_dir=self.output_dir,
                format=self.output_format,
            )
        return self._storage

    # ========================================================================
    # 主流程
    # ========================================================================

    def run(self, input_path: str = None) -> List[dict]:
        """
        运行完整的预处理流水线。

        分为两个阶段:
        Phase 1: 提取文本 + 清洗 + 分块（逐文件）
        Phase 2: TF-IDF 全局拟合 + 向量化 + 去重 + 存储（跨文件批量）
        """
        input_path = input_path or self.input_dir

        logger.info("=" * 50)
        logger.info("RAG 预处理流水线启动")
        if settings.USE_TFIDF:
            logger.info(f"  嵌入模式: TF-IDF 离线 (维度={settings.TFIDF_MAX_FEATURES})")
        else:
            logger.info(f"  嵌入模型: {settings.EMBEDDING_MODEL_NAME}")
        logger.info(f"  分块: 相似度≥{self.chunk_config.similarity_threshold}, "
                     f"{self.chunk_config.min_chunk_chars}~{self.chunk_config.max_chunk_chars}字符, "
                     f"重叠{self.chunk_config.overlap_ratio*100:.0f}%")
        if self.chunk_config.deduplicate:
            logger.info(f"  去重: 已启用 (阈值={self.chunk_config.dedup_threshold})")
        if settings.ENABLE_TEXT_CLEANING:
            logger.info(f"  文本清洗: 已启用")
        if settings.INCREMENTAL_MODE and not self.force_reprocess:
            logger.info(f"  增量处理: 已启用")
        logger.info(f"  知识库: {input_path}")
        logger.info(f"  输出: {self.output_dir}")
        logger.info("=" * 50)

        # Step 1: 收集&过滤文件
        all_files = self._collect_files(input_path)
        if not all_files:
            logger.warning(f"未找到可处理的文件。请将文档放入: {input_path}")
            logger.info(f"支持的类型: {', '.join(sorted(self._all_supported_extensions()))}")
            return []

        files_to_process = self._filter_incremental(all_files)
        if not files_to_process:
            logger.info("所有文件均为最新，无需处理。使用 --force 强制重新处理。")
            return []

        if self.dry_run:
            logger.info(f"\n[预览模式] 将处理 {len(files_to_process)} 个文件:")
            for f in files_to_process:
                logger.info(f"  - {Path(f).name}")
            return []

        logger.info(f"共 {len(all_files)} 个文件，其中 {len(files_to_process)} 个需要处理")

        # ====================================================================
        # Phase 1: 逐文件提取文本 → 清洗 → 分块
        # ====================================================================
        logger.info("\n" + "-" * 40)
        logger.info("Phase 1: 文本提取 & 语义分块")
        logger.info("-" * 40)

        all_file_data = []  # 存储每个文件的处理中间结果
        total_start = time.time()

        for i, file_path in enumerate(files_to_process, 1):
            logger.info(f"\n[{i}/{len(files_to_process)}] {Path(file_path).name}")

            try:
                file_data = self._extract_and_chunk(file_path)
                if file_data:
                    all_file_data.append(file_data)
            except Exception as e:
                logger.error(f"  处理失败: {e}", exc_info=True)

        if not all_file_data:
            logger.warning("没有成功提取到任何文本内容")
            return []

        total_chunks = sum(len(d["chunks"]) for d in all_file_data)
        logger.info(f"\nPhase 1 完成: {len(all_file_data)} 个文件, 共 {total_chunks} 个 chunk")

        # ====================================================================
        # Phase 2: TF-IDF 全局拟合 → 向量化 → 去重 → 存储
        # ====================================================================
        logger.info("\n" + "-" * 40)
        logger.info("Phase 2: 向量化 & 存储")
        logger.info("-" * 40)

        # 自动检测向量数据库
        from storage.vectordb_factory import auto_detect_storage
        vectordb_result = auto_detect_storage(embedder=self.embedder)
        logger.info(f"存储后端: {vectordb_result.db_type} — {vectordb_result.message}")
        vectordb = vectordb_result.storage if vectordb_result.is_vectordb else None

        # 收集所有 chunk 文本用于 TF-IDF 全局拟合
        all_chunk_texts = []
        for fd in all_file_data:
            all_chunk_texts.extend([c["text"] for c in fd["chunks"]])

        # TF-IDF 全局拟合（一次性拟合所有 chunk）
        if settings.USE_TFIDF:
            logger.info(f"TF-IDF 全局拟合: {len(all_chunk_texts)} 个文本...")
            self.embedder.fit(all_chunk_texts)

        # 逐文件编码和存储
        results = []
        total_vectordb_inserted = 0
        for fd in all_file_data:
            file_path = fd["file_path"]
            chunks = fd["chunks"]

            try:
                # 向量化
                chunk_texts = [c["text"] for c in chunks]
                embeddings = self.embedder.encode(chunk_texts)
                logger.info(f"  向量化: {Path(file_path).name} — {embeddings.shape[1]}维 × {embeddings.shape[0]}条")

                # Chunk 去重
                dedup_count = 0
                if self.chunk_config.deduplicate and len(chunks) > 1:
                    chunks, embeddings = deduplicate_chunks(
                        chunks, embeddings,
                        threshold=self.chunk_config.dedup_threshold,
                    )
                    dedup_count = len(fd["chunks"]) - len(chunks)
                    if dedup_count > 0:
                        logger.info(f"  去重: 移除 {dedup_count} 个重复 chunk")
                    # 去重后同步更新 chunk_texts
                    chunk_texts = [c["text"] for c in chunks]

                # 构建 chunk 元数据（包含 source_file 用于向量库更新过滤）
                source_filename = Path(file_path).name
                chunk_metadatas = []
                for c in chunks:
                    meta = {"source_file": source_filename}
                    if "page" in c:
                        meta["page"] = c["page"]
                    if "heading" in c:
                        meta["heading"] = c["heading"]
                    if "overlap_from_prev" in c:
                        meta["overlap_from_prev"] = c["overlap_from_prev"]
                    chunk_metadatas.append(meta)

                # 存储到本地文件（始终作为备份）
                output_path = self.storage.save(
                    chunks=chunks,
                    embeddings=embeddings,
                    source_file=file_path,
                    source_metadata=fd["metadata"],
                    chunk_metadatas=chunk_metadatas,
                )
                logger.info(f"  文件: {Path(output_path).name}")

                # 同时写入向量数据库（如果可用）
                vectordb_inserted = 0
                if vectordb:
                    try:
                        # 先删除该文件的旧向量（支持文档更新场景）
                        deleted_count = 0
                        if hasattr(vectordb, 'delete_by_source'):
                            deleted_count = vectordb.delete_by_source(source_filename)
                            if deleted_count > 0:
                                logger.info(f"  向量DB: 已清理旧版本 {deleted_count} 条")

                        chunk_ids = [f"{Path(file_path).stem}_{i:04d}" for i in range(len(chunks))]
                        vectordb.insert(
                            vectors=embeddings,
                            texts=chunk_texts,
                            metadatas=chunk_metadatas,
                            ids=chunk_ids,
                        )
                        vectordb_inserted = len(chunks)
                        total_vectordb_inserted += vectordb_inserted
                        logger.info(f"  向量DB: {vectordb_inserted} 条已写入 {vectordb_result.db_type}")
                    except Exception as e:
                        logger.warning(f"  向量DB写入失败: {e}")

                results.append({
                    "file": file_path,
                    "status": "success",
                    "processor": fd["processor_name"],
                    "text_length": fd["text_length"],
                    "chunk_count": len(chunks),
                    "dedup_removed": dedup_count,
                    "output_path": output_path,
                    "vectordb_inserted": vectordb_inserted,
                })

            except Exception as e:
                logger.error(f"  向量化/存储失败: {Path(file_path).name} — {e}", exc_info=True)
                results.append({
                    "file": file_path,
                    "status": "error",
                    "error": str(e),
                })

        # 更新增量 manifest
        if not self.dry_run:
            self._update_manifest(files_to_process, results)

        # 断开向量数据库
        if vectordb:
            vectordb.disconnect()

        # 汇总
        total_time = time.time() - total_start
        self._print_summary(results, total_time, total_chunks, vectordb_result, total_vectordb_inserted)

        return results

    # ========================================================================
    # Phase 1: 文本提取 & 分块
    # ========================================================================

    def _extract_and_chunk(self, file_path: str) -> Optional[Dict[str, Any]]:
        """处理单个文件：提取文本 → 清洗 → 分块"""
        # 匹配处理器
        processor = find_processor(file_path)
        logger.info(f"  处理器: {processor.processor_name}")

        # 提取文本
        result = processor.process(file_path)
        if result["error"]:
            logger.error(f"  提取失败: {result['error']}")
            return None

        text = result["text"]
        logger.info(f"  提取文本: {len(text)} 字符")

        # 文本清洗
        if settings.ENABLE_TEXT_CLEANING:
            text = clean_text(text)

        # 语义分块
        chunks = self.chunker.chunk(text)
        if not chunks:
            logger.warning("  分块结果为空")
            return None
        logger.info(f"  语义分块: {len(chunks)} 个 chunk")

        # 增强元数据：页码、处理器名
        enhanced_chunks = []
        for c in chunks:
            enhanced = dict(c)
            # 从原始文本中估算 chunk 所在页码（如果处理器提供了页面信息）
            if "page_count" in result.get("metadata", {}):
                enhanced["source_pages"] = result["metadata"].get("page_count", 0)
            enhanced_chunks.append(enhanced)

        return {
            "file_path": file_path,
            "chunks": enhanced_chunks,
            "metadata": result["metadata"],
            "processor_name": processor.processor_name,
            "text_length": len(text),
        }

    # ========================================================================
    # 文件收集 & 增量过滤
    # ========================================================================

    def _collect_files(self, input_path: str) -> List[str]:
        """收集待处理的文件列表"""
        path = Path(input_path)
        if not path.exists():
            raise FileNotFoundError(
                f"路径不存在: {input_path}\n"
                f"请创建该目录并放入文档，或修改 settings.py 中的 KNOWLEDGE_BASE_DIR"
            )

        if path.is_file():
            try:
                find_processor(str(path))
                return [str(path.absolute())]
            except ValueError:
                raise ValueError(f"不支持的文件类型: {input_path}")

        files = []
        for ext in self._all_supported_extensions():
            for f in path.rglob(f"*{ext}"):
                abs_path = str(f.absolute())
                if str(self.output_dir) not in abs_path:
                    files.append(abs_path)
        return sorted(files)

    def _all_supported_extensions(self) -> set:
        """从已注册处理器中获取所有支持的文件扩展名"""
        exts = set()
        for p in get_all_processors():
            if hasattr(p, 'SUPPORTED_EXTENSIONS'):
                exts.update(p.SUPPORTED_EXTENSIONS)
        return exts

    def _filter_incremental(self, all_files: List[str]) -> List[str]:
        """
        增量过滤：跳过哈希未变化的已处理文件。

        返回需要（重新）处理的文件列表。
        """
        if self.force_reprocess or not settings.INCREMENTAL_MODE:
            return all_files

        manifest = self._load_manifest()
        to_process = []
        skipped = 0

        for file_path in all_files:
            try:
                file_hash = compute_file_hash(file_path)
            except Exception:
                # 文件不可读，仍然尝试处理
                to_process.append(file_path)
                continue

            if file_path in manifest and manifest[file_path] == file_hash:
                skipped += 1
            else:
                to_process.append(file_path)

        if skipped > 0:
            logger.info(f"增量跳过 {skipped} 个未变化的文件")
        return to_process

    def _load_manifest(self) -> Dict[str, str]:
        """加载增量处理记录"""
        if not self._manifest_path.exists():
            return {}
        try:
            with open(self._manifest_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def _update_manifest(self, processed_files: List[str], results: List[dict]):
        """更新增量处理记录（只记录成功处理的文件）"""
        manifest = self._load_manifest()
        for file_path in processed_files:
            # 只记录成功文件
            file_results = [r for r in results if r["file"] == file_path]
            if file_results and file_results[0]["status"] == "success":
                try:
                    manifest[file_path] = compute_file_hash(file_path)
                except Exception:
                    pass

        with open(self._manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)

    # ========================================================================
    # 汇总报告
    # ========================================================================

    def _print_summary(self, results: List[dict], total_time: float, total_chunks: int,
                       vectordb_result=None, total_vectordb_inserted: int = 0):
        """打印处理汇总"""
        success = [r for r in results if r.get("status") == "success"]
        errors = [r for r in results if r.get("status") == "error"]

        logger.info("\n" + "=" * 50)
        logger.info("处理完成 — 汇总")
        logger.info(f"  处理文件数: {len(results)} (成功: {len(success)}, 失败: {len(errors)})")
        logger.info(f"  总 chunk 数: {total_chunks}")
        total_dedup = sum(r.get("dedup_removed", 0) for r in success)
        if total_dedup > 0:
            logger.info(f"  去重移除: {total_dedup} 个重复 chunk")
        logger.info(f"  总耗时: {format_duration(total_time)}")
        logger.info(f"  文件存储: {Path(self.output_dir).absolute()}")
        if vectordb_result and vectordb_result.is_vectordb:
            logger.info(f"  向量数据库: {vectordb_result.db_type} — 已写入 {total_vectordb_inserted} 条")
        elif vectordb_result:
            logger.info(f"  向量数据库: 不可用 ({vectordb_result.message})")
            logger.info(f"  💡 安装 chromadb 后即可自动启用向量数据库: pip install chromadb")

        if errors:
            logger.info("失败文件:")
            for e in errors:
                pname = Path(e['file']).name if 'file' in e else '?'
                logger.info(f"  - {pname}: {e.get('error', 'unknown')}")

        # 合并所有向量文件
        if len(success) > 1:
            try:
                merged_path = self.storage.merge_to_single_file()
                logger.info(f"  合并文件: {Path(merged_path).name}")
            except Exception as e:
                logger.warning(f"  合并失败: {e}")


# ============================================================================
# CLI 入口
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="RAG 数据预处理系统 — 自动解析知识库文档并向量化存储",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python main.py                              # 扫描 knowledge_base/ 文件夹
  python main.py --input ./my_docs/           # 指定输入目录
  python main.py --output ./my_vectors/       # 指定输出目录
  python main.py --force                      # 强制重新处理所有文件
  python main.py --dry-run                    # 预览，不实际处理
  python main.py --show-settings              # 查看当前配置
  python main.py --list-formats               # 查看支持的文件格式
        """,
    )

    parser.add_argument(
        "--input", "-i", type=str, default=None,
        help=f"输入目录路径 (默认: {settings.KNOWLEDGE_BASE_DIR})",
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help=f"输出目录 (默认: {settings.OUTPUT_DIR})",
    )
    parser.add_argument(
        "--format", "-f", type=str, choices=["csv", "excel"], default=None,
        help=f"输出格式 (默认: {settings.OUTPUT_FORMAT})",
    )
    parser.add_argument(
        "--similarity", "-s", type=float, default=0.6,
        help="语义分块相似度阈值 0~1 (默认: 0.6)",
    )
    parser.add_argument(
        "--max-chars", type=int, default=1000,
        help="单个 chunk 最大字符数 (默认: 1000)",
    )
    parser.add_argument(
        "--min-chars", type=int, default=80,
        help="单个 chunk 最小字符数 (默认: 80)",
    )
    parser.add_argument(
        "--overlap", type=float, default=0.15,
        help="chunk 间重叠比例 0~0.3 (默认: 0.15)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="强制重新处理所有文件（忽略增量记录）",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="预览模式：列出待处理文件，不实际处理",
    )
    parser.add_argument(
        "--list-formats", action="store_true",
        help="列出所有支持的文件格式",
    )
    parser.add_argument(
        "--show-settings", action="store_true",
        help="查看当前 settings.py 中的配置",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="详细日志输出",
    )

    args = parser.parse_args()

    if args.show_settings:
        settings.show_settings()
        return

    if args.list_formats:
        print("支持的文件格式:")
        for p in get_all_processors():
            exts = getattr(p, 'SUPPORTED_EXTENSIONS', None)
            if exts:
                print(f"  {p.processor_name}: {', '.join(sorted(exts))}")
            else:
                print(f"  {p.processor_name}: 见 config.py")
        return

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # 确保必要目录
    settings.ensure_directories()

    # 构建配置
    chunk_config = ChunkConfig(
        similarity_threshold=args.similarity,
        max_chunk_chars=args.max_chars,
        min_chunk_chars=args.min_chars,
        overlap_ratio=args.overlap,
    )

    # 运行流水线
    pipeline = RAGPipeline(
        input_dir=args.input,
        output_dir=args.output,
        output_format=args.format,
        chunk_config=chunk_config,
        force_reprocess=args.force,
        dry_run=args.dry_run,
    )
    results = pipeline.run()

    errors = [r for r in results if r.get("status") == "error"]
    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
