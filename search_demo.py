#!/usr/bin/env python3
"""
RAG 检索演示 — 3 层召回 + 多路合路 + LLM 答案生成

使用方式:
    # 交互式检索 (默认)
    python search_demo.py

    # 单次查询
    python search_demo.py --query "Python开发经验"

    # 指定向量文件
    python search_demo.py --input ./output/all_vectors.csv --query "关键词"

    # 交互式模式
    python search_demo.py --interactive
"""

import argparse
import logging
import sys
from pathlib import Path

import settings
from embedder import Embedder
from retrieval import LocalRetriever, CrossEncoderReranker, SearchPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("Search")


def build_pipeline(vector_file: str = None, use_rerank: bool = True):
    """构建检索流水线"""
    # 1. 加载向量数据
    vector_file = vector_file or str(Path(settings.OUTPUT_DIR) / "all_vectors.csv")
    if not Path(vector_file).exists():
        logger.error(f"向量文件不存在: {vector_file}")
        logger.info("请先运行 main.py 生成向量数据")
        return None

    # 2. 嵌入器（先创建，用于检索器重新编码）
    embedder = Embedder(
        use_tfidf=settings.USE_TFIDF,
        model_name=settings.EMBEDDING_MODEL_NAME,
        tfidf_max_features=settings.TFIDF_MAX_FEATURES,
    )

    # 1. 加载向量数据（TF-IDF模式下传入embedder以重新编码）
    logger.info(f"加载向量数据: {vector_file}")
    retriever = LocalRetriever()
    retriever.load_from_csv(vector_file, embedder=embedder)
    logger.info(f"  加载 {retriever.count()} 条记录")

    # 3. Reranker (可选)
    reranker = None
    if use_rerank:
        try:
            reranker = CrossEncoderReranker()
            logger.info("  Reranker: CrossEncoder 已加载")
        except Exception as e:
            logger.warning(f"  Reranker 不可用: {e}")

    # 4. LLM 配置
    llm_config = {
        "api_key": settings.LLM_API_KEY,
        "base_url": settings.LLM_BASE_URL,
        "model": settings.LLM_MODEL,
    }

    # 5. 构建流水线
    pipeline = SearchPipeline(retriever, embedder, reranker, llm_config)
    logger.info("检索流水线就绪\n")
    return pipeline


def print_result(response):
    """格式化打印检索结果"""
    print(f"\n{'=' * 60}")
    print(f"查询: {response.query}")
    print(f"耗时: {response.elapsed_ms:.0f}ms")
    print(f"召回统计: {response.layer_stats}")
    print(f"{'=' * 60}")

    if response.answer:
        print(f"\n[答案]\n{response.answer}")

    print(f"\n[检索结果 Top {len(response.results)}]:")
    for i, r in enumerate(response.results, 1):
        print(f"\n--- [{i}] 相关度: {r.score:.4f} | 来源: {r.source} ---")
        # 截断显示 + 安全编码
        text = r.text.replace('\n', ' ')[:200]
        if len(r.text) > 200:
            text += "..."
        # 安全打印（处理 Windows GBK 编码问题）
        try:
            print(f"    {text}")
        except UnicodeEncodeError:
            print(f"    {text.encode('ascii', errors='replace').decode('ascii')}")
        source_file = r.metadata.get("source_file", "")
        if source_file:
            print(f"    文件: {source_file}")

    print(f"\n{'=' * 60}\n")


def interactive_mode(pipeline):
    """交互式检索模式"""
    print("\n" + "=" * 60)
    print("RAG 检索演示 — 交互模式")
    print("输入查询进行检索，输入 'quit' 退出")
    print("=" * 60)

    while True:
        try:
            query = input("\n[查询]: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见!")
            break

        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            print("再见!")
            break

        response = pipeline.search(query, top_k=5, use_rerank=True, use_llm=True)
        print_result(response)


def main():
    parser = argparse.ArgumentParser(
        description="RAG 检索演示 — 3层召回 + 多路合路",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python search_demo.py                                    # 交互模式
  python search_demo.py --query "Python开发经验"           # 单次查询
  python search_demo.py --input ./output/all_vectors.csv   # 指定向量文件
  python search_demo.py --no-rerank                        # 不启用精排
        """,
    )

    parser.add_argument("--query", "-q", type=str, help="查询文本")
    parser.add_argument("--input", "-i", type=str, default=None,
                        help=f"向量文件路径 (默认: {settings.OUTPUT_DIR}/all_vectors.csv)")
    parser.add_argument("--top-k", "-k", type=int, default=5, help="返回数量 (默认: 5)")
    parser.add_argument("--no-rerank", action="store_true", help="禁用重排序")
    parser.add_argument("--no-llm", action="store_true", help="禁用 LLM 答案合成")
    parser.add_argument("--interactive", action="store_true", help="交互式模式 (默认)")
    parser.add_argument("--verbose", "-v", action="store_true", help="详细日志")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # 构建流水线
    pipeline = build_pipeline(
        vector_file=args.input,
        use_rerank=not args.no_rerank,
    )
    if pipeline is None:
        sys.exit(1)

    # 单次查询
    if args.query:
        response = pipeline.search(
            args.query,
            top_k=args.top_k,
            use_rerank=not args.no_rerank,
            use_llm=not args.no_llm,
        )
        print_result(response)
        return

    # 交互模式
    interactive_mode(pipeline)


if __name__ == "__main__":
    main()
