"""
文本向量化模块

支持两种嵌入模式:
1. TF-IDF 离线模式 (默认) — 使用 sklearn TfidfVectorizer，无需联网
2. Sentence-Transformers 模式 — 需联网下载
模型，语义理解更强

两种模式使用完全相同的接口，可无缝切换。
"""

import logging
from typing import List, Union
import numpy as np

logger = logging.getLogger(__name__)


class Embedder:
    """
    文本嵌入器（双模式）。

    离线 TF-IDF 模式 (默认):
        embedder = Embedder(use_tfidf=True)
        vectors = embedder.encode(["文本1", "文本2"])

    Sentence-Transformers 模式 (需联网):
        embedder = Embedder(use_tfidf=False, model_name="paraphrase-multilingual-MiniLM-L12-v2")
        vectors = embedder.encode(["文本1", "文本2"])

    重要: 使用 TF-IDF 模式时，必须先调用 fit() 在整个语料上训练，
          然后再调用 encode() 编码。或者使用 fit_encode() 一步完成。
    """

    def __init__(
        self,
        use_tfidf: bool = True,
        model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
        batch_size: int = 32,
        device: str = None,
        tfidf_max_features: int = 512,  # TF-IDF 特征维度
    ):
        """
        Args:
            use_tfidf: True=离线TF-IDF模式, False=sentence-transformers模式
            model_name: sentence-transformers 模型名称 (仅 use_tfidf=False 时生效)
            batch_size: 批量编码大小
            device: 设备选择 (仅 use_tfidf=False 时生效)
            tfidf_max_features: TF-IDF 最大特征数（嵌入维度）
        """
        self.use_tfidf = use_tfidf
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        self.tfidf_max_features = tfidf_max_features

        self._model = None
        self._dim = None
        self._tfidf_vectorizer = None  # sklearn TfidfVectorizer

    @property
    def dim(self) -> int:
        """嵌入向量维度"""
        return self._dim or (self.tfidf_max_features if self.use_tfidf else 384)

    def fit(self, texts: List[str]):
        """
        在语料上训练 TF-IDF 向量器（仅 TF-IDF 模式需要）。

        Args:
            texts: 全部待编码文本，用于构建 TF-IDF 词汇表
        """
        if not self.use_tfidf:
            return  # sentence-transformers 模式不需要 fit

        from sklearn.feature_extraction.text import TfidfVectorizer

        logger.info(f"正在构建 TF-IDF 词汇表 (max_features={self.tfidf_max_features})...")
        self._tfidf_vectorizer = TfidfVectorizer(
            max_features=self.tfidf_max_features,
            analyzer='char_wb',         # 字符级 n-gram，支持中英文
            ngram_range=(2, 4),          # 2~4 字符 n-gram
            lowercase=True,
            strip_accents='unicode',
        )
        self._tfidf_vectorizer.fit(texts)
        self._dim = min(self.tfidf_max_features, len(self._tfidf_vectorizer.vocabulary_))
        logger.info(f"TF-IDF 词汇表构建完成，维度: {self._dim}")

    def fit_encode(
        self,
        texts: Union[str, List[str]],
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        训练并编码（仅 TF-IDF 模式需要，简化调用）。
        在 sentence-transformers 模式下等同于 encode()。

        Args:
            texts: 单个字符串或字符串列表
            show_progress: 是否显示进度条 (TF-IDF 模式忽略)

        Returns:
            numpy 数组
        """
        single_input = isinstance(texts, str)
        if single_input:
            texts = [texts]

        if self.use_tfidf:
            self.fit(texts)
            return self.encode(texts)
        else:
            return self.encode(texts, show_progress)

    def encode(self, texts: Union[str, List[str]], show_progress: bool = False) -> np.ndarray:
        """
        将文本编码为向量。

        Args:
            texts: 单个字符串或字符串列表
            show_progress: 是否显示进度条

        Returns:
            numpy 数组，shape=(n_texts, dim)，已 L2 归一化
        """
        single_input = isinstance(texts, str)
        if single_input:
            texts = [texts]

        if len(texts) == 0:
            return np.array([])

        if self.use_tfidf:
            embeddings = self._encode_tfidf(texts)
        else:
            embeddings = self._encode_sentence_transformers(texts, show_progress)

        if single_input:
            return embeddings[0]
        return embeddings

    def encode_single(self, text: str) -> np.ndarray:
        """编码单条文本，返回1维向量"""
        return self.encode([text])[0]

    # --- 内部方法 ---

    def _encode_tfidf(self, texts: List[str]) -> np.ndarray:
        """使用 TF-IDF 编码文本"""
        if self._tfidf_vectorizer is None:
            # 未 fit 时自动 fit
            self.fit(texts)

        vectors = self._tfidf_vectorizer.transform(texts).toarray().astype(np.float32)

        # L2 归一化
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10
        vectors = vectors / norms

        return vectors

    def _encode_sentence_transformers(self, texts: List[str], show_progress: bool) -> np.ndarray:
        """使用 Sentence-Transformers 模型编码文本"""
        if self._model is None:
            self._load_sentence_transformer_model()

        if len(texts) <= self.batch_size:
            embeddings = self._model.encode(
                texts,
                show_progress_bar=show_progress,
                batch_size=self.batch_size,
                normalize_embeddings=True,
            )
        else:
            all_embeddings = []
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i:i + self.batch_size]
                batch_emb = self._model.encode(
                    batch,
                    show_progress_bar=show_progress,
                    normalize_embeddings=True,
                )
                all_embeddings.append(batch_emb)
            embeddings = np.vstack(all_embeddings)

        return embeddings

    def _load_sentence_transformer_model(self):
        """加载 sentence-transformers 模型（需要网络）"""
        try:
            from sentence_transformers import SentenceTransformer

            logger.info(f"正在加载嵌入模型: {self.model_name}")
            logger.info("首次加载需从 HuggingFace 下载模型，请确保网络可访问 huggingface.co")
            self._model = SentenceTransformer(self.model_name, device=self.device)
            self._dim = self._model.get_sentence_embedding_dimension()
            logger.info(f"模型加载完成，嵌入维度: {self._dim}")

            try:
                import torch
                if torch.cuda.is_available():
                    logger.info(f"使用 GPU: {torch.cuda.get_device_name(0)}")
                else:
                    logger.info("使用 CPU")
            except ImportError:
                logger.info("使用 CPU（PyTorch 未安装 CUDA 支持）")

        except ImportError as e:
            raise ImportError(
                f"缺少 sentence-transformers 库。请运行: pip install sentence-transformers\n"
                f"或使用 TF-IDF 离线模式: Embedder(use_tfidf=True)\n"
                f"原始错误: {e}"
            )
        except Exception as e:
            raise RuntimeError(
                f"无法加载模型 '{self.model_name}'。\n"
                f"可能原因: 网络不通、HuggingFace 不可访问、模型名称错误。\n"
                f"解决方案: 使用 TF-IDF 离线模式 Embedder(use_tfidf=True)\n"
                f"原始错误: {type(e).__name__}: {e}"
            )
