"""
文档处理器抽象基类

所有文档类型处理器必须继承此类并实现 can_handle() 和 extract_text()。
新增文件类型只需：创建子类 → 实现两个方法 → 注册到 processors/__init__.py
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional


class BaseProcessor(ABC):
    """
    文档处理器基类。

    子类需要实现:
        - can_handle(file_path) -> bool
        - extract_text(file_path) -> str
    可选覆盖:
        - processor_name (str): 处理器名称
        - extract_metadata(file_path) -> dict
    """

    # 子类可覆盖此属性
    processor_name: str = "base"

    @abstractmethod
    def can_handle(self, file_path: str) -> bool:
        """
        判断此处理器能否处理给定文件。
        通常基于文件扩展名判断。

        Args:
            file_path: 文件路径

        Returns:
            True 表示可以处理此文件
        """
        ...

    @abstractmethod
    def extract_text(self, file_path: str) -> str:
        """
        从文件中提取纯文本内容。

        Args:
            file_path: 文件路径

        Returns:
            提取的文本字符串
        """
        ...

    def extract_metadata(self, file_path: str) -> Dict[str, Any]:
        """
        提取文件元数据。子类可覆盖以提供更丰富的元数据。

        Args:
            file_path: 文件路径

        Returns:
            元数据字典
        """
        path = Path(file_path)
        stat = path.stat()
        return {
            "file_name": path.name,
            "file_relative_path": str(path.relative_to(path.cwd())) if path.is_relative_to(path.cwd()) else path.name,
            "file_size_bytes": stat.st_size,
            "file_extension": path.suffix.lower(),
            "processor": self.processor_name,
        }

    def process(self, file_path: str) -> Dict[str, Any]:
        """
        模板方法：处理单个文件，返回标准化结果。

        Args:
            file_path: 文件路径

        Returns:
            包含 text, metadata, error (可选) 的字典
        """
        result = {
            "text": "",
            "metadata": self.extract_metadata(file_path),
            "error": None,
        }

        if not Path(file_path).exists():
            result["error"] = f"文件不存在: {file_path}"
            return result

        try:
            result["text"] = self.extract_text(file_path)
            if not result["text"].strip():
                result["error"] = "未能提取到任何文本内容"
        except Exception as e:
            result["error"] = f"{type(e).__name__}: {str(e)}"

        return result
