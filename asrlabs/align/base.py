"""对齐后端抽象基类——策略模式核心"""

from abc import ABC, abstractmethod

import numpy as np
from asrlabs.models import TranscriptionResult


class BaseAligner(ABC):
    """所有对齐后端的抽象基类

    子类需要:
    1. 设置类属性 name, display_name
    2. 实现 load_model() 和 align()
    """

    name: str = ""
    display_name: str = ""

    def __init__(self, config: dict):
        """初始化对齐后端

        Args:
            config: 配置字典，至少包含 extras
        """
        self.config = config
        self._model = None
        self._loaded = False

    @abstractmethod
    def load_model(self) -> None:
        """加载/初始化对齐模型"""
        ...

    @abstractmethod
    def align(
        self,
        audio: str | np.ndarray,
        result: TranscriptionResult,
        language: str | None = None,
    ) -> TranscriptionResult:
        """对齐音频和文本

        Args:
            audio: 音频文件路径或 numpy 数组
            result: 已有听写结果（可能不含时间戳）
            language: 语言代码（None 则从 result 推断）

        Returns:
            带时间戳的 TranscriptionResult
        """
        ...

    def unload(self) -> None:
        """释放模型资源"""
        self._model = None
        self._loaded = False

    def _ensure_loaded(self) -> None:
        """保证模型已加载"""
        if not self._loaded:
            self.load_model()
            self._loaded = True
