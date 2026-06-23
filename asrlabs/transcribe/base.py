"""听写后端抽象基类——策略模式核心"""

from abc import ABC, abstractmethod

import numpy as np
from asrlabs.models import TranscriptionResult


class BaseTranscriber(ABC):
    """所有听写后端的抽象基类

    子类需要:
    1. 设置类属性 name, display_name, supports_timestamps, recommended_aligner
    2. 实现 load_model() 和 transcribe()
    """

    # 元信息——子类必须覆盖
    name: str = ""
    display_name: str = ""
    supports_timestamps: bool = False
    recommended_aligner: str | None = None

    def __init__(self, config: dict):
        """初始化听写后端

        Args:
            config: 配置字典，至少包含 device, compute_type, language, beam_size, extras
        """
        self.config = config
        self._model = None
        self._loaded = False

    @abstractmethod
    def load_model(self) -> None:
        """加载/初始化模型

        子类在此方法中完成模型加载逻辑。
        由 transcribe() 首次调用时自动触发（延迟加载）。
        """
        ...

    @abstractmethod
    def transcribe(
        self, audio: str | np.ndarray, **kwargs
    ) -> TranscriptionResult:
        """执行听写

        Args:
            audio: 音频文件路径或 numpy 数组
            **kwargs: 额外参数，透传底层库

        Returns:
            TranscriptionResult 统一结果
        """
        ...

    def unload(self) -> None:
        """释放模型资源，子类可覆盖"""
        self._model = None
        self._loaded = False

    def _ensure_loaded(self) -> None:
        """保证模型已加载（延迟加载）"""
        if not self._loaded:
            self.load_model()
            self._loaded = True
