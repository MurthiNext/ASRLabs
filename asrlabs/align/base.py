"""对齐后端抽象基类——策略模式核心"""

from abc import ABC, abstractmethod

import numpy as np
from asrlabs.models import TranscriptionResult


ALIGNER_REGISTRY: dict[str, type["BaseAligner"]] = {}


def register_aligner(cls: type["BaseAligner"]) -> type["BaseAligner"]:
    """装饰器：将对齐后端注册到全局注册表"""
    if not cls.name:
        raise ValueError(f"{cls.__name__} 必须设置 name 类属性")
    ALIGNER_REGISTRY[cls.name] = cls
    return cls


def get_aligner(name: str, config: dict) -> "BaseAligner | None":
    """工厂方法：按名称创建对齐后端实例

    Args:
        name: 对齐器名称（whisper_align / qwen3_align / none）
        config: 配置字典

    Returns:
        BaseAligner 实例，name 为 "none" 时返回 None

    Raises:
        ValueError: 未知的对齐器名称
    """
    if name == "none":
        return None
    if name not in ALIGNER_REGISTRY:
        available = ", ".join(ALIGNER_REGISTRY.keys())
        raise ValueError(f"未知的对齐器: {name}，可用: {available}")
    return ALIGNER_REGISTRY[name](config)


def list_aligners() -> list[dict]:
    """列出所有已注册的对齐后端"""
    return [
        {"name": cls.name, "display_name": cls.display_name}
        for cls in ALIGNER_REGISTRY.values()
    ]


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
            config: 配置字典，至少包含 model_path, device, extras
        """
        self.config = config
        self.model_path = config.get("model_path", "")  # 本地模型路径，空则用默认
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
