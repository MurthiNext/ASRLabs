"""听写后端抽象基类与注册表——策略模式核心"""

from abc import ABC, abstractmethod

import numpy as np
from asrlabs.models import TranscriptionResult


class BaseTranscriber(ABC):
    """所有听写后端的抽象基类

    子类需要:
    1. 设置类属性 name, display_name, supports_timestamps, recommended_aligner
    2. 实现 load_model() 和 transcribe()

    用户可直接导入使用:
        from asrlabs.transcribe import WhisperTranscriber
        t = WhisperTranscriber({"model": "whisper-large-v3", "device": "cuda"})
        t.load_model()
        result = t.transcribe("audio.wav")
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


# ── 注册表基础设施 ──

TRANSCRIBER_REGISTRY: dict[str, type[BaseTranscriber]] = {}
"""全局注册表：{模型名称: 后端类}"""

_PREFIX_REGISTRY: dict[str, type[BaseTranscriber]] = {}
"""前缀注册表：用于匹配 whisper-base/whisper-large-v3 等到同一个类"""


def register_transcriber(cls: type[BaseTranscriber]) -> type[BaseTranscriber]:
    """装饰器：将听写后端注册到全局注册表"""
    if not cls.name:
        raise ValueError(f"{cls.__name__} 必须设置 name 类属性")
    TRANSCRIBER_REGISTRY[cls.name] = cls
    return cls


def get_transcriber(name: str, config: dict) -> BaseTranscriber:
    """工厂方法：按名称创建听写后端实例

    Args:
        name: 模型名称，如 "whisper-base", "faster-whisper-large-v3"
        config: 配置字典

    Returns:
        BaseTranscriber 实例

    Raises:
        ValueError: 未知的模型名称
    """
    # 精确匹配
    if name in TRANSCRIBER_REGISTRY:
        return TRANSCRIBER_REGISTRY[name](config)

    # 前缀匹配（如 whisper-large-v3 → whisper- 前缀 → WhisperTranscriber）
    for prefix, cls in _PREFIX_REGISTRY.items():
        if name.startswith(prefix):
            return cls(config)

    available = ", ".join(TRANSCRIBER_REGISTRY.keys())
    raise ValueError(f"未知的听写模型: {name}，可用模型: {available}")


def list_transcribers() -> list[dict]:
    """列出所有已注册的听写后端"""
    return [
        {
            "name": cls.name,
            "display_name": cls.display_name,
            "supports_timestamps": cls.supports_timestamps,
        }
        for cls in TRANSCRIBER_REGISTRY.values()
    ]
