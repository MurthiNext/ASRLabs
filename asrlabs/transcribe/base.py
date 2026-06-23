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
    supported_devices: list[str] = ["cpu", "cuda"]  # 子类可覆盖添加 vulkan 等

    def __init__(self, config: dict):
        """初始化听写后端

        Args:
            config: 配置字典，至少包含 model_path, device, compute_type, language, beam_size, extras
        """
        import logging

        self.config = config
        self.model_path = config.get("model_path", "")  # 本地模型路径，空则用默认
        self._model = None
        self._loaded = False

        # Vulkan 回退：仅 CTranslate2 (faster-whisper) 支持，其他模型自动切到 cpu
        raw_device = config.get("device", "auto")
        if raw_device == "vulkan" and "vulkan" not in self.supported_devices:
            logger = logging.getLogger(__name__)
            logger.warning(
                "%s 不支持 vulkan（仅 faster-whisper 的 CTranslate2 后端支持），"
                "已自动回退到 cpu。支持: %s",
                self.name, ", ".join(self.supported_devices),
            )
            config["device"] = "cpu"

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
"""全局注册表：{引擎名: 后端类}"""


def register_transcriber(cls: type[BaseTranscriber]) -> type[BaseTranscriber]:
    """装饰器：将听写后端注册到全局注册表"""
    if not cls.name:
        raise ValueError(f"{cls.__name__} 必须设置 name 类属性")
    TRANSCRIBER_REGISTRY[cls.name] = cls
    return cls


def get_transcriber(name: str, config: dict) -> BaseTranscriber:
    """工厂方法：按引擎名创建听写后端实例

    Args:
        name: 引擎名，如 "whisper", "faster-whisper"
        config: 配置字典（含 model_path, device, language 等）

    Returns:
        BaseTranscriber 实例

    Raises:
        ValueError: 未知的引擎名
    """
    if name not in TRANSCRIBER_REGISTRY:
        available = ", ".join(TRANSCRIBER_REGISTRY.keys())
        raise ValueError(f"未知的听写引擎: {name}，可用引擎: {available}")
    return TRANSCRIBER_REGISTRY[name](config)


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
