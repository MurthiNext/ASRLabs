"""听写后端——策略模式注册表"""

from asrlabs.transcribe.base import BaseTranscriber

# 全局注册表：{模型名称: 后端类}
TRANSCRIBER_REGISTRY: dict[str, type[BaseTranscriber]] = {}


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
    # 尝试精确匹配
    if name in TRANSCRIBER_REGISTRY:
        return TRANSCRIBER_REGISTRY[name](config)

    # 尝试前缀匹配（如 whisper-base → whisper 类处理所有 whisper-*）
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


# 前缀注册表——用于匹配 whisper-base/whisper-large-v3 等到同一个类
_PREFIX_REGISTRY: dict[str, type[BaseTranscriber]] = {}

# 导入具体后端触发注册
from asrlabs.transcribe import whisper  # noqa: F401 — 确保 @register_transcriber 被执行
from asrlabs.transcribe import faster_whisper  # noqa: F401
from asrlabs.transcribe import qwen3_asr  # noqa: F401
from asrlabs.transcribe import cohere  # noqa: F401
from asrlabs.transcribe import granite_speech  # noqa: F401
