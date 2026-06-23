"""对齐后端——策略模式注册表"""

from asrlabs.align.base import BaseAligner

ALIGNER_REGISTRY: dict[str, type[BaseAligner]] = {}


def register_aligner(cls: type[BaseAligner]) -> type[BaseAligner]:
    """装饰器：将对齐后端注册到全局注册表"""
    if not cls.name:
        raise ValueError(f"{cls.__name__} 必须设置 name 类属性")
    ALIGNER_REGISTRY[cls.name] = cls
    return cls


def get_aligner(name: str, config: dict) -> BaseAligner | None:
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
