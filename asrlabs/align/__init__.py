"""对齐后端——策略模式注册表"""

from asrlabs.align.base import (
    ALIGNER_REGISTRY,
    BaseAligner,
    get_aligner,
    list_aligners,
    register_aligner,
)
from asrlabs.align.whisper_align import WhisperAligner  # noqa: F401 — 触发 @register_aligner 装饰器

__all__ = [
    "ALIGNER_REGISTRY",
    "BaseAligner",
    "get_aligner",
    "list_aligners",
    "register_aligner",
    "WhisperAligner",
]
