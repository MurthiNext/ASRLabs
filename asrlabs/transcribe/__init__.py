"""听写后端——策略模式注册表

用户可直接导入模型类:
    from asrlabs.transcribe import WhisperTranscriber
    from asrlabs.transcribe import FasterWhisperTranscriber
    from asrlabs.transcribe import Qwen3ASRTranscriber
    from asrlabs.transcribe import GraniteSpeechTranscriber
    from asrlabs.transcribe import CohereTranscriber
"""

from asrlabs.transcribe.base import (  # noqa: F401 — re-export
    TRANSCRIBER_REGISTRY,
    BaseTranscriber,
    get_transcriber,
    list_transcribers,
    register_transcriber,
)

# ── 本地模型 —— 导入即注册 ──
from asrlabs.transcribe.whisper import WhisperTranscriber  # noqa: F401
from asrlabs.transcribe.faster_whisper import FasterWhisperTranscriber  # noqa: F401
from asrlabs.transcribe.qwen3_asr import Qwen3ASRTranscriber  # noqa: F401
from asrlabs.transcribe.granite_speech import GraniteSpeechTranscriber  # noqa: F401
from asrlabs.transcribe.cohere import CohereTranscriber  # noqa: F401  # 本地模式

__all__ = [
    "TRANSCRIBER_REGISTRY",
    "BaseTranscriber",
    "get_transcriber",
    "list_transcribers",
    "register_transcriber",
    "WhisperTranscriber",
    "FasterWhisperTranscriber",
    "Qwen3ASRTranscriber",
    "GraniteSpeechTranscriber",
    "CohereTranscriber",
]
