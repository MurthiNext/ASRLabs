"""OpenAI Whisper 听写后端——通过 stable-ts 调用"""

import numpy as np
from asrlabs.models import TranscriptionResult, Segment, Word
from asrlabs.transcribe.base import BaseTranscriber
from asrlabs.transcribe import register_transcriber, _PREFIX_REGISTRY


@register_transcriber
class WhisperTranscriber(BaseTranscriber):
    """OpenAI Whisper 听写后端（使用 stable-ts）

    模型命名: whisper-tiny, whisper-base, whisper-small,
              whisper-medium, whisper-large-v3
    """

    name = "whisper-base"  # 注册名（前缀匹配: whisper-*）
    display_name = "OpenAI Whisper (stable-ts)"
    supports_timestamps = True
    recommended_aligner = "whisper_align"

    def load_model(self) -> None:
        """加载 Whisper 模型（通过 stable-ts）"""
        import stable_whisper

        # 提取模型尺寸（如 whisper-base → base）
        size = self.config.get("model", "base").split("-", 1)[1]
        device = self.config.get("device", "auto")
        if device == "auto":
            device = "cuda" if self._cuda_available() else "cpu"

        self._model = stable_whisper.load_model(size, device=device)

    def transcribe(
        self, audio: str | np.ndarray, **kwargs
    ) -> TranscriptionResult:
        """执行听写"""
        self._ensure_loaded()

        extras = {**self.config.get("extras", {}), **kwargs}
        language = self.config.get("language", "auto")
        if language == "auto":
            language = None

        # stable-ts transcribe 返回 WhisperResult 对象
        result = self._model.transcribe(
            audio,
            language=language,
            word_timestamps=True,
            vad=extras.get("vad", True),
            **{k: v for k, v in extras.items() if k != "vad"},
        )

        # 转换为统一 TranscriptionResult
        segments = []
        for seg in result.segments:
            words = []
            if hasattr(seg, "words") and seg.words:
                for w in seg.words:
                    words.append(Word(
                        text=w.word.strip(),
                        start=w.start,
                        end=w.end,
                        confidence=getattr(w, "probability", 1.0),
                    ))
            segments.append(Segment(
                text=seg.text.strip(),
                start=seg.start,
                end=seg.end,
                words=words,
                confidence=getattr(seg, "avg_logprob", 0.0),
            ))

        return TranscriptionResult(
            text=result.text.strip(),
            segments=segments,
            language=getattr(result, "language", "auto"),
            duration=getattr(result, "duration", 0.0),
            model=self.config.get("model", "whisper-base"),
            has_timestamps=True,
        )

    @staticmethod
    def _cuda_available() -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False


# 注册前缀匹配——使 whisper-tiny/whisper-small 等都路由到此类
_PREFIX_REGISTRY["whisper-"] = WhisperTranscriber
