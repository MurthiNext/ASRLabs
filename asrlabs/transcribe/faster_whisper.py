"""Faster Whisper 听写后端——通过 stable-ts 的 CTranslate2 后端调用"""

import numpy as np
from asrlabs.models import TranscriptionResult, Segment, Word
from asrlabs.transcribe.base import BaseTranscriber
from asrlabs.transcribe import register_transcriber, _PREFIX_REGISTRY


@register_transcriber
class FasterWhisperTranscriber(BaseTranscriber):
    """Faster Whisper 听写后端（CTranslate2 引擎，通过 stable-ts）

    模型命名: faster-whisper-tiny, faster-whisper-base,
              faster-whisper-large-v3
    """

    name = "faster-whisper-base"
    display_name = "Faster Whisper (CTranslate2 via stable-ts)"
    supports_timestamps = True
    recommended_aligner = "whisper_align"

    def load_model(self) -> None:
        """加载 Faster Whisper 模型（通过 stable-ts load_faster_whisper）"""
        import stable_whisper

        # 提取模型尺寸（faster-whisper-large-v3 → large-v3）
        size = self.config.get("model", self.name).split("-", 2)[2]
        device = self.config.get("device", "auto")
        compute_type = self.config.get("compute_type", "float16")
        if device == "auto":
            device = "cuda" if self._cuda_available() else "cpu"

        self._model = stable_whisper.load_faster_whisper(
            size, device=device, compute_type=compute_type
        )

    def transcribe(
        self, audio: str | np.ndarray, **kwargs
    ) -> TranscriptionResult:
        """执行听写"""
        self._ensure_loaded()

        extras = {**self.config.get("extras", {}), **kwargs}
        language = self.config.get("language", "auto")
        if language == "auto":
            language = None
        beam_size = self.config.get("beam_size", 5)

        result = self._model.transcribe(
            audio,
            language=language,
            beam_size=beam_size,
            word_timestamps=True,
            vad=extras.get("vad", True),
            **{k: v for k, v in extras.items() if k not in ("vad",)},
        )

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
            model=self.config.get("model", "faster-whisper-base"),
            has_timestamps=True,
        )

    @staticmethod
    def _cuda_available() -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False


_PREFIX_REGISTRY["faster-whisper-"] = FasterWhisperTranscriber
