"""OpenAI Whisper 听写后端——通过 stable-ts 调用"""

import numpy as np
from asrlabs.models import TranscriptionResult, Segment, Word
from asrlabs.transcribe.base import BaseTranscriber
from asrlabs.transcribe.base import register_transcriber


@register_transcriber
class WhisperTranscriber(BaseTranscriber):
    """OpenAI Whisper 听写后端（使用 stable-ts）

    模型命名: whisper-tiny, whisper-base, whisper-small,
              whisper-medium, whisper-large-v3
    """

    name = "whisper"
    display_name = "OpenAI Whisper (stable-ts)"
    supports_timestamps = True
    recommended_aligner = "whisper_align"

    def load_model(self) -> None:
        """加载 Whisper 模型（通过 stable-ts）

        model_path 为空时使用 "base"（stable-ts 内置尺寸），
        否则可以是本地 .pt 文件路径或 stable-ts 尺寸名（tiny/base/small/medium/large-v3）。
        """
        import stable_whisper

        model_path = self.model_path or "base"
        device = self.config.get("device", "auto")
        if device == "auto":
            device = "cuda" if self._cuda_available() else "cpu"

        self._model = stable_whisper.load_model(model_path, device=device)

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
            model=self.name,
            has_timestamps=True,
        )

    @staticmethod
    def _cuda_available() -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

