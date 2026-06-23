"""Whisper 对齐器——通过 stable-ts 的 align() 和 refine() 实现"""

import numpy as np
from asrlabs.models import TranscriptionResult, Segment, Word
from asrlabs.align.base import BaseAligner
from asrlabs.align.base import register_aligner


@register_aligner
class WhisperAligner(BaseAligner):
    """Stable Whisper 对齐器

    利用 Whisper 交叉注意力机制对齐文本和音频。
    仅适用于 Whisper/Faster Whisper 听写模型。
    """

    name = "whisper_align"
    display_name = "Stable Whisper Align"

    def load_model(self) -> None:
        """加载同款 Whisper 模型用于对齐"""
        import stable_whisper

        # 对齐需要与听写相同尺寸的模型，默认使用 base
        size = self.config.get("extras", {}).get("model_size", "base")
        device = self.config.get("device", "auto")
        if device == "auto":
            device = "cuda" if self._cuda_available() else "cpu"

        self._model = stable_whisper.load_model(size, device=device)

    def align(
        self,
        audio: str | np.ndarray,
        result: TranscriptionResult,
        language: str | None = None,
    ) -> TranscriptionResult:
        """对齐音频和文本"""
        self._ensure_loaded()

        lang = language or result.language
        if lang == "auto":
            lang = None

        do_refine = self.config.get("extras", {}).get("refine", True)

        # stable-ts align: 将完整文本对齐到音频
        aligned = self._model.align(audio, result.text, language=lang)

        # 可选：微调边界
        if do_refine:
            aligned = self._model.refine(audio, aligned)

        # 转换为统一结果
        segments = []
        for seg in aligned.segments:
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
            ))

        return TranscriptionResult(
            text=aligned.text.strip(),
            segments=segments,
            language=lang or result.language,
            duration=result.duration,
            model=result.model,
            has_timestamps=True,
        )

    @staticmethod
    def _cuda_available() -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
