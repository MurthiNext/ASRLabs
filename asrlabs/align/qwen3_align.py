"""Qwen3 Forced Aligner 对齐器——通过 qwen-asr 的 Qwen3ForcedAligner 实现"""

import numpy as np
from asrlabs.models import TranscriptionResult, Segment, Word
from asrlabs.align.base import BaseAligner
from asrlabs.align.base import register_aligner


@register_aligner
class Qwen3Aligner(BaseAligner):
    """Qwen3 Forced Aligner 对齐器

    独立对齐器，适用于任何听写模型产出的文本。
    支持 11 种语言，单段最长 5 分钟，需要 GPU。
    """

    name = "qwen3_align"
    display_name = "Qwen3 Forced Aligner"

    def load_model(self) -> None:
        """加载 Qwen3 Forced Aligner 模型

        model_path 为空时默认使用 Qwen/Qwen3-ForcedAligner-0.6B，
        否则可以是 HuggingFace 模型 ID 或本地缓存路径。
        """
        from qwen_asr import Qwen3ForcedAligner
        import torch

        model_path = self.model_path or "Qwen/Qwen3-ForcedAligner-0.6B"
        device = self.config.get("device", "auto")
        if device == "auto":
            device = "cuda:0" if torch.cuda.is_available() else "cpu"

        self._model = Qwen3ForcedAligner.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            device_map=device,
        )

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
            lang = "Chinese"  # Qwen3ForcedAligner 需要明确语言

        # 按 segment 逐个对齐（每个 segment ≤5 分钟）
        aligned_segments = []
        for seg in result.segments:
            if not seg.text.strip():
                aligned_segments.append(seg)
                continue

            try:
                align_results = self._model.align(
                    audio=audio,
                    text=seg.text.strip(),
                    language=lang,
                )
            except Exception:
                # 对齐失败则保留原 segment（无时间戳）
                aligned_segments.append(seg)
                continue

            words = []
            if align_results:
                for token in align_results[0]:
                    words.append(Word(
                        text=token.text,
                        start=token.start_time,
                        end=token.end_time,
                    ))

                aligned_segments.append(Segment(
                    text=seg.text.strip(),
                    start=words[0].start if words else 0.0,
                    end=words[-1].end if words else 0.0,
                    words=words,
                ))
            else:
                aligned_segments.append(seg)

        # 重新拼接全文
        full_text = " ".join(s.text for s in aligned_segments if s.text)

        return TranscriptionResult(
            text=full_text.strip(),
            segments=aligned_segments,
            language=lang,
            duration=result.duration,
            model=result.model,
            has_timestamps=any(
                s.words and s.words[0].start > 0 for s in aligned_segments if s.words
            ),
        )
