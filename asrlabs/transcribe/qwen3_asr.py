"""Qwen3 ASR 听写后端——通过 qwen-asr 库调用"""

import numpy as np
from asrlabs.models import TranscriptionResult, Segment
from asrlabs.transcribe.base import BaseTranscriber
from asrlabs.transcribe import register_transcriber, _PREFIX_REGISTRY


@register_transcriber
class Qwen3ASRTranscriber(BaseTranscriber):
    """Qwen3 ASR 听写后端

    模型命名: qwen3-asr-0.6b, qwen3-asr-1.7b
    """

    name = "qwen3-asr-1.7b"
    display_name = "Qwen3 ASR"
    supports_timestamps = False  # 需要 forced_aligner 才有时间戳
    recommended_aligner = "qwen3_align"

    def load_model(self) -> None:
        """加载 Qwen3 ASR 模型"""
        from qwen_asr import Qwen3ASRModel
        import torch

        model_name = self._get_hf_model_name()
        device = self.config.get("device", "auto")

        self._model = Qwen3ASRModel.from_pretrained(
            model_name,
            dtype=torch.bfloat16,
            device_map=device if device != "cpu" else None,
        )

    def transcribe(
        self, audio: str | np.ndarray, **kwargs
    ) -> TranscriptionResult:
        """执行听写"""
        self._ensure_loaded()

        language = self.config.get("language", "auto")
        if language == "auto":
            language = None

        results = self._model.transcribe(audio=audio, language=language)

        # Qwen3ASR 返回列表，通常单段返回一个结果
        if not results:
            return TranscriptionResult(
                text="",
                language=language or "auto",
                model=self.config.get("model", "qwen3-asr-1.7b"),
            )

        full_text = " ".join(r.text for r in results if r.text)
        segments = []
        for r in results:
            segments.append(Segment(
                text=r.text.strip() if r.text else "",
                start=0.0,  # Qwen3 不带 forced_aligner 时无时间戳
                end=0.0,
            ))

        detected_lang = getattr(results[0], "language", language or "auto")

        return TranscriptionResult(
            text=full_text.strip(),
            segments=segments if segments else [Segment(full_text.strip(), 0.0, 0.0)],
            language=detected_lang,
            model=self.config.get("model", "qwen3-asr-1.7b"),
            has_timestamps=False,
        )

    def _get_hf_model_name(self) -> str:
        """将模型名转换为 HuggingFace 模型 ID"""
        model = self.config.get("model", "qwen3-asr-1.7b")
        variant = model.split("-", 2)[2]  # 1.7b 或 0.6b
        return f"Qwen/Qwen3-ASR-{variant.upper()}"


_PREFIX_REGISTRY["qwen3-asr-"] = Qwen3ASRTranscriber
