"""Cohere Transcribe 听写后端

需要 transformers >= 4.52 + trust_remote_code。
云端 API 模式待后续支持。

使用方式:
    from asrlabs.transcribe import CohereTranscriber
    t = CohereTranscriber({
        "model": "cohere-transcribe",
        "model_path": "CohereLabs/cohere-transcribe-03-2026",
        "device": "cuda",
    })
    t.load_model()
    result = t.transcribe("audio.wav")
"""

import numpy as np
from asrlabs.models import TranscriptionResult, Segment
from asrlabs.transcribe.base import BaseTranscriber
from asrlabs.transcribe.base import register_transcriber


@register_transcriber
class CohereTranscriber(BaseTranscriber):
    """Cohere Transcribe 听写后端（transformers pipeline）"""

    name = "cohere-transcribe"
    display_name = "Cohere Transcribe"
    supports_timestamps = False
    recommended_aligner = "qwen3_align"

    def load_model(self) -> None:
        """通过 transformers pipeline 加载（避免 generate 兼容问题）"""
        from transformers import pipeline

        model_path = self.model_path or "CohereLabs/cohere-transcribe-03-2026"

        device = self.config.get("device", "auto")
        if device == "auto":
            import torch
            device = "cuda:0" if torch.cuda.is_available() else "cpu"

        self._pipe = pipeline(
            "automatic-speech-recognition",
            model=model_path,
            trust_remote_code=True,
            device=device,
        )
        self._model = True  # pipeline 已加载

    def transcribe(
        self, audio: str | np.ndarray, **kwargs
    ) -> TranscriptionResult:
        """执行听写"""
        self._ensure_loaded()

        language = self.config.get("language", "auto")
        if language == "auto":
            language = "en"

        # pipeline 音频输入
        if isinstance(audio, np.ndarray):
            # VAD 分出的 numpy chunk → pipeline 接受 dict
            pipe_input = {"array": audio, "sampling_rate": 16000}
        else:
            pipe_input = audio

        result = self._pipe(pipe_input,
                            generate_kwargs={"language": language, "max_new_tokens": 256})

        text = result["text"].strip() if isinstance(result, dict) else str(result).strip()
        return TranscriptionResult(
            text=text,
            segments=[Segment(text, 0.0, 0.0)],
            language=language,
            model="cohere-transcribe",
            has_timestamps=False,
        )
