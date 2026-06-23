"""Cohere Transcribe 听写后端

需要 transformers >= 5.4.0（原生支持 CohereAsrForConditionalGeneration）。
云端 API 模式待后续支持。

使用方式:
    from asrlabs.transcribe import CohereTranscriber
    t = CohereTranscriber({
        "model": "cohere-transcribe",
        "model_path": "CohereLabs/cohere-transcribe-03-2026",  # 或本地路径
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
    """Cohere Transcribe 听写后端（transformers >= 5.4.0 原生支持）"""

    name = "cohere-transcribe"
    display_name = "Cohere Transcribe"
    supports_timestamps = False
    recommended_aligner = "qwen3_align"

    def load_model(self) -> None:
        """加载 Cohere Transcribe 模型（本地 transformers）"""
        import torch
        from transformers import AutoProcessor, CohereAsrForConditionalGeneration

        model_path = self.model_path or "CohereLabs/cohere-transcribe-03-2026"

        # CPU 用 float32，GPU 可用 bfloat16
        device = self.config.get("device", "auto")
        if device == "cpu" or (device == "auto" and not torch.cuda.is_available()):
            torch_dtype = torch.float32
        else:
            torch_dtype = torch.bfloat16

        self._processor = AutoProcessor.from_pretrained(model_path)
        self._model = CohereAsrForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map=device,
        )

    def transcribe(
        self, audio: str | np.ndarray, **kwargs
    ) -> TranscriptionResult:
        """执行听写"""
        self._ensure_loaded()

        language = self.config.get("language", "auto")
        if language == "auto":
            language = "en"

        return self._transcribe_local(audio, language)

    # ── 云端模式 —— 后续支持 ──
    # def _transcribe_cloud(self, audio, language):
    #     """通过 Cohere 云端 API 听写（TODO）"""
    #     ...

    def _transcribe_local(
        self, audio: str | np.ndarray, language: str
    ) -> TranscriptionResult:
        """通过本地 transformers 模型听写"""
        from transformers.audio_utils import load_audio as hf_load_audio
        import torch

        if isinstance(audio, str):
            audio_array = hf_load_audio(audio, sampling_rate=16000)
        else:
            audio_array = audio

        inputs = self._processor(
            audio=audio_array, sampling_rate=16000, return_tensors="pt",
            language=language,
        ).to(self._model.device)

        with torch.no_grad():
            outputs = self._model.generate(**inputs, max_new_tokens=256)

        text = self._processor.decode(outputs[0], skip_special_tokens=True)

        return TranscriptionResult(
            text=text.strip(),
            segments=[Segment(text.strip(), 0.0, 0.0)],
            language=language,
            model="cohere-transcribe",
            has_timestamps=False,
        )
