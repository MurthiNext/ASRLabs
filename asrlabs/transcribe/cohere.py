"""Cohere Transcribe 听写后端

当前状态: 本地模式可用（transformers >= 4.52 + trust_remote_code），
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
    """Cohere Transcribe 听写后端

    模型命名: cohere-transcribe
    当前支持: local 模式（HuggingFace transformers 本地运行，trust_remote_code=True）
    后续计划: cloud 模式（Cohere API）
    """

    name = "cohere-transcribe"
    display_name = "Cohere Transcribe"
    supports_timestamps = False
    recommended_aligner = "qwen3_align"

    def load_model(self) -> None:
        """加载本地 transformers 模型

        Cohere Transcribe 的模型架构代码在 HF 仓库中（非 transformers 主线），
        需要 trust_remote_code=True。后续 transformers >= 5.4.0 有原生支持。
        """
        from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

        model_path = self.model_path or "CohereLabs/cohere-transcribe-03-2026"

        self._processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True
        )
        self._model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_path,
            trust_remote_code=True,
            device_map=self.config.get("device", "auto"),
        )

    def transcribe(
        self, audio: str | np.ndarray, **kwargs
    ) -> TranscriptionResult:
        """执行听写（当前仅本地模式）"""
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
            audio_array, sampling_rate=16000, return_tensors="pt",
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
