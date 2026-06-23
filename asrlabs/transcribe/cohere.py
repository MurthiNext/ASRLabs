"""Cohere Transcribe 听写后端

当前状态: 本地模式可用，云端 API 模式待后续支持。

使用方式（需手动导入）:
    from asrlabs.transcribe.cohere import CohereTranscriber
    t = CohereTranscriber({"model": "cohere-transcribe", "device": "cuda",
                           "extras": {"mode": "local"}})
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
    当前支持: local 模式（HuggingFace transformers 本地运行）
    后续计划: cloud 模式（Cohere API）
    """

    name = "cohere-transcribe"
    display_name = "Cohere Transcribe"
    supports_timestamps = False
    recommended_aligner = "qwen3_align"

    def load_model(self) -> None:
        """加载本地 transformers 模型

        TODO: 后续支持 cloud 模式时，根据 config.extras.mode 切换
              mode == "cloud" → cohere.ClientV2(api_key=...)
              mode == "local" → 当前逻辑
        """
        from transformers import AutoProcessor, CohereAsrForConditionalGeneration

        model_id = "CohereLabs/cohere-transcribe-03-2026"
        self._processor = AutoProcessor.from_pretrained(model_id)
        self._model = CohereAsrForConditionalGeneration.from_pretrained(
            model_id, device_map=self.config.get("device", "auto")
        )

    def transcribe(
        self, audio: str | np.ndarray, **kwargs
    ) -> TranscriptionResult:
        """执行听写（当前仅本地模式）"""
        self._ensure_loaded()

        language = self.config.get("language", "auto")
        if language == "auto":
            language = "en"

        # TODO: 后续支持 cloud 模式分支
        return self._transcribe_local(audio, language)

    # ── 云端模式 —— 后续支持 ──
    # def _transcribe_cloud(self, audio, language):
    #     """通过 Cohere 云端 API 听写（TODO）"""
    #     if isinstance(audio, np.ndarray):
    #         raise ValueError("Cohere 云端模式需要文件路径，不支持 numpy 数组")
    #     with open(audio, "rb") as f:
    #         response = self._client.audio.transcriptions.create(
    #             model="cohere-transcribe-03-2026",
    #             language=language,
    #             file=f,
    #         )
    #     text = response.text if hasattr(response, "text") else str(response)
    #     return TranscriptionResult(
    #         text=text.strip(),
    #         segments=[Segment(text.strip(), 0.0, 0.0)],
    #         language=language,
    #         model="cohere-transcribe",
    #         has_timestamps=False,
    #     )

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
            audio_array, sampling_rate=16000, return_tensors="pt", language=language
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
