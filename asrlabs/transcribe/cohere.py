"""Cohere Transcribe 听写后端——支持云端 API 和本地 transformers 两种模式"""

import numpy as np
from asrlabs.models import TranscriptionResult, Segment
from asrlabs.transcribe.base import BaseTranscriber
from asrlabs.transcribe import register_transcriber


@register_transcriber
class CohereTranscriber(BaseTranscriber):
    """Cohere Transcribe 听写后端

    模型命名: cohere-transcribe
    支持两种模式:
    - cloud: 通过 Cohere API（需要 API key）
    - local: 通过 HuggingFace transformers 本地运行
    """

    name = "cohere-transcribe"
    display_name = "Cohere Transcribe"
    supports_timestamps = False
    recommended_aligner = "qwen3_align"

    def load_model(self) -> None:
        """根据模式加载模型"""
        mode = self.config.get("extras", {}).get("mode", "local")

        if mode == "cloud":
            # 云端模式不需要预加载模型
            import cohere
            api_key = self.config.get("extras", {}).get("api_key")
            if not api_key:
                raise ValueError("Cohere 云端模式需要提供 api_key")
            self._client = cohere.ClientV2(api_key=api_key)
            self._model = True  # 标记已就绪
        else:
            from transformers import AutoProcessor, CohereAsrForConditionalGeneration
            model_id = "CohereLabs/cohere-transcribe-03-2026"
            self._processor = AutoProcessor.from_pretrained(model_id)
            self._model = CohereAsrForConditionalGeneration.from_pretrained(
                model_id, device_map=self.config.get("device", "auto")
            )

    def transcribe(
        self, audio: str | np.ndarray, **kwargs
    ) -> TranscriptionResult:
        """执行听写"""
        self._ensure_loaded()

        mode = self.config.get("extras", {}).get("mode", "local")
        language = self.config.get("language", "auto")
        if language == "auto":
            language = "en"  # Cohere 默认识别英语，需要显式指定

        if mode == "cloud":
            return self._transcribe_cloud(audio, language)
        else:
            return self._transcribe_local(audio, language)

    def _transcribe_cloud(
        self, audio: str | np.ndarray, language: str
    ) -> TranscriptionResult:
        """通过 Cohere 云端 API 听写"""
        if isinstance(audio, np.ndarray):
            raise ValueError("Cohere 云端模式需要文件路径，不支持 numpy 数组")

        with open(audio, "rb") as f:
            response = self._client.audio.transcriptions.create(
                model="cohere-transcribe-03-2026",
                language=language,
                file=f,
            )

        text = response.text if hasattr(response, "text") else str(response)
        return TranscriptionResult(
            text=text.strip(),
            segments=[Segment(text.strip(), 0.0, 0.0)],
            language=language,
            model="cohere-transcribe",
            has_timestamps=False,
        )

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
