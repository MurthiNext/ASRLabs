"""Qwen3 ASR 听写后端——通过 qwen-asr 库调用"""

import tempfile
from pathlib import Path
import numpy as np
from asrlabs.models import TranscriptionResult, Segment
from asrlabs.transcribe.base import BaseTranscriber
from asrlabs.transcribe.base import register_transcriber


@register_transcriber
class Qwen3ASRTranscriber(BaseTranscriber):
    """Qwen3 ASR 听写后端"""

    name = "qwen3-asr"
    display_name = "Qwen3 ASR"
    supports_timestamps = False
    recommended_aligner = "qwen3_align"

    def load_model(self) -> None:
        """加载 Qwen3 ASR 模型

        model_path 为空时默认使用 Qwen/Qwen3-ASR-1.7B，
        否则可以是 HuggingFace 模型 ID 或本地缓存路径。
        """
        from qwen_asr import Qwen3ASRModel
        import torch

        model_path = self.model_path or "Qwen/Qwen3-ASR-1.7B"
        device = self.config.get("device", "auto")

        self._model = Qwen3ASRModel.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            device_map=device if device != "cpu" else None,
        )

    def transcribe(
        self, audio: str | np.ndarray, **kwargs
    ) -> TranscriptionResult:
        """执行听写

        Qwen3ASRModel.transcribe 不接受裸 numpy 数组，需要文件路径。
        VAD 分出的 numpy chunk 先写临时 WAV。
        """
        self._ensure_loaded()

        language = self.config.get("language", "auto")
        if language == "auto":
            language = None

        # numpy 数组 -> 临时 WAV
        if isinstance(audio, np.ndarray):
            import soundfile as sf
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                tmp_path = f.name
            sf.write(tmp_path, audio, 16000, subtype="PCM_16")
            try:
                results = self._model.transcribe(audio=tmp_path, language=language)
            finally:
                Path(tmp_path).unlink(missing_ok=True)
        else:
            results = self._model.transcribe(audio=audio, language=language)

        if not results:
            return TranscriptionResult(
                text="",
                language=language or "auto",
                model=self.name,
            )

        full_text = " ".join(r.text for r in results if r.text)
        segments = []
        for r in results:
            segments.append(Segment(
                text=r.text.strip() if r.text else "",
                start=0.0,
                end=0.0,
            ))

        detected_lang = getattr(results[0], "language", language or "auto")

        return TranscriptionResult(
            text=full_text.strip(),
            segments=segments if segments else [Segment(full_text.strip(), 0.0, 0.0)],
            language=detected_lang,
            model=self.name,
            has_timestamps=False,
        )
