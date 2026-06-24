"""Kotoba Whisper v2.2 日语听写后端

通过 HuggingFace transformers.pipeline + trust_remote_code 加载
kotoba-tech/kotoba-whisper-v2.2（Distil-Whisper，日语特化）。

使用方式:
    from asrlabs.transcribe import KotobaWhisperTranscriber
    t = KotobaWhisperTranscriber({"language": "ja"})
    t.load_model()
    result = t.transcribe("audio.wav")
"""

import logging
import numpy as np
from asrlabs.models import TranscriptionResult, Segment
from asrlabs.transcribe.base import BaseTranscriber
from asrlabs.transcribe.base import register_transcriber

logger = logging.getLogger(__name__)


@register_transcriber
class KotobaWhisperTranscriber(BaseTranscriber):
    """Kotoba Whisper v2.2 听写后端（transformers pipeline + trust_remote_code）

    模型默认针对日语，language 默认 "ja"（可经 config 覆盖）。
    模型原生 pipeline 内部含 15s 子分块 + 批量推理。
    """

    name = "kotoba-whisper"
    display_name = "Kotoba Whisper v2.2 (日本語)"
    supports_timestamps = True
    recommended_aligner = None  # pipeline 自身已含时间戳改进

    def load_model(self) -> None:
        """加载 Kotoba Whisper v2.2 pipeline

        model_path 为空时使用官方 HF ID "kotoba-tech/kotoba-whisper-v2.2"，
        否则视作本地模型目录或自定义 HF ID。
        """
        import torch
        from transformers import pipeline

        model_path = self.model_path or "kotoba-tech/kotoba-whisper-v2.2"
        device = self.config.get("device", "auto")
        if device == "auto":
            device = "cuda:0" if self._cuda_available() else "cpu"

        is_cuda = isinstance(device, str) and device.startswith("cuda")
        torch_dtype = torch.float16 if is_cuda else torch.float32
        model_kwargs = {"attn_implementation": "sdpa"} if is_cuda else {}

        self._model = pipeline(
            model=model_path,
            torch_dtype=torch_dtype,
            device=device,
            model_kwargs=model_kwargs,
            chunk_length_s=15,
            trust_remote_code=True,
        )

    def transcribe(
        self, audio: str | np.ndarray, **kwargs
    ) -> TranscriptionResult:
        """执行听写

        Kotoba 默认日语，用户可通过 config.language 覆盖。
        extras 透传 add_punctuation / batch_size 等参数给底层 pipeline。
        """
        self._ensure_loaded()

        # Kotoba 默认日语，与其它后端的 "auto" 默认不同
        language = self.config.get("language", "ja")
        generate_kwargs = {"language": language, "task": "transcribe"}

        extras = {**self.config.get("extras", {}), **kwargs}

        pipe_kwargs: dict = {"generate_kwargs": generate_kwargs}

        # 标点恢复（可选，需安装 punctuators 库）
        if extras.pop("add_punctuation", False):
            pipe_kwargs["add_punctuation"] = True
        # 批大小（默认 8）
        batch_size = extras.pop("batch_size", 8)
        if batch_size:
            pipe_kwargs["batch_size"] = batch_size

        result = self._model(audio, **pipe_kwargs)
        return self._parse_output(result, language)

    def _parse_output(
        self, result: dict, language: str
    ) -> TranscriptionResult:
        """将 pipeline 输出解析为 TranscriptionResult

        pipeline 基础模式返回 {"text": "..."}；
        diarization 场景返回 {"text": "...", "chunks": [...]}，
        每个 chunk 可含 timestamp=(start, end)。
        """
        full_text = (result.get("text") or "").strip()

        segments = []
        chunks = result.get("chunks")
        if chunks and isinstance(chunks, list):
            for chunk in chunks:
                text = (chunk.get("text") or "").strip()
                start, end = 0.0, 0.0
                ts = chunk.get("timestamp")
                if isinstance(ts, (list, tuple)) and len(ts) == 2:
                    start = float(ts[0]) if ts[0] is not None else 0.0
                    end = float(ts[1]) if ts[1] is not None else 0.0
                segments.append(Segment(text=text, start=start, end=end, words=[]))

        has_ts = bool(segments)
        if not segments:
            segments = [Segment(text=full_text, start=0.0, end=0.0, words=[])]

        return TranscriptionResult(
            text=full_text,
            segments=segments,
            language=language,
            duration=0.0,
            model=self.name,
            has_timestamps=has_ts,
        )

    @staticmethod
    def _cuda_available() -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
