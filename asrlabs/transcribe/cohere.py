"""Cohere Transcribe 听写后端

当前: transformers 4.x + trust_remote_code（等待 qwen-asr 兼容 transformers >= 5.4 后升级）。

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

import logging
import numpy as np
from asrlabs.models import TranscriptionResult, Segment
from asrlabs.transcribe.base import BaseTranscriber
from asrlabs.transcribe.base import register_transcriber

logger = logging.getLogger(__name__)


@register_transcriber
class CohereTranscriber(BaseTranscriber):
    """Cohere Transcribe 听写后端（transformers 4.x + trust_remote_code）"""

    name = "cohere-transcribe"
    display_name = "Cohere Transcribe"
    supports_timestamps = False
    recommended_aligner = "qwen3_align"

    def load_model(self) -> None:
        from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

        model_path = self.model_path or "CohereLabs/cohere-transcribe-03-2026"
        self._processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True,
        )
        self._model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_path,
            trust_remote_code=True,
            device_map=self.config.get("device", "auto"),
        )
        # 修补缓存模型文件的 decoder_attention_mask bug
        self._patch_model_generate()

    def _patch_model_generate(self):
        """修补 Cohere 缓存模型文件的 generate() 中 decoder_attention_mask=None 问题"""
        import pathlib
        cache_dir = pathlib.Path.home() / ".cache" / "huggingface" / "modules" \
                    / "transformers_modules" / "cohere_hyphen_transcribe_hyphen_03_hyphen_2026"
        target = cache_dir / "modeling_cohere_asr.py"
        if not target.exists():
            return
        content = target.read_text(encoding="utf-8")
        if "elif decoder_attention_mask is None:" in content:
            return  # 已修补

        old = """        decoder_attention_mask = kwargs.pop("decoder_attention_mask", None)
        if decoder_input_ids is not None and decoder_attention_mask is None:
            decoder_attention_mask = torch.ones_like(
                decoder_input_ids, dtype=torch.long, device=decoder_input_ids.device
            )

        generation_kwargs = dict(kwargs)"""
        new = """        decoder_attention_mask = kwargs.pop("decoder_attention_mask", None)
        if decoder_input_ids is not None and decoder_attention_mask is None:
            decoder_attention_mask = torch.ones_like(
                decoder_input_ids, dtype=torch.long, device=decoder_input_ids.device
            )
        elif decoder_attention_mask is None:
            batch_size = input_features.shape[0] if input_features is not None else 1
            decoder_attention_mask = torch.ones(
                (batch_size, 1), dtype=torch.long,
                device=input_features.device if input_features is not None else "cpu",
            )

        generation_kwargs = dict(kwargs)"""
        if old in content:
            target.write_text(content.replace(old, new), encoding="utf-8")
            logger.info("Cohere 模型 generate() 已修补 (decoder_attention_mask)")

    def transcribe(
        self, audio: str | np.ndarray, **kwargs
    ) -> TranscriptionResult:
        self._ensure_loaded()

        language = self.config.get("language", "auto")
        if language == "auto":
            language = "en"

        return self._transcribe_local(audio, language)

    def _transcribe_local(
        self, audio: str | np.ndarray, language: str
    ) -> TranscriptionResult:
        import torch
        import torchaudio

        if isinstance(audio, str):
            import soundfile as sf
            audio_array, sr = sf.read(audio, dtype="float32")
            if audio_array.ndim > 1:
                audio_array = audio_array.mean(axis=1)
            if sr != 16000:
                t = torch.from_numpy(audio_array).unsqueeze(0)
                audio_array = torchaudio.transforms.Resample(sr, 16000)(t).squeeze(0).numpy()
        else:
            audio_array = audio

        inputs = self._processor(
            audio=audio_array, sampling_rate=16000, return_tensors="pt",
            language=language,
        )
        inputs = {k: v.to(device=self._model.device, dtype=self._model.dtype)
                  if v.dtype in (torch.float32, torch.float16, torch.bfloat16)
                  else v.to(device=self._model.device)
                  for k, v in inputs.items()}

        # 确保 decoder_input_ids 存在（Cohere encoder-decoder 必需）
        if "decoder_input_ids" not in inputs and "input_ids" not in inputs:
            bos = getattr(self._model.config, "decoder_start_token_id", None) or 0
            inputs["decoder_input_ids"] = torch.tensor(
                [[bos]], device=self._model.device, dtype=torch.long,
            )

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
