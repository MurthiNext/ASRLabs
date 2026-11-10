"""IBM Granite Speech 听写后端——通过 transformers 调用"""

import numpy as np
from asrlabs.models import TranscriptionResult, Segment
from asrlabs.transcribe.base import BaseTranscriber
from asrlabs.transcribe.base import register_transcriber


@register_transcriber
class GraniteSpeechTranscriber(BaseTranscriber):
    """IBM Granite Speech 听写后端

    模型命名: granite-speech-2b, granite-speech-2b-plus, granite-speech-8b
    """

    name = "granite-speech"
    display_name = "IBM Granite Speech"
    supports_timestamps = False  # 基础版无时间戳，2b-plus 有
    recommended_aligner = "qwen3_align"

    def load_model(self) -> None:
        """加载 Granite Speech 模型

        model_path 为空时默认使用 ibm-granite/granite-speech-4.1-2b，
        否则可以是 HuggingFace 模型 ID 或本地缓存路径。
        """
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
        import torch

        model_path = self.model_path or "ibm-granite/granite-speech-4.1-2b"
        device = self.config.get("device", "auto")
        if device == "cpu" or (device == "auto" and not torch.cuda.is_available()):
            dtype = torch.float32
        else:
            dtype = torch.bfloat16

        self._model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_path,
            device_map=device,
            dtype=dtype,
        )
        self._processor = AutoProcessor.from_pretrained(
            model_path, fix_mistral_regex=True,
        )
        self._tokenizer = self._processor.tokenizer

    def transcribe(
        self, audio: str | np.ndarray, **kwargs
    ) -> TranscriptionResult:
        """执行听写"""
        self._ensure_loaded()
        import torch

        # 加载音频
        if isinstance(audio, str):
            import soundfile as sf
            import torchaudio
            wav_np, sr = sf.read(audio, dtype="float32")
            if wav_np.ndim > 1:
                wav_np = wav_np.mean(axis=1)
            if sr != 16000:
                t = torch.from_numpy(wav_np).unsqueeze(0)
                wav = torchaudio.transforms.Resample(sr, 16000)(t)
            else:
                wav = torch.from_numpy(wav_np).unsqueeze(0)
        else:
            wav = torch.from_numpy(audio).float()
            if wav.ndim == 1:
                wav = wav.unsqueeze(0)

        # 构建对话格式 prompt
        user_prompt = "<|audio|>transcribe the speech with proper punctuation and capitalization."
        chat = [{"role": "user", "content": user_prompt}]
        prompt = self._tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True
        )

        device = self._model.device if hasattr(self._model, "device") else "cpu"
        model_inputs = self._processor(
            prompt, wav, device=device, return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            model_outputs = self._model.generate(
                **model_inputs, max_new_tokens=200, do_sample=False, num_beams=1
            )

        # 解码（跳过输入 token）
        new_tokens = model_outputs[0, model_inputs["input_ids"].shape[-1]:].unsqueeze(0)
        text = self._tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0]

        return TranscriptionResult(
            text=text.strip(),
            segments=[Segment(text.strip(), 0.0, 0.0)],
            language=self.config.get("language", "auto"),
            model=self.name,
            has_timestamps=False,
        )
