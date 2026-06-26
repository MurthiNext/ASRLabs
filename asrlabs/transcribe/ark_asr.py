"""ARK-ASR 听写后端——通过 transformers + trust_remote_code 调用

使用方式:
    from asrlabs.transcribe import ArkASRTranscriber
    t = ArkASRTranscriber({"model_path": "AutoArk-AI/ARK-ASR-3B", "device": "cuda"})
    t.load_model()
    result = t.transcribe("audio.wav")
"""

import logging
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from asrlabs.models import TranscriptionResult, Segment
from asrlabs.transcribe.base import BaseTranscriber, register_transcriber

logger = logging.getLogger(__name__)


@register_transcriber
class ArkASRTranscriber(BaseTranscriber):
    """ARK-ASR 听写后端（transformers + trust_remote_code）

    模型默认 AutoArk-AI/ARK-ASR-3B（3B，BF16），单段上限 30s，无时间戳。
    """

    name = "ark-asr"
    display_name = "ARK-ASR"
    supports_timestamps = False
    recommended_aligner = "qwen3_align"

    def load_model(self) -> None:
        """加载 ARK-ASR 模型 / 处理器 / 分词器

        三者均需 trust_remote_code=True 以加载自定义 arkasr 远程代码。
        精度: cuda -> bfloat16, cpu -> float32（与 granite 一致）。
        """
        import torch
        from transformers import (
            AutoModelForCausalLM,
            AutoProcessor,
            AutoTokenizer,
        )

        model_path = self.model_path or "AutoArk-AI/ARK-ASR-3B"
        device = self.config.get("device", "auto")
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        is_cuda = isinstance(device, str) and device.startswith("cuda")
        torch_dtype = torch.bfloat16 if is_cuda else torch.float32

        self._model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            attn_implementation="sdpa",
        ).to(device)
        self._model.eval()

        self._processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True,
        )
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True,
        )
        self._device = device
        self._dtype = torch_dtype

        # 预构造 bad_words_ids，过滤非 ASR 控制标记（官方推荐做法）
        self._bad_words_ids = self._build_bad_words_ids()
        logger.info("ARK-ASR 模型已加载到 %s（%s）", device, torch_dtype)

    def transcribe(
        self, audio: str | np.ndarray, **kwargs: Any,
    ) -> TranscriptionResult:
        """执行听写

        ARK 的 apply_chat_template 需要音频文件路径，VAD 产出的 numpy chunk
        先写临时 WAV 再调用（与 qwen3_asr 相同的模式）。
        """
        self._ensure_loaded()

        if isinstance(audio, np.ndarray):
            import soundfile as sf
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                tmp_path = f.name
            sf.write(tmp_path, audio, 16000, subtype="PCM_16")
            try:
                return self._run_inference(tmp_path, kwargs)
            finally:
                Path(tmp_path).unlink(missing_ok=True)
        return self._run_inference(audio, kwargs)

    def _run_inference(
        self, audio_path: str, kwargs: dict,
    ) -> TranscriptionResult:
        """构建 chat template -> generate -> 解码

        核心流程:
          1. 构造 [{"role":"user","content":[{"type":"audio","path":...},{"type":"text",...}]}]
          2. processor.apply_chat_template 生成模型输入（audio_max_length=30s）
          3. model.generate(do_sample=False, bad_words_ids=...) 贪心解码
          4. 跳过输入 token 解码生成文本
        """
        import torch

        max_new_tokens = kwargs.pop("max_new_tokens", 256)

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "path": audio_path},
                    {"type": "text", "text": "Please transcribe this audio."},
                ],
            }
        ]

        inputs = self._processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            return_tensors="pt",
            sampling_rate=16000,
            audio_padding="longest",
            text_kwargs={"padding": "longest"},
            audio_max_length=30 * 16000,
        )
        inputs = inputs.to(self._device)
        if "audios" in inputs:
            inputs["audios"] = inputs["audios"].to(dtype=self._dtype)

        with torch.inference_mode():
            outputs = self._model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                pad_token_id=self._tokenizer.pad_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
                bad_words_ids=self._bad_words_ids,
            )

        # 跳过输入部分，只解码新生成的 token
        new_tokens = outputs[:, inputs.input_ids.shape[1]:]
        text = self._tokenizer.batch_decode(
            new_tokens, skip_special_tokens=True,
        )[0]

        text = text.strip()
        return TranscriptionResult(
            text=text,
            segments=[Segment(text=text, start=0.0, end=0.0)],
            language=self.config.get("language", "auto"),
            model=self.name,
            has_timestamps=False,
        )

    def _build_bad_words_ids(self) -> list[list[int]]:
        """构造 bad_words_ids，屏蔽非 ASR 控制标记

        保留 eos_token_id，屏蔽其余 special tokens 及 <...> 形态的占位符，
        防止模型在转录文本中吐出控制标记（官方推荐做法）。
        """
        tokenizer = self._tokenizer
        eos_ids = tokenizer.eos_token_id
        keep_ids = {eos_ids} if isinstance(eos_ids, int) else set(eos_ids or [])
        bad_ids = set(tokenizer.all_special_ids) - keep_ids
        bad_ids.update(
            token_id
            for token, token_id in tokenizer.get_added_vocab().items()
            if token.startswith("<") and token.endswith(">")
            and token_id not in keep_ids
        )
        return [[token_id] for token_id in sorted(bad_ids)]

    def unload(self) -> None:
        """释放模型资源"""
        super().unload()
        self._processor = None
        self._tokenizer = None
        self._bad_words_ids = None
