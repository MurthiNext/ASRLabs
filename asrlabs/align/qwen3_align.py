"""Qwen3 Forced Aligner 对齐器——通过 qwen-asr 的 Qwen3ForcedAligner 实现"""

import logging
import tempfile
from pathlib import Path
import numpy as np
from asrlabs.models import TranscriptionResult, Segment, Word
from asrlabs.align.base import BaseAligner
from asrlabs.align.base import register_aligner

logger = logging.getLogger(__name__)

# Qwen3ForcedAligner 限制：单段最长 5 分钟
_MAX_AUDIO_SECONDS = 280  # ~4.7 分钟，留余量
_TARGET_SR = 16000        # Qwen3 要求 16kHz


@register_aligner
class Qwen3Aligner(BaseAligner):
    """Qwen3 Forced Aligner 对齐器

    对齐策略:
      1. 合并全部 segment 文本，按字符数比例切分为 < 5 分钟的块
      2. 每块提取音频 → 重采样到 16kHz → 写临时 WAV → align()
      3. 对齐结果直接作为新 segment（带词级时间戳）
    """

    name = "qwen3_align"
    display_name = "Qwen3 Forced Aligner"

    def load_model(self) -> None:
        from qwen_asr import Qwen3ForcedAligner
        import torch

        model_path = self.model_path or "Qwen/Qwen3-ForcedAligner-0.6B"
        device = self.config.get("device", "auto")
        if device == "auto":
            device = "cuda:0" if torch.cuda.is_available() else "cpu"

        self._model = Qwen3ForcedAligner.from_pretrained(
            model_path, dtype=torch.bfloat16, device_map=device,
        )

    def align(
        self,
        audio: str | np.ndarray,
        result: TranscriptionResult,
        language: str | None = None,
    ) -> TranscriptionResult:
        self._ensure_loaded()

        lang = language or result.language
        if lang == "auto":
            lang = "Chinese"

        total_duration = self._get_audio_duration(audio)

        valid_segs = [s for s in result.segments if s.text.strip()]
        if not valid_segs:
            return result

        total_chars = sum(len(s.text) for s in valid_segs)

        # 短文本直接对齐
        if total_chars < 100:
            return self._align_single(audio, result, lang)

        chunk_plans = self._plan_chunks(valid_segs, total_chars, total_duration)
        logger.info(
            "对齐: %d segments, %d 字符, %.1fs → %d 个块",
            len(valid_segs), total_chars, total_duration, len(chunk_plans),
        )

        aligned_segments = []
        for i, plan in enumerate(chunk_plans):
            try:
                chunk_text = plan["text"]
                logger.info("  块 %d/%d: %.1f-%.1fs, %d 字符",
                            i + 1, len(chunk_plans),
                            plan["start_sec"], plan["end_sec"], len(chunk_text))

                # 提取音频块 → 重采样到 16kHz → 写临时 WAV
                chunk_audio = self._read_audio_chunk(
                    audio, plan["start_sec"], plan["end_sec"], total_duration,
                )
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    tmp_path = f.name
                self._write_wav(tmp_path, chunk_audio, _TARGET_SR)

                try:
                    align_results = self._model.align(
                        audio=tmp_path,
                        text=chunk_text,
                        language=lang,
                    )
                    seg = self._tokens_to_segment(align_results, plan["start_sec"])
                    if seg:
                        aligned_segments.append(seg)
                    else:
                        aligned_segments.append(
                            Segment(text=chunk_text,
                                    start=plan["start_sec"], end=plan["end_sec"])
                        )
                finally:
                    Path(tmp_path).unlink(missing_ok=True)

            except KeyboardInterrupt:
                logger.warning("用户中断")
                raise
            except Exception as e:
                logger.warning("块 %d 对齐失败 (%s)", i + 1, e)
                aligned_segments.append(
                    Segment(text=plan["text"],
                            start=plan["start_sec"], end=plan["end_sec"])
                )

        full_text = " ".join(s.text for s in aligned_segments if s.text)
        return TranscriptionResult(
            text=full_text.strip(),
            segments=aligned_segments,
            language=lang,
            duration=total_duration,
            model=result.model,
            has_timestamps=any(
                s.words and len(s.words) > 0 and s.words[0].end > 0
                for s in aligned_segments
            ),
        )

    # ── 辅助方法 ──

    def _align_single(self, audio, result, lang):
        text = result.text.strip()
        if not text:
            return result
        try:
            align_results = self._model.align(audio=audio, text=text, language=lang)
        except Exception:
            return result
        seg = self._tokens_to_segment(align_results, 0.0)
        if seg:
            result.segments = [seg]
            result.has_timestamps = True
        return result

    def _tokens_to_segment(self, align_results, time_offset):
        if not align_results or not align_results[0]:
            logger.warning("_tokens_to_segment: align_results 为空")
            return None
        tokens = list(align_results[0])
        if not tokens:
            logger.warning("_tokens_to_segment: tokens 列表为空")
            return None
        words = [
            Word(text=t.text,
                 start=round(float(t.start_time) + time_offset, 3),
                 end=round(float(t.end_time) + time_offset, 3))
            for t in tokens
        ]
        text = "".join(w.text for w in words)  # CJK 不用空格拼接
        return Segment(text=text, start=words[0].start, end=words[-1].end, words=words)

    def _plan_chunks(self, segments, total_chars, total_duration):
        chars_per_sec = total_chars / max(total_duration, 1)
        plans = []
        current_segs, current_chars = [], 0

        for seg in segments:
            seg_chars = len(seg.text)
            if current_chars > 0 and (current_chars + seg_chars) / max(chars_per_sec, 1) > _MAX_AUDIO_SECONDS:
                prev_end = plans[-1]["end_sec"] if plans else 0.0
                plans.append({
                    "text": " ".join(s.text.strip() for s in current_segs),
                    "start_sec": prev_end,
                    "end_sec": prev_end + current_chars / chars_per_sec,
                })
                current_segs, current_chars = [], 0
            current_segs.append(seg)
            current_chars += seg_chars

        if current_segs:
            prev_end = plans[-1]["end_sec"] if plans else 0.0
            plans.append({
                "text": " ".join(s.text.strip() for s in current_segs),
                "start_sec": prev_end,
                "end_sec": total_duration,
            })
        if plans:
            plans[0]["start_sec"] = 0.0
        return plans

    def _read_audio_chunk(self, audio, start_sec, end_sec, total_duration):
        """读取音频片段并重采样到 _TARGET_SR"""
        import soundfile as sf
        import torchaudio

        pad = 0.3
        start_sec = max(0, start_sec - pad)
        end_sec = min(total_duration, end_sec + pad)

        if isinstance(audio, np.ndarray):
            data = audio[int(start_sec * _TARGET_SR):int(end_sec * _TARGET_SR)]
            return data.astype(np.float32)

        sr = sf.info(audio).samplerate
        data, _ = sf.read(
            audio,
            start=int(start_sec * sr),
            frames=int((end_sec - start_sec) * sr),
            dtype="float32",
        )
        if data.ndim > 1:
            data = data.mean(axis=1)

        # 重采样到 16kHz
        if sr != _TARGET_SR:
            import torch
            t = torch.from_numpy(data).unsqueeze(0)
            resampler = torchaudio.transforms.Resample(sr, _TARGET_SR)
            data = resampler(t).squeeze(0).numpy()

        return data.astype(np.float32)

    def _write_wav(self, path, audio, sr):
        """写 WAV 文件（Qwen3 最稳定的输入方式）"""
        import soundfile as sf
        sf.write(path, audio, sr, subtype="PCM_16")

    def _get_audio_duration(self, audio):
        if isinstance(audio, str):
            import soundfile as sf
            return sf.info(audio).duration
        return len(audio) / _TARGET_SR
