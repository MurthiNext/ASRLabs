"""Qwen3 Forced Aligner 对齐器——通过 qwen-asr 的 Qwen3ForcedAligner 实现"""

import logging
import numpy as np
from asrlabs.models import TranscriptionResult, Segment, Word
from asrlabs.align.base import BaseAligner
from asrlabs.align.base import register_aligner

logger = logging.getLogger(__name__)

# Qwen3ForcedAligner 限制：单段最长 5 分钟
_MAX_AUDIO_SECONDS = 280  # ~4.7 分钟，留余量


@register_aligner
class Qwen3Aligner(BaseAligner):
    """Qwen3 Forced Aligner 对齐器

    独立对齐器，适用于任何听写模型产出的文本。
    支持 11 种语言，单段最长 5 分钟，需要 GPU。

    对齐策略:
      1. 合并全部 segment 文本
      2. 按字符数比例估算时间位置，切分音频为 < 5 分钟的块
      3. 每块合并文本 + 提取对应音频片段 → 一次 align() 调用
      4. 对齐结果直接作为新的 segment（带有词级时间戳）
    """

    name = "qwen3_align"
    display_name = "Qwen3 Forced Aligner"

    def load_model(self) -> None:
        """加载 Qwen3 Forced Aligner 模型"""
        from qwen_asr import Qwen3ForcedAligner
        import torch

        model_path = self.model_path or "Qwen/Qwen3-ForcedAligner-0.6B"
        device = self.config.get("device", "auto")
        if device == "auto":
            device = "cuda:0" if torch.cuda.is_available() else "cpu"

        self._model = Qwen3ForcedAligner.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            device_map=device,
        )

    def align(
        self,
        audio: str | np.ndarray,
        result: TranscriptionResult,
        language: str | None = None,
    ) -> TranscriptionResult:
        """对齐音频和文本"""
        self._ensure_loaded()

        lang = language or result.language
        if lang == "auto":
            lang = "Chinese"

        total_duration = self._get_audio_duration(audio)
        sr = self._get_audio_sr(audio)

        # 获取有效 segment 的完整文本
        valid_segs = [s for s in result.segments if s.text.strip()]
        if not valid_segs:
            return result

        total_chars = sum(len(s.text) for s in valid_segs)

        # 短文本 -> 直接对齐
        if total_chars < 100:
            return self._align_single(audio, result, lang)

        # 按字符数比例切分为 < _MAX_AUDIO_SECONDS 的块
        chunk_plans = self._plan_chunks(valid_segs, total_chars, total_duration)

        logger.info(
            "对齐: %d segments, %d 字符, %.1fs → %d 个块",
            len(valid_segs), total_chars, total_duration, len(chunk_plans),
        )

        aligned_segments = []
        for i, plan in enumerate(chunk_plans):
            try:
                chunk_text = plan["text"]
                chunk_start = plan["start_sec"]
                chunk_end = plan["end_sec"]

                logger.info("  块 %d/%d: %.1f-%.1fs, %d 字符",
                            i + 1, len(chunk_plans), chunk_start, chunk_end, len(chunk_text))

                chunk_audio = self._extract_audio_chunk(
                    audio, chunk_start, chunk_end, total_duration, sr,
                )

                align_results = self._model.align(
                    audio=(chunk_audio, sr),
                    text=chunk_text,
                    language=lang,
                )

                # 对齐结果直接作为新 segment（带词级时间戳）
                seg = self._tokens_to_segment(align_results, chunk_start)
                if seg:
                    aligned_segments.append(seg)
                else:
                    # 对齐失败 -> 保留原始文本（无时间戳）
                    aligned_segments.append(
                        Segment(text=chunk_text, start=chunk_start, end=chunk_end)
                    )

            except KeyboardInterrupt:
                logger.warning("用户中断")
                raise
            except Exception as e:
                logger.warning("块 %d 对齐失败 (%s)", i + 1, e)
                aligned_segments.append(
                    Segment(text=plan["text"], start=plan["start_sec"], end=plan["end_sec"])
                )

        full_text = " ".join(s.text for s in aligned_segments if s.text)
        return TranscriptionResult(
            text=full_text.strip(),
            segments=aligned_segments,
            language=lang,
            duration=total_duration,
            model=result.model,
            has_timestamps=any(
                s.words and len(s.words) > 0 and s.start > 0 for s in aligned_segments
            ),
        )

    # ── 辅助方法 ──

    def _align_single(
        self, audio, result: TranscriptionResult, lang: str
    ) -> TranscriptionResult:
        """短文本直接对齐"""
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

    def _tokens_to_segment(self, align_results, time_offset: float) -> Segment | None:
        """将 Qwen3ForcedAligner 返回的 token 列表转为 Segment"""
        if not align_results or not align_results[0]:
            return None
        tokens = list(align_results[0])
        if not tokens:
            return None
        words = [
            Word(text=t.text, start=t.start_time + time_offset, end=t.end_time + time_offset)
            for t in tokens
        ]
        text = " ".join(w.text for w in words)
        return Segment(text=text, start=words[0].start, end=words[-1].end, words=words)

    def _plan_chunks(
        self,
        segments: list[Segment],
        total_chars: int,
        total_duration: float,
    ) -> list[dict]:
        """按字符数比例规划音频块

        Returns:
            [{text, start_sec, end_sec}, ...]
        """
        chars_per_sec = total_chars / max(total_duration, 1)
        plans = []
        current_segs = []
        current_chars = 0

        for seg in segments:
            seg_chars = len(seg.text)
            seg_est_dur = seg_chars / max(chars_per_sec, 1)

            if current_chars > 0 and (current_chars + seg_chars) / chars_per_sec > _MAX_AUDIO_SECONDS:
                # 当前块已满，结算
                plans.append({
                    "text": " ".join(s.text.strip() for s in current_segs),
                    "start_sec": 0 if not plans else plans[-1]["end_sec"],
                    "end_sec": 0 if not plans else plans[-1]["end_sec"] + current_chars / chars_per_sec,
                })
                current_segs = []
                current_chars = 0

            current_segs.append(seg)
            current_chars += seg_chars

        if current_segs:
            prev_end = plans[-1]["end_sec"] if plans else 0.0
            plans.append({
                "text": " ".join(s.text.strip() for s in current_segs),
                "start_sec": prev_end,
                "end_sec": total_duration,
            })

        # 修正第一块的 start_sec
        if plans:
            plans[0]["start_sec"] = 0.0

        return plans

    def _get_audio_duration(self, audio: str | np.ndarray) -> float:
        if isinstance(audio, str):
            import soundfile as sf
            return sf.info(audio).duration
        return len(audio) / 16000

    def _get_audio_sr(self, audio: str | np.ndarray) -> int:
        if isinstance(audio, str):
            import soundfile as sf
            return sf.info(audio).samplerate
        return 16000

    def _extract_audio_chunk(
        self,
        audio: str | np.ndarray,
        start_sec: float,
        end_sec: float,
        total_duration: float,
        sr: int,
    ) -> np.ndarray:
        """提取音频片段，加少量 padding"""
        pad = 0.3
        start_sec = max(0, start_sec - pad)
        end_sec = min(total_duration, end_sec + pad)

        if isinstance(audio, np.ndarray):
            return audio[int(start_sec * sr):int(end_sec * sr)].astype(np.float32)

        import soundfile as sf
        data, _ = sf.read(
            audio,
            start=int(start_sec * sr),
            frames=int((end_sec - start_sec) * sr),
            dtype="float32",
        )
        if data.ndim > 1:
            data = data.mean(axis=1)
        return data.astype(np.float32)
