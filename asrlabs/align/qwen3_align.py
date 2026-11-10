"""Qwen3 Forced Aligner 对齐器——通过 qwen-asr 的 Qwen3ForcedAligner 实现"""

import logging
import unicodedata
from pathlib import Path
import numpy as np
from asrlabs.models import TranscriptionResult, Segment, Word
from asrlabs.align.base import BaseAligner
from asrlabs.align.base import register_aligner
from asrlabs.utils.audio import temp_wav_from_array
from asrlabs.utils.language import ISO6391_TO_ENGLISH

logger = logging.getLogger(__name__)

# Qwen3ForcedAligner 限制：单段最长 5 分钟
_MAX_AUDIO_SECONDS = 280  # ~4.7 分钟，留余量
_TARGET_SR = 16000        # Qwen3 要求 16kHz


@register_aligner
class Qwen3Aligner(BaseAligner):
    """Qwen3 Forced Aligner 对齐器

    对齐策略:
      1. 合并全部 segment 文本，按字符数比例切分为 < 5 分钟的块
      2. 每块提取音频 -> 重采样到 16kHz -> 写临时 WAV -> align()
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
            lang = "zh"
        qwen_lang = ISO6391_TO_ENGLISH.get(lang, "English")

        total_duration = self._get_audio_duration(audio)

        valid_segs = [s for s in result.segments if s.text.strip()]
        if not valid_segs:
            return result

        total_chars = sum(len(s.text) for s in valid_segs)

        # 短文本直接对齐
        if total_chars < 100:
            return self._align_single(audio, result, qwen_lang)

        chunk_plans = self._plan_chunks(valid_segs, total_chars, total_duration)
        logger.info(
            "对齐: %d segments, %d 字符, %.1fs -> %d 个块",
            len(valid_segs), total_chars, total_duration, len(chunk_plans),
        )

        aligned_segments = []
        for i, plan in enumerate(chunk_plans):
            try:
                chunk_text = plan["text"]
                logger.info("  块 %d/%d: %.1f-%.1fs, %d 字符",
                            i + 1, len(chunk_plans),
                            plan["start_sec"], plan["end_sec"], len(chunk_text))

                # 提取音频块 -> 重采样到 16kHz -> 写临时 WAV
                chunk_audio = self._read_audio_chunk(
                    audio, plan["start_sec"], plan["end_sec"], total_duration,
                )
                with temp_wav_from_array(chunk_audio, _TARGET_SR) as tmp_path:
                    align_results = self._model.align(
                        audio=tmp_path,
                        text=chunk_text,
                        language=qwen_lang,
                    )
                    seg = self._tokens_to_segment(
                        align_results, plan["start_sec"], original_text=chunk_text,
                    )
                    if seg:
                        aligned_segments.append(seg)
                    else:
                        aligned_segments.append(
                            Segment(text=chunk_text,
                                    start=plan["start_sec"], end=plan["end_sec"])
                        )

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
        seg = self._tokens_to_segment(align_results, 0.0, original_text=text)
        if seg:
            result.segments = [seg]
            result.has_timestamps = True
        return result

    def _tokens_to_segment(self, align_results, time_offset, original_text=""):
        if not align_results or not align_results[0]:
            logger.warning("_tokens_to_segment: align_results 为空")
            return None
        tokens = list(align_results[0])
        if not tokens:
            logger.warning("_tokens_to_segment: tokens 列表为空")
            return None

        # 优先尝试标点回填；失败则降级为原始拼接逻辑
        if original_text:
            words = self._reinsert_puncts(tokens, time_offset, original_text)
            if words is not None:
                text = "".join(w.text for w in words)  # CJK 不用空格拼接
                return Segment(text=text, start=words[0].start,
                               end=words[-1].end, words=words)

        words = [
            Word(text=t.text,
                 start=round(float(t.start_time) + time_offset, 3),
                 end=round(float(t.end_time) + time_offset, 3))
            for t in tokens
        ]
        text = "".join(w.text for w in words)  # CJK 不用空格拼接
        return Segment(text=text, start=words[0].start, end=words[-1].end, words=words)

    # 非 P* 类别但需视为标点的符号 (原硬编码集合中的 Sm/So 类字符)
    _EXTRA_PUNCT = frozenset("～♪◇○●◎■□★☆")

    @staticmethod
    def _is_punct(ch: str) -> bool:
        """判断字符是否为标点——基于 Unicode 类别 P* (覆盖所有语言)。
        """
        if not ch:
            return False
        return unicodedata.category(ch).startswith("P") or ch in Qwen3Aligner._EXTRA_PUNCT

    def _reinsert_puncts(self, tokens, time_offset, original_text):
        """将原始文本中的标点回填到 Qwen 对齐结果中。

        核心算法:
          1. 把 Qwen tokens 展开成单字符流 (仅含非标点非空白字符),
             每字符继承所属 token 的时间戳
          2. 逐字符扫描原始文本: 非标点字符从字符流取下一个并累积成 word;
             遇到标点则输出累积的 word, 再输出标点作为独立 Word
          3. 保持 Qwen 原始 token 边界 (token 索引切换时输出当前累积的 word)
          4. 标点落在 token 内部时自动切开该 token
          5. 标点 Word 时间戳取前一个 word 的 end (与 CTC 行为一致)

        降级条件: 原文去标点去空白后的纯字符序列与 token 纯字符序列不一致时
                  返回 None, 调用方降级为原始拼接逻辑 (不做标点回填)。
                  降级校验对 original 和 token 做对称过滤, 保证多语言下可靠。
        """
        # 构建字符流: [(char, start, end, token_idx)] —— 仅含非标点非空白字符
        char_stream = []
        for tidx, t in enumerate(tokens):
            for ch in t.text:
                if ch.isspace() or self._is_punct(ch):
                    continue  # 跳过 token 内的空白/标点, 保持与原文过滤一致
                char_stream.append((
                    ch,
                    round(float(t.start_time) + time_offset, 3),
                    round(float(t.end_time) + time_offset, 3),
                    tidx,
                ))

        # 降级校验: 原文与 token 去标点去空白后的纯字符序列必须一致 (对称过滤)
        original_pure = "".join(
            ch for ch in original_text
            if not ch.isspace() and not self._is_punct(ch)
        )
        token_pure = "".join(
            ch for t in tokens for ch in t.text
            if not ch.isspace() and not self._is_punct(ch)
        )
        if original_pure != token_pure:
            logger.debug(
                "标点回填降级: 原文纯字符(%d)与token纯字符(%d)不一致",
                len(original_pure), len(token_pure),
            )
            return None

        result_words: list[Word] = []
        cur_chars: list[str] = []
        cur_start: float | None = None
        cur_end: float | None = None
        cur_tidx: int = -1
        ci = 0  # char_stream 指针

        def _flush():
            nonlocal cur_chars, cur_start, cur_end, cur_tidx
            if cur_chars:
                result_words.append(Word(
                    text="".join(cur_chars),
                    start=cur_start if cur_start is not None else 0.0,
                    end=cur_end if cur_end is not None else 0.0,
                ))
            cur_chars = []
            cur_start = None
            cur_end = None
            cur_tidx = -1

        for ch in original_text:
            if ch.isspace():
                continue  # 跳过空白，不切割当前 word
            if self._is_punct(ch):
                prev_end = cur_end
                _flush()
                # 标点时间戳: 优先取前一个 word 的 end，否则取下一个字符的 start
                punct_time = prev_end
                if punct_time is None and ci < len(char_stream):
                    punct_time = char_stream[ci][1]
                if punct_time is None:
                    punct_time = 0.0
                result_words.append(Word(text=ch, start=punct_time, end=punct_time))
            else:
                if ci < len(char_stream):
                    _, ts, te, tidx = char_stream[ci]
                    ci += 1
                    # token 边界切换: 输出当前累积的 word，保持 Qwen 原始分词边界
                    if cur_tidx != -1 and tidx != cur_tidx:
                        _flush()
                    if cur_start is None:
                        cur_start = ts
                    cur_end = te
                    cur_tidx = tidx
                    cur_chars.append(ch)
                else:
                    # 字符流耗尽 (不应发生，降级保护)
                    if cur_start is None:
                        cur_start = 0.0
                    cur_chars.append(ch)

        _flush()

        if not result_words:
            return None
        return result_words

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

    def _get_audio_duration(self, audio):
        if isinstance(audio, str):
            import soundfile as sf
            return sf.info(audio).duration
        return len(audio) / _TARGET_SR
