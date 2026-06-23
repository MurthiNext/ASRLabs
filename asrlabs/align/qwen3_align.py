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

    对齐策略（避免逐句调用 GPU 导致卡死）:
      1. 将所有 segment 文本合并
      2. 按字符数比例估算每个 segment 在音频中的时间位置
      3. 将音频切分为 < 5 分钟的块
      4. 每块内的 segment 合并文本，一次对齐调用处理一个音频块
    """

    name = "qwen3_align"
    display_name = "Qwen3 Forced Aligner"

    def load_model(self) -> None:
        """加载 Qwen3 Forced Aligner 模型

        model_path 为空时默认使用 Qwen/Qwen3-ForcedAligner-0.6B，
        否则可以是 HuggingFace 模型 ID 或本地缓存路径。
        """
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

        # 获取音频总时长
        total_duration = self._get_audio_duration(audio)

        # 仅有一个 segment 或文本很短 → 直接对齐
        total_chars = sum(len(s.text) for s in result.segments if s.text.strip())
        if len(result.segments) <= 1 or total_chars < 100:
            return self._align_single(audio, result, lang)

        # 将音频 + 文本切分为 < _MAX_AUDIO_SECONDS 的块
        logger.info(
            "对齐: %d segments, 共 %d 字符, 音频 %.1fs → 切分为 %.0fs 块",
            len(result.segments), total_chars, total_duration, _MAX_AUDIO_SECONDS,
        )
        chunks = self._split_into_chunks(result.segments, total_chars, total_duration)

        if not chunks:
            return result

        # 获取音频采样率
        sr = self._get_audio_sr(audio)

        logger.info("切分为 %d 个块，逐块对齐...", len(chunks))
        aligned_segments = []
        char_offset = 0  # 累计字符偏移，用于估算时间位置
        for i, chunk_segs in enumerate(chunks):
            try:
                chunk_chars = sum(len(s.text) for s in chunk_segs)
                # 按字符占比估算该块的起止时间
                chunk_start = (char_offset / max(total_chars, 1)) * total_duration
                chunk_end = ((char_offset + chunk_chars) / max(total_chars, 1)) * total_duration
                char_offset += chunk_chars

                chunk_audio = self._extract_audio_chunk(audio, chunk_start, chunk_end,
                                                        total_duration, sr)

                # 合并文本
                chunk_text = " ".join(s.text.strip() for s in chunk_segs if s.text.strip())
                if not chunk_text:
                    aligned_segments.extend(chunk_segs)
                    continue

                logger.info("  块 %d/%d: %.1f-%.1fs, %d 字符",
                            i + 1, len(chunks), chunk_start, chunk_end, len(chunk_text))

                # Qwen3ForcedAligner 接受 (numpy_array, sample_rate) 元组
                align_results = self._model.align(
                    audio=(chunk_audio, sr),
                    text=chunk_text,
                    language=lang,
                )

                # 将词级时间戳映射回 segment
                if align_results and align_results[0]:
                    tokens = list(align_results[0])
                    aligned_segs = self._distribute_tokens_to_segments(
                        chunk_segs, tokens, chunk_start,
                    )
                    aligned_segments.extend(aligned_segs)
                else:
                    aligned_segments.extend(chunk_segs)

            except KeyboardInterrupt:
                logger.warning("用户中断，返回已对齐的部分结果")
                raise
            except Exception as e:
                logger.warning("块 %d 对齐失败 (%s)，保留原文本", i + 1, e)
                aligned_segments.extend(chunk_segs)

        full_text = " ".join(s.text for s in aligned_segments if s.text)
        return TranscriptionResult(
            text=full_text.strip(),
            segments=aligned_segments,
            language=lang,
            duration=result.duration or total_duration,
            model=result.model,
            has_timestamps=any(
                s.words and s.words[0].start > 0 for s in aligned_segments if s.words
            ),
        )

    def _align_single(
        self, audio, result: TranscriptionResult, lang: str
    ) -> TranscriptionResult:
        """单段对齐（文本短、segment 少的简单情况）"""
        text = result.text.strip()
        if not text:
            return result
        try:
            align_results = self._model.align(audio=audio, text=text, language=lang)
        except Exception:
            return result

        words = []
        if align_results and align_results[0]:
            for token in align_results[0]:
                words.append(Word(
                    text=token.text, start=token.start_time, end=token.end_time,
                ))
        if words:
            result.segments = [
                Segment(text=text, start=words[0].start, end=words[-1].end, words=words)
            ]
            result.has_timestamps = True
        return result

    def _get_audio_duration(self, audio: str | np.ndarray) -> float:
        """获取音频时长（秒）"""
        if isinstance(audio, str):
            import soundfile as sf
            info = sf.info(audio)
            return info.duration
        else:
            return len(audio) / 16000  # 假设 16kHz

    def _split_into_chunks(
        self,
        segments: list[Segment],
        total_chars: int,
        total_duration: float,
    ) -> list[list[Segment]]:
        """按比例将 segments 分配到音频时间块中

        估算: 每个 segment 的时间 = 其字符数占比 × 总时长
        """
        chunks = []
        current_chunk: list[Segment] = []
        current_est_duration = 0.0
        chars_per_sec = total_chars / max(total_duration, 1)

        for seg in segments:
            if not seg.text.strip():
                current_chunk.append(seg)
                continue

            seg_chars = len(seg.text)
            seg_est_dur = seg_chars / max(chars_per_sec, 1)

            if current_est_duration + seg_est_dur > _MAX_AUDIO_SECONDS and current_chunk:
                chunks.append(current_chunk)
                current_chunk = []
                current_est_duration = 0.0

            current_chunk.append(seg)
            current_est_duration += seg_est_dur

        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    def _get_audio_sr(self, audio: str | np.ndarray) -> int:
        """获取音频采样率"""
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
        """从音频中提取 [start_sec, end_sec] 片段"""
        # 加少量 padding 避免边界截断
        pad = 0.3
        start_sec = max(0, start_sec - pad)
        end_sec = min(total_duration, end_sec + pad)

        if isinstance(audio, np.ndarray):
            start_idx = int(start_sec * sr)
            end_idx = int(end_sec * sr)
            return audio[start_idx:end_idx].astype(np.float32)

        import soundfile as sf
        start_frame = int(start_sec * sr)
        num_frames = int((end_sec - start_sec) * sr)

        data, _ = sf.read(audio, start=start_frame, frames=num_frames, dtype="float32")
        if data.ndim > 1:
            data = data.mean(axis=1)
        return data.astype(np.float32)

    def _distribute_tokens_to_segments(
        self,
        segments: list[Segment],
        tokens: list,
        chunk_offset: float,
    ) -> list[Segment]:
        """将词级时间戳按字符位置分配回各个 segment

        tokens: Qwen3ForcedAligner 返回的 token 列表，每个有 text/start_time/end_time
        chunk_offset: 当前音频块的起始偏移（秒）
        """
        # 为每个 segment 找到对应的 tokens
        full_text = "".join(t.text for t in tokens)
        aligned = []
        char_pos = 0

        for seg in segments:
            seg_text = seg.text.strip()
            if not seg_text:
                aligned.append(seg)
                continue

            # 在 full_text 中查找 seg_text 的位置
            seg_words = []
            seg_tokens = self._find_span_tokens(tokens, seg_text, char_pos)
            if seg_tokens:
                for tok in seg_tokens:
                    seg_words.append(Word(
                        text=tok.text,
                        start=tok.start_time + chunk_offset,
                        end=tok.end_time + chunk_offset,
                    ))
                aligned.append(Segment(
                    text=seg_text,
                    start=seg_words[0].start,
                    end=seg_words[-1].end,
                    words=seg_words,
                ))
                char_pos = full_text.find(seg_text, char_pos) + len(seg_text)
            else:
                aligned.append(seg)

        return aligned

    def _find_span_tokens(self, tokens: list, target: str, start_pos: int) -> list:
        """在 token 列表中查找对应 target 文本的 token 子序列"""
        # 归一化: 去掉空格后匹配
        flat_text = "".join(t.text for t in tokens)
        target_compact = target.replace(" ", "")

        idx = flat_text.find(target_compact, start_pos)
        if idx < 0:
            # 模糊匹配：找最长的 token 子序列
            return []

        # 找到 idx 对应的 token 范围
        token_start = 0
        pos = 0
        for i, t in enumerate(tokens):
            if pos + len(t.text) > idx:
                token_start = i
                break
            pos += len(t.text)

        token_end = token_start
        pos = 0
        for i in range(token_start, len(tokens)):
            pos += len(tokens[i].text)
            if pos >= len(target_compact):
                token_end = i + 1
                break

        return tokens[token_start:token_end]
