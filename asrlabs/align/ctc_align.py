"""CTC Forced Aligner 对齐器——基于原版 ctc-forced-aligner 底层 API

通过 Wav2Vec2/HuBERT/MMS CTC 模型实现文本-音频强制对齐。
- 拉丁语系: word 级时间戳
- CJK (zh/ja/ko): 自动 char 级 + uroman 罗马化
- 支持 1136+ 语言（MMS 模型）

依赖 (optional extra):
    pip install -e .[ctc]
    需 ffmpeg 在 PATH；需 C++ 编译环境（Windows: MSVC Build Tools）
"""

import logging
import tempfile
from pathlib import Path

import numpy as np

from asrlabs.models import TranscriptionResult, Segment, Word
from asrlabs.align.base import BaseAligner, register_aligner

logger = logging.getLogger(__name__)

# CTC 模型固定 16kHz 采样率
_TARGET_SR = 16000
# generate_emissions 默认参数: 30s 窗口 + 2s 上下文, 批大小 4
_WINDOW_SEC = 30
_CONTEXT_SEC = 2
_BATCH_SIZE = 4

# BCP47 → ISO 639-3 映射（ctc-forced-aligner 要求 ISO 639-3 语言代码）
_BCP47_TO_ISO6393: dict[str, str] = {
    "zh": "cmn", "zh-cn": "cmn", "zh-tw": "cmn", "zh-hans": "cmn", "zh-hant": "cmn",
    "en": "eng", "en-us": "eng", "en-gb": "eng",
    "ja": "jpn",
    "ko": "kor",
    "ar": "ara", "ru": "rus", "de": "deu", "fr": "fra",
    "es": "spa", "it": "ita", "pt": "por",
    "tr": "tur", "fa": "fas", "he": "heb",
    "uk": "ukr", "pl": "pol", "nl": "nld",
    "vi": "vie", "th": "tha", "id": "ind",
}

# CJK ISO 639-3 集合（用于决定 Word 拼接是否加空格）
_CJK_ISO3 = {"cmn", "jpn", "kor"}


@register_aligner
class CtcAligner(BaseAligner):
    """CTC Forced Aligner 对齐器

    对齐策略:
      1. 语言映射 BCP47 → ISO 639-3
      2. 音频准备: path 直接用, ndarray 写临时 wav
      3. generate_emissions 分块推理 → CTC emission 矩阵
      4. preprocess_text 规范化 + uroman 罗马化 (CJK 自动 char 级)
      5. get_alignments + get_spans → 强制对齐路径
      6. postprocess_results → word/char 级时间戳
      7. 按原始 segment 字符数比例分组为 Segment 列表
    """

    name = "ctc_align"
    display_name = "CTC Forced Aligner (MMS/Wav2Vec2)"

    def load_model(self) -> None:
        """加载 CTC 对齐模型与 tokenizer"""
        try:
            from ctc_forced_aligner import load_alignment_model
        except ImportError as e:
            raise ImportError(
                "ctc_align 需要 ctc extra, 请执行: pip install -e .[ctc]"
            ) from e
        import torch

        # 默认 MMS-300m-1130 (1136 语言, CC-BY-NC 4.0 非商用)
        # 商用请改用 WAV2VEC2_ASR_LARGE_960H (MIT) 等模型
        model_path = self.model_path or "MahmoudAshraf/mms-300m-1130-forced-aligner"
        device = self.config.get("device", "auto")
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if str(device).startswith("cuda") else torch.float32

        logger.info("加载 CTC 对齐模型: %s @ %s (%s)", model_path, device, dtype)
        self._model, self._tokenizer = load_alignment_model(
            device, model_path, dtype=dtype,
        )
        self._device = device
        self._dtype = dtype

    def align(
        self,
        audio: str | np.ndarray,
        result: TranscriptionResult,
        language: str | None = None,
    ) -> TranscriptionResult:
        """对齐音频与文本, 返回带 word/char 级时间戳的结果"""
        self._ensure_loaded()

        lang = language or result.language or "auto"
        iso3 = self._map_language(lang)
        text = result.text.strip()
        if not text:
            return result

        tmp_path: str | None = None
        try:
            audio_path, tmp_path = self._prepare_audio(audio)
            duration = self._get_duration(audio_path)
            aligned = self._run_alignment(audio_path, text, iso3)
        finally:
            if tmp_path:
                Path(tmp_path).unlink(missing_ok=True)

        if not aligned:
            return result

        segments = self._group_results(aligned, result, iso3)
        if not segments:
            return result

        cjk = iso3 in _CJK_ISO3
        full_text = "".join(s.text for s in segments) if cjk \
            else " ".join(s.text for s in segments)
        return TranscriptionResult(
            text=full_text.strip(),
            segments=segments,
            language=lang if lang != "auto" else iso3,
            duration=duration,
            model=result.model,
            has_timestamps=True,
        )

    # ── 辅助方法 ──

    def _map_language(self, lang: str) -> str:
        """BCP47 语言代码 → ISO 639-3

        Args:
            lang: BCP47 代码 (zh/en/ja ...) 或 "auto"

        Returns:
            ISO 639-3 代码 (cmn/eng/jpn ...), 未知时回退 eng
        """
        if not lang or lang == "auto":
            return "eng"
        key = lang.lower().strip()
        if key in _BCP47_TO_ISO6393:
            return _BCP47_TO_ISO6393[key]
        main = key.split("-")[0]
        if main in _BCP47_TO_ISO6393:
            return _BCP47_TO_ISO6393[main]
        logger.warning("未知语言代码 %s, 回退到 eng", lang)
        return "eng"

    def _prepare_audio(self, audio: str | np.ndarray) -> tuple[str, str | None]:
        """准备音频文件路径

        Args:
            audio: 文件路径或 numpy 数组 (假设已 16kHz)

        Returns:
            (audio_path, tmp_path), tmp_path 非 None 时调用方需清理
        """
        if isinstance(audio, str):
            return audio, None
        # ndarray → 临时 wav (CTC load_audio 走 ffmpeg, 需文件输入)
        import soundfile as sf
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp_path = f.name
        sf.write(tmp_path, np.asarray(audio, dtype=np.float32), _TARGET_SR, subtype="PCM_16")
        return tmp_path, tmp_path

    def _get_duration(self, audio_path: str) -> float:
        """获取音频时长 (秒)"""
        import soundfile as sf
        return float(sf.info(audio_path).duration)

    def _run_alignment(self, audio_path: str, text: str, iso3: str) -> list[dict]:
        """执行 CTC 强制对齐核心流程

        主要实现逻辑:
          1. load_audio: ffmpeg 解码 → 16kHz 单声道 tensor
          2. generate_emissions: 30s 窗口分块批量推理 → log_softmax emission
          3. preprocess_text: 规范化 + uroman 罗马化, CJK 自动切 char 级
          4. get_alignments: CTC forced_align (C++ 扩展) → 对齐路径
          5. get_spans: 计算每个 token 的帧区间
          6. postprocess_results: 帧区间 → 秒级时间戳

        Args:
            audio_path: 音频文件路径
            text: 待对齐文本
            iso3: ISO 639-3 语言代码

        Returns:
            [{start, end, text, score}, ...] 列表
        """
        from ctc_forced_aligner import (
            load_audio,
            generate_emissions,
            preprocess_text,
            get_alignments,
            get_spans,
            postprocess_results,
        )

        waveform = load_audio(audio_path, self._dtype, self._device)
        emissions, stride = generate_emissions(
            self._model,
            waveform,
            window_length=_WINDOW_SEC,
            context_length=_CONTEXT_SEC,
            batch_size=_BATCH_SIZE,
        )

        # romanize=True 启用 uroman; preprocess_text 对 chi/jpn 内部自动切 char 级
        tokens_starred, text_starred = preprocess_text(
            text,
            romanize=True,
            language=iso3,
            split_size="word",
            star_frequency="segment",
        )

        segments, scores, blank_id = get_alignments(emissions, tokens_starred, self._tokenizer)
        spans = get_spans(tokens_starred, segments, blank_id)
        results = postprocess_results(text_starred, spans, stride, scores, 0.0)
        return results

    def _group_results(
        self,
        results: list[dict],
        original: TranscriptionResult,
        iso3: str,
    ) -> list[Segment]:
        """将对齐结果分组为 Segment 列表

        策略:
          - 原始仅 0/1 段: 全部 Word 合并为单个 Segment
          - 原始多段: 按各段文本字符数比例分配 results, 保留段落结构

        Args:
            results: postprocess_results 输出
            original: 原始听写结果 (提供段落结构参考)
            iso3: ISO 639-3 语言代码 (决定 CJK 拼接方式)

        Returns:
            带时间戳与 words 的 Segment 列表
        """
        cjk = iso3 in _CJK_ISO3
        words = [
            Word(
                text=str(r["text"]),
                start=round(float(r["start"]), 3),
                end=round(float(r["end"]), 3),
                confidence=round(float(r.get("score", 1.0)), 4),
            )
            for r in results
        ]
        if not words:
            return []

        valid_segs = [s for s in original.segments if s.text.strip()]

        # 单段或无段: 合并为一个 Segment
        if len(valid_segs) <= 1:
            return [self._build_segment(words, cjk)]

        # 多段: 按字符数比例分配
        total_chars = sum(len(s.text) for s in valid_segs)
        if total_chars == 0:
            return [self._build_segment(words, cjk)]

        segments: list[Segment] = []
        idx = 0
        for seg in valid_segs:
            count = max(1, round(len(words) * len(seg.text) / total_chars))
            seg_words = words[idx: idx + count]
            idx += count
            if not seg_words:
                continue
            segments.append(self._build_segment(seg_words, cjk))

        # 处理尾部剩余 words (比例分配可能产生余数)
        if idx < len(words):
            tail = words[idx:]
            if segments:
                last = segments[-1]
                last.words.extend(tail)
                last.end = tail[-1].end
                last.text = (last.text + self._join_text(tail, cjk)).strip()
            else:
                segments.append(self._build_segment(tail, cjk))

        return segments

    @staticmethod
    def _join_text(words: list[Word], cjk: bool) -> str:
        """按语言习惯拼接 Word 文本"""
        if cjk:
            return "".join(w.text for w in words)
        return " ".join(w.text for w in words)

    def _build_segment(self, words: list[Word], cjk: bool) -> Segment:
        """从 Word 列表构造 Segment"""
        return Segment(
            text=self._join_text(words, cjk).strip(),
            start=words[0].start,
            end=words[-1].end,
            words=words,
        )
