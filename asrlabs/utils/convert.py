"""格式转换与分句工具"""

import logging
import unicodedata

from asrlabs.models import TranscriptionResult, Segment, Word
from asrlabs.utils.language import is_cjk as _is_cjk_lang

logger = logging.getLogger(__name__)

# 句末标点：遇此标点切分 segment，标点附在句尾
_SENTENCE_END_PUNCTS = frozenset("。！？!?…\n")
# 单个 segment 最大字符数兜底（超过则在 word 边界强制截断）
_DEFAULT_MAX_CHARS = 40


def _extract_primary_lang(lang: str | None) -> str:
    """提取 ISO 639-1 供 pysbd 使用"""
    if not lang or lang == "auto":
        return "en"
    return lang.lower()


def apply_split_logic(
    result: TranscriptionResult, logic: str, max_chars: int = _DEFAULT_MAX_CHARS
) -> TranscriptionResult:
    """按指定逻辑分句，返回新 TranscriptionResult"""
    if logic == "pysbd":
        result = _split_pysbd(result)
    elif logic == "punct":
        result = _split_punct(result)
    else:
        raise ValueError(f"未知分句逻辑: {logic}")

    # 兜底: 超长 segment 强制截断
    if max_chars > 0:
        result = _split_by_max_length(result, max_chars)

    return result


def _split_pysbd(result: TranscriptionResult) -> TranscriptionResult:
    """pysbd 语义分句，保留词级时间戳"""
    import pysbd

    lang_code = _extract_primary_lang(result.language)
    try:
        seg = pysbd.Segmenter(language=lang_code, clean=False)
    except ValueError:
        seg = pysbd.Segmenter(language="en", clean=False)

    new_segments: list[Segment] = []
    for segment in result.segments:
        text = segment.text.strip()
        if not text:
            continue

        char_to_word = _build_char_to_word_map(text, segment.words)
        sentences = seg.segment(text)

        search_from = 0
        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue
            start_char = text.find(sent, search_from)
            if start_char == -1:
                new_segments.append(Segment(text=sent, start=segment.start, end=segment.end))
                continue
            end_char = start_char + len(sent)
            search_from = end_char

            if segment.words and char_to_word:
                word_indices: set[int] = set()
                for ci in range(start_char, end_char):
                    if ci in char_to_word:
                        word_indices.add(char_to_word[ci])
                if word_indices:
                    widx_list = sorted(word_indices)
                    start_time = segment.words[widx_list[0]].start
                    end_time = segment.words[widx_list[-1]].end
                    sent_words = [segment.words[i] for i in widx_list]
                    new_segments.append(Segment(
                        text=sent, start=start_time, end=end_time,
                        words=sent_words,
                    ))
                    continue

            new_segments.append(Segment(text=sent, start=0.0, end=0.0))

    has_ts = any(s.end > 0 for s in new_segments)
    joined = _join_cjk(result.language, [s.text for s in new_segments])
    return TranscriptionResult(
        text=joined,
        segments=new_segments,
        language=result.language,
        duration=result.duration,
        model=result.model,
        has_timestamps=has_ts,
    )


def _split_punct(result: TranscriptionResult) -> TranscriptionResult:
    """遇句末标点切分，标点附在句尾（不独立成段）

    主要逻辑:
      1. 逐字符遍历 segment 文本
      2. 遇句末标点（。！？!?…）→ 标点追加到当前文本末尾，输出当前 segment
      3. 遇句内标点（、，,；;）→ 追加到当前文本，不切分
      4. 其他标点 → 追加到当前文本，不切分
      5. 通过 char_to_word 映射保留词级时间戳
    """
    new_segments: list[Segment] = []
    for segment in result.segments:
        text = segment.text
        if not text.strip():
            continue

        char_to_word = _build_char_to_word_map(text, segment.words)
        current_text = ""
        current_word_indices: set[int] = set()

        for ci, ch in enumerate(text):
            if _is_punct(ch):
                # 标点追加到当前文本末尾（不独立成段）
                current_text += ch
                if ci in char_to_word:
                    current_word_indices.add(char_to_word[ci])

                # 句末标点 → 切分
                if ch in _SENTENCE_END_PUNCTS:
                    if current_text.strip():
                        new_segments.append(_make_segment(
                            current_text.strip(), current_word_indices,
                            segment.words,
                        ))
                    current_text = ""
                    current_word_indices.clear()
            elif not ch.isspace():
                current_text += ch
                if ci in char_to_word:
                    current_word_indices.add(char_to_word[ci])
            else:
                # 空白追加但不记录到 word 映射
                current_text += ch

        # 处理尾部剩余
        if current_text.strip():
            new_segments.append(_make_segment(
                current_text.strip(), current_word_indices,
                segment.words,
            ))

    has_ts = any(s.end > 0 for s in new_segments)
    joined = _join_cjk(result.language, [s.text for s in new_segments])
    return TranscriptionResult(
        text=joined,
        segments=new_segments,
        language=result.language,
        duration=result.duration,
        model=result.model,
        has_timestamps=has_ts,
    )


def _is_punct(ch: str) -> bool:
    """判断是否为标点字符"""
    return bool(ch) and unicodedata.category(ch).startswith("P")


def _build_char_to_word_map(text: str, words: list[Word]) -> dict[int, int]:
    """构建字符位置 → word 索引的映射"""
    char_to_word: dict[int, int] = {}
    if not words:
        return char_to_word
    char_pos = 0
    for widx, word in enumerate(words):
        while char_pos < len(text) and text[char_pos] in (' ', '\t', '\n', '\r'):
            char_pos += 1
        wtext = word.text.strip()
        wlen = len(wtext)
        for offset in range(wlen):
            if char_pos + offset < len(text):
                char_to_word[char_pos + offset] = widx
        char_pos += wlen
    return char_to_word


def _make_segment(
    text: str, word_indices: set[int], words: list[Word]
) -> Segment:
    """根据 word_indices 创建有时间戳的 Segment"""
    if not word_indices or not words:
        return Segment(text=text, start=0.0, end=0.0)
    idxs = sorted(word_indices)
    return Segment(
        text=text,
        start=words[idxs[0]].start,
        end=words[idxs[-1]].end,
        words=[words[i] for i in idxs],
    )


def _join_cjk(lang: str | None, texts: list[str]) -> str:
    """CJK 语言不用空格拼接"""
    if _is_cjk_lang(lang):
        return "".join(texts)
    return " ".join(texts)


def _split_by_max_length(
    result: TranscriptionResult, max_chars: int = _DEFAULT_MAX_CHARS
) -> TranscriptionResult:
    """超长 segment 在 word 边界强制截断

    主要逻辑:
      1. 遍历所有 segment，长度 <= max_chars 的保留不变
      2. 超长的 segment 按 word 累积字符数，达到 max_chars 时切分
      3. CJK 语言在字符级切分（word 可能是单字），非 CJK 在 word 边界切分
      4. 切分后每个子 segment 继承对应的 words 和时间戳
    """
    cjk = _is_cjk_lang(result.language)
    new_segments: list[Segment] = []

    for seg in result.segments:
        text = seg.text.strip()
        if len(text) <= max_chars:
            new_segments.append(seg)
            continue

        if not seg.words:
            # 无 word 级时间戳：按字符数等分
            for i in range(0, len(text), max_chars):
                chunk = text[i:i + max_chars]
                new_segments.append(Segment(
                    text=chunk, start=seg.start, end=seg.end,
                ))
            continue

        # 有 word 级时间戳：按 word 累积切分
        cur_words: list[Word] = []
        cur_chars: int = 0

        for w in seg.words:
            cur_words.append(w)
            cur_chars += len(w.text)

            if cur_chars >= max_chars:
                new_segments.append(Segment(
                    text=_join_text(cur_words, cjk).strip(),
                    start=cur_words[0].start,
                    end=cur_words[-1].end,
                    words=cur_words,
                ))
                cur_words = []
                cur_chars = 0

        # 尾部剩余
        if cur_words:
            new_segments.append(Segment(
                text=_join_text(cur_words, cjk).strip(),
                start=cur_words[0].start,
                end=cur_words[-1].end,
                words=cur_words,
            ))

    return TranscriptionResult(
        text=result.text,
        segments=new_segments,
        language=result.language,
        duration=result.duration,
        model=result.model,
        has_timestamps=result.has_timestamps,
    )


def _join_text(words: list[Word], cjk: bool) -> str:
    """按语言习惯拼接 Word 文本"""
    if cjk:
        return "".join(w.text for w in words)
    return " ".join(w.text for w in words)
