"""标点归一化后处理——根据语言统一全角/半角标点"""

import re
from asrlabs.models import TranscriptionResult
from asrlabs.utils.language import is_cjk

# ── 标点映射表 ──

# 半角 -> 全角 (CJK)
_HALF_TO_FULL = str.maketrans({
    ",": "，", ".": "。", "!": "！", "?": "？",
    ":": "：", ";": "；", "-": "—", "~": "〜",
    "(": "（", ")": "）",
})

# 全角 -> 半角 (Western)
_FULL_TO_HALF = str.maketrans({
    "，": ",", "。": ".", "！": "!", "？": "?",
    "：": ":", "；": ";", "—": "-", "〜": "~",
    "（": "(", "）": ")",
})


def normalize_punctuation(text: str, language: str) -> str:
    """根据语言归一化标点符号"""
    if is_cjk(language):
        text = text.translate(_HALF_TO_FULL)
        # 修复常见的标点组合
        text = _fix_cjk_sequences(text)
    else:
        text = text.translate(_FULL_TO_HALF)
        text = _fix_western_sequences(text)

    return text


def _fix_cjk_sequences(text: str) -> str:
    """修复 CJK 标点序列的常见问题"""
    # 重复标点压缩（！！！->！, ？？？->？, 4个及以上才压缩）
    text = re.sub(r"！{4,}", "！", text)
    text = re.sub(r"？{4,}", "？", text)
    text = re.sub(r"。{2,}", "……", text)  # 多个句号->省略号
    # 连续标点去重
    text = re.sub(r"、、+", "、", text)
    # 引号匹配
    return text


def _fix_western_sequences(text: str) -> str:
    """修复 Western 标点序列的常见问题"""
    # 重复标点压缩（4个及以上才压缩）
    text = re.sub(r"!{4,}", "!", text)
    text = re.sub(r"\?{4,}", "?", text)
    text = re.sub(r"\.{4,}", "...", text)
    # 句点后空格修复 (去掉多余空格)
    text = re.sub(r"\.\s{2,}", ". ", text)
    # 逗号后空格确保
    text = re.sub(r",(\S)", r", \1", text)
    return text


def normalize_result(result: TranscriptionResult, language: str) -> TranscriptionResult:
    """对 TranscriptionResult 全文执行标点归一化

    language 为 "auto" 时不做任何处理直接返回原结果。
    """
    if not language or language == "auto":
        return result
    result.text = normalize_punctuation(result.text, language)
    for seg in result.segments:
        seg.text = normalize_punctuation(seg.text, language)
    return result
