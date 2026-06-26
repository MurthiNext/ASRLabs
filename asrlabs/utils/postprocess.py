"""标点归一化后处理——根据语言统一全角/半角标点"""

import re

# CJK 语言列表
_CJK_LANGS = frozenset({
    "zh", "ja", "ko", "chinese", "japanese", "korean",
    "cantonese", "yue", "cmn", "wuu", "hak",
})

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
    """根据语言归一化标点符号

    Args:
        text: 待处理文本
        language: 语言代码 (zh/ja/ko/en/fr 等)

    Returns:
        标点归一化后的文本
    """
    lang_lower = (language or "").lower()
    is_cjk = lang_lower in _CJK_LANGS

    if is_cjk:
        text = text.translate(_HALF_TO_FULL)
        # 修复常见的标点组合
        text = _fix_cjk_sequences(text)
    else:
        text = text.translate(_FULL_TO_HALF)
        text = _fix_western_sequences(text)

    return text


def _fix_cjk_sequences(text: str) -> str:
    """修复 CJK 标点序列的常见问题"""
    # 重复标点压缩（！->！, ？？->？）
    text = re.sub(r"！{2,}", "！", text)
    text = re.sub(r"？{2,}", "？", text)
    text = re.sub(r"。{2,}", "……", text)  # 多个句号->省略号
    # 连续标点去重
    text = re.sub(r"、、+", "、", text)
    # 空格->顿号不是合理变换，跳过
    # 引号匹配
    return text


def _fix_western_sequences(text: str) -> str:
    """修复 Western 标点序列的常见问题"""
    # 重复标点压缩
    text = re.sub(r"!{3,}", "!", text)
    text = re.sub(r"\?{3,}", "?", text)
    text = re.sub(r"\.{4,}", "...", text)
    # 句点后空格修复 (去掉多余空格)
    text = re.sub(r"\.\s{2,}", ". ", text)
    # 逗号后空格确保
    text = re.sub(r",(\S)", r", \1", text)
    return text
