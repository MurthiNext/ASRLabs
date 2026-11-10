"""语言代码工具——统一使用 ISO 639-1"""

# ── ISO 639-1 → 各后端的输入格式 ──

# ISO 639-1 → Qwen3 English（调用 Qwen3 API 时使用）
ISO6391_TO_ENGLISH: dict[str, str] = {
    "zh": "Chinese",      "yue": "Cantonese",
    "en": "English",
    "ja": "Japanese",
    "ko": "Korean",
    "ar": "Arabic",       "de": "German",       "fr": "French",
    "es": "Spanish",      "pt": "Portuguese",   "id": "Indonesian",
    "it": "Italian",      "ru": "Russian",      "th": "Thai",
    "vi": "Vietnamese",   "tr": "Turkish",      "hi": "Hindi",
    "ms": "Malay",        "nl": "Dutch",        "sv": "Swedish",
    "da": "Danish",       "fi": "Finnish",      "pl": "Polish",
    "cs": "Czech",        "tl": "Filipino",     "fil": "Filipino",
    "fa": "Persian",      "el": "Greek",        "ro": "Romanian",
    "hu": "Hungarian",    "mk": "Macedonian",
    "no": "Norwegian",    "he": "Hebrew",       "uk": "Ukrainian",
    "bg": "Bulgarian",    "hr": "Croatian",     "sk": "Slovak",
    "lt": "Lithuanian",   "lv": "Latvian",      "et": "Estonian",
    "sl": "Slovenian",    "sr": "Serbian",
}

# ISO 639-1 → ISO 639-3（调用 CTC 对齐器时使用）
ISO6391_TO_ISO6393: dict[str, str] = {
    "zh": "cmn",  "en": "eng",  "ja": "jpn",  "ko": "kor",
    "ar": "ara",  "ru": "rus",  "de": "deu",  "fr": "fra",
    "es": "spa",  "it": "ita",  "pt": "por",
    "tr": "tur",  "fa": "fas",  "he": "heb",
    "uk": "ukr",  "pl": "pol",  "nl": "nld",
    "vi": "vie",  "th": "tha",  "id": "ind",
    "sv": "swe",  "da": "dan",  "fi": "fin",
    "el": "ell",  "cs": "ces",  "ro": "ron",
    "hu": "hun",  "sk": "slk",  "bg": "bul",
    "hr": "hrv",  "lt": "lit",  "lv": "lav",
    "no": "nor",  "et": "est",  "sl": "slv",
    "sr": "srp",  "ms": "msa",  "hi": "hin",
    "yue": "yue",
}

# ── 后端输出 → ISO 639-1 反向映射 ──

ENGLISH_TO_ISO6391: dict[str, str] = {
    v.lower(): k for k, v in ISO6391_TO_ENGLISH.items()
}

ISO6393_TO_ISO6391: dict[str, str] = {
    v: k for k, v in ISO6391_TO_ISO6393.items()
}

# ── CJK ──

_CJK = frozenset({"zh", "ja", "ko", "yue"})


def is_cjk(lang: str | None) -> bool:
    """判断语言代码是否为 CJK"""
    if not lang or lang == "auto":
        return False
    return lang.lower() in _CJK


def is_valid_iso6391(code: str) -> bool:
    """校验是否为有效的 ISO 639-1 代码"""
    return code in ISO6391_TO_ENGLISH


def validate_iso6391(code: str) -> str:
    """校验并返回规范化的 ISO 639-1 代码

    Raises:
        ValueError: 无效的 ISO 639-1 代码
    """
    if not is_valid_iso6391(code):
        raise ValueError(
            f"无效的 ISO 639-1 语言代码: {code!r}，"
            f"支持的代码: {', '.join(sorted(ISO6391_TO_ENGLISH))}"
        )
    return code


def validate_language_option(code: str) -> str:
    """用于 Click 回调: 接受 'auto' 或有效 ISO 639-1"""
    if code == "auto":
        return code
    return validate_iso6391(code)
