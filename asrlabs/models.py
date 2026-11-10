"""核心数据模型——贯穿全流程的数据结构"""

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Word:
    """单个词及其时间戳"""
    text: str
    start: float
    end: float
    confidence: float = 1.0


@dataclass
class Segment:
    """一个语音片段"""
    text: str
    start: float
    end: float
    words: list[Word] = field(default_factory=list)
    confidence: float = 0.0


@dataclass
class TranscriptionResult:
    """一次完整听写的结果"""
    text: str
    segments: list[Segment] = field(default_factory=list)
    language: str = "auto"
    duration: float = 0.0
    model: str = ""
    has_timestamps: bool = False

    def _format_timestamp(self, seconds: float) -> str:
        """将秒数格式化为 SRT 时间戳 HH:MM:SS,mmm"""
        ms = int(round(seconds * 1000))
        h = ms // 3600000
        m = (ms % 3600000) // 60000
        s = (ms % 60000) // 1000
        millis = ms % 1000
        return f"{h:02d}:{m:02d}:{s:02d},{millis:03d}"

    def to_txt(self) -> str:
        """输出纯文本"""
        return self.text

    def to_srt(self, max_chars_per_line: int = 42) -> str:
        """输出 SRT 字幕格式"""
        if not self.has_timestamps:
            raise ValueError("无法生成 SRT：结果不含时间戳，请先运行对齐")
        lines = []
        for i, seg in enumerate(self.segments, 1):
            if not seg.text.strip():
                continue
            lines.append(str(i))
            start_ts = self._format_timestamp(seg.start)
            end_ts = self._format_timestamp(seg.end)
            lines.append(f"{start_ts} --> {end_ts}")
            text = seg.text.strip()
            if len(text) > max_chars_per_line:
                wrapped = _wrap_text(text, max_chars_per_line, self.language)
                text = "\n".join(wrapped[:2])
            lines.append(text)
            lines.append("")
        return "\n".join(lines)

    def to_json(self) -> str:
        """输出 JSON 格式（保留完整元数据）"""
        def _convert(obj):
            if isinstance(obj, (Word, Segment, TranscriptionResult)):
                result = {}
                for field_def in obj.__dataclass_fields__:
                    value = getattr(obj, field_def)
                    result[field_def] = _convert(value)
                return result
            elif isinstance(obj, list):
                return [_convert(item) for item in obj]
            return obj
        return json.dumps(_convert(self), ensure_ascii=False, indent=2)

    def to_lrc(self) -> str:
        """输出 LRC 歌词格式"""
        if not self.has_timestamps:
            raise ValueError("无法生成 LRC：结果不含时间戳，请先运行对齐")
        lines = []
        for seg in self.segments:
            if not seg.text.strip():
                continue
            start = seg.start
            mins = int(start // 60)
            secs = start - mins * 60
            lines.append(f"[{mins:02d}:{secs:05.2f}]{seg.text.strip()}")
        return "\n".join(lines)

    def save(self, path: str | Path, fmt: str | None = None) -> Path:
        """保存结果到文件"""
        path = Path(path)
        if fmt is None:
            fmt = path.suffix.lstrip(".")
        if fmt not in ("json", "srt", "txt", "lrc"):
            raise ValueError(f"不支持的输出格式: {fmt}，支持 json/srt/txt/lrc")
        if fmt == "json":
            content = self.to_json()
        elif fmt == "srt":
            content = self.to_srt()
        elif fmt == "lrc":
            content = self.to_lrc()
        else:
            content = self.to_txt()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path


def _wrap_text(text: str, max_chars: int, lang: str | None = None) -> list[str]:
    """将超长文本按最大字符数换行

    主要逻辑:
      1. CJK 语言（zh/ja/ko）：在任意字符边界截断
      2. 拉丁语系：在空格边界截断，避免截断单词
      3. 尽量均匀分配行数
    """
    from asrlabs.utils.language import is_cjk as _is_cjk_lang

    if len(text) <= max_chars:
        return [text]

    cjk = _is_cjk_lang(lang)

    if cjk:
        # CJK: 按字符数等分
        lines = []
        for i in range(0, len(text), max_chars):
            lines.append(text[i:i + max_chars])
        return lines

    # 拉丁语系: 按空格截断
    words_list = text.split(" ")
    lines: list[str] = []
    cur_line = ""
    for w in words_list:
        if cur_line:
            if len(cur_line) + 1 + len(w) <= max_chars:
                cur_line += " " + w
            else:
                lines.append(cur_line)
                cur_line = w
        else:
            cur_line = w
    if cur_line:
        lines.append(cur_line)
    return lines
