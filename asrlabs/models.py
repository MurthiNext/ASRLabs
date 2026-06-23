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

    def to_srt(self) -> str:
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
            lines.append(seg.text.strip())
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

    def save(self, path: str | Path, fmt: str | None = None) -> Path:
        """保存结果到文件

        Args:
            path: 输出路径，扩展名决定格式（.json/.srt/.txt）
            fmt: 强制格式，若为 None 则从扩展名推断
        """
        path = Path(path)
        if fmt is None:
            fmt = path.suffix.lstrip(".")
        if fmt not in ("json", "srt", "txt"):
            raise ValueError(f"不支持的输出格式: {fmt}，支持 json/srt/txt")
        if fmt == "json":
            content = self.to_json()
        elif fmt == "srt":
            content = self.to_srt()
        else:
            content = self.to_txt()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path
