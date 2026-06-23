"""格式解析工具——SRT/VTT 解析、JSON 反序列化、格式检测"""

import json
import re
from pathlib import Path
from asrlabs.models import TranscriptionResult, Segment


def parse_srt(path: str | Path) -> TranscriptionResult:
    """解析 SRT 字幕文件为 TranscriptionResult

    Args:
        path: SRT 文件路径

    Returns:
        TranscriptionResult（从 SRT 提取的文本 + 时间戳）
    """
    content = Path(path).read_text(encoding="utf-8")
    # SRT 时间戳正则: 00:00:00,000 --> 00:00:02,000
    timestamp_pattern = re.compile(
        r"(\d{2}):(\d{2}):(\d{2}),(\d{3})\s*-->\s*"
        r"(\d{2}):(\d{2}):(\d{2}),(\d{3})"
    )

    segments = []
    blocks = content.strip().split("\n\n")
    for block in blocks:
        lines = block.strip().split("\n")
        if len(lines) < 2:
            continue
        # 跳过序号行
        text_start = 1
        match = timestamp_pattern.search(lines[1] if len(lines) > 1 else "")
        if not match:
            # 尝试在其他行找时间戳
            for i, line in enumerate(lines):
                match = timestamp_pattern.search(line)
                if match:
                    text_start = i + 1
                    break
        if not match:
            continue

        start_sec = (
            int(match.group(1)) * 3600
            + int(match.group(2)) * 60
            + int(match.group(3))
            + int(match.group(4)) / 1000
        )
        end_sec = (
            int(match.group(5)) * 3600
            + int(match.group(6)) * 60
            + int(match.group(7))
            + int(match.group(8)) / 1000
        )
        text = " ".join(lines[text_start:]).strip()
        if text:
            segments.append(Segment(text=text, start=start_sec, end=end_sec))

    full_text = " ".join(s.text for s in segments)
    return TranscriptionResult(
        text=full_text,
        segments=segments,
        has_timestamps=len(segments) > 0,
    )


def parse_json_result(path: str | Path) -> TranscriptionResult:
    """从 JSON 文件反序列化 TranscriptionResult

    Args:
        path: JSON 文件路径

    Returns:
        TranscriptionResult 实例
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))

    segments = []
    for seg_data in data.get("segments", []):
        words = []
        for w_data in seg_data.get("words", []):
            from asrlabs.models import Word

            words.append(Word(
                text=w_data["text"],
                start=w_data["start"],
                end=w_data["end"],
                confidence=w_data.get("confidence", 1.0),
            ))
        segments.append(Segment(
            text=seg_data["text"],
            start=seg_data["start"],
            end=seg_data["end"],
            words=words,
            confidence=seg_data.get("confidence", 0.0),
        ))

    return TranscriptionResult(
        text=data.get("text", ""),
        segments=segments,
        language=data.get("language", "auto"),
        duration=data.get("duration", 0.0),
        model=data.get("model", ""),
        has_timestamps=data.get("has_timestamps", False),
    )


def detect_reference_format(path: str | Path) -> str:
    """检测参考文件的格式

    Args:
        path: 文件路径

    Returns:
        格式名: "json", "srt", "vtt", "txt"

    Raises:
        ValueError: 不支持的格式
    """
    suffix = Path(path).suffix.lower()
    format_map = {
        ".json": "json",
        ".srt": "srt",
        ".vtt": "vtt",
        ".txt": "txt",
    }
    if suffix not in format_map:
        raise ValueError(
            f"不支持的文件格式: {suffix}，支持的格式: {', '.join(format_map.keys())}"
        )
    return format_map[suffix]


def load_reference(path: str | Path) -> TranscriptionResult:
    """自动检测格式并加载参考文件

    Args:
        path: 参考文件路径（.json / .srt / .vtt / .txt）

    Returns:
        TranscriptionResult 实例
    """
    fmt = detect_reference_format(path)
    if fmt == "json":
        return parse_json_result(path)
    elif fmt == "srt":
        return parse_srt(path)
    elif fmt == "vtt":
        return parse_srt(path)  # VTT 与 SRT 结构相似，SRT 解析器兼容基本格式
    elif fmt == "txt":
        text = Path(path).read_text(encoding="utf-8").strip()
        return TranscriptionResult(
            text=text,
            segments=[Segment(text, 0.0, 0.0)],
            has_timestamps=False,
        )
