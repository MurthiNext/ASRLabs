"""格式解析工具——JSON 反序列化、格式检测"""

import json
from pathlib import Path
from asrlabs.models import TranscriptionResult, Segment, Word


def parse_json_result(path: str | Path) -> TranscriptionResult:
    """从 JSON 文件反序列化 TranscriptionResult"""
    data = json.loads(Path(path).read_text(encoding="utf-8"))

    segments = []
    for seg_data in data.get("segments", []):
        words = []
        for w_data in seg_data.get("words", []):
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


def load_reference(path: str | Path) -> TranscriptionResult:
    """加载参考文件"""
    suffix = Path(path).suffix.lower()
    if suffix == ".json":
        return parse_json_result(path)
    elif suffix == ".txt":
        text = Path(path).read_text(encoding="utf-8").strip()
        return TranscriptionResult(
            text=text,
            segments=[Segment(text, 0.0, 0.0)],
            has_timestamps=False,
        )
    else:
        raise ValueError(
            f"不支持的文件格式: {suffix}，支持 .json / .txt"
        )
