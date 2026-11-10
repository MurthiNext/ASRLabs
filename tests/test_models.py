"""核心数据模型测试"""

import json
import tempfile
from pathlib import Path
from asrlabs.models import Word, Segment, TranscriptionResult


class TestWord:
    def test_create_word_default_confidence(self):
        w = Word(text="hello", start=0.0, end=1.0)
        assert w.text == "hello"
        assert w.start == 0.0
        assert w.end == 1.0
        assert w.confidence == 1.0

    def test_create_word_with_confidence(self):
        w = Word(text="world", start=1.0, end=2.0, confidence=0.85)
        assert w.confidence == 0.85


class TestSegment:
    def test_create_segment_defaults(self):
        seg = Segment(text="hello world", start=0.0, end=2.0)
        assert seg.text == "hello world"
        assert seg.words == []
        assert seg.confidence == 0.0

    def test_create_segment_with_words(self):
        words = [
            Word("hello", 0.0, 1.0, 0.9),
            Word("world", 1.0, 2.0, 0.8),
        ]
        seg = Segment(text="hello world", start=0.0, end=2.0, words=words)
        assert len(seg.words) == 2


class TestTranscriptionResult:
    def _make_result(self, has_timestamps=True):
        words1 = [Word("hello", 0.0, 1.0, 0.9), Word("world", 1.0, 2.0, 0.8)]
        words2 = [Word("goodbye", 3.0, 4.0, 0.7)]
        segments = [
            Segment("hello world", 0.0, 2.0, words1),
            Segment("goodbye", 3.0, 4.0, words2),
        ]
        return TranscriptionResult(
            text="hello world goodbye",
            segments=segments,
            language="en",
            duration=4.0,
            model="whisper-base",
            has_timestamps=has_timestamps,
        )

    def test_to_txt(self):
        result = self._make_result()
        txt = result.to_txt()
        assert txt == "hello world goodbye"

    def test_to_srt(self):
        result = self._make_result()
        srt = result.to_srt()
        assert "1\n" in srt
        assert "00:00:00,000 --> 00:00:02,000" in srt
        assert "hello world" in srt
        assert "2\n" in srt
        assert "00:00:03,000 --> 00:00:04,000" in srt
        assert "goodbye" in srt

    def test_to_srt_without_timestamps_raises(self):
        result = self._make_result(has_timestamps=False)
        import pytest as pt
        with pt.raises(ValueError, match="时间戳"):
            result.to_srt()

    def test_to_json_roundtrip(self):
        result = self._make_result()
        json_str = result.to_json()
        data = json.loads(json_str)
        assert data["text"] == "hello world goodbye"
        assert data["language"] == "en"
        assert len(data["segments"]) == 2
        assert data["has_timestamps"] is True

    def test_save_txt(self):
        result = self._make_result()
        with tempfile.TemporaryDirectory() as tmp:
            out = result.save(Path(tmp) / "result.txt")
            assert out.suffix == ".txt"
            assert Path(out).read_text(encoding="utf-8") == "hello world goodbye"

    def test_save_auto_detect_format(self):
        result = self._make_result()
        with tempfile.TemporaryDirectory() as tmp:
            out = result.save(Path(tmp) / "result.srt")
            assert out.suffix == ".srt"

    def test_has_timestamps_false_without_word_data(self):
        result = TranscriptionResult(
            text="hello",
            segments=[Segment("hello", 0.0, 0.0)],
            has_timestamps=False,
        )
        assert result.has_timestamps is False
