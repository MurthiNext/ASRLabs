"""Kotoba Whisper 后端测试——使用 mock pipeline，不下载真实模型"""

import numpy as np
import pytest
from asrlabs.models import TranscriptionResult


class FakePipeline:
    """模拟 transformers pipeline 的可调用对象"""

    def __init__(self, output: dict):
        self.output = output
        self.last_call = None

    def __call__(self, audio, **kwargs):
        self.last_call = (audio, kwargs)
        return self.output


def make_transcriber(config=None):
    """构造一个跳过 load_model 的 KotobaWhisperTranscriber"""
    from asrlabs.transcribe.kotoba import KotobaWhisperTranscriber
    t = KotobaWhisperTranscriber(config or {})
    # 跳过真实模型加载
    t._loaded = True
    return t


def test_registered():
    """后端应通过装饰器注册到 TRANSCRIBER_REGISTRY"""
    from asrlabs.transcribe import TRANSCRIBER_REGISTRY
    assert "kotoba-whisper" in TRANSCRIBER_REGISTRY
    assert TRANSCRIBER_REGISTRY["kotoba-whisper"].name == "kotoba-whisper"


def test_class_attrs():
    from asrlabs.transcribe.kotoba import KotobaWhisperTranscriber
    assert KotobaWhisperTranscriber.supports_timestamps is True
    assert KotobaWhisperTranscriber.recommended_aligner is None


def test_parse_plain_text_output():
    """pipeline 返回 {"text": "..."} 时应解析为单段结果"""
    t = make_transcriber()
    t._model = FakePipeline({"text": "こんにちは"})
    result = t.transcribe(np.zeros(16000, dtype=np.float32))

    assert isinstance(result, TranscriptionResult)
    assert result.text == "こんにちは"
    assert result.language == "ja"  # 默认日语
    assert result.model == "kotoba-whisper"
    assert len(result.segments) == 1
    assert result.segments[0].text == "こんにちは"


def test_language_override():
    """config.language 可覆盖默认日语"""
    t = make_transcriber({"language": "en"})
    t._model = FakePipeline({"text": "hello"})
    result = t.transcribe(np.zeros(16000, dtype=np.float32))
    assert result.language == "en"

    # generate_kwargs 应携带覆盖后的语言
    _, kwargs = t._model.last_call
    assert kwargs["generate_kwargs"]["language"] == "en"


def test_parse_chunks_with_timestamps():
    """pipeline 返回 chunks（含 timestamp）时解析为多段带时间戳结果"""
    t = make_transcriber()
    t._model = FakePipeline({
        "text": "全体テキスト",
        "chunks": [
            {"text": "第一文", "timestamp": (0.0, 1.5)},
            {"text": "第二文", "timestamp": (1.5, 3.0)},
        ],
    })
    result = t.transcribe(np.zeros(16000, dtype=np.float32))

    assert result.has_timestamps is True
    assert len(result.segments) == 2
    assert result.segments[0].text == "第一文"
    assert result.segments[0].start == 0.0
    assert result.segments[0].end == 1.5
    assert result.segments[1].start == 1.5
    assert result.segments[1].end == 3.0


def test_add_punctuation_passthrough():
    """extras.add_punctuation=True 时应传递给 pipeline"""
    t = make_transcriber({"extras": {"add_punctuation": True}})
    t._model = FakePipeline({"text": "テスト。"})
    t.transcribe(np.zeros(16000, dtype=np.float32))

    _, kwargs = t._model.last_call
    assert kwargs.get("add_punctuation") is True


def test_batch_size_passthrough():
    """extras.batch_size 应传递给 pipeline"""
    t = make_transcriber({"extras": {"batch_size": 4}})
    t._model = FakePipeline({"text": "x"})
    t.transcribe(np.zeros(16000, dtype=np.float32))

    _, kwargs = t._model.last_call
    assert kwargs.get("batch_size") == 4


def test_generate_kwargs_always_present():
    """无论语言如何，generate_kwargs 必须包含 task=transcribe"""
    t = make_transcriber()
    t._model = FakePipeline({"text": "x"})
    t.transcribe(np.zeros(16000, dtype=np.float32))

    _, kwargs = t._model.last_call
    assert kwargs["generate_kwargs"]["task"] == "transcribe"


def test_reexport():
    """KotobaWhisperTranscriber 应可从 asrlabs.transcribe 直接导入"""
    from asrlabs.transcribe import KotobaWhisperTranscriber as K
    from asrlabs.transcribe.kotoba import KotobaWhisperTranscriber as K2
    assert K is K2


def test_listed_in_transcribers():
    """list_transcribers() 应包含 kotoba-whisper"""
    from asrlabs.transcribe import list_transcribers
    names = [t["name"] for t in list_transcribers()]
    assert "kotoba-whisper" in names
