"""Pipeline 集成测试——使用 mock 模型验证端到端流程"""

import tempfile
from pathlib import Path
import pytest
import numpy as np


class MockTranscriber:
    """Mock 听写模型——用于测试 pipeline"""
    name = "mock"
    display_name = "Mock"
    supports_timestamps = True
    recommended_aligner = None

    def __init__(self, config):
        self.config = config
        self._loaded = False

    def load_model(self):
        self._loaded = True

    def transcribe(self, audio, **kwargs):
        from asrlabs.models import TranscriptionResult, Segment, Word
        return TranscriptionResult(
            text="这是一段测试文本",
            segments=[
                Segment(
                    text="这是一段测试文本",
                    start=0.0,
                    end=2.0,
                    words=[
                        Word("这是", 0.0, 0.5, 0.9),
                        Word("一段", 0.5, 1.0, 0.9),
                        Word("测试", 1.0, 1.5, 0.9),
                        Word("文本", 1.5, 2.0, 0.9),
                    ],
                )
            ],
            language="zh",
            duration=2.0,
            model="mock",
            has_timestamps=True,
        )

    def unload(self):
        self._loaded = False

    def _ensure_loaded(self):
        if not self._loaded:
            self.load_model()


class TestRunner:
    def test_runner_single_file(self, monkeypatch):
        """测试 Runner 单文件处理流程"""
        # 注册 mock 模型
        from asrlabs.transcribe import TRANSCRIBER_REGISTRY
        monkeypatch.setitem(TRANSCRIBER_REGISTRY, "mock", MockTranscriber)

        # 模拟预处理步骤，避免实际加载音频文件
        import asrlabs.pipeline.runner as runner_mod
        monkeypatch.setattr(
            runner_mod, "preprocess_audio",
            lambda audio, config: [np.zeros(16000, dtype=np.float32)]
        )

        # 创建临时配置
        yaml_content = """
transcriber:
  model: mock
  language: zh
aligner:
  name: none
output:
  formats: [json, txt]
  dir: ./output
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as cf:
            cf.write(yaml_content)
            config_path = cf.name

        # 创建伪音频文件
        import soundfile
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as af:
            # 生成 1 秒静音
            silence = np.zeros(16000, dtype=np.float32)
            soundfile.write(af.name, silence, 16000)
            audio_path = af.name

        try:
            from asrlabs.pipeline.runner import Runner
            runner = Runner(config_path)
            result = runner.run(audio_path)

            assert result.text == "这是一段测试文本"
            assert result.language == "zh"
            assert result.has_timestamps is True
            assert len(result.segments) == 1
        finally:
            Path(config_path).unlink(missing_ok=True)
            Path(audio_path).unlink(missing_ok=True)
            # 清理输出
            import shutil
            shutil.rmtree(Path(audio_path).parent / "output", ignore_errors=True)

    def test_merge_multiple_results(self):
        """测试多段结果合并"""
        from asrlabs.pipeline.runner import Runner
        from asrlabs.models import TranscriptionResult, Segment

        # 构造 Runner 最小实例
        runner = Runner.__new__(Runner)

        r1 = TranscriptionResult(
            text="第一段",
            segments=[Segment("第一段", 0.0, 2.0)],
            duration=2.0,
            model="mock",
            has_timestamps=True,
        )
        r2 = TranscriptionResult(
            text="第二段",
            segments=[Segment("第二段", 0.0, 3.0)],
            duration=3.0,
            model="mock",
            has_timestamps=True,
        )

        merged = runner._merge_results([r1, r2])
        assert merged.text == "第一段 第二段"
        assert merged.duration == 5.0
        assert len(merged.segments) == 2
