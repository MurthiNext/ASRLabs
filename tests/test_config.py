"""配置系统测试"""

import os
import tempfile
from pathlib import Path
import pytest
from asrlabs.config import (
    TranscriberConfig,
    AlignerConfig,
    AudioConfig,
    OutputConfig,
    LoggingConfig,
    ProjectConfig,
    load_config,
    _expand_env_vars,
)


class TestExpandEnvVars:
    def test_expand_dollar_brace(self):
        os.environ["TEST_VAR"] = "resolved_value"
        result = _expand_env_vars("prefix_${TEST_VAR}_suffix")
        assert result == "prefix_resolved_value_suffix"

    def test_no_expansion_for_plain_string(self):
        result = _expand_env_vars("no_variable_here")
        assert result == "no_variable_here"

    def test_missing_env_var_raises(self):
        with pytest.raises(ValueError, match="环境变量"):
            _expand_env_vars("${NONEXISTENT_VAR_12345}")


class TestLoadConfig:
    def test_load_minimal_config(self):
        yaml_content = """
transcriber:
  model: whisper
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            cfg = load_config(f.name)
        os.unlink(f.name)
        assert cfg.transcriber.model == "whisper"
        assert cfg.transcriber.model_path == ""  # 默认值
        assert cfg.audio.sample_rate == 16000  # 默认值

    def test_load_full_config(self):
        yaml_content = """
transcriber:
  model: faster-whisper
  model_path: large-v3
  device: cuda
  compute_type: float16
  language: zh
  beam_size: 5
  extras:
    temperature: [0.0, 0.2, 0.4]
    vad_filter: true

aligner:
  name: whisper_align
  extras:
    refine: true

audio:
  sample_rate: 16000
  vad: true
  max_segment_length: 30
  min_silence_dur: 0.5

output:
  formats: [json, srt]
  dir: ./out
  keep_segments: true

logging:
  level: DEBUG
  file: asrlabs.log
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            cfg = load_config(f.name)
        os.unlink(f.name)
        assert cfg.transcriber.model == "faster-whisper"
        assert cfg.transcriber.model_path == "large-v3"
        assert cfg.transcriber.device == "cuda"
        assert cfg.transcriber.language == "zh"
        assert cfg.transcriber.extras == {"temperature": [0.0, 0.2, 0.4], "vad_filter": True}
        assert cfg.aligner.name == "whisper_align"
        assert cfg.output.formats == ["json", "srt"]
        assert cfg.output.keep_segments is True

    def test_missing_transcriber_raises(self):
        yaml_content = "audio:\n  sample_rate: 16000\n"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            with pytest.raises(ValueError, match="transcriber"):
                load_config(f.name)
        os.unlink(f.name)
