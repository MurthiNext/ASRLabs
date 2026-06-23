"""YAML 配置解析——参照 GalTransl ConfigHelper 设计"""

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml


# ── 配置数据类 ──

@dataclass
class TranscriberConfig:
    """听写模型配置"""
    model: str = "whisper"  # 引擎名: whisper, faster-whisper, qwen3-asr, granite-speech
    model_path: str = ""    # 本地模型路径/ID，空则使用引擎默认
    device: str = "auto"
    compute_type: str = "float16"
    language: str = "auto"
    beam_size: int = 3
    extras: dict = field(default_factory=dict)


@dataclass
class AlignerConfig:
    """对齐器配置"""
    name: str = "none"
    extras: dict = field(default_factory=dict)


@dataclass
class AudioConfig:
    """音频预处理配置"""
    sample_rate: int = 16000
    vad: bool = True
    max_segment_length: float = 30.0
    min_silence_dur: float = 0.5


@dataclass
class OutputConfig:
    """输出配置"""
    formats: list[str] = field(default_factory=lambda: ["json"])
    dir: str = "./output"   # 输出目录
    name: str = ""          # 自定义文件名 stem（空则用输入文件名）
    keep_segments: bool = False


@dataclass
class LoggingConfig:
    """日志配置"""
    level: str = "INFO"
    file: str | None = None


@dataclass
class ProjectConfig:
    """项目总配置"""
    transcriber: TranscriberConfig
    aligner: AlignerConfig = field(default_factory=AlignerConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


# ── 环境变量展开 ──

_ENV_VAR_PATTERN = re.compile(r"\$\{(\w+)\}")


def _expand_env_vars(value: str) -> str:
    """展开字符串中的 ${VAR} 环境变量引用"""
    def _replace(match):
        var_name = match.group(1)
        val = os.environ.get(var_name)
        if val is None:
            raise ValueError(f"环境变量 {var_name} 未设置，配置中引用了 ${{{var_name}}}")
        return val
    return _ENV_VAR_PATTERN.sub(_replace, value)


def _expand_env_in_dict(data: dict) -> dict:
    """递归展开字典中所有字符串值的环境变量"""
    result = {}
    for key, value in data.items():
        if isinstance(value, str):
            result[key] = _expand_env_vars(value)
        elif isinstance(value, dict):
            result[key] = _expand_env_in_dict(value)
        elif isinstance(value, list):
            result[key] = [
                _expand_env_vars(v) if isinstance(v, str) else v
                for v in value
            ]
        else:
            result[key] = value
    return result


# ── 配置加载 ──

def load_config(path: str | Path) -> ProjectConfig:
    """加载 YAML 配置文件

    Args:
        path: config.yaml 文件路径

    Returns:
        ProjectConfig 实例

    Raises:
        ValueError: 配置缺失必要字段或格式错误
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在: {path}")

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    raw = _expand_env_in_dict(raw)

    if "transcriber" not in raw:
        raise ValueError("配置文件缺少 transcriber 字段")

    tc_raw = raw["transcriber"]
    transcriber = TranscriberConfig(
        model=tc_raw.get("model", "whisper"),
        model_path=tc_raw.get("model_path", ""),
        device=tc_raw.get("device", "auto"),
        compute_type=tc_raw.get("compute_type", "float16"),
        language=tc_raw.get("language", "auto"),
        beam_size=tc_raw.get("beam_size", 5),
        extras=tc_raw.get("extras", {}),
    )

    al_raw = raw.get("aligner", {})
    aligner = AlignerConfig(
        name=al_raw.get("name", "none"),
        extras=al_raw.get("extras", {}),
    )

    au_raw = raw.get("audio", {})
    audio = AudioConfig(
        sample_rate=au_raw.get("sample_rate", 16000),
        vad=au_raw.get("vad", True),
        max_segment_length=au_raw.get("max_segment_length", 30.0),
        min_silence_dur=au_raw.get("min_silence_dur", 0.5),
    )

    out_raw = raw.get("output", {})
    output = OutputConfig(
        formats=out_raw.get("formats", ["json"]),
        dir=out_raw.get("dir", "./output"),
        name=out_raw.get("name", ""),
        keep_segments=out_raw.get("keep_segments", False),
    )

    log_raw = raw.get("logging", {})
    logging = LoggingConfig(
        level=log_raw.get("level", "INFO"),
        file=log_raw.get("file", None),
    )

    return ProjectConfig(
        transcriber=transcriber,
        aligner=aligner,
        audio=audio,
        output=output,
        logging=logging,
    )
