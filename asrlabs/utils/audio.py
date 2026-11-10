"""音频工具函数——加载、信息提取、格式转换

使用 soundfile 读取音频（避免 torchcodec 依赖 FFmpeg DLL），
torchaudio 仅用于重采样。
"""

import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator
import numpy as np
import soundfile as sf


def load_audio(path: str | Path, target_sr: int = 16000) -> tuple[np.ndarray, int]:
    """加载音频文件并重采样到目标采样率

    Args:
        path: 音频文件路径
        target_sr: 目标采样率，默认 16000

    Returns:
        (audio_array, sample_rate) 元组，audio 为 mono float32 numpy 数组
    """
    # soundfile 读取（不依赖 FFmpeg）
    data, sr = sf.read(str(path), dtype="float32", always_2d=False)

    # 转 mono
    if data.ndim > 1:
        data = data.mean(axis=1)

    # 重采样
    if sr != target_sr:
        import torch
        import torchaudio.transforms as T

        tensor = torch.from_numpy(data).unsqueeze(0)  # (1, samples)
        resampler = T.Resample(sr, target_sr)
        data = resampler(tensor).squeeze(0).numpy()

    return data.astype(np.float32), target_sr


def get_audio_duration(path: str | Path) -> float:
    """获取音频时长（秒）"""
    info = sf.info(str(path))
    return info.duration


def get_audio_info(path: str | Path) -> dict:
    """获取音频文件的元信息

    Returns:
        dict with keys: duration, sample_rate, num_channels, num_frames
    """
    info = sf.info(str(path))
    return {
        "duration": info.duration,
        "sample_rate": info.samplerate,
        "num_channels": info.channels,
        "num_frames": info.frames,
    }


@contextmanager
def temp_wav_from_array(audio: np.ndarray, sr: int = 16000) -> Generator[str, Any, None]:
    """将 numpy 数组写入临时 WAV 文件，退出时自动清理

    Args:
        audio: float32 numpy 数组 (mono)
        sr: 采样率，默认 16000

    Yields:
        临时 WAV 文件路径
    """
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp_path = f.name
    try:
        sf.write(tmp_path, np.asarray(audio, dtype=np.float32), sr, subtype="PCM_16")
        yield tmp_path
    finally:
        Path(tmp_path).unlink(missing_ok=True)
