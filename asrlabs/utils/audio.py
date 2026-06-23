"""音频工具函数——加载、信息提取、格式转换"""

from pathlib import Path
import numpy as np


def load_audio(path: str | Path, target_sr: int = 16000) -> tuple[np.ndarray, int]:
    """加载音频文件并重采样到目标采样率

    Args:
        path: 音频文件路径
        target_sr: 目标采样率，默认 16000

    Returns:
        (audio_array, sample_rate) 元组，audio 为 mono float32 numpy 数组
    """
    import torchaudio

    waveform, sr = torchaudio.load(str(path), normalize=True)
    # 转 mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    # 重采样
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        waveform = resampler(waveform)
    return waveform.squeeze(0).numpy().astype(np.float32), target_sr


def get_audio_duration(path: str | Path) -> float:
    """获取音频时长（秒）"""
    import torchaudio

    info = torchaudio.info(str(path))
    return info.num_frames / info.sample_rate


def get_audio_info(path: str | Path) -> dict:
    """获取音频文件的元信息

    Returns:
        dict with keys: duration, sample_rate, num_channels, num_frames
    """
    import torchaudio

    info = torchaudio.info(str(path))
    return {
        "duration": info.num_frames / info.sample_rate,
        "sample_rate": info.sample_rate,
        "num_channels": info.num_channels,
        "num_frames": info.num_frames,
    }
