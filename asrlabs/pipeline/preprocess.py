"""音频预处理——重采样、VAD 分段、静音检测"""

from pathlib import Path
import numpy as np
from asrlabs.config import AudioConfig


def preprocess_audio(
    audio_path: str | Path, config: AudioConfig
) -> list[np.ndarray]:
    """预处理音频：加载、重采样、VAD 分段

    Args:
        audio_path: 音频文件路径
        config: 音频配置

    Returns:
        分段后的音频片段列表，每个为 float32 numpy 数组 (sr=config.sample_rate)
    """
    from asrlabs.utils.audio import load_audio

    audio, sr = load_audio(audio_path, target_sr=config.sample_rate)

    if not config.vad:
        return [audio]

    # 使用 Silero VAD 进行语音活动检测和分段
    segments = _vad_split(
        audio,
        sr=config.sample_rate,
        max_segment_length=config.max_segment_length,
        min_silence_dur=config.min_silence_dur,
    )

    return segments if segments else [audio]


def _vad_split(
    audio: np.ndarray,
    sr: int = 16000,
    max_segment_length: float = 30.0,
    min_silence_dur: float = 0.5,
) -> list[np.ndarray]:
    """使用 Silero VAD 将长音频分割为语音片段

    Args:
        audio: 音频 numpy 数组
        sr: 采样率
        max_segment_length: 单段最大长度（秒）
        min_silence_dur: 最小静音时长（秒）

    Returns:
        语音片段列表
    """
    try:
        import torch

        model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
        )
        (get_speech_timestamps, _, _, _, _) = utils

        # Silero VAD 需要 torch tensor
        audio_tensor = torch.from_numpy(audio)

        speech_timestamps = get_speech_timestamps(
            audio_tensor,
            model,
            threshold=0.5,
            min_speech_duration_ms=250,
            min_silence_duration_ms=int(min_silence_dur * 1000),
        )

        if not speech_timestamps:
            return [audio]

        segments = []
        for ts in speech_timestamps:
            start_sample = ts["start"]
            end_sample = ts["end"]
            duration = (end_sample - start_sample) / sr

            # 长段进一步切分
            if duration > max_segment_length:
                max_samples = int(max_segment_length * sr)
                for offset in range(0, end_sample - start_sample, max_samples):
                    chunk_end = min(offset + max_samples, end_sample - start_sample)
                    segments.append(audio[start_sample + offset:start_sample + chunk_end])
            else:
                segments.append(audio[start_sample:end_sample])

        return segments if segments else [audio]

    except Exception:
        # VAD 失败则返回完整音频
        return [audio]
