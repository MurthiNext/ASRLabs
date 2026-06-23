"""音频预处理——重采样、VAD 分段、静音检测

分段策略:
  1. Silero VAD 检测语音区域（只过滤纯静音，不按短停顿切分）
  2. 合并间距 < merge_gap 的相邻语音区域
  3. 按 max_segment_length 定长切块，块间保留 overlap 重叠
  4. 丢弃短于 min_segment_length 的碎片
"""

import logging
from pathlib import Path
import numpy as np
from asrlabs.config import AudioConfig

logger = logging.getLogger(__name__)

# VAD 内部参数（不暴露给用户配置）
_SPEECH_THRESHOLD = 0.5          # 语音检测阈值
_MIN_SPEECH_DURATION_MS = 500    # 最短语音段（更短视为噪音，丢弃）
_MIN_SILENCE_DURATION_MS = 300   # VAD 内部静音判定（较激进，后续合并会补偿）
_MERGE_GAP = 1.5                 # 合并间距（秒）：相邻语音区隔 < 此值则合并
_CHUNK_OVERLAP = 0.5             # 切块重叠（秒）：块间保留上下文连续


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

    segments = _vad_segment(
        audio,
        sr=config.sample_rate,
        max_segment_length=config.max_segment_length,
    )

    logger.info(
        "VAD 分段完成: %d 段，总时长 %.1fs → %.1fs",
        len(segments),
        len(audio) / sr,
        sum(len(s) for s in segments) / sr,
    )
    return segments if segments else [audio]


def _vad_segment(
    audio: np.ndarray,
    sr: int = 16000,
    max_segment_length: float = 30.0,
) -> list[np.ndarray]:
    """VAD 检测 → 合并相邻区域 → 定长切块 → 丢弃碎片

    Args:
        audio: mono float32 numpy 数组
        sr: 采样率
        max_segment_length: 单段最大长度（秒）

    Returns:
        语音片段列表
    """
    try:
        import torch

        # 1. Silero VAD 检测语音时间戳
        logger.info("加载 Silero VAD 模型（首次自动下载，约 3MB）...")
        model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            trust_repo=True,
            verbose=False,
        )
        (get_speech_timestamps, _, _, _, _) = utils

        audio_tensor = torch.from_numpy(audio)
        speech_ts = get_speech_timestamps(
            audio_tensor,
            model,
            threshold=_SPEECH_THRESHOLD,
            min_speech_duration_ms=_MIN_SPEECH_DURATION_MS,
            min_silence_duration_ms=_MIN_SILENCE_DURATION_MS,
        )

        if not speech_ts:
            logger.info("VAD 未检测到语音，返回完整音频")
            return [audio]

        # 2. 合并间距 < _MERGE_GAP 的相邻语音区域
        merged = _merge_adjacent(speech_ts, sr, merge_gap=_MERGE_GAP)

        # 3. 定长切块（块间 _CHUNK_OVERLAP 重叠）
        chunks = _chunk_segments(audio, merged, sr, max_segment_length, overlap=_CHUNK_OVERLAP)

        logger.info(
            "VAD: %d 个语音区 → 合并为 %d 个区域 → 切成 %d 个块",
            len(speech_ts), len(merged), len(chunks),
        )
        return chunks if chunks else [audio]

    except Exception as e:
        logger.warning("VAD 失败 (%s)，使用完整音频", e)
        return [audio]


def _merge_adjacent(
    timestamps: list[dict],
    sr: int,
    merge_gap: float = 1.5,
) -> list[dict]:
    """合并间距小于 merge_gap 秒的相邻语音区域

    Args:
        timestamps: Silero VAD 返回的语音区域列表 [{"start": int, "end": int}, ...]
        sr: 采样率
        merge_gap: 合并阈值（秒），间距小于此值则合并

    Returns:
        合并后的区域列表
    """
    if not timestamps:
        return []

    gap_samples = int(merge_gap * sr)
    merged = [timestamps[0].copy()]

    for ts in timestamps[1:]:
        last = merged[-1]
        if ts["start"] - last["end"] <= gap_samples:
            # 可合并：扩展上一个区域的结束位置
            last["end"] = ts["end"]
        else:
            merged.append(ts.copy())

    return merged


def _chunk_segments(
    audio: np.ndarray,
    regions: list[dict],
    sr: int,
    max_duration: float,
    overlap: float = 0.5,
) -> list[np.ndarray]:
    """将语音区域按 max_duration 定长切块

    Args:
        audio: 完整音频数组
        regions: 合并后的语音区域列表
        sr: 采样率
        max_duration: 单块最大时长（秒）
        overlap: 块间重叠（秒）

    Returns:
        音频片段 numpy 数组列表
    """
    max_samples = int(max_duration * sr)
    overlap_samples = int(overlap * sr)
    step = max(max_samples - overlap_samples, 1)  # 每次至少前进 1 采样点
    min_samples = int(0.5 * sr)  # 最短 0.5s，更短丢弃

    chunks = []
    for region in regions:
        start = region["start"]
        end = region["end"]

        offset = 0
        while start + offset < end:
            chunk_start = start + offset
            chunk_end = min(chunk_start + max_samples, end)
            chunk = audio[chunk_start:chunk_end]

            if len(chunk) >= min_samples:
                chunks.append(chunk)

            if chunk_end >= end:
                break
            offset += step  # 前进非重叠部分

    return chunks

