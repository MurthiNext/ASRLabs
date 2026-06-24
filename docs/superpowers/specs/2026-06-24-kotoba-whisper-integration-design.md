# Kotoba Whisper v2.2 后端集成设计

## 概述

为 ASRLabs 添加 `kotoba-whisper` 听写后端，使用 [kotoba-tech/kotoba-whisper-v2.2](https://huggingface.co/kotoba-tech/kotoba-whisper-v2.2) 模型，主要面向日语语音识别场景。

## 技术分析

| 项目 | 详情 |
|------|------|
| 架构 | Distil-Whisper（基于 Whisper Large-v3, 756M 参数） |
| 依赖库 | HuggingFace `transformers` v4.39+ |
| 加载方式 | `pipeline(..., trust_remote_code=True)` |
| 语言 | 日语为主（ReazonSpeech 语料库） |
| 内部处理 | 15 秒窗口分块 + 批量推理 |
| 输出格式 | dict: `{"text": "..."}` 或含 chunks |
| 时间戳 | 支持（内部通过 stable-ts 改进） |
| 许可证 | Apache 2.0 |

### 集成可行性

项目已具备集成条件：
- `transformers` 依赖已存在（granite-speech、cohere 共用）
- `trust_remote_code` 模式已有先例（`cohere.py`）
- 装饰器注册机制直接可用

## 实现方案

### 方案选择：pipeline() 包装器（选定）

新建 `asrlabs/transcribe/kotoba.py`，使用 `transformers.pipeline()` + `trust_remote_code=True` 加载模型。

### 新增文件

**`asrlabs/transcribe/kotoba.py`**（~80-100 行）

```python
"""Kotoba Whisper v2.2 日语听写后端"""

import numpy as np
from asrlabs.models import TranscriptionResult, Segment, Word
from asrlabs.transcribe.base import BaseTranscriber, register_transcriber


@register_transcriber
class KotobaWhisperTranscriber(BaseTranscriber):
    name = "kotoba-whisper"
    display_name = "Kotoba Whisper v2.2 (日本語)"
    supports_timestamps = True
    recommended_aligner = None

    def load_model(self) -> None:
        """加载 Kotoba Whisper v2.2 pipeline"""
        import torch
        from transformers import pipeline

        device = self.config.get("device", "auto")
        if device == "auto":
            device = "cuda:0" if torch.cuda.is_available() else "cpu"

        is_cuda = device.startswith("cuda") or device == "auto" and torch.cuda.is_available()
        torch_dtype = torch.float16 if is_cuda else torch.float32
        model_kwargs = {}
        if is_cuda:
            model_kwargs["attn_implementation"] = "sdpa"

        self._model = pipeline(
            model="kotoba-tech/kotoba-whisper-v2.2",
            torch_dtype=torch_dtype,
            device=device,
            model_kwargs=model_kwargs,
            chunk_length_s=15,
            trust_remote_code=True,
        )

    def transcribe(self, audio: str | np.ndarray, **kwargs) -> TranscriptionResult:
        self._ensure_loaded()

        # Kotoba 默认日语，用户可通过 config.language 覆盖
        lang = self.config.get("language", "ja")
        generate_kwargs = {"language": lang, "task": "transcribe"}

        extras = {**self.config.get("extras", {}), **kwargs}

        # 构建 pipeline 调用参数
        pipe_kwargs = {
            "generate_kwargs": generate_kwargs,
        }
        # 标点恢复（可选，需安装 punctuators 库）
        if extras.pop("add_punctuation", False):
            pipe_kwargs["add_punctuation"] = True
        # 批大小
        batch_size = extras.pop("batch_size", 8)
        if batch_size:
            pipe_kwargs["batch_size"] = batch_size

        result = self._model(audio, **pipe_kwargs)

        # 解析 pipeline 输出
        return self._parse_output(result, lang)

    def _parse_output(self, result: dict, language: str) -> TranscriptionResult:
        """将 pipeline 输出解析为 TranscriptionResult"""
        # 提取全文
        full_text = result.get("text", "").strip()
        # 当前 pipeline 基本模式返回 {"text": "..."}，无逐段时间戳
        # chunks 字段在 diarization 场景下才出现
        segments = []
        chunks = result.get("chunks")
        if chunks and isinstance(chunks, list):
            for chunk in chunks:
                text = chunk.get("text", "").strip()
                seg = Segment(text=text, start=0.0, end=0.0, words=[])
                # 如果 chunk 有时间戳字段则提取
                ts = chunk.get("timestamp")
                if ts and isinstance(ts, (list, tuple)) and len(ts) == 2:
                    seg.start = ts[0] or 0.0
                    seg.end = ts[1] or 0.0
                segments.append(seg)

        return TranscriptionResult(
            text=full_text,
            segments=segments if segments else [Segment(text=full_text, start=0.0, end=0.0, words=[])],
            language=language,
            duration=0.0,
            model=self.name,
            has_timestamps=bool(segments),
        )

    @staticmethod
    def _cuda_available() -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
```

### 需修改的文件

| 文件 | 改动内容 |
|------|----------|
| `asrlabs/transcribe/__init__.py` | 添加 `from . import kotoba`（约 1 行） |
| `asrlabs/cli.py` | `init` 命令的 YAML 模板中添加 kotoba-whisper 示例项（约 5 行） |

### 无需修改的文件

- `config.py` — 配置字典直接传入，无需新增配置字段
- `base.py` — 注册机制完全兼容
- `runner.py` — Runner 透明调用
- `models.py` — 输出类型无变化
- `pipeline/preprocess.py` — VAD 分段自然兼容（pipeline 内部 15s 子分块叠加无害）

## 使用方式

### 命令行
```bash
asrlab transcribe audio.wav -m kotoba-whisper -l ja
```

### 配置文件
```yaml
transcriber:
  model: kotoba-whisper
  language: ja
  extras:
    add_punctuation: true
    batch_size: 8
```

## 工作流

```
audio.wav → Runner.preprocess_audio() → [numpy chunks]
  → 对每个 chunk: KotobaWhisperTranscriber.transcribe()
    → pipeline(chunk, generate_kwargs={language: "ja", task: "transcribe"})
    → pipeline 内部 15s 子分块 + 批量推理
    → 返回 {"text": "..."}
  → Runner._merge_results() 合并各 chunk
  → Runner._save_output() → JSON/SRT/TXT
```

## 不做的事（明确排除）

- 不集成说话人分离（diarization）— 需 `pyannote.audio` + `diarizers`，依赖过重
- 不修改 VAD 策略 — 当前双重分块不影响正确性
- 不添加新的配置项 — Kotoba 的专属参数通过 `extras` 透传
- `punctuators` 列为可选依赖，不写入 `pyproject.toml` 核心依赖

## 验证方式

1. `pytest tests/ -v` — 确保现有 27 个测试全部通过
2. 手动端到端测试：
   ```bash
   .venv/Scripts/python.exe -m asrlabs transcribe 日本語音声.wav -m kotoba-whisper -l ja -f json,srt
   ```
3. 验证输出 JSON 包含正确日语文本
4. 验证 `asrlab list transcribers` 输出中包含 `kotoba-whisper`