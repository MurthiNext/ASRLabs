# ASRLabs

**ASR 工具箱 —— 整合多款开源 ASR 模型，开箱即用**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)[![PyTorch](https://img.shields.io/badge/PyTorch-2.12-ee4c2c.svg)](https://pytorch.org/)

## 目录

- [项目简介](#项目简介)
- [支持引擎](#支持引擎)
- [快速开始](#快速开始)
  - [安装](#安装)
  - [基本使用](#基本使用)
  - [配置文件](#配置文件)
- [命令参考](#命令参考)
  - [transcribe —— 听写](#transcribe-命令行参考)
  - [align —— 对齐](#align-命令行参考)
  - [list / init](#其他命令)
- [项目结构](#项目结构)
- [扩展开发](#扩展开发)
- [依赖环境](#依赖环境)
- [致谢](#致谢)

## 项目简介

ASRLabs 是一个 Python ASR 工具箱，参照 [GalTransl](https://github.com/GalTransl/GalTransl) 的分层架构设计，整合业界领先的开源语音识别模型，提供统一接口、策略模式可扩展、开箱即用的听写与对齐体验。

**核心功能**：

- **听写 (Transcription)** —— 音频 → 文本。支持 5 种引擎，VAD 自动分段，统一输出 JSON/SRT/TXT
- **对齐 (Alignment)** —— 文本 + 音频 → 带时间戳文本。听写与对齐分离，任意组合

## 支持引擎

### 听写引擎

| 引擎 | `-m` 参数 | 后端 | 时间戳 | 推荐对齐器 |
|---|---|---|---|---|
| OpenAI Whisper | `whisper` | stable-ts | ✅ 内置 | `whisper_align` |
| Faster Whisper | `faster-whisper` | stable-ts (CTranslate2) | ✅ 内置 | `whisper_align` |
| Qwen3 ASR | `qwen3-asr` | qwen-asr | ❌ | `qwen3_align` |
| IBM Granite Speech | `granite-speech` | transformers | ❌ | `qwen3_align` |
| Cohere Transcribe | `cohere-transcribe` | transformers | ❌ | `qwen3_align` |

> **Whisper / Faster Whisper** 统一使用 [stable-ts](https://github.com/jianfch/stable-ts) 作为后端，获得更精准的静音抑制、VAD 预处理和词级时间戳。
>
> **Faster Whisper** 支持 `--device vulkan`（CTranslate2 后端）。
>
> **Cohere Transcribe** 当前仅支持本地 transformers 模式，云端 API 待后续支持。

### 对齐引擎

| 对齐器 | `--aligner` 参数 | 后端 | 适用模型 | 限制 |
|---|---|---|---|---|
| Whisper Align | `whisper_align` | stable-ts align() + refine() | 仅 whisper / faster-whisper | 需同款模型权重 |
| Qwen3 Forced Aligner | `qwen3_align` | qwen-asr | 任意模型 | GPU, 单段 ≤ 5 分钟 |

## 快速开始

### 安装

```bash
git clone https://github.com/MurthiNext/ASRLabs.git
cd ASRLabs
python -m venv .venv
.venv\Scripts\activate  # Windows

# 安装依赖（CUDA 13 环境）
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu130

# 安装 ASRLabs
pip install -e .
```

<details>
<summary>其他 PyTorch 索引</summary>

```bash
# CPU 环境
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# CUDA 12.8 环境
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128
```
</details>

### 基本使用

```bash
# 列出可用引擎
asrlab list transcribers
asrlab list aligners

# 生成配置模板
asrlab init

# ── 听写 ──

# Faster Whisper — 本地 CTranslate2 模型（推荐，速度快）
asrlab transcribe audio.wav -m faster-whisper --model-path large-v3 --device cuda --lang ja

# OpenAI Whisper — 指定模型尺寸
asrlab transcribe audio.wav -m whisper --model-path large-v3 --device cuda

# Qwen3 ASR — HuggingFace 模型
asrlab transcribe audio.wav -m qwen3-asr --model-path Qwen/Qwen3-ASR-1.7B --device cuda

# Granite Speech — 本地路径
asrlab transcribe audio.wav -m granite-speech --model-path "H:\Models\ibm-granite\granite-speech-4.1-2b" --device cuda

# Cohere Transcribe — 本地路径
asrlab transcribe audio.wav -m cohere-transcribe --model-path "H:\Models\CohereLabs\cohere-transcribe-03-2026" --device cuda --lang ja

# 输出 SRT 字幕
asrlab transcribe audio.wav -m whisper --model-path large-v3 -f json,srt

# ── 对齐 ──

# Qwen3 对齐器（通用，适合无时间戳的听写结果）
asrlab align audio.wav result.json --aligner qwen3_align --model-path "H:\Models\Qwen\Qwen3-ForcedAligner-0.6B" --device cuda

# Whisper 对齐器（仅适用 Whisper 系列听写结果）
asrlab align audio.wav result.json --aligner whisper_align --model-path large-v3 --device cuda

# 输出 SRT
asrlab align audio.wav result.json --aligner qwen3_align -f srt,json
```

### 配置文件

`asrlab init` 生成 `config.yaml`，支持所有参数预设。使用 `-c` 加载后，CLI 参数会覆盖配置中的对应项。

```yaml
transcriber:
  model: whisper              # 引擎名
  model_path: large-v3        # 本地模型路径 / HF ID / stable-ts 尺寸名
  device: cuda                # cuda | cpu | vulkan | auto
  compute_type: float16       # float16 | int8 | float32
  language: auto              # auto | zh | en | ja | ko ...
  beam_size: 5
  extras:                     # 透传底层库特有参数
    temperature: [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    vad_filter: true

aligner:
  name: qwen3_align           # whisper_align | qwen3_align | none
  extras: {}

audio:
  sample_rate: 16000
  vad: true                   # Silero VAD 自动分段
  max_segment_length: 30.0    # 单段最大 30 秒
  min_silence_dur: 0.5

output:
  formats: [json]             # json, srt, txt
  dir: ./output
  keep_segments: false
```

> `${VAR}` 语法支持环境变量引用，适合存放 API Key 等敏感信息。

## 命令参考

### `transcribe` 命令行参考

```
asrlab transcribe <音频文件|目录> [选项]
```

| 选项 | 默认值 | 说明 |
|---|---|---|
| `-m, --model` | `whisper` | 引擎名（见上表） |
| `--model-path` | `""` | 模型路径 / HF ID / stable-ts 尺寸名 |
| `-f, --formats` | `json` | 输出格式，逗号分隔 |
| `-l, --lang` | `auto` | 语言代码 |
| `--aligner` | — | 对齐器名称（可选，输出带时间戳） |
| `-c, --config` | — | 配置文件路径 |
| `-o, --output-dir` | `./output` | 输出目录 |
| `--batch` | — | 批量处理目录下所有音频 |
| `--device` | `auto` | `cuda` / `cpu` / `vulkan` / `auto` |
| `--compute-type` | `float16` | `float16` / `int8` / `float32` |

### `align` 命令行参考

```
asrlab align <音频文件> [参考文件] [选项]
```

| 选项 | 默认值 | 说明 |
|---|---|---|
| `参考文件` | — | `.json` / `.srt` / `.vtt` / `.txt` 文件（自动检测格式） |
| `-t, --text` | — | 直接指定文本（与参考文件互斥） |
| `-f, --formats` | `json` | 输出格式 |
| `-l, --lang` | `auto` | 语言代码（纯文本 `.txt` 时建议指定） |
| `--aligner` | `qwen3_align` | 对齐器名称 |
| `--model-path` | `""` | 对齐模型路径 / HF ID |
| `--device` | `auto` | `cuda` / `cpu` / `auto` |

### 其他命令

```bash
# 列出可用引擎
asrlab list transcribers
asrlab list aligners

# 生成配置模板
asrlab init
```

## 项目结构

```
ASRLabs/
├── asrlabs/                     # 核心包
│   ├── __init__.py              # 版本、入口
│   ├── cli.py                   # Click 命令行
│   ├── config.py                # YAML 配置解析
│   ├── models.py                # 数据模型 (Word, Segment, TranscriptionResult)
│   ├── transcribe/              # 听写后端（策略模式）
│   │   ├── base.py              #   BaseTranscriber + 注册表
│   │   ├── whisper.py           #   OpenAI Whisper
│   │   ├── faster_whisper.py    #   Faster Whisper (CTranslate2)
│   │   ├── qwen3_asr.py         #   Qwen3 ASR
│   │   ├── granite_speech.py    #   IBM Granite Speech
│   │   └── cohere.py            #   Cohere Transcribe
│   ├── align/                   # 对齐后端（策略模式）
│   │   ├── base.py              #   BaseAligner + 注册表
│   │   ├── whisper_align.py     #   Whisper 对齐器
│   │   └── qwen3_align.py       #   Qwen3 Forced Aligner
│   ├── pipeline/                # 编排层
│   │   ├── runner.py            #   Runner 主编排器
│   │   └── preprocess.py        #   重采样 + Silero VAD 分段
│   └── utils/                   # 工具函数
│       ├── audio.py             #   音频加载
│       └── formats.py           #   SRT/JSON 解析
├── sample_project/              # 示例项目
├── tests/                       # 单元测试
├── requirements.txt
├── pyproject.toml
└── README.md
```

## 扩展开发

新增听写引擎只需两步：

```python
# asrlabs/transcribe/my_model.py
from asrlabs.transcribe.base import BaseTranscriber, register_transcriber

@register_transcriber
class MyModelTranscriber(BaseTranscriber):
    name = "my-model"             # CLI 使用 -m my-model
    display_name = "My ASR Model"
    supports_timestamps = False   # 模型是否自带时间戳
    recommended_aligner = "qwen3_align"

    def load_model(self) -> None:
        model_path = self.model_path or "default/model/id"
        # 加载模型...

    def transcribe(self, audio, **kwargs):
        # 返回 TranscriptionResult...
```

然后在 `asrlabs/transcribe/__init__.py` 的导入列表中加入一行即可。

## 依赖环境

| 依赖 | 版本 | 说明 |
|---|---|---|
| Python | >= 3.10 | |
| PyTorch | >= 2.12 | CUDA 13.0 推荐 |
| transformers | >= 4.52, < 5.0 | Qwen3 ASR 兼容性要求 |
| stable-ts | >= 2.17 | Whisper / Faster Whisper 后端 |
| faster-whisper | >= 1.0 | CTranslate2 引擎 |
| qwen-asr | >= 0.0.6 | Qwen3 ASR + ForcedAligner |
| soundfile | >= 0.12 | 音频 I/O |
| nvidia-cublas-cu12 | — | CUDA 13 环境下为 CTranslate2 提供 CUDA 12 DLL |

> **注意**：`transformers` 需锁定 `< 5.0`，因 `qwen-asr` 尚不兼容 5.x 的配置系统和生成 API。等待 qwen-asr 更新后可移除此限制。

## 致谢

本项目结构参照 [GalTransl](https://github.com/GalTransl/GalTransl) 设计，受益于其策略模式 + 编排层的清晰架构。感谢以下开源项目：

- [stable-ts](https://github.com/jianfch/stable-ts) — Whisper 词级时间戳增强
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) — CTranslate2 加速推理
- [Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR) — 多语言 ASR + ForcedAligner
- [Cohere Transcribe](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026) — Conformer-based ASR
- [IBM Granite Speech](https://huggingface.co/ibm-granite/granite-speech-4.1-2b) — IBM 语音模型
- [Silero VAD](https://github.com/snakers4/silero-vad) — 语音活动检测

## License

此项目使用 [MIT License](LICENSE).