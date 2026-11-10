# ASRLabs

**ASR 工具箱 —— 整合多款开源 ASR 模型，开箱即用**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)[![PyTorch](https://img.shields.io/badge/PyTorch-2.12-ee4c2c.svg)](https://pytorch.org/)

## 目录

- [项目简介](#项目简介)
- [支持引擎](#支持引擎)
- [快速开始](#快速开始)
- [命令参考](#命令参考)
- [格式转换](#格式转换)
- [输出控制](#输出控制)
- [优化特性](#优化特性)
- [项目结构](#项目结构)
- [扩展开发](#扩展开发)
- [依赖环境](#依赖环境)
- [致谢](#致谢)

## 项目简介

ASRLabs 是一个 Python ASR 工具箱，整合业界领先的开源语音识别模型，提供统一接口、策略模式可扩展、开箱即用的听写与对齐体验。

**核心功能**：

- **听写 (Transcription)** —— 音频 → 文本，输出 JSON
- **对齐 (Alignment)** —— 文本 + 音频 → 带时间戳的 JSON
- **转换 (Convert)** —— JSON → SRT/LRC/TXT，支持 pysbd 语义分句或遇标点切分

> transcribe / align 仅输出 JSON，格式转换统一交给 `convert` 命令处理。

## 支持引擎

### 听写引擎

受限于 Qwen3ASR 较低的 transformers 后端版本支持，无法使用 transformers 5.x 后端，因此部分引擎会启用 trust_remote_code 或 patch 依赖库来解决兼容性问题，请知悉此内容。

| 引擎               | `-m` 参数           | 后端                    | 时间戳     | 推荐对齐器      |
| ------------------ | ------------------- | ----------------------- | ---------- | --------------- |
| OpenAI Whisper     | `whisper`           | stable-ts               | ✅ 内置    | `whisper_align` |
| Faster Whisper     | `faster-whisper`    | stable-ts (CTranslate2) | ✅ 内置    | `whisper_align` |
| Qwen3 ASR          | `qwen3-asr`         | qwen-asr                | ❌         | `qwen3_align`   |
| IBM Granite Speech | `granite-speech`    | transformers            | ❌         | `qwen3_align`   |
| Cohere Transcribe  | `cohere-transcribe` | transformers            | ❌         | `qwen3_align`   |
| Kotoba Whisper     | `kotoba-whisper`    | transformers            | ✅ 内置    | —               |
| ARK-ASR            | `ark-asr`           | transformers            | ❌         | `qwen3_align`   |

> **Whisper / Faster Whisper** 统一使用 [stable-ts](https://github.com/jianfch/stable-ts) 作为后端，获得更精准的静音抑制、VAD 预处理和词级时间戳。
>
> **Faster Whisper** 独家支持 `--device vulkan`（CTranslate2 后端加速）。
>
> **Kotoba Whisper** 是日语特化 Distil-Whisper，通过 transformers pipeline 加载，pipeline 内部自带 15s 子分块与批量推理。语言默认 `ja`，与其它引擎的 `auto` 默认不同。

### 对齐引擎

| 对齐器               | `--aligner` 参数 | 后端                          | 适用模型                     | 限制               |
| -------------------- | ---------------- | ----------------------------- | ---------------------------- | ------------------ |
| Whisper Align        | `whisper_align`  | stable-ts align() + refine()  | 仅 whisper / faster-whisper  | 需同款模型权重     |
| Qwen3 Forced Aligner | `qwen3_align`    | qwen-asr                      | 任意模型                     | GPU, 单段 ≤ 5 分钟 |
| CTC Forced Aligner   | `ctc_align`      | ctc-forced-aligner (MMS/Wav2Vec2/HuBERT) | 任意模型           | 需 `[ctc]` extra + ffmpeg |

> 听写产出的 JSON 直接作为参考文件传给对齐器，无需额外格式转换。
>
> **CTC Forced Aligner** 基于 [MahmoudAshraf97/ctc-forced-aligner](https://github.com/MahmoudAshraf97/ctc-forced-aligner)，内置 uroman 罗马化，支持 1136+ 语言。CJK (zh/ja/ko) 自动 char 级对齐，拉丁语系 word 级。需 `pip install -e .[ctc]`。默认模型 MMS-300m-1130 为 **CC-BY-NC 4.0（非商用）**，商用请通过 `--model-path` 指定 MIT 模型（如 `facebook/wav2vec2-large-960h-lv60-self`）。

## 快速开始

### 安装

```bash
git clone https://github.com/MurthiNext/ASRLabs.git
cd ASRLabs
python -m venv .venv
.venv\Scripts\activate  # Windows

# CUDA 13 环境
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu130
pip install -e .

# CPU / CUDA 12.8 只需换索引:
#   --extra-index-url https://download.pytorch.org/whl/cpu
#   --extra-index-url https://download.pytorch.org/whl/cu128
```

### 基本使用

```bash
# 列出可用引擎
asrlabs list transcribers
asrlabs list aligners
asrlabs list transcribers --json # 以 JSON 形式输出

# 生成配置模板
asrlabs init

# ── 听写（仅输出 JSON）──
asrlabs transcribe audio.wav -m whisper --model-path large-v3 -d ./output
asrlabs transcribe audio.wav -m faster-whisper -o my_result -d ./output --device cuda
asrlabs transcribe audio.wav -m qwen3-asr --model-path Qwen/Qwen3-ASR-1.7B --device cuda
asrlabs transcribe audio.wav -m cohere-transcribe --model-path local/model --lang ja
asrlabs transcribe audio.wav -m kotoba-whisper
asrlabs transcribe audio.wav -m ark-asr --model-path AutoArk-AI/ARK-ASR-3B --device cuda

# 批量处理
asrlabs transcribe ./audio_dir/ -m faster-whisper -d ./results --batch

# ── 对齐（仅输出 JSON）──
asrlabs align audio.wav result.json -d ./output
asrlabs align audio.wav result.json --aligner qwen3_align --device cuda
asrlabs align audio.wav -t "transcript text" -l ja

# ── 格式转换 ──
asrlabs convert result.json                          # pysbd 分句 → SRT
asrlabs convert result.json -f srt,lrc               # 同时输出 SRT + LRC
asrlabs convert result.json -f txt -l punct          # 遇标点切分 → 纯文本
asrlabs convert ./output_dir/ --batch -f srt -d ./subtitles/
```

### 配置文件

`asrlabs init` 生成 `config.yaml`。CLI 参数优先级高于配置项。

```yaml
transcriber:
  model: whisper              # 引擎名
  model_path: large-v3        # 本地模型路径 / HF ID / stable-ts 尺寸名
  device: cuda                # cuda | cpu | vulkan | auto
  compute_type: float16       # float16 | int8 | float32
  language: auto              # auto | ISO 639-1 代码 (zh/en/ja/ko ...)
  beam_size: 3                # 搜索宽度（越小越快）
  extras:
    temperature: [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    vad_filter: true

aligner:
  name: qwen3_align           # whisper_align | qwen3_align | ctc_align | none
  extras: {}

audio:
  sample_rate: 16000
  vad: true                   # Silero VAD 自动分段
  max_segment_length: 30.0    # 单段最大 30 秒
  min_silence_dur: 0.5

output:
  dir: ""
  name: ""
  keep_segments: false
# transcribe/align 仅输出 JSON；格式转换请使用 asrlabs convert
```

> `${VAR}` 语法支持环境变量引用，适合存放 API Key 等敏感信息。

## 命令参考

可能不会及时更新，有必要请使用 `--help` 参数查看具体情况。

### `transcribe` 命令行参考

```
asrlabs transcribe <音频文件|目录> [选项]
```

| 选项             | 默认值    | 说明                                |
| ---------------- | --------- | ----------------------------------- |
| `-m, --model`    | `whisper` | 引擎名（见引擎表）                  |
| `--model-path`   | `""`      | 模型路径 / HF ID / stable-ts 尺寸名 |
| `-l, --lang`     | `auto`    | ISO 639-1 语言代码（zh/en/ja ...）  |
| `--aligner`      | —         | 对齐器名称（可选）                  |
| `-c, --config`   | —         | 配置文件路径                        |
| `-d, --dir`      | `""`      | 输出目录（空=与音频同目录）         |
| `-o, --output`   | `""`      | 输出文件名 stem（不含扩展名）       |
| `--batch`        | —         | 批量处理目录（仅允许 -d）           |
| `--device`       | `auto`    | `cuda` / `cpu` / `vulkan` / `auto`  |
| `--compute-type` | `float16` | `float16` / `int8` / `float32`      |

### `align` 命令行参考

```
asrlabs align <音频文件> [参考文件] [选项]
```

| 选项            | 默认值        | 说明                                           |
| --------------- | ------------- | ---------------------------------------------- |
| `参考文件`      | —             | `.json` / `.srt` / `.vtt` / `.txt`（自动检测） |
| `-t, --text`    | —             | 直接指定文本（与参考文件互斥）                 |
| `-l, --lang`    | `auto`        | ISO 639-1 语言代码                              |
| `--aligner`     | `qwen3_align` | 对齐器名称                                     |
| `--model-path`  | `""`          | 对齐模型路径                                   |
| `-d, --dir`     | `""`          | 输出目录                                       |
| `-o, --output`  | `""`          | 输出文件名 stem                                |
| `-c, --config`  | —             | 配置文件路径                                   |
| `--device`      | `auto`        | `cuda` / `cpu` / `vulkan` / `auto`             |

### `convert` 命令行参考

```
asrlabs convert <JSON 文件|目录> [选项]
```

| 选项           | 默认值   | 说明                                                    |
| -------------- | -------- | ------------------------------------------------------- |
| `-f, --format` | `srt`    | 输出格式，逗号分隔: `srt`, `lrc`, `txt`                   |
| `-l, --logic`  | `pysbd`  | 分句逻辑: `pysbd`（语义分句）/ `punct`（遇标点切分）      |
| `-d, --dir`    | `""`     | 输出目录（空=与 JSON 同目录）                            |
| `-o, --output` | `""`     | 输出文件名 stem（不含扩展名）                            |
| `--batch`      | —        | 批量处理目录下所有 `.json` 文件                           |

### 其他命令

```bash
asrlabs list transcribers   # 列出所有听写引擎
asrlabs list aligners       # 列出所有对齐器
asrlabs init                # 生成配置模板
```

## 格式转换

`convert` 命令从 transcribe/align 产出的 JSON 文件生成字幕或纯文本。

### 输出格式

| 格式  | 说明                              | 需要时间戳 |
| ----- | --------------------------------- | ---------- |
| `srt` | 标准 SRT 字幕，带序号 + 时间戳 + 文本 | 是         |
| `lrc` | LRC 歌词格式，`[mm:ss.xx]文本`      | 是         |
| `txt` | 纯文本，一句一行                    | 否         |

### 分句逻辑

| 选项       | 说明                                                       |
| ---------- | ---------------------------------------------------------- |
| `pysbd` —   | 基于 [pysbd](https://github.com/nipunsadvilkar/pySBD)，按语义边界分句，自动匹配语言 |
| `punct` — | 遇标点即切分（Unicode P* 类别），每个标点符号独立成段       |

分句时自动保留词级时间戳：通过字符位置 → Word 索引映射，确保每个句子精确对应音频片段。

## 输出控制

`-d` 和 `-o` 分别控制输出目录和文件名：

```bash
# 与音频同目录（默认）
asrlabs transcribe audio.wav -m whisper
# → audio.json

# 指定目录
asrlabs transcribe audio.wav -m whisper -d ./results
# → ./results/audio.json

# 指定文件名
asrlabs transcribe audio.wav -m whisper -o transcript
# → transcript.json

# 同时指定
asrlabs transcribe audio.wav -m whisper -d ./results -o transcript
# → ./results/transcript.json
```

## 优化特性

### 标点归一化

自动根据语言调整标点风格：

- **日语/中文**：全角 `。！？、`
- **英语/西方**：半角 `. ! ? ,`

无需额外配置，在输出前自动处理。

### VAD 分段策略

两步式智能分段：

1. VAD 检测 → 合并间距 < 2.0s 的语音区 → 每块最长 30s
2. 块间 0.5s 重叠避免边界截断：避免数百次逐句模型调用

### 段间上下文

Whisper/Faster-Whisper 逐段听写时自动传递段末文本作为 `initial_prompt`，减少边界吞音和断裂。

### 语言代码规范

所有命令的 `-l/--lang` 参数统一使用 **ISO 639-1** 代码（`zh` / `en` / `ja` / `ko` ...），特殊值 `auto` 启用自动检测。CLI 入口处会校验代码有效性，无效代码直接报错。

JSON 输出中的 `language` 字段也固定为 ISO 639-1 格式，确保各后端产出一致。

## 项目结构

```
ASRLabs/
├── asrlabs/                     # 核心包
│   ├── cli.py                   # Click 命令行
│   ├── config.py                # YAML 配置解析
│   ├── models.py                # 数据模型 (Word, Segment, TranscriptionResult)
│   ├── transcribe/              # 听写后端（策略模式）
│   │   ├── base.py              #   BaseTranscriber + 注册表
│   │   ├── whisper.py           #   OpenAI Whisper (stable-ts)
│   │   ├── faster_whisper.py    #   Faster Whisper (CTranslate2)
│   │   ├── qwen3_asr.py         #   Qwen3 ASR
│   │   ├── granite_speech.py    #   IBM Granite Speech
│   │   ├── cohere.py            #   Cohere Transcribe
│   │   ├── kotoba.py            #   Kotoba Whisper
│   │   └── ark_asr.py           #   ARK-ASR (transformers + chat template)
│   ├── align/                   # 对齐后端
│   │   ├── base.py              #   BaseAligner + 注册表
│   │   ├── whisper_align.py     #   Whisper 对齐器
│   │   ├── qwen3_align.py       #   Qwen3 Forced Aligner
│   │   └── ctc_align.py         #   CTC Forced Aligner (MMS/Wav2Vec2)
│   ├── pipeline/                # 编排层
│   │   ├── runner.py            #   Runner 主编排器
│   │   └── preprocess.py        #   重采样 + Silero VAD
│   └── utils/
│       ├── audio.py             #   音频加载 (soundfile)
│       ├── convert.py           #   分句 + 格式转换
│       ├── formats.py           #   SRT/JSON 解析
│       ├── language.py          #   语言代码统一工具 (ISO 639-1)
│       └── postprocess.py       #   标点归一化
├── sample_project/              # 示例项目
├── tests/                       # 单元测试
├── pyproject.toml
├── requirements.txt
└── README.md
```

## 扩展开发

新增听写引擎只需两步：

```python
# asrlabs/transcribe/my_model.py
from asrlabs.transcribe.base import BaseTranscriber, register_transcriber

@register_transcriber
class MyModelTranscriber(BaseTranscriber):
    name = "my-model"
    display_name = "My ASR Model"
    supports_timestamps = False
    recommended_aligner = "qwen3_align"

    def load_model(self) -> None: ...
    def transcribe(self, audio, **kwargs) -> TranscriptionResult: ...
```

然后在 `asrlabs/transcribe/__init__.py` 导入即可。

## 依赖环境

| 依赖               | 版本           | 说明                                      |
| ------------------ | -------------- | ----------------------------------------- |
| Python             | >= 3.10        |                                           |
| PyTorch            | >= 2.12        | CUDA 13.0 推荐                            |
| transformers       | >= 4.52, < 5.0 | Qwen3 ASR 兼容性约束                      |
| stable-ts          | >= 2.17        | Whisper / Faster Whisper 后端             |
| faster-whisper     | >= 1.0         | CTranslate2 引擎                          |
| qwen-asr           | >= 0.0.6       | Qwen3 ASR + ForcedAligner                 |
| pysbd              | >= 0.3.4       | 语义分句（convert 命令）                  |
| soundfile          | >= 0.12        | 音频 I/O                                  |
| nvidia-cublas-cu12 | —              | CUDA 13 下为 CTranslate2 提供 CUDA 12 DLL |
| ctc-forced-aligner | optional `[ctc]` | CTC 对齐后端, 需 ffmpeg + C++ 编译环境    |
| uroman / nltk / Unidecode | optional `[ctc]` | CTC 对齐器的罗马化与文本规范化依赖  |

> `transformers` 锁定 `< 5.0` 因 `qwen-asr 0.0.6` 不兼容 5.x 的配置与生成 API。需要 `transformers 5.x` 的后端可能难以实现。

## 致谢

感谢以下开源项目提供的相关技术：

- [GalTransl](https://github.com/GalTransl/GalTransl) — 项目灵感，架构设计
- [Silero VAD](https://github.com/snakers4/silero-vad) — 语音活动检测
- [pySBD](https://github.com/nipunsadvilkar/pySBD) — 多语言句子边界检测
- [stable-ts](https://github.com/jianfch/stable-ts) — Whisper 词级时间戳增强
- [ctc-forced-aligner](https://github.com/MahmoudAshraf97/ctc-forced-aligner) — Wav2Vec2/Hubert/MMS CTC 强制对齐
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) — CTranslate2 加速推理
- [Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR) — 多语言 ASR + ForcedAligner
- [Cohere Transcribe](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026) — Conformer-based ASR
- [IBM Granite Speech](https://huggingface.co/ibm-granite/granite-speech-4.1-2b) — IBM 语音模型
- [Kotoba Whisper v2.2](https://huggingface.co/kotoba-tech/kotoba-whisper-v2.2) — 日语特化 Distil-Whisper
- [ARK-ASR](https://huggingface.co/AutoArk-AI/ARK-ASR-3B) — 3B 多语言 ASR (Whisper 编码器 + Qwen 解码器)

## License

此项目使用 [MIT License](LICENSE) 开源。
