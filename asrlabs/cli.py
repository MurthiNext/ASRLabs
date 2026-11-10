"""CLI 命令行接口——Click 实现"""

import sys
from pathlib import Path
import click

from asrlabs import __version__
from asrlabs.transcribe import list_transcribers, get_transcriber
from asrlabs.align import list_aligners, get_aligner
from asrlabs.pipeline.runner import Runner
from asrlabs.utils.formats import load_reference
from asrlabs.utils.language import validate_language_option
from asrlabs.utils.convert import apply_split_logic


@click.group()
@click.version_option(__version__, prog_name="asrlab")
def main():
    """ASRLabs - ASR 工具箱，整合多款开源 ASR 模型，开箱即用"""
    pass


@main.command()
@click.argument("audio", type=click.Path(exists=True))
@click.option(
    "-m", "--model", default="whisper",
    help="引擎名: whisper | faster-whisper | qwen3-asr | granite-speech | cohere-transcribe | kotoba-whisper | ark-asr"
)
@click.option(
    "--model-path", default=None,
    help="本地模型路径或 HuggingFace ID（空则使用引擎默认）"
)
@click.option("-l", "--lang", default="auto", callback=lambda ctx, param, value: validate_language_option(value),
              help="语言代码（ISO 639-1: zh/en/ja/ko ...，auto 自动检测）")
@click.option("--aligner", default=None, help="对齐器名称（whisper_align / qwen3_align）")
@click.option("-c", "--config", "config_path", default=None,
              type=click.Path(exists=True), help="配置文件路径")
@click.option("-d", "--dir", default=None, help="输出目录")
@click.option("-o", "--output", default=None, help="输出文件名 stem（不含扩展名）")
@click.option("--batch", is_flag=True, help="批量处理目录（仅允许 -d）")
@click.option(
    "--device",
    type=click.Choice(["cuda", "cpu", "vulkan", "auto"]),
    default=None,
    help="设备: cuda / cpu / vulkan / auto（默认 auto）"
)
@click.option(
    "--compute-type",
    type=click.Choice(["float16", "int8", "float32"]),
    default=None,
    help="计算精度: float16 / int8 / float32（默认 float16）"
)
def transcribe(audio, model, model_path, lang, aligner, config_path,
               dir, output, batch, device, compute_type):
    """听写——音频转文本

    AUDIO: 音频文件路径或目录（--batch 批量处理时）
    """
    if config_path:
        runner = Runner(config_path, device=device, compute_type=compute_type,
                        model_path=model_path)
        # 用 CLI 参数覆盖配置
        if model:
            runner.cfg.transcriber.model = model
        if lang != "auto":
            runner.cfg.transcriber.language = lang
        if dir:
            runner.cfg.output.dir = dir
        if output:
            runner.cfg.output.name = output
        if aligner:
            runner.cfg.aligner.name = aligner
        # 需要重建 transcriber（配置已变）
        runner._transcriber = None
    else:
        # 无配置文件，构建临时配置
        from asrlabs.config import (
            ProjectConfig, TranscriberConfig, AlignerConfig,
            AudioConfig, OutputConfig, LoggingConfig,
        )
        runner = Runner.__new__(Runner)
        runner.cfg = ProjectConfig(
            transcriber=TranscriberConfig(
                model=model,
                model_path=model_path or "",
                language=lang,
                device=device or "auto",
                compute_type=compute_type or "float16",
            ),
            aligner=AlignerConfig(name=aligner or "none"),
            audio=AudioConfig(),
            output=OutputConfig(
                dir=dir or "",
                name=output or "",
            ),
            logging=LoggingConfig(),
        )
        runner._transcriber = None
        runner._aligner = None
        runner._transcriber_config = {
            "model": model,
            "model_path": model_path or "",
            "device": device or "auto",
            "compute_type": compute_type or "float16", "language": lang,
            "beam_size": 5, "extras": {},
        }
        runner._aligner_config = {
            "name": aligner or "", "device": device or "auto", "extras": {},
        }
        runner._setup_logging = lambda: None

    audio_path = Path(audio)
    if batch or audio_path.is_dir():
        # 批量模式：只允许 -d，不允许 -o
        if output:
            raise click.UsageError("批量模式下仅允许 -d 指定目录，不支持 -o 指定文件名")
        results = runner.run_batch(audio_path)
        click.echo(f"批量处理完成，共 {len(results)} 个文件")
    else:
        result = runner.run(audio_path)
        click.echo(f"听写完成: {audio_path.name}")
        click.echo(f"语言: {result.language}")
        click.echo(f"文本: {result.text[:200]}{'...' if len(result.text) > 200 else ''}")


@main.command()
@click.argument("audio", type=click.Path(exists=True))
@click.argument("reference", required=False, type=click.Path(exists=True))
@click.option("-t", "--text", default=None, help="直接指定文本（与 REFERENCE 互斥）")
@click.option("-l", "--lang", default="auto", callback=lambda ctx, param, value: validate_language_option(value),
              help="语言代码（ISO 639-1，纯文本时必需）")
@click.option("--aligner", default="qwen3_align", help="对齐器: whisper_align / qwen3_align")
@click.option("-d", "--dir", default=None, help="输出目录")
@click.option("-o", "--output", default=None, help="输出文件名 stem（不含扩展名）")
@click.option("-c", "--config", "config_path", default=None,
              type=click.Path(exists=True), help="配置文件路径")
@click.option(
    "--model-path", default=None,
    help="对齐模型路径或 HF ID（空则使用引擎默认）"
)
@click.option(
    "--device",
    type=click.Choice(["cuda", "cpu", "vulkan", "auto"]),
    default=None,
    help="设备: cuda / cpu / vulkan / auto（默认 auto）"
)
def align(audio, reference, text, lang, aligner, dir, output, config_path, model_path, device):
    """对齐——为文本添加时间戳

    AUDIO: 音频文件路径
    REFERENCE: 参考文件（.json/.srt/.vtt/.txt），可用 -t 代替
    """
    if reference and text:
        raise click.UsageError("REFERENCE 和 -t/--text 互斥，请只指定一个")
    if not reference and not text:
        raise click.UsageError("需要指定 REFERENCE 文件或 -t/--text")

    # 获取文本
    if reference:
        try:
            result = load_reference(reference)
        except ValueError as e:
            click.echo(f"错误: {e}", err=True)
            sys.exit(1)
    else:
        from asrlabs.models import TranscriptionResult, Segment
        if lang == "auto":
            click.echo("警告: 纯文本模式建议用 -l 指定语言", err=True)
        result = TranscriptionResult(
            text=text,
            segments=[Segment(text, 0.0, 0.0)],
            language=lang,
            has_timestamps=False,
        )

    # 创建对齐器
    if config_path:
        from asrlabs.config import load_config
        cfg = load_config(config_path)
        aligner_config = {
            "name": aligner,
            "model_path": model_path or "",
            "device": device or cfg.transcriber.device,
            "extras": cfg.aligner.extras,
        }
    else:
        aligner_config = {
            "name": aligner,
            "model_path": model_path or "",
            "device": device or "auto",
            "extras": {},
        }

    al = get_aligner(aligner, aligner_config)
    if al is None:
        click.echo("错误: 对齐器不存在", err=True)
        sys.exit(1)

    click.echo(f"使用对齐器: {aligner}")
    aligned = al.align(audio, result, language=lang if lang != "auto" else None)

    from asrlabs.utils.postprocess import normalize_result
    lang_tag = lang if lang != "auto" else (aligned.language if aligned.language != "auto" else None)
    aligned = normalize_result(aligned, lang_tag or "auto")

    # 输出 JSON
    if dir or output:
        out_dir = Path(dir) if dir else Path(audio).parent
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = output or Path(audio).stem
        out_path = out_dir / f"{stem}.json"
    else:
        out_path = Path(audio).with_suffix(".aligned.json")
    aligned.save(out_path)
    click.echo(f"输出: {out_path}")


@main.command()
@click.argument("json_file", type=click.Path(exists=True))
@click.option(
    "-f", "--format", "formats", default="srt",
    help="输出格式，逗号分隔: srt,lrc,txt"
)
@click.option("-d", "--dir", default=None, help="输出目录")
@click.option("-o", "--output", default=None, help="输出文件名 stem（不含扩展名）")
@click.option("--batch", is_flag=True, help="批量处理目录（仅允许 -d）")
@click.option(
    "-l", "--logic",
    type=click.Choice(["pysbd", "punct"]),
    default="pysbd",
    help="分句逻辑: pysbd（语义分句）/ punct（遇标点切分）"
)
@click.option(
    "--max-chars", default=40, show_default=True,
    help="单条字幕最大字符数（超过则强制截断，0=不限制）"
)
def convert(json_file, formats, dir, output, batch, logic, max_chars):
    """格式转换——将 JSON 结果转为其他格式

    JSON_FILE: JSON 结果文件路径，--batch 时为目录
    """
    fmt_list = [f.strip() for f in formats.split(",")]
    invalid = [f for f in fmt_list if f not in ("srt", "lrc", "txt", "json")]
    if invalid:
        raise click.UsageError(f"不支持的格式: {', '.join(invalid)}，支持 srt/lrc/txt/json")

    json_path = Path(json_file)
    if batch or json_path.is_dir():
        if output:
            raise click.UsageError("批量模式下仅允许 -d 指定目录，不支持 -o 指定文件名")
        json_files = sorted(json_path.glob("*.json"))
        if not json_files:
            click.echo(f"目录 {json_path} 中没有 JSON 文件")
            return
        for jf in json_files:
            _convert_single(jf, fmt_list, dir, output, logic, max_chars)
        click.echo(f"批量转换完成，共 {len(json_files)} 个文件")
    else:
        _convert_single(json_path, fmt_list, dir, output, logic, max_chars)


def _convert_single(json_path, fmt_list, dir, output, logic, max_chars=40):
    """单文件转换"""
    result = _load_result(json_path)
    result = apply_split_logic(result, logic, max_chars=max_chars)

    out_dir = Path(dir) if dir else json_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = output or json_path.stem

    for fmt in fmt_list:
        if fmt in ("srt", "lrc") and not result.has_timestamps:
            click.echo(f"跳过 {fmt}: {json_path.name} 不含有效时间戳")
            continue
        out_path = out_dir / f"{stem}.{fmt}"
        result.save(out_path, fmt=fmt if fmt == "json" else None)
        click.echo(f"输出: {out_path}")


def _load_result(json_path):
    """从 JSON 文件加载 TranscriptionResult"""
    import json
    from asrlabs.models import TranscriptionResult, Segment, Word

    data = json.loads(json_path.read_text(encoding="utf-8"))
    raw_segs = data.get("segments", [])
    segments = []
    for rs in raw_segs:
        words = []
        for rw in rs.get("words", []):
            words.append(Word(
                text=rw["text"],
                start=rw["start"],
                end=rw["end"],
                confidence=rw.get("confidence", 1.0),
            ))
        segments.append(Segment(
            text=rs["text"],
            start=rs.get("start", 0.0),
            end=rs.get("end", 0.0),
            words=words,
            confidence=rs.get("confidence", 0.0),
        ))
    return TranscriptionResult(
        text=data.get("text", ""),
        segments=segments,
        language=data.get("language", "auto"),
        duration=data.get("duration", 0.0),
        model=data.get("model", ""),
        has_timestamps=data.get("has_timestamps", False),
    )


@main.group()
def list():
    """列出可用模型"""
    pass


@list.command()
@click.option("--json", "as_json", is_flag=True, help="以 JSON 格式输出（供外部程序解析）")
def transcribers(as_json):
    """列出所有听写后端"""
    items = list_transcribers()
    if as_json:
        import json as _json
        click.echo(_json.dumps(items, ensure_ascii=False, indent=2))
        return
    for t in items:
        ts = "✅" if t["supports_timestamps"] else "❌"
        click.echo(f"  {t['name']:30s} {t['display_name']:40s} 时间戳: {ts}")


@list.command()
@click.option("--json", "as_json", is_flag=True, help="以 JSON 格式输出（供外部程序解析）")
def aligners(as_json):
    """列出所有对齐后端"""
    items = list_aligners()
    if as_json:
        import json as _json
        click.echo(_json.dumps(items, ensure_ascii=False, indent=2))
        return
    for a in items:
        click.echo(f"  {a['name']:20s} {a['display_name']}")


@main.command()
def init():
    """生成示例配置文件"""
    config_content = """# ASRLabs 项目配置
# 生成方式: asrlab init
#
# 可用引擎:
#   whisper          — OpenAI Whisper (stable-ts), --model-path 指定尺寸/路径
#   faster-whisper   — Faster Whisper (CTranslate2, 支持 vulkan)
#   qwen3-asr        — Qwen3 ASR, --model-path 指定 HF ID 或本地路径
#   granite-speech   — IBM Granite Speech, --model-path 指定 HF ID 或本地路径
#   kotoba-whisper   — Kotoba Whisper v2.2 日语听写（transformers pipeline）
#   ark-asr          — ARK-ASR 3B 多语言听写（transformers, chat template）

# ── 听写模型 ──
transcriber:
  model: whisper                   # 引擎名: whisper | faster-whisper | qwen3-asr | granite-speech | kotoba-whisper | ark-asr
  model_path: ""                   # 本地模型路径（空则使用引擎默认）
  device: auto                     # cuda | cpu | vulkan | auto
  compute_type: float16            # float16 | int8 | float32（仅 faster-whisper / vulkan）
  language: auto                   # auto | zh | en | ja | ko ...
  beam_size: 5
  extras: {}                       # 透传底层库特有参数

# ── 对齐器（可选） ──
aligner:
  name: none                       # whisper_align | qwen3_align | none
  extras: {}

# ── 音频预处理 ──
audio:
  sample_rate: 16000
  vad: true
  max_segment_length: 30.0
  min_silence_dur: 0.5

# ── 输出 ──
output:
  dir: ""                           # 输出目录（空=同音频目录，-d 覆盖）
  name: ""                          # 输出文件名 stem（-o 覆盖，空则用输入文件名）
  keep_segments: false
# transcribe/align 仅输出 JSON；格式转换请使用 asrlab convert

# ── 日志 ──
logging:
  level: INFO
  file: null
"""
    out = Path("config.yaml")
    if out.exists():
        click.confirm("config.yaml 已存在，是否覆盖？", abort=True)
    out.write_text(config_content, encoding="utf-8")
    click.echo(f"配置模板已生成: {out}")


if __name__ == "__main__":
    main()
