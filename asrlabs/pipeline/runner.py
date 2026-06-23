"""Runner 主编排器——串联预处理 -> 听写 -> 对齐 -> 输出"""

import logging
from pathlib import Path
from asrlabs.config import ProjectConfig, load_config
from asrlabs.models import TranscriptionResult, Segment
from asrlabs.transcribe import get_transcriber
from asrlabs.align import get_aligner
from asrlabs.pipeline.preprocess import preprocess_audio

logger = logging.getLogger(__name__)


class Runner:
    """主编排器

    用法:
        runner = Runner("config.yaml")
        result = runner.run("audio.wav")
        results = runner.run_batch("./audio_dir/")

    device 和 compute_type 可通过参数覆盖 YAML 配置中的值，
    支持 "cuda"、"cpu"、"vulkan" 三种设备类型。
    """

    def __init__(
        self,
        config_path: str | Path,
        device: str | None = None,
        compute_type: str | None = None,
        model_path: str | None = None,
    ):
        """初始化 Runner

        Args:
            config_path: YAML 配置文件路径
            device: 覆盖设备类型 (cuda/cpu/vulkan)，None 则使用配置值
            compute_type: 覆盖计算精度 (float16/int8)，None 则使用配置值
            model_path: 覆盖模型路径，None 则使用配置值
        """
        self.cfg = load_config(config_path)

        # 允许 CLI 参数覆盖配置中的值
        tc = self.cfg.transcriber
        if device is not None:
            tc.device = device
        if compute_type is not None:
            tc.compute_type = compute_type
        if model_path is not None:
            tc.model_path = model_path

        self._setup_logging()

        # 构建配置字典（透传给后端）
        self._transcriber_config = {
            "model": tc.model,
            "model_path": tc.model_path,
            "device": tc.device,
            "compute_type": tc.compute_type,
            "language": tc.language,
            "beam_size": tc.beam_size,
            "extras": tc.extras,
        }

        al = self.cfg.aligner
        self._aligner_config = {
            "name": al.name,
            "device": tc.device,
            "extras": al.extras,
        }

        # 延迟创建实例
        self._transcriber = None
        self._aligner = None

    def run(self, audio: str | Path) -> TranscriptionResult:
        """处理单个音频文件

        Args:
            audio: 音频文件路径

        Returns:
            TranscriptionResult（含时间戳，如果配置了对齐器）
        """
        audio = Path(audio)
        if not audio.exists():
            raise FileNotFoundError(f"音频文件不存在: {audio}")

        logger.info(f"开始处理: {audio.name}")

        # 1. 预处理
        logger.info("预处理中...")
        segments = preprocess_audio(audio, self.cfg.audio)
        logger.info(f"音频分为 {len(segments)} 段")

        # 2. 听写（逐段，带上下文传递）
        transcriber = self._get_transcriber()
        all_results = []
        prev_text = ""
        for i, segment in enumerate(segments):
            logger.info(f"听写段 {i + 1}/{len(segments)}...")
            kwargs = {}
            if prev_text and transcriber.supports_timestamps:
                # Whisper/Faster-Whisper 支持 initial_prompt 提供上下文
                kwargs["initial_prompt"] = prev_text[-200:]  # 取最后 200 字符
            result = transcriber.transcribe(segment, **kwargs)
            if result.text.strip():
                prev_text = result.text
            all_results.append(result)

        # 合并多段结果
        combined = self._merge_results(all_results)

        # 3. 对齐（可选）
        if self._should_align(combined):
            logger.info("对齐中...")
            aligner = self._get_aligner()
            combined = aligner.align(
                str(audio), combined, language=self.cfg.transcriber.language
            )

        # 4. 标点归一化
        combined = self._normalize_punctuation(combined)

        # 5. 输出
        self._save_output(audio, combined)

        logger.info(f"完成: {audio.name}")
        return combined

    def run_batch(self, audio_dir: str | Path) -> list[TranscriptionResult]:
        """批量处理目录下所有音频文件

        Args:
            audio_dir: 音频目录路径

        Returns:
            TranscriptionResult 列表
        """
        audio_dir = Path(audio_dir)
        audio_extensions = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".wma"}
        audio_files = [
            f for f in audio_dir.iterdir()
            if f.suffix.lower() in audio_extensions
        ]

        results = []
        for audio_file in audio_files:
            try:
                result = self.run(audio_file)
                results.append(result)
            except Exception as e:
                logger.error(f"处理失败 {audio_file.name}: {e}")
        return results

    def _get_transcriber(self):
        if self._transcriber is None:
            self._transcriber = get_transcriber(
                self.cfg.transcriber.model, self._transcriber_config
            )
        return self._transcriber

    def _get_aligner(self):
        if self._aligner is None and self.cfg.aligner.name != "none":
            self._aligner = get_aligner(
                self.cfg.aligner.name, self._aligner_config
            )
        return self._aligner

    def _normalize_punctuation(self, result: TranscriptionResult) -> TranscriptionResult:
        """标点归一化"""
        from asrlabs.utils.postprocess import normalize_punctuation
        lang = result.language or self.cfg.transcriber.language or "auto"
        if lang == "auto":
            return result
        result.text = normalize_punctuation(result.text, lang)
        for seg in result.segments:
            seg.text = normalize_punctuation(seg.text, lang)
        return result

    def _should_align(self, result: TranscriptionResult) -> bool:
        """判断是否需要对齐"""
        name = self.cfg.aligner.name
        if name in ("none", "", None):
            return False
        if result.has_timestamps:
            return False
        return True

    def _merge_results(
        self, results: list[TranscriptionResult]
    ) -> TranscriptionResult:
        """合并多段听写结果"""
        if len(results) == 1:
            return results[0]

        all_text = " ".join(r.text for r in results)
        all_segments = []
        time_offset = 0.0
        detected_lang = results[0].language

        for r in results:
            for seg in r.segments:
                # 偏移词级时间戳
                offset_words = []
                for w in seg.words:
                    offset_words.append(type(w)(
                        text=w.text,
                        start=w.start + time_offset,
                        end=w.end + time_offset,
                        confidence=w.confidence,
                    ))
                all_segments.append(Segment(
                    text=seg.text,
                    start=seg.start + time_offset,
                    end=seg.end + time_offset,
                    words=offset_words,
                    confidence=seg.confidence,
                ))
            time_offset += r.duration

        return TranscriptionResult(
            text=all_text,
            segments=all_segments,
            language=detected_lang,
            duration=time_offset,
            model=results[0].model,
            has_timestamps=results[0].has_timestamps,
        )

    def _save_output(self, audio: Path, result: TranscriptionResult):
        """按配置写出所有格式

        - cfg.output.name 非空 → <dir>/<name>.<fmt>
        - cfg.output.name 为空 → <dir>/<audio.stem>.<fmt>
        """
        out_dir = Path(self.cfg.output.dir)
        if not out_dir.is_absolute():
            out_dir = audio.parent / out_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        stem = self.cfg.output.name or audio.stem
        for fmt in self.cfg.output.formats:
            if fmt == "srt" and not result.has_timestamps:
                logger.warning("跳过 SRT：结果不含时间戳（模型 %s 不产出时间戳）", result.model)
                continue
            out_path = out_dir / f"{stem}.{fmt}"
            result.save(out_path)
            logger.info(f"输出: {out_path}")

    def _setup_logging(self):
        """配置日志"""
        log_cfg = self.cfg.logging
        level = getattr(logging, log_cfg.level.upper(), logging.INFO)
        kwargs = {"level": level}
        if log_cfg.file:
            kwargs["filename"] = log_cfg.file
        logging.basicConfig(
            format="%(asctime)s [%(levelname)s] %(message)s",
            **kwargs,
        )
