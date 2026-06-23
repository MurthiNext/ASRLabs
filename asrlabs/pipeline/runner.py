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
    ):
        """初始化 Runner

        Args:
            config_path: YAML 配置文件路径
            device: 覆盖设备类型 (cuda/cpu/vulkan)，None 则使用配置值
            compute_type: 覆盖计算精度 (float16/int8)，None 则使用配置值
        """
        self.cfg = load_config(config_path)

        # 允许 CLI 参数覆盖配置中的 device / compute_type
        tc = self.cfg.transcriber
        if device is not None:
            tc.device = device
        if compute_type is not None:
            tc.compute_type = compute_type

        self._setup_logging()

        # 构建配置字典（透传给后端）
        self._transcriber_config = {
            "model": tc.model,
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

        # 2. 听写
        transcriber = self._get_transcriber()
        all_results = []
        for i, segment in enumerate(segments):
            logger.info(f"听写段 {i + 1}/{len(segments)}...")
            result = transcriber.transcribe(segment)
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

        # 4. 输出
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

    def _should_align(self, result: TranscriptionResult) -> bool:
        """判断是否需要对齐"""
        if self.cfg.aligner.name == "none":
            return False
        if result.has_timestamps and self.cfg.aligner.name == "":
            return False
        if self._get_aligner() is not None:
            return True
        return False

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
                all_segments.append(Segment(
                    text=seg.text,
                    start=seg.start + time_offset,
                    end=seg.end + time_offset,
                    words=seg.words,  # 词级时间戳也需偏移，暂略
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
        """按配置写出所有格式"""
        output_dir = Path(self.cfg.output.dir)
        if not output_dir.is_absolute():
            output_dir = audio.parent / output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        stem = audio.stem
        for fmt in self.cfg.output.formats:
            out_path = output_dir / f"{stem}.{fmt}"
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
