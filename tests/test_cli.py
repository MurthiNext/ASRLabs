"""CLI 测试"""

from click.testing import CliRunner
from asrlabs.cli import main


def test_main_help():
    """测试主命令帮助"""
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "ASRLabs" in result.output


def test_transcribe_help():
    """测试 transcribe 子命令帮助"""
    runner = CliRunner()
    result = runner.invoke(main, ["transcribe", "--help"])
    assert result.exit_code == 0
    assert "--model" in result.output


def test_align_help():
    """测试 align 子命令帮助"""
    runner = CliRunner()
    result = runner.invoke(main, ["align", "--help"])
    assert result.exit_code == 0


def test_list_transcribers():
    """测试列出听写后端"""
    runner = CliRunner()
    result = runner.invoke(main, ["list", "transcribers"])
    assert result.exit_code == 0
    assert "whisper" in result.output.lower() or "Whisper" in result.output


def test_list_aligners():
    """测试列出对齐后端"""
    runner = CliRunner()
    result = runner.invoke(main, ["list", "aligners"])
    assert result.exit_code == 0
    assert "align" in result.output.lower()


def test_init_generates_config():
    """测试 init 生成配置文件"""
    import os
    import tempfile
    from pathlib import Path

    runner = CliRunner()
    orig_cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as tmp:
        try:
            os.chdir(tmp)
            result = runner.invoke(main, ["init"])
            assert result.exit_code == 0
            assert Path("config.yaml").exists()
            content = Path("config.yaml").read_text()
            assert "transcriber:" in content
            assert "whisper-base" in content
        finally:
            os.chdir(orig_cwd)


def test_align_missing_reference():
    """测试对齐缺少参考文件时报错"""
    runner = CliRunner()
    result = runner.invoke(main, ["align", "nonexistent.wav"])
    assert result.exit_code != 0


def test_transcribe_missing_file():
    """测试听写缺少文件时报错"""
    runner = CliRunner()
    result = runner.invoke(main, ["transcribe", "nonexistent.wav"])
    assert result.exit_code != 0
