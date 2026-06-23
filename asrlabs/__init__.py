"""ASRLabs - ASR 工具箱，整合多款开源 ASR 模型，开箱即用"""

__version__ = "0.1.0"


def _patch_qwen_asr_for_transformers_v5():
    """修复 qwen-asr 0.0.6 与 transformers >= 5.0 的兼容性

    transformers 5.x 中 check_model_inputs 签名从 func=None 变为 func（必选），
    qwen-asr 使用的 @check_model_inputs() 调用方式在新版中报错。
    此补丁将 @check_model_inputs() 替换为 @check_model_inputs。
    """
    import sys
    if "qwen_asr" in sys.modules:
        return  # 已导入，无需补丁

    # 在 qwen_asr 首次导入前，预先修补源文件
    import importlib.util
    spec = importlib.util.find_spec("qwen_asr")
    if spec is None or spec.origin is None:
        return
    import pathlib
    pkg_dir = pathlib.Path(spec.origin).parent
    target = pkg_dir / "core" / "transformers_backend" / "modeling_qwen3_asr.py"
    if not target.exists():
        return

    content = target.read_text(encoding="utf-8")
    if "@check_model_inputs()" in content:
        target.write_text(
            content.replace("@check_model_inputs()", "@check_model_inputs"),
            encoding="utf-8",
        )


_patch_qwen_asr_for_transformers_v5()
