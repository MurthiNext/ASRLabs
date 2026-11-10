"""pytest 共享 fixtures"""

import pytest
from pathlib import Path


@pytest.fixture
def sample_audio_dir():
    """示例音频目录"""
    return Path(__file__).parent / "fixtures"
