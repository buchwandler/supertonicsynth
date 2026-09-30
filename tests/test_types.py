import numpy as np
import pytest
from supertonicsynth.types import SynthesisConfig, SynthesisResult


def test_config_validation():
    with pytest.raises(ValueError):
        SynthesisConfig(steps=0)
    with pytest.raises(ValueError):
        SynthesisConfig(speed=3.0)


def test_wav(tmp_path):
    result = SynthesisResult(np.zeros(100, dtype=np.float32), 1000, 0.1, 1)
    path = result.write_wav(tmp_path / "out.wav")
    assert path.stat().st_size > 44
