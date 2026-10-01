import numpy as np
import pytest

from supertonicsynth.types import SynthesisConfig, SynthesisResult
from supertonicsynth.voice_level import VoiceLevelConfig


def test_config_validation():
    with pytest.raises(ValueError):
        SynthesisConfig(steps=0)
    with pytest.raises(ValueError):
        SynthesisConfig(speed=3.0)


@pytest.mark.parametrize("output_gain", [True, -1.0, float("nan"), float("inf")])
def test_config_rejects_invalid_output_gain(output_gain):
    with pytest.raises(ValueError, match="output_gain"):
        SynthesisConfig(output_gain=output_gain)


def test_config_validates_postprocessing_controls():
    with pytest.raises(ValueError, match="normalize_audio"):
        SynthesisConfig(normalize_audio=1)
    with pytest.raises(ValueError, match="voice_level"):
        SynthesisConfig(voice_level=object())
    config = SynthesisConfig(
        normalize_audio=False,
        output_gain=0.5,
        voice_level=VoiceLevelConfig(mode="calibrated", gain_db=-3.0),
    )
    assert config.normalize_audio is False
    assert config.output_gain == 0.5
    assert config.voice_level.gain_db == -3.0


def test_wav(tmp_path):
    result = SynthesisResult(np.zeros(100, dtype=np.float32), 1000, 0.1, 1)
    path = result.write_wav(tmp_path / "out.wav")
    assert path.stat().st_size > 44
