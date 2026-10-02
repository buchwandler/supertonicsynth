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


@pytest.mark.parametrize(
    "kwargs",
    [
        {"steps": True},
        {"steps": 1.5},
        {"speed": True},
        {"speed": float("nan")},
        {"speed": float("inf")},
        {"speed": 0.6},
        {"speed": 2.1},
        {"max_chunk_length": True},
        {"max_chunk_length": 10.5},
        {"max_chunk_length": 9},
        {"silence_duration": True},
        {"silence_duration": float("nan")},
        {"silence_duration": float("inf")},
        {"silence_duration": -0.1},
        {"seed": True},
        {"seed": -1},
        {"seed": 2**32},
        {"seed": 1.5},
    ],
)
def test_synthesis_config_rejects_invalid_values(kwargs):
    with pytest.raises(ValueError):
        SynthesisConfig(**kwargs)
