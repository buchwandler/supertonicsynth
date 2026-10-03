import numpy as np
import pytest

from supertonicsynth.config import MAX_SEED, MAX_SPEED, MAX_STEPS, MIN_SPEED, MIN_STEPS
from supertonicsynth.errors import (
    EmptyTextError,
    InvalidGenerationConfigError,
    InvalidLanguageError,
    InvalidRequestError,
    SynthesisInputTooLongError,
)
from supertonicsynth.types import (
    AtomicSynthesisResult,
    GenerationConfig,
    RequestMeasure,
    SynthesisConfig,
    SynthesisRequest,
    SynthesisResult,
)
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


def test_synthesis_request_validation():
    request = SynthesisRequest(id="segment-1", text="Hello.", language="en")
    assert (request.id, request.text, request.language) == ("segment-1", "Hello.", "en")
    assert SynthesisRequest(id="x", text="Hello.", language="na").language == "na"

    with pytest.raises(InvalidRequestError, match="id"):
        SynthesisRequest(id="  ", text="Hello.", language="en")
    with pytest.raises(InvalidRequestError, match="id"):
        SynthesisRequest(id=1, text="Hello.", language="en")
    with pytest.raises(EmptyTextError, match="string"):
        SynthesisRequest(id="x", text=1, language="en")
    with pytest.raises(EmptyTextError, match="empty"):
        SynthesisRequest(id="x", text=" \n", language="en")
    with pytest.raises(InvalidLanguageError, match="non-empty"):
        SynthesisRequest(id="x", text="Hello.", language=" ")
    with pytest.raises(InvalidLanguageError, match="unsupported"):
        SynthesisRequest(id="x", text="Hello.", language="xx")


def test_generation_config_defaults_and_bounds():
    assert GenerationConfig() == GenerationConfig(steps=5, speed=1.05, seed=None)
    assert GenerationConfig(steps=MIN_STEPS, speed=MIN_SPEED, seed=0).seed == 0
    assert GenerationConfig(steps=MAX_STEPS, speed=MAX_SPEED, seed=MAX_SEED).steps == MAX_STEPS


@pytest.mark.parametrize(
    "kwargs",
    [
        {"steps": True},
        {"steps": 1.5},
        {"steps": MIN_STEPS - 1},
        {"steps": MAX_STEPS + 1},
        {"speed": True},
        {"speed": float("nan")},
        {"speed": float("inf")},
        {"speed": MIN_SPEED - 0.01},
        {"speed": MAX_SPEED + 0.01},
        {"seed": True},
        {"seed": -1},
        {"seed": MAX_SEED + 1},
        {"seed": 1.5},
    ],
)
def test_generation_config_rejects_invalid_values(kwargs):
    with pytest.raises(InvalidGenerationConfigError):
        GenerationConfig(**kwargs)


def test_atomic_result_preserves_request_and_audio_helpers(tmp_path):
    source_metadata = {"voice": "F1"}
    result = AtomicSynthesisResult(
        id="segment-1",
        audio=np.array([0.0, 0.5, -0.5, 1.0], dtype=np.float64),
        sample_rate=4,
        text="Exact source text.",
        language="en",
        warnings=("notice",),
        metadata=source_metadata,
    )
    source_metadata["voice"] = "M1"

    assert result.id == "segment-1"
    assert result.text == "Exact source text."
    assert result.language == "en"
    assert result.audio.dtype == np.float32
    assert result.audio.ndim == 1
    assert result.metadata == {"voice": "F1"}
    assert result.duration_seconds == 1.0
    assert result.pcm16().dtype == np.int16
    path = result.write_wav(tmp_path / "nested" / "out.wav")
    assert path.is_file()
    assert path.stat().st_size > 44


@pytest.mark.parametrize(
    "audio",
    [np.array([], dtype=np.float32), np.zeros((2, 2), dtype=np.float32)],
)
def test_atomic_result_rejects_invalid_shape(audio):
    with pytest.raises(InvalidRequestError, match="audio"):
        AtomicSynthesisResult("x", audio, 1000, "text", "en")


@pytest.mark.parametrize("audio", [np.array([np.nan]), np.array([np.inf])])
def test_atomic_result_rejects_non_finite_audio(audio):
    with pytest.raises(InvalidRequestError, match="finite"):
        AtomicSynthesisResult("x", audio, 1000, "text", "en")


@pytest.mark.parametrize("sample_rate", [0, -1, True, 1.5])
def test_atomic_result_rejects_invalid_sample_rate(sample_rate):
    with pytest.raises(InvalidRequestError, match="sample_rate"):
        AtomicSynthesisResult("x", np.ones(2), sample_rate, "text", "en")


def test_atomic_result_validates_other_fields():
    with pytest.raises(InvalidRequestError, match="id"):
        AtomicSynthesisResult(" ", np.ones(2), 1000, "text", "en")
    with pytest.raises(InvalidRequestError, match="text"):
        AtomicSynthesisResult("x", np.ones(2), 1000, 1, "en")
    with pytest.raises(InvalidRequestError, match="language"):
        AtomicSynthesisResult("x", np.ones(2), 1000, "text", " ")
    with pytest.raises(InvalidRequestError, match="warnings"):
        AtomicSynthesisResult("x", np.ones(2), 1000, "text", "en", warnings=["bad"])
    with pytest.raises(InvalidRequestError, match="metadata"):
        AtomicSynthesisResult("x", np.ones(2), 1000, "text", "en", metadata=None)


def test_request_measure_derives_fit_and_validates_consistency():
    assert RequestMeasure(amount=4, maximum=5).fits is True
    assert RequestMeasure(amount=6, maximum=5).fits is False
    unknown = RequestMeasure(amount=6, maximum=None)
    assert (unknown.unit, unknown.fits) == ("tokens", None)
    with pytest.raises(InvalidRequestError, match="fits"):
        RequestMeasure(amount=6, maximum=5, fits=True)


def test_input_too_long_error_exposes_structured_capacity():
    error = SynthesisInputTooLongError(
        text_length=20,
        token_count=12,
        max_tokens=10,
        model_id="supertonic-3",
    )
    assert (error.text_length, error.token_count, error.max_tokens, error.model_id) == (
        20,
        12,
        10,
        "supertonic-3",
    )
