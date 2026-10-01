import io
import wave

import numpy as np
import pytest

from supertonicsynth.audio import (
    AUDIO_PEAK_EPSILON,
    audio_to_int16_bytes,
    finish_audio,
    float_to_int16,
    prepare_audio,
    write_wav,
)
from supertonicsynth.errors import ModelInferenceError


def test_prepare_audio_normalizes_peak_and_preserves_float32():
    audio = np.array([0.2, -0.4], dtype=np.float64)

    result = prepare_audio(audio, normalize=True)

    assert result.dtype == np.float32
    np.testing.assert_allclose(result, [0.5, -1.0])


def test_prepare_audio_keeps_near_silence_safe():
    audio = np.array([AUDIO_PEAK_EPSILON / 2, -AUDIO_PEAK_EPSILON / 2], dtype=np.float32)

    result = prepare_audio(audio, normalize=True)

    np.testing.assert_array_equal(result, np.zeros_like(audio))


def test_prepare_audio_without_normalization_leaves_samples_unchanged():
    audio = np.array([0.2, -0.4], dtype=np.float32)

    result = prepare_audio(audio, normalize=False)

    np.testing.assert_array_equal(result, audio)


@pytest.mark.parametrize(
    "audio",
    [np.array([[0.1, 0.2]], dtype=np.float32), np.array([np.nan], dtype=np.float32)],
)
def test_prepare_audio_rejects_invalid_shape_or_non_finite_samples(audio):
    with pytest.raises(ModelInferenceError):
        prepare_audio(audio, normalize=False)


def test_finish_audio_applies_gain_and_clips():
    audio = np.array([0.2, -0.4], dtype=np.float32)

    result = finish_audio(audio, output_gain=2.0)

    np.testing.assert_array_equal(result, np.array([0.4, -0.8], dtype=np.float32))
    clipped = finish_audio(np.array([0.8, -0.8], dtype=np.float32), output_gain=2.0)
    np.testing.assert_array_equal(clipped, np.array([1.0, -1.0], dtype=np.float32))


@pytest.mark.parametrize("gain", [True, -1.0, float("nan"), float("inf")])
def test_finish_audio_rejects_invalid_output_gain(gain):
    with pytest.raises(ModelInferenceError, match="output_gain"):
        finish_audio(np.array([0.1], dtype=np.float32), output_gain=gain)


def test_pcm16_conversion_clips_before_converting():
    audio = np.array([-2.0, 0.0, 2.0], dtype=np.float32)

    result = float_to_int16(audio)

    np.testing.assert_array_equal(result, np.array([-32767, 0, 32767], dtype=np.int16))
    assert audio_to_int16_bytes(audio) == result.tobytes()


def test_write_wav_writes_mono_pcm16():
    stream = io.BytesIO()
    write_wav(stream, np.array([0.0, 0.5], dtype=np.float32), 22050)
    stream.seek(0)

    with wave.open(stream, "rb") as wav:
        assert wav.getnchannels() == 1
        assert wav.getsampwidth() == 2
        assert wav.getframerate() == 22050
        assert wav.getnframes() == 2


def test_write_wav_rejects_invalid_sample_rate():
    with pytest.raises(ValueError, match="sample_rate"):
        write_wav(io.BytesIO(), np.array([0.0], dtype=np.float32), 0)
