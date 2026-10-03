from types import SimpleNamespace

import numpy as np
import pytest

import supertonicsynth.runtime as runtime_module
from supertonicsynth._onnxvoice import ResolvedSupertonicBundle
from supertonicsynth.errors import (
    ClosedRuntimeError,
    InvalidRequestError,
    ModelInferenceError,
    SynthesisInputTooLongError,
)
from supertonicsynth.runtime import SupertonicRuntime
from supertonicsynth.style import VoiceStyle
from supertonicsynth.types import GenerationConfig, SynthesisRequest
from supertonicsynth.voice_level import VoiceLevelConfig


class FakeRuntime:
    def __init__(self, audio=None, sample_rate=1000):
        self.calls = []
        self.closed = False
        self.audio = np.ones(20, dtype=np.float32) * 0.25 if audio is None else audio
        self.sample_rate = sample_rate

    def infer(self, token_ids, **kwargs):
        self.calls.append((token_ids, kwargs))
        return SimpleNamespace(audio=self.audio, sample_rate=self.sample_rate)

    def close(self):
        self.closed = True


def make_runtime(tmp_path, *, audio=None, sample_rate=1000, max_input_tokens=None):
    indexer = tmp_path / "unicode_indexer.json"
    indexer.write_text("[" + ",".join(str(i) for i in range(128)) + "]")
    config = tmp_path / "tts.json"
    config.write_text("{}")
    style_path = tmp_path / "F1.json"
    style_path.write_text(
        '{"style_ttl":{"dims":[1,2],"data":[[1,2]]},"style_dp":{"dims":[1,2],"data":[[3,4]]}}'
    )
    installation = SimpleNamespace(kind="bundle")
    metadata = {"managed": True}
    if max_input_tokens is not None:
        metadata["runtime"] = {"max_input_tokens": max_input_tokens}
    bundle = ResolvedSupertonicBundle(
        ref="supertonic:supertonic-3",
        bundle_id="supertonic-3",
        config_path=config,
        unicode_indexer_path=indexer,
        model_paths={},
        style_paths={"F1": style_path},
        sample_rate=1000,
        source_revision="catalog-revision",
        metadata=metadata,
        installation=installation,
    )
    fake = FakeRuntime(audio=audio, sample_rate=sample_rate)
    return SupertonicRuntime(fake, bundle), fake


def test_atomic_synthesis_encodes_once_infers_once_and_never_chunks(tmp_path, monkeypatch):
    runtime, fake = make_runtime(tmp_path)
    request = SynthesisRequest(id="seg-123", text="A caller-shaped request.", language="en")
    encode_calls = []
    original_encode = runtime.frontend.encode

    def encode(*args, **kwargs):
        encode_calls.append((args, kwargs))
        return original_encode(*args, **kwargs)

    def fail_chunking(*_args, **_kwargs):
        raise AssertionError("atomic synthesis must not call chunk_text")

    monkeypatch.setattr(runtime.frontend, "encode", encode)
    monkeypatch.setattr(runtime_module, "chunk_text", fail_chunking)
    result = runtime.synthesize(request, voice="F1", config=GenerationConfig(seed=123))

    assert len(encode_calls) == 1
    assert len(fake.calls) == 1
    assert fake.calls[0][1]["seed"] == 123
    assert result.id == request.id
    assert result.text == request.text
    assert result.language == request.language
    assert result.audio.ndim == 1
    assert result.metadata["token_count"] == len(fake.calls[0][0])
    assert result.metadata["voice_ref"] == "supertonic:supertonic-3/F1"
    assert "chunks" not in result.metadata
    assert "output_gain" not in result.metadata
    assert "normalize_audio" not in result.metadata


def test_measure_request_uses_frontend_without_inference(tmp_path):
    runtime, fake = make_runtime(tmp_path, max_input_tokens=100)
    request = SynthesisRequest(id="seg", text="Measure me.", language="en")

    measure = runtime.measure_request(request)

    assert measure.amount > 0
    assert measure.maximum == 100
    assert measure.unit == "tokens"
    assert measure.fits is True
    assert fake.calls == []


def test_atomic_synthesis_rejects_request_over_declared_capacity(tmp_path):
    runtime, fake = make_runtime(tmp_path, max_input_tokens=2)
    request = SynthesisRequest(id="seg", text="Too long.", language="en")

    measure = runtime.measure_request(request)
    assert measure.fits is False
    with pytest.raises(SynthesisInputTooLongError) as exc_info:
        runtime.synthesize(request, voice="F1")

    error = exc_info.value
    assert error.text_length == len(request.text)
    assert error.token_count == measure.amount
    assert error.max_tokens == 2
    assert error.model_id == "supertonic-3"
    assert fake.calls == []


def test_unknown_capacity_does_not_invent_a_fit_or_split(tmp_path):
    runtime, fake = make_runtime(tmp_path)
    request = SynthesisRequest(id="seg", text="No declared model limit.", language="en")

    assert runtime.max_input_tokens is None
    assert runtime.measure_request(request).fits is None
    result = runtime.synthesize(request, voice="F1")
    assert result.id == "seg"
    assert len(fake.calls) == 1


def test_atomic_synthesis_supports_custom_style_without_managed_identity(tmp_path):
    runtime, fake = make_runtime(tmp_path)
    custom = VoiceStyle(
        ttl=np.array([[1.0, 2.0]], dtype=np.float32),
        dp=np.array([[3.0, 4.0]], dtype=np.float32),
        name="F1",
    )

    result = runtime.synthesize(
        SynthesisRequest(id="custom", text="Custom style.", language="en"),
        voice=custom,
        voice_level=VoiceLevelConfig(mode="calibrated"),
    )

    assert len(fake.calls) == 1
    assert fake.calls[0][1]["style_ttl"] is custom.ttl
    assert result.metadata["voice"] == "F1"
    assert result.metadata["voice_ref"] is None
    assert result.metadata["voice_level"]["source"] == "missing_identity"


def test_atomic_synthesis_applies_explicit_voice_gain_without_mastering(tmp_path):
    runtime, fake = make_runtime(tmp_path, audio=np.array([0.25, -0.25], dtype=np.float32))

    result = runtime.synthesize(
        SynthesisRequest(id="gain", text="Apply static gain.", language="en"),
        voice="F1",
        voice_level=VoiceLevelConfig(gain_db=6.0),
    )

    np.testing.assert_allclose(result.audio, np.array([0.25, -0.25]) * 10 ** (6 / 20))
    assert len(fake.calls) == 1
    assert result.metadata["voice_level"]["source"] == "override"


@pytest.mark.parametrize(
    "audio,sample_rate",
    [
        (np.array([], dtype=np.float32), 1000),
        (np.zeros((2, 2), dtype=np.float32), 1000),
        (np.array([np.nan], dtype=np.float32), 1000),
        (np.array([np.inf], dtype=np.float32), 1000),
        (np.ones(2, dtype=np.float32), 22050),
    ],
)
def test_atomic_synthesis_rejects_invalid_runtime_audio(tmp_path, audio, sample_rate):
    runtime, _ = make_runtime(tmp_path, audio=audio, sample_rate=sample_rate)
    request = SynthesisRequest(id="bad-audio", text="Check audio.", language="en")

    with pytest.raises(ModelInferenceError):
        runtime.synthesize(request, voice="F1")


def test_atomic_synthesis_rejects_wrong_request_and_closed_runtime(tmp_path):
    runtime, _ = make_runtime(tmp_path)
    with pytest.raises(InvalidRequestError, match="SynthesisRequest"):
        runtime.synthesize("not a request")
    runtime.close()
    with pytest.raises(ClosedRuntimeError):
        runtime.measure_request(SynthesisRequest(id="seg", text="Closed.", language="en"))
    with pytest.raises(ClosedRuntimeError):
        runtime.synthesize(SynthesisRequest(id="seg", text="Closed.", language="en"), voice="F1")


def test_atomic_hash_includes_exact_source_text(tmp_path):
    runtime, _ = make_runtime(tmp_path)
    first = runtime.synthesize(
        SynthesisRequest(id="same", text="Exact text.", language="en"), voice="F1"
    )
    second = runtime.synthesize(
        SynthesisRequest(id="same", text="Exact text. ", language="en"), voice="F1"
    )

    assert first.text == "Exact text."
    assert second.text == "Exact text. "
    assert first.metadata["synthesis_hash"] != second.metadata["synthesis_hash"]
