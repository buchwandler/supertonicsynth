from types import SimpleNamespace

import numpy as np
import pytest

import supertonicsynth.voice_level as voice_level_module
from supertonicsynth._onnxvoice import (
    ResolvedSupertonicBundle,
    installation_to_bundle_info,
)
from supertonicsynth.errors import InvalidRequestError
from supertonicsynth.runtime import SupertonicRuntime
from supertonicsynth.style import VoiceStyle
from supertonicsynth.types import SynthesisConfig
from supertonicsynth.voice_level import (
    VoiceCalibrationCatalog,
    VoiceCalibrationKey,
    VoiceLevelConfig,
)


class FakeRuntime:
    def __init__(self, audio=None):
        self.calls = []
        self.closed = False
        self.audio = (
            np.ones(100, dtype=np.float32) * 0.1
            if audio is None
            else np.asarray(audio, dtype=np.float32)
        )

    def infer(self, token_ids, **kwargs):
        self.calls.append((token_ids, kwargs))
        return SimpleNamespace(audio=self.audio, sample_rate=1000)

    def close(self):
        self.closed = True


def make_runtime(tmp_path, *, audio=None, style_names=("M1",), managed=False, local=False):
    indexer = tmp_path / "unicode_indexer.json"
    indexer.write_text("[" + ",".join(str(i) for i in range(128)) + "]")
    config = tmp_path / "tts.json"
    config.write_text("{}")
    styles = {}
    for name in style_names:
        style = tmp_path / f"{name}.json"
        style.write_text(
            '{"style_ttl":{"dims":[1,2],"data":[[1,2]]},"style_dp":{"dims":[1,2],"data":[[3,4]]}}'
        )
        styles[name] = style
    installation = None
    metadata = {}
    if managed:
        installation = SimpleNamespace(kind="bundle", metadata={"managed": True})
        metadata = {"managed": True}
    if local:
        installation = SimpleNamespace(kind="external", metadata={"managed": False})
        metadata = {"managed": False, "external": True}
    bundle = ResolvedSupertonicBundle(
        "supertonic:st3" if managed else None,
        "supertonic-3",
        config,
        indexer,
        {},
        styles,
        1000,
        "source-revision",
        metadata,
        installation,
    )
    fake = FakeRuntime(audio)
    return SupertonicRuntime(fake, bundle), fake, bundle


def test_runtime_fake(tmp_path):
    runtime, fake, _ = make_runtime(tmp_path)
    result = runtime.synthesize_text("hello", voice="M1", language="en", seed=4)
    assert result.sample_rate == 1000
    assert result.audio.size == 100
    assert fake.calls[0][1]["seed"] == 4
    runtime.close()
    assert fake.closed


def test_runtime_rejects_derived_seed_overflow_before_inference(tmp_path):
    runtime, fake, _ = make_runtime(tmp_path)
    config = SynthesisConfig(max_chunk_length=10, seed=0xFFFFFFFF)

    with pytest.raises(InvalidRequestError, match="maximum seed"):
        runtime.synthesize_text("First. Second.", config=config)

    assert fake.calls == []


def test_managed_voice_identity_overrides_and_postprocessing(tmp_path):
    raw = np.array([0.2, -0.4], dtype=np.float32)
    runtime, _, bundle = make_runtime(tmp_path, audio=raw, style_names=("F1",), managed=True)
    config = SynthesisConfig(
        normalize_audio=True,
        output_gain=1.0,
        voice_level=VoiceLevelConfig(),
    )

    result = runtime.synthesize_text(
        "count to one",
        voice="F1",
        language="de",
        config=config,
        normalize_audio=False,
        output_gain=0.5,
        voice_level=VoiceLevelConfig(mode="calibrated", gain_db=-6.0),
    )

    expected = raw * 10 ** (-6 / 20) * 0.5
    np.testing.assert_allclose(result.audio, expected, rtol=1e-6)
    assert result.audio.dtype == np.float32
    assert result.metadata["bundle"] == "supertonic:supertonic-3"
    assert result.metadata["voice_ref"] == "supertonic:supertonic-3/F1"
    assert result.metadata["backing_ref"] == "supertonic:supertonic-3"
    assert result.metadata["voice_level"] == {
        "mode": "calibrated",
        "applied": True,
        "gain_db": -6.0,
        "source": "override",
        "calibration_key": "supertonic:supertonic-3/F1@de",
        "voice_ref": "supertonic:supertonic-3/F1",
        "language": "de",
        "reason": "an explicit gain_db override was selected",
        "catalog_revision": None,
    }
    assert runtime.last_voice_level_application is not None
    assert runtime.last_voice_level_application.key == VoiceCalibrationKey(
        "supertonic:supertonic-3/F1", "de"
    )
    assert config.normalize_audio is True
    assert config.output_gain == 1.0
    assert config.voice_level.mode == "off"
    identity = result.metadata["synthesis_identity"]
    assert identity["voice_ref"] == result.metadata["voice_ref"]
    assert identity["backing_ref"] == result.metadata["backing_ref"]
    assert identity["source_revision"] == "source-revision"
    assert identity["voice_level"]["calibration_key"] == "supertonic:supertonic-3/F1@de"
    assert len(result.metadata["synthesis_hash"]) == 64
    assert bundle.ref == "supertonic:st3"


def test_runtime_clips_after_static_gain_and_output_gain(tmp_path):
    runtime, _, _ = make_runtime(
        tmp_path,
        audio=np.array([0.8, -0.8], dtype=np.float32),
        style_names=("F1",),
        managed=True,
    )

    result = runtime.synthesize_text(
        "count to one",
        voice="F1",
        language="en",
        config=SynthesisConfig(
            normalize_audio=False,
            output_gain=2.0,
            voice_level=VoiceLevelConfig(mode="calibrated", gain_db=6.0),
        ),
    )

    np.testing.assert_array_equal(result.audio, np.array([1.0, -1.0], dtype=np.float32))


def test_unmanaged_and_custom_style_do_not_claim_catalog_identity(tmp_path):
    audio = np.array([0.2, -0.4], dtype=np.float32)
    runtime, _, _ = make_runtime(
        tmp_path, audio=audio, style_names=("F1",), managed=False, local=True
    )

    local_result = runtime.synthesize_text(
        "count to one",
        voice="F1",
        language="en",
        config=SynthesisConfig(
            normalize_audio=False,
            voice_level=VoiceLevelConfig(mode="calibrated"),
        ),
    )
    np.testing.assert_array_equal(local_result.audio, audio)
    assert local_result.metadata["voice_ref"] is None
    assert local_result.metadata["backing_ref"] is None
    assert local_result.metadata["voice_level"]["source"] == "missing_identity"

    custom = VoiceStyle(
        ttl=np.array([[1.0, 2.0]], dtype=np.float32),
        dp=np.array([[3.0, 4.0]], dtype=np.float32),
        name="F1",
    )
    custom_result = runtime.synthesize_text(
        "count to one",
        voice=custom,
        language="en",
        normalize_audio=False,
        voice_level=VoiceLevelConfig(mode="calibrated"),
    )
    assert custom_result.metadata["voice"] == "F1"
    assert custom_result.metadata["voice_ref"] is None
    assert custom_result.metadata["voice_level"]["source"] == "missing_identity"

    override_result = runtime.synthesize_text(
        "count to one",
        voice=custom,
        language="en",
        normalize_audio=False,
        voice_level=VoiceLevelConfig(mode="calibrated", gain_db=6.0),
    )
    np.testing.assert_allclose(override_result.audio, audio * 10 ** (6 / 20), rtol=1e-6)
    assert override_result.metadata["voice_level"]["source"] == "override"


def test_custom_style_name_never_claims_managed_calibration_identity(tmp_path):
    runtime, _, _ = make_runtime(
        tmp_path, audio=np.array([0.2, -0.4], dtype=np.float32), style_names=("F1",), managed=True
    )
    custom = VoiceStyle(
        ttl=np.array([[1.0, 2.0]], dtype=np.float32),
        dp=np.array([[3.0, 4.0]], dtype=np.float32),
        name="F1",
    )

    result = runtime.synthesize_text(
        "count to one",
        voice=custom,
        language="en",
        normalize_audio=False,
        voice_level=VoiceLevelConfig(mode="calibrated"),
    )

    assert result.metadata["voice_ref"] is None
    assert result.metadata["voice_level"]["source"] == "missing_identity"
    np.testing.assert_array_equal(result.audio, np.array([0.2, -0.4], dtype=np.float32))


def test_synthesis_hash_changes_with_catalog_revision(tmp_path, monkeypatch):
    runtime, _, _ = make_runtime(tmp_path, style_names=("F1",), managed=True)
    first_catalog = VoiceCalibrationCatalog(1, "bs1770", "test", -24.0, {}, {}, revision="a" * 64)
    second_catalog = VoiceCalibrationCatalog(1, "bs1770", "test", -24.0, {}, {}, revision="b" * 64)
    monkeypatch.setattr(voice_level_module, "default_voice_calibration", lambda: first_catalog)
    first = runtime.synthesize_text(
        "count to one",
        voice="F1",
        language="de",
        voice_level=VoiceLevelConfig(mode="calibrated"),
    )
    monkeypatch.setattr(voice_level_module, "default_voice_calibration", lambda: second_catalog)
    second = runtime.synthesize_text(
        "count to one",
        voice="F1",
        language="de",
        voice_level=VoiceLevelConfig(mode="calibrated"),
    )

    assert first.metadata["voice_level"]["catalog_revision"] == "a" * 64
    assert second.metadata["voice_level"]["catalog_revision"] == "b" * 64
    assert first.metadata["synthesis_hash"] != second.metadata["synthesis_hash"]


def test_synthesis_hash_includes_exact_source_text(tmp_path):
    runtime, _, _ = make_runtime(tmp_path, style_names=("F1",), managed=True)
    first = runtime.synthesize_text("count to one", voice="F1", language="en")
    second = runtime.synthesize_text("count to one ", voice="F1", language="en")

    assert first.metadata["synthesis_hash"] != second.metadata["synthesis_hash"]


def test_installed_bundle_ref_uses_canonical_installation_identity(tmp_path):
    indexer = tmp_path / "unicode_indexer.json"
    config = tmp_path / "tts.json"
    paths = {("config", None): config, ("unicode_indexer", None): indexer}
    for component in (
        "duration_predictor",
        "text_encoder",
        "vector_estimator",
        "vocoder",
    ):
        paths[("model", component)] = tmp_path / f"{component}.onnx"

    def artifact(role, component=None):
        return SimpleNamespace(path=paths[(role, component)])

    installation = SimpleNamespace(
        id="supertonic-3",
        system="supertonic",
        ref="supertonic:supertonic-3",
        kind="bundle",
        artifacts=(),
        metadata={"managed": True},
        sample_rate=1000,
        artifact=artifact,
    )

    resolved = installation_to_bundle_info(installation)

    assert resolved.ref == "supertonic:supertonic-3"
    assert resolved.bundle_id == "supertonic-3"


def test_runtime_does_not_measure_loudness(tmp_path, monkeypatch):
    import audiosig

    def fail_measurement(*args, **kwargs):
        raise AssertionError("runtime synthesis must not measure loudness")

    monkeypatch.setattr(audiosig, "measure_loudness", fail_measurement)
    runtime, _, _ = make_runtime(tmp_path)

    result = runtime.synthesize_text("count to one", voice="M1", language="en")

    assert result.audio.size == 100
