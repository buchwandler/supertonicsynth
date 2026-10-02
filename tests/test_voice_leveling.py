import json

import numpy as np
import pytest

from supertonicsynth.voice_level import (
    CalibrationDataError,
    VoiceCalibrationKey,
    VoiceLevelConfig,
    apply_voice_level_calibration,
    default_voice_calibration,
    load_voice_calibration,
)

VOICE_REF = "supertonic:supertonic-3/F1"
CALIBRATION_KEY = f"{VOICE_REF}@de"


def catalog_payload(voices=None):
    return {
        "schema": 1,
        "method": "bs1770",
        "corpus": "test-corpus-v1",
        "reference_lufs": -24.0,
        "generated_with": {"audiosig": "test"},
        "voices": voices or {},
    }


def write_catalog(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_key_uses_semantic_voice_ref_and_normalizes_language():
    key = VoiceCalibrationKey(VOICE_REF, " DE ")

    assert key.language == "de"
    assert str(key) == CALIBRATION_KEY
    assert VoiceCalibrationKey.parse(f"{VOICE_REF}@EN") == VoiceCalibrationKey(
        "supertonic:supertonic-3/F1", "en"
    )


@pytest.mark.parametrize(
    "value",
    [
        "supertonic:supertonic-3:F1@en",
        "supertonic:supertonic-3/@en",
        "supertonic:supertonic-3/F1@",
        "supertonic:supertonic-3/F1@en@de",
        "supertonic:../bundle/F1@en",
        "supertonic:supertonic-3/F1/other@en",
    ],
)
def test_key_rejects_malformed_semantic_refs(value):
    with pytest.raises(CalibrationDataError):
        VoiceCalibrationKey.parse(value)


@pytest.mark.parametrize("language", ["", "   ", "de @en", "de_de"])
def test_key_rejects_empty_or_malformed_language(language):
    with pytest.raises(CalibrationDataError):
        VoiceCalibrationKey(VOICE_REF, language)


def test_loader_parses_records_and_computes_canonical_revision(tmp_path):
    payload = catalog_payload(
        {
            CALIBRATION_KEY: {
                "gain_db": -2.5,
                "measured_lufs": -21.5,
                "reference_lufs": -24.0,
                "mad_lu": 0.1,
                "samples": 3,
                "method": "bs1770",
                "corpus_version": "test-corpus-v1",
            }
        }
    )
    first = load_voice_calibration(write_catalog(tmp_path / "first.json", payload))
    reordered = dict(reversed(list(payload.items())))
    second = load_voice_calibration(write_catalog(tmp_path / "second.json", reordered))
    key = VoiceCalibrationKey(VOICE_REF, "de")

    assert first.revision is not None and len(first.revision) == 64
    assert first.revision == second.revision
    assert first.voices[key].gain_db == -2.5
    assert first.voices[key].samples == 3


@pytest.mark.parametrize(
    "payload",
    [
        {"schema": 1},
        {**catalog_payload(), "extra": True},
        {**catalog_payload(), "schema": True},
        {**catalog_payload(), "method": "other"},
        {**catalog_payload(), "generated_with": {"audiosig": 1}},
        {**catalog_payload(), "reference_lufs": float("nan")},
    ],
)
def test_loader_rejects_invalid_top_level_fields_and_values(tmp_path, payload):
    with pytest.raises(CalibrationDataError):
        load_voice_calibration(write_catalog(tmp_path / "bad.json", payload))


def test_loader_rejects_duplicate_json_fields(tmp_path):
    path = tmp_path / "duplicate.json"
    path.write_text('{"schema":1,"schema":1}', encoding="utf-8")

    with pytest.raises(CalibrationDataError, match="duplicate JSON"):
        load_voice_calibration(path)


@pytest.mark.parametrize(
    "record",
    [
        {"extra": 1, "gain_db": 0.0},
        {"measured_lufs": -20.0},
        {"gain_db": True},
        {"gain_db": float("nan")},
        {"gain_db": 0.0, "samples": True},
        {"gain_db": 0.0, "samples": 0},
        {"gain_db": 0.0, "method": "other"},
        {"gain_db": 0.0, "corpus_version": 1},
    ],
)
def test_loader_rejects_invalid_records(tmp_path, record):
    payload = catalog_payload({CALIBRATION_KEY: record})

    with pytest.raises(CalibrationDataError):
        load_voice_calibration(write_catalog(tmp_path / "bad-record.json", payload))


def test_loader_rejects_duplicate_normalized_calibration_identities(tmp_path):
    payload = catalog_payload(
        {
            f"{VOICE_REF}@de": {"gain_db": 0.0},
            f"{VOICE_REF}@DE": {"gain_db": 1.0},
        }
    )

    with pytest.raises(CalibrationDataError, match="duplicate normalized"):
        load_voice_calibration(write_catalog(tmp_path / "duplicate-normalized.json", payload))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"mode": "invalid"},
        {"gain_db": True},
        {"gain_db": float("nan")},
        {"gain_db": float("inf")},
    ],
)
def test_voice_level_config_rejects_invalid_values(kwargs):
    with pytest.raises(ValueError):
        VoiceLevelConfig(**kwargs)


def test_explicit_gain_overrides_catalog_and_does_not_require_identity(tmp_path):
    catalog = load_voice_calibration(
        write_catalog(
            tmp_path / "catalog.json",
            catalog_payload({CALIBRATION_KEY: {"gain_db": -6.0}}),
        )
    )
    audio = np.array([0.25, -0.5], dtype=np.float32)

    result, application = apply_voice_level_calibration(
        audio,
        VoiceLevelConfig(mode="calibrated", gain_db=6.0),
        None,
        catalog=catalog,
    )

    np.testing.assert_allclose(result, audio * 10 ** (6 / 20), rtol=1e-6)
    assert application.source == "override"
    assert application.key is None
    assert application.catalog_revision is None


def test_catalog_gain_uses_decibels_without_clipping(tmp_path):
    catalog = load_voice_calibration(
        write_catalog(
            tmp_path / "catalog.json",
            catalog_payload({CALIBRATION_KEY: {"gain_db": 6.0}}),
        )
    )
    audio = np.array([0.8], dtype=np.float32)

    result, application = apply_voice_level_calibration(
        audio,
        VoiceLevelConfig(mode="calibrated"),
        VoiceCalibrationKey(VOICE_REF, "de"),
        catalog=catalog,
    )

    np.testing.assert_allclose(result, audio * 10 ** (6 / 20), rtol=1e-6)
    assert result[0] > 1.0
    assert application.source == "catalog"
    assert application.applied is True
    assert application.catalog_revision == catalog.revision
    assert application.calibration_key == VoiceCalibrationKey(VOICE_REF, "de")
    assert application.reason == "a matching calibration was selected"


def test_off_and_missing_identity_or_calibration_leave_audio_unchanged(tmp_path):
    audio = np.array([0.25, -0.5], dtype=np.float32)
    key = VoiceCalibrationKey(VOICE_REF, "de")
    empty_catalog = load_voice_calibration(
        write_catalog(tmp_path / "empty.json", catalog_payload())
    )

    off, application = apply_voice_level_calibration(audio, VoiceLevelConfig(), key)
    np.testing.assert_array_equal(off, audio)
    assert application.source == "off"
    assert application.reason == "voice-level calibration is disabled"

    missing_identity, application = apply_voice_level_calibration(
        audio, VoiceLevelConfig(mode="calibrated"), None, catalog=empty_catalog
    )
    np.testing.assert_array_equal(missing_identity, audio)
    assert application.source == "missing_identity"
    assert application.reason == "the voice has no stable managed semantic identity"

    missing_calibration, application = apply_voice_level_calibration(
        audio, VoiceLevelConfig(mode="calibrated"), key, catalog=empty_catalog
    )
    np.testing.assert_array_equal(missing_calibration, audio)
    assert application.source == "missing_calibration"
    assert application.reason == "no calibration entry matches this voice/language"
    assert application.catalog_revision == empty_catalog.revision


def test_default_catalog_loads_from_packaged_resources():
    catalog = default_voice_calibration()

    assert catalog.schema == 1
    assert catalog.method == "bs1770"
    assert catalog.corpus == "supertonicsynth-counting-1-to-10-v1"
    assert catalog.reference_lufs == -24.0
    assert len(catalog.voices) == 252
    assert catalog.revision is not None and len(catalog.revision) == 64

    de_key = VoiceCalibrationKey("supertonic:supertonic-3/F1", "de")
    de = catalog.voices[de_key]
    assert de.gain_db == pytest.approx(-7.350321819495406)
    assert de.measured_lufs == pytest.approx(-16.649678180504594)
    assert de.mad_lu == pytest.approx(0.1121306640854769)
    assert de.samples == 3
    assert de.method == "bs1770"
    assert de.corpus_version == "supertonicsynth-counting-1-to-10-v1"

    assert VoiceCalibrationKey("supertonic:supertonic-3/F1", "cs") not in catalog.voices
    for voice in ("F1", "F2", "F3", "F4", "F5", "M1", "M2", "M3", "M4", "M5"):
        assert VoiceCalibrationKey(f"supertonic:supertonic-3/{voice}", "hr") not in catalog.voices


def test_packaged_calibration_records_match_benchmark_policy():
    catalog = default_voice_calibration()

    for key, record in catalog.voices.items():
        assert key.language not in {"hr", "na"}
        assert record.samples == 3
        assert record.method == "bs1770"
        assert record.reference_lufs == -24.0
        assert record.corpus_version == catalog.corpus
        assert record.mad_lu is not None and record.mad_lu <= 0.75
        assert record.measured_lufs is not None
        assert record.gain_db == pytest.approx(-24.0 - record.measured_lufs)
