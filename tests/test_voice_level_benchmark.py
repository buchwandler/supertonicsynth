import json
import math

import pytest

import benchmarks.voice_level_benchmark as benchmark
from benchmarks.voice_level_benchmark import (
    EXPECTED_LANGUAGES,
    aggregate_measurements,
    build_report,
    expected_keys,
    load_policy,
    prepare_counting_stimulus,
    prepare_stimuli,
)
from supertonicsynth import SUPPORTED_LANGUAGES, UNKNOWN_LANGUAGE
from supertonicsynth.voice_level import VoiceCalibrationKey


def _measurement(key, repeat, loudness):
    voice_ref, language = key.rsplit("@", 1)
    return {
        "voice_ref": voice_ref,
        "language": language,
        "calibration_key": key,
        "repeat": repeat,
        "integrated_lufs": loudness,
    }


def test_counting_stimulus_is_prepared_and_keeps_provenance():
    first = prepare_counting_stimulus("en")
    second = prepare_counting_stimulus("en")

    assert first == second
    assert first["source_text"] == "1, 2, 3, 4, 5, 6, 7, 8, 9, 10."
    assert len(first["count_words"]) == 10
    assert all(isinstance(word, str) and word for word in first["count_words"])
    assert first["prepared_text"] == first["counting_text"]
    assert first["number_renderer"] == "numeralform"
    assert first["text_preparer"] == "spokenform"
    assert first["preparation_warnings"] == [
        "[NUMBERS] caller-managed number categories for language 'en'"
    ]
    assert not any(character.isdecimal() for character in first["prepared_text"])


def test_prepare_only_writes_complete_language_provenance_without_opening_runtime(
    monkeypatch, tmp_path
):
    output = tmp_path / "counting_stimuli.json"
    monkeypatch.setattr(
        benchmark.SupertonicRuntime,
        "from_pretrained",
        staticmethod(lambda *_args, **_kwargs: pytest.fail("runtime must not open")),
    )

    exit_code = benchmark.main(["--prepare-only", "--stimuli-output", str(output)])

    manifest = json.loads(output.read_text(encoding="utf-8"))
    assert exit_code == (0 if manifest["complete"] else 1)
    assert manifest["requested_languages"] == list(EXPECTED_LANGUAGES)
    assert manifest["supported_spoken_languages"] == list(EXPECTED_LANGUAGES)
    assert manifest["unknown_language_excluded"] == UNKNOWN_LANGUAGE
    assert set(manifest["stimuli"]) | {row["language"] for row in manifest["failures"]} == set(
        EXPECTED_LANGUAGES
    )
    assert manifest["counting_source"]["number_range"] == [1, 10]


def test_counting_stimuli_cover_every_preparable_language_without_english_fallback():
    stimuli, failures = prepare_stimuli()

    assert set(stimuli) | {row["language"] for row in failures} == set(EXPECTED_LANGUAGES)
    assert set(EXPECTED_LANGUAGES) == set(SUPPORTED_LANGUAGES) - {UNKNOWN_LANGUAGE}
    assert UNKNOWN_LANGUAGE not in stimuli
    assert UNKNOWN_LANGUAGE not in {row["language"] for row in failures}
    assert all(row["language"] != "en" for row in failures)
    assert "hi" in stimuli
    assert all(row["language"] == language for language, row in stimuli.items())
    assert all(
        not any(character.isdecimal() for character in row["prepared_text"])
        for row in stimuli.values()
    )
    assert all(row["phase"] == "stimulus_preparation" for row in failures)
    assert all(row["language"] != "en" for row in failures)
    assert "hr" in stimuli or "hr" in {row["language"] for row in failures}


def test_calibration_matrix_excludes_unknown_language():
    voice_refs = [f"supertonic:supertonic-3/{name}" for name in ("F1", "M1")]

    keys = expected_keys(voice_refs)

    assert len(keys) == 2 * len(SUPPORTED_LANGUAGES)
    assert all(not key.endswith("@na") for key in keys)
    assert VoiceCalibrationKey("supertonic:supertonic-3/F1", "de") in {
        VoiceCalibrationKey.parse(key) for key in keys
    }


def test_aggregate_measurements_computes_eligible_high_variability_and_incomplete():
    policy = load_policy()
    eligible_key = "supertonic:supertonic-3/F1@en"
    variable_key = "supertonic:supertonic-3/F1@de"
    incomplete_key = "supertonic:supertonic-3/F1@fr"
    measurements = [
        _measurement(eligible_key, 0, -16.0),
        _measurement(eligible_key, 1, -16.2),
        _measurement(eligible_key, 2, -15.8),
        _measurement(variable_key, 0, -16.0),
        _measurement(variable_key, 1, -18.0),
        _measurement(variable_key, 2, -20.0),
        _measurement(incomplete_key, 0, -19.0),
    ]

    rows = {
        row["calibration_key"]: row
        for row in aggregate_measurements(
            [eligible_key, variable_key, incomplete_key], measurements, policy
        )
    }

    assert rows[eligible_key]["status"] == "eligible"
    assert rows[eligible_key]["median_lufs"] == -16.0
    assert rows[eligible_key]["mad_lu"] == pytest.approx(0.2)
    assert rows[eligible_key]["gain_db"] == -8.0
    assert rows[variable_key]["status"] == "high_variability"
    assert rows[variable_key]["gain_db"] is None
    assert rows[incomplete_key]["status"] == "incomplete"
    assert rows[incomplete_key]["gain_db"] is None


def test_aggregate_rejects_duplicate_or_non_finite_measurements():
    policy = load_policy()
    key = "supertonic:supertonic-3/F1@en"

    with pytest.raises(ValueError, match="duplicate measurement repeat"):
        aggregate_measurements(
            [key], [_measurement(key, 0, -16.0), _measurement(key, 0, -16.0)], policy
        )

    with pytest.raises(ValueError, match="finite"):
        aggregate_measurements([key], [_measurement(key, 0, math.nan)], policy)


def test_report_retains_expected_full_matrix_and_measurement_provenance():
    policy = load_policy()
    voice_refs = ["supertonic:supertonic-3/F1"]
    key = f"{voice_refs[0]}@en"
    stimulus = prepare_counting_stimulus("en")
    measurements = [
        {
            **_measurement(key, repeat, -16.0),
            "seed": int(policy["seed"]) + repeat,
            "sample_rate": 44100,
            "duration_seconds": 4.2,
            "synthesis_hash": f"hash-{repeat}",
        }
        for repeat in range(int(policy["repeats"]))
    ]

    report = build_report(
        model="supertonic-3",
        catalog_voice_refs=voice_refs,
        selected_keys=[key],
        stimuli={"en": stimulus},
        stimulus_failures=[],
        measurements=measurements,
        failures=[],
        policy=policy,
    )

    assert len(report["matrix"]["expected_keys"]) == len(EXPECTED_LANGUAGES)
    assert report["coverage"]["expected"] == len(EXPECTED_LANGUAGES)
    assert report["coverage"]["measured"] == 1
    assert (
        next(row for row in report["aggregates"] if row["calibration_key"] == key)["status"]
        == "eligible"
    )
    assert report["stimuli"]["en"] == stimulus
