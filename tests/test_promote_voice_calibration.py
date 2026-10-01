import hashlib
import json

import pytest

from benchmarks.promote_voice_calibration import (
    EXPECTED_LANGUAGES,
    _merge_reports,
    build_candidate,
    build_review_template,
)
from benchmarks.promote_voice_calibration import main as promote_main
from benchmarks.voice_level_benchmark import load_policy, prepare_counting_stimulus
from supertonicsynth import load_voice_calibration
from supertonicsynth.voice_level import VoiceCalibrationKey

VOICE_REF = "supertonic:supertonic-3/F1"
CALIBRATION_KEY = f"{VOICE_REF}@en"


def _report(repeat, loudness):
    policy = load_policy()
    key = VoiceCalibrationKey(VOICE_REF, "en")
    expected_keys = sorted(
        str(VoiceCalibrationKey(VOICE_REF, language)) for language in EXPECTED_LANGUAGES
    )
    return {
        "schema": 1,
        "corpus": policy["name"],
        "model": "supertonic-3",
        "policy": policy,
        "generated_with": {
            "supertonicsynth": "test-version",
            "audiosig": "test-version",
            "onnxvoice": "test-version",
            "spokenform": "test-version",
            "numeralform": "test-version",
        },
        "matrix": {
            "catalog_voice_refs": [VOICE_REF],
            "languages": list(EXPECTED_LANGUAGES),
            "expected_keys": expected_keys,
            "selected_keys": [str(key)],
        },
        "stimuli": {"en": prepare_counting_stimulus("en")},
        "stimulus_failures": [],
        "coverage": {"complete": False},
        "measurements": [
            {
                "voice_ref": VOICE_REF,
                "language": "en",
                "calibration_key": CALIBRATION_KEY,
                "repeat": repeat,
                "seed": policy["seed"] + repeat,
                "sample_rate": 44100,
                "duration_seconds": 4.2,
                "integrated_lufs": loudness,
                "synthesis_hash": hashlib.sha256(f"repeat-{repeat}".encode()).hexdigest(),
            }
        ],
        "failures": [],
    }


def _report_hash(report):
    return hashlib.sha256(
        json.dumps(report, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _review(
    reports, hashes, *, decision="approve", rationale="Reviewed measured report statistics."
):
    return {
        "schema": 1,
        "reviewer": "reviewer@example.test",
        "reviewed_at": "2026-01-01T12:00:00Z",
        "source_reports": sorted(hashes),
        "decisions": {CALIBRATION_KEY: {"decision": decision, "rationale": rationale}},
    }


def test_multi_report_promotion_requires_measured_review_and_preserves_only_approved_rows():
    reports = [_report(0, -16.0), _report(1, -16.2), _report(2, -15.8)]
    hashes = [_report_hash(report) for report in reports]
    review = _review(reports, hashes)

    candidate, summary = build_candidate(reports, review, hashes, allow_partial=True)

    assert list(candidate["voices"]) == [CALIBRATION_KEY]
    assert candidate["voices"][CALIBRATION_KEY] == {
        "gain_db": -8.0,
        "measured_lufs": -16.0,
        "reference_lufs": -24.0,
        "mad_lu": pytest.approx(0.2),
        "samples": 3,
        "method": "bs1770",
        "corpus_version": "supertonicsynth-counting-1-to-10-v1",
    }
    assert summary["matrix_complete"] is False
    assert summary["promoted"] == 1
    assert summary["review_record"]["reviewed_decisions"][CALIBRATION_KEY]["decision"] == "approve"


def test_review_hashes_must_match_exact_measurement_reports():
    reports = [_report(0, -16.0), _report(1, -16.0), _report(2, -16.0)]
    hashes = [_report_hash(report) for report in reports]
    review = _review(reports, ["0" * 64, *hashes[1:]])

    with pytest.raises(ValueError, match="source_reports do not match"):
        build_candidate(reports, review, hashes, allow_partial=True)


def test_approval_requires_a_completed_non_placeholder_review_rationale():
    reports = [_report(0, -16.0), _report(1, -16.0), _report(2, -16.0)]
    hashes = [_report_hash(report) for report in reports]
    review = _review(
        reports,
        hashes,
        rationale="Review this measured result before approving it.",
    )

    with pytest.raises(ValueError, match="completed review rationale"):
        build_candidate(reports, review, hashes, allow_partial=True)


def test_high_variability_result_cannot_be_approved():
    reports = [_report(0, -16.0), _report(1, -18.0), _report(2, -20.0)]
    hashes = [_report_hash(report) for report in reports]
    review = _review(reports, hashes)

    with pytest.raises(ValueError, match="only eligible measurements"):
        build_candidate(reports, review, hashes, allow_partial=True)


def test_multi_report_merge_rejects_duplicate_repeat_measurements():
    reports = [_report(0, -16.0), _report(0, -16.1)]

    with pytest.raises(ValueError, match="duplicate measurement repeat"):
        _merge_reports(reports)


def test_review_template_is_bound_to_report_hashes_and_lists_only_eligible_results():
    reports = [_report(0, -16.0), _report(1, -16.0), _report(2, -16.0)]
    hashes = [_report_hash(report) for report in reports]

    template = build_review_template(reports, hashes)

    assert template["source_reports"] == sorted(hashes)
    assert list(template["decisions"]) == [CALIBRATION_KEY]
    assert template["decisions"][CALIBRATION_KEY]["status"] == "eligible"
    assert template["decisions"][CALIBRATION_KEY]["decision"] == "exclude"


def test_promotion_cli_writes_a_loader_compatible_reviewed_candidate(tmp_path):
    reports = [_report(0, -16.0), _report(1, -16.2), _report(2, -15.8)]
    report_paths = []
    for index, report in enumerate(reports):
        path = tmp_path / f"report-{index}.json"
        path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        report_paths.append(path)
    review_path = tmp_path / "review.json"
    candidate_path = tmp_path / "candidate.json"

    assert (
        promote_main(
            [
                *(str(path) for path in report_paths),
                "--review-template",
                "--output",
                str(review_path),
            ]
        )
        == 0
    )
    review = json.loads(review_path.read_text(encoding="utf-8"))
    review["reviewer"] = "reviewer@example.test"
    review["reviewed_at"] = "2026-01-01T12:00:00Z"
    review["decisions"][CALIBRATION_KEY].update(
        decision="approve",
        rationale="Reviewed repeat measurements and verified the policy-derived gain.",
    )
    review_path.write_text(json.dumps(review, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    assert (
        promote_main(
            [
                *(str(path) for path in report_paths),
                "--review",
                str(review_path),
                "--output",
                str(candidate_path),
                "--allow-partial",
            ]
        )
        == 0
    )
    catalog = load_voice_calibration(candidate_path)
    assert catalog.voices[VoiceCalibrationKey.parse(CALIBRATION_KEY)].gain_db == -8.0
    assert candidate_path.with_suffix(".json.review.json").is_file()


def test_promotion_rejects_unknown_language_sentinel():
    reports = [_report(0, -16.0)]
    row = reports[0]["measurements"][0]
    row["language"] = "na"
    row["calibration_key"] = f"{VOICE_REF}@na"
    reports[0]["matrix"]["selected_keys"] = [f"{VOICE_REF}@na"]

    with pytest.raises(ValueError, match="selected keys are invalid"):
        _merge_reports(reports)
