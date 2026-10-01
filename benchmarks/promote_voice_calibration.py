#!/usr/bin/env python3
"""Merge measured reports and create a reviewed candidate calibration catalog."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from supertonicsynth import SUPPORTED_LANGUAGES, UNKNOWN_LANGUAGE
from supertonicsynth.voice_level import VoiceCalibrationKey

PRODUCTION_CATALOG = (
    Path(__file__).resolve().parents[1]
    / "supertonicsynth"
    / "data"
    / "voice_level_calibration.json"
)
DEFAULT_OUTPUT = Path("benchmarks/output/voice_level_calibration/candidate_catalog.json")
EXPECTED_LANGUAGES = tuple(sorted(set(SUPPORTED_LANGUAGES) - {UNKNOWN_LANGUAGE}))


def _object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_object_pairs)


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _required_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _sha256_digest(value: Any, name: str) -> str:
    digest = _required_string(value, name)
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return digest


def _normalize_report(report: Mapping[str, Any]) -> dict[str, Any]:
    if (
        isinstance(report.get("schema"), bool)
        or not isinstance(report.get("schema"), int)
        or report["schema"] != 1
    ):
        raise ValueError("unsupported measurement report schema")
    policy = report.get("policy")
    matrix = report.get("matrix")
    generated_with = report.get("generated_with")
    stimuli = report.get("stimuli")
    if not all(isinstance(item, Mapping) for item in (policy, matrix, generated_with, stimuli)):
        raise ValueError("measurement report is missing policy, matrix, versions, or stimuli")
    corpus = _required_string(report.get("corpus"), "corpus")
    model = _required_string(report.get("model"), "model")
    repeats = policy.get("repeats")
    seed = policy.get("seed")
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 2:
        raise ValueError("measurement policy repeats must be at least two")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("measurement policy seed must be an integer")
    _finite_number(policy.get("reference_lufs"), "reference_lufs")
    minimum = _finite_number(policy.get("min_gain_db"), "min_gain_db")
    maximum = _finite_number(policy.get("max_gain_db"), "max_gain_db")
    max_mad = _finite_number(policy.get("max_mad_lu"), "max_mad_lu")
    if minimum > maximum or max_mad < 0:
        raise ValueError("measurement policy gain or variability limits are invalid")

    raw_voices = matrix.get("catalog_voice_refs")
    raw_languages = matrix.get("languages")
    raw_expected = matrix.get("expected_keys")
    raw_selected = matrix.get("selected_keys")
    if not isinstance(raw_voices, list) or not raw_voices:
        raise ValueError("measurement matrix must identify canonical catalog voice refs")
    if not isinstance(raw_languages, list) or tuple(raw_languages) != EXPECTED_LANGUAGES:
        raise ValueError("measurement matrix must cover all supported spoken languages in order")
    voice_refs = [_required_string(ref, "catalog voice ref") for ref in raw_voices]
    if len(voice_refs) != len(set(voice_refs)):
        raise ValueError("measurement matrix contains duplicate catalog voice refs")
    for voice_ref in voice_refs:
        if not voice_ref.startswith("supertonic:"):
            raise ValueError(f"non-canonical voice ref in report: {voice_ref}")
    expected_keys = sorted(
        str(VoiceCalibrationKey(voice_ref, language))
        for voice_ref in voice_refs
        for language in EXPECTED_LANGUAGES
    )
    if raw_expected != expected_keys:
        raise ValueError("measurement report expected keys do not match its catalog matrix")
    if (
        not isinstance(raw_selected, list)
        or any(not isinstance(key, str) for key in raw_selected)
        or len(raw_selected) != len(set(raw_selected))
        or not set(raw_selected) <= set(expected_keys)
    ):
        raise ValueError("measurement report selected keys are invalid")
    if not isinstance(generated_with, Mapping) or any(
        not isinstance(name, str) or not isinstance(value, str) or not value
        for name, value in generated_with.items()
    ):
        raise ValueError("generated_with must map package names to non-empty versions")
    for language, stimulus in stimuli.items():
        if language not in EXPECTED_LANGUAGES or not isinstance(stimulus, Mapping):
            raise ValueError(f"invalid stimulus report for language {language!r}")
        if stimulus.get("language") != language or not stimulus.get("prepared_text"):
            raise ValueError(f"stimulus for {language!r} is missing prepared text provenance")
        if (
            stimulus.get("number_renderer") != "numeralform"
            or stimulus.get("text_preparer") != "spokenform"
        ):
            raise ValueError(
                f"stimulus for {language!r} does not use the declared preparation tools"
            )

    measurements = report.get("measurements")
    failures = report.get("failures")
    stimulus_failures = report.get("stimulus_failures")
    if not isinstance(measurements, list) or not isinstance(failures, list):
        raise ValueError("measurement report arrays are malformed")
    if not isinstance(stimulus_failures, list):
        raise ValueError("stimulus_failures must be a list")
    if any(not isinstance(row, Mapping) for row in failures):
        raise ValueError("failure entries must be objects")
    if any(
        not isinstance(row, Mapping) or row.get("language") not in EXPECTED_LANGUAGES
        for row in stimulus_failures
    ):
        raise ValueError("stimulus preparation failures must identify a spoken language")
    return {
        "corpus": corpus,
        "model": model,
        "policy": dict(policy),
        "matrix": {
            "catalog_voice_refs": voice_refs,
            "languages": list(raw_languages),
            "expected_keys": expected_keys,
        },
        "selected_keys": list(raw_selected),
        "generated_with": dict(generated_with),
        "stimuli": dict(stimuli),
        "measurements": measurements,
        "failures": failures,
        "stimulus_failures": stimulus_failures,
    }


def _merge_reports(
    reports: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not reports:
        raise ValueError("at least one measurement report is required")
    normalized = [_normalize_report(report) for report in reports]
    first = normalized[0]
    for report in normalized[1:]:
        for name in ("corpus", "model", "policy", "matrix", "generated_with"):
            if report[name] != first[name]:
                raise ValueError(f"measurement reports disagree on {name}")

    stimuli: dict[str, Mapping[str, Any]] = {}
    measurements: dict[tuple[str, int], dict[str, Any]] = {}
    failures: list[dict[str, Any]] = []
    stimulus_failures: dict[str, Mapping[str, Any]] = {}
    for report in normalized:
        for language, stimulus in report["stimuli"].items():
            previous = stimuli.get(language)
            if previous is not None and previous != stimulus:
                raise ValueError(f"measurement reports disagree on the {language!r} stimulus")
            stimuli[language] = stimulus
        for row in report["stimulus_failures"]:
            if not isinstance(row, Mapping) or row.get("language") not in EXPECTED_LANGUAGES:
                raise ValueError("invalid stimulus preparation failure")
            stimulus_failures[str(row["language"])] = row
        for row in report["measurements"]:
            if not isinstance(row, Mapping):
                raise ValueError("measurement entries must be objects")
            key = _required_string(row.get("calibration_key"), "calibration_key")
            parsed = VoiceCalibrationKey.parse(key)
            if str(parsed) != key or parsed.voice_ref not in first["matrix"]["catalog_voice_refs"]:
                raise ValueError(f"measurement uses an unknown or non-canonical identity: {key}")
            if key not in report["selected_keys"]:
                raise ValueError(f"measurement was not selected in its report: {key}")
            if parsed.language not in report["stimuli"]:
                raise ValueError(f"measurement has no prepared stimulus provenance: {key}")
            if parsed.language not in EXPECTED_LANGUAGES:
                raise ValueError(f"measurement uses an unsupported calibration language: {key}")
            if row.get("voice_ref") != parsed.voice_ref or row.get("language") != parsed.language:
                raise ValueError(f"measurement identity fields disagree for {key}")
            repeat = row.get("repeat")
            if (
                isinstance(repeat, bool)
                or not isinstance(repeat, int)
                or not 0 <= repeat < first["policy"]["repeats"]
            ):
                raise ValueError(f"measurement repeat is outside the declared policy for {key}")
            if row.get("seed") != first["policy"]["seed"] + repeat:
                raise ValueError(f"measurement seed does not match the fixed policy for {key}")
            _finite_number(row.get("integrated_lufs"), f"{key}.integrated_lufs")
            sample_rate = row.get("sample_rate")
            if (
                isinstance(sample_rate, bool)
                or not isinstance(sample_rate, int)
                or sample_rate <= 0
            ):
                raise ValueError(f"measurement sample rate is invalid for {key}")
            _finite_number(row.get("duration_seconds"), f"{key}.duration_seconds")
            _required_string(row.get("synthesis_hash"), f"{key}.synthesis_hash")
            identity = (key, repeat)
            if identity in measurements:
                raise ValueError(f"duplicate measurement repeat {repeat} for {key}")
            measurements[identity] = dict(row)
        failures.extend(row for row in report["failures"] if isinstance(row, Mapping))

    if set(stimuli) & set(stimulus_failures):
        raise ValueError("a language has both a prepared stimulus and a preparation failure")
    return (
        {
            **first,
            "stimuli": stimuli,
            "stimulus_failures": list(stimulus_failures.values()),
            "measurements": list(measurements.values()),
            "failures": failures,
        },
        list(measurements.values()),
    )


def _aggregates(merged: Mapping[str, Any]) -> list[dict[str, Any]]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in merged["measurements"]:
        groups[str(row["calibration_key"])].append(row)
    policy = merged["policy"]
    rows = []
    for key in merged["matrix"]["expected_keys"]:
        values = [float(row["integrated_lufs"]) for row in groups.get(key, ())]
        repeats = len(values)
        median = statistics.median(values) if values else None
        mad = statistics.median(abs(value - median) for value in values) if values else None
        if repeats != policy["repeats"]:
            status = "incomplete"
            gain_db = None
        elif mad is not None and mad > policy["max_mad_lu"]:
            status = "high_variability"
            gain_db = None
        else:
            status = "eligible"
            requested_gain = policy["reference_lufs"] - median
            gain_db = min(policy["max_gain_db"], max(policy["min_gain_db"], requested_gain))
        voice_ref, language = key.rsplit("@", 1)
        rows.append(
            {
                "voice_ref": voice_ref,
                "language": language,
                "calibration_key": key,
                "median_lufs": median,
                "mad_lu": mad,
                "repeat_count": repeats,
                "status": status,
                "gain_db": gain_db,
            }
        )
    return rows


def _review_metadata(review: Mapping[str, Any], report_hashes: Sequence[str]) -> dict[str, Any]:
    if (
        isinstance(review.get("schema"), bool)
        or not isinstance(review.get("schema"), int)
        or review["schema"] != 1
    ):
        raise ValueError("unsupported review manifest schema")
    reviewer = _required_string(review.get("reviewer"), "reviewer")
    reviewed_at = _required_string(review.get("reviewed_at"), "reviewed_at")
    try:
        timestamp = datetime.fromisoformat(reviewed_at.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("reviewed_at must be an ISO 8601 timestamp") from error
    if timestamp.tzinfo is None:
        raise ValueError("reviewed_at must include a timezone")
    expected_hashes = [
        _sha256_digest(digest, f"report_hashes[{index}]")
        for index, digest in enumerate(report_hashes)
    ]
    sources = review.get("source_reports")
    if (
        not isinstance(sources, list)
        or any(not isinstance(source, str) for source in sources)
        or len(sources) != len(set(sources))
        or sorted(sources) != sorted(expected_hashes)
    ):
        raise ValueError("review manifest source_reports do not match the input reports")
    decisions = review.get("decisions")
    if not isinstance(decisions, Mapping):
        raise ValueError("review decisions must be an object")
    checked: dict[str, dict[str, str]] = {}
    for key, decision in decisions.items():
        parsed = VoiceCalibrationKey.parse(key)
        if str(parsed) != key or not isinstance(decision, Mapping):
            raise ValueError(f"invalid review decision for {key!r}")
        choice = decision.get("decision")
        rationale = _required_string(decision.get("rationale"), f"{key}.rationale")
        if choice not in {"approve", "exclude"}:
            raise ValueError(f"{key}.decision must be 'approve' or 'exclude'")
        if choice == "approve" and rationale == "Review this measured result before approving it.":
            raise ValueError(f"{key} approval needs a completed review rationale")
        checked[key] = {"decision": str(choice), "rationale": rationale}
    return {
        "reviewer": reviewer,
        "reviewed_at": reviewed_at,
        "source_reports": sorted(report_hashes),
        "decisions": checked,
    }


def build_review_template(
    reports: Sequence[Mapping[str, Any]], report_hashes: Sequence[str]
) -> dict[str, Any]:
    merged, _ = _merge_reports(reports)
    aggregates = _aggregates(merged)
    return {
        "schema": 1,
        "reviewer": "",
        "reviewed_at": "",
        "source_reports": sorted(report_hashes),
        "decisions": {
            row["calibration_key"]: {
                "decision": "exclude",
                "rationale": "Review this measured result before approving it.",
                "status": row["status"],
                "median_lufs": row["median_lufs"],
                "mad_lu": row["mad_lu"],
                "gain_db": row["gain_db"],
            }
            for row in aggregates
            if row["status"] == "eligible"
        },
    }


def build_candidate(
    reports: Sequence[Mapping[str, Any]],
    review: Mapping[str, Any],
    report_hashes: Sequence[str],
    *,
    allow_partial: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    merged, _ = _merge_reports(reports)
    expected_keys = set(merged["matrix"]["expected_keys"])
    aggregate_rows = _aggregates(merged)
    complete_keys = {
        row["calibration_key"] for row in aggregate_rows if row["status"] != "incomplete"
    }
    matrix_complete = (
        complete_keys == expected_keys
        and set(merged["stimuli"]) == set(EXPECTED_LANGUAGES)
        and not merged["stimulus_failures"]
        and not merged["failures"]
    )
    if not matrix_complete and not allow_partial:
        missing = sorted(expected_keys - complete_keys)
        unavailable = sorted(set(EXPECTED_LANGUAGES) - set(merged["stimuli"]))
        raise ValueError(
            "incomplete calibration matrix requires --allow-partial; "
            f"missing identities={len(missing)}, unavailable languages={unavailable}"
        )

    reviewed = _review_metadata(review, report_hashes)
    aggregate_by_key = {row["calibration_key"]: row for row in aggregate_rows}
    unknown_reviews = set(reviewed["decisions"]) - expected_keys
    if unknown_reviews:
        raise ValueError(
            f"review manifest contains unknown calibration keys: {sorted(unknown_reviews)}"
        )
    voices: dict[str, dict[str, Any]] = {}
    counts = {"eligible": 0, "high_variability": 0, "incomplete": 0, "reviewed": 0, "promoted": 0}
    for key, aggregate in aggregate_by_key.items():
        status = str(aggregate["status"])
        counts[status] += 1
        decision = reviewed["decisions"].get(key)
        if decision is None:
            continue
        counts["reviewed"] += 1
        if decision["decision"] != "approve":
            continue
        if status != "eligible":
            raise ValueError(f"only eligible measurements may be approved: {key} ({status})")
        voices[key] = {
            "gain_db": float(aggregate["gain_db"]),
            "measured_lufs": float(aggregate["median_lufs"]),
            "reference_lufs": float(merged["policy"]["reference_lufs"]),
            "mad_lu": float(aggregate["mad_lu"]),
            "samples": int(aggregate["repeat_count"]),
            "method": "bs1770",
            "corpus_version": merged["corpus"],
        }
        counts["promoted"] += 1
    if not voices:
        raise ValueError("review manifest approves no eligible measurements")

    catalog = {
        "schema": 1,
        "method": "bs1770",
        "corpus": merged["corpus"],
        "reference_lufs": float(merged["policy"]["reference_lufs"]),
        "generated_with": dict(sorted(merged["generated_with"].items())),
        "voices": dict(sorted(voices.items())),
    }
    review_record = {
        "schema": 1,
        **reviewed,
        "matrix_complete": matrix_complete,
        "catalog_voice_refs": merged["matrix"]["catalog_voice_refs"],
        "languages": list(EXPECTED_LANGUAGES),
        "reviewed_decisions": {
            key: {
                **decision,
                "median_lufs": aggregate_by_key[key]["median_lufs"],
                "mad_lu": aggregate_by_key[key]["mad_lu"],
                "gain_db": aggregate_by_key[key]["gain_db"],
            }
            for key, decision in reviewed["decisions"].items()
        },
    }
    return catalog, {**counts, "matrix_complete": matrix_complete, "review_record": review_record}


def _write_json(path: Path, value: Mapping[str, Any], *, force: bool) -> None:
    if path.exists() and not force:
        raise SystemExit(f"output already exists: {path}; pass --force to replace it")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", type=Path)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--review", type=Path)
    group.add_argument("--review-template", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report_bytes = [path.read_bytes() for path in args.reports]
    reports = [json.loads(raw, object_pairs_hook=_object_pairs) for raw in report_bytes]
    if any(not isinstance(report, Mapping) for report in reports):
        raise SystemExit("measurement reports must contain JSON objects")
    report_hashes = [hashlib.sha256(raw).hexdigest() for raw in report_bytes]
    if len(report_hashes) != len(set(report_hashes)):
        raise SystemExit("duplicate measurement reports were supplied")
    output = args.output.resolve()
    if output == PRODUCTION_CATALOG.resolve():
        raise SystemExit("promotion never writes directly to the packaged production catalog")
    report_paths = {path.resolve() for path in args.reports}
    if output in report_paths:
        raise SystemExit("output must not overwrite an input measurement report")
    if args.review is not None and args.review.resolve() in report_paths:
        raise SystemExit("review manifest must not also be an input measurement report")

    if args.review_template:
        template = build_review_template(reports, report_hashes)
        _write_json(output, template, force=args.force)
        print(f"Review template: {output}")
        return 0

    review = _read_json(args.review)
    if not isinstance(review, Mapping):
        raise SystemExit("review manifest must contain a JSON object")
    candidate, summary = build_candidate(
        reports,
        review,
        report_hashes,
        allow_partial=args.allow_partial,
    )
    review_path = output.with_suffix(output.suffix + ".review.json")
    if review_path in report_paths or review_path == args.review.resolve():
        raise SystemExit("review record must not overwrite an input report or review manifest")
    if not args.force and (output.exists() or review_path.exists()):
        raise SystemExit(
            f"candidate or review output already exists; pass --force to replace: {output}"
        )
    _write_json(output, candidate, force=args.force)
    _write_json(review_path, summary["review_record"], force=args.force)
    for name in ("eligible", "high_variability", "incomplete", "reviewed", "promoted"):
        print(f"{name.replace('_', ' ').title()}: {summary[name]}")
    print(f"Full matrix: {summary['matrix_complete']}")
    print(f"Candidate: {output}")
    print(f"Review record: {review_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
