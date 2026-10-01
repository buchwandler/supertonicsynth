#!/usr/bin/env python3
"""Measure prepared counting speech for static Supertonic voice calibration."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from audiosig import measure_loudness
from numeralform import render
from spokenform import NumberPolicy, prepare_language

from supertonicsynth import (
    SUPPORTED_LANGUAGES,
    UNKNOWN_LANGUAGE,
    SupertonicRuntime,
    SynthesisConfig,
    VoiceLevelConfig,
)
from supertonicsynth.runtime import canonical_voice_ref
from supertonicsynth.voice_level import VoiceCalibrationKey

POLICY_PATH = Path(__file__).with_name("data") / "voice_level_policy.json"
DEFAULT_OUTPUT = Path("benchmarks/output/voice_level_calibration/measurements.json")
DEFAULT_STIMULI_OUTPUT = Path("benchmarks/output/voice_level_calibration/counting_stimuli.json")
STIMULUS_SOURCE = ", ".join(str(number) for number in range(1, 11)) + "."
EXPECTED_LANGUAGES = tuple(sorted(set(SUPPORTED_LANGUAGES) - {UNKNOWN_LANGUAGE}))


class StimulusPreparationError(ValueError):
    """Raised when a language has no supported counting-text preparation path."""


def load_policy(path: Path = POLICY_PATH) -> dict[str, Any]:
    policy = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(policy, dict)
        or isinstance(policy.get("schema"), bool)
        or not isinstance(policy.get("schema"), int)
        or policy["schema"] != 1
    ):
        raise ValueError("voice-level policy has an unsupported schema")
    if not isinstance(policy.get("name"), str) or not policy["name"]:
        raise ValueError("policy name must be a non-empty string")
    repeats = policy.get("repeats")
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 2:
        raise ValueError("policy repeats must be an integer of at least two")
    seed = policy.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("policy seed must be an integer")
    steps = policy.get("steps")
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError("policy steps must be a positive integer")
    for name in ("reference_lufs", "min_gain_db", "max_gain_db", "max_mad_lu"):
        value = policy.get(name)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ValueError(f"policy {name} must be finite")
    if policy["min_gain_db"] > policy["max_gain_db"] or policy["max_mad_lu"] < 0:
        raise ValueError("voice-level policy limits are invalid")
    return policy


def prepare_counting_stimulus(language: str) -> dict[str, Any]:
    if language not in EXPECTED_LANGUAGES:
        raise StimulusPreparationError(
            f"{language!r} is not a supported spoken Supertonic language"
        )
    try:
        number_words = [render(number, locale=language) for number in range(1, 11)]
        counting_text = ", ".join(number_words) + "."
        prepared = prepare_language(
            counting_text,
            language=language,
            number_policy=NumberPolicy.CALLER_MANAGED,
        )
    except Exception as error:
        raise StimulusPreparationError(
            f"could not prepare counting stimulus for {language!r}: {error}"
        ) from error
    expected_warning = f"[NUMBERS] caller-managed number categories for language {language!r}"
    unexpected_warnings = tuple(
        warning for warning in prepared.warnings if warning != expected_warning
    )
    if unexpected_warnings:
        raise StimulusPreparationError(
            f"Spokenform reported unsupported preparation for {language!r}: "
            + "; ".join(unexpected_warnings)
        )
    if any(character.isdecimal() for character in prepared.spoken_text):
        raise StimulusPreparationError(
            f"counting stimulus for {language!r} still contains numeric digits"
        )
    return {
        "source_text": STIMULUS_SOURCE,
        "count_words": number_words,
        "counting_text": counting_text,
        "prepared_text": prepared.spoken_text,
        "language": language,
        "number_renderer": "numeralform",
        "text_preparer": "spokenform",
        "preparation_warnings": list(prepared.warnings),
    }


def prepare_stimuli(
    languages: Sequence[str] = EXPECTED_LANGUAGES,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, str]]]:
    stimuli: dict[str, dict[str, Any]] = {}
    failures: list[dict[str, str]] = []
    for language in languages:
        try:
            stimuli[language] = prepare_counting_stimulus(language)
        except StimulusPreparationError as error:
            failures.append(
                {
                    "language": language,
                    "phase": "stimulus_preparation",
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
    return stimuli, failures


def build_stimulus_manifest(
    languages: Sequence[str],
    stimuli: Mapping[str, Mapping[str, Any]],
    failures: Sequence[Mapping[str, str]],
    *,
    corpus: str,
) -> dict[str, Any]:
    return {
        "schema": 1,
        "corpus": corpus,
        "generated_with": generated_with(),
        "counting_source": {
            "source_text": STIMULUS_SOURCE,
            "number_range": [1, 10],
            "number_renderer": "numeralform",
            "text_preparer": "spokenform",
        },
        "supported_spoken_languages": list(EXPECTED_LANGUAGES),
        "requested_languages": list(languages),
        "unknown_language_excluded": UNKNOWN_LANGUAGE,
        "stimuli": {language: dict(row) for language, row in sorted(stimuli.items())},
        "failures": list(failures),
        "complete": set(stimuli) == set(languages) and not failures,
    }


def generated_with() -> dict[str, str]:
    names = ("supertonicsynth", "audiosig", "onnxvoice", "spokenform", "numeralform")
    versions: dict[str, str] = {}
    for name in names:
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = "unknown"
    return versions


def expected_keys(voice_refs: Sequence[str]) -> list[str]:
    return sorted(
        str(VoiceCalibrationKey(voice_ref, language))
        for voice_ref in voice_refs
        for language in EXPECTED_LANGUAGES
    )


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def aggregate_measurements(
    keys: Sequence[str],
    measurements: Sequence[Mapping[str, Any]],
    policy: Mapping[str, Any],
) -> list[dict[str, Any]]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    seen_repeats: set[tuple[str, int]] = set()
    for measurement in measurements:
        key = str(measurement["calibration_key"])
        repeat = measurement["repeat"]
        if isinstance(repeat, bool) or not isinstance(repeat, int) or repeat < 0:
            raise ValueError(f"measurement repeat for {key} must be a non-negative integer")
        identity = (key, repeat)
        if identity in seen_repeats:
            raise ValueError(f"duplicate measurement repeat {repeat} for {key}")
        seen_repeats.add(identity)
        _finite_number(measurement["integrated_lufs"], f"{key}.integrated_lufs")
        groups[key].append(measurement)

    aggregates = []
    for key in sorted(keys):
        rows = sorted(groups.get(key, ()), key=lambda row: int(row["repeat"]))
        values = [float(row["integrated_lufs"]) for row in rows]
        repeat_count = len(rows)
        median_lufs = statistics.median(values) if values else None
        mad_lu = statistics.median(abs(value - median_lufs) for value in values) if values else None
        if repeat_count != int(policy["repeats"]):
            status = "incomplete"
            gain_db = None
        elif mad_lu is not None and mad_lu > float(policy["max_mad_lu"]):
            status = "high_variability"
            gain_db = None
        else:
            status = "eligible"
            requested_gain = float(policy["reference_lufs"]) - float(median_lufs)
            gain_db = min(
                float(policy["max_gain_db"]),
                max(float(policy["min_gain_db"]), requested_gain),
            )
        voice_ref, language = key.rsplit("@", 1)
        aggregates.append(
            {
                "voice_ref": voice_ref,
                "language": language,
                "calibration_key": key,
                "median_lufs": median_lufs,
                "mad_lu": mad_lu,
                "repeat_count": repeat_count,
                "status": status,
                "gain_db": gain_db,
            }
        )
    return aggregates


def build_report(
    *,
    model: str,
    catalog_voice_refs: Sequence[str],
    selected_keys: Sequence[str],
    stimuli: Mapping[str, Mapping[str, Any]],
    stimulus_failures: Sequence[Mapping[str, str]],
    measurements: Sequence[Mapping[str, Any]],
    failures: Sequence[Mapping[str, Any]],
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    all_keys = expected_keys(catalog_voice_refs)
    aggregates = aggregate_measurements(all_keys, measurements, policy)
    fully_measured = sum(aggregate["repeat_count"] == policy["repeats"] for aggregate in aggregates)
    failed_keys = len(all_keys) - fully_measured
    return {
        "schema": 1,
        "corpus": str(policy["name"]),
        "model": model,
        "generated_with": generated_with(),
        "policy": dict(policy),
        "matrix": {
            "catalog_voice_refs": sorted(catalog_voice_refs),
            "languages": list(EXPECTED_LANGUAGES),
            "expected_keys": all_keys,
            "selected_keys": sorted(selected_keys),
        },
        "stimuli": {language: dict(row) for language, row in sorted(stimuli.items())},
        "stimulus_failures": list(stimulus_failures),
        "coverage": {
            "expected": len(all_keys),
            "selected": len(selected_keys),
            "measured": fully_measured,
            "failed": failed_keys,
            "failure_count": len(failures) + len(stimulus_failures),
            "complete": fully_measured == len(all_keys) and not failures and not stimulus_failures,
        },
        "measurements": list(measurements),
        "failures": list(failures),
        "aggregates": aggregates,
    }


def _write_report(path: Path, report: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="supertonic-3")
    parser.add_argument("--voice", action="append", help="exact managed style name; repeatable")
    parser.add_argument("--language", action="append", help="spoken language code; repeatable")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--refresh-catalog", action="store_true")
    exclusive = parser.add_mutually_exclusive_group()
    exclusive.add_argument("--list-only", action="store_true")
    exclusive.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--stimuli-output", type=Path, default=DEFAULT_STIMULI_OUTPUT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    policy = load_policy()
    languages = tuple(args.language) if args.language else EXPECTED_LANGUAGES
    unsupported = set(languages) - set(EXPECTED_LANGUAGES)
    if unsupported:
        raise SystemExit(f"unsupported calibration language(s): {sorted(unsupported)}")
    if len(set(languages)) != len(languages):
        raise SystemExit("language filters must not contain duplicates")
    if args.prepare_only and args.voice:
        raise SystemExit("--voice cannot be combined with --prepare-only")
    if args.prepare_only:
        stimuli, stimulus_failures = prepare_stimuli(languages)
        manifest = build_stimulus_manifest(
            languages,
            stimuli,
            stimulus_failures,
            corpus=str(policy["name"]),
        )
        _write_report(args.stimuli_output, manifest)
        print(f"Prepared stimuli: {len(stimuli)}")
        print(f"Preparation failures: {len(stimulus_failures)}")
        print(f"Stimulus manifest: {args.stimuli_output}")
        return 0 if manifest["complete"] else 1

    with SupertonicRuntime.from_pretrained(
        args.model,
        cache_dir=args.cache_dir,
        offline=args.offline,
        refresh_catalog=args.refresh_catalog,
    ) as runtime:
        catalog_voice_refs = {
            name: canonical_voice_ref(runtime.bundle, name) for name in runtime.voice_names
        }
        if any(ref is None for ref in catalog_voice_refs.values()):
            raise SystemExit(
                "managed bundle contains a style without a canonical semantic voice ref"
            )
        all_voice_refs = [str(catalog_voice_refs[name]) for name in runtime.voice_names]
        selected_voices = tuple(args.voice) if args.voice else runtime.voice_names
        unknown_voices = set(selected_voices) - set(runtime.voice_names)
        if unknown_voices:
            raise SystemExit(f"unknown managed voice style(s): {sorted(unknown_voices)}")
        if len(set(selected_voices)) != len(selected_voices):
            raise SystemExit("voice filters must not contain duplicates")
        if args.list_only:
            for name in selected_voices:
                voice_ref = str(catalog_voice_refs[name])
                for language in languages:
                    print(VoiceCalibrationKey(voice_ref, language))
            return 0

        stimuli, stimulus_failures = prepare_stimuli(languages)
        selected_voice_refs = [str(catalog_voice_refs[name]) for name in selected_voices]
        selected = {
            str(VoiceCalibrationKey(voice_ref, language))
            for voice_ref in selected_voice_refs
            for language in languages
        }
        measurements: list[dict[str, Any]] = []
        failures: list[dict[str, Any]] = []
        config = SynthesisConfig(
            steps=int(policy["steps"]),
            seed=int(policy["seed"]),
            normalize_audio=True,
            output_gain=1.0,
            voice_level=VoiceLevelConfig(mode="off"),
        )
        total = len(selected) * int(policy["repeats"])
        completed = 0
        stimulus_errors = {row["language"]: row for row in stimulus_failures}
        for name in selected_voices:
            voice_ref = str(catalog_voice_refs[name])
            for language in languages:
                key = str(VoiceCalibrationKey(voice_ref, language))
                if language in stimulus_errors:
                    for repeat in range(int(policy["repeats"])):
                        completed += 1
                        print(
                            f"[{completed}/{total}] {key} repeat {repeat + 1} skipped: "
                            "stimulus preparation failed",
                            flush=True,
                        )
                    continue
                text = str(stimuli[language]["prepared_text"])
                for repeat in range(int(policy["repeats"])):
                    completed += 1
                    print(f"[{completed}/{total}] {key} repeat {repeat + 1}", flush=True)
                    try:
                        result = runtime.synthesize_text(
                            text,
                            voice=name,
                            language=language,
                            config=config,
                            seed=int(policy["seed"]) + repeat,
                        )
                        loudness = measure_loudness(
                            result.audio,
                            sample_rate=result.sample_rate,
                        )
                        integrated_lufs = _finite_number(
                            loudness.integrated_lufs,
                            "integrated_lufs",
                        )
                        measurements.append(
                            {
                                "voice_ref": voice_ref,
                                "language": language,
                                "calibration_key": key,
                                "repeat": repeat,
                                "seed": int(policy["seed"]) + repeat,
                                "sample_rate": result.sample_rate,
                                "duration_seconds": result.duration,
                                "integrated_lufs": integrated_lufs,
                                "synthesis_hash": result.metadata["synthesis_hash"],
                            }
                        )
                    except Exception as error:
                        failures.append(
                            {
                                "voice_ref": voice_ref,
                                "language": language,
                                "calibration_key": key,
                                "repeat": repeat,
                                "phase": "synthesis_or_measurement",
                                "error_type": type(error).__name__,
                                "error": str(error),
                            }
                        )

    report = build_report(
        model=args.model,
        catalog_voice_refs=all_voice_refs,
        selected_keys=sorted(selected),
        stimuli=stimuli,
        stimulus_failures=stimulus_failures,
        measurements=measurements,
        failures=failures,
        policy=policy,
    )
    _write_report(args.output, report)
    print(f"Measurements: {len(measurements)}")
    print(f"Failures: {len(failures) + len(stimulus_failures)}")
    print(f"Report: {args.output}")
    return 0 if report["coverage"]["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
