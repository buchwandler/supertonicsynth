from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files
from pathlib import Path
from typing import Any, Literal

import audiosig
import numpy as np

from .audio import prepare_audio

_SUPPORTED_SCHEMA = 1
_SUPPORTED_METHOD = "bs1770"
_TOP_FIELDS = {"schema", "method", "corpus", "reference_lufs", "generated_with", "voices"}
_RECORD_FIELDS = {
    "gain_db",
    "measured_lufs",
    "reference_lufs",
    "mad_lu",
    "samples",
    "method",
    "corpus_version",
}
_SAFE_COMPONENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")
_LANGUAGE_TAG = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*\Z")

VoiceLevelMode = Literal["off", "calibrated"]
VoiceLevelSource = Literal["off", "override", "catalog", "missing_identity", "missing_calibration"]


@dataclass(frozen=True, slots=True)
class VoiceLevelConfig:
    """Static engine-local gain calibration settings."""

    mode: VoiceLevelMode = "off"
    gain_db: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.mode, str) or self.mode not in {"off", "calibrated"}:
            raise ValueError("mode must be 'off' or 'calibrated'")
        if self.gain_db is not None:
            if isinstance(self.gain_db, bool) or not isinstance(self.gain_db, (int, float)):
                raise ValueError("gain_db must be a finite number or None")
            if not np.isfinite(self.gain_db):
                raise ValueError("gain_db must be finite")


class CalibrationDataError(ValueError):
    """Raised when calibration data is malformed or unsafe."""


def _validate_voice_ref(value: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise CalibrationDataError("voice_ref must be a non-empty normalized semantic ref")
    if not value.startswith("supertonic:") or value.count(":") != 1:
        raise CalibrationDataError("voice_ref must have supertonic:<bundle-id>/<voice-id> form")
    bundle_and_voice = value.removeprefix("supertonic:")
    if bundle_and_voice.count("/") != 1:
        raise CalibrationDataError("voice_ref must contain exactly one child-voice separator")
    bundle_id, voice_id = bundle_and_voice.split("/", 1)
    if not _SAFE_COMPONENT.fullmatch(bundle_id) or not _SAFE_COMPONENT.fullmatch(voice_id):
        raise CalibrationDataError("voice_ref contains an empty or unsafe bundle/style ID")
    return value


def _normalize_language(value: str) -> str:
    if not isinstance(value, str):
        raise CalibrationDataError("language must be a non-empty normalized language tag")
    normalized = value.strip().lower()
    if not normalized or not _LANGUAGE_TAG.fullmatch(normalized):
        raise CalibrationDataError("language must be a non-empty normalized language tag")
    return normalized


@dataclass(frozen=True, slots=True, order=True)
class VoiceCalibrationKey:
    voice_ref: str
    language: str

    def __post_init__(self) -> None:
        _validate_voice_ref(self.voice_ref)
        object.__setattr__(self, "language", _normalize_language(self.language))

    def __str__(self) -> str:
        return f"{self.voice_ref}@{self.language}"

    @classmethod
    def parse(cls, value: str) -> VoiceCalibrationKey:
        if not isinstance(value, str) or value.count("@") != 1:
            raise CalibrationDataError("calibration key must have <voice-ref>@<language> form")
        voice_ref, language = value.split("@", 1)
        return cls(voice_ref=voice_ref, language=language)


@dataclass(frozen=True, slots=True)
class VoiceLevelCalibration:
    gain_db: float
    measured_lufs: float | None = None
    reference_lufs: float | None = None
    mad_lu: float | None = None
    samples: int | None = None
    method: str = _SUPPORTED_METHOD
    corpus_version: str | None = None


@dataclass(frozen=True, slots=True)
class VoiceCalibrationCatalog:
    schema: int
    method: str
    corpus: str
    reference_lufs: float
    generated_with: Mapping[str, str]
    voices: Mapping[VoiceCalibrationKey, VoiceLevelCalibration]
    revision: str | None = None


@dataclass(frozen=True, slots=True)
class VoiceLevelApplication:
    applied: bool
    gain_db: float
    source: VoiceLevelSource
    key: VoiceCalibrationKey | None
    mode: VoiceLevelMode = "off"
    catalog_revision: str | None = None
    reason: str = ""

    @property
    def calibration_key(self) -> VoiceCalibrationKey | None:
        return self.key


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CalibrationDataError(f"{name} must be a finite number")
    converted = float(value)
    if not np.isfinite(converted):
        raise CalibrationDataError(f"{name} must be finite")
    return converted


def _optional_finite(value: Any, name: str) -> float | None:
    return None if value is None else _finite(value, name)


def _object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CalibrationDataError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _validate_record(key: VoiceCalibrationKey, raw: Any) -> VoiceLevelCalibration:
    if not isinstance(raw, Mapping):
        raise CalibrationDataError(f"record for {key} must be an object")
    unknown = set(raw) - _RECORD_FIELDS
    if unknown:
        raise CalibrationDataError(f"record for {key} has unknown field(s): {sorted(unknown)}")
    if "gain_db" not in raw:
        raise CalibrationDataError(f"record for {key} is missing gain_db")
    method = raw.get("method", _SUPPORTED_METHOD)
    if method != _SUPPORTED_METHOD:
        raise CalibrationDataError(f"record for {key} has unsupported method: {method!r}")
    samples = raw.get("samples")
    if samples is not None and (
        isinstance(samples, bool) or not isinstance(samples, int) or samples <= 0
    ):
        raise CalibrationDataError(f"record for {key} samples must be a positive integer")
    corpus_version = raw.get("corpus_version")
    if corpus_version is not None and (not isinstance(corpus_version, str) or not corpus_version):
        raise CalibrationDataError(f"record for {key} corpus_version must be a non-empty string")
    return VoiceLevelCalibration(
        gain_db=_finite(raw["gain_db"], f"{key}.gain_db"),
        measured_lufs=_optional_finite(raw.get("measured_lufs"), f"{key}.measured_lufs"),
        reference_lufs=_optional_finite(raw.get("reference_lufs"), f"{key}.reference_lufs"),
        mad_lu=_optional_finite(raw.get("mad_lu"), f"{key}.mad_lu"),
        samples=samples,
        method=method,
        corpus_version=corpus_version,
    )


def _canonical_catalog(
    *,
    corpus: str,
    generated_with: Mapping[str, str],
    reference_lufs: float,
    voices: Mapping[VoiceCalibrationKey, VoiceLevelCalibration],
) -> dict[str, Any]:
    return {
        "schema": _SUPPORTED_SCHEMA,
        "method": _SUPPORTED_METHOD,
        "corpus": corpus,
        "reference_lufs": reference_lufs,
        "generated_with": dict(sorted(generated_with.items())),
        "voices": {
            str(key): {
                "gain_db": record.gain_db,
                "measured_lufs": record.measured_lufs,
                "reference_lufs": record.reference_lufs,
                "mad_lu": record.mad_lu,
                "samples": record.samples,
                "method": record.method,
                "corpus_version": record.corpus_version,
            }
            for key, record in sorted(voices.items(), key=lambda item: str(item[0]))
        },
    }


def _validate_catalog(raw: Any) -> VoiceCalibrationCatalog:
    if not isinstance(raw, Mapping):
        raise CalibrationDataError("calibration catalog must be an object")
    unknown = set(raw) - _TOP_FIELDS
    if unknown:
        raise CalibrationDataError(f"calibration catalog has unknown field(s): {sorted(unknown)}")
    missing = _TOP_FIELDS - set(raw)
    if missing:
        raise CalibrationDataError(f"calibration catalog is missing field(s): {sorted(missing)}")
    if isinstance(raw["schema"], bool) or not isinstance(raw["schema"], int):
        raise CalibrationDataError("schema must be an integer")
    if raw["schema"] != _SUPPORTED_SCHEMA:
        raise CalibrationDataError(f"unsupported calibration schema: {raw['schema']!r}")
    if raw["method"] != _SUPPORTED_METHOD:
        raise CalibrationDataError(f"unsupported calibration method: {raw['method']!r}")
    corpus = raw["corpus"]
    if not isinstance(corpus, str) or not corpus:
        raise CalibrationDataError("corpus must be a non-empty string")
    generated_with = raw["generated_with"]
    if not isinstance(generated_with, Mapping) or any(
        not isinstance(key, str) or not key or not isinstance(value, str) or not value
        for key, value in generated_with.items()
    ):
        raise CalibrationDataError("generated_with must be a non-empty-string mapping")
    voices_raw = raw["voices"]
    if not isinstance(voices_raw, Mapping):
        raise CalibrationDataError("voices must be an object")
    voices: dict[VoiceCalibrationKey, VoiceLevelCalibration] = {}
    for raw_key, record in voices_raw.items():
        key = VoiceCalibrationKey.parse(raw_key)
        if key in voices:
            raise CalibrationDataError(f"duplicate normalized calibration key: {key}")
        voices[key] = _validate_record(key, record)
    reference_lufs = _finite(raw["reference_lufs"], "reference_lufs")
    canonical = _canonical_catalog(
        corpus=corpus,
        generated_with=generated_with,
        reference_lufs=reference_lufs,
        voices=voices,
    )
    serialized = json.dumps(
        canonical,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return VoiceCalibrationCatalog(
        schema=_SUPPORTED_SCHEMA,
        method=_SUPPORTED_METHOD,
        corpus=corpus,
        reference_lufs=reference_lufs,
        generated_with=dict(sorted(generated_with.items())),
        voices=dict(sorted(voices.items(), key=lambda item: str(item[0]))),
        revision=hashlib.sha256(serialized).hexdigest(),
    )


def _load_json(handle: Any, source: Any) -> VoiceCalibrationCatalog:
    try:
        raw = json.load(handle, object_pairs_hook=_object_pairs)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise CalibrationDataError(f"invalid calibration JSON: {source}") from exc
    return _validate_catalog(raw)


def load_voice_calibration(path: Path | str) -> VoiceCalibrationCatalog:
    """Load and strictly validate a runtime calibration catalog."""
    source = Path(path)
    try:
        with source.open("r", encoding="utf-8") as handle:
            return _load_json(handle, source)
    except UnicodeDecodeError as exc:
        raise CalibrationDataError(f"invalid calibration JSON: {source}") from exc


@lru_cache(maxsize=1)
def default_voice_calibration() -> VoiceCalibrationCatalog:
    resource = files("supertonicsynth").joinpath("data", "voice_level_calibration.json")
    with resource.open("r", encoding="utf-8") as handle:
        return _load_json(handle, resource)


def apply_voice_level_calibration(
    audio: np.ndarray,
    config: VoiceLevelConfig,
    key: VoiceCalibrationKey | None,
    *,
    catalog: VoiceCalibrationCatalog | None = None,
) -> tuple[np.ndarray, VoiceLevelApplication]:
    """Apply one deterministic static gain without measuring the waveform."""
    result = prepare_audio(audio, normalize=False)
    catalog_revision = None
    if config.gain_db is not None:
        gain = float(config.gain_db)
        source: VoiceLevelSource = "override"
    elif config.mode != "calibrated":
        gain = 0.0
        source = "off"
    elif key is None:
        gain = 0.0
        source = "missing_identity"
    else:
        selected_catalog = catalog if catalog is not None else default_voice_calibration()
        catalog_revision = selected_catalog.revision
        selected = selected_catalog.voices.get(key)
        if selected is None:
            gain = 0.0
            source = "missing_calibration"
        else:
            gain = selected.gain_db
            source = "catalog"
    if gain:
        result = np.asarray(audiosig.apply_gain_db(result, gain, clip=False), dtype=np.float32)
    else:
        result = result.copy()
    reasons = {
        "off": "voice-level calibration is disabled",
        "override": "an explicit gain_db override was selected",
        "catalog": "a matching calibration was selected",
        "missing_identity": "the voice has no stable managed semantic identity",
        "missing_calibration": "no calibration entry matches this voice/language",
    }
    application = VoiceLevelApplication(
        applied=bool(gain),
        gain_db=gain,
        source=source,
        key=key,
        mode=config.mode,
        catalog_revision=catalog_revision,
        reason=reasons[source],
    )
    return result, application


__all__ = [
    "CalibrationDataError",
    "VoiceCalibrationCatalog",
    "VoiceCalibrationKey",
    "VoiceLevelApplication",
    "VoiceLevelCalibration",
    "VoiceLevelConfig",
    "VoiceLevelMode",
    "apply_voice_level_calibration",
    "default_voice_calibration",
    "load_voice_calibration",
]
