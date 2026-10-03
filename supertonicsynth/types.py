from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from numbers import Real
from pathlib import Path
from typing import Any, Literal

import numpy as np

from .audio import float_to_int16, write_wav
from .config import (
    AVAILABLE_LANGUAGES,
    DEFAULT_SPEED,
    DEFAULT_STEPS,
    MAX_SEED,
    MAX_SPEED,
    MAX_STEPS,
    MIN_SPEED,
    MIN_STEPS,
)
from .errors import (
    EmptyTextError,
    InvalidGenerationConfigError,
    InvalidLanguageError,
    InvalidRequestError,
)
from .voice_level import VoiceLevelConfig


@dataclass(frozen=True, slots=True)
class SynthesisRequest:
    """One caller-shaped atomic Supertonic synthesis request."""

    id: str
    text: str
    language: str

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id.strip():
            raise InvalidRequestError("request id must be a non-empty string")
        if not isinstance(self.text, str):
            raise EmptyTextError("request text must be a string")
        if not self.text.strip():
            raise EmptyTextError("request text must not be empty or whitespace-only")
        if not isinstance(self.language, str) or not self.language.strip():
            raise InvalidLanguageError("language must be a non-empty string")

        if self.language not in AVAILABLE_LANGUAGES:
            raise InvalidLanguageError(f"unsupported language {self.language!r}")


@dataclass(frozen=True, slots=True)
class GenerationConfig:
    """Model generation controls for one atomic inference."""

    steps: int = DEFAULT_STEPS
    speed: float = DEFAULT_SPEED
    seed: int | None = None

    def __post_init__(self) -> None:
        if isinstance(self.steps, bool) or not isinstance(self.steps, int):
            raise InvalidGenerationConfigError("steps must be an integer")
        if not MIN_STEPS <= self.steps <= MAX_STEPS:
            raise InvalidGenerationConfigError(f"steps must be between {MIN_STEPS} and {MAX_STEPS}")
        if (
            isinstance(self.speed, bool)
            or not isinstance(self.speed, Real)
            or not np.isfinite(self.speed)
        ):
            raise InvalidGenerationConfigError("speed must be a finite real number")
        if not MIN_SPEED <= self.speed <= MAX_SPEED:
            raise InvalidGenerationConfigError(f"speed must be between {MIN_SPEED} and {MAX_SPEED}")
        if self.seed is not None:
            if isinstance(self.seed, bool) or not isinstance(self.seed, int):
                raise InvalidGenerationConfigError("seed must be an integer or None")
            if not 0 <= self.seed <= MAX_SEED:
                raise InvalidGenerationConfigError(f"seed must be between 0 and {MAX_SEED}")


@dataclass(frozen=True, slots=True)
class RequestMeasure:
    """Encoded request size and its relationship to a known model capacity."""

    amount: int
    maximum: int | None
    unit: Literal["tokens"] = "tokens"
    fits: bool | None = None

    def __post_init__(self) -> None:
        if isinstance(self.amount, bool) or not isinstance(self.amount, int) or self.amount < 0:
            raise InvalidRequestError("amount must be a non-negative integer")
        if self.maximum is not None and (
            isinstance(self.maximum, bool) or not isinstance(self.maximum, int) or self.maximum <= 0
        ):
            raise InvalidRequestError("maximum must be a positive integer or None")
        if self.unit != "tokens":
            raise InvalidRequestError("unit must be 'tokens'")
        expected_fits = None if self.maximum is None else self.amount <= self.maximum
        if self.fits is not None and (
            not isinstance(self.fits, bool) or self.fits != expected_fits
        ):
            raise InvalidRequestError("fits must match amount and maximum")
        object.__setattr__(self, "fits", expected_fits)


@dataclass(slots=True)
class AtomicSynthesisResult:
    """One atomic Supertonic inference result."""

    id: str
    audio: np.ndarray
    sample_rate: int
    text: str
    language: str
    warnings: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id.strip():
            raise InvalidRequestError("result id must be a non-empty string")
        if not isinstance(self.text, str):
            raise InvalidRequestError("result text must be a string")
        if not isinstance(self.language, str) or not self.language.strip():
            raise InvalidRequestError("result language must be a non-empty string")
        if (
            isinstance(self.sample_rate, bool)
            or not isinstance(self.sample_rate, int)
            or self.sample_rate <= 0
        ):
            raise InvalidRequestError("sample_rate must be a positive integer")
        try:
            audio = np.asarray(self.audio, dtype=np.float32)
        except (TypeError, ValueError, OverflowError) as exc:
            raise InvalidRequestError("audio must be a float32-compatible vector") from exc
        if audio.ndim != 1 or audio.size == 0:
            raise InvalidRequestError("audio must be a non-empty one-dimensional vector")
        if not np.all(np.isfinite(audio)):
            raise InvalidRequestError("audio must contain only finite samples")
        if not isinstance(self.warnings, tuple) or any(
            not isinstance(warning, str) for warning in self.warnings
        ):
            raise InvalidRequestError("warnings must be a tuple of strings")
        if not isinstance(self.metadata, Mapping):
            raise InvalidRequestError("metadata must be a mapping")
        self.audio = audio
        self.metadata = dict(self.metadata)

    @property
    def duration_seconds(self) -> float:
        return float(self.audio.size / self.sample_rate)

    def pcm16(self) -> np.ndarray:
        return float_to_int16(self.audio)

    def write_wav(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        write_wav(target, self.audio, self.sample_rate)
        return target


@dataclass(frozen=True, slots=True)
class SynthesisConfig:
    steps: int = 5
    speed: float = 1.05
    max_chunk_length: int | None = None
    silence_duration: float = 0.3
    seed: int | None = None
    normalize_audio: bool = True
    output_gain: float = 1.0
    voice_level: VoiceLevelConfig = field(default_factory=VoiceLevelConfig)

    def __post_init__(self) -> None:
        if isinstance(self.steps, bool) or not isinstance(self.steps, int):
            raise ValueError("steps must be an integer")
        if not MIN_STEPS <= self.steps <= MAX_STEPS:
            raise ValueError(f"steps must be between {MIN_STEPS} and {MAX_STEPS}")
        if (
            isinstance(self.speed, bool)
            or not isinstance(self.speed, Real)
            or not np.isfinite(self.speed)
        ):
            raise ValueError("speed must be a finite real number")
        if not MIN_SPEED <= self.speed <= MAX_SPEED:
            raise ValueError(f"speed must be between {MIN_SPEED} and {MAX_SPEED}")
        if self.max_chunk_length is not None:
            if isinstance(self.max_chunk_length, bool) or not isinstance(
                self.max_chunk_length, int
            ):
                raise ValueError("max_chunk_length must be an integer or None")
            if self.max_chunk_length < 10:
                raise ValueError("max_chunk_length must be at least 10")
        if (
            isinstance(self.silence_duration, bool)
            or not isinstance(self.silence_duration, Real)
            or not np.isfinite(self.silence_duration)
        ):
            raise ValueError("silence_duration must be a finite real number")
        if self.silence_duration < 0:
            raise ValueError("silence_duration must be non-negative")
        if self.seed is not None:
            if isinstance(self.seed, bool) or not isinstance(self.seed, int):
                raise ValueError("seed must be an integer or None")
            if not 0 <= self.seed <= MAX_SEED:
                raise ValueError(f"seed must be between 0 and {MAX_SEED}")
        if not isinstance(self.normalize_audio, bool):
            raise ValueError("normalize_audio must be a bool")
        if (
            isinstance(self.output_gain, bool)
            or not isinstance(self.output_gain, Real)
            or not np.isfinite(self.output_gain)
            or self.output_gain < 0
        ):
            raise ValueError("output_gain must be a finite non-negative number")
        if not isinstance(self.voice_level, VoiceLevelConfig):
            raise ValueError("voice_level must be a VoiceLevelConfig")


@dataclass(frozen=True, slots=True)
class SynthesisResult:
    audio: np.ndarray
    sample_rate: int
    duration: float
    chunks: int
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def pcm16(self) -> np.ndarray:
        return float_to_int16(self.audio)

    def write_wav(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        write_wav(target, self.audio, self.sample_rate)
        return target


@dataclass(frozen=True, slots=True)
class RuntimeDiagnostics:
    bundle_ref: str | None
    providers_requested: tuple[str, ...]
    sessions: tuple[str, ...]
    raw: Any = None
