from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from numbers import Real
from pathlib import Path
from typing import Any

import numpy as np

from .audio import float_to_int16, write_wav
from .config import MAX_SEED, MAX_SPEED, MAX_STEPS, MIN_SPEED, MIN_STEPS
from .voice_level import VoiceLevelConfig


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
