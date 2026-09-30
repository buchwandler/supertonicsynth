from __future__ import annotations

import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from collections.abc import Mapping

import numpy as np

from .config import MAX_SPEED, MAX_STEPS, MIN_SPEED, MIN_STEPS


@dataclass(frozen=True, slots=True)
class SynthesisConfig:
    steps: int = 5
    speed: float = 1.05
    max_chunk_length: int | None = None
    silence_duration: float = 0.3
    seed: int | None = None

    def __post_init__(self) -> None:
        if not MIN_STEPS <= self.steps <= MAX_STEPS:
            raise ValueError(f"steps must be between {MIN_STEPS} and {MAX_STEPS}")
        if not MIN_SPEED <= self.speed <= MAX_SPEED:
            raise ValueError(f"speed must be between {MIN_SPEED} and {MAX_SPEED}")
        if self.max_chunk_length is not None and self.max_chunk_length < 10:
            raise ValueError("max_chunk_length must be at least 10")
        if self.silence_duration < 0:
            raise ValueError("silence_duration must be non-negative")


@dataclass(frozen=True, slots=True)
class SynthesisResult:
    audio: np.ndarray
    sample_rate: int
    duration: float
    chunks: int
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def pcm16(self) -> np.ndarray:
        audio = np.asarray(self.audio, dtype=np.float32).reshape(-1)
        return (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)

    def write_wav(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(target), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(self.sample_rate)
            wav.writeframes(self.pcm16().tobytes())
        return target


@dataclass(frozen=True, slots=True)
class RuntimeDiagnostics:
    bundle_ref: str | None
    providers_requested: tuple[str, ...]
    sessions: tuple[str, ...]
    raw: Any = None
