from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .errors import InvalidVoiceStyleError


@dataclass(frozen=True, slots=True)
class VoiceStyle:
    ttl: np.ndarray
    dp: np.ndarray
    name: str | None = None
    path: Path | None = None


def _array(value: object, key: str) -> np.ndarray:
    if not isinstance(value, dict):
        raise InvalidVoiceStyleError(f"{key} must be an object")
    dims = value.get("dims")
    data = value.get("data")
    if (
        not isinstance(dims, list)
        or not dims
        or not all(isinstance(v, int) and v > 0 for v in dims)
    ):
        raise InvalidVoiceStyleError(f"{key}.dims must contain positive integers")
    try:
        array = np.asarray(data, dtype=np.float32).reshape(*dims)
    except (TypeError, ValueError) as exc:
        raise InvalidVoiceStyleError(f"{key} has invalid data/dims: {exc}") from exc
    if array.shape[0] != 1:
        raise InvalidVoiceStyleError(f"{key} batch dimension must be 1")
    if not np.all(np.isfinite(array)):
        raise InvalidVoiceStyleError(f"{key} contains non-finite values")
    return np.ascontiguousarray(array)


def load_voice_style(path: str | Path, *, name: str | None = None) -> VoiceStyle:
    source = Path(path)
    try:
        data = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise InvalidVoiceStyleError(f"Could not read voice style {source}: {exc}") from exc
    if not isinstance(data, dict):
        raise InvalidVoiceStyleError("voice style JSON must contain an object")
    try:
        ttl = _array(data["style_ttl"], "style_ttl")
        dp = _array(data["style_dp"], "style_dp")
    except KeyError as exc:
        raise InvalidVoiceStyleError(f"voice style is missing {exc.args[0]!r}") from exc
    return VoiceStyle(ttl=ttl, dp=dp, name=name or source.stem, path=source)
