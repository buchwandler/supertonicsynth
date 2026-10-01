from __future__ import annotations

from pathlib import Path
from typing import Any

from .runtime import SupertonicRuntime
from .types import SynthesisConfig, SynthesisResult


def synthesize(
    text: str,
    *,
    model: str = "supertonic-3",
    voice: str = "M1",
    language: str = "na",
    config: SynthesisConfig | None = None,
    cache_dir: str | Path | None = None,
    **runtime_options: Any,
) -> SynthesisResult:
    with SupertonicRuntime.from_pretrained(
        model, cache_dir=cache_dir, **runtime_options
    ) as runtime:
        return runtime.synthesize_text(text, voice=voice, language=language, config=config)
