from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ._onnxvoice import (
    ResolvedSupertonicBundle,
    install_pretrained_bundle,
    open_installed_bundle,
    open_local_bundle,
)
from .config import (
    AVAILABLE_LANGUAGES,
    DEFAULT_LANGUAGE,
    DEFAULT_MAX_CHUNK_LENGTH,
    DEFAULT_MAX_CHUNK_LENGTH_KO,
    DEFAULT_MODEL,
    DEFAULT_SILENCE_DURATION,
    DEFAULT_SPEED,
    DEFAULT_STEPS,
    DEFAULT_VOICE,
    MAX_TEXT_LENGTH,
)
from .errors import (
    ClosedRuntimeError,
    InvalidLanguageError,
    InvalidRequestError,
    InvalidVoiceStyleError,
    ModelInferenceError,
)
from .frontend import SupertonicFrontend
from .style import VoiceStyle, load_voice_style
from .text_split import chunk_text
from .types import RuntimeDiagnostics, SynthesisConfig, SynthesisResult


class SupertonicRuntime:
    def __init__(
        self, runtime: Any, bundle: ResolvedSupertonicBundle, *, owns_runtime: bool = True
    ) -> None:
        self._runtime = runtime
        self.bundle = bundle
        self._owns_runtime = owns_runtime
        self._closed = False
        self.frontend = SupertonicFrontend(bundle.unicode_indexer_path)
        self.sample_rate = int(bundle.sample_rate or 44_100)

    @classmethod
    def from_pretrained(
        cls,
        ref: str = DEFAULT_MODEL,
        *,
        cache_dir: str | Path | None = None,
        offline: bool = False,
        refresh_catalog: bool = False,
        force_download: bool = False,
        providers: Sequence[Any] | str | None = None,
        provider_options: Mapping[str, Any] | None = None,
        session_options: Any | None = None,
        progress: Any | None = None,
    ) -> SupertonicRuntime:
        resolved = install_pretrained_bundle(
            ref,
            cache_dir=cache_dir,
            offline=offline,
            refresh_catalog=refresh_catalog,
            force_download=force_download,
            progress=progress,
        )
        runtime = open_installed_bundle(
            resolved,
            cache_dir=cache_dir,
            providers=providers,
            provider_options=provider_options,
            session_options=session_options,
        )
        return cls(runtime, resolved)

    @classmethod
    def from_local(
        cls,
        root: str | Path,
        *,
        providers: Sequence[Any] | str | None = None,
        provider_options: Mapping[str, Any] | None = None,
        session_options: Any | None = None,
    ) -> SupertonicRuntime:
        runtime, resolved = open_local_bundle(
            root,
            providers=providers,
            provider_options=provider_options,
            session_options=session_options,
        )
        return cls(runtime, resolved)

    @property
    def voice_names(self) -> tuple[str, ...]:
        return tuple(sorted(self.bundle.style_paths))

    def get_voice_style(self, voice: str = DEFAULT_VOICE) -> VoiceStyle:
        try:
            path = self.bundle.style_paths[voice]
        except KeyError as exc:
            available = ", ".join(self.voice_names) or "none"
            raise InvalidVoiceStyleError(
                f"Unknown voice style {voice!r}; available: {available}"
            ) from exc
        return load_voice_style(path, name=voice)

    def synthesize_text(
        self,
        text: str,
        *,
        voice: str | VoiceStyle = DEFAULT_VOICE,
        language: str = DEFAULT_LANGUAGE,
        config: SynthesisConfig | None = None,
        steps: int | None = None,
        speed: float | None = None,
        max_chunk_length: int | None = None,
        silence_duration: float | None = None,
        seed: int | None = None,
    ) -> SynthesisResult:
        if self._closed:
            raise ClosedRuntimeError("runtime is closed")
        if not isinstance(text, str) or not text.strip():
            raise InvalidRequestError("text cannot be empty")
        if len(text) > MAX_TEXT_LENGTH:
            raise InvalidRequestError(f"text exceeds maximum length {MAX_TEXT_LENGTH}")
        if language not in AVAILABLE_LANGUAGES:
            raise InvalidLanguageError(f"Unsupported language {language!r}")
        base = config or SynthesisConfig()
        effective = SynthesisConfig(
            steps=base.steps if steps is None else steps,
            speed=base.speed if speed is None else speed,
            max_chunk_length=base.max_chunk_length
            if max_chunk_length is None
            else max_chunk_length,
            silence_duration=base.silence_duration
            if silence_duration is None
            else silence_duration,
            seed=base.seed if seed is None else seed,
        )
        style = self.get_voice_style(voice) if isinstance(voice, str) else voice
        limit = effective.max_chunk_length or (
            DEFAULT_MAX_CHUNK_LENGTH_KO if language == "ko" else DEFAULT_MAX_CHUNK_LENGTH
        )
        chunks = chunk_text(text, limit)
        if not chunks:
            raise InvalidRequestError("text produced no synthesis chunks")
        audio_parts: list[np.ndarray] = []
        for index, chunk in enumerate(chunks):
            batch = self.frontend.encode(chunk, language)
            call_seed = None if effective.seed is None else effective.seed + index
            try:
                result = self._runtime.infer(
                    batch.text_ids.tolist(),
                    text_mask=batch.text_mask,
                    style_ttl=style.ttl,
                    style_dp=style.dp,
                    steps=effective.steps,
                    speed=effective.speed,
                    seed=call_seed,
                )
            except Exception as exc:
                if isinstance(exc, (ValueError, TypeError)):
                    raise
                raise ModelInferenceError(str(exc)) from exc
            audio = np.asarray(getattr(result, "audio"), dtype=np.float32).squeeze()
            if audio.ndim != 1 or audio.size == 0 or not np.all(np.isfinite(audio)):
                raise ModelInferenceError("runtime returned invalid audio")
            result_rate = int(getattr(result, "sample_rate", self.sample_rate))
            if result_rate != self.sample_rate:
                raise ModelInferenceError(
                    f"runtime sample rate changed from {self.sample_rate} to {result_rate}"
                )
            audio_parts.append(audio)
        if effective.silence_duration > 0 and len(audio_parts) > 1:
            silence = np.zeros(
                int(round(effective.silence_duration * self.sample_rate)), dtype=np.float32
            )
            interleaved: list[np.ndarray] = []
            for index, audio in enumerate(audio_parts):
                interleaved.append(audio)
                if index + 1 < len(audio_parts):
                    interleaved.append(silence)
            audio_parts = interleaved
        audio = np.concatenate(audio_parts).astype(np.float32, copy=False)
        voice_name = style.name if isinstance(voice, VoiceStyle) else voice
        return SynthesisResult(
            audio=audio,
            sample_rate=self.sample_rate,
            duration=float(audio.size / self.sample_rate),
            chunks=len(chunks),
            metadata={
                "bundle": self.bundle.ref or self.bundle.bundle_id,
                "voice": voice_name,
                "language": language,
                "steps": effective.steps,
                "speed": effective.speed,
                "seed": effective.seed,
            },
        )

    def diagnostics(self) -> RuntimeDiagnostics:
        raw = self._runtime.diagnostics() if hasattr(self._runtime, "diagnostics") else None
        sessions: tuple[str, ...] = ()
        providers: tuple[str, ...] = ()
        if raw is not None:
            raw_sessions = getattr(raw, "sessions", ())
            sessions = tuple(
                str(getattr(item, "component", getattr(item, "name", item)))
                for item in raw_sessions
            )
            providers = tuple(getattr(self._runtime, "providers", ()) or ())
        return RuntimeDiagnostics(
            bundle_ref=self.bundle.ref, providers_requested=providers, sessions=sessions, raw=raw
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._owns_runtime:
            close = getattr(self._runtime, "close", None)
            if close is not None:
                close()

    def __enter__(self) -> SupertonicRuntime:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
