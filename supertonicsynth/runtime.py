from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from . import __version__
from ._onnxvoice import (
    ResolvedSupertonicBundle,
    install_pretrained_bundle,
    open_installed_bundle,
    open_local_bundle,
)
from .audio import finish_audio, prepare_audio
from .config import (
    AVAILABLE_LANGUAGES,
    DEFAULT_LANGUAGE,
    DEFAULT_MAX_CHUNK_LENGTH,
    DEFAULT_MAX_CHUNK_LENGTH_KO,
    DEFAULT_MODEL,
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
from .voice_level import (
    VoiceCalibrationKey,
    VoiceLevelApplication,
    VoiceLevelConfig,
    apply_voice_level_calibration,
)


def _is_managed_bundle(bundle: ResolvedSupertonicBundle) -> bool:
    installation = bundle.installation
    return (
        installation is not None
        and bundle.metadata.get("managed") is not False
        and bundle.metadata.get("external") is not True
        and getattr(installation, "kind", None) != "external"
    )


def canonical_backing_ref(bundle: ResolvedSupertonicBundle) -> str | None:
    if not _is_managed_bundle(bundle) or not bundle.bundle_id:
        return None
    return f"supertonic:{bundle.bundle_id}"


def canonical_voice_ref(
    bundle: ResolvedSupertonicBundle,
    voice: str | VoiceStyle,
) -> str | None:
    backing_ref = canonical_backing_ref(bundle)
    if not isinstance(voice, str) or backing_ref is None or voice not in bundle.style_paths:
        return None
    return f"{backing_ref}/{voice}"


def _synthesis_hash(identity: Mapping[str, Any], source_text: str) -> str:
    serialized = json.dumps(
        identity,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(serialized + b"\0" + source_text.encode("utf-8")).hexdigest()


class SupertonicRuntime:
    def __init__(
        self, runtime: Any, bundle: ResolvedSupertonicBundle, *, owns_runtime: bool = True
    ) -> None:
        self._runtime = runtime
        self.bundle = bundle
        self._owns_runtime = owns_runtime
        self._closed = False
        self._last_voice_level_application: VoiceLevelApplication | None = None
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

    @property
    def last_voice_level_application(self) -> VoiceLevelApplication | None:
        return self._last_voice_level_application

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
        normalize_audio: bool | None = None,
        output_gain: float | None = None,
        voice_level: VoiceLevelConfig | None = None,
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
            max_chunk_length=(
                base.max_chunk_length if max_chunk_length is None else max_chunk_length
            ),
            silence_duration=(
                base.silence_duration if silence_duration is None else silence_duration
            ),
            seed=base.seed if seed is None else seed,
            normalize_audio=(base.normalize_audio if normalize_audio is None else normalize_audio),
            output_gain=base.output_gain if output_gain is None else output_gain,
            voice_level=base.voice_level if voice_level is None else voice_level,
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
            audio = np.asarray(result.audio, dtype=np.float32).squeeze()
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
            for index, part in enumerate(audio_parts):
                interleaved.append(part)
                if index + 1 < len(audio_parts):
                    interleaved.append(silence)
            audio_parts = interleaved
        audio = np.concatenate(audio_parts).astype(np.float32, copy=False)
        audio = prepare_audio(audio, normalize=effective.normalize_audio)
        backing_ref = canonical_backing_ref(self.bundle)
        voice_ref = canonical_voice_ref(self.bundle, voice)
        calibration_key = (
            VoiceCalibrationKey(voice_ref, language) if voice_ref is not None else None
        )
        audio, application = apply_voice_level_calibration(
            audio, effective.voice_level, calibration_key
        )
        audio = finish_audio(audio, output_gain=effective.output_gain)
        self._last_voice_level_application = application
        voice_name = style.name if isinstance(voice, VoiceStyle) else voice
        application_key = application.calibration_key
        voice_level_metadata = {
            "mode": application.mode,
            "applied": application.applied,
            "gain_db": application.gain_db,
            "source": application.source,
            "calibration_key": str(application_key) if application_key else None,
            "voice_ref": application_key.voice_ref if application_key else None,
            "language": application_key.language if application_key else language,
            "reason": application.reason,
            "catalog_revision": application.catalog_revision,
        }
        synthesis_identity = {
            "supertonicsynth_version": __version__,
            "voice_ref": voice_ref,
            "backing_ref": backing_ref,
            "source_revision": self.bundle.source_revision,
            "language": language,
            "steps": effective.steps,
            "speed": effective.speed,
            "max_chunk_length": limit,
            "silence_duration": effective.silence_duration,
            "seed": effective.seed,
            "normalize_audio": effective.normalize_audio,
            "output_gain": effective.output_gain,
            "voice_level": {
                "mode": application.mode,
                "source": application.source,
                "gain_db": application.gain_db,
                "calibration_key": str(application_key) if application_key else None,
                "catalog_revision": application.catalog_revision,
            },
        }
        return SynthesisResult(
            audio=audio,
            sample_rate=self.sample_rate,
            duration=float(audio.size / self.sample_rate),
            chunks=len(chunks),
            metadata={
                "bundle": backing_ref or self.bundle.ref or self.bundle.bundle_id,
                "voice": voice_name,
                "voice_ref": voice_ref,
                "backing_ref": backing_ref,
                "language": language,
                "steps": effective.steps,
                "speed": effective.speed,
                "seed": effective.seed,
                "normalize_audio": effective.normalize_audio,
                "output_gain": effective.output_gain,
                "voice_level": voice_level_metadata,
                "synthesis_identity": synthesis_identity,
                "synthesis_hash": _synthesis_hash(synthesis_identity, text),
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
