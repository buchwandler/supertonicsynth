from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from types import MappingProxyType
from typing import Any

from ._onnxvoice import list_pretrained_bundles
from .config import UNKNOWN_LANGUAGE
from .errors import InvalidLanguageError, RuntimeCapabilityError


@dataclass(frozen=True, slots=True)
class DescribedVoice:
    id: str
    gender: str = "unknown"
    language: str = "unknown"
    locale: str = "unknown"
    language_label: str = "unknown"
    languages: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id:
            raise RuntimeCapabilityError("discovered voice id must be a non-empty string")
        for field_name in ("gender", "language", "locale", "language_label"):
            if not isinstance(getattr(self, field_name), str):
                raise RuntimeCapabilityError(f"voice {field_name} must be a string")
        if not isinstance(self.languages, tuple) or any(
            not isinstance(language, str) or not language for language in self.languages
        ):
            raise RuntimeCapabilityError("voice languages must be a tuple of non-empty strings")
        if not isinstance(self.metadata, Mapping):
            raise RuntimeCapabilityError("voice metadata must be a mapping")
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True, slots=True)
class DiscoveredModel:
    id: str
    ref: str
    display_name: str
    version: str | None
    sample_rate: int | None
    aliases: tuple[str, ...]
    languages: tuple[str, ...]
    voices: tuple[DescribedVoice, ...]
    default_voice: str | None
    source_revision: str | None
    max_input_tokens: int | None
    runtime_available: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for field_name in ("id", "ref", "display_name"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value:
                raise RuntimeCapabilityError(f"model {field_name} must be a non-empty string")
        for field_name in ("version", "default_voice", "source_revision"):
            value = getattr(self, field_name)
            if value is not None and (not isinstance(value, str) or not value):
                raise RuntimeCapabilityError(
                    f"model {field_name} must be a non-empty string or None"
                )
        if self.sample_rate is not None and (
            isinstance(self.sample_rate, bool)
            or not isinstance(self.sample_rate, int)
            or self.sample_rate <= 0
        ):
            raise RuntimeCapabilityError("model sample_rate must be a positive integer or None")
        if self.max_input_tokens is not None and (
            isinstance(self.max_input_tokens, bool)
            or not isinstance(self.max_input_tokens, int)
            or self.max_input_tokens <= 0
        ):
            raise RuntimeCapabilityError(
                "model max_input_tokens must be a positive integer or None"
            )
        if not isinstance(self.aliases, tuple) or any(
            not isinstance(alias, str) or not alias for alias in self.aliases
        ):
            raise RuntimeCapabilityError("model aliases must be a tuple of non-empty strings")
        if not isinstance(self.languages, tuple) or any(
            not isinstance(language, str) or not language for language in self.languages
        ):
            raise RuntimeCapabilityError("model languages must be a tuple of non-empty strings")
        if not isinstance(self.voices, tuple) or any(
            not isinstance(voice, DescribedVoice) for voice in self.voices
        ):
            raise RuntimeCapabilityError("model voices must be a tuple of DescribedVoice records")
        if not isinstance(self.runtime_available, bool):
            raise RuntimeCapabilityError("runtime_available must be a bool")
        if not isinstance(self.metadata, Mapping):
            raise RuntimeCapabilityError("model metadata must be a mapping")
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @property
    def voice_ids(self) -> tuple[str, ...]:
        return tuple(voice.id for voice in self.voices)


def _metadata_string(metadata: Mapping[str, Any], key: str) -> str | None:
    value = metadata.get(key)
    return value if isinstance(value, str) and value else None


def _max_input_tokens(metadata: Mapping[str, Any]) -> int | None:
    value = metadata.get("max_input_tokens")
    runtime = metadata.get("runtime")
    if value is None and isinstance(runtime, Mapping):
        value = runtime.get("max_input_tokens")
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise RuntimeCapabilityError("catalog max_input_tokens must be a positive integer")
    return value


def _describe_model(item: Any) -> DiscoveredModel:
    try:
        model_id = item.id
        metadata = item.metadata
        if not isinstance(model_id, str) or not model_id:
            raise RuntimeCapabilityError("catalog model id must be a non-empty string")
        if not isinstance(metadata, Mapping):
            raise RuntimeCapabilityError("catalog model metadata must be a mapping")

        raw_languages = metadata.get("language_codes", metadata.get("languages", ()))
        if not isinstance(raw_languages, (tuple, list)):
            raise RuntimeCapabilityError("catalog model languages must be a sequence")
        languages = tuple(language for language in raw_languages if language != UNKNOWN_LANGUAGE)
        if any(not isinstance(language, str) or not language for language in languages):
            raise RuntimeCapabilityError("catalog model languages must contain non-empty strings")

        raw_voices = item.voices
        if not isinstance(raw_voices, (tuple, list)):
            raise RuntimeCapabilityError("catalog voice inventory must be a sequence")
        voice_metadata = metadata.get("voices", ())
        voice_records = (
            {
                record.get("name"): dict(record)
                for record in voice_metadata
                if isinstance(record, Mapping) and isinstance(record.get("name"), str)
            }
            if isinstance(voice_metadata, (tuple, list))
            else {}
        )
        voices = tuple(
            DescribedVoice(
                id=voice_id,
                languages=languages,
                metadata=voice_records.get(voice_id, {}),
            )
            for voice_id in raw_voices
        )
        aliases = tuple(item.aliases)
        sample_rate = item.sample_rate
        source_revision = _metadata_string(metadata, "source_revision")
        version_value = _metadata_string(metadata, "version") or _metadata_string(
            metadata, "model_version"
        )
        display_name = (
            _metadata_string(metadata, "display_name")
            or _metadata_string(metadata, "name")
            or model_id
        )
        return DiscoveredModel(
            id=model_id,
            ref=f"supertonic:{model_id}",
            display_name=display_name,
            version=version_value,
            sample_rate=sample_rate,
            aliases=aliases,
            languages=languages,
            voices=voices,
            default_voice=item.default_voice,
            source_revision=source_revision,
            max_input_tokens=_max_input_tokens(metadata),
            metadata=metadata,
        )
    except RuntimeCapabilityError:
        raise
    except (AttributeError, TypeError, ValueError) as exc:
        raise RuntimeCapabilityError(f"invalid OnnxVoice catalog record: {exc}") from exc


def discover_models(
    *,
    language: str | None = None,
    offline: bool = False,
    refresh: bool = False,
    cache_dir: str | Path | None = None,
    catalog_path: str | Path | None = None,
) -> tuple[DiscoveredModel, ...]:
    if language is not None and (not isinstance(language, str) or not language.strip()):
        raise InvalidLanguageError("language filter must be a non-empty string or None")
    if language == UNKNOWN_LANGUAGE:
        return ()
    items = list_pretrained_bundles(
        cache_dir=cache_dir,
        offline=offline,
        refresh=refresh,
        catalog_path=catalog_path,
        language=language,
    )
    models = tuple(_describe_model(item) for item in items)
    if language is None:
        return models
    return tuple(model for model in models if language in model.languages)


def _distribution_version(distribution: str) -> str | None:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return None


def runtime_identity(model: DiscoveredModel | None = None) -> dict[str, str | None]:
    if model is not None and not isinstance(model, DiscoveredModel):
        raise TypeError("model must be a DiscoveredModel or None")
    from . import __version__

    engine_version = _distribution_version("supertonicsynth") or __version__
    source_revision = model.source_revision if model is not None else None
    model_revision = (model.version or source_revision) if model is not None else None
    return {
        "engine_version": engine_version,
        "runtime_revision": _distribution_version("onnxvoice"),
        "catalog_revision": source_revision,
        "model_revision": model_revision,
    }
