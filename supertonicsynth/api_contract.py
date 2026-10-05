"""Versioned public contract for the atomic request API."""

from __future__ import annotations

from dataclasses import dataclass

from .runtime import SupertonicRuntime
from .types import AtomicSynthesisResult, SynthesisRequest

REQUEST_API_VERSION = 1


@dataclass(frozen=True, slots=True)
class RequestApiContract:
    """Inspectable request API types, methods, and supported capabilities."""

    version: int
    request_type: type[SynthesisRequest]
    result_type: type[AtomicSynthesisResult]
    runtime_type: type[SupertonicRuntime]
    synthesis_method: str
    measurement_method: str
    discovery_method: str
    runtime_identity_method: str
    caller_owns_text_boundaries: bool
    supports_request_measurement: bool
    supports_named_voices: bool
    supports_reference_voice: bool
    supports_linguistic_tokens: bool
    supports_pronunciation_overrides: bool
    supports_whole_request_phonemes: bool
    supports_speakers: bool
    supports_word_timings: bool
    supports_voice_level: bool


def request_api_contract() -> RequestApiContract:
    """Return the immutable request API contract without opening model assets."""

    return RequestApiContract(
        version=REQUEST_API_VERSION,
        request_type=SynthesisRequest,
        result_type=AtomicSynthesisResult,
        runtime_type=SupertonicRuntime,
        synthesis_method="synthesize",
        measurement_method="measure_request",
        discovery_method="discover_models",
        runtime_identity_method="runtime_identity",
        caller_owns_text_boundaries=True,
        supports_request_measurement=True,
        supports_named_voices=True,
        supports_reference_voice=False,
        supports_linguistic_tokens=False,
        supports_pronunciation_overrides=False,
        supports_whole_request_phonemes=False,
        supports_speakers=False,
        supports_word_timings=False,
        supports_voice_level=True,
    )
