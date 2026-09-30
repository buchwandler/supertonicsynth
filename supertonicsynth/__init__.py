"""SupertonicSynth — Supertonic-3 synthesis backed by OnnxVoice."""

try:
    from ._version import __version__, __version_tuple__
except ImportError:
    __version__ = "0.0.0+unknown"
    __version_tuple__ = (0, 0, 0)

from .asset_manager import BundleAssetManager
from .config import (
    AVAILABLE_LANGUAGES,
    DEFAULT_LANGUAGE,
    DEFAULT_MODEL,
    DEFAULT_VOICE,
    SUPPORTED_LANGUAGES,
    UNKNOWN_LANGUAGE,
)
from .convenience import synthesize
from .errors import *
from .runtime import SupertonicRuntime
from .style import VoiceStyle, load_voice_style
from .types import RuntimeDiagnostics, SynthesisConfig, SynthesisResult

__all__ = [
    "AVAILABLE_LANGUAGES",
    "BundleAssetManager",
    "DEFAULT_LANGUAGE",
    "DEFAULT_MODEL",
    "DEFAULT_VOICE",
    "RuntimeDiagnostics",
    "SUPPORTED_LANGUAGES",
    "SupertonicRuntime",
    "SynthesisConfig",
    "SynthesisResult",
    "UNKNOWN_LANGUAGE",
    "VoiceStyle",
    "__version__",
    "__version_tuple__",
    "load_voice_style",
    "synthesize",
]
