"""SupertonicSynth, Supertonic-3 synthesis backed by OnnxVoice."""

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
from .errors import (
    AssetCacheError,
    AssetDownloadError,
    AssetError,
    BundleNotFoundError,
    CatalogUnavailableError,
    ClosedRuntimeError,
    InvalidLanguageError,
    InvalidRequestError,
    InvalidVoiceStyleError,
    ModelInferenceError,
    OfflineAssetError,
    OptionalDependencyError,
    RuntimeCapabilityError,
    SessionCreationError,
    SupertonicSynthError,
)
from .runtime import SupertonicRuntime
from .style import VoiceStyle, load_voice_style
from .types import RuntimeDiagnostics, SynthesisConfig, SynthesisResult
from .voice_level import (
    CalibrationDataError,
    VoiceCalibrationCatalog,
    VoiceCalibrationKey,
    VoiceLevelApplication,
    VoiceLevelCalibration,
    VoiceLevelConfig,
    VoiceLevelMode,
    apply_voice_level_calibration,
    default_voice_calibration,
    load_voice_calibration,
)

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
    "CalibrationDataError",
    "VoiceCalibrationCatalog",
    "VoiceCalibrationKey",
    "VoiceLevelApplication",
    "VoiceLevelCalibration",
    "VoiceLevelConfig",
    "VoiceLevelMode",
    "apply_voice_level_calibration",
    "default_voice_calibration",
    "load_voice_calibration",
    "AssetCacheError",
    "AssetDownloadError",
    "AssetError",
    "BundleNotFoundError",
    "CatalogUnavailableError",
    "ClosedRuntimeError",
    "InvalidLanguageError",
    "InvalidRequestError",
    "InvalidVoiceStyleError",
    "ModelInferenceError",
    "OfflineAssetError",
    "OptionalDependencyError",
    "RuntimeCapabilityError",
    "SessionCreationError",
    "SupertonicSynthError",
]
