class SupertonicSynthError(Exception):
    """Base error for SupertonicSynth."""


class OptionalDependencyError(SupertonicSynthError):
    pass


class AssetError(SupertonicSynthError):
    pass


class AssetDownloadError(AssetError):
    pass


class AssetCacheError(AssetError):
    pass


class CatalogUnavailableError(AssetError):
    pass


class BundleNotFoundError(AssetError):
    pass


class OfflineAssetError(AssetError):
    pass


class SessionCreationError(SupertonicSynthError):
    pass


class ModelInferenceError(SupertonicSynthError):
    pass


class RuntimeCapabilityError(SupertonicSynthError):
    pass


class InvalidRequestError(SupertonicSynthError):
    pass


class InvalidLanguageError(InvalidRequestError):
    pass


class InvalidVoiceStyleError(InvalidRequestError):
    pass


class ClosedRuntimeError(SupertonicSynthError):
    pass
