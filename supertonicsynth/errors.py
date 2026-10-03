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


class EmptyTextError(InvalidRequestError):
    pass


class InvalidGenerationConfigError(InvalidRequestError, ValueError):
    pass


class SynthesisInputTooLongError(InvalidRequestError):
    def __init__(
        self,
        *,
        text_length: int,
        token_count: int,
        max_tokens: int | None,
        model_id: str | None,
    ) -> None:
        self.text_length = text_length
        self.token_count = token_count
        self.max_tokens = max_tokens
        self.model_id = model_id
        maximum = "unknown" if max_tokens is None else str(max_tokens)
        model = "unknown model" if model_id is None else model_id
        super().__init__(
            f"encoded request has {token_count} tokens, exceeding maximum {maximum} for {model}"
        )


class InvalidLanguageError(InvalidRequestError):
    pass


class InvalidVoiceStyleError(InvalidRequestError):
    pass


class ClosedRuntimeError(SupertonicSynthError):
    pass
