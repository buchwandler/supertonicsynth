from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import (
    AssetCacheError,
    AssetDownloadError,
    AssetError,
    BundleNotFoundError,
    CatalogUnavailableError,
    ModelInferenceError,
    OfflineAssetError,
    OptionalDependencyError,
    RuntimeCapabilityError,
    SessionCreationError,
)

_COMPONENTS = ("duration_predictor", "text_encoder", "vector_estimator", "vocoder")


@dataclass(frozen=True, slots=True)
class ResolvedSupertonicBundle:
    ref: str | None
    bundle_id: str | None
    config_path: Path
    unicode_indexer_path: Path
    model_paths: Mapping[str, Path]
    style_paths: Mapping[str, Path]
    sample_rate: int | None
    source_revision: str | None
    metadata: Mapping[str, Any]
    installation: Any | None = None


def _onnxvoice() -> Any:
    try:
        import onnxvoice
    except ModuleNotFoundError as exc:
        raise OptionalDependencyError(
            "OnnxVoice with Supertonic support is required. Install supertonicsynth[cpu] or supertonicsynth[gpu]."
        ) from exc
    return onnxvoice


def normalize_supertonic_ref(ref: str) -> str:
    if not isinstance(ref, str) or not ref.strip():
        raise BundleNotFoundError("Supertonic bundle reference must not be empty")
    value = ref.strip()
    if ":" not in value:
        return f"supertonic:{value}"
    system, item_id = value.split(":", 1)
    if system.casefold() != "supertonic" or not item_id:
        raise BundleNotFoundError(f"Not a Supertonic bundle reference: {ref!r}")
    return f"supertonic:{item_id}"


def normalize_provider_request(
    providers: Sequence[Any] | str | None, provider_options: Mapping[str, Any] | None
) -> tuple[str | Sequence[str] | None, Any | None]:
    if providers is None or isinstance(providers, str):
        return providers, provider_options
    names: list[str] = []
    options: list[dict[str, Any]] = []
    explicit = False
    for provider in providers:
        if hasattr(provider, "name"):
            names.append(str(provider.name))
            value = dict(getattr(provider, "options", {}) or {})
            options.append(value)
            explicit = explicit or bool(value)
        elif isinstance(provider, tuple):
            name, raw = provider
            names.append(str(name))
            options.append(dict(raw or {}))
            explicit = True
        else:
            names.append(str(provider))
            options.append(dict(provider_options or {}))
    if not names:
        raise ValueError("providers must not be empty")
    return names, options if explicit else provider_options


def _map_error(exc: Exception, operation: str) -> Exception:
    name = type(exc).__name__
    message = str(exc) or name
    if name in {"AssetNotFoundError", "NotInstalledError"}:
        return BundleNotFoundError(message)
    if name == "OfflineError":
        return OfflineAssetError(message)
    if name in {"IntegrityError", "ManifestError", "UnsafePathError", "LockError"}:
        return AssetCacheError(message)
    if name == "CatalogError":
        return CatalogUnavailableError(message)
    if name in {"RuntimeContractError", "CapabilityError", "UnsupportedSystemError"}:
        return RuntimeCapabilityError(message)
    if name == "OptionalDependencyError":
        return OptionalDependencyError(message)
    if operation == "infer":
        return ModelInferenceError(f"Supertonic inference failed: {message}")
    if operation == "open":
        return SessionCreationError(f"Could not open Supertonic runtime: {message}")
    return AssetDownloadError(message) if operation == "install" else AssetError(message)


def _call(operation: str, fn: Callable[[], Any]) -> Any:
    try:
        return fn()
    except (FileNotFoundError, ValueError, TypeError):
        raise
    except Exception as exc:
        mapped = _map_error(exc, operation)
        raise mapped from exc


def _local_files(root: Path) -> dict[str, Path]:
    files = {
        "config": root / "onnx" / "tts.json",
        "unicode_indexer": root / "onnx" / "unicode_indexer.json",
        **{component: root / "onnx" / f"{component}.onnx" for component in _COMPONENTS},
    }
    styles = root / "voice_styles"
    if styles.is_dir():
        for path in sorted(styles.glob("*.json")):
            files[f"voice_style:{path.stem}"] = path
    missing = [
        str(path)
        for key, path in files.items()
        if not key.startswith("voice_style:") and not path.is_file()
    ]
    if missing:
        raise FileNotFoundError("Missing Supertonic bundle files: " + ", ".join(missing))
    return files


def open_local_bundle(
    root: str | Path,
    *,
    providers: Sequence[Any] | str | None = None,
    provider_options: Mapping[str, Any] | None = None,
    session_options: Any | None = None,
) -> tuple[Any, ResolvedSupertonicBundle]:
    root_path = Path(root)
    requested, options = normalize_provider_request(providers, provider_options)
    module = _onnxvoice()
    runtime = _call(
        "open",
        lambda: module.open_local(
            system="supertonic",
            files=_local_files(root_path),
            metadata={"managed": False, "bundle_name": root_path.name},
            runtime={"layout": "supertonic-3-v1"},
            sample_rate=44_100,
            providers=requested,
            provider_options=options,
            session_options=session_options,
        ),
    )
    installation = getattr(runtime, "installation", None)
    resolved = (
        installation_to_bundle_info(installation, ref=None)
        if installation is not None
        else ResolvedSupertonicBundle(
            ref=None,
            bundle_id=root_path.name,
            config_path=root_path / "onnx/tts.json",
            unicode_indexer_path=root_path / "onnx/unicode_indexer.json",
            model_paths={c: root_path / "onnx" / f"{c}.onnx" for c in _COMPONENTS},
            style_paths={p.stem: p for p in sorted((root_path / "voice_styles").glob("*.json"))},
            sample_rate=44_100,
            source_revision=None,
            metadata={"managed": False},
            installation=None,
        )
    )
    return runtime, resolved


def install_pretrained_bundle(
    ref: str,
    *,
    cache_dir: str | Path | None = None,
    offline: bool = False,
    refresh_catalog: bool = False,
    force_download: bool = False,
    progress: Any | None = None,
) -> ResolvedSupertonicBundle:
    normalized = normalize_supertonic_ref(ref)
    module = _onnxvoice()
    manager = module.OnnxVoice(cache_dir=cache_dir, offline=offline)
    installation = _call(
        "install",
        lambda: manager.install(
            normalized, refresh=refresh_catalog, force=force_download, progress=progress
        ),
    )
    return installation_to_bundle_info(installation, ref=normalized)


def open_installed_bundle(
    resolved: ResolvedSupertonicBundle,
    *,
    cache_dir: str | Path | None = None,
    providers: Sequence[Any] | str | None = None,
    provider_options: Mapping[str, Any] | None = None,
    session_options: Any | None = None,
) -> Any:
    requested, options = normalize_provider_request(providers, provider_options)
    module = _onnxvoice()
    manager = module.OnnxVoice(cache_dir=cache_dir)
    return _call(
        "open",
        lambda: manager.open(
            resolved.installation,
            providers=requested,
            provider_options=options,
            session_options=session_options,
        ),
    )


def installation_to_bundle_info(installation: Any, *, ref: str | None) -> ResolvedSupertonicBundle:
    if installation is None:
        raise RuntimeCapabilityError("Supertonic installation is unavailable")
    try:
        config = Path(installation.artifact("config").path)
        indexer = Path(installation.artifact("unicode_indexer").path)
        models = {
            component: Path(installation.artifact("model", component=component).path)
            for component in _COMPONENTS
        }
    except (KeyError, AttributeError) as exc:
        raise RuntimeCapabilityError(
            "Supertonic installation is missing required model/config artifacts"
        ) from exc
    styles: dict[str, Path] = {}
    for artifact in getattr(installation, "artifacts", ()):
        if getattr(artifact, "role", None) == "voice_style" and getattr(
            artifact, "component", None
        ):
            styles[str(artifact.component)] = Path(artifact.path)
    raw = dict(getattr(installation, "metadata", {}) or {})
    return ResolvedSupertonicBundle(
        ref=ref or getattr(installation, "ref", None),
        bundle_id=getattr(installation, "id", None),
        config_path=config,
        unicode_indexer_path=indexer,
        model_paths=models,
        style_paths=styles,
        sample_rate=getattr(installation, "sample_rate", None),
        source_revision=str(raw.get("source_revision")) if raw.get("source_revision") else None,
        metadata=raw,
        installation=installation,
    )


def available_providers() -> tuple[str, ...]:
    return tuple(_call("open", lambda: _onnxvoice().available_providers()))
