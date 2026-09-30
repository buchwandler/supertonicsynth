from __future__ import annotations
import os
from pathlib import Path
from typing import Any
from ._onnxvoice import normalize_supertonic_ref
from .errors import OptionalDependencyError


def _offline_value(value: bool | None) -> bool:
    if value is not None:
        return value
    return os.environ.get("SUPERTONICSYNTH_OFFLINE", "").casefold() in {"1", "true", "yes", "on"}


class BundleAssetManager:
    """Thin compatibility facade over OnnxVoice's Supertonic catalog/store."""

    def __init__(
        self,
        cache_dir: str | Path | None = None,
        *,
        catalog_path: str | Path | None = None,
        offline: bool | None = None,
        progress: Any | None = None,
    ) -> None:
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.catalog_path = Path(catalog_path) if catalog_path is not None else None
        self.offline = _offline_value(offline)
        self.progress = progress

    def _manager(self) -> Any:
        try:
            import onnxvoice
        except ModuleNotFoundError as exc:
            raise OptionalDependencyError(
                "OnnxVoice is required for Supertonic bundle assets"
            ) from exc
        sources = {"supertonic": str(self.catalog_path)} if self.catalog_path is not None else None
        return onnxvoice.OnnxVoice(
            cache_dir=self.cache_dir, catalog_sources=sources, offline=self.offline
        )

    def list_bundles(self, *, refresh: bool = False) -> tuple[Any, ...]:
        return tuple(self._manager().list("supertonic", refresh=refresh, progress=self.progress))

    def resolve_bundle(
        self,
        bundle: str = "supertonic-3",
        *,
        download: bool = True,
        refresh_catalog: bool = False,
        force_download: bool = False,
    ) -> Any:
        manager = self._manager()
        ref = normalize_supertonic_ref(bundle)
        return (
            manager.install(
                ref, refresh=refresh_catalog, force=force_download, progress=self.progress
            )
            if download
            else manager.resolve(ref)
        )
