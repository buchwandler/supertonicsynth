from types import SimpleNamespace

import pytest

import supertonicsynth._onnxvoice as onnxvoice_boundary
from supertonicsynth.discovery import (
    DescribedVoice,
    DiscoveredModel,
    discover_models,
    runtime_identity,
)
from supertonicsynth.errors import (
    BundleNotFoundError,
    CatalogUnavailableError,
    OfflineAssetError,
    OptionalDependencyError,
    RuntimeCapabilityError,
)


class CatalogError(Exception):
    pass


class OfflineError(Exception):
    pass


class UnsupportedSystemError(Exception):
    pass


class AssetNotFoundError(Exception):
    pass


def _catalog_item():
    return SimpleNamespace(
        id="supertonic-3",
        aliases=("supertonic", "st3"),
        sample_rate=44100,
        voices=("F1", "M1"),
        default_voice="F1",
        metadata={
            "display_name": "Supertonic 3",
            "version": "3.0",
            "language_codes": ("en", "de", "na"),
            "source_revision": "a" * 40,
            "runtime": {"layout": "supertonic-3-v1", "max_input_tokens": 512},
            "voices": [
                {"name": "F1", "artifact_component": "F1"},
                {"name": "M1", "artifact_component": "M1"},
            ],
        },
    )


class FakeOnnxVoice:
    item = _catalog_item()
    calls = []
    failure = None

    def __init__(self, **kwargs):
        self.constructor_kwargs = kwargs
        type(self).constructor_calls.append(kwargs)

    def list(self, system, *, language=None, refresh=False):
        type(self).calls.append((system, language, refresh))
        if type(self).failure is not None:
            raise type(self).failure
        return (self.item,)

    def install(self, *_args, **_kwargs):
        pytest.fail("metadata discovery must not install model assets")

    def open(self, *_args, **_kwargs):
        pytest.fail("metadata discovery must not open a runtime")


FakeOnnxVoice.constructor_calls = []


def use_fake_catalog(monkeypatch):
    FakeOnnxVoice.calls = []
    FakeOnnxVoice.constructor_calls = []
    FakeOnnxVoice.failure = None
    FakeOnnxVoice.item = _catalog_item()
    monkeypatch.setattr(
        onnxvoice_boundary,
        "_onnxvoice",
        lambda: SimpleNamespace(OnnxVoice=FakeOnnxVoice),
    )


def test_discover_models_normalizes_catalog_without_install_or_open(tmp_path, monkeypatch):
    use_fake_catalog(monkeypatch)
    catalog_path = tmp_path / "catalog.json"

    models = discover_models(
        language="de",
        offline=True,
        refresh=True,
        cache_dir=tmp_path / "cache",
        catalog_path=catalog_path,
    )

    assert isinstance(models, tuple)
    assert len(models) == 1
    model = models[0]
    assert isinstance(model, DiscoveredModel)
    assert model.id == "supertonic-3"
    assert model.ref == "supertonic:supertonic-3"
    assert model.display_name == "Supertonic 3"
    assert model.version == "3.0"
    assert model.sample_rate == 44100
    assert model.aliases == ("supertonic", "st3")
    assert model.languages == ("en", "de")
    assert model.default_voice == "F1"
    assert model.source_revision == "a" * 40
    assert model.max_input_tokens == 512
    assert model.voice_ids == ("F1", "M1")
    assert all(isinstance(voice, DescribedVoice) for voice in model.voices)
    assert model.voices[0].gender == "unknown"
    assert model.voices[0].languages == ("en", "de")
    assert model.voices[0].metadata["artifact_component"] == "F1"
    assert FakeOnnxVoice.constructor_calls == [
        {
            "cache_dir": tmp_path / "cache",
            "catalog_sources": {"supertonic": str(catalog_path)},
            "offline": True,
        }
    ]
    assert FakeOnnxVoice.calls == [("supertonic", "de", True)]


def test_discovery_language_filter_excludes_unknown_sentinel(tmp_path, monkeypatch):
    use_fake_catalog(monkeypatch)
    all_models = discover_models()
    assert len(all_models) == 1
    assert discover_models(language="fr") == ()
    before = len(FakeOnnxVoice.constructor_calls)
    assert discover_models(language="na") == ()
    assert len(FakeOnnxVoice.constructor_calls) == before


def test_discovery_reports_unknown_capacity_when_catalog_has_no_limit(monkeypatch):
    use_fake_catalog(monkeypatch)
    FakeOnnxVoice.item.metadata["runtime"] = {"layout": "supertonic-3-v1", "chunk_size": 512}

    model = discover_models()[0]

    assert model.max_input_tokens is None


@pytest.mark.parametrize(
    "failure,error_type",
    [
        (CatalogError("catalog unavailable"), CatalogUnavailableError),
        (OfflineError("no cached catalog"), OfflineAssetError),
        (UnsupportedSystemError("system unsupported"), RuntimeCapabilityError),
        (AssetNotFoundError("bundle missing"), BundleNotFoundError),
    ],
)
def test_discovery_maps_onnxvoice_catalog_errors(monkeypatch, failure, error_type):
    use_fake_catalog(monkeypatch)
    FakeOnnxVoice.failure = failure

    with pytest.raises(error_type):
        discover_models()


def test_discovery_maps_malformed_catalog_record(monkeypatch):
    use_fake_catalog(monkeypatch)
    FakeOnnxVoice.item = SimpleNamespace(id="broken", metadata={})

    with pytest.raises(RuntimeCapabilityError, match="invalid OnnxVoice catalog record"):
        discover_models()


def test_discovery_rejects_invalid_capacity_metadata(monkeypatch):
    use_fake_catalog(monkeypatch)
    FakeOnnxVoice.item.metadata["max_input_tokens"] = 0

    with pytest.raises(RuntimeCapabilityError, match="max_input_tokens"):
        discover_models()


def test_runtime_identity_without_and_with_model(monkeypatch):
    use_fake_catalog(monkeypatch)
    model = discover_models()[0]

    without_model = runtime_identity()
    with_model = runtime_identity(model)

    assert set(without_model) == {
        "engine_version",
        "request_api_version",
        "runtime_revision",
        "catalog_revision",
        "model_revision",
    }
    assert without_model["engine_version"]
    assert without_model["catalog_revision"] is None
    assert without_model["request_api_version"] == "1"
    assert without_model["model_revision"] is None
    assert with_model["catalog_revision"] == "a" * 40
    assert with_model["model_revision"] == "3.0"


def test_runtime_identity_rejects_foreign_model_objects():
    with pytest.raises(TypeError, match="DiscoveredModel"):
        runtime_identity(object())


def test_optional_dependency_error_remains_public(monkeypatch):
    def fail_import():
        raise OptionalDependencyError("missing OnnxVoice")

    monkeypatch.setattr(onnxvoice_boundary, "_onnxvoice", fail_import)

    with pytest.raises(OptionalDependencyError, match="missing OnnxVoice"):
        discover_models()
