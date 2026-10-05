import subprocess
import sys
from dataclasses import FrozenInstanceError

import pytest

import supertonicsynth
from supertonicsynth import (
    AtomicSynthesisResult,
    DescribedVoice,
    GenerationConfig,
    RequestApiContract,
    RequestMeasure,
    SupertonicRuntime,
    SynthesisRequest,
    request_api_contract,
)


def test_request_api_version_is_public():
    assert supertonicsynth.REQUEST_API_VERSION == 1
    assert "REQUEST_API_VERSION" in supertonicsynth.__all__
    assert isinstance(request_api_contract(), RequestApiContract)


def test_request_api_contract_is_dependency_light():
    code = """
import sys
import supertonicsynth
from supertonicsynth import (
    AtomicSynthesisResult,
    GenerationConfig,
    RequestApiContract,
    RequestMeasure,
    SynthesisRequest,
    SupertonicRuntime,
    request_api_contract,
)

contract = request_api_contract()
assert contract.request_type is SynthesisRequest
assert contract.result_type is AtomicSynthesisResult
assert contract.runtime_type is SupertonicRuntime
assert isinstance(GenerationConfig(), GenerationConfig)
assert RequestMeasure(amount=1, maximum=None).fits is None
assert isinstance(contract, RequestApiContract)
assert 'onnxvoice' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)


def test_request_api_contract_declares_types_methods_and_capabilities():
    contract = request_api_contract()

    assert contract.version == supertonicsynth.REQUEST_API_VERSION == 1
    assert contract.request_type is SynthesisRequest
    assert contract.result_type is AtomicSynthesisResult
    assert contract.runtime_type is SupertonicRuntime
    assert contract.synthesis_method == "synthesize"
    assert contract.measurement_method == "measure_request"
    assert contract.discovery_method == "discover_models"
    assert contract.runtime_identity_method == "runtime_identity"
    assert contract.caller_owns_text_boundaries is True
    assert contract.supports_request_measurement is True
    assert contract.supports_named_voices is True
    assert contract.supports_reference_voice is False
    assert contract.supports_linguistic_tokens is False
    assert contract.supports_pronunciation_overrides is False
    assert contract.supports_whole_request_phonemes is False
    assert contract.supports_speakers is False
    assert contract.supports_word_timings is False
    assert contract.supports_voice_level is True

    with pytest.raises(FrozenInstanceError):
        contract.version = 2


def test_no_model_request_surface_records_and_runtime_methods():
    request = SynthesisRequest(id="contract", text="One caller-shaped request.", language="en")
    generation = GenerationConfig()
    measure = RequestMeasure(amount=1, maximum=None)
    voice = DescribedVoice(id="F1", languages=("en",))
    model = supertonicsynth.DiscoveredModel(
        id="supertonic-3",
        ref="supertonic:supertonic-3",
        display_name="Supertonic 3",
        version=None,
        sample_rate=44100,
        aliases=("supertonic",),
        languages=("en",),
        voices=(voice,),
        default_voice="F1",
        source_revision=None,
        max_input_tokens=None,
    )

    assert request.text == "One caller-shaped request."
    assert generation.seed is None
    assert measure.fits is None
    assert model.voice_ids == ("F1",)
    assert callable(SupertonicRuntime.from_pretrained)
    assert callable(SupertonicRuntime.measure_request)
    assert callable(SupertonicRuntime.synthesize)
    assert callable(SupertonicRuntime.close)
    assert "onnxvoice" not in sys.modules
