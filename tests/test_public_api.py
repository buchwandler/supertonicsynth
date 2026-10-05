import subprocess
import sys

import supertonicsynth

REQUIRED_PUBLIC_API = (
    "SynthesisRequest",
    "GenerationConfig",
    "REQUEST_API_VERSION",
    "RequestApiContract",
    "request_api_contract",
    "AtomicSynthesisResult",
    "RequestMeasure",
    "SynthesisInputTooLongError",
    "EmptyTextError",
    "InvalidGenerationConfigError",
    "DiscoveredModel",
    "DescribedVoice",
    "discover_models",
    "runtime_identity",
)


def test_required_api_is_exported_from_package_root():
    assert set(REQUIRED_PUBLIC_API) <= set(supertonicsynth.__all__)
    assert all(hasattr(supertonicsynth, name) for name in REQUIRED_PUBLIC_API)


def test_clean_public_import_does_not_import_onnxvoice_or_open_model_runtime():
    code = """
import sys
from supertonicsynth import (
    AtomicSynthesisResult,
    DescribedVoice,
    DiscoveredModel,
    GenerationConfig,
    REQUEST_API_VERSION,
    RequestApiContract,
    request_api_contract,
    RequestMeasure,
    SynthesisInputTooLongError,
    SynthesisRequest,
    SupertonicRuntime,
    discover_models,
    runtime_identity,
)
assert 'onnxvoice' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)
