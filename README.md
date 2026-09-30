# supertonicsynth

A flat-layout Python synthesis package for Supertonic-3, using OnnxVoice for model storage, integrity verification, ONNX sessions, providers, and runtime diagnostics.

## Design

`supertonicsynth` owns:

- Supertonic text normalization and Unicode indexing.
- language tokens.
- voice-style JSON loading.
- long-text chunking.
- synthesis configuration and WAV/result handling.

OnnxVoice owns:

- the `supertonic` catalog/store.
- downloads and checksums.
- the four ONNX sessions.
- provider configuration.
- the Supertonic model-ready inference loop.

## Install

```bash
pip install "supertonicsynth[cpu]"
```

The package uses dynamic VCS versioning via `setuptools_scm`; there is no static version in `pyproject.toml`.

The pretrained model catalog targets the archived Supertonic TTS weights at `supertone-oss-archive/supertonic-3`; model downloading remains an OnnxVoice responsibility.

## Usage

```python
from supertonicsynth import SupertonicRuntime

with SupertonicRuntime.from_pretrained("supertonic-3") as tts:
    result = tts.synthesize_text(
        "Welcome to SupertonicSynth.",
        voice="M1",
        language="en",
        steps=5,
        seed=1234,
    )
    result.write_wav("output.wav")
```

Local bundle:

```python
with SupertonicRuntime.from_local("/models/supertonic-3") as tts:
    result = tts.synthesize_text("Hello.", voice="F1", language="en")
```

## Runtime prerequisite

The included package targets the staged OnnxVoice work described in the sibling implementation briefs. Until an OnnxVoice release includes `system="supertonic"`, local/pretrained opening will fail with a runtime capability error.

## Repository layout

There is deliberately **no `src/` layer**:

```text
supertonicsynth/
  __init__.py
  runtime.py
  frontend.py
  ...
tests/
examples/
pyproject.toml
```

## Licensing

SupertonicSynth project code is Apache-2.0. Portions of the frontend/text behavior are derived from the archived MIT-licensed `supertone-oss-archive/supertonic-py`; the upstream MIT notice is preserved under `licenses/SUPERTONIC-PY-MIT.txt` and in `NOTICE`.

Supertonic-3 model assets are not included in this package and retain their upstream OpenRAIL-M model license.
