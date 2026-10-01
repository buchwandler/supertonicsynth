# supertonicsynth

A flat-layout Python synthesis package for Supertonic-3. OnnxVoice owns catalog resolution, managed asset installation and integrity, ONNX Runtime sessions, providers, and inference. SupertonicSynth owns text synthesis, language selection, voice styles, chunking, static voice-level calibration, output gain, and synthesis metadata.

## Install

```bash
pip install "supertonicsynth[cpu]"
```

The project uses dynamic VCS versioning through `setuptools_scm`. Model weights are not included in the wheel.

The current development checkout uses a locally adapted OnnxVoice semantic voice-ref API. The dependency lower bound will be updated only after the first compatible OnnxVoice release is available.

## Python usage

```python
from supertonicsynth import (
    SupertonicRuntime,
    SynthesisConfig,
    VoiceLevelConfig,
)

config = SynthesisConfig(
    voice_level=VoiceLevelConfig(mode="calibrated"),
)

with SupertonicRuntime.from_pretrained("supertonic-3") as tts:
    result = tts.synthesize_text(
        "A prepared German sentence.",
        voice="F1",
        language="de",
        config=config,
    )
    result.write_wav("output.wav")

print(result.metadata["voice_ref"])
print(result.metadata["voice_level"]["calibration_key"])
```

`result.pcm16()` and `result.write_wav(path)` use the same finite-audio validation and clipping policy.

## CLI

```bash
supertonicsynth synthesize \
  "A prepared German sentence." \
  --model supertonic-3 \
  --voice F1 \
  --language de \
  --voice-level calibrated \
  --no-normalize-audio \
  --output-gain 1.0 \
  -o output.wav
```

Use `--voice-gain-db FLOAT` for an explicit static dB override. That override takes precedence over the catalog and works for local or custom styles. `--output-gain FLOAT` is a separate request-level linear gain. Final output is clipped to `[-1, 1]` for safety.

## Voice identity and language

OnnxVoice semantic voice refs are the canonical voice identity. A managed `F1` style from `supertonic-3` has the voice ref `supertonic:supertonic-3/F1`. Language remains a separate synthesis condition. Calibration therefore uses a pair such as `(supertonic:supertonic-3/F1, de)`, serialized as `supertonic:supertonic-3/F1@de`. The `@de` suffix belongs to the calibration key, not the voice ref.

Automatic catalog calibration applies only to managed OnnxVoice installations and catalog-backed string styles. Local bundles and caller-created `VoiceStyle` values do not receive calibration by matching their directory or style name. They remain unchanged unless an explicit `voice_level.gain_db` override is provided.

## Loudness calibration and output gain

Calibration is an offline, reviewed static correction. The benchmark measures prepared speech and the promotion tool stores measured gain against the canonical voice ref and synthesis language. Normal synthesis performs no LUFS measurement and does not contact a network service for calibration. Missing identity or missing calibration leaves the audio unchanged.

`VoiceLevelConfig(mode="off")` is the default. `mode="calibrated"` opts into the packaged catalog. `normalize_audio` is a separate deterministic peak-normalization control. `output_gain` is a separate linear user-requested gain. Neither feature performs final program mastering, which remains an external responsibility.

The packaged catalog is currently empty. The full measurement matrix is blocked because the local Spokenform and Numeralform checkouts do not support Croatian (`hr`); the benchmark fails for that language instead of borrowing English. See the [benchmark and promotion guide](benchmarks/README.md). The `na` unknown-language sentinel is not a spoken language and is intentionally excluded from calibration coverage.

## Further documentation

- [Architecture](docs/architecture.md)
- [Troubleshooting](docs/troubleshooting.md)
- [Benchmark and promotion](benchmarks/README.md)

## Local bundle

```python
with SupertonicRuntime.from_local("/models/supertonic-3") as tts:
    result = tts.synthesize_text("Hello.", voice="F1", language="en")
```

Local bundles are unmanaged and do not claim a canonical OnnxVoice voice ref for automatic calibration.

## Repository layout

The package deliberately has no `src/` layer:

```text
supertonicsynth/
tests/
examples/
benchmarks/
pyproject.toml
```

## Licensing

SupertonicSynth project code is Apache-2.0. Portions of the frontend and text behavior are derived from the archived MIT-licensed `supertone-oss-archive/supertonic-py`; the upstream MIT notice is preserved under `licenses/SUPERTONIC-PY-MIT.txt` and in `NOTICE`.

Supertonic-3 model assets are not included in this package and retain their upstream OpenRAIL-M model license.
