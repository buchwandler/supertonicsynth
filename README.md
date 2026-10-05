# supertonicsynth

A flat-layout Python synthesis package for Supertonic-3. OnnxVoice owns catalog resolution, managed asset installation and integrity, ONNX Runtime sessions, providers, and inference. SupertonicSynth owns text synthesis, language selection, voice styles, chunking, static voice-level calibration, output gain, and synthesis metadata.

## Install

```bash
pip install "supertonicsynth[cpu]"
```

The project uses dynamic VCS versioning through `setuptools_scm`. Model weights are not included in the wheel.

The package requires OnnxVoice >=0.2.0,<0.3. The 0.2.x line provides the Supertonic catalog, semantic voice refs, managed installation, local-open, and runtime APIs used by this package.

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

### Atomic synthesis

`SupertonicRuntime.synthesize` is the strict one-request API. It encodes one request and performs at most one inference. It does not split text, join audio chunks, insert inter-chunk silence, peak-normalize, or apply output gain. `synthesize_text` and package-level `synthesize` remain convenience APIs that preserve the existing multi-chunk composition behavior.

```python
from supertonicsynth import GenerationConfig, SupertonicRuntime, SynthesisRequest

request = SynthesisRequest(id="job-42", text="A prepared German sentence.", language="de")
with SupertonicRuntime.from_pretrained("supertonic-3") as tts:
    measurement = tts.measure_request(request)
    result = tts.synthesize(
        request,
        voice="F1",
        config=GenerationConfig(steps=5, speed=1.05, seed=42),
    )
    result.write_wav("atomic.wav")

print(measurement.amount, measurement.maximum, measurement.fits)
```

`RequestMeasure` reports encoded token count and only reports a maximum when model metadata declares `max_input_tokens`. If no maximum is declared, `maximum` and `fits` are `None`; callers must not infer a capacity limit.

`REQUEST_API_VERSION` is `1`. `request_api_contract()` returns a frozen, dependency-light `RequestApiContract` describing the request/result/runtime types, method names, and capabilities; inspecting it does not import OnnxVoice or open model assets. Text boundaries belong to the caller. Named voices, request measurement, and voice-level controls are supported; reference voices, linguistic tokens, pronunciation overrides, whole-request phonemes, speakers, and word timings are not.

`measure_request()` is the public preflight API and shares token accounting with `synthesize()`. If the catalog declares a maximum, an oversized request raises `SynthesisInputTooLongError` before inference; with no declared maximum, `fits` remains `None` rather than guessing.

### Metadata-only discovery

```python
from supertonicsynth import discover_models, runtime_identity

for model in discover_models(language="de"):
    print(model.id, model.voice_ids, model.max_input_tokens)
    print(runtime_identity(model))
```

Discovery reads typed OnnxVoice catalog metadata. It does not install model assets or open runtime sessions. Pass `offline=True` to use cached catalog metadata only. Optional catalog fields, including `max_input_tokens`, remain unknown when the catalog does not declare them.

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

### List voices

```bash
supertonicsynth voices --model supertonic-3
```

`voices` lists catalog voice metadata without installing model assets or opening runtime sessions. `--offline` uses cached catalog metadata only; omit it to allow catalog metadata retrieval. Use `--refresh-catalog`, `--cache-dir`, and `--catalog-path` to control catalog lookup.

## Voice identity and language

OnnxVoice semantic voice refs are the canonical voice identity. A managed `F1` style from `supertonic-3` has the voice ref `supertonic:supertonic-3/F1`. Language remains a separate synthesis condition. Calibration therefore uses a pair such as `(supertonic:supertonic-3/F1, de)`, serialized as `supertonic:supertonic-3/F1@de`. The `@de` suffix belongs to the calibration key, not the voice ref.

Automatic catalog calibration applies only to managed OnnxVoice installations and catalog-backed string styles. Local bundles and caller-created `VoiceStyle` values do not receive calibration by matching their directory or style name. They remain unchanged unless an explicit `voice_level.gain_db` override is provided.

## Loudness calibration and output gain

Calibration is an offline, reviewed static correction. The benchmark measures prepared speech and the promotion tool stores measured gain against the canonical voice ref and synthesis language. Normal synthesis performs no LUFS measurement and does not contact a network service for calibration. Missing identity or missing calibration leaves the audio unchanged.

`VoiceLevelConfig(mode="off")` is the default. `mode="calibrated"` opts into the packaged catalog. `normalize_audio` is a separate deterministic peak-normalization control. `output_gain` is a separate linear user-requested gain. Neither feature performs final program mastering, which remains an external responsibility.

The package ships 252 reviewed, statistically eligible voice/language calibrations from the Supertonic-3 counting benchmark. Coverage is partial, not the full 310-key matrix: 48 completed but high-variability identities and all 10 Croatian (`hr`) identities are intentionally absent because a Croatian counting stimulus could not be prepared. Missing entries remain unchanged at runtime (0 dB). The `na` unknown-language sentinel is excluded because it is not a spoken calibration language. See the [benchmark and promotion guide](https://github.com/buchwandler/supertonicsynth/blob/main/benchmarks/README.md).

## Further documentation

- [Architecture](https://github.com/buchwandler/supertonicsynth/blob/main/docs/architecture.md)
- [Troubleshooting](https://github.com/buchwandler/supertonicsynth/blob/main/docs/troubleshooting.md)
- [Benchmark and promotion](https://github.com/buchwandler/supertonicsynth/blob/main/benchmarks/README.md)

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
