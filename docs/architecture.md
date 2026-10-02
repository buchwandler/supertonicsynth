# Architecture

## Ownership

SupertonicSynth owns text validation and normalization, language selection, voice-style loading, chunking, synthesis configuration, static voice-level calibration, output gain, result metadata, PCM/WAV conversion, and offline calibration tooling.

OnnxVoice owns catalog retrieval, canonical bundle and semantic voice identity, managed installation, checksums and storage, ONNX Runtime sessions, providers, inference execution, and runtime diagnostics. Calibration remains engine-local.

## Identity model

A voice has one canonical OnnxVoice semantic ref, for example `supertonic:supertonic-3/F1`. Its backing asset ref is `supertonic:supertonic-3`. Synthesis language is an independent dimension, so calibration keys pair `(voice_ref, language)` and serialize as `supertonic:supertonic-3/F1@de`. The language suffix is not part of the voice ref.

Managed installation identity comes from the canonical installed bundle ID, never from the caller's alias. A style string is eligible only when it names a style in the managed bundle. A local bundle or a caller-created `VoiceStyle` has no automatic calibration identity, even if its path or label resembles a packaged style.

## Audio pipeline

For each request, the runtime:

1. runs model inference for each text chunk and validates each returned mono finite waveform;
2. concatenates speech and requested inter-chunk silence;
3. validates the completed float32 audio and optionally peak-normalizes it;
4. constructs the managed semantic voice ref and language calibration key when trusted identity exists;
5. selects one explicit gain, catalog gain, or unchanged result for the complete utterance;
6. applies request `output_gain` and final safety clipping;
7. returns the waveform and synthesis metadata.

Normal synthesis never measures LUFS. Loudness measurement exists only in offline benchmark tooling. Static calibration uses `audiosig.apply_gain_db(..., clip=False)` so final clipping remains at the output boundary.

## Gain controls

`VoiceLevelConfig` selects `off` or `calibrated` mode and may contain an explicit `gain_db`. An explicit dB override has highest precedence and does not require a managed voice identity. Otherwise, calibrated mode looks up the `(voice_ref, language)` pair. Off mode, missing identity, and missing calibration use 0 dB.

`normalize_audio` is deterministic peak normalization before calibration. `output_gain` is a separate finite non-negative linear gain after static calibration. These engine controls are not document or program mastering.

## Calibration catalog

The packaged catalog uses schema 1 and method `bs1770`. The loader rejects duplicate JSON fields, unknown fields, unsupported schema or method, malformed Supertonic semantic refs, invalid normalized language keys, duplicate normalized identities, booleans in numeric fields, non-finite values, and non-positive sample counts. It computes a deterministic SHA-256 revision over canonical JSON. Package resources are loaded with `importlib.resources`, not filesystem-layout assumptions.

Only measured and reviewed values belong in production data. The benchmark renders counts 1 through 10 with `numeralform`, then prepares the language-specific words with `spokenform`. Reports retain exact stimuli, tool versions, voice refs, synthesis hashes, fixed repeat seeds, loudness measurements, and matrix coverage. Promotion merges compatible reports, rejects duplicate identity/repeat measurements, requires a source-hash-bound review manifest, and writes a separate candidate plus review record. It never writes production data. The packaged catalog contains 252 reviewed, statistically eligible entries from the Supertonic-3 counting benchmark. Coverage is partial: of the 310 expected voice/language identities, 48 high-variability measurements and all 10 incomplete Croatian (`hr`) identities are intentionally absent. Croatian stimulus preparation is unsupported in the local numeralform and Spokenform checkouts. Missing entries remain unchanged at runtime (0 dB). The `na` unknown-language sentinel is excluded rather than mapped to English.

## Reproducibility metadata

Results include canonical `voice_ref` and `backing_ref`, the selected calibration key, source, gain, catalog revision, and reason. `synthesis_identity` records the engine version, canonical identity and source revision, language, effective synthesis controls, and calibration selection. `synthesis_hash` is SHA-256 over canonical JSON identity plus the exact source text. It is reproducibility metadata, not an authenticity signature.

## Dependency boundary

SupertonicSynth requires OnnxVoice >=0.2.0,<0.3. The 0.2.x line provides the Supertonic catalog, semantic voice-ref, managed-installation, local-open, and runtime APIs used by this package.
