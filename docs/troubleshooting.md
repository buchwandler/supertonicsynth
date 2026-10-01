# Troubleshooting

## Calibration is off

`VoiceLevelConfig` defaults to `mode="off"`, preserving opt-in catalog correction. Select `VoiceLevelConfig(mode="calibrated")` or pass `--voice-level calibrated` to request catalog lookup.

## `missing_identity`

Automatic calibration requires a managed OnnxVoice installation and a string style name present in that installation's catalog-backed style paths. A local directory, a locally opened bundle, or a caller-created `VoiceStyle` does not get a packaged gain by matching its name. Use an explicit `gain_db`/`--voice-gain-db` override when a deliberate request-specific correction is desired.

## `missing_calibration`

The voice has a stable managed identity, but no catalog entry matches the pair `(voice_ref, language)`. Check the metadata fields `voice_ref`, `language`, and `calibration_key`. The key suffix is a separate language dimension, for example `supertonic:supertonic-3/F1@de`; it is not a different voice ref. Missing entries leave the waveform unchanged.

`na` is the unknown/default language sentinel, not a calibration language. It is not mapped to English.

## Invalid calibration data

A malformed catalog raises `CalibrationDataError`. The loader is intentionally strict. Check for duplicate JSON object keys, unknown fields, invalid or duplicate normalized keys, unsupported schema or method, non-finite numbers, boolean numeric values, and non-positive `samples`. Fix the source catalog instead of bypassing validation.

## Unexpected level changes

The processing order is peak normalization, one static calibration gain on the complete utterance, request `output_gain`, then final safety clipping. `normalize_audio` defaults to `True`; set it to `False` when preserving the model's pre-calibration peak is required. Static gain may temporarily exceed the normalized range because it uses `clip=False`; `finish_audio` performs final clipping.

`output_gain` is a finite, non-negative linear multiplier. It is not the calibration dB value and is not final mastering. WAV/PCM16 conversion clips through the shared audio helpers.

## Incomplete language benchmark

The calibration benchmark prepares numbers 1 through 10 with `numeralform`, then runs the language-specific text through `spokenform`. In the current sibling checkouts, Croatian (`hr`) is missing from both locale registries. Its stimulus preparation fails explicitly, so the full voice/language matrix is incomplete and the packaged catalog remains empty. Do not map `hr` to another language or promote a partial report as full coverage. `na` is separately excluded as the unknown-language sentinel.

## OnnxVoice dependency version

The development checkout currently relies on a locally adapted OnnxVoice semantic voice-ref API. Do not guess a released dependency bound. Raise the project lower bound only after an OnnxVoice release containing that API is available and verified.
