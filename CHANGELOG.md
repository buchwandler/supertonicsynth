# Changelog

## Unreleased

- Add opt-in static voice calibration keyed by canonical OnnxVoice semantic voice ref and synthesis language, with strict catalog loading and request-level gain controls.
- Add shared finite-audio validation, peak normalization, clipping, PCM16/WAV conversion, calibration diagnostics, and reproducibility metadata.
- Add deterministic counting benchmarks, language-specific Spokenform/Numeralform stimuli, and review-gated multi-report candidate promotion. Keep the packaged calibration catalog empty while `hr` stimulus preparation is unsupported.
- Bootstrap Supertonic-3 frontend/runtime package around the staged OnnxVoice `supertonic` adapter and catalog.
