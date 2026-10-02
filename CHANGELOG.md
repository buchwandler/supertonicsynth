# Changelog

## Unreleased

- Add opt-in static voice calibration keyed by canonical OnnxVoice semantic voice ref and synthesis language, with strict catalog loading and request-level gain controls.
- Add shared finite-audio validation, peak normalization, clipping, PCM16/WAV conversion, calibration diagnostics, and reproducibility metadata.
- Add deterministic counting benchmarks, language-specific Spokenform/Numeralform stimuli, and review-gated multi-report candidate promotion. Populate the partial packaged Supertonic-3 BS.1770 catalog from the counting-1-to-10 benchmark with 252 eligible voice/language calibrations; 48 high-variability identities, all 10 Croatian (`hr`) identities, and the `na` unknown-language sentinel remain uncalibrated. Missing entries remain unchanged at runtime (0 dB).
- Bootstrap Supertonic-3 frontend/runtime package around the staged OnnxVoice `supertonic` adapter and catalog.
