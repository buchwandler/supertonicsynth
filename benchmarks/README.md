# Supertonic voice calibration benchmark

The benchmark measures fully prepared counting speech through a managed Supertonic runtime. It renders the sequence 1 through 10 with `numeralform`, then prepares that language-specific text with `spokenform`. Both dependencies must be importable in the workspace. In this sibling checkout, install them with `python -m pip install -e ../numeralform -e ../spokenform`. The report records the source sequence, rendered words, exact prepared text, tool versions, semantic voice refs, synthesis hashes, repeat seeds, and BS.1770 measurements.

The matrix is derived from the installed managed bundle's catalog-backed styles and `SUPPORTED_LANGUAGES`. `UNKNOWN_LANGUAGE` (`na`) is not included. No alias or caller-created style is used for catalog identities. Synthesis runs with calibration off, normalization on, output gain 1.0, five steps, and three fixed repeat seeds. The packaged catalog is never modified by either tool.

Generate and inspect the complete stimulus set without opening the model runtime:

```bash
python benchmarks/voice_level_benchmark.py --prepare-only \
  --stimuli-output benchmarks/output/voice_level_calibration/counting_stimuli.json
```

The stimulus manifest lists every supported spoken language, exact prepared text, tool versions, and per-language preparation failures. It returns nonzero while any requested language is unavailable.

Run the complete matrix offline when model assets are installed:

```bash
python benchmarks/voice_level_benchmark.py --offline
```

Run a smaller diagnostic report:

```bash
python benchmarks/voice_level_benchmark.py \
  --offline --voice F1 --language en \
  --output benchmarks/output/voice_level_calibration/f1-en.json
```

The exit status is nonzero for any preparation, synthesis, or measurement failure. The report is still written and marks missing or variable entries. To collect disjoint language reports, run one `--language` selection per report, then supply every report to the promotion tool.

## Reviewed candidate promotion

Create a review template after collecting reports:

```bash
python benchmarks/promote_voice_calibration.py \
  benchmarks/output/voice_level_calibration/report-en.json \
  benchmarks/output/voice_level_calibration/report-de.json \
  --review-template \
  --output benchmarks/output/voice_level_calibration/review.json
```

Inspect the measurements, complete the reviewer and timestamp fields, and explicitly approve only individually reviewed, complete, low-variability entries with a rationale. Then build a separate candidate:

```bash
python benchmarks/promote_voice_calibration.py \
  benchmarks/output/voice_level_calibration/report-en.json \
  benchmarks/output/voice_level_calibration/report-de.json \
  --review benchmarks/output/voice_level_calibration/review.json \
  --output benchmarks/output/voice_level_calibration/candidate.json
```

The review manifest is tied to the exact input report bytes. Incomplete matrix coverage requires `--allow-partial`; incomplete and high-variability identities are never promoted. The candidate is written separately with a companion review record. The tool refuses to write the packaged production catalog. Review the candidate and review record before any deliberate production catalog edit.

## Language preparation limits

Generation fails per language rather than borrowing English or leaving numeric digits behind. In the current local sibling checkouts, 30 supported spoken languages prepare successfully, but `hr` is not registered by `numeralform` or `spokenform`. The benchmark records `hr` as a preparation failure and the full matrix therefore remains incomplete. Do not mark calibration coverage complete or package a full-matrix catalog until Croatian stimulus preparation has a verified locale-specific path. The `na` sentinel remains excluded, not mapped to English.
