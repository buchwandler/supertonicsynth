---
schema_version: 2
object_type: release_entry
versioning:
  schema_version: 1
  revision: 1
entry_id: entry-0002
release_version: v0.1.0
kind: added
summary:
  Added static voice level calibration with catalog or explicit dB gain, peak
  normalization, and voice level metadata
status: accepted
audience: null
scopes: []
source_refs:
  - git:8c87254fec22b4d4f312038a16f9a238bc9fbca8
paths:
  - supertonicsynth/voice_level.py
  - supertonicsynth/audio.py
  - supertonicsynth/data/voice_level_calibration.json
  - supertonicsynth/runtime.py
  - supertonicsynth/types.py
  - supertonicsynth/__main__.py
  - benchmarks/voice_level_benchmark.py
  - benchmarks/promote_voice_calibration.py
issues: []
prs: []
sources:
  - git:8c87254fec22b4d4f312038a16f9a238bc9fbca8
contributors:
  - "@holgern"
breaking: false
internal: false
order: 2
---

VoiceLevelConfig selects an offline calibrated static gain per semantic voice ref and language, with an explicit dB override taking precedence. Peak normalization and request-level output gain stay separate controls. Includes the counting benchmark and reviewed promotion tooling; the packaged catalog ships empty in this release.
