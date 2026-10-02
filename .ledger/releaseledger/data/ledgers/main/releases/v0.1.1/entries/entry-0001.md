---
schema_version: 2
object_type: release_entry
versioning:
  schema_version: 1
  revision: 1
entry_id: entry-0001
release_version: v0.1.1
kind: added
summary:
  Added 252 reviewed voice and language calibrations to the packaged voice
  level calibration catalog
status: accepted
audience: null
scopes: []
source_refs:
  - git:1614c4cc1c0b49701be2f78a92de16516ad42573
paths:
  - supertonicsynth/data/voice_level_calibration.json
  - tests/test_voice_leveling.py
issues: []
prs: []
sources:
  - git:1614c4cc1c0b49701be2f78a92de16516ad42573
contributors:
  - "@holgern"
breaking: false
internal: false
order: 1
---

Coverage is partial by design: 48 completed but high-variability identities and all 10 Croatian identities are absent from the 310-key matrix. Missing identities continue to resolve to 0 dB and leave audio unchanged.
