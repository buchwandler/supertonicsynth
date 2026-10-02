---
schema_version: 2
object_type: release_entry
versioning:
  schema_version: 1
  revision: 1
entry_id: entry-0002
release_version: v0.1.1
kind: changed
summary:
  Changed SynthesisConfig validation to enforce integer, finite, and seed range
  constraints and reject overflowing seeds
status: accepted
audience: null
scopes: []
source_refs:
  - git:6a261a52e47b402d56f970f66bac3a65f9d4e1b2
paths:
  - supertonicsynth/types.py
  - supertonicsynth/runtime.py
  - supertonicsynth/config.py
  - tests/test_types.py
  - tests/test_runtime_fake.py
issues: []
prs: []
sources:
  - git:6a261a52e47b402d56f970f66bac3a65f9d4e1b2
contributors:
  - "@holgern"
breaking: false
internal: false
order: 2
---
