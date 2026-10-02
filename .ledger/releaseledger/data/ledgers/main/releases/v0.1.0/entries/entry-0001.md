---
schema_version: 2
object_type: release_entry
versioning:
  schema_version: 1
  revision: 1
entry_id: entry-0001
release_version: v0.1.0
kind: added
summary:
  Added the SupertonicSynth synthesis package with text chunking, voice and
  language selection, WAV output, and a CLI
status: accepted
audience: null
scopes: []
source_refs:
  - git:70688c003a35e495cde226ecee118d4ebb3b911f
  - git:1ef14f0de10cc52db0243d5a72525ecc6f8e92b9
paths:
  - LICENSE
  - README.md
  - pyproject.toml
  - supertonicsynth/__init__.py
  - supertonicsynth/__main__.py
  - supertonicsynth/runtime.py
  - supertonicsynth/types.py
  - supertonicsynth/errors.py
  - supertonicsynth/text_split.py
issues: []
prs: []
sources:
  - git:70688c003a35e495cde226ecee118d4ebb3b911f
  - git:1ef14f0de10cc52db0243d5a72525ecc6f8e92b9
contributors:
  - "@holgern"
breaking: false
internal: false
order: 1
---

SupertonicRuntime wraps the OnnxVoice supertonic adapter for text synthesis with per-language conditioning, semantic voice refs, deterministic chunking, request-level output gain, and synthesis metadata. Ships Apache-2.0 licensing, VCS-based packaging, typed errors, examples, and a synthesize/voices CLI.
