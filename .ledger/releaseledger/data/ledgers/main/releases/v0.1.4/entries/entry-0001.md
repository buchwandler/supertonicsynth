---
schema_version: 2
object_type: release_entry
versioning:
  schema_version: 1
  revision: 1
entry_id: entry-0001
release_version: v0.1.4
kind: changed
summary:
  Changed Supertonic-3 request measurement, discovery, and synthesis to enforce
  a versioned 512 model-token limit
status: accepted
audience: null
scopes: []
source_refs:
  - git:ebfaf679c994a5b8d0cadb6c03244cdeb39f03cb
paths:
  - .github/workflows/python-publish.yml
  - README.md
  - docs/architecture.md
  - supertonicsynth/config.py
  - supertonicsynth/discovery.py
  - supertonicsynth/errors.py
  - supertonicsynth/runtime.py
  - tests/test_discovery.py
  - tests/test_request_api_contract.py
  - tests/test_runtime_atomic.py
  - tests/test_types.py
issues: []
prs: []
sources:
  - git:ebfaf679c994a5b8d0cadb6c03244cdeb39f03cb
contributors:
  - "@holgern"
breaking: false
internal: false
order: 1
---
