---
schema_version: 2
object_type: release_entry
versioning:
  schema_version: 1
  revision: 1
entry_id: entry-0001
release_version: v0.1.3
kind: added
summary:
  Added a versioned request API contract with preflight measurement and capacity
  validation
status: accepted
audience: null
scopes: []
source_refs:
  - git:7cee17f150c1ed1a56d487575e13286134daaa12
paths:
  - .github/workflows/python-publish.yml
  - .github/workflows/tests.yml
  - README.md
  - docs/architecture.md
  - docs/troubleshooting.md
  - pyproject.toml
  - supertonicsynth/__init__.py
  - supertonicsynth/api_contract.py
  - supertonicsynth/discovery.py
  - tests/test_discovery.py
  - tests/test_public_api.py
  - tests/test_request_api_contract.py
  - tests/test_runtime_atomic.py
issues: []
prs: []
sources:
  - git:7cee17f150c1ed1a56d487575e13286134daaa12
contributors:
  - "@holgern"
breaking: false
internal: false
order: 1
---
