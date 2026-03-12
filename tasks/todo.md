# Memory Engine v0 — Sprint

## Steps
- [x] Step 1: Project scaffolding (pyproject.toml, structure, models, store init)
- [x] Step 2: LLM & embedding abstraction layer
- [x] Step 3: MemoryObserver (extract memories from conversations)
- [x] Step 4: MemoryStore (SQLite CRUD, decay, merge)
- [x] Step 5: MemoryRetriever (hybrid search + LLM re-ranking)
- [x] Step 6: PromptBuilder (format memories for injection)
- [x] Step 7: MemoryEngine orchestrator
- [x] Step 8: CLI demo
- [x] Step 9: Tests (25/25 passing)
- [x] Step 10: Verification & review

## Review
- All 25 tests pass with mock providers (no API calls needed)
- Fixed: pyproject.toml build-backend, test thresholds for mock embeddings
- Structure matches plan exactly
