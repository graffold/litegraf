# Ralph Progress Log

This file tracks progress across iterations. Agents update this file
after each iteration and it's included in prompts for context.

## Codebase Patterns (Study These First)

- **Sync/async dual API**: Every public async method (`aquery`, `ainsert`) has a sync wrapper that calls `run_sync(self.async_method(...))`. New parameters must be added to both and threaded through.
- **Per-query vs per-instance config**: Instance-level config goes as `@dataclass` fields (resolved in `__post_init__`). Per-call options go as keyword args on `query()`/`aquery()`.
- **Retrieval entry point**: `_similarity_search()` is the single internal method for all retrieval — mode routing branches here.

---


## 2026-04-13 - US-001
- Added `QueryMode` Literal type (`"naive"`, `"local"`, `"global"`, `"hybrid"`, `"mix"`) and `QUERY_MODES` validation set to `src/pipeline/dx/models.py`
- Added `mode: QueryMode = "hybrid"` keyword argument to both `aquery()` and `query()` in `src/pipeline/litegraf.py`
- Added runtime validation: invalid mode raises `ValueError`
- Passed `mode` through to `_similarity_search()` for future routing (US-002/US-003 will add entity/relationship embedding paths)
- `naive` mode preserves current chunk-only behavior exactly; other modes currently fall back to the same path until entity/relationship indexes exist
- Exported `QueryMode` from `src/pipeline/__init__.py`
- Files changed: `src/pipeline/dx/models.py`, `src/pipeline/litegraf.py`, `src/pipeline/__init__.py`
- **Learnings:**
  - `LiteGraf` is a `@dataclass` with `__post_init__` for backend resolution — new fields go on the class, but `mode` is per-query not per-instance so it's a method param only
  - Sync wrappers (`query`, `insert`) delegate to async versions via `run_sync()` — any new param must be threaded through both
  - Existing test suite has pre-existing failures: missing `pytest-asyncio` for async tests, missing `neo4j` for backend resolution tests — not related to this change
  - `_similarity_search` is the retrieval entry point — this is where mode-specific routing will be added when entity/relationship embeddings land
---
