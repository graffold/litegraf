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

## 2026-04-13 - US-002
- Implemented entity description embeddings during insert
- Updated extraction prompt to request `{name, type, description}` for entities
- `_store_extraction` now batch-embeds entity descriptions via `embed_documents()` and stores `description` + `embedding` properties on entity nodes
- Added `_ensure_entity_vector_index()` — creates `entity_embeddings` Neo4j vector index (768-dim, cosine) on init
- Implemented `_search_entity_index()` for querying the entity_embeddings index
- Refactored `_similarity_search()` into mode-aware routing: naive→chunks, local/hybrid→entities, mix→all
- Added `_search_chunk_index()` extracted from old monolithic method
- Incremental: re-inserting content that produces the same entity will upsert (MERGE) the node with updated description/embedding
- Files changed: `src/pipeline/litegraf.py`
- **Learnings:**
  - `upsert_node` uses MERGE on `id` property then `SET n += $props` — so adding `description` and `embedding` to props automatically handles incremental updates
  - Batch embedding via `embed_documents()` is more efficient than per-entity `embed_query()`
  - The vector index CREATE uses `IF NOT EXISTS` so it's safe to call on every init
  - Pre-existing test failures: missing `neo4j`, `pytest-asyncio`, `hypothesis` packages — not related to this change
---

## 2026-04-13 - US-004
- Added `max_gleaning: int = 1` dataclass field to `LiteGraf` under a new `# --- Extraction ---` section
- Modified `_extract_chunk()` to perform additional gleaning passes when `max_gleaning > 1`
- Each gleaning pass sends a prompt listing already-found entities and asks "Did you miss any entities or relationships?"
- New entities/relationships are deduplicated by normalized name (entities) and (source, target, type) tuple (relationships) before merging
- Early termination when a gleaning pass returns nothing new
- `max_gleaning=1` (default) preserves exact single-pass behavior — the gleaning code path is skipped entirely
- Files changed: `src/pipeline/litegraf.py`
- **Learnings:**
  - A `GleaningExtractor` processor already exists in `src/pipeline/processors/gleaning_extractor.py` with sync `invoke()` calls — but integrating gleaning directly into `_extract_chunk` using the async `self._llm.extract()` path is simpler and avoids sync/async mismatch
  - The existing `GleaningExtractor` uses a different prompt format (no `description` field) — the inline implementation matches the existing extraction prompt style (with `description`) for consistency with US-002/US-003 entity/relationship embedding
  - Pre-existing test failures: missing `hypothesis`, `pytest-asyncio` packages — not related to this change
---

## 2026-04-13 - US-005
- Added `enable_entity_merge: bool = True` dataclass field to `LiteGraf` under the `# --- Extraction ---` section
- Made `_store_extraction` async to support LLM calls for entity description merging
- When `enable_entity_merge` is True and an entity already exists in the graph with a different description, the LLM is called to produce a merged summary
- The merged description replaces the new description before embedding, so the re-embedded vector reflects the merged text
- When `enable_entity_merge` is False, or the entity is new, or descriptions match, no extra LLM call is made
- Updated `ainsert` call site to `await self._store_extraction(...)` 
- Files changed: `src/pipeline/litegraf.py`
- **Learnings:**
  - `_store_extraction` was sync but needed to become async to call `self._llm.ainvoke()` for merge — since it's only called from `ainsert` (async), this is safe
  - Entity node IDs follow the pattern `{label}:{name}` — used this to query for existing descriptions before upsert
  - The merge step runs before batch embedding, so the merged description gets properly embedded in the same pass
  - Pre-existing test failures: missing `hypothesis`, `pytest-asyncio`, `neo4j` packages — not related to this change
---

## 2026-04-13 - US-006
- Added `RerankerProvider` ABC to `src/pipeline/interfaces.py` with `rerank(query, candidates) -> scored list` method
- Created `src/pipeline/backends/cross_encoder_reranker.py` — `CrossEncoderReranker` using sentence-transformers `CrossEncoder`
- Added reranker resolution to `BackendRegistry` in `src/pipeline/dx/registry.py` — string shorthand `"cross-encoder"`, class reference, or instance passthrough
- Added `reranker: str | RerankerProvider | type[RerankerProvider] | None = None` field to `LiteGraf` dataclass
- Resolved `_reranker` in `__post_init__` via `BackendRegistry.resolve_reranker()` (only when not None)
- Wired reranker into `aquery()` after `_similarity_search()` and before context assembly/LLM call
- When no reranker is set (`None` default), retrieval results are returned as-is — zero behavior change
- Exported `RerankerProvider` from `src/pipeline/__init__.py`
- Files changed: `src/pipeline/interfaces.py`, `src/pipeline/backends/cross_encoder_reranker.py`, `src/pipeline/dx/registry.py`, `src/pipeline/litegraf.py`, `src/pipeline/__init__.py`
- **Learnings:**
  - The `BackendRegistry._resolve()` method handles all three input forms (string, class, instance) generically — adding a new backend type only requires a new registry dict, lazy loader, and `resolve_*` classmethod
  - Reranker is per-instance config (not per-query) since it's a heavyweight model load — follows the same pattern as `embedding` and `llm`
  - The `register()` method's category map also needs updating when adding a new backend type
  - Pre-existing test failures: missing `hypothesis`, `pytest-asyncio`, `neo4j` packages — not related to this change
---
