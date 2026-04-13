# PRD: LiteGraf Multi-Mode Retrieval & Extraction Improvements

**Author:** Ralph
**Status:** Draft
**Created:** 2026-04-13
**Last Updated:** 2026-04-13

---

## 1. Overview

Improve litegraf's retrieval quality and ingestion pipeline by adopting proven patterns from LightRAG (HKUDS). The core thesis: embedding entities and relationships separately (not just chunks) and querying across multiple retrieval modes dramatically improves answer quality for knowledge graph RAG.

## 2. Goals

- Match LightRAG's retrieval quality via multi-mode query (local/global/hybrid)
- Improve entity extraction recall via gleaning
- Support entity deduplication and description merging across documents
- Keep the single-class, sensible-defaults API surface

## 3. Non-Goals

- WebUI / server mode (separate project)
- Multimodal document parsing (future)
- Replacing Neo4j as default graph store

---

## 4. Architecture Changes

### 4.1 New Vector Indexes

Three separate embedding spaces:
- **Chunk embeddings** (existing) — raw text chunks
- **Entity embeddings** (new) — embedded entity descriptions
- **Relationship embeddings** (new) — embedded relationship descriptions

### 4.2 Query Modes

| Mode | Retrieves from | Use case |
|------|---------------|----------|
| `naive` | Chunk embeddings only | Simple similarity search (current behavior) |
| `local` | Entity embeddings → expand neighborhoods | Entity-centric questions ("What does TP53 do?") |
| `global` | Relationship embeddings → high-level patterns | Broad questions ("How are kinases regulated?") |
| `hybrid` | Entity + Relationship embeddings combined | Default for best coverage |
| `mix` | All three (entities + relationships + chunks) | Maximum recall, use with reranker |

### 4.3 Extraction Pipeline Enhancement

```
Text → Chunk → LLM Extract → (Gleaning loop) → Merge with existing entities → Embed descriptions → Store
```

---

## 5. User Stories

### US-001: Multi-mode query parameter

**As a** developer using litegraf,
**I want to** specify a query mode (naive, local, global, hybrid, mix),
**So that** I can control retrieval strategy based on my question type.

**Acceptance Criteria:**
- `kg.query("...", mode="hybrid")` works
- Default mode is `hybrid`
- `naive` mode preserves current behavior exactly
- Mode is available on both `query()` and `aquery()`

---

### US-002: Entity description embeddings

**As a** developer,
**I want** entity descriptions to be embedded and stored in a separate vector index,
**So that** local-mode queries can find relevant entities by semantic similarity.

**Acceptance Criteria:**
- During insert, each extracted entity's description is embedded
- Embeddings stored in a dedicated Neo4j vector index (`entity_embeddings`)
- Entity nodes gain a `description` and `embedding` property
- Incremental: new inserts update embeddings for merged entities

---

### US-003: Relationship description embeddings

**As a** developer,
**I want** relationship descriptions to be embedded and stored in a separate vector index,
**So that** global-mode queries can find relevant high-level patterns.

**Acceptance Criteria:**
- During insert, each extracted relationship's description is embedded
- Embeddings stored in a dedicated Neo4j vector index (`relationship_embeddings`)
- Relationship edges gain a `description` and `embedding` property

---

### US-004: Entity gleaning (multi-pass extraction)

**As a** developer,
**I want** the extraction step to re-prompt the LLM for missed entities,
**So that** dense documents don't lose important entities on a single pass.

**Acceptance Criteria:**
- New parameter `max_gleaning: int = 1` on `LiteGraf`
- When `max_gleaning > 1`, after initial extraction the LLM is asked "Did you miss any entities or relationships?"
- Additional entities/rels are merged into the extraction result
- Setting `max_gleaning=1` (default) preserves current single-pass behavior

---

### US-005: Entity merge/summarization on re-insert

**As a** developer,
**I want** entities that appear across multiple documents to have their descriptions merged via LLM summarization,
**So that** the knowledge graph accumulates richer descriptions over time.

**Acceptance Criteria:**
- When upserting an entity that already exists, fetch existing description
- If descriptions differ, call LLM to produce a merged summary
- New parameter `enable_entity_merge: bool = True`
- Merged description is re-embedded

---

### US-006: Reranker interface and injection

**As a** developer,
**I want to** optionally provide a reranker that re-scores retrieved results,
**So that** I can improve precision without changing retrieval logic.

**Acceptance Criteria:**
- New `RerankerProvider` interface with `rerank(query, candidates) -> scored list`
- `LiteGraf(reranker="cross-encoder")` or `LiteGraf(reranker=MyReranker())`
- When no reranker is set, retrieval results are returned as-is
- Reranker runs after retrieval, before context assembly

---

### US-007: Token budget control for context assembly

**As a** developer,
**I want** query context to respect a token budget,
**So that** large retrievals don't overflow the LLM context window.

**Acceptance Criteria:**
- New parameters: `max_context_tokens: int = 8000`
- Context assembly truncates/prioritizes chunks within budget
- Entities and relationships each get a proportional share
- Token counting uses tiktoken (or configurable tokenizer)

---

### US-008: Document deletion with KG cleanup

**As a** developer,
**I want to** delete a previously inserted document and have the KG cleaned up,
**So that** stale or incorrect data can be removed.

**Acceptance Criteria:**
- `kg.delete(doc_id)` removes all chunks for that document
- Entities/relationships unique to that document are removed
- Shared entities/relationships are rebuilt from remaining sources
- Dedup index is updated so the same content can be re-inserted

---

### US-009: Conversation history for multi-turn queries

**As a** developer,
**I want to** pass conversation history to the query method,
**So that** follow-up questions have context from prior turns.

**Acceptance Criteria:**
- `kg.query("...", history=[{"role": "user", "content": "..."}, ...])` 
- History is passed to the LLM for response generation only (not retrieval)
- History is optional, default empty

---

### US-010: Insert custom KG data

**As a** developer,
**I want to** insert pre-extracted entities and relationships directly,
**So that** I can integrate external knowledge sources without re-extraction.

**Acceptance Criteria:**
- `kg.insert_kg(entities=[...], relationships=[...])` 
- Entities and relationships are upserted and embedded
- Dedup still applies (by entity name + type)
- No LLM calls are made

---

## 6. Technical Notes

- Entity/relationship descriptions should be generated during extraction (update the extraction prompt to return descriptions, not just name/type)
- The extraction prompt needs to output: `{name, type, description}` for entities and `{source, target, type, description, keywords}` for relationships
- Graph neighborhood expansion for local mode: given top-k entities, fetch 1-hop neighbors and their connecting relationships
- For global mode: given top-k relationships, include the source/target entity descriptions as context

## 7. Priority & Sequencing

| Phase | Stories | Rationale |
|-------|---------|-----------|
| Phase 1 | US-002, US-003, US-004, US-005 | Ingestion improvements (no query API changes) |
| Phase 2 | US-001, US-007 | Multi-mode query + token control |
| Phase 3 | US-006, US-009 | Reranker + conversation history |
| Phase 4 | US-008, US-010 | Delete + custom KG insert |

## 8. Open Questions

- [ ] Should `hybrid` or `mix` be the default mode?
- [ ] Do we want a `response_type` parameter (bullet points, paragraphs, etc.) or keep that out of scope?
- [ ] Should entity merge use a dedicated smaller/cheaper LLM call vs the main LLM?
- [ ] Max parallel inserts — should we expose `max_parallel_insert` like LightRAG does?
