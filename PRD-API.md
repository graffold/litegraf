# PRD: LiteGraf API Server

**Author:** Ralph
**Status:** Draft
**Created:** 2026-04-13
**Last Updated:** 2026-04-13

---

## 1. Overview

A lightweight HTTP API server that exposes litegraf's knowledge graph capabilities over REST. Designed to integrate with chat frontends (Open WebUI, custom UIs) via Ollama-compatible endpoints, while also providing first-class REST endpoints for document management, querying, and KG manipulation.

## 2. Goals

- Expose litegraf insert/query/delete over HTTP with streaming support
- Ollama-compatible chat interface so any Ollama-aware frontend works out of the box
- Async document indexing with progress tracking
- Simple auth (API key + optional JWT)
- Single deployable process, production-ready with Gunicorn

## 3. Non-Goals

- WebUI (separate project, consumes this API)
- Multi-tenant billing/metering
- GraphQL (REST only for now)
- Replacing the Python library API — the server wraps `LiteGraf`, it doesn't replace it

## 4. Tech Stack

- FastAPI + Uvicorn (dev/single-worker)
- Gunicorn + Uvicorn workers (production)
- Auto-generated OpenAPI docs at `/docs` (Swagger) and `/redoc`
- Configuration via `.env` file + CLI args (CLI takes precedence)

---

## 5. Architecture

```
┌─────────────────────────────────────────────┐
│  litegraf-server                            │
│                                             │
│  /api/chat ──────┐                          │
│  /api/generate ──┤── Ollama compat layer    │
│                  │                          │
│  /query ─────────┤                          │
│  /query/stream ──┤── Query endpoints        │
│                  │                          │
│  /documents/* ───┤── Document management    │
│                  │                          │
│  /entities/* ────┤── KG CRUD                │
│  /relationships/*┤                          │
│                  │                          │
│  /export ────────┘── Data export            │
│                                             │
│         ┌──────────────┐                    │
│         │   LiteGraf   │  (library)         │
│         └──────────────┘                    │
└─────────────────────────────────────────────┘
```

### Request Priority

When indexing and querying share the same LLM concurrency pool:
1. **Query** — highest priority (user-facing latency)
2. **Entity merge** — medium (maintains KG quality)
3. **Extraction** — lowest (background indexing)

---

## 6. User Stories

### US-001: Query endpoint

**As a** client application,
**I want to** POST a query and receive a JSON response with the answer and references,
**So that** I can integrate litegraf into any application.

**Acceptance Criteria:**
- `POST /query` with body `{"query": "...", "mode": "hybrid"}`
- Response includes `answer`, `references[]`, `duration_seconds`
- Supports `mode`: naive, local, global, hybrid, mix
- Optional `only_context: true` returns references without LLM synthesis
- Optional `include_chunk_content: true` includes retrieved text in references

---

### US-002: Streaming query

**As a** frontend developer,
**I want to** receive the LLM response as a server-sent event stream,
**So that** users see tokens appear in real-time.

**Acceptance Criteria:**
- `POST /query/stream` with same body as `/query`
- Response is `text/event-stream` (SSE)
- Each event contains a token chunk
- Final event includes references metadata
- Connection closes cleanly on completion or error

---

### US-003: Ollama-compatible chat endpoint

**As a** user of Open WebUI or similar Ollama-aware frontends,
**I want** litegraf to appear as an Ollama model,
**So that** I can query my knowledge graph from any Ollama-compatible chat UI.

**Acceptance Criteria:**
- `POST /api/chat` accepts Ollama chat format (`model`, `messages[]`, `stream`)
- `POST /api/generate` accepts Ollama generate format
- `GET /api/tags` returns litegraf as an available model (`litegraf:latest`)
- Query mode selectable via message prefix (`/local`, `/global`, `/mix`, etc.)
- Session-management requests (title generation) forwarded to underlying LLM

---

### US-004: User prompt separation

**As a** developer,
**I want to** provide a `user_prompt` that guides LLM output formatting without affecting retrieval,
**So that** I can request specific output formats (bullet points, mermaid diagrams) without degrading search quality.

**Acceptance Criteria:**
- `/query` body accepts optional `user_prompt` field
- `user_prompt` is injected into the LLM prompt after context assembly
- `user_prompt` does NOT influence retrieval/embedding search
- Ollama endpoints support `[prompt in brackets]` syntax: `/mix[use bullet points] What is TP53?`

---

### US-005: Document upload

**As a** user,
**I want to** upload documents via HTTP for indexing,
**So that** I don't need Python code to add content to the knowledge graph.

**Acceptance Criteria:**
- `POST /documents/upload` accepts multipart file upload (txt, pdf, csv, docx)
- Returns `{"track_id": "..."}` immediately
- Indexing happens asynchronously in background
- Configurable max upload size (default 100MB)

---

### US-006: Document text insert

**As a** client application,
**I want to** insert raw text via the API,
**So that** I can programmatically feed content without file uploads.

**Acceptance Criteria:**
- `POST /documents/text` with body `{"content": "...", "id": "optional-id"}`
- `POST /documents/texts` for batch insert `{"contents": ["...", "..."]}`
- Returns `track_id` for async progress tracking
- Deduplication still applies

---

### US-007: Async indexing progress tracking

**As a** frontend,
**I want to** poll document processing status,
**So that** I can show progress to users.

**Acceptance Criteria:**
- `GET /track_status/{track_id}` returns status object
- Status includes: `status` (pending/processing/completed/failed), `progress` (0-100), `error` (if failed), `result` (on completion)
- Track IDs expire after configurable TTL (default 24h)

---

### US-008: Document scan

**As an** operator,
**I want to** trigger a scan of an input directory for new files,
**So that** I can drop files into a folder and have them indexed.

**Acceptance Criteria:**
- `POST /documents/scan` scans configured `input_dir`
- Returns list of newly discovered files and their track IDs
- Already-indexed files (by content hash) are skipped
- Configurable `input_dir` via `.env` or CLI

---

### US-009: Document deletion

**As a** user,
**I want to** delete a document via the API,
**So that** stale content is removed from the knowledge graph.

**Acceptance Criteria:**
- `DELETE /documents/{doc_id}`
- Triggers KG cleanup (removes unique entities/rels, rebuilds shared ones)
- Returns confirmation with count of removed entities/relationships

---

### US-010: Entity CRUD

**As a** power user or UI,
**I want to** create, read, update, and delete entities via the API,
**So that** I can manually curate the knowledge graph.

**Acceptance Criteria:**
- `GET /entities?search=TP53` — search/list entities
- `POST /entities` — create entity `{"name": "...", "type": "...", "description": "..."}`
- `PUT /entities/{name}` — update entity properties
- `DELETE /entities/{name}` — delete entity and its relationships
- All mutations trigger re-embedding of affected descriptions

---

### US-011: Relationship CRUD

**As a** power user or UI,
**I want to** manage relationships between entities via the API,
**So that** I can correct or enrich the knowledge graph manually.

**Acceptance Criteria:**
- `GET /relationships?entity=TP53` — list relationships for an entity
- `POST /relationships` — create `{"source": "...", "target": "...", "type": "...", "description": "..."}`
- `PUT /relationships/{id}` — update relationship properties
- `DELETE /relationships/{id}` — delete relationship

---

### US-012: KG export

**As a** researcher,
**I want to** export the knowledge graph in common formats,
**So that** I can analyze it externally or back it up.

**Acceptance Criteria:**
- `GET /export?format=csv` (default)
- Supported formats: `csv`, `json`, `graphml`
- Optional `include_embeddings=true` for full vector data
- Streams large exports rather than buffering in memory

---

### US-013: API key authentication

**As an** operator,
**I want to** protect the API with an API key,
**So that** unauthorized clients cannot access the server.

**Acceptance Criteria:**
- Configure via `LITEGRAF_API_KEY` env var
- Clients pass key in `X-API-Key` header
- Configurable whitelist paths (e.g., `/health` always open)
- 401 response with clear error when key is missing/invalid

---

### US-014: JWT authentication

**As an** operator deploying for multiple users,
**I want** JWT-based auth with username/password login,
**So that** I can control access per user.

**Acceptance Criteria:**
- `POST /auth/login` with `{"username": "...", "password": "..."}` returns JWT
- Accounts configured via `AUTH_ACCOUNTS` env var
- Passwords stored as bcrypt hashes
- Configurable token expiry (default 4h)
- JWT validated on all protected endpoints

---

### US-015: Bypass mode

**As a** chat user,
**I want to** sometimes talk to the LLM directly without RAG retrieval,
**So that** I can have general conversation in the same interface.

**Acceptance Criteria:**
- `/bypass` prefix in Ollama chat skips RAG, passes directly to LLM
- Chat history is included in the LLM call
- Works via both Ollama endpoints and `/query` with `mode: "bypass"`

---

### US-016: Health and info endpoints

**As an** operator or load balancer,
**I want** health check and server info endpoints,
**So that** I can monitor the service.

**Acceptance Criteria:**
- `GET /health` returns `{"status": "ok"}` (always unauthenticated)
- `GET /info` returns server version, configured backends, workspace name
- Health check verifies graph store connectivity

---

### US-017: Workspace isolation

**As an** operator running multiple tenants,
**I want to** isolate data between different workspaces on the same server,
**So that** one deployment can serve multiple independent knowledge graphs.

**Acceptance Criteria:**
- Configurable via `--workspace` CLI arg or `WORKSPACE` env var
- Workspace name prefixes/isolates all storage (graph labels, vector indexes, cache keys)
- Multiple server instances can share the same backend with different workspaces
- Default workspace is empty string (no prefix)

---

## 7. Configuration

### Environment Variables (`.env`)

| Variable | Default | Description |
|----------|---------|-------------|
| `HOST` | `0.0.0.0` | Server bind address |
| `PORT` | `9621` | Server port |
| `WORKERS` | `1` | Gunicorn worker count |
| `WORKING_DIR` | `./litegraf_storage` | Data persistence directory |
| `INPUT_DIR` | `./inputs` | Directory to scan for documents |
| `WORKSPACE` | `` | Workspace name for data isolation |
| `MAX_ASYNC` | `16` | Max concurrent LLM calls |
| `MAX_PARALLEL_INSERT` | `2` | Max files indexed in parallel |
| `TIMEOUT` | `150` | LLM request timeout (seconds) |
| `MAX_UPLOAD_SIZE` | `100MB` | Max file upload size |
| `LITEGRAF_API_KEY` | `` | API key (empty = no auth) |
| `AUTH_ACCOUNTS` | `` | JWT accounts (`user:hash,...`) |
| `TOKEN_SECRET` | `` | JWT signing secret |
| `TOKEN_EXPIRE_HOURS` | `4` | JWT token expiry |
| `WHITELIST_PATHS` | `/health` | Paths exempt from auth |

### CLI Arguments (override `.env`)

```
litegraf-server --port 9621 --workspace myproject --working-dir ./data --log-level INFO
litegraf-server-prod --workers 4  # Gunicorn mode
```

---

## 8. Priority & Sequencing

| Phase | Stories | Rationale |
|-------|---------|-----------|
| Phase 1 | US-001, US-002, US-005, US-006, US-007, US-016 | Core query + insert + health (MVP) |
| Phase 2 | US-003, US-004, US-015 | Ollama compat + chat UX |
| Phase 3 | US-008, US-009, US-010, US-011 | Document management + KG CRUD |
| Phase 4 | US-012, US-013, US-014, US-017 | Export, auth, multi-tenancy |

## 9. Open Questions

- [ ] Should we support WebSocket as alternative to SSE for streaming?
- [ ] Rate limiting per API key — needed for Phase 1 or defer?
- [ ] Should `/api/chat` support multi-turn conversation history natively, or expect the client to manage it?
- [ ] Do we want a `/graph/visualize` endpoint that returns a D3-compatible JSON subgraph?
- [ ] Pagination strategy for entity/relationship list endpoints?
