# Product: litegraf

litegraf is a lightweight, pluggable knowledge graph ingestion and enrichment pipeline for biomedical / life sciences data. It extracts entities and relationships from text (plain text, PDFs, tabular data) using LLMs and stores them in a graph database.

## Core Value Proposition
- Single entry point via `LiteGraf()` dataclass — sensible defaults, override only what you need
- Pluggable backends for graph storage, LLM, embeddings, and job persistence
- Idempotent inserts via content-hash deduplication
- Sync + async dual API (`insert()`/`ainsert()`, `query()`/`aquery()`)
- LLM response caching and async concurrency limiting built in

## Domain
Biomedical / life sciences knowledge graphs. Primary entities: proteins, diseases, genes, and their relationships. Datasets and benchmarks drawn from biomedical corpora (BC5CDR, ChemProt, GAD).

## Key User Flows
1. **Insert** — text → LLM extracts entities/relationships → stored in graph DB
2. **Query** — question → vector similarity search → LLM synthesizes answer
3. **Context-only** — `only_context=True` for retrieval without LLM synthesis
4. **CLI / TUI** — `litegraf` CLI and `litegraf-tui` interactive terminal for benchmarks and operations
5. **Benchmarks** — 4-axis suite (extraction, KG quality, query, throughput) validating pipeline components

## License
AGPL-3.0-only
