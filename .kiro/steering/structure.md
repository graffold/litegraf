# Project Structure

```
src/pipeline/                  # Main package (wheel target)
├── __init__.py                # Re-exports: LiteGraf, GraphStore, EmbeddingProvider, LLMProvider, JobStore
├── litegraf.py                # LiteGraf — single entry point class (dataclass)
├── interfaces.py              # Four core ABCs: GraphStore, EmbeddingProvider, LLMProvider, JobStore
├── config.py                  # PipelineConfig — env-var-based configuration
├── cli.py                     # CLI entry point (`litegraf` command)
├── tui.py                     # Interactive TUI entry point (`litegraf-tui` command)
│
├── backends/                  # Pluggable backend implementations
│   ├── neo4j_store.py         # GraphStore → Neo4j
│   ├── ollama_llm.py          # LLMProvider → Ollama
│   ├── bedrock_llm.py         # LLMProvider → AWS Bedrock
│   ├── cloudflare_llm.py      # LLMProvider → Cloudflare Workers AI
│   ├── local_embeddings.py    # EmbeddingProvider → sentence-transformers
│   └── sqlite_job_store.py    # JobStore → SQLite
│
├── dx/                        # Developer experience layer
│   ├── models.py              # Data models: InsertResult, QueryResult, ContextChunk
│   ├── cache.py               # LLM response caching
│   ├── dedup.py               # Content-hash deduplication
│   ├── limiter.py             # Async concurrency limiter
│   ├── registry.py            # BackendRegistry — resolves string shorthands to classes
│   └── sync_utils.py          # run_sync() helper for sync wrappers
│
├── ingest/                    # Text ingestion and extraction pipeline
│   ├── chunker.py             # Text chunking strategies
│   ├── kg_pipeline.py         # Knowledge graph construction pipeline
│   ├── extraction_models.py   # Extraction data models
│   └── ...                    # Various ingestors (biorxiv, pmc, csv, etc.)
│
├── processors/                # Plugin processors (extend via ProcessorBase ABC)
│   ├── base.py                # ProcessorBase ABC (name, process, graph_store)
│   └── ...                    # ~30 processors (entity resolution, enrichment, etc.)
│
├── enrichment/                # Graph enrichment orchestration
│   ├── base.py
│   ├── enrichment_orchestrator.py
│   └── ...
│
├── utils/                     # Shared utilities
│   ├── retry.py
│   ├── database_config.py
│   └── context_graph_interfaces.py
│
└── benchmarks/                # 4-axis benchmark suite
    ├── __main__.py            # Benchmark CLI entry
    ├── run_benchmark.py       # Benchmark runner
    ├── compare_providers.py   # LLM provider comparison
    ├── datasets/              # Gold-standard biomedical datasets (BC5CDR, ChemProt, GAD)
    ├── metrics/               # Extraction, KG quality, query, throughput metrics
    ├── generators/            # Synthetic test data generators
    ├── competitors/           # Competitor framework runners (LightRAG, MS GraphRAG, etc.)
    └── results/               # Benchmark result JSON files

tests/
├── conftest.py                # Shared fixtures (mock LLMs, DBs, embedders, sample data)
├── unit/                      # Unit tests (no external services)
├── integration/               # Integration tests (require live services)
├── properties/                # Property-based tests (Hypothesis)
└── helpers/                   # Test helper utilities
```

## Architecture Patterns
- **Interface-driven**: All backends implement ABCs from `interfaces.py`
- **Backend registry**: String shorthands (e.g., `"neo4j"`, `"ollama"`) resolved via `dx/registry.py`
- **Decorator/wrapper pattern**: DX features (cache, rate limiter) wrap LLMProvider instances
- **Plugin processors**: Extend `ProcessorBase` for custom processing without modifying core
- **Dataclass models**: `InsertResult`, `QueryResult`, `ContextChunk` in `dx/models.py`
- **Sync/async dual API**: Async is primary; sync methods use `run_sync()` wrapper
