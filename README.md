# litegraf

```
▀▀      ▀▀▀▀▀▀▀ ▀▀▀▀▀▀▀ ▀▀▀▀▀▀▀  ▀▀▀▀▀  ▀▀▀▀▀▀   ▀▀▀▀▀  ▀▀▀▀▀▀▀
▀▀        ▀▀▀     ▀▀▀   ▀▀      ▀▀      ▀▀   ▀▀ ▀▀   ▀▀ ▀▀
▀▀        ▀▀▀     ▀▀▀   ▀▀▀▀▀   ▀▀  ▀▀▀ ▀▀▀▀▀▀  ▀▀▀▀▀▀▀ ▀▀▀▀▀
▀▀        ▀▀▀     ▀▀▀   ▀▀      ▀▀   ▀▀ ▀▀  ▀▀  ▀▀   ▀▀ ▀▀
▀▀▀▀▀▀▀ ▀▀▀▀▀▀▀   ▀▀▀   ▀▀▀▀▀▀▀  ▀▀▀▀▀▀ ▀▀   ▀▀ ▀▀   ▀▀ ▀▀
  v0.1.0  lightweight knowledge graph benchmark suite

  Ollama        http://localhost:11434  (3 models)
  Bedrock       123456789  (user)
  Embeddings    sentence-transformers  (5.4.0)
  Datasets      3/3 cached  (ready)

CLI
  bench      Run benchmarks with graphical output
  show       Render a previous benchmark result JSON
  insert     Insert text into the knowledge graph
  query      Query the knowledge graph
  status     Show system status and connectivity

Interactive mode  Ctrl+C to exit

  1  Show system status
  2  Run all benchmarks
  3  Benchmark: extraction only
  4  Benchmark: throughput only
  5  Compare LLM providers
  6  Insert text into KG
  7  Query the KG
  8  Render a result JSON

litegraf ❯
```

A lightweight, pluggable knowledge graph ingestion pipeline. Feed it text, PDFs, or tabular data — it extracts entities and relationships using LLMs and stores them in a graph database.

One class. Sensible defaults. Works out of the box.

```python
from pipeline import LiteGraf

kg = LiteGraf()
kg.insert("TP53 is associated with multiple cancers including breast cancer.")
result = kg.query("What cancers are associated with TP53?")
print(result.answer)
```

## Features

- **Single entry point** — `LiteGraf()` with all-defaults constructor, override only what you need
- **Pluggable backends** — Neo4j, Ollama, sentence-transformers, SQLite out of the box; swap via string shorthands or your own classes
- **Idempotent inserts** — content-hash deduplication, re-running is a no-op
- **LLM response caching** — same prompt = cached response, saves tokens during development
- **Async concurrency limiter** — caps concurrent LLM calls to prevent rate-limit explosions
- **Sync + async API** — `insert()` / `ainsert()`, `query()` / `aquery()`
- **`only_context` mode** — retrieve context without LLM synthesis, plug into your own prompt chains
- **CLI** — `litegraf run` and `litegraf enrich` for command-line usage
- **Processor plugins** — add custom processors without modifying core code

## Install

```bash
pip install litegraf

# With Neo4j support:
pip install "litegraf[neo4j]"

# Everything:
pip install "litegraf[all]"
```

## Quick Start

```python
from pipeline import LiteGraf

# All defaults — Neo4j + Ollama + sentence-transformers + SQLite
kg = LiteGraf()
kg.insert("Interleukin-6 is elevated in inflammatory diseases.")
result = kg.query("What role does IL-6 play in disease?")
print(result.answer)
```

### Custom backends

```python
kg = LiteGraf(
    graph_store="neo4j",
    graph_uri="bolt://localhost:7687",
    llm="ollama",
    llm_model="llama3",
    embedding_model="all-mpnet-base-v2",
    chunk_token_size=512,
)
```

### Async batch insert

```python
import asyncio

async def main():
    kg = LiteGraf(max_async_calls=8)
    result = await kg.ainsert([
        "BRCA1 mutations increase breast cancer risk.",
        "Metformin is used to treat type 2 diabetes.",
    ])
    print(f"{result.chunks_processed} chunks, {result.entities_extracted} entities")

asyncio.run(main())
```

### Context-only retrieval

```python
context = kg.query("What proteins relate to heart disease?", only_context=True)
# context.answer is None — just the retrieved chunks
for chunk in context.context:
    print(f"[{chunk.score:.2f}] {chunk.text[:100]}...")
```

### Bring your own backends

```python
from pipeline.backends.neo4j_store import Neo4jGraphStore
from pipeline.backends.ollama_llm import OllamaLLMProvider

graph = Neo4jGraphStore(uri="bolt://localhost:7687", auth=("neo4j", "pass"))
llm = OllamaLLMProvider(model="mistral")
kg = LiteGraf(graph_store=graph, llm=llm, enable_cache=False)
```

## CLI

```bash
litegraf run --query "TP53 cancer" --max-results 10 --graph-uri bolt://localhost:7687
litegraf enrich --file data.csv --graph-uri bolt://localhost:7687
litegraf run --query "BRCA1" --config config.yaml
```

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `graph_store` | `"neo4j"` | Graph backend (string, class, or instance) |
| `llm` | `"ollama"` | LLM backend |
| `embedding` | `"local"` | Embedding backend |
| `chunk_token_size` | `512` | Tokens per chunk |
| `enable_cache` | `True` | Cache LLM responses to disk |
| `enable_dedup` | `True` | Skip duplicate content on insert |
| `max_async_calls` | `16` | Max concurrent LLM calls |

## Custom Backends

Implement any of the four interfaces:

```python
from pipeline.interfaces import GraphStore

class MyGraphStore(GraphStore):
    def execute_query(self, query, params=None): ...
    def upsert_node(self, label, properties): ...
    def upsert_relationship(self, source_id, rel_type, target_id, properties=None): ...
    def close(self): ...

kg = LiteGraf(graph_store=MyGraphStore())
```

## Benchmarks

litegraf includes a 4-axis benchmark suite that validates each pipeline component
independently — no graph database required, just an LLM endpoint:

| Axis | What it measures |
|------|-----------------|
| `extraction` | NER F1 against gold-standard biomedical datasets (bc5cdr, chemprot, gad) |
| `kg-quality` | Entity consolidation accuracy, contradiction detection, provenance validation |
| `query` | Mode routing, answer relevance (LLM-as-judge), latency percentiles |
| `throughput` | Docs/min, embedding rate, peak memory, estimated token cost |

```bash
# Interactive TUI
uv run litegraf-tui

# Or directly
uv run litegraf-tui bench --all
uv run litegraf-tui bench --compare-models --docs 10
```

> End-to-end KG construction benchmarks (Neo4j insert → query) live in
> [graffold-api](https://github.com/graffold/graffold-api), which uses these
> pre-validated components.

See [`src/pipeline/benchmarks/README.md`](src/pipeline/benchmarks/README.md) for full details.

## Development

```bash
git clone https://github.com/graffold/litegraf.git
cd litegraf
pip install -e ".[all]"
pytest tests/ -m "not integration"
ruff check pipeline/
```

## License

[AGPL-3.0-only](LICENSE) — free to use, modify, and distribute. If you modify it or use it in a network service, you must make your source code available under the same license.
