# LiteGraf

Lightweight knowledge graph ingestion and query pipeline. Insert text or documents, extract entities and relationships with an LLM, store them in a graph database, and query with natural language.

```python
from pipeline.litegraf import LiteGraf

kg = LiteGraf()
kg.insert("TP53 is associated with multiple cancers including breast and lung cancer.")
result = kg.query("What cancers are associated with TP53?")
print(result.answer)
```

## Features

- **Single entry point** — `LiteGraf` dataclass with sensible defaults, override only what you need
- **Pluggable backends** — Neo4j, Memgraph, Ollama, Cloudflare Workers AI, AWS Bedrock
- **Sync and async** — `insert()` / `ainsert()`, `query()` / `aquery()`
- **Content deduplication** — hash-based, idempotent inserts
- **LLM response caching** — disk-based, avoids redundant API calls
- **Rate limiting** — async concurrency limiter for LLM providers
- **PDF and document ingestion** — via MarkItDown + PyMuPDF
- **Benchmarking suite** — compare extraction quality across LLM providers
- **Enrichment pipeline** — entity resolution, ontology integration, evidence scoring

## Install

Requires Python 3.11+.

```bash
pip install litegraf
```

With optional backends:

```bash
pip install litegraf[neo4j]       # Neo4j graph store
pip install litegraf[bedrock]     # AWS Bedrock LLM
pip install litegraf[all]         # Everything
```

Or from source with [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/graffold/litegraf.git
cd litegraf
uv sync --all-extras
```

## Quick Start

### Default setup (Ollama + Neo4j)

Start Ollama and Neo4j locally, then:

```python
from pipeline.litegraf import LiteGraf

kg = LiteGraf()  # connects to localhost defaults

# Insert text
kg.insert("BRCA1 interacts with RAD51 in DNA repair pathways.")

# Insert a PDF
kg.insert(open("paper.pdf", "rb").read())

# Query
result = kg.query("What proteins interact with BRCA1?")
print(result.answer)
print(result.context)  # retrieved graph context
```

### Cloudflare Workers AI (free tier)

```python
kg = LiteGraf(
    llm="cloudflare",
    llm_model="@cf/meta/llama-3.1-8b-instruct-fp8",
)
```

### Memgraph backend

```python
kg = LiteGraf(
    graph_store="memgraph",
    graph_uri="bolt://localhost:7687",
    graph_user="",
    graph_password="",
)
```

### Async usage

```python
import asyncio
from pipeline.litegraf import LiteGraf

async def main():
    kg = LiteGraf()
    await kg.ainsert("TP53 suppresses tumor growth.")
    result = await kg.aquery("What does TP53 do?")
    print(result.answer)

asyncio.run(main())
```

### Query modes

```python
# Full pipeline: retrieve context → LLM synthesis
result = kg.query("What cancers involve TP53?")

# Context only (bring your own LLM prompt)
result = kg.query("TP53", mode="only_context")
for chunk in result.context:
    print(chunk.text, chunk.score)
```

## Configuration

All parameters can be set via the `LiteGraf` constructor:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `graph_store` | `"neo4j"` | Graph backend: `"neo4j"`, `"memgraph"`, or instance |
| `graph_uri` | `"bolt://localhost:7687"` | Bolt connection URI |
| `graph_user` | `"neo4j"` | Graph database username |
| `graph_password` | `""` | Graph database password |
| `llm` | `"ollama"` | LLM provider: `"ollama"`, `"cloudflare"`, `"bedrock"` |
| `llm_model` | `"llama3"` | Model name/ID |
| `embedding` | `"local"` | Embedding provider (local sentence-transformers) |
| `chunk_token_size` | `512` | Tokens per chunk |
| `enable_cache` | `True` | Cache LLM responses to disk |
| `enable_dedup` | `True` | Skip duplicate content on insert |

## Benchmarks

Compare extraction quality across LLM providers on biomedical datasets:

```bash
python -m pipeline.benchmarks
```

Results are published to `docs/` for GitHub Pages viewing.

## Development

```bash
uv sync --all-extras --group dev
uv run pytest
uv run ruff check src/
```

## License

[AGPL-3.0](LICENSE)
