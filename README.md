# litegraf

> **Note:** This package name is a placeholder — rename before publishing.

A pluggable knowledge-graph ingestion and enrichment pipeline for biomedical literature. Extracts entities and relationships from PubMed abstracts, bioRxiv preprints, PMC full-text articles, and tabular data, then stores them in a graph database.

## Features

- **Abstract interfaces** for graph storage, embeddings, LLM extraction, and job persistence — swap backends without touching pipeline code
- **Default backends** included: Neo4j, sentence-transformers, Ollama, SQLite
- **CLI** for running ingestion and enrichment from the command line
- **Processor plugin architecture** — add new processors without modifying core code
- **Property-based test suite** with Hypothesis

## Quick Start

```bash
pip install .
# or with Neo4j support:
pip install ".[neo4j]"
```

### CLI Usage

```bash
# Run ingestion pipeline
biokg-ingest run --query "TP53 cancer" --max-results 10 --graph-uri bolt://localhost:7687

# Run enrichment pipeline
biokg-ingest enrich --file data.csv --graph-uri bolt://localhost:7687

# Load config from YAML
biokg-ingest run --query "BRCA1" --config config.yaml
```

### Python API

```python
from pipeline.interfaces import GraphStore, EmbeddingProvider, LLMProvider
from pipeline.backends import Neo4jGraphStore, LocalEmbeddingProvider, OllamaLLMProvider
from pipeline.ingest.kg_pipeline import KGPipeline

graph = Neo4jGraphStore(uri="bolt://localhost:7687", auth=("neo4j", "pass"))
embedder = LocalEmbeddingProvider()  # all-mpnet-base-v2, 768 dims
llm = OllamaLLMProvider()           # llama3 on localhost:11434

pipeline = KGPipeline(graph, embedder, llm)
await pipeline.run(search_term="TP53 cancer", max_results=5)
```

### Custom Backends

Implement any of the four interfaces to plug in your own backend:

```python
from pipeline.interfaces import GraphStore

class MyGraphStore(GraphStore):
    def execute_query(self, query, params=None): ...
    def upsert_node(self, label, properties): ...
    def upsert_relationship(self, source_id, rel_type, target_id, properties=None): ...
    def close(self): ...
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `NEO4J_URI` | `bolt://localhost:7687` | Neo4j connection URI |
| `NEO4J_USER` | `neo4j` | Neo4j username |
| `NEO4J_PASSWORD` | `password` | Neo4j password |
| `NEO4J_DATABASE` | `neo4j` | Neo4j database name |
| `ENTREZ_EMAIL` | | Email for NCBI Entrez API |
| `ENTREZ_API_KEY` | | API key for NCBI Entrez |

### YAML Config

```yaml
graph_uri: "bolt://localhost:7687"
graph_user: "neo4j"
graph_password: "password"
graph_database: "neo4j"
embedding_model: "all-mpnet-base-v2"
llm_model: "llama3"
llm_url: "http://localhost:11434"
```

## Package Structure

```
pipeline/
├── interfaces.py          # GraphStore, EmbeddingProvider, LLMProvider, JobStore
├── config.py              # Environment-based configuration
├── cli.py                 # CLI entrypoints
├── backends/              # Default backend implementations
│   ├── neo4j_store.py     # Neo4jGraphStore (optional neo4j dep)
│   ├── local_embeddings.py # LocalEmbeddingProvider (sentence-transformers)
│   ├── ollama_llm.py      # OllamaLLMProvider
│   └── sqlite_job_store.py # SQLiteJobStore
├── ingest/                # Ingestion pipeline modules
├── enrichment/            # Enrichment pipeline modules
├── processors/            # Pluggable processor modules
└── utils/                 # Standalone utilities
```

## Optional Dependencies

```bash
pip install ".[neo4j]"    # Neo4j driver
pip install ".[bedrock]"  # AWS Bedrock SDK
pip install ".[redis]"    # Redis client
pip install ".[all]"      # Everything
```

## Development

```bash
# Install dev dependencies
pip install -e ".[all]"
pip install pytest hypothesis pytest-asyncio

# Run tests
pytest tests/ -m "not integration"

# Run property-based tests
pytest tests/properties/ -m properties

# Lint
ruff check pipeline/
```

## License

MIT
