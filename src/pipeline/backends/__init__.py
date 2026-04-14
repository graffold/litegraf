"""Default backend implementations for the pipeline interfaces.

Re-exports:
- SQLiteJobStore (pipeline.backends.sqlite_job_store)
- LocalEmbeddingProvider (pipeline.backends.local_embeddings)
- OllamaLLMProvider (pipeline.backends.ollama_llm)
- Neo4jGraphStore (pipeline.backends.neo4j_store)

Backends that have uninstalled dependencies are silently skipped so the
module remains importable even when optional extras are missing.
"""

__all__ = [
    "CloudflareLLMProvider",
    "LocalEmbeddingProvider",
    "Neo4jGraphStore",
    "OllamaLLMProvider",
    "SQLiteJobStore",
]

try:
    from pipeline.backends.sqlite_job_store import SQLiteJobStore
except ImportError:
    pass

try:
    from pipeline.backends.local_embeddings import LocalEmbeddingProvider
except ImportError:
    pass

try:
    from pipeline.backends.ollama_llm import OllamaLLMProvider
except ImportError:
    pass

try:
    from pipeline.backends.cloudflare_llm import CloudflareLLMProvider
except ImportError:
    pass

try:
    from pipeline.backends.neo4j_store import Neo4jGraphStore
except ImportError:
    pass
