"""Lightweight configuration for the pipeline package.

Reads settings from environment variables, providing sensible defaults
for standalone open-source usage.
"""
import os


class PipelineConfig:
    """Read-only configuration from environment variables."""

    # Entrez / NCBI
    ENTREZ_EMAIL = os.environ.get("ENTREZ_EMAIL", "")
    ENTREZ_API_KEY = os.environ.get("ENTREZ_API_KEY", "")

    # Graph database (Memgraph default, Neo4j compatible)
    GRAPH_URI = os.environ.get("GRAPH_URI", os.environ.get("NEO4J_URI", "bolt://localhost:7687"))
    GRAPH_USER = os.environ.get("GRAPH_USER", os.environ.get("NEO4J_USER", ""))
    GRAPH_PASSWORD = os.environ.get("GRAPH_PASSWORD", os.environ.get("NEO4J_PASSWORD", ""))
    GRAPH_DATABASE = os.environ.get("GRAPH_DATABASE", os.environ.get("NEO4J_DATABASE", ""))

    # Backwards compat aliases
    NEO4J_URI = GRAPH_URI
    NEO4J_USER = GRAPH_USER
    NEO4J_PASSWORD = GRAPH_PASSWORD
    NEO4J_DATABASE = GRAPH_DATABASE

    # Tokenizers
    TOKENIZERS_PARALLELISM = os.environ.get("TOKENIZERS_PARALLELISM", "false")

    # Chunking — token-based chunker settings
    # Default 1024 tokens per chunk (up from 512) to reduce LLM call count.
    # Llama 3.1 8B has 128K context; 1024-token chunks are well within budget.
    CHUNK_MAX_TOKENS = int(os.environ.get("CHUNK_MAX_TOKENS", "1024"))
    CHUNK_OVERLAP_TOKENS = int(os.environ.get("CHUNK_OVERLAP_TOKENS", "128"))

    # LLM / Bedrock
    BEDROCK_MODEL_ID = os.environ.get("BEDROCK_MODEL_ID", "")
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
    HUGGINGFACE_API_KEY = os.environ.get("HUGGINGFACE_API_KEY", "")
    SAGEMAKER_ENDPOINT_NAME = os.environ.get("SAGEMAKER_ENDPOINT_NAME", "")

    # Cloudflare Workers AI
    CF_ACCOUNT_ID = os.environ.get("CF_ACCOUNT_ID", "")
    CF_API_TOKEN = os.environ.get("CF_API_TOKEN", "")
    CF_MODEL = os.environ.get("CF_MODEL", "@cf/meta/llama-3.1-8b-instruct")

    # Vision
    VISION_MODEL_ID = os.environ.get("VISION_MODEL_ID", "")

    # Sentence-level provenance toggle
    ENABLE_SENTENCE_PROVENANCE = (
        os.environ.get("ENABLE_SENTENCE_PROVENANCE", "true").lower() == "true"
    )

    @classmethod
    def get_config(cls, key: str, default: str = "") -> str:
        """Get a config value by key name, matching src.config.Config interface."""
        return getattr(cls, key, None) or os.environ.get(key, default)

    @classmethod
    def get_consolidation_enabled(cls) -> bool:
        return os.environ.get("CONSOLIDATION_ENABLED", "false").lower() == "true"

    @classmethod
    def get_consolidation_mode(cls) -> str:
        return os.environ.get("CONSOLIDATION_MODE", "comprehensive")
