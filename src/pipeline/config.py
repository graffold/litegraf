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

    # Neo4j
    NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
    NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "password")
    NEO4J_DATABASE = os.environ.get("NEO4J_DATABASE", "neo4j")

    # Tokenizers
    TOKENIZERS_PARALLELISM = os.environ.get("TOKENIZERS_PARALLELISM", "false")

    # LLM / Bedrock
    BEDROCK_MODEL_ID = os.environ.get("BEDROCK_MODEL_ID", "")
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
    HUGGINGFACE_API_KEY = os.environ.get("HUGGINGFACE_API_KEY", "")
    SAGEMAKER_ENDPOINT_NAME = os.environ.get("SAGEMAKER_ENDPOINT_NAME", "")

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
