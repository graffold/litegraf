"""Backend registry — resolves string shorthands, class references, and instances.

Supports three input forms for each backend type:

1. **String shorthand** (``"neo4j"``, ``"ollama"``, …) → looked up and instantiated.
2. **Class reference** (a subclass of the ABC) → instantiated with kwargs.
3. **Pre-configured instance** → returned unchanged.
"""

from __future__ import annotations

from typing import Any

from pipeline.interfaces import EmbeddingProvider, GraphStore, JobStore, LLMProvider, RerankerProvider


def _lazy_neo4j() -> type[GraphStore]:
    from pipeline.backends.neo4j_store import Neo4jGraphStore

    return Neo4jGraphStore


def _lazy_local_embedding() -> type[EmbeddingProvider]:
    from pipeline.backends.local_embeddings import LocalEmbeddingProvider

    return LocalEmbeddingProvider


def _lazy_ollama() -> type[LLMProvider]:
    from pipeline.backends.ollama_llm import OllamaLLMProvider

    return OllamaLLMProvider


def _lazy_bedrock_llm() -> type[LLMProvider]:
    from pipeline.backends.bedrock_llm import BedrockLLMProvider

    return BedrockLLMProvider


def _lazy_cloudflare_llm() -> type[LLMProvider]:
    from pipeline.backends.cloudflare_llm import CloudflareLLMProvider

    return CloudflareLLMProvider


def _lazy_sqlite() -> type[JobStore]:
    from pipeline.backends.sqlite_job_store import SQLiteJobStore

    return SQLiteJobStore


def _lazy_cross_encoder() -> type[RerankerProvider]:
    from pipeline.backends.cross_encoder_reranker import CrossEncoderReranker

    return CrossEncoderReranker


# Mapping from shorthand → lazy loader returning the class
_GRAPH_STORES: dict[str, Any] = {"neo4j": _lazy_neo4j}
_EMBEDDING_PROVIDERS: dict[str, Any] = {"local": _lazy_local_embedding}
_LLM_PROVIDERS: dict[str, Any] = {"ollama": _lazy_ollama, "bedrock": _lazy_bedrock_llm, "cloudflare": _lazy_cloudflare_llm}
_JOB_STORES: dict[str, Any] = {"sqlite": _lazy_sqlite}
_RERANKERS: dict[str, Any] = {"cross-encoder": _lazy_cross_encoder}


class BackendRegistry:
    """Resolves backend specifications to live instances."""

    @classmethod
    def resolve_graph_store(
        cls, spec: str | GraphStore | type[GraphStore], **kwargs: Any
    ) -> GraphStore:
        return cls._resolve(spec, _GRAPH_STORES, GraphStore, "graph_store", **kwargs)

    @classmethod
    def resolve_embedding(
        cls, spec: str | EmbeddingProvider | type[EmbeddingProvider], **kwargs: Any
    ) -> EmbeddingProvider:
        return cls._resolve(
            spec, _EMBEDDING_PROVIDERS, EmbeddingProvider, "embedding", **kwargs
        )

    @classmethod
    def resolve_llm(
        cls, spec: str | LLMProvider | type[LLMProvider], **kwargs: Any
    ) -> LLMProvider:
        return cls._resolve(spec, _LLM_PROVIDERS, LLMProvider, "llm", **kwargs)

    @classmethod
    def resolve_job_store(
        cls, spec: str | JobStore | type[JobStore], **kwargs: Any
    ) -> JobStore:
        return cls._resolve(spec, _JOB_STORES, JobStore, "job_store", **kwargs)

    @classmethod
    def resolve_reranker(
        cls, spec: str | RerankerProvider | type[RerankerProvider], **kwargs: Any
    ) -> RerankerProvider:
        return cls._resolve(spec, _RERANKERS, RerankerProvider, "reranker", **kwargs)

    @classmethod
    def register(cls, category: str, name: str, backend_cls: type) -> None:
        """Register a custom backend shorthand at runtime.

        Args:
            category: One of ``"graph_store"``, ``"embedding"``, ``"llm"``, ``"job_store"``.
            name: The string shorthand (e.g. ``"milvus"``).
            backend_cls: The backend class to register.
        """
        registry = {
            "graph_store": _GRAPH_STORES,
            "embedding": _EMBEDDING_PROVIDERS,
            "llm": _LLM_PROVIDERS,
            "job_store": _JOB_STORES,
            "reranker": _RERANKERS,
        }.get(category)
        if registry is None:
            raise ValueError(
                f"Unknown category '{category}'. Use: graph_store, embedding, llm, job_store"
            )
        registry[name] = lambda _cls=backend_cls: _cls

    # -- internal ------------------------------------------------------------

    @staticmethod
    def _resolve(
        spec: Any,
        registry: dict[str, Any],
        abc: type,
        label: str,
        **kwargs: Any,
    ) -> Any:
        # Instance — pass through
        if isinstance(spec, abc):
            return spec

        # String shorthand — look up and instantiate
        if isinstance(spec, str):
            loader = registry.get(spec)
            if loader is None:
                available = list(registry.keys())
                raise ValueError(f"Unknown {label} '{spec}'. Available: {available}")
            cls = loader()
            return cls(**kwargs)

        # Class reference — instantiate
        if isinstance(spec, type) and issubclass(spec, abc):
            return spec(**kwargs)

        raise TypeError(
            f"{label} must be a string shorthand, a {abc.__name__} subclass, "
            f"or a {abc.__name__} instance — got {type(spec).__name__}"
        )
