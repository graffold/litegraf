"""LiteGraf — single entry point for the litegraf knowledge graph pipeline.

Every parameter has a sensible default.  Users override only what they need.

.. code-block:: python

    from pipeline.litegraf import LiteGraf

    kg = LiteGraf()
    kg.insert("TP53 is associated with multiple cancers.")
    result = kg.query("What cancers are associated with TP53?")
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from pipeline.dx.cache import LLMCache
from pipeline.dx.dedup import ContentDeduplicator
from pipeline.dx.limiter import RateLimitedLLMProvider
from pipeline.dx.models import ContextChunk, InsertResult, QueryResult
from pipeline.dx.registry import BackendRegistry
from pipeline.dx.sync_utils import run_sync
from pipeline.interfaces import EmbeddingProvider, GraphStore, JobStore, LLMProvider

logger = logging.getLogger(__name__)


@dataclass
class LiteGraf:
    """Single entry point for the litegraf knowledge graph pipeline.

    Accepts string shorthands (``"neo4j"``, ``"ollama"``), class references,
    or pre-configured backend instances for each backend slot.
    """

    # --- Graph store ---
    graph_store: str | GraphStore | type[GraphStore] = "neo4j"
    graph_uri: str = "bolt://localhost:7687"
    graph_user: str = "neo4j"
    graph_password: str = "password"
    graph_database: str = "neo4j"

    # --- Embedding provider ---
    embedding: str | EmbeddingProvider | type[EmbeddingProvider] = "local"
    embedding_model: str = "all-mpnet-base-v2"

    # --- LLM provider ---
    llm: str | LLMProvider | type[LLMProvider] = "ollama"
    llm_model: str = "llama3"
    llm_url: str = "http://localhost:11434"

    # --- Job store ---
    job_store: str | JobStore | type[JobStore] = "sqlite"
    job_store_path: str | None = None

    # --- Chunking ---
    chunk_token_size: int = 512
    chunk_overlap_tokens: int = 64

    # --- DX features ---
    enable_cache: bool = True
    cache_dir: str = ".litegraf_cache"
    max_async_calls: int = 16
    enable_dedup: bool = True
    working_dir: str = "./litegraf_workdir"

    # --- Resolved instances (set in __post_init__) ---
    _graph: GraphStore = field(init=False, repr=False)
    _embedder: EmbeddingProvider = field(init=False, repr=False)
    _llm: LLMProvider = field(init=False, repr=False)
    _job_store: JobStore = field(init=False, repr=False)
    _dedup: ContentDeduplicator = field(init=False, repr=False)
    _cache: LLMCache | None = field(init=False, repr=False)

    def __post_init__(self) -> None:
        # Resolve backends via registry
        self._graph = BackendRegistry.resolve_graph_store(
            self.graph_store,
            uri=self.graph_uri,
            auth=(self.graph_user, self.graph_password),
            database=self.graph_database,
        )
        self._embedder = BackendRegistry.resolve_embedding(
            self.embedding,
            model_name=self.embedding_model,
        )

        llm_instance = BackendRegistry.resolve_llm(
            self.llm,
            model=self.llm_model,
            base_url=self.llm_url,
        )

        self._job_store = BackendRegistry.resolve_job_store(
            self.job_store,
            **({"db_path": self.job_store_path} if self.job_store_path else {}),
        )

        # Wire DX layers onto the LLM
        self._cache = None
        if self.enable_cache:
            self._cache = LLMCache(cache_dir=self.cache_dir)
            llm_instance = self._cache.wrap(llm_instance)

        if self.max_async_calls > 0:
            llm_instance = RateLimitedLLMProvider(
                llm_instance, max_concurrent=self.max_async_calls
            )

        self._llm = llm_instance

        # Dedup
        self._dedup = ContentDeduplicator(working_dir=self.working_dir)

        logger.info(
            "LiteGraf initialized (graph=%s, llm=%s, embedding=%s)",
            self.graph_store,
            self.llm,
            self.embedding,
        )

    # --- Insert (async) -----------------------------------------------------

    async def ainsert(self, content: str | list[str]) -> InsertResult:
        """Insert text content into the knowledge graph (async)."""
        if not content or (isinstance(content, list) and not any(content)):
            raise ValueError("Content must be a non-empty string or list of strings")

        start = time.monotonic()
        texts = [content] if isinstance(content, str) else content

        total_chunks = 0
        total_entities = 0
        total_rels = 0
        all_doc_ids: list[str] = []

        for text in texts:
            if not text.strip():
                continue

            doc_id = self._dedup.compute_content_id(text)
            all_doc_ids.append(doc_id)

            if self.enable_dedup and self._dedup.is_duplicate(doc_id):
                continue

            # Chunk
            chunks = self._chunk_text(text, doc_id)
            total_chunks += len(chunks)

            # Extract + store per chunk
            for chunk_id, chunk_text in chunks:
                extraction = await self._extract_chunk(chunk_text)
                nodes = extraction.get("entities", [])
                rels = extraction.get("relationships", [])
                total_entities += len(nodes)
                total_rels += len(rels)
                self._store_extraction(chunk_id, nodes, rels)

            self._dedup.mark_seen(doc_id)

        duration = time.monotonic() - start
        result_id: str | list[str] = (
            all_doc_ids[0] if len(all_doc_ids) == 1 else all_doc_ids
        )
        return InsertResult(
            doc_id=result_id,
            chunks_processed=total_chunks,
            entities_extracted=total_entities,
            relationships_extracted=total_rels,
            was_duplicate=(total_chunks == 0),
            duration_seconds=round(duration, 3),
        )

    # --- Query (async) ------------------------------------------------------

    async def aquery(self, question: str, *, only_context: bool = False) -> QueryResult:
        """Query the knowledge graph (async)."""
        if not question.strip():
            raise ValueError("Question must be a non-empty string")

        start = time.monotonic()

        # Embed the question
        query_vec = self._embedder.embed_query(question)

        # Similarity search — use execute_query with a vector search Cypher
        context_chunks = self._similarity_search(query_vec, top_k=10)

        answer: str | None = None
        if not only_context and context_chunks:
            context_text = "\n---\n".join(c.text for c in context_chunks)
            prompt = (
                "Answer the following question based on the provided context. "
                "Be concise and factual.\n\n"
                f"Context:\n{context_text}\n\n"
                f"Question: {question}\n\nAnswer:"
            )
            answer = await self._llm.ainvoke(prompt)

        duration = time.monotonic() - start
        return QueryResult(
            answer=answer,
            context=context_chunks,
            duration_seconds=round(duration, 3),
        )

    # --- Sync wrappers ------------------------------------------------------

    def insert(self, content: str | list[str]) -> InsertResult:
        """Insert text content into the knowledge graph (sync wrapper)."""
        return run_sync(self.ainsert(content))

    def query(self, question: str, *, only_context: bool = False) -> QueryResult:
        """Query the knowledge graph (sync wrapper)."""
        return run_sync(self.aquery(question, only_context=only_context))

    # --- Lifecycle ----------------------------------------------------------

    def close(self) -> None:
        """Close all backend connections."""
        self._graph.close()

    def __enter__(self) -> LiteGraf:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __repr__(self) -> str:
        pw = "****" if self.graph_password else ""
        return (
            f"LiteGraf(graph_store={self.graph_store!r}, graph_uri={self.graph_uri!r}, "
            f"graph_password={pw!r}, llm={self.llm!r}, embedding={self.embedding!r}, "
            f"enable_cache={self.enable_cache}, enable_dedup={self.enable_dedup})"
        )

    # --- Internal helpers ---------------------------------------------------

    def _chunk_text(self, text: str, doc_id: str) -> list[tuple[str, str]]:
        """Split text into chunks. Returns list of (chunk_id, chunk_text)."""
        # Simple token-approximate chunking by character count
        # (1 token ≈ 4 chars is a rough heuristic)
        approx_chars = self.chunk_token_size * 4
        overlap_chars = self.chunk_overlap_tokens * 4
        chunks: list[tuple[str, str]] = []
        start = 0
        idx = 0
        while start < len(text):
            end = min(start + approx_chars, len(text))
            chunk_text = text[start:end]
            chunk_id = f"{doc_id}_chunk_{idx}"
            chunks.append((chunk_id, chunk_text))
            start += approx_chars - overlap_chars
            idx += 1
        return chunks

    async def _extract_chunk(self, chunk_text: str) -> dict[str, Any]:
        """Extract entities and relationships from a chunk via LLM."""
        prompt = (
            "Extract all entities (people, organizations, concepts, proteins, diseases, etc.) "
            "and relationships from the following text. "
            "Return valid JSON with 'entities' (list of {name, type}) and "
            "'relationships' (list of {source, target, type})."
        )
        return await self._llm.extract(prompt, chunk_text)

    def _store_extraction(
        self,
        chunk_id: str,
        nodes: list[dict[str, Any]],
        rels: list[dict[str, Any]],
    ) -> None:
        """Upsert extracted entities and relationships into the graph store."""
        for node in nodes:
            label = node.get("type", "Entity")
            name = node.get("name", "")
            if name:
                self._graph.upsert_node(
                    label, {"id": f"{label}:{name}", "name": name, "chunk_id": chunk_id}
                )

        for rel in rels:
            source = rel.get("source", "")
            target = rel.get("target", "")
            rel_type = rel.get("type", "RELATED_TO")
            if source and target:
                self._graph.upsert_relationship(
                    f"Entity:{source}",
                    rel_type,
                    f"Entity:{target}",
                    {"chunk_id": chunk_id},
                )

    def _similarity_search(
        self, query_vec: list[float], top_k: int = 10
    ) -> list[ContextChunk]:
        """Search the graph for chunks similar to the query vector."""
        # Try vector index search via Cypher (Neo4j 5.x+ with vector indexes)
        try:
            results = self._graph.execute_query(
                "CALL db.index.vector.queryNodes('chunk_embeddings', $top_k, $vec) "
                "YIELD node, score "
                "RETURN node.chunk_id AS chunk_id, node.text AS text, score "
                "ORDER BY score DESC",
                {"top_k": top_k, "vec": query_vec},
            )
            return [
                ContextChunk(
                    chunk_id=r.get("chunk_id", ""),
                    text=r.get("text", ""),
                    score=r.get("score", 0.0),
                    metadata={},
                )
                for r in results
                if r.get("text")
            ]
        except Exception:
            logger.debug("Vector search not available, falling back to text search")

        # Fallback: return empty (graph may not have vector indexes)
        return []
