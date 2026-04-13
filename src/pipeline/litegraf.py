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
from typing import Any, Protocol

import tiktoken

from pipeline.dx.cache import LLMCache
from pipeline.dx.dedup import ContentDeduplicator
from pipeline.dx.limiter import RateLimitedLLMProvider
from pipeline.dx.models import QUERY_MODES, ContextChunk, DeleteResult, InsertResult, QueryMode, QueryResult
from pipeline.dx.registry import BackendRegistry
from pipeline.dx.sync_utils import run_sync
from pipeline.interfaces import EmbeddingProvider, GraphStore, JobStore, LLMProvider, RerankerProvider


class TokenCounter(Protocol):
    """Protocol for token counting callables: ``(text: str) -> int``."""

    def __call__(self, text: str) -> int: ...


def _default_token_counter() -> TokenCounter:
    """Return a tiktoken-based token counter (cl100k_base encoding)."""
    enc = tiktoken.get_encoding("cl100k_base")
    return lambda text: len(enc.encode(text))

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

    # --- Extraction ---
    max_gleaning: int = 1
    enable_entity_merge: bool = True

    # --- Reranker ---
    reranker: str | RerankerProvider | type[RerankerProvider] | None = None

    # --- Token budget ---
    max_context_tokens: int = 8000
    tokenizer: TokenCounter | None = None

    # --- Resolved instances (set in __post_init__) ---
    _graph: GraphStore = field(init=False, repr=False)
    _embedder: EmbeddingProvider = field(init=False, repr=False)
    _llm: LLMProvider = field(init=False, repr=False)
    _job_store: JobStore = field(init=False, repr=False)
    _dedup: ContentDeduplicator = field(init=False, repr=False)
    _cache: LLMCache | None = field(init=False, repr=False)
    _reranker: RerankerProvider | None = field(init=False, repr=False)
    _token_counter: TokenCounter = field(init=False, repr=False)

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

        # Reranker (optional)
        self._reranker = (
            BackendRegistry.resolve_reranker(self.reranker)
            if self.reranker is not None
            else None
        )

        # Token counter
        self._token_counter = self.tokenizer or _default_token_counter()

        # Ensure vector indexes exist
        self._ensure_entity_vector_index()
        self._ensure_relationship_vector_index()

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
                await self._store_extraction(chunk_id, chunk_text, nodes, rels)

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

    async def aquery(self, question: str, *, only_context: bool = False, mode: QueryMode = "hybrid") -> QueryResult:
        """Query the knowledge graph (async)."""
        if not question.strip():
            raise ValueError("Question must be a non-empty string")
        if mode not in QUERY_MODES:
            raise ValueError(f"Unknown query mode {mode!r}, expected one of {sorted(QUERY_MODES)}")

        start = time.monotonic()

        # Embed the question
        query_vec = self._embedder.embed_query(question)

        # Similarity search — use execute_query with a vector search Cypher
        context_chunks = self._similarity_search(query_vec, top_k=10, mode=mode)

        # Rerank if a reranker is configured
        if self._reranker and context_chunks:
            candidates = [
                {"text": c.text, "chunk_id": c.chunk_id, "score": c.score, "metadata": c.metadata}
                for c in context_chunks
            ]
            reranked = self._reranker.rerank(question, candidates)
            context_chunks = [
                ContextChunk(
                    chunk_id=r["chunk_id"],
                    text=r["text"],
                    score=r["score"],
                    metadata=r.get("metadata", {}),
                )
                for r in reranked
            ]

        # Apply token budget
        context_chunks = self._apply_token_budget(context_chunks)

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

    # --- Delete (async) -----------------------------------------------------

    async def adelete(self, doc_id: str) -> DeleteResult:
        """Delete a document and clean up its entities/relationships from the KG (async).

        Chunks belonging to *doc_id* are removed.  Entities and relationships
        that are unique to those chunks are deleted.  Shared entities (referenced
        by other documents) are kept.  The dedup index is updated so the same
        content can be re-inserted.
        """
        chunk_prefix = f"{doc_id}_chunk_"

        # 1. Delete relationships unique to this document
        #    A relationship is unique if its chunk_id belongs to this doc.
        rel_result = self._graph.execute_query(
            "MATCH ()-[r]->() WHERE r.chunk_id STARTS WITH $prefix "
            "DELETE r RETURN count(r) AS cnt",
            {"prefix": chunk_prefix},
        )
        rels_removed = rel_result[0]["cnt"] if rel_result else 0

        # 2. Delete entity nodes whose chunk_id belongs to this doc AND
        #    that have no remaining relationships from other documents.
        ent_result = self._graph.execute_query(
            "MATCH (n:Entity) WHERE n.chunk_id STARTS WITH $prefix "
            "AND NOT EXISTS { MATCH (n)-[r]-() WHERE NOT r.chunk_id STARTS WITH $prefix } "
            "AND NOT EXISTS { MATCH (n)-[r]-() WHERE r.chunk_id IS NULL } "
            "DELETE n RETURN count(n) AS cnt",
            {"prefix": chunk_prefix},
        )
        ents_removed = ent_result[0]["cnt"] if ent_result else 0

        # 3. Delete chunk nodes
        chunk_del = self._graph.execute_query(
            "MATCH (c:Chunk) WHERE c.chunk_id STARTS WITH $prefix "
            "DETACH DELETE c RETURN count(c) AS cnt",
            {"prefix": chunk_prefix},
        )
        chunks_removed = chunk_del[0]["cnt"] if chunk_del else 0

        # 4. Update dedup index so the same content can be re-inserted
        self._dedup.remove_seen(doc_id)

        return DeleteResult(
            doc_id=doc_id,
            chunks_removed=chunks_removed,
            entities_removed=ents_removed,
            relationships_removed=rels_removed,
        )

    # --- Sync wrappers ------------------------------------------------------

    def insert(self, content: str | list[str]) -> InsertResult:
        """Insert text content into the knowledge graph (sync wrapper)."""
        return run_sync(self.ainsert(content))

    def delete(self, doc_id: str) -> DeleteResult:
        """Delete a document and clean up its KG data (sync wrapper)."""
        return run_sync(self.adelete(doc_id))

    def query(self, question: str, *, only_context: bool = False, mode: QueryMode = "hybrid") -> QueryResult:
        """Query the knowledge graph (sync wrapper)."""
        return run_sync(self.aquery(question, only_context=only_context, mode=mode))

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

    def _ensure_entity_vector_index(self) -> None:
        """Create the entity_embeddings vector index if it doesn't exist."""
        try:
            self._graph.execute_query(
                "CREATE VECTOR INDEX entity_embeddings IF NOT EXISTS "
                "FOR (n:Entity) ON (n.embedding) "
                "OPTIONS {indexConfig: {`vector.dimensions`: 768, `vector.similarity_function`: 'cosine'}}"
            )
        except Exception:
            logger.debug("Could not create entity_embeddings index (may already exist or DB unavailable)")

    def _ensure_relationship_vector_index(self) -> None:
        """Create the relationship_embeddings vector index if it doesn't exist."""
        try:
            self._graph.execute_query(
                "CREATE VECTOR INDEX relationship_embeddings IF NOT EXISTS "
                "FOR ()-[r:RELATED_TO]-() ON (r.embedding) "
                "OPTIONS {indexConfig: {`vector.dimensions`: 768, `vector.similarity_function`: 'cosine'}}"
            )
        except Exception:
            logger.debug("Could not create relationship_embeddings index (may already exist or DB unavailable)")

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
        """Extract entities and relationships from a chunk via LLM.

        When max_gleaning > 1, performs additional passes asking the LLM
        for missed entities/relationships and merges results.
        """
        prompt = (
            "Extract all entities (people, organizations, concepts, proteins, diseases, etc.) "
            "and relationships from the following text. "
            "Return valid JSON with 'entities' (list of {name, type, description}) and "
            "'relationships' (list of {source, target, type, description}). "
            "Each 'description' should be a concise summary of the entity or relationship."
        )
        result = await self._llm.extract(prompt, chunk_text)

        if self.max_gleaning <= 1:
            return result

        all_entities: list[dict[str, Any]] = list(result.get("entities", []))
        all_rels: list[dict[str, Any]] = list(result.get("relationships", []))
        seen_names: set[str] = {e.get("name", "").strip().lower() for e in all_entities}
        seen_rels: set[tuple[str, str, str]] = {
            (r.get("source", "").lower(), r.get("target", "").lower(), r.get("type", "").lower())
            for r in all_rels
        }

        for _ in range(self.max_gleaning - 1):
            entity_names = ", ".join(e.get("name", "") for e in all_entities) or "(none)"
            gleaning_prompt = (
                "The following entities have already been extracted from the text below:\n"
                f"Already found: {entity_names}\n\n"
                "Did you miss any entities or relationships? "
                "Re-read the text carefully and return ONLY new ones not listed above.\n"
                "Return valid JSON with 'entities' (list of {name, type, description}) and "
                "'relationships' (list of {source, target, type, description}).\n"
                "If nothing was missed, return: {\"entities\": [], \"relationships\": []}\n\n"
                f"Text: {chunk_text}"
            )
            try:
                extra = await self._llm.extract(gleaning_prompt, "")
            except Exception:
                logger.debug("Gleaning pass failed, stopping early")
                break

            new_entities = [
                e for e in extra.get("entities", [])
                if e.get("name", "").strip().lower() not in seen_names
            ]
            new_rels = [
                r for r in extra.get("relationships", [])
                if (r.get("source", "").lower(), r.get("target", "").lower(), r.get("type", "").lower()) not in seen_rels
            ]

            if not new_entities and not new_rels:
                break

            for e in new_entities:
                seen_names.add(e.get("name", "").strip().lower())
            for r in new_rels:
                seen_rels.add((r.get("source", "").lower(), r.get("target", "").lower(), r.get("type", "").lower()))
            all_entities.extend(new_entities)
            all_rels.extend(new_rels)

        return {"entities": all_entities, "relationships": all_rels}

    async def _store_extraction(
        self,
        chunk_id: str,
        chunk_text: str,
        nodes: list[dict[str, Any]],
        rels: list[dict[str, Any]],
    ) -> None:
        """Upsert extracted entities and relationships into the graph store."""
        # --- Store chunk node ---
        chunk_embedding = self._embedder.embed_query(chunk_text)
        self._graph.upsert_node("Chunk", {
            "id": chunk_id,
            "chunk_id": chunk_id,
            "text": chunk_text,
            "embedding": chunk_embedding,
        })
        # --- Entity merge: fetch existing descriptions and merge via LLM ---
        if self.enable_entity_merge:
            for node in nodes:
                name = node.get("name", "")
                label = node.get("type", "Entity")
                new_desc = node.get("description", "")
                if not (name and new_desc):
                    continue
                node_id = f"{label}:{name}"
                try:
                    existing = self._graph.execute_query(
                        "MATCH (n {id: $id}) RETURN n.description AS description",
                        {"id": node_id},
                    )
                except Exception:
                    continue
                old_desc = (existing[0].get("description") or "") if existing else ""
                if old_desc and old_desc != new_desc:
                    try:
                        merged = await self._llm.ainvoke(
                            "Merge these two descriptions of the same entity into one concise summary.\n\n"
                            f"Description 1: {old_desc}\n\nDescription 2: {new_desc}\n\nMerged description:"
                        )
                        if merged and merged.strip():
                            node["description"] = merged.strip()
                    except Exception:
                        logger.debug("Entity merge LLM call failed for %s", node_id)

        # Embed entity descriptions in batch
        descriptions = []
        entity_indices: list[int] = []
        for i, node in enumerate(nodes):
            desc = node.get("description", "")
            if node.get("name") and desc:
                descriptions.append(desc)
                entity_indices.append(i)

        embeddings: list[list[float]] = []
        if descriptions:
            embeddings = self._embedder.embed_documents(descriptions)

        emb_map: dict[int, list[float]] = dict(zip(entity_indices, embeddings))

        for i, node in enumerate(nodes):
            label = node.get("type", "Entity")
            name = node.get("name", "")
            if name:
                props: dict[str, Any] = {
                    "id": f"{label}:{name}",
                    "name": name,
                    "chunk_id": chunk_id,
                }
                desc = node.get("description", "")
                if desc:
                    props["description"] = desc
                if i in emb_map:
                    props["embedding"] = emb_map[i]
                self._graph.upsert_node(label, props)

        # Embed relationship descriptions in batch
        rel_descriptions = []
        rel_indices: list[int] = []
        for i, rel in enumerate(rels):
            desc = rel.get("description", "")
            if rel.get("source") and rel.get("target") and desc:
                rel_descriptions.append(desc)
                rel_indices.append(i)

        rel_embeddings: list[list[float]] = []
        if rel_descriptions:
            rel_embeddings = self._embedder.embed_documents(rel_descriptions)

        rel_emb_map: dict[int, list[float]] = dict(zip(rel_indices, rel_embeddings))

        for i, rel in enumerate(rels):
            source = rel.get("source", "")
            target = rel.get("target", "")
            rel_type = rel.get("type", "RELATED_TO")
            if source and target:
                rel_props: dict[str, Any] = {"chunk_id": chunk_id}
                desc = rel.get("description", "")
                if desc:
                    rel_props["description"] = desc
                if i in rel_emb_map:
                    rel_props["embedding"] = rel_emb_map[i]
                self._graph.upsert_relationship(
                    f"Entity:{source}",
                    rel_type,
                    f"Entity:{target}",
                    rel_props,
                )

    def _apply_token_budget(self, chunks: list[ContextChunk]) -> list[ContextChunk]:
        """Truncate context chunks to fit within *max_context_tokens*.

        Chunks are already sorted by score (descending).  We partition them
        into three buckets — entity, relationship, and plain chunk — and give
        each bucket a proportional share of the budget.  Within each bucket
        the highest-scored items are kept first.
        """
        if not chunks:
            return chunks

        budget = self.max_context_tokens
        count = self._token_counter

        # Partition by type
        buckets: dict[str, list[ContextChunk]] = {"entity": [], "relationship": [], "chunk": []}
        for c in chunks:
            kind = c.metadata.get("type", "chunk")
            buckets.setdefault(kind, buckets["chunk"]).append(c)

        # Non-empty bucket count determines proportional share
        active = {k: v for k, v in buckets.items() if v}
        if not active:
            return []

        share = budget // len(active)
        remainder = budget - share * len(active)

        result: list[ContextChunk] = []
        for i, (_, items) in enumerate(active.items()):
            bucket_budget = share + (remainder if i == 0 else 0)
            used = 0
            for c in items:
                tokens = count(c.text)
                if used + tokens > bucket_budget:
                    break
                used += tokens
                result.append(c)

        # Re-sort by score descending
        result.sort(key=lambda c: c.score, reverse=True)
        return result

    def _similarity_search(
        self, query_vec: list[float], top_k: int = 10, *, mode: QueryMode = "hybrid"
    ) -> list[ContextChunk]:
        """Search the graph for chunks similar to the query vector."""
        chunks: list[ContextChunk] = []

        # Chunk-based search (used by naive, mix)
        if mode in ("naive", "mix"):
            chunks.extend(self._search_chunk_index(query_vec, top_k))

        # Entity-based search (used by local, hybrid, mix)
        if mode in ("local", "hybrid", "mix"):
            chunks.extend(self._search_entity_index(query_vec, top_k))

        # Relationship-based search (used by global, hybrid, mix)
        if mode in ("global", "hybrid", "mix"):
            chunks.extend(self._search_relationship_index(query_vec, top_k))

        if not chunks:
            # Fallback: try chunk search for any mode
            chunks = self._search_chunk_index(query_vec, top_k)

        # Deduplicate by chunk_id, keep highest score
        seen: dict[str, ContextChunk] = {}
        for c in chunks:
            key = c.chunk_id or c.text[:100]
            if key not in seen or c.score > seen[key].score:
                seen[key] = c
        result = sorted(seen.values(), key=lambda c: c.score, reverse=True)
        return result[:top_k]

    def _search_chunk_index(
        self, query_vec: list[float], top_k: int
    ) -> list[ContextChunk]:
        """Search the chunk_embeddings vector index."""
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
            logger.debug("chunk_embeddings index not available")
            return []

    def _search_entity_index(
        self, query_vec: list[float], top_k: int
    ) -> list[ContextChunk]:
        """Search the entity_embeddings vector index."""
        try:
            results = self._graph.execute_query(
                "CALL db.index.vector.queryNodes('entity_embeddings', $top_k, $vec) "
                "YIELD node, score "
                "RETURN node.id AS entity_id, node.name AS name, "
                "node.description AS description, score "
                "ORDER BY score DESC",
                {"top_k": top_k, "vec": query_vec},
            )
            return [
                ContextChunk(
                    chunk_id=r.get("entity_id", ""),
                    text=r.get("description", r.get("name", "")),
                    score=r.get("score", 0.0),
                    metadata={"type": "entity", "name": r.get("name", "")},
                )
                for r in results
                if r.get("description") or r.get("name")
            ]
        except Exception:
            logger.debug("entity_embeddings index not available")
            return []

    def _search_relationship_index(
        self, query_vec: list[float], top_k: int
    ) -> list[ContextChunk]:
        """Search the relationship_embeddings vector index."""
        try:
            results = self._graph.execute_query(
                "CALL db.index.vector.queryRelationships('relationship_embeddings', $top_k, $vec) "
                "YIELD relationship, score "
                "RETURN relationship.description AS description, "
                "relationship.chunk_id AS chunk_id, type(relationship) AS rel_type, score "
                "ORDER BY score DESC",
                {"top_k": top_k, "vec": query_vec},
            )
            return [
                ContextChunk(
                    chunk_id=r.get("chunk_id", ""),
                    text=r.get("description", ""),
                    score=r.get("score", 0.0),
                    metadata={"type": "relationship", "rel_type": r.get("rel_type", "")},
                )
                for r in results
                if r.get("description")
            ]
        except Exception:
            logger.debug("relationship_embeddings index not available")
            return []
