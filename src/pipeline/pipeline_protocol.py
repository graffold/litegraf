"""Pipeline protocol and base class for multi-stage ingestion pipelines.

Defines PipelineProtocol (runtime-checkable structural typing interface)
and PipelineBase (abstract base with default stage implementations).
"""

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Protocol, runtime_checkable

from pipeline.entity_schema import EntitySchema
from pipeline.hooks import PostHook, PreHook
from pipeline.interfaces import EmbeddingProvider, GraphStore, LLMProvider
from pipeline.pipeline_config import PipelineConfig
from pipeline.pipeline_context import PipelineContext, StageResult
from pipeline.source_adapter import SourceAdapter

logger = logging.getLogger(__name__)

STAGE_ORDER = [
    "fetch",
    "extract",
    "deduplicate",
    "chunk",
    "extract_entities",
    "store_graph",
    "embed",
    "post_process",
]


@runtime_checkable
class PipelineProtocol(Protocol):
    """Formal protocol for multi-stage ingestion pipelines."""

    async def fetch(self, ctx: PipelineContext) -> StageResult: ...
    async def extract(self, ctx: PipelineContext) -> StageResult: ...
    async def deduplicate(self, ctx: PipelineContext) -> StageResult: ...
    async def chunk(self, ctx: PipelineContext) -> StageResult: ...
    async def extract_entities(self, ctx: PipelineContext) -> StageResult: ...
    async def store_graph(self, ctx: PipelineContext) -> StageResult: ...
    async def embed(self, ctx: PipelineContext) -> StageResult: ...
    async def post_process(self, ctx: PipelineContext) -> StageResult: ...
    async def run(self, ctx: PipelineContext) -> PipelineContext: ...
    async def close(self) -> None: ...


class PipelineBase:
    """Abstract base implementing PipelineProtocol with default stages.

    Subclasses must implement fetch, extract, and deduplicate.
    Default implementations are provided for chunk, extract_entities,
    store_graph, embed, and post_process.
    """

    STAGE_ORDER = STAGE_ORDER

    def __init__(
        self,
        graph_store: GraphStore,
        embedding_provider: EmbeddingProvider,
        llm_provider: LLMProvider,
        config: PipelineConfig,
        source_adapter: SourceAdapter,
        entity_schema: EntitySchema,
    ) -> None:
        self.graph_store = graph_store
        self.embedding_provider = embedding_provider
        self.llm_provider = llm_provider
        self.config = config
        self.source_adapter = source_adapter
        self.entity_schema = entity_schema
        self._pre_hooks: dict[str, list[PreHook]] = {
            s: [] for s in STAGE_ORDER
        }
        self._post_hooks: dict[str, list[PostHook]] = {
            s: [] for s in STAGE_ORDER
        }

    # --- Abstract methods (subclasses must implement) ---

    async def fetch(self, ctx: PipelineContext) -> StageResult:
        """Fetch raw records from the data source."""
        raise NotImplementedError(
            "Subclasses must implement fetch()"
        )

    async def extract(self, ctx: PipelineContext) -> StageResult:
        """Extract text content from raw records."""
        raise NotImplementedError(
            "Subclasses must implement extract()"
        )

    async def deduplicate(self, ctx: PipelineContext) -> StageResult:
        """Remove duplicate records."""
        raise NotImplementedError(
            "Subclasses must implement deduplicate()"
        )

    # --- Default implementations ---

    async def chunk(self, ctx: PipelineContext) -> StageResult:
        """Token-based chunking using config parameters.

        Operates on ctx.deduplicated_records (preferred) or
        ctx.extracted_text as fallback.
        """
        start = time.monotonic()
        items_processed = 0

        # Determine source texts
        source_items = ctx.deduplicated_records or ctx.extracted_text
        approx_chars = self.config.chunk_size * 4
        overlap_chars = self.config.chunk_overlap * 4
        chunks: list[dict[str, Any]] = []

        for i, item in enumerate(source_items):
            text = item.get("text", "") or item.get("content", "")
            if not text:
                continue
            text_start = 0
            chunk_idx = 0
            while text_start < len(text):
                end = min(text_start + approx_chars, len(text))
                chunk_text = text[text_start:end]
                chunks.append({
                    "chunk_id": f"item_{i}_chunk_{chunk_idx}",
                    "text": chunk_text,
                    "source_index": i,
                })
                text_start += approx_chars - overlap_chars
                chunk_idx += 1
            items_processed += 1

        ctx.chunks = chunks
        duration = time.monotonic() - start
        return StageResult(
            stage_name="chunk",
            success=True,
            duration_seconds=round(duration, 4),
            items_processed=items_processed,
        )

    async def extract_entities(self, ctx: PipelineContext) -> StageResult:
        """Extract entities using LLMProvider with EntitySchema prompt."""
        start = time.monotonic()
        items_processed = 0
        all_entities: list[dict[str, Any]] = []
        all_relationships: list[dict[str, Any]] = []

        for chunk in ctx.chunks:
            chunk_text = chunk.get("text", "")
            if not chunk_text:
                continue
            try:
                extraction = await self.llm_provider.extract(
                    self.entity_schema.extraction_prompt, chunk_text
                )
                entities = extraction.get("entities", [])
                relationships = extraction.get("relationships", [])
                all_entities.extend(entities)
                all_relationships.extend(relationships)
                items_processed += 1
            except Exception as exc:
                logger.warning(
                    "Entity extraction failed for chunk %s: %s",
                    chunk.get("chunk_id", "unknown"),
                    exc,
                )

        ctx.entities = all_entities
        ctx.relationships = all_relationships
        duration = time.monotonic() - start
        return StageResult(
            stage_name="extract_entities",
            success=True,
            duration_seconds=round(duration, 4),
            items_processed=items_processed,
        )

    async def store_graph(self, ctx: PipelineContext) -> StageResult:
        """Upsert entities and relationships via GraphStore."""
        start = time.monotonic()
        items_processed = 0

        for entity in ctx.entities:
            name = entity.get("name", "")
            label = entity.get("type", "Entity")
            if not name:
                continue
            props: dict[str, Any] = {
                "id": f"{label}:{name}",
                "name": name,
            }
            desc = entity.get("description", "")
            if desc:
                props["description"] = desc
            self.graph_store.upsert_node(label, props)
            items_processed += 1

        for rel in ctx.relationships:
            source = rel.get("source", "")
            target = rel.get("target", "")
            rel_type = rel.get("type", "RELATED_TO")
            if not (source and target):
                continue
            rel_props: dict[str, Any] = {}
            desc = rel.get("description", "")
            if desc:
                rel_props["description"] = desc
            self.graph_store.upsert_relationship(
                f"Entity:{source}", rel_type, f"Entity:{target}", rel_props
            )
            items_processed += 1

        duration = time.monotonic() - start
        return StageResult(
            stage_name="store_graph",
            success=True,
            duration_seconds=round(duration, 4),
            items_processed=items_processed,
        )

    async def embed(self, ctx: PipelineContext) -> StageResult:
        """Embed chunks using EmbeddingProvider."""
        start = time.monotonic()
        items_processed = 0

        texts = [c.get("text", "") for c in ctx.chunks if c.get("text")]
        if texts:
            vectors = self.embedding_provider.embed_documents(texts)
            embeddings: list[dict[str, Any]] = []
            for i, chunk in enumerate(ctx.chunks):
                if chunk.get("text") and i < len(vectors):
                    embeddings.append({
                        "chunk_id": chunk.get("chunk_id", f"chunk_{i}"),
                        "text": chunk["text"],
                        "embedding": vectors[i],
                    })
                    items_processed += 1
            ctx.embeddings = embeddings

        duration = time.monotonic() - start
        return StageResult(
            stage_name="embed",
            success=True,
            duration_seconds=round(duration, 4),
            items_processed=items_processed,
        )

    async def post_process(self, ctx: PipelineContext) -> StageResult:
        """No-op post-processing stage. Override for custom logic."""
        return StageResult(
            stage_name="post_process",
            success=True,
            duration_seconds=0.0,
            items_processed=0,
        )

    # --- Orchestration ---

    async def run(
        self,
        ctx: PipelineContext,
        *,
        progress_callback: (
            Callable[[int, int, str], Awaitable[None]] | None
        ) = None,
        cancellation_event: asyncio.Event | None = None,
    ) -> PipelineContext:
        """Execute all pipeline stages in order.

        For each stage: check cancellation → run pre-hooks → execute stage
        → run post-hooks → record result. Stops on cancellation or failure.
        """
        total_stages = len(self.STAGE_ORDER)
        cumulative_items = 0

        for stage_idx, stage_name in enumerate(self.STAGE_ORDER):
            # Check cancellation before each stage
            if cancellation_event and cancellation_event.is_set():
                logger.info(
                    "Pipeline cancelled before stage '%s'", stage_name
                )
                break

            ctx.current_stage = stage_name
            ctx.stage_timestamps[stage_name] = datetime.now(timezone.utc)

            # Run pre-hooks
            for hook in self._pre_hooks.get(stage_name, []):
                try:
                    await hook(ctx)
                except Exception as exc:
                    logger.warning(
                        "Pre-hook for stage '%s' raised: %s",
                        stage_name,
                        exc,
                    )

            # Execute stage
            stage_method = getattr(self, stage_name)
            try:
                result = await stage_method(ctx)
            except Exception as exc:
                result = StageResult(
                    stage_name=stage_name,
                    success=False,
                    duration_seconds=0.0,
                    error=str(exc),
                )
                ctx.stage_results.append(result)
                logger.error(
                    "Stage '%s' failed: %s", stage_name, exc
                )
                break

            ctx.stage_results.append(result)
            cumulative_items += result.items_processed

            # Run post-hooks
            for hook in self._post_hooks.get(stage_name, []):
                try:
                    await hook(ctx, result)
                except Exception as exc:
                    logger.warning(
                        "Post-hook for stage '%s' raised: %s",
                        stage_name,
                        exc,
                    )

            # Progress callback at stage boundary
            if progress_callback:
                try:
                    await progress_callback(
                        cumulative_items, total_stages, stage_name
                    )
                except Exception as exc:
                    logger.debug(
                        "Progress callback error: %s", exc
                    )

            # Check cancellation after stage (between stages)
            if cancellation_event and cancellation_event.is_set():
                logger.info(
                    "Pipeline cancelled after stage '%s'",
                    stage_name,
                )
                break

        return ctx

    # --- Hook registration ---

    def add_pre_hook(
        self, stage_name: str, callback: PreHook
    ) -> None:
        """Register a pre-stage hook for the given stage."""
        if stage_name not in self._pre_hooks:
            self._pre_hooks[stage_name] = []
        self._pre_hooks[stage_name].append(callback)

    def add_post_hook(
        self, stage_name: str, callback: PostHook
    ) -> None:
        """Register a post-stage hook for the given stage."""
        if stage_name not in self._post_hooks:
            self._post_hooks[stage_name] = []
        self._post_hooks[stage_name].append(callback)

    # --- Lifecycle ---

    async def close(self) -> None:
        """Clean up resources."""
        try:
            self.graph_store.close()
        except Exception as exc:
            logger.debug("Error closing graph store: %s", exc)
