"""Integration tests: pipeline construction with mock/default backends.

Verifies that pipeline modules accept injected backends and expose expected
methods without requiring real external services (Neo4j, Ollama, etc.).

Validates: Requirements 15.1, 15.2, 15.4
"""

from __future__ import annotations

import logging
from typing import Any

import pytest

from pipeline.interfaces import (
    EmbeddingProvider,
    GraphStore,
    JobStore,
    LLMProvider,
)

# ---------------------------------------------------------------------------
# Concrete mock implementations of all four interfaces
# ---------------------------------------------------------------------------


class MockGraphStore(GraphStore):
    """In-memory GraphStore for integration testing."""

    def __init__(self) -> None:
        self.nodes: list[dict[str, Any]] = []
        self.relationships: list[dict[str, Any]] = []
        self.queries: list[str] = []

    def execute_query(
        self, query: str, params: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        self.queries.append(query)
        return []

    def upsert_node(self, label: str, properties: dict[str, Any]) -> str:
        node_id = properties.get("id", f"{label}_{len(self.nodes)}")
        self.nodes.append({"label": label, "properties": properties})
        return str(node_id)

    def upsert_relationship(
        self,
        source_id: str,
        rel_type: str,
        target_id: str,
        properties: dict[str, Any] | None = None,
    ) -> None:
        self.relationships.append(
            {
                "source": source_id,
                "type": rel_type,
                "target": target_id,
                "properties": properties or {},
            }
        )

    def close(self) -> None:
        pass


class MockEmbeddingProvider(EmbeddingProvider):
    """Deterministic EmbeddingProvider for integration testing."""

    def embed_query(self, text: str) -> list[float]:
        return [0.1] * 768

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * 768 for _ in texts]


class MockLLMProvider(LLMProvider):
    """Stub LLMProvider for integration testing."""

    def invoke(self, prompt: str, **kwargs: Any) -> str:
        return '{"entities": [], "relationships": []}'

    async def ainvoke(self, prompt: str, **kwargs: Any) -> str:
        return '{"entities": [], "relationships": []}'

    async def extract(self, prompt: str, text: str) -> dict[str, Any]:
        return {"entities": [], "relationships": []}


class MockJobStore(JobStore):
    """In-memory JobStore for integration testing."""

    def __init__(self) -> None:
        self._store: dict[str, dict[str, Any]] = {}

    async def save(self, job_id: str, metadata: dict[str, Any]) -> None:
        self._store[job_id] = dict(metadata)

    async def load(self, job_id: str) -> dict[str, Any] | None:
        data = self._store.get(job_id)
        return dict(data) if data else None

    async def delete(self, job_id: str) -> None:
        self._store.pop(job_id, None)

    async def list_jobs(self) -> list[dict[str, Any]]:
        return [dict(v) for v in self._store.values()]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_graph_store() -> MockGraphStore:
    return MockGraphStore()


@pytest.fixture
def mock_embedding_provider() -> MockEmbeddingProvider:
    return MockEmbeddingProvider()


@pytest.fixture
def mock_llm_provider() -> MockLLMProvider:
    return MockLLMProvider()


@pytest.fixture
def mock_job_store() -> MockJobStore:
    return MockJobStore()


# ---------------------------------------------------------------------------
# 12.1 — Pipeline construction with mock backends
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestKGPipelineConstruction:
    """KGPipeline accepts injected mock backends."""

    def test_construction_succeeds(
        self,
        mock_graph_store: MockGraphStore,
        mock_embedding_provider: MockEmbeddingProvider,
        mock_llm_provider: MockLLMProvider,
    ) -> None:
        from pipeline.ingest.kg_pipeline import KGPipeline

        pipeline = KGPipeline(
            graph_store=mock_graph_store,
            embedding_provider=mock_embedding_provider,
            llm_provider=mock_llm_provider,
        )
        assert pipeline.db is mock_graph_store
        assert pipeline.embedder is mock_embedding_provider
        assert pipeline.llm is mock_llm_provider

    def test_has_run_method(
        self,
        mock_graph_store: MockGraphStore,
        mock_embedding_provider: MockEmbeddingProvider,
        mock_llm_provider: MockLLMProvider,
    ) -> None:
        from pipeline.ingest.kg_pipeline import KGPipeline

        pipeline = KGPipeline(
            graph_store=mock_graph_store,
            embedding_provider=mock_embedding_provider,
            llm_provider=mock_llm_provider,
        )
        assert callable(getattr(pipeline, "run", None))


@pytest.mark.integration
class TestEmbeddingPipelineConstruction:
    """EmbeddingPipeline accepts injected mock backends."""

    def test_construction_succeeds(
        self,
        mock_graph_store: MockGraphStore,
        mock_embedding_provider: MockEmbeddingProvider,
        mock_llm_provider: MockLLMProvider,
    ) -> None:
        from pipeline.ingest.embedding_pipeline import EmbeddingPipeline

        pipeline = EmbeddingPipeline(
            graph_store=mock_graph_store,
            embedding_provider=mock_embedding_provider,
            llm_provider=mock_llm_provider,
            database="test",
        )
        assert pipeline.db is mock_graph_store
        assert pipeline.embedder is mock_embedding_provider
        assert pipeline.llm is mock_llm_provider

    def test_has_expected_methods(
        self,
        mock_graph_store: MockGraphStore,
        mock_embedding_provider: MockEmbeddingProvider,
        mock_llm_provider: MockLLMProvider,
    ) -> None:
        from pipeline.ingest.embedding_pipeline import EmbeddingPipeline

        pipeline = EmbeddingPipeline(
            graph_store=mock_graph_store,
            embedding_provider=mock_embedding_provider,
            llm_provider=mock_llm_provider,
            database="test",
        )
        # EmbeddingPipeline should have embedding-related methods
        assert hasattr(pipeline, "db")
        assert hasattr(pipeline, "embedder")


@pytest.mark.integration
class TestEnrichmentOrchestratorConstruction:
    """EnrichmentOrchestrator accepts injected mock backends."""

    def test_construction_succeeds(
        self,
        mock_graph_store: MockGraphStore,
        mock_llm_provider: MockLLMProvider,
    ) -> None:
        from pipeline.enrichment.enrichment_orchestrator import (
            EnrichmentOrchestrator,
        )

        orchestrator = EnrichmentOrchestrator(
            graph_store=mock_graph_store,
            llm_provider=mock_llm_provider,
        )
        assert orchestrator.db is mock_graph_store
        assert orchestrator.llm is mock_llm_provider

    def test_has_enrich_csv_method(
        self,
        mock_graph_store: MockGraphStore,
        mock_llm_provider: MockLLMProvider,
    ) -> None:
        from pipeline.enrichment.enrichment_orchestrator import (
            EnrichmentOrchestrator,
        )

        orchestrator = EnrichmentOrchestrator(
            graph_store=mock_graph_store,
            llm_provider=mock_llm_provider,
        )
        assert callable(getattr(orchestrator, "enrich_csv", None))


@pytest.mark.integration
class TestIngestionJobManagerConstruction:
    """IngestionJobManager accepts injected mock JobStore."""

    def test_construction_succeeds(self, mock_job_store: MockJobStore) -> None:
        from src.services.ingestion.ingestion_job_manager import (
            IngestionJobManager,
        )

        manager = IngestionJobManager(
            job_store=mock_job_store,
            logger=logging.getLogger("test"),
        )
        assert manager.job_store is mock_job_store

    def test_has_expected_attributes(self, mock_job_store: MockJobStore) -> None:
        from src.services.ingestion.ingestion_job_manager import (
            IngestionJobManager,
        )

        manager = IngestionJobManager(
            job_store=mock_job_store,
            logger=logging.getLogger("test"),
        )
        assert hasattr(manager, "active_tasks")
        assert hasattr(manager, "cancellation_events")


@pytest.mark.integration
class TestSQLiteJobStoreAsBackend:
    """SQLiteJobStore works as a real JobStore backend in pipeline modules."""

    @pytest.mark.asyncio
    async def test_sqlite_job_store_round_trip(self, tmp_path) -> None:
        from pipeline.backends.sqlite_job_store import SQLiteJobStore

        store = SQLiteJobStore(db_path=str(tmp_path / "test_jobs.db"))
        try:
            await store.save("job-1", {"status": "running", "progress": 50})
            loaded = await store.load("job-1")
            assert loaded == {"status": "running", "progress": 50}
        finally:
            await store.close()

    @pytest.mark.asyncio
    async def test_sqlite_job_store_in_job_manager(self, tmp_path) -> None:
        from pipeline.backends.sqlite_job_store import SQLiteJobStore
        from src.services.ingestion.ingestion_job_manager import (
            IngestionJobManager,
        )

        store = SQLiteJobStore(db_path=str(tmp_path / "test_jobs.db"))
        try:
            manager = IngestionJobManager(
                job_store=store,
                logger=logging.getLogger("test"),
            )
            assert manager.job_store is store
        finally:
            await store.close()


@pytest.mark.integration
class TestMockGraphStoreContextManager:
    """GraphStore context manager protocol works with mock."""

    def test_context_manager(self, mock_graph_store: MockGraphStore) -> None:
        with mock_graph_store as gs:
            assert gs is mock_graph_store
            gs.execute_query("RETURN 1")
        assert "RETURN 1" in mock_graph_store.queries
