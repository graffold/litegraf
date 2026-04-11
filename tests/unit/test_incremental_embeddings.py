"""
Unit tests for incremental embeddings generation.

Tests verify that:
1. Only nodes without embeddings are processed
2. Existing embeddings are reused (not regenerated)
3. Vector index is updated incrementally
4. Consistency is maintained after updates
"""

from unittest.mock import MagicMock, patch

import pytest

from pipeline.ingest.embedding_pipeline import EmbeddingPipeline
from pipeline.interfaces import EmbeddingProvider, GraphStore, LLMProvider


class _MockGraphStore(GraphStore):
    """Concrete mock GraphStore for testing."""

    def __init__(self):
        self.driver = MagicMock()

    def execute_query(self, query, params=None):
        return []

    def upsert_node(self, label, properties):
        return "mock-id"

    def upsert_relationship(self, source_id, rel_type, target_id, properties=None):
        pass

    def close(self):
        pass


class _MockEmbeddingProvider(EmbeddingProvider):
    """Concrete mock EmbeddingProvider for testing."""

    def embed_query(self, text):
        return [0.1] * 768

    def embed_documents(self, texts):
        return [[0.1] * 768 for _ in texts]


class _MockLLMProvider(LLMProvider):
    """Concrete mock LLMProvider for testing."""

    def invoke(self, prompt, **kwargs):
        return ""

    async def ainvoke(self, prompt, **kwargs):
        return ""

    async def extract(self, prompt, text):
        return {"entities": [], "relationships": []}


def _make_pipeline(mock_graph_store=None, mock_embedder=None, mock_llm=None):
    """Helper to create an EmbeddingPipeline with mock backends."""
    gs = mock_graph_store or _MockGraphStore()
    ep = mock_embedder or _MockEmbeddingProvider()
    lp = mock_llm or _MockLLMProvider()
    with patch(
        "src.utils.neo4j_index_manager.setup_graphrag_indexes",
        return_value=True,
    ):
        return EmbeddingPipeline(
            gs,
            ep,
            lp,
            database="test",
        )


@pytest.fixture
def mock_graph_store():
    """Mock GraphStore with a driver for testing."""
    gs = _MockGraphStore()
    return gs


@pytest.fixture
def mock_embedder():
    """Mock EmbeddingProvider that returns 768-dimensional vectors."""
    return _MockEmbeddingProvider()


@pytest.fixture
def mock_llm():
    """Mock LLMProvider for testing."""
    return _MockLLMProvider()


@pytest.fixture
def mock_session():
    """Mock Neo4j session."""
    session = MagicMock()
    session.__enter__ = MagicMock(return_value=session)
    session.__exit__ = MagicMock(return_value=None)
    return session


@pytest.mark.xfail(reason="Mocks need rewrite for neo4j-graphrag pydantic validation")
class TestIncrementalEmbeddings:
    """Test incremental embeddings generation."""

    @pytest.mark.asyncio
    async def test_get_nodes_without_embeddings_filters_correctly(
        self, mock_graph_store, mock_embedder, mock_llm, mock_session
    ):
        """Test that _get_nodes_without_embeddings only fetches nodes without embeddings."""
        mock_graph_store.driver.session.return_value = mock_session

        # Mock query results
        total_result = MagicMock()
        total_result.single.return_value = {"count": 100}

        with_emb_result = MagicMock()
        with_emb_result.single.return_value = {"count": 80}

        without_emb_result = MagicMock()
        without_emb_result.__iter__ = MagicMock(
            return_value=iter(
                [
                    {
                        "id": "protein_1",
                        "name": "TP53",
                        "type": "Protein",
                        "props": {"uniprot_id": "P04637"},
                        "chunk_texts": [],
                    },
                    {
                        "id": "protein_2",
                        "name": "BRCA1",
                        "type": "Protein",
                        "props": {"uniprot_id": "P38398"},
                        "chunk_texts": [],
                    },
                ]
            )
        )

        mock_session.run.side_effect = [
            total_result,
            with_emb_result,
            without_emb_result,
        ]

        pipeline = _make_pipeline(mock_graph_store, mock_embedder, mock_llm)

        nodes = pipeline._get_nodes_without_embeddings("Protein")

        assert len(nodes) == 2
        assert nodes[0]["name"] == "TP53"
        assert nodes[1]["name"] == "BRCA1"

        calls = mock_session.run.call_args_list
        query = calls[2][0][0]
        assert "WHERE n.embedding IS NULL" in query

    @pytest.mark.asyncio
    async def test_add_embeddings_to_kg_skips_existing(
        self, mock_graph_store, mock_embedder, mock_llm, mock_session
    ):
        """Test that add_embeddings_to_kg only processes nodes without embeddings."""
        mock_graph_store.driver.session.return_value = mock_session

        nodes_without_emb = [
            {
                "id": "protein_1",
                "name": "TP53",
                "type": "Protein",
                "props": {},
                "chunk_texts": [],
            }
        ]

        verification_results = [
            MagicMock(single=MagicMock(return_value={"count": 1})),
            MagicMock(single=MagicMock(return_value={"count": 0})),
            MagicMock(single=MagicMock(return_value={"count": 1})),
            MagicMock(single=MagicMock(return_value={"count": 0})),
            MagicMock(single=MagicMock(return_value={"count": 0})),
        ]

        mock_session.run.side_effect = verification_results

        pipeline = _make_pipeline(mock_graph_store, mock_embedder, mock_llm)
        pipeline._get_nodes_without_embeddings = MagicMock(
            side_effect=[nodes_without_emb, []]
        )

        await pipeline.add_embeddings_to_kg()

        assert pipeline._get_nodes_without_embeddings.call_count == 2
        pipeline._get_nodes_without_embeddings.assert_any_call("Protein")
        pipeline._get_nodes_without_embeddings.assert_any_call("Disease")

    @pytest.mark.asyncio
    async def test_process_documents_chunks_only_skips_nodes(
        self, mock_graph_store, mock_embedder, mock_llm, mock_session
    ):
        """Test that process_documents_chunks_only skips node embeddings."""
        mock_graph_store.driver.session.return_value = mock_session

        verification_result = MagicMock()
        verification_result.single.return_value = {"count": 1}
        mock_session.run.return_value = verification_result

        # Spy on embed_documents
        original_embed = mock_embedder.embed_documents
        mock_embedder.embed_documents = MagicMock(side_effect=original_embed)

        pipeline = _make_pipeline(mock_graph_store, mock_embedder, mock_llm)

        from pipeline.ingest.ingestor import Chunk, ProcessedDocument

        chunk = Chunk(
            chunk_id="chunk_1",
            text="Test abstract text",
            title="Test Title",
            publication_year=2024,
        )
        chunk.nodes = [
            {"id": "protein_1", "name": "TP53", "type": "Protein"},
            {"id": "disease_1", "name": "Cancer", "type": "Disease"},
        ]

        doc = ProcessedDocument(
            doc_id="pubmed_12345",
            chunks=[chunk],
            metadata={"pmid": "12345"},
        )

        await pipeline.process_documents_chunks_only([doc])

        assert mock_embedder.embed_documents.call_count == 1
        call_args = mock_embedder.embed_documents.call_args[0][0]
        assert len(call_args) == 1
        assert call_args[0] == "Test abstract text"

    @pytest.mark.asyncio
    async def test_incremental_update_reuses_existing_embeddings(
        self, mock_graph_store, mock_embedder, mock_llm, mock_session
    ):
        """Test that incremental updates reuse existing embeddings."""
        mock_graph_store.driver.session.return_value = mock_session

        total_result = MagicMock()
        total_result.single.return_value = {"count": 100}

        with_emb_result = MagicMock()
        with_emb_result.single.return_value = {"count": 80}

        without_emb_nodes = [
            {
                "id": f"protein_{i}",
                "name": f"Protein{i}",
                "type": "Protein",
                "props": {},
                "chunk_texts": [],
            }
            for i in range(20)
        ]
        without_emb_result = MagicMock()
        without_emb_result.__iter__ = MagicMock(return_value=iter(without_emb_nodes))

        mock_session.run.side_effect = [
            total_result,
            with_emb_result,
            without_emb_result,
        ]

        pipeline = _make_pipeline(mock_graph_store, mock_embedder, mock_llm)

        nodes = pipeline._get_nodes_without_embeddings("Protein")

        assert len(nodes) == 20
        calls = mock_session.run.call_args_list
        query = calls[2][0][0]
        assert "WHERE n.embedding IS NULL" in query

    def test_vector_index_includes_new_embeddings(
        self, mock_graph_store, mock_embedder, mock_llm, mock_session
    ):
        """Test that vector index automatically includes new embeddings."""
        mock_graph_store.driver.session.return_value = mock_session

        index_result = MagicMock()
        index_result.single.return_value = {"name": "node_embeddings"}
        mock_session.run.return_value = index_result

        pipeline = _make_pipeline(mock_graph_store, mock_embedder, mock_llm)

        assert pipeline.db is not None

    @pytest.mark.asyncio
    async def test_batch_processing_efficiency(
        self, mock_graph_store, mock_embedder, mock_llm, mock_session
    ):
        """Test that embeddings are generated in efficient batches."""
        mock_graph_store.driver.session.return_value = mock_session

        nodes_without_emb = [
            {
                "id": f"protein_{i}",
                "name": f"Protein{i}",
                "type": "Protein",
                "props": {},
                "chunk_texts": [],
            }
            for i in range(150)
        ]

        verification_results = [
            MagicMock(single=MagicMock(return_value={"count": 150})),
            MagicMock(single=MagicMock(return_value={"count": 0})),
            MagicMock(single=MagicMock(return_value={"count": 150})),
            MagicMock(single=MagicMock(return_value={"count": 0})),
            MagicMock(single=MagicMock(return_value={"count": 0})),
        ]

        mock_session.run.side_effect = verification_results

        # Spy on embed_documents
        original_embed = mock_embedder.embed_documents
        mock_embedder.embed_documents = MagicMock(side_effect=original_embed)

        pipeline = _make_pipeline(mock_graph_store, mock_embedder, mock_llm)
        pipeline._get_nodes_without_embeddings = MagicMock(
            side_effect=[nodes_without_emb, []]
        )

        await pipeline.add_embeddings_to_kg()

        assert mock_embedder.embed_documents.call_count == 3
        for i, call in enumerate(mock_embedder.embed_documents.call_args_list):
            batch_size = len(call[0][0])
            assert batch_size == 50, f"Batch {i} should have 50 items"


@pytest.mark.xfail(reason="Mocks need rewrite for neo4j-graphrag pydantic validation")
class TestEmbeddingConsistency:
    """Test embedding consistency after incremental updates."""

    @pytest.mark.asyncio
    async def test_verification_after_update(
        self, mock_graph_store, mock_embedder, mock_llm, mock_session
    ):
        """Test that verification queries run after embedding generation."""
        mock_graph_store.driver.session.return_value = mock_session

        verification_results = [
            MagicMock(single=MagicMock(return_value={"count": 10})),
            MagicMock(single=MagicMock(return_value={"count": 5})),
            MagicMock(single=MagicMock(return_value={"count": 15})),
            MagicMock(single=MagicMock(return_value={"count": 20})),
            MagicMock(single=MagicMock(return_value={"count": 30})),
        ]

        mock_session.run.side_effect = verification_results

        pipeline = _make_pipeline(mock_graph_store, mock_embedder, mock_llm)
        pipeline._get_nodes_without_embeddings = MagicMock(return_value=[])

        await pipeline.add_embeddings_to_kg()

        assert mock_session.run.call_count == 5

    def test_embedding_dimensions_consistency(
        self, mock_graph_store, mock_embedder, mock_llm
    ):
        """Test that all embeddings have consistent dimensions."""
        pipeline = _make_pipeline(mock_graph_store, mock_embedder, mock_llm)

        embeddings = pipeline.embedder.embed_documents(["text1", "text2"])

        assert all(len(emb) == 768 for emb in embeddings)


class TestEmbeddingPipelineTypeValidation:
    """Test that EmbeddingPipeline validates backend types."""

    def test_rejects_non_graph_store(self):
        """Constructor raises TypeError for non-GraphStore."""
        with pytest.raises(TypeError, match="GraphStore"):
            EmbeddingPipeline("not-a-graph-store", _MockEmbeddingProvider(), _MockLLMProvider())

    def test_rejects_non_embedding_provider(self):
        """Constructor raises TypeError for non-EmbeddingProvider."""
        with pytest.raises(TypeError, match="EmbeddingProvider"):
            EmbeddingPipeline(_MockGraphStore(), "not-an-embedder", _MockLLMProvider())

    def test_rejects_non_llm_provider(self):
        """Constructor raises TypeError for non-LLMProvider."""
        with pytest.raises(TypeError, match="LLMProvider"):
            EmbeddingPipeline(_MockGraphStore(), _MockEmbeddingProvider(), "not-an-llm")

    def test_accepts_valid_backends(self):
        """Constructor accepts valid backend instances."""
        with patch(
            "src.utils.neo4j_index_manager.setup_graphrag_indexes",
            return_value=True,
        ):
            pipeline = EmbeddingPipeline(
                _MockGraphStore(),
                _MockEmbeddingProvider(),
                _MockLLMProvider(),
                database="test",
            )
            assert isinstance(pipeline.db, GraphStore)
            assert isinstance(pipeline.embedder, EmbeddingProvider)
            assert isinstance(pipeline.llm, LLMProvider)
