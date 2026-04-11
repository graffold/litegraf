"""Unit tests for pipeline.backends.neo4j_store.Neo4jGraphStore."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from pipeline.interfaces import GraphStore


class TestNeo4jGraphStoreImportGuard:
    """Verify the import guard raises when neo4j is missing."""

    def test_import_error_message(self):
        """If neo4j is absent, importing the module raises ImportError."""
        import importlib
        import sys

        # Temporarily hide neo4j
        real_neo4j = sys.modules.get("neo4j")
        sys.modules["neo4j"] = None  # type: ignore[assignment]
        try:
            # Force re-import
            if "pipeline.backends.neo4j_store" in sys.modules:
                del sys.modules["pipeline.backends.neo4j_store"]
            with pytest.raises(ImportError, match="Install biokg-ingest\\[neo4j\\]"):
                importlib.import_module("pipeline.backends.neo4j_store")
        finally:
            # Restore
            if real_neo4j is not None:
                sys.modules["neo4j"] = real_neo4j
            elif "neo4j" in sys.modules:
                del sys.modules["neo4j"]
            # Re-import the real module
            if "pipeline.backends.neo4j_store" in sys.modules:
                del sys.modules["pipeline.backends.neo4j_store"]


class TestNeo4jGraphStoreABC:
    """Verify Neo4jGraphStore satisfies the GraphStore ABC."""

    def test_is_subclass_of_graph_store(self):
        from pipeline.backends.neo4j_store import Neo4jGraphStore

        assert issubclass(Neo4jGraphStore, GraphStore)

    def test_no_remaining_abstract_methods(self):
        from pipeline.backends.neo4j_store import Neo4jGraphStore

        assert not getattr(Neo4jGraphStore, "__abstractmethods__", set())


@patch("neo4j.GraphDatabase.driver")
class TestNeo4jGraphStoreOperations:
    """Test Neo4jGraphStore methods with a mocked driver."""

    def _make_store(self, mock_driver_cls: MagicMock):
        from pipeline.backends.neo4j_store import Neo4jGraphStore

        return Neo4jGraphStore(
            uri="bolt://localhost:7687",
            auth=("neo4j", "password"),
            database="testdb",
        )

    def test_execute_query(self, mock_driver_cls: MagicMock):
        store = self._make_store(mock_driver_cls)
        mock_session = MagicMock()
        mock_record = MagicMock()
        mock_record.data.return_value = {"name": "Alice"}
        mock_session.run.return_value = [mock_record]
        store._driver.session.return_value.__enter__ = MagicMock(
            return_value=mock_session
        )
        store._driver.session.return_value.__exit__ = MagicMock(return_value=False)

        result = store.execute_query("MATCH (n) RETURN n", {"limit": 10})
        assert result == [{"name": "Alice"}]

    def test_upsert_node_with_id(self, mock_driver_cls: MagicMock):
        store = self._make_store(mock_driver_cls)
        mock_session = MagicMock()
        mock_record = MagicMock()
        mock_record.data.return_value = {"node_id": "gene_123"}
        mock_session.run.return_value = [mock_record]
        store._driver.session.return_value.__enter__ = MagicMock(
            return_value=mock_session
        )
        store._driver.session.return_value.__exit__ = MagicMock(return_value=False)

        node_id = store.upsert_node("Gene", {"id": "gene_123", "name": "TP53"})
        assert node_id == "gene_123"

    def test_upsert_node_derives_id(self, mock_driver_cls: MagicMock):
        store = self._make_store(mock_driver_cls)
        mock_session = MagicMock()
        mock_record = MagicMock()
        mock_record.data.return_value = {"node_id": "Gene:TP53"}
        mock_session.run.return_value = [mock_record]
        store._driver.session.return_value.__enter__ = MagicMock(
            return_value=mock_session
        )
        store._driver.session.return_value.__exit__ = MagicMock(return_value=False)

        node_id = store.upsert_node("Gene", {"name": "TP53"})
        # Should derive an id since none was provided
        assert isinstance(node_id, str)
        assert len(node_id) > 0

    def test_upsert_relationship(self, mock_driver_cls: MagicMock):
        store = self._make_store(mock_driver_cls)
        mock_session = MagicMock()
        mock_session.run.return_value = []
        store._driver.session.return_value.__enter__ = MagicMock(
            return_value=mock_session
        )
        store._driver.session.return_value.__exit__ = MagicMock(return_value=False)

        # Should not raise
        store.upsert_relationship("gene_1", "INTERACTS_WITH", "gene_2", {"score": 0.95})

    def test_upsert_relationship_no_properties(self, mock_driver_cls: MagicMock):
        store = self._make_store(mock_driver_cls)
        mock_session = MagicMock()
        mock_session.run.return_value = []
        store._driver.session.return_value.__enter__ = MagicMock(
            return_value=mock_session
        )
        store._driver.session.return_value.__exit__ = MagicMock(return_value=False)

        store.upsert_relationship("gene_1", "INTERACTS_WITH", "gene_2")

    def test_close(self, mock_driver_cls: MagicMock):
        store = self._make_store(mock_driver_cls)
        store.close()
        store._driver.close.assert_called_once()

    def test_context_manager(self, mock_driver_cls: MagicMock):
        from pipeline.backends.neo4j_store import Neo4jGraphStore

        store = Neo4jGraphStore(
            uri="bolt://localhost:7687",
            auth=("neo4j", "password"),
        )
        with store as s:
            assert s is store
        store._driver.close.assert_called_once()
