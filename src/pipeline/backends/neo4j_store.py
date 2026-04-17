"""Neo4j graph store backend using the ``neo4j`` Python driver.

Wraps the :pypi:`neo4j` driver directly to satisfy the
:class:`~pipeline.interfaces.GraphStore` ABC.  The constructor accepts
``uri``, ``auth``, and ``database`` parameters.

The ``neo4j`` package is an optional dependency.  If it is not installed,
importing this module raises :class:`ImportError` with installation
instructions.
"""

from __future__ import annotations

import logging
from typing import Any

try:
    import neo4j
except ImportError:
    raise ImportError("Install biokg-ingest[neo4j]")

from pipeline.interfaces import GraphStore

logger = logging.getLogger(__name__)

_DEFAULT_DATABASE = "neo4j"
_BATCH_SIZE = 500


class Neo4jGraphStore(GraphStore):
    """GraphStore backed by a Neo4j database.

    Parameters
    ----------
    uri:
        Bolt or Neo4j connection URI (e.g. ``"bolt://localhost:7687"``).
        If *None*, reads from ``NEO4J_URI`` environment variable or
        ``src.config.Config.NEO4J_URI``.
    auth:
        Tuple of ``(username, password)`` for authentication.
        If *None*, reads from config/environment.
    database:
        Neo4j database name.  Defaults to ``"neo4j"``.
    user:
        Username (alternative to *auth* tuple).
    password:
        Password (alternative to *auth* tuple).
    """

    def __init__(
        self,
        uri: str | None = None,
        auth: tuple[str, str] | None = None,
        database: str = _DEFAULT_DATABASE,
        *,
        user: str | None = None,
        password: str | None = None,
    ) -> None:
        # Resolve connection parameters from config if not provided
        if uri is None or auth is None or (user is None and auth is None):
            try:
                from pipeline.config import PipelineConfig as Config

                uri = uri or Config.NEO4J_URI
                if auth is None:
                    _user = user or Config.NEO4J_USER
                    _password = password or Config.NEO4J_PASSWORD
                    auth = (_user, _password)
            except ImportError:
                import os

                uri = uri or os.environ.get("NEO4J_URI", "bolt://localhost:7687")
                if auth is None:
                    _user = user or os.environ.get("NEO4J_USER", "neo4j")
                    _password = password or os.environ.get("NEO4J_PASSWORD", "password")
                    auth = (_user, _password)

        self._uri = uri
        self._auth = auth
        self._database = database
        self._driver: neo4j.Driver = neo4j.GraphDatabase.driver(uri, auth=auth)

        # Pending batch buffers
        self._node_buffer: list[tuple[str, dict[str, Any]]] = []
        self._rel_buffer: list[tuple[str, str, str, dict[str, Any]]] = []

        # Ensure a range index on `id` for fast lookups
        try:
            with self._driver.session(database=self._database) as session:
                session.run("CREATE INDEX node_id_index IF NOT EXISTS FOR (n:Entity) ON (n.id)")
        except Exception:
            pass

    # -- GraphStore interface ------------------------------------------------

    def execute_query(
        self, query: str, params: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Execute a Cypher query and return result records as dicts."""
        with self._driver.session(database=self._database) as session:
            result = session.run(query, parameters=params or {})
            return [record.data() for record in result]

    # Alias used by kg_pipeline and other modules
    _execute_cypher = execute_query

    def upsert_node(self, label: str, properties: dict[str, Any]) -> str:
        """Buffer a node upsert. Flushes automatically at batch size."""
        node_id = properties.get("id") or self._derive_id(label, properties)
        props = {**properties, "id": node_id}
        self._node_buffer.append((label, props))
        if len(self._node_buffer) >= _BATCH_SIZE:
            self.flush()
        return node_id

    def upsert_relationship(
        self,
        source_id: str,
        rel_type: str,
        target_id: str,
        properties: dict[str, Any] | None = None,
    ) -> None:
        """Buffer a relationship upsert. Flushes automatically at batch size."""
        self._rel_buffer.append((source_id, rel_type, target_id, properties or {}))
        if len(self._rel_buffer) >= _BATCH_SIZE:
            self.flush()

    def flush(self) -> None:
        """Flush all buffered nodes and relationships in batched transactions."""
        if self._node_buffer:
            self._flush_nodes()
        if self._rel_buffer:
            self._flush_rels()

    def close(self) -> None:
        """Flush pending writes and close the driver."""
        self.flush()
        self._driver.close()

    # -- batch internals -----------------------------------------------------

    def _flush_nodes(self) -> None:
        """Write all buffered nodes in one transaction per label."""
        by_label: dict[str, list[dict[str, Any]]] = {}
        for label, props in self._node_buffer:
            by_label.setdefault(label, []).append(props)
        self._node_buffer.clear()

        with self._driver.session(database=self._database) as session:
            for label, batch in by_label.items():
                query = (
                    "UNWIND $batch AS row "
                    f"MERGE (n:`{label}` {{id: row.id}}) "
                    "SET n += row"
                )
                session.run(query, parameters={"batch": batch})
                logger.debug("Flushed %d %s nodes", len(batch), label)

    def _flush_rels(self) -> None:
        """Write all buffered relationships in one transaction per type."""
        by_type: dict[str, list[dict[str, Any]]] = {}
        for src, rel_type, tgt, props in self._rel_buffer:
            by_type.setdefault(rel_type, []).append(
                {"source_id": src, "target_id": tgt, "props": props}
            )
        self._rel_buffer.clear()

        with self._driver.session(database=self._database) as session:
            for rel_type, batch in by_type.items():
                query = (
                    "UNWIND $batch AS row "
                    "MATCH (a {id: row.source_id}), (b {id: row.target_id}) "
                    f"MERGE (a)-[r:`{rel_type}`]->(b) "
                    "SET r += row.props"
                )
                session.run(query, parameters={"batch": batch})
                logger.debug("Flushed %d %s rels", len(batch), rel_type)

    # -- helpers -------------------------------------------------------------

    @staticmethod
    def _derive_id(label: str, properties: dict[str, Any]) -> str:
        """Derive a deterministic identifier from label and properties."""
        parts = [label] + [str(v) for _, v in sorted(properties.items())]
        return ":".join(parts)
