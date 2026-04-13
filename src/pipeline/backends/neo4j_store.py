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

        # Ensure a range index on `id` for fast lookups (silences cartesian product warnings)
        try:
            with self._driver.session(database=self._database) as session:
                session.run("CREATE INDEX node_id_index IF NOT EXISTS FOR (n:Entity) ON (n.id)")
        except Exception:
            pass  # index may already exist or DB may not support IF NOT EXISTS

    # -- GraphStore interface ------------------------------------------------

    def execute_query(
        self, query: str, params: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Execute a Cypher query and return result records as dicts."""
        with self._driver.session(database=self._database) as session:
            result = session.run(query, parameters=params or {})
            return [record.data() for record in result]

    def upsert_node(self, label: str, properties: dict[str, Any]) -> str:
        """Create or update a node using MERGE.

        The node is matched/created on its ``id`` property.  If ``id`` is not
        present in *properties*, one is derived from the label and the sorted
        property values.

        Returns the node's ``id`` property.
        """
        node_id = properties.get("id") or self._derive_id(label, properties)
        props = {**properties, "id": node_id}

        query = (
            f"MERGE (n:`{label}` {{id: $id}}) SET n += $props RETURN n.id AS node_id"
        )
        records = self.execute_query(query, {"id": node_id, "props": props})
        return str(records[0]["node_id"]) if records else node_id

    def upsert_relationship(
        self,
        source_id: str,
        rel_type: str,
        target_id: str,
        properties: dict[str, Any] | None = None,
    ) -> None:
        """Create or update a relationship using MERGE between two nodes."""
        query = (
            "MATCH (a {id: $source_id}), (b {id: $target_id}) "
            f"MERGE (a)-[r:`{rel_type}`]->(b) "
        )
        if properties:
            query += "SET r += $props"

        self.execute_query(
            query,
            {
                "source_id": source_id,
                "target_id": target_id,
                "props": properties or {},
            },
        )

    def close(self) -> None:
        """Close the underlying Neo4j driver and release connections."""
        self._driver.close()

    # -- helpers -------------------------------------------------------------

    @staticmethod
    def _derive_id(label: str, properties: dict[str, Any]) -> str:
        """Derive a deterministic identifier from label and properties."""
        parts = [label] + [str(v) for _, v in sorted(properties.items())]
        return ":".join(parts)
