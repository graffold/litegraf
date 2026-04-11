import logging
import time

from neo4j_graphrag.retrievers import Text2CypherRetriever, VectorRetriever

from pipeline.ingest.ingestor import ProcessedDocument
from pipeline.interfaces import EmbeddingProvider, GraphStore, LLMProvider
# Import disease hierarchy enricher
try:
    from pipeline.processors.disease_hierarchy_enricher import DiseaseHierarchyEnricher

    HAS_HIERARCHY_ENRICHER = True
except ImportError:
    DiseaseHierarchyEnricher = type(None)  # type: ignore
    HAS_HIERARCHY_ENRICHER = False

logger = logging.getLogger(__name__)
class EmbeddingPipeline:
    def __init__(
        self,
        graph_store: GraphStore,
        embedding_provider: EmbeddingProvider,
        llm_provider: LLMProvider,
        *,
        database: str = "neo4j",
        store_embedding_properties: bool = False,
        optimize_text_storage: bool = True,
    ):
        """
        Initialize the embedding pipeline.

        Args:
            graph_store: GraphStore backend for graph database operations
            embedding_provider: EmbeddingProvider backend for text embeddings
            llm_provider: LLMProvider backend for LLM operations
            database: Database name
            store_embedding_properties: Whether to store embeddings as node properties (default False for efficiency)
            optimize_text_storage: Whether to avoid storing redundant abstract_text when contained in full_text (default True)
        """
        # --- Type validation for injected backends ---
        if not isinstance(graph_store, GraphStore):
            raise TypeError(
                f"graph_store must be a GraphStore instance, got {type(graph_store).__name__}"
            )
        if not isinstance(embedding_provider, EmbeddingProvider):
            raise TypeError(
                f"embedding_provider must be an EmbeddingProvider instance, got {type(embedding_provider).__name__}"
            )
        if not isinstance(llm_provider, LLMProvider):
            raise TypeError(
                f"llm_provider must be an LLMProvider instance, got {type(llm_provider).__name__}"
            )

        self.db = graph_store
        self.embedder = embedding_provider
        self.llm = llm_provider

        self.database = database
        self.store_embedding_properties = store_embedding_properties
        self.optimize_text_storage = optimize_text_storage
        logger.info("Initializing EmbeddingPipeline with injected backends")

        # Initialize disease hierarchy enricher
        self.hierarchy_enricher = None
        if (
            HAS_HIERARCHY_ENRICHER and database and database != "test"
        ):  # Skip for test databases
            try:
                self.hierarchy_enricher = DiseaseHierarchyEnricher(database=database)  # type: ignore
                logger.info("Disease hierarchy enricher initialized")
            except Exception as e:
                logger.warning(f"Could not initialize disease hierarchy enricher: {e}")

        self.vector_retriever = None
        self.text2cypher_retriever = None

        self._create_indexes()
        # Defer retriever initialization — only needed for querying, not embedding generation
        self._retrievers_initialized = False

    def _get_driver(self):
        """Get the underlying Neo4j driver from the graph store, if available.

        Returns the driver attribute from the graph store backend, or None
        if the backend doesn't expose a driver (e.g., non-Neo4j backends).
        """
        return getattr(self.db, "driver", None)

    def _get_driver_session(self):
        """Get a Neo4j session from the graph store's driver.

        Returns a context-manager session, or raises RuntimeError if the
        backend doesn't expose a driver.
        """
        driver = self._get_driver()
        if driver is None:
            raise RuntimeError(
                "Graph store backend does not expose a 'driver' attribute. "
                "Neo4j-specific operations require a Neo4j-compatible backend."
            )
        return driver.session(database=self.database)

    def _create_indexes(self):
        """Create vector and full-text indexes using neo4j_graphrag standards."""
        driver = self._get_driver()
        if driver is None:
            logger.warning(
                "Graph store backend does not expose a 'driver' attribute. "
                "Skipping Neo4j-specific index creation."
            )
            return

        try:
            from neo4j_graphrag.indexes import create_vector_index, create_fulltext_index
        except ImportError:
            create_vector_index = None
            create_fulltext_index = None

        def _setup_graphrag_indexes(driver, database, dimensions=768, await_online=True):
            """Set up neo4j-graphrag vector and fulltext indexes."""
            if create_vector_index is None:
                return False
            try:
                create_vector_index(
                    driver,
                    name="node_embeddings",
                    label="Entity",
                    embedding_property="embedding",
                    dimensions=dimensions,
                    similarity_fn="cosine",
                )
                create_vector_index(
                    driver,
                    name="chunk_embeddings",
                    label="Chunk",
                    embedding_property="embedding",
                    dimensions=dimensions,
                    similarity_fn="cosine",
                )
                create_fulltext_index(
                    driver,
                    name="node_fulltext",
                    label="Entity",
                    node_properties=["name", "full_text", "abstract_text"],
                )
                if await_online:
                    with driver.session(database=database) as session:
                        session.run("CALL db.awaitIndexes(300)")
                return True
            except Exception as e:
                logger.warning(f"graphrag index setup failed: {e}")
                return False

        success = _setup_graphrag_indexes(
            driver=driver,
            database=self.database,
            dimensions=768,
            await_online=True,
        )

        if success:
            logger.info(
                "✓ All indexes created successfully using neo4j_graphrag standards"
            )
            return
        logger.warning("⚠ Some indexes failed, trying manual fallback...")

        # Fallback to manual creation if neo4j_graphrag fails
        try:
            with driver.session(database=self.database) as session:
                index_exists = session.run(
                    "SHOW INDEXES WHERE name = 'node_embeddings'"
                ).single()
                if not index_exists:
                    session.run("""
                        CREATE VECTOR INDEX node_embeddings IF NOT EXISTS
                        FOR (n:Entity) ON (n.embedding)
                        OPTIONS {indexConfig: {`vector.dimensions`: 768, `vector.similarity_function`: 'cosine'}}
                    """)
                    logger.info("Created vector index: node_embeddings for Entity")
                fulltext_exists = session.run(
                    "SHOW INDEXES WHERE name = 'node_fulltext'"
                ).single()
                if not fulltext_exists:
                    session.run("""
                        CREATE FULLTEXT INDEX node_fulltext IF NOT EXISTS
                        FOR (n:Entity) ON EACH [n.name, n.full_text, n.abstract_text]
                    """)
                    logger.info(
                        "Created full-text index: node_fulltext for name, full_text, and abstract_text"
                    )
                chunk_index_exists = session.run(
                    "SHOW INDEXES WHERE name = 'chunk_embeddings'"
                ).single()
                if not chunk_index_exists:
                    session.run("""
                        CREATE VECTOR INDEX chunk_embeddings IF NOT EXISTS
                        FOR (c:Chunk) ON (c.embedding)
                        OPTIONS {indexConfig: {`vector.dimensions`: 768, `vector.similarity_function`: 'cosine'}}
                    """)
                    logger.info("Created vector index: chunk_embeddings")
                session.run("CALL db.awaitIndexes(300)")
                time.sleep(5)
        except Exception as e:
            logger.error(f"Failed to create or await indexes: {e}")
            raise

    def _initialize_retrievers(self):
        """Initialize retrievers after index creation. Called lazily on first use."""
        if self._retrievers_initialized:
            return
        try:
            with self._get_driver_session() as session:
                index_exists = session.run(
                    "SHOW INDEXES WHERE name = 'node_embeddings'"
                ).single()
                if not index_exists:
                    raise Exception("node_embeddings index not found after creation")
                chunk_index_exists = session.run(
                    "SHOW INDEXES WHERE name = 'chunk_embeddings'"
                ).single()
                if not chunk_index_exists:
                    raise Exception("chunk_embeddings index not found after creation")

            logger.info("Initializing neo4j-graphrag retrievers")
            driver = self._get_driver()
            self.vector_retriever = VectorRetriever(
                driver, "node_embeddings", embedder=self.embedder
            )
            self.text2cypher_retriever = Text2CypherRetriever(
                driver=driver, llm=self.llm
            )
            logger.info("neo4j-graphrag retrievers initialized successfully")

            self._retrievers_initialized = True

        except Exception as e:
            logger.error(f"Failed to initialize retrievers: {e}")
            raise

    def cleanup_embedding_properties(self, dry_run: bool = True):
        """
        Remove embedding properties from nodes after vector indexes are created.
        This optimizes storage while maintaining vector search functionality.
        """
        try:
            with self._get_driver_session() as session:
                # Check current storage usage
                stats_query = """
                MATCH (n)
                WHERE n.embedding IS NOT NULL
                WITH size(n.embedding) as emb_size, count(n) as node_count
                RETURN node_count, avg(emb_size) as avg_dimensions,
                       (node_count * avg(emb_size) * 4) as approx_bytes
                """
                stats = session.run(stats_query).single()
                if stats:
                    node_count = stats["node_count"]
                    avg_dims = stats["avg_dimensions"]
                    approx_mb = (
                        stats["approx_bytes"] / (1024 * 1024)
                        if stats["approx_bytes"]
                        else 0
                    )

                    logger.info(
                        f"Found {node_count} nodes with embeddings ({avg_dims:.0f} dims each)"
                    )
                    logger.info(
                        f"Estimated storage: {approx_mb:.1f} MB in embedding properties"
                    )

                    if dry_run:
                        logger.info(
                            "DRY RUN: Would remove embedding properties from nodes"
                        )
                        logger.info(
                            "Vector indexes will continue to work for similarity search"
                        )
                        return {
                            "action": "dry_run",
                            "nodes_affected": node_count,
                            "storage_mb": approx_mb,
                        }
                    # Remove embedding properties but keep other properties
                    cleanup_query = """
                        MATCH (n)
                        WHERE n.embedding IS NOT NULL
                        REMOVE n.embedding
                        RETURN count(n) as cleaned_count
                        """
                    result = session.run(cleanup_query).single()
                    cleaned_count = result["cleaned_count"] if result else 0

                    logger.info(
                        f"Removed embedding properties from {cleaned_count} nodes"
                    )
                    logger.info(f"Freed approximately {approx_mb:.1f} MB of storage")
                    logger.info(
                        "Vector indexes remain functional for similarity search"
                    )

                    return {
                        "action": "cleaned",
                        "nodes_affected": cleaned_count,
                        "storage_freed_mb": approx_mb,
                    }
                logger.info("No embedding properties found to clean up")
                return {"action": "none_found", "nodes_affected": 0, "storage_mb": 0}

        except Exception as e:
            logger.error(f"Failed to cleanup embedding properties: {e}")
            raise

    def _get_nodes_without_embeddings(self, label: str) -> list[dict]:
        """Fetch nodes of given label without embeddings."""
        try:
            with self._get_driver_session() as session:
                total_count = session.run(
                    f"MATCH (n:{label}) RETURN count(n) AS count"
                ).single()["count"]
                logger.info(f"Total {label} nodes in database: {total_count}")

                # Check how many already have embeddings
                with_embeddings = session.run(
                    f"MATCH (n:{label}) WHERE n.embedding IS NOT NULL RETURN count(n) AS count"
                ).single()["count"]
                logger.info(f"{label} nodes with embeddings: {with_embeddings}")

                # Get nodes without embeddings, including their properties for better embedding text
                result = session.run(
                    f"""
                    MATCH (n:{label}) WHERE n.embedding IS NULL
                    OPTIONAL MATCH (n)-[:FROM_CHUNK]->(c:Chunk)
                    RETURN n.id AS id, n.name AS name, '{label}' AS type,
                           properties(n) AS props,
                           collect(DISTINCT c.text)[0..3] AS chunk_texts
                    """
                )
                nodes = []
                for record in result:
                    node_data = {
                        "id": record["id"],
                        "name": record["name"],
                        "type": record["type"],
                        "props": record["props"] or {},
                        "chunk_texts": record["chunk_texts"] or [],
                    }
                    nodes.append(node_data)
                logger.info(f"Found {len(nodes)} {label} nodes without embeddings")
                if total_count > 0 and len(nodes) == total_count:
                    logger.warning(
                        f"All {label} nodes lack embeddings. Check embedding pipeline."
                    )
                return nodes
        except Exception as e:
            logger.error(f"Failed to fetch {label} nodes without embeddings: {e}")
            raise

    async def add_embeddings_to_kg(self):
        """Generate and store embeddings for existing Protein and Disease nodes without embeddings."""
        logger.info("Adding embeddings to existing KG nodes")
        try:
            protein_nodes = self._get_nodes_without_embeddings("Protein")
            disease_nodes = self._get_nodes_without_embeddings("Disease")
            all_nodes = protein_nodes + disease_nodes

            if not all_nodes:
                logger.info("No nodes found without embeddings")
                return

            batch_size = 50  # Smaller batch size for better performance
            for i in range(0, len(all_nodes), batch_size):
                batch = all_nodes[i : i + batch_size]
                # Create meaningful text for embedding that includes enriched properties
                node_texts = []
                for node in batch:
                    # Include properties in the text for better semantic representation
                    props = node.get("props", {})
                    text_parts = [f"{node['type']}: {node['name']}"]

                    # Add important properties to the embedding text
                    for key, value in props.items():
                        if key not in [
                            "id",
                            "name",
                            "embedding",
                            "full_text",
                            "abstract_text",
                        ] and value not in [None, "NA", ""]:
                            text_parts.append(f"{key}: {value}")

                    # Add chunk texts if available
                    if node.get("chunk_texts"):
                        for i_chunk, chunk_text in enumerate(
                            node["chunk_texts"][:2]
                        ):  # Use up to 2 abstracts
                            if chunk_text:
                                text_parts.append(
                                    f"Abstract_{i_chunk + 1}: {chunk_text[:200]}..."
                                )  # Truncate for performance

                    node_texts.append(" | ".join(text_parts))

                try:
                    embeddings = self.embedder.embed_documents(node_texts)
                    with self._get_driver_session() as session:
                        for node, embedding, node_text in zip(
                            batch, embeddings, node_texts, strict=False
                        ):
                            # Store the combined abstracts as abstract_text
                            abstract_text = ""
                            if node.get("chunk_texts"):
                                abstract_text = " ".join(
                                    [text for text in node["chunk_texts"] if text]
                                )

                            # Update existing node - Always store embedding property for Vector Index
                            result = session.run(
                                """
                                MATCH (n {id: $id})
                                SET n.embedding = $embedding, n.full_text = $full_text, n.abstract_text = $abstract_text, n:Entity
                                RETURN n
                                """,
                                id=node["id"],
                                embedding=embedding,
                                full_text=node_text,
                                abstract_text=abstract_text,
                            )

                            if result.single():
                                logger.debug(
                                    f"Updated existing {node['type']} node with embedding: {node['name']}"
                                )
                            else:
                                logger.warning(
                                    f"Could not find node with id: {node['id']}"
                                )

                    logger.info(
                        f"Processed batch {i // batch_size + 1} of {len(all_nodes) // batch_size + 1}"
                    )
                except Exception as e:
                    logger.error(f"Failed to process batch {i // batch_size + 1}: {e}")
                    continue

            # Verification section
            with self._get_driver_session() as session:
                protein_count = session.run(
                    "MATCH (n:Protein) WHERE n.embedding IS NOT NULL RETURN count(n) AS count"
                ).single()["count"]
                disease_count = session.run(
                    "MATCH (n:Disease) WHERE n.embedding IS NOT NULL RETURN count(n) AS count"
                ).single()["count"]
                entity_count = session.run(
                    "MATCH (n:Entity) WHERE n.embedding IS NOT NULL RETURN count(n) AS count"
                ).single()["count"]
                chunk_count = session.run(
                    "MATCH (c:Chunk) WHERE c.embedding IS NOT NULL RETURN count(c) AS count"
                ).single()["count"]
                rel_count = session.run(
                    "MATCH ()-[:ASSOCIATED_WITH]->() RETURN count(*) AS count"
                ).single()["count"]
                logger.info(
                    f"Verification: {protein_count} Protein nodes, {disease_count} Disease nodes, {entity_count} Entity nodes, {chunk_count} Chunk nodes, {rel_count} ASSOCIATED_WITH relationships"
                )

            if protein_count == 0 and disease_count == 0:
                logger.error("No embeddings stored for Protein or Disease nodes.")
            if entity_count == 0:
                logger.error("No Entity nodes with embeddings.")
            if chunk_count == 0:
                logger.warning("No Chunk nodes with embeddings.")
            if rel_count == 0:
                logger.error("No ASSOCIATED_WITH relationships found.")
        except Exception as e:
            logger.error(f"Failed to add embeddings to KG: {e}")
            raise

    async def process_documents(
        self, docs: list[ProcessedDocument]
    ) -> list[ProcessedDocument]:
        """Generate and store embeddings for chunks and KG nodes."""
        logger.info(f"Generating embeddings for {len(docs)} documents")
        if not docs:
            logger.warning("No documents provided for embedding.")
            return docs
        for doc in docs:
            if not doc.chunks:
                logger.warning(f"Document {doc.doc_id} has no chunks.")
                continue
            for chunk in doc.chunks:
                try:
                    if not chunk.text:
                        logger.warning(
                            f"Chunk {chunk.chunk_id} has empty text. Skipping."
                        )
                        continue
                    chunk_embedding = self.embedder.embed_documents([chunk.text])[0]
                    chunk.embedding = chunk_embedding
                    with self._get_driver_session() as session:
                        # Extract PMID from doc_id (format: "pubmed_12345")
                        pmid = (
                            doc.doc_id.replace("pubmed_", "")
                            if doc.doc_id.startswith("pubmed_")
                            else doc.doc_id
                        )
                        session.run(
                            """
                            MERGE (c:Chunk {chunk_id: $chunk_id})
                            SET c.text = $text, c.embedding = $embedding, c.doc_id = $doc_id, c.pmid = $pmid, c.title = $title, c.publication_year = $publication_year
                            WITH c
                            MERGE (a:Abstract {pmid: $pmid})
                            MERGE (a)-[:HAS_CHUNK]->(c)
                            """,
                            chunk_id=chunk.chunk_id,
                            text=chunk.text,
                            embedding=chunk_embedding,
                            doc_id=doc.doc_id,
                            pmid=pmid,
                            title=chunk.title or "",
                            publication_year=chunk.publication_year,
                        )
                        logger.debug(f"Stored embedding for chunk: {chunk.chunk_id}")
                    if not chunk.nodes:
                        logger.debug(f"Chunk {chunk.chunk_id} has no nodes.")
                    for node in chunk.nodes:
                        node_text = f"{node['type']}: {node.get('name', 'unknown')}"
                        node_embedding = self.embedder.embed_documents([node_text])[0]
                        node_label = node.get("type", "Protein")

                        # Create full_text content for search
                        full_text_parts = [
                            f"Type: {node['type']}",
                            f"Name: {node.get('name', 'unknown')}",
                            f"ID: {node['id']}",
                        ]
                        if chunk.text:
                            full_text_parts.append(f"Abstract: {chunk.text}")
                        full_text = " | ".join(full_text_parts)

                        # Optimize text storage
                        if (
                            self.optimize_text_storage
                            and chunk.text
                            and chunk.text in full_text
                        ):
                            abstract_text_to_store = None
                        else:
                            abstract_text_to_store = chunk.text

                        with self._get_driver_session() as session:
                            if self.store_embedding_properties:
                                if abstract_text_to_store is not None:
                                    result = session.run(
                                        f"""
                                        MATCH (n:{node_label} {{id: $id}})
                                        SET n.embedding = $embedding, n.full_text = $full_text, n.abstract_text = $abstract_text, n:Entity
                                        RETURN n
                                        """,
                                        id=node["id"],
                                        embedding=node_embedding,
                                        full_text=full_text,
                                        abstract_text=abstract_text_to_store,
                                    )
                                else:
                                    result = session.run(
                                        f"""
                                        MATCH (n:{node_label} {{id: $id}})
                                        SET n.embedding = $embedding, n.full_text = $full_text, n:Entity
                                        REMOVE n.abstract_text
                                        RETURN n
                                        """,
                                        id=node["id"],
                                        embedding=node_embedding,
                                        full_text=full_text,
                                    )
                            elif abstract_text_to_store is not None:
                                result = session.run(
                                    f"""
                                    MATCH (n:{node_label} {{id: $id}})
                                    SET n.full_text = $full_text, n.abstract_text = $abstract_text, n:Entity
                                    RETURN n
                                    """,
                                    id=node["id"],
                                    full_text=full_text,
                                    abstract_text=abstract_text_to_store,
                                )
                            else:
                                result = session.run(
                                    f"""
                                    MATCH (n:{node_label} {{id: $id}})
                                    SET n.full_text = $full_text, n:Entity
                                    REMOVE n.abstract_text
                                    RETURN n
                                    """,
                                    id=node["id"],
                                    full_text=full_text,
                                )
                            if result.single():
                                logger.debug(
                                    f"Updated existing {node_label} node with embedding: {node['id']} (name: {node.get('name', 'unknown')})"
                                )
                            else:
                                # If node doesn't exist, create it
                                if self.store_embedding_properties:
                                    if abstract_text_to_store is not None:
                                        session.run(
                                            f"""
                                            MERGE (n:{node_label} {{id: $id}})
                                            ON CREATE SET n.first_seen = datetime()
                                            ON MATCH SET n.last_seen = datetime()
                                            SET n.embedding = $embedding, n.name = $name, n.type = $type, n.full_text = $full_text, n.abstract_text = $abstract_text, n:Entity
                                            """,
                                            id=node["id"],
                                            embedding=node_embedding,
                                            name=node.get("name", "unknown"),
                                            type=node_label,
                                            full_text=full_text,
                                            abstract_text=abstract_text_to_store,
                                        )
                                    else:
                                        session.run(
                                            f"""
                                            MERGE (n:{node_label} {{id: $id}})
                                            ON CREATE SET n.first_seen = datetime()
                                            ON MATCH SET n.last_seen = datetime()
                                            SET n.embedding = $embedding, n.name = $name, n.type = $type, n.full_text = $full_text, n:Entity
                                            """,
                                            id=node["id"],
                                            embedding=node_embedding,
                                            name=node.get("name", "unknown"),
                                            type=node_label,
                                            full_text=full_text,
                                        )
                                elif abstract_text_to_store is not None:
                                    session.run(
                                        f"""
                                        MERGE (n:{node_label} {{id: $id}})
                                        ON CREATE SET n.first_seen = datetime()
                                        ON MATCH SET n.last_seen = datetime()
                                        SET n.name = $name, n.type = $type, n.full_text = $full_text, n.abstract_text = $abstract_text, n:Entity
                                        """,
                                        id=node["id"],
                                        name=node.get("name", "unknown"),
                                        type=node_label,
                                        full_text=full_text,
                                        abstract_text=abstract_text_to_store,
                                    )
                                else:
                                    session.run(
                                        f"""
                                        MERGE (n:{node_label} {{id: $id}})
                                        ON CREATE SET n.first_seen = datetime()
                                        ON MATCH SET n.last_seen = datetime()
                                        SET n.name = $name, n.type = $type, n.full_text = $full_text, n:Entity
                                        """,
                                        id=node["id"],
                                        name=node.get("name", "unknown"),
                                        type=node_label,
                                        full_text=full_text,
                                    )
                                logger.debug(
                                    f"Created new {node_label} node with embedding: {node['id']} (name: {node.get('name', 'unknown')})"
                                )

                            # Ensure chunk-node relationship exists
                            session.run(
                                f"""
                                MATCH (n:{node_label} {{id: $id}})
                                MERGE (c:Chunk {{chunk_id: $chunk_id}})
                                MERGE (n)-[:FROM_CHUNK]->(c)
                                """,
                                id=node["id"],
                                chunk_id=chunk.chunk_id,
                            )
                    if not chunk.relationships:
                        logger.debug(f"Chunk {chunk.chunk_id} has no relationships.")
                    for rel in chunk.relationships:
                        source_label = rel.get("source_type", "Protein")
                        target_label = rel.get("target_type", "Disease")
                        rel_type = rel.get("type", "ASSOCIATED_WITH")
                        try:
                            with self._get_driver_session() as session:
                                pattern_id = (
                                    f"{rel['source_id']}-{rel_type}-{rel['target_id']}"
                                )
                                session.run(
                                    f"""
                                    MATCH (s:{source_label} {{id: $source_id}})
                                    MATCH (t:{target_label} {{id: $target_id}})
                                    MERGE (s)-[r:{rel_type}]->(t)
                                    SET r.source_doc = $source_doc,
                                        r.pattern_id = $pattern_id,
                                        r.original_type = $rel_type,
                                        r.canonical_type = $rel_type
                                    """,
                                    source_id=rel["source_id"],
                                    target_id=rel["target_id"],
                                    source_doc=doc.doc_id,
                                    pattern_id=pattern_id,
                                    rel_type=rel_type,
                                )
                            logger.debug(
                                f"Stored relationship: {source_label}({rel['source_id']}) -[{rel_type}]-> {target_label}({rel['target_id']}) from {doc.doc_id}"
                            )
                        except Exception as e:
                            logger.error(
                                f"Failed to store relationship for {rel['source_id']} -> {rel['target_id']}: {e}"
                            )
                            continue
                    logger.info(
                        f"Processed chunk {chunk.chunk_id} with {len(chunk.nodes)} nodes and {len(chunk.relationships)} relationships"
                    )
                except Exception as e:
                    logger.error(f"Embedding failed for chunk {chunk.chunk_id}: {e}")
                    continue
        with self._get_driver_session() as session:
            protein_count = session.run(
                "MATCH (n:Protein) WHERE n.embedding IS NOT NULL RETURN count(n) AS count"
            ).single()["count"]
            disease_count = session.run(
                "MATCH (n:Disease) WHERE n.embedding IS NOT NULL RETURN count(n) AS count"
            ).single()["count"]
            entity_count = session.run(
                "MATCH (n:Entity) WHERE n.embedding IS NOT NULL RETURN count(n) AS count"
            ).single()["count"]
            chunk_count = session.run(
                "MATCH (c:Chunk) WHERE c.embedding IS NOT NULL RETURN count(c) AS count"
            ).single()["count"]
            rel_count = session.run(
                "MATCH ()-[:ASSOCIATED_WITH]->() RETURN count(*) AS count"
            ).single()["count"]
            alzheimer_count = session.run(
                "MATCH (d:Disease) WHERE toLower(d.name) CONTAINS 'alzheimer' RETURN count(d) AS count"
            ).single()["count"]
            logger.info(
                f"Verification: {protein_count} Protein nodes, {disease_count} Disease nodes, {entity_count} Entity nodes, {chunk_count} Chunk nodes, {rel_count} ASSOCIATED_WITH relationships, {alzheimer_count} Alzheimer nodes"
            )
            if protein_count == 0 and disease_count == 0:
                logger.error("No embeddings stored for Protein or Disease nodes.")
            if entity_count == 0:
                logger.error("No Entity nodes with embeddings.")
            if chunk_count == 0:
                logger.error("No Chunk nodes with embeddings.")
            if rel_count == 0:
                logger.error("No ASSOCIATED_WITH relationships found.")
            if alzheimer_count == 0:
                logger.warning("No Disease nodes with 'Alzheimer' in name.")
        return docs

    async def process_documents_chunks_only(
        self, docs: list[ProcessedDocument]
    ) -> list[ProcessedDocument]:
        """Generate and store embeddings ONLY for text chunks, skip graph nodes."""
        logger.info(
            f"Generating chunk embeddings only for {len(docs)} documents (skipping graph nodes)"
        )
        if not docs:
            logger.warning("No documents provided for embedding.")
            return docs
        for doc in docs:
            if not doc.chunks:
                logger.warning(f"Document {doc.doc_id} has no chunks.")
                continue
            for chunk in doc.chunks:
                try:
                    if not chunk.text:
                        logger.warning(
                            f"Chunk {chunk.chunk_id} has empty text. Skipping."
                        )
                        continue

                    # Only process chunk embeddings
                    chunk_embedding = self.embedder.embed_documents([chunk.text])[0]
                    chunk.embedding = chunk_embedding

                    # Extract PMID from doc_id (format: "pubmed_12345")
                    pmid = (
                        doc.doc_id.replace("pubmed_", "")
                        if doc.doc_id.startswith("pubmed_")
                        else doc.doc_id
                    )
                    with self._get_driver_session() as session:
                        session.run(
                            """
                            MERGE (c:Chunk {chunk_id: $chunk_id})
                            SET c.text = $text, c.embedding = $embedding, c.doc_id = $doc_id, c.pmid = $pmid, c.title = $title, c.publication_year = $publication_year
                            WITH c
                            MERGE (a:Abstract {pmid: $pmid})
                            MERGE (a)-[:HAS_CHUNK]->(c)
                            """,
                            chunk_id=chunk.chunk_id,
                            text=chunk.text,
                            embedding=chunk_embedding,
                            doc_id=doc.doc_id,
                            pmid=pmid,
                            title=chunk.title or "",
                            publication_year=chunk.publication_year,
                        )
                        logger.debug(
                            f"Stored chunk embedding via Neo4j: {chunk.chunk_id}"
                        )

                    # Skip graph node embeddings - they will be added later with enriched properties
                    logger.debug(
                        f"Skipping {len(chunk.nodes)} node embeddings for later enriched processing"
                    )

                except Exception as e:
                    logger.error(
                        f"Chunk embedding failed for chunk {chunk.chunk_id}: {e}"
                    )
                    continue

        # Show chunk embedding statistics
        with self._get_driver_session() as session:
            chunk_count = session.run(
                "MATCH (c:Chunk) WHERE c.embedding IS NOT NULL RETURN count(c) AS count"
            ).single()["count"]
            logger.info(
                f"Chunk embedding verification: {chunk_count} chunks with embeddings"
            )

        return docs

    async def enrich_disease_hierarchy(self):
        """Enrich disease nodes with hierarchical relationships from MONDO ontology."""
        if not self.hierarchy_enricher:
            logger.warning("Disease hierarchy enricher not available")
            return

        try:
            logger.info(
                "Enriching disease nodes with hierarchical relationships and consolidating extracted diseases"
            )
            stats = self.hierarchy_enricher.enrich_disease_hierarchy(
                dry_run=False, consolidate_extracted=True
            )

            if stats["mondo_matches_found"] > 0:
                logger.info(
                    f"Successfully enriched {stats['mondo_matches_found']} diseases with hierarchy"
                )
                logger.info(
                    f"Created {stats['parent_relationships_created']} parent relationships"
                )
                logger.info(
                    f"Created {stats['ancestor_relationships_created']} ancestor relationships"
                )

            if stats.get("nodes_merged", 0) > 0:
                logger.info(
                    f"Consolidated {stats['nodes_merged']} extracted disease nodes with hierarchy"
                )
                logger.info(
                    f"Transferred {stats['relationships_transferred']} relationships to hierarchical nodes"
                )

            if stats["mondo_matches_found"] == 0 and stats.get("nodes_merged", 0) == 0:
                logger.info(
                    "No new hierarchical relationships or consolidations created"
                )

        except Exception as e:
            logger.error(f"Failed to enrich disease hierarchy: {e}")

    def close(self):
        if hasattr(self, "db") and self.db:
            self.db.close()
        if hasattr(self, "hierarchy_enricher") and self.hierarchy_enricher:
            self.hierarchy_enricher.close()
        logger.info("EmbeddingPipeline closed")
