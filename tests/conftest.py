"""Shared pytest fixtures for mocking external dependencies.

This module provides reusable fixtures for:
- LLM API mocking (Ollama, Bedrock, SageMaker, OpenAI)
- Database connection mocking (Neo4j, Neptune)
- Common test utilities

These fixtures enable fast, reliable unit tests without external service dependencies.
"""

from __future__ import annotations

import json
import os
from unittest.mock import AsyncMock, MagicMock

import pytest
from hypothesis import settings

# ============================================================================
# Hypothesis Configuration
# ============================================================================

# Configure Hypothesis profiles for CI vs local development
# CI: Lower max_examples (50) for faster execution
# Local: Higher max_examples (100) for thorough testing
settings.register_profile("ci", max_examples=20)
settings.register_profile("dev", max_examples=30)

# Auto-detect CI environment and load appropriate profile
if os.getenv("CI") == "true" or os.getenv("GITHUB_ACTIONS") == "true":
    settings.load_profile("ci")
else:
    settings.load_profile("dev")


# ============================================================================
# LLM Mocking Fixtures
# ============================================================================


@pytest.fixture
def mock_llm_response():
    """Mock LLM response object with content attribute."""
    response = MagicMock()
    response.content = "Mock LLM response"
    return response


@pytest.fixture
def mock_llm(mock_llm_response):
    """Mock LLM instance with invoke method.

    Returns a MagicMock that simulates LLM behavior:
    - invoke() returns a response object with .content attribute
    - generate() returns a simple string response
    """
    llm = MagicMock()
    llm.invoke = MagicMock(return_value=mock_llm_response)
    llm.generate = MagicMock(return_value="Mock LLM response")
    return llm


@pytest.fixture
def mock_async_llm(mock_llm_response):
    """Mock async LLM instance with ainvoke method."""
    llm = MagicMock()
    llm.ainvoke = AsyncMock(return_value=mock_llm_response)
    llm.agenerate = AsyncMock(return_value="Mock LLM response")
    return llm


@pytest.fixture
def mock_ollama_llm():
    """Mock Ollama LLM with typical response format."""
    llm = MagicMock()
    response = MagicMock()
    response.content = json.dumps(
        {
            "entities": [{"name": "TP53", "type": "Protein"}],
            "relationships": [
                {"source": "TP53", "target": "Cancer", "type": "ASSOCIATED_WITH"}
            ],
        }
    )
    llm.invoke = MagicMock(return_value=response)
    return llm


@pytest.fixture
def mock_bedrock_llm():
    """Mock AWS Bedrock LLM with typical response format."""
    llm = MagicMock()
    response = MagicMock()
    response.content = "Mock Bedrock response"
    llm.invoke = MagicMock(return_value=response)
    return llm


@pytest.fixture
def mock_sagemaker_llm():
    """Mock AWS SageMaker LLM with typical response format."""
    llm = MagicMock()
    response = MagicMock()
    response.content = "Mock SageMaker response"
    llm.invoke = MagicMock(return_value=response)
    return llm


@pytest.fixture
def mock_openai_llm():
    """Mock OpenAI LLM with typical response format."""
    llm = MagicMock()
    response = MagicMock()
    response.content = "Mock OpenAI response"
    llm.invoke = MagicMock(return_value=response)
    return llm


# ============================================================================
# Database Mocking Fixtures
# ============================================================================


@pytest.fixture
def mock_neo4j_driver():
    """Mock Neo4j driver with session management."""
    driver = MagicMock()
    session = MagicMock()
    session.run = MagicMock(return_value=[])
    session.close = MagicMock()
    driver.session = MagicMock(return_value=session)
    driver.close = MagicMock()
    return driver


@pytest.fixture
def mock_neo4j_database(mock_neo4j_driver):
    """Mock Neo4jDatabase instance with common methods.

    Provides mocked methods:
    - _execute_cypher() returns empty list
    - query() returns empty list
    - close() does nothing
    """
    db = MagicMock()
    db._driver = mock_neo4j_driver
    db._execute_cypher = MagicMock(return_value=[])
    db.query = MagicMock(return_value=[])
    db.close = MagicMock()
    return db


@pytest.fixture
def mock_neptune_client():
    """Mock Neptune OpenCypher client."""
    client = MagicMock()
    client.execute_query = MagicMock(return_value={"results": []})
    return client


@pytest.fixture
def mock_neptune_database(mock_neptune_client):
    """Mock NeptuneDatabase instance with common methods.

    Provides mocked methods:
    - execute_query() returns empty results
    - close() does nothing
    """
    db = MagicMock()
    db._client = mock_neptune_client
    db.execute_query = MagicMock(return_value={"results": []})
    db.close = MagicMock()
    return db


# ============================================================================
# Embedding Mocking Fixtures
# ============================================================================


@pytest.fixture
def mock_embedder():
    """Mock embedding model with encode method.

    Returns embeddings with shape (len(texts), 384).
    """
    import numpy as np

    embedder = MagicMock()

    def encode_side_effect(texts):
        if isinstance(texts, str):
            return np.random.rand(384)
        return np.random.rand(len(texts), 384)

    embedder.encode = MagicMock(side_effect=encode_side_effect)
    embedder.embed_query = MagicMock(return_value=[0.1] * 384)
    return embedder


# ============================================================================
# Graph Schema Fixtures
# ============================================================================


@pytest.fixture
def sample_graph_schema():
    """Sample graph schema for testing GraphTools."""
    return {
        "node_labels": ["Protein", "Disease", "Chunk", "Abstract"],
        "relationship_types": ["ASSOCIATED_WITH", "INTERACTS_WITH", "HAS_CHUNK"],
        "property_keys": {
            "Protein": ["name", "id", "uniprot_id", "gene_symbol"],
            "Disease": ["name", "id", "mondo_id"],
            "Chunk": ["text", "chunk_id"],
            "Abstract": ["pmid", "title", "abstract"],
        },
    }


# ============================================================================
# Test Data Fixtures
# ============================================================================


@pytest.fixture
def sample_entities():
    """Sample entity extraction results."""
    return [
        {"name": "TP53", "type": "Protein", "properties": {"uniprot_id": "P04637"}},
        {
            "name": "Cancer",
            "type": "Disease",
            "properties": {"mondo_id": "MONDO:0004992"},
        },
        {"name": "BRCA1", "type": "Protein", "properties": {"uniprot_id": "P38398"}},
    ]


@pytest.fixture
def sample_relationships():
    """Sample relationship extraction results."""
    return [
        {
            "source": "TP53",
            "target": "Cancer",
            "type": "ASSOCIATED_WITH",
            "properties": {"confidence": 0.95, "pmid": "12345"},
        },
        {
            "source": "BRCA1",
            "target": "Cancer",
            "type": "ASSOCIATED_WITH",
            "properties": {"confidence": 0.92, "pmid": "67890"},
        },
    ]


@pytest.fixture
def sample_cypher_results():
    """Sample Cypher query results."""
    return [
        {"p.name": "TP53", "p.id": "P04637", "p.type": "Protein"},
        {"p.name": "BRCA1", "p.id": "P38398", "p.type": "Protein"},
    ]
