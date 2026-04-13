"""Data models returned by LiteGraf insert and query operations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

QueryMode = Literal["naive", "local", "global", "hybrid", "mix"]
QUERY_MODES: set[str] = {"naive", "local", "global", "hybrid", "mix"}


@dataclass
class ContextChunk:
    """A chunk of context retrieved during a query."""

    chunk_id: str
    text: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class InsertResult:
    """Result of an insert operation."""

    doc_id: str | list[str]
    chunks_processed: int
    entities_extracted: int
    relationships_extracted: int
    was_duplicate: bool
    duration_seconds: float


@dataclass
class QueryResult:
    """Result of a query operation."""

    answer: str | None
    context: list[ContextChunk]
    duration_seconds: float
