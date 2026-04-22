"""Pipeline context and stage result models.

Provides Pydantic v2 models for mutable pipeline state (PipelineContext)
and individual stage execution results (StageResult). These flow through
pipeline stages, accumulating data and tracking progress.
"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class StageResult(BaseModel):
    """Output of a single pipeline stage."""

    stage_name: str
    success: bool
    duration_seconds: float
    items_processed: int = 0
    items_skipped: int = 0
    error: str | None = None


class PipelineContext(BaseModel):
    """Mutable state flowing through pipeline stages."""

    source_records: list[dict[str, Any]] = Field(default_factory=list)
    extracted_text: list[dict[str, Any]] = Field(default_factory=list)
    deduplicated_records: list[dict[str, Any]] = Field(default_factory=list)
    chunks: list[dict[str, Any]] = Field(default_factory=list)
    entities: list[dict[str, Any]] = Field(default_factory=list)
    relationships: list[dict[str, Any]] = Field(default_factory=list)
    embeddings: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    current_stage: str = ""
    stage_timestamps: dict[str, datetime] = Field(default_factory=dict)
    stage_results: list[StageResult] = Field(default_factory=list)

    @property
    def is_complete(self) -> bool:
        """True iff all 8 stages completed successfully."""
        return (
            all(r.success for r in self.stage_results) and len(self.stage_results) == 8
        )
