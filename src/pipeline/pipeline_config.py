"""Runtime configuration for pipeline execution.

Provides a Pydantic v2 model controlling chunking, batching, concurrency,
and feature toggles for pipeline runs. Distinct from pipeline.config which
reads environment-level settings.
"""

from pydantic import BaseModel, field_validator, ValidationInfo


class PipelineConfig(BaseModel):
    """Runtime configuration for pipeline execution."""

    chunk_size: int = 2048
    chunk_overlap: int = 256
    batch_size: int = 50
    max_concurrent_extractions: int = 16
    enable_dedup: bool = True
    enable_entity_merge: bool = True
    max_gleaning: int = 1

    @field_validator("chunk_size")
    @classmethod
    def chunk_size_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError("chunk_size must be >= 1")
        return v

    @field_validator("chunk_overlap")
    @classmethod
    def overlap_less_than_chunk(cls, v: int, info: ValidationInfo) -> int:
        if "chunk_size" in info.data and v >= info.data["chunk_size"]:
            raise ValueError("chunk_overlap must be < chunk_size")
        return v
