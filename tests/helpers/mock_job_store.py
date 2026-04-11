"""Mock JobStore for testing IngestionJobManager without Redis."""

from __future__ import annotations

from typing import Any

from pipeline.interfaces import JobStore


class MockJobStore(JobStore):
    """In-memory JobStore implementation for unit tests."""

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
