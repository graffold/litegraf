"""Unit tests for pipeline.backends.sqlite_job_store.SQLiteJobStore.

Validates Requirements 6.1, 6.2, 6.3, 6.4, 6.5:
- Implements JobStore interface using local SQLite
- Auto-creates database file and schema on first use
- Supports concurrent read access (WAL mode)
- Upserts on save with existing job_id
- Round-trip save/load equivalence
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import aiosqlite
import pytest

from pipeline.backends.sqlite_job_store import SQLiteJobStore
from pipeline.interfaces import JobStore


@pytest.fixture
def db_path(tmp_path: Path) -> str:
    """Return a temporary database path for test isolation."""
    return str(tmp_path / "test_jobs.db")


@pytest.fixture
def store(db_path: str) -> SQLiteJobStore:
    """Return a fresh SQLiteJobStore pointed at a temp database."""
    return SQLiteJobStore(db_path=db_path)


# -- ABC conformance --------------------------------------------------------


class TestABCConformance:
    """Requirement 6.1: SQLiteJobStore SHALL implement the JobStore interface."""

    def test_is_subclass_of_job_store(self) -> None:
        assert issubclass(SQLiteJobStore, JobStore)

    def test_instance_is_job_store(self, store: SQLiteJobStore) -> None:
        assert isinstance(store, JobStore)


# -- Auto-creation ----------------------------------------------------------


class TestAutoCreation:
    """Requirement 6.2: Auto-create database file and schema on first use."""

    @pytest.mark.asyncio
    async def test_creates_db_file_on_first_operation(self, db_path: str) -> None:
        store = SQLiteJobStore(db_path=db_path)
        assert not os.path.exists(db_path)
        await store.save("job-1", {"status": "running"})
        assert os.path.exists(db_path)
        await store.close()

    @pytest.mark.asyncio
    async def test_creates_parent_directories(self, tmp_path: Path) -> None:
        nested = str(tmp_path / "a" / "b" / "c" / "jobs.db")
        store = SQLiteJobStore(db_path=nested)
        await store.save("job-1", {"status": "done"})
        assert os.path.exists(nested)
        await store.close()

    @pytest.mark.asyncio
    async def test_creates_jobs_table(self, db_path: str) -> None:
        store = SQLiteJobStore(db_path=db_path)
        await store.save("job-1", {"x": 1})
        await store.close()

        async with aiosqlite.connect(db_path) as db:
            async with db.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='jobs'"
            ) as cur:
                row = await cur.fetchone()
        assert row is not None

    @pytest.mark.asyncio
    async def test_default_db_path(self) -> None:
        store = SQLiteJobStore()
        expected = os.path.join(Path.home(), ".biokg-ingest", "jobs.db")
        assert store._db_path == expected


# -- WAL mode ---------------------------------------------------------------


class TestWALMode:
    """Requirement 6.3: Concurrent read access via WAL mode."""

    @pytest.mark.asyncio
    async def test_wal_mode_enabled(self, db_path: str) -> None:
        store = SQLiteJobStore(db_path=db_path)
        await store.save("job-1", {"status": "running"})

        async with aiosqlite.connect(db_path) as db:
            async with db.execute("PRAGMA journal_mode;") as cur:
                row = await cur.fetchone()
        assert row is not None
        assert row[0].lower() == "wal"
        await store.close()


# -- save / load / delete / list_jobs ----------------------------------------


class TestSaveLoad:
    """Requirements 6.4, 6.5: Upsert semantics and round-trip equivalence."""

    @pytest.mark.asyncio
    async def test_save_and_load_basic(self, store: SQLiteJobStore) -> None:
        metadata = {"status": "running", "progress": 42}
        await store.save("job-1", metadata)
        loaded = await store.load("job-1")
        assert loaded == metadata
        await store.close()

    @pytest.mark.asyncio
    async def test_load_nonexistent_returns_none(self, store: SQLiteJobStore) -> None:
        result = await store.load("does-not-exist")
        assert result is None
        await store.close()

    @pytest.mark.asyncio
    async def test_save_upserts_existing(self, store: SQLiteJobStore) -> None:
        await store.save("job-1", {"status": "running"})
        await store.save("job-1", {"status": "completed", "result": "ok"})
        loaded = await store.load("job-1")
        assert loaded == {"status": "completed", "result": "ok"}
        await store.close()

    @pytest.mark.asyncio
    async def test_delete_removes_job(self, store: SQLiteJobStore) -> None:
        await store.save("job-1", {"status": "running"})
        await store.delete("job-1")
        assert await store.load("job-1") is None
        await store.close()

    @pytest.mark.asyncio
    async def test_delete_nonexistent_is_noop(self, store: SQLiteJobStore) -> None:
        # Should not raise
        await store.delete("ghost")
        await store.close()

    @pytest.mark.asyncio
    async def test_list_jobs_empty(self, store: SQLiteJobStore) -> None:
        jobs = await store.list_jobs()
        assert jobs == []
        await store.close()

    @pytest.mark.asyncio
    async def test_list_jobs_returns_all(self, store: SQLiteJobStore) -> None:
        await store.save("a", {"name": "alpha"})
        await store.save("b", {"name": "beta"})
        await store.save("c", {"name": "gamma"})
        jobs = await store.list_jobs()
        names = sorted(j["name"] for j in jobs)
        assert names == ["alpha", "beta", "gamma"]
        await store.close()

    @pytest.mark.asyncio
    async def test_round_trip_various_types(self, store: SQLiteJobStore) -> None:
        """Verify round-trip for different JSON-serializable value types."""
        metadata: dict[str, Any] = {
            "string": "hello",
            "integer": 42,
            "float": 3.14,
            "boolean": True,
            "null": None,
            "nested": {"a": [1, 2, 3]},
        }
        await store.save("typed", metadata)
        loaded = await store.load("typed")
        assert loaded == metadata
        await store.close()


# -- Lifecycle ---------------------------------------------------------------


class TestLifecycle:
    """Verify close() behaviour."""

    @pytest.mark.asyncio
    async def test_close_sets_db_to_none(self, store: SQLiteJobStore) -> None:
        await store.save("job-1", {"x": 1})
        assert store._db is not None
        await store.close()
        assert store._db is None

    @pytest.mark.asyncio
    async def test_close_idempotent(self, store: SQLiteJobStore) -> None:
        await store.close()
        await store.close()  # should not raise
