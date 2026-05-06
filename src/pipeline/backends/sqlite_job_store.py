"""SQLite-backed job store using aiosqlite.

Provides a zero-configuration JobStore implementation that persists job
metadata as JSON in a local SQLite database.  WAL mode is enabled for
concurrent read access.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import aiosqlite

from pipeline.interfaces import JobStore

_DEFAULT_DB_PATH = os.path.join(Path.home(), ".litegraf", "jobs.db")

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS jobs (
    job_id   TEXT PRIMARY KEY,
    metadata TEXT NOT NULL
);
"""


class SQLiteJobStore(JobStore):
    """Async SQLite job store.

    Parameters
    ----------
    db_path:
        Path to the SQLite database file.  Defaults to
        ``~/.litegraf/jobs.db``.  The parent directory is created
        automatically if it does not exist.
    """

    def __init__(self, db_path: str | None = None) -> None:
        self._db_path = db_path or _DEFAULT_DB_PATH
        self._db: aiosqlite.Connection | None = None

    async def _ensure_db(self) -> aiosqlite.Connection:
        """Return an open connection, creating the DB/table on first call."""
        if self._db is not None:
            return self._db

        os.makedirs(os.path.dirname(self._db_path), exist_ok=True)
        self._db = await aiosqlite.connect(self._db_path)
        await self._db.execute("PRAGMA journal_mode=WAL;")
        await self._db.execute(_CREATE_TABLE_SQL)
        await self._db.commit()
        return self._db

    # -- JobStore interface ----------------------------------------------------

    async def save(self, job_id: str, metadata: dict[str, Any]) -> None:
        """Persist job metadata.  Upserts if *job_id* already exists."""
        db = await self._ensure_db()
        await db.execute(
            "INSERT OR REPLACE INTO jobs (job_id, metadata) VALUES (?, ?)",
            (job_id, json.dumps(metadata)),
        )
        await db.commit()

    async def load(self, job_id: str) -> dict[str, Any] | None:
        """Load job metadata by ID.  Returns ``None`` if not found."""
        db = await self._ensure_db()
        async with db.execute(
            "SELECT metadata FROM jobs WHERE job_id = ?", (job_id,)
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            return None
        return json.loads(row[0])

    async def delete(self, job_id: str) -> None:
        """Remove persisted job state."""
        db = await self._ensure_db()
        await db.execute("DELETE FROM jobs WHERE job_id = ?", (job_id,))
        await db.commit()

    async def list_jobs(self) -> list[dict[str, Any]]:
        """Return all persisted job metadata dictionaries."""
        db = await self._ensure_db()
        async with db.execute("SELECT metadata FROM jobs") as cursor:
            rows = await cursor.fetchall()
        return [json.loads(row[0]) for row in rows]

    # -- Lifecycle -------------------------------------------------------------

    async def close(self) -> None:
        """Close the underlying database connection."""
        if self._db is not None:
            await self._db.close()
            self._db = None
