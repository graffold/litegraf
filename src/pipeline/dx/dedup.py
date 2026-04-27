"""Content-addressed deduplication using hash-based tracking.

Chunk-level: MD5 hash makes ``insert()`` idempotent — re-inserting the same text is a no-op.
Source-level: SHA256 hash skips entire files that haven't changed, avoiding chunking + LLM costs.

Both indices are persisted to JSON files so dedup survives restarts.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


class ContentDeduplicator:
    """Hash-based content deduplication tracker with chunk-level and source-level layers."""

    _INDEX_FILENAME = "dedup_index.json"
    _SOURCE_INDEX_FILENAME = "source_dedup_index.json"

    def __init__(self, working_dir: str = "./litegraf_workdir") -> None:
        self._working_dir = working_dir
        self._index_path = os.path.join(working_dir, self._INDEX_FILENAME)
        self._source_index_path = os.path.join(working_dir, self._SOURCE_INDEX_FILENAME)
        self._seen: set[str] = set()
        self._source_seen: dict[str, dict] = {}
        self._load()

    # -- source-level API (SHA256) -------------------------------------------

    @staticmethod
    def compute_source_hash(content: str | bytes) -> str:
        """Return ``'src-{sha256hex}'`` for the full source content."""
        data = content.encode("utf-8") if isinstance(content, str) else content
        return f"src-{hashlib.sha256(data).hexdigest()}"

    def is_source_duplicate(self, source_hash: str) -> bool:
        """Check whether this source file has been ingested before."""
        return source_hash in self._source_seen

    def mark_source_seen(self, source_hash: str, metadata: dict | None = None) -> None:
        """Record a source file as ingested with optional metadata."""
        self._source_seen[source_hash] = metadata or {}
        self._save_source_index()

    def remove_source(self, source_hash: str) -> None:
        """Remove a source from the index so it can be re-ingested."""
        self._source_seen.pop(source_hash, None)
        self._save_source_index()

    def get_source_metadata(self, source_hash: str) -> dict | None:
        """Return stored metadata for a source, or None."""
        return self._source_seen.get(source_hash)

    # -- public API ----------------------------------------------------------

    @staticmethod
    def compute_content_id(content: str, prefix: str = "doc") -> str:
        """Return a deterministic ID: ``'{prefix}-{md5hex}'``."""
        md5_hex = hashlib.md5(content.encode("utf-8")).hexdigest()
        return f"{prefix}-{md5_hex}"

    def is_duplicate(self, content_id: str) -> bool:
        """Check whether *content_id* has been seen before."""
        return content_id in self._seen

    def mark_seen(self, content_id: str) -> None:
        """Record *content_id* as processed and persist to disk."""
        self._seen.add(content_id)
        self._save()

    def remove_seen(self, content_id: str) -> None:
        """Remove *content_id* from the index so it can be re-inserted."""
        self._seen.discard(content_id)
        self._save()

    def clear(self) -> None:
        """Reset both dedup indices (in-memory and on disk)."""
        self._seen.clear()
        self._source_seen.clear()
        self._save()
        self._save_source_index()

    # -- persistence ---------------------------------------------------------

    def _load(self) -> None:
        if os.path.exists(self._index_path):
            try:
                with open(self._index_path) as f:
                    self._seen = set(json.load(f))
                logger.debug(
                    "Loaded %d dedup entries from %s", len(self._seen), self._index_path
                )
            except (json.JSONDecodeError, TypeError):
                logger.warning(
                    "Corrupted dedup index at %s — starting fresh", self._index_path
                )
                self._seen = set()
        if os.path.exists(self._source_index_path):
            try:
                with open(self._source_index_path) as f:
                    self._source_seen = json.load(f)
                logger.debug(
                    "Loaded %d source dedup entries from %s",
                    len(self._source_seen), self._source_index_path,
                )
            except (json.JSONDecodeError, TypeError):
                logger.warning(
                    "Corrupted source dedup index at %s — starting fresh",
                    self._source_index_path,
                )
                self._source_seen = {}

    def _save(self) -> None:
        Path(self._working_dir).mkdir(parents=True, exist_ok=True)
        with open(self._index_path, "w") as f:
            json.dump(sorted(self._seen), f)

    def _save_source_index(self) -> None:
        Path(self._working_dir).mkdir(parents=True, exist_ok=True)
        with open(self._source_index_path, "w") as f:
            json.dump(self._source_seen, f)
