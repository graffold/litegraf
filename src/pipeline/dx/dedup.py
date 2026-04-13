"""Content-addressed deduplication using MD5 hashing.

Makes ``insert()`` idempotent — re-inserting the same text is a no-op.
The seen-IDs index is persisted to a JSON file so dedup survives restarts.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


class ContentDeduplicator:
    """MD5-based content deduplication tracker."""

    _INDEX_FILENAME = "dedup_index.json"

    def __init__(self, working_dir: str = "./litegraf_workdir") -> None:
        self._working_dir = working_dir
        self._index_path = os.path.join(working_dir, self._INDEX_FILENAME)
        self._seen: set[str] = set()
        self._load()

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
        """Reset the dedup index (in-memory and on disk)."""
        self._seen.clear()
        self._save()

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

    def _save(self) -> None:
        Path(self._working_dir).mkdir(parents=True, exist_ok=True)
        with open(self._index_path, "w") as f:
            json.dump(sorted(self._seen), f)
