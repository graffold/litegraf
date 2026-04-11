"""Unit tests for the litegraf DX layer.

Covers:
- 9.1: ContentDeduplicator
- 9.2: LLMCache and CachedLLMProvider
- 9.3: RateLimitedLLMProvider
- 9.4: BackendRegistry
- 9.5: LiteGraf.__post_init__
- 9.6: run_sync()
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import patch

import pytest

from pipeline.dx.cache import CachedLLMProvider, LLMCache
from pipeline.dx.dedup import ContentDeduplicator
from pipeline.dx.limiter import RateLimitedLLMProvider
from pipeline.dx.registry import BackendRegistry
from pipeline.dx.sync_utils import run_sync
from pipeline.interfaces import (
    EmbeddingProvider,
    GraphStore,
    JobStore,
    LLMProvider,
)

# ---------------------------------------------------------------------------
# Helpers: concrete mock implementations of the ABCs
# ---------------------------------------------------------------------------


class StubLLMProvider(LLMProvider):
    """Minimal concrete LLMProvider for testing."""

    def __init__(self, **kwargs: Any) -> None:
        self.invoke_calls: list[str] = []
        self.ainvoke_calls: list[str] = []
        self.extract_calls: list[tuple[str, str]] = []

    def invoke(self, prompt: str, **kwargs: Any) -> str:
        self.invoke_calls.append(prompt)
        return f"sync-response-for:{prompt}"

    async def ainvoke(self, prompt: str, **kwargs: Any) -> str:
        self.ainvoke_calls.append(prompt)
        return f"async-response-for:{prompt}"

    async def extract(self, prompt: str, text: str) -> dict[str, Any]:
        self.extract_calls.append((prompt, text))
        return {"entities": [], "relationships": []}


class StubGraphStore(GraphStore):
    """Minimal concrete GraphStore for testing."""

    def __init__(self, **kwargs: Any) -> None:
        self.closed = False

    def execute_query(
        self, query: str, params: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        return []

    def upsert_node(self, label: str, properties: dict[str, Any]) -> str:
        return properties.get("id", "node-1")

    def upsert_relationship(
        self,
        source_id: str,
        rel_type: str,
        target_id: str,
        properties: dict[str, Any] | None = None,
    ) -> None:
        pass

    def close(self) -> None:
        self.closed = True


class StubEmbeddingProvider(EmbeddingProvider):
    """Minimal concrete EmbeddingProvider for testing."""

    def __init__(self, **kwargs: Any) -> None:
        pass

    def embed_query(self, text: str) -> list[float]:
        return [0.1] * 384

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * 384 for _ in texts]


class StubJobStore(JobStore):
    """Minimal concrete JobStore for testing."""

    def __init__(self, **kwargs: Any) -> None:
        pass

    async def save(self, job_id: str, metadata: dict[str, Any]) -> None:
        pass

    async def load(self, job_id: str) -> dict[str, Any] | None:
        return None

    async def delete(self, job_id: str) -> None:
        pass

    async def list_jobs(self) -> list[dict[str, Any]]:
        return []


# ===========================================================================
# 9.1  ContentDeduplicator
# ===========================================================================


class TestContentDeduplicator:
    """Tests for pipeline.dx.dedup.ContentDeduplicator."""

    def test_compute_content_id_deterministic(self):
        """Same content always produces the same ID."""
        dedup = ContentDeduplicator.__new__(ContentDeduplicator)
        dedup._seen = set()
        id1 = dedup.compute_content_id("hello world")
        id2 = dedup.compute_content_id("hello world")
        assert id1 == id2

    def test_compute_content_id_prefix(self):
        """ID starts with the given prefix."""
        cid = ContentDeduplicator.compute_content_id("abc", prefix="chunk")
        assert cid.startswith("chunk-")

    def test_compute_content_id_different_content(self):
        """Different content produces different IDs."""
        id1 = ContentDeduplicator.compute_content_id("aaa")
        id2 = ContentDeduplicator.compute_content_id("bbb")
        assert id1 != id2

    def test_duplicate_detection(self, tmp_path):
        """mark_seen → is_duplicate returns True."""
        dedup = ContentDeduplicator(working_dir=str(tmp_path / "wd"))
        cid = dedup.compute_content_id("test content")
        assert not dedup.is_duplicate(cid)
        dedup.mark_seen(cid)
        assert dedup.is_duplicate(cid)

    def test_persistence_across_instances(self, tmp_path):
        """Dedup index survives a new instance pointing at the same dir."""
        wd = str(tmp_path / "wd")
        dedup1 = ContentDeduplicator(working_dir=wd)
        cid = dedup1.compute_content_id("persist me")
        dedup1.mark_seen(cid)

        dedup2 = ContentDeduplicator(working_dir=wd)
        assert dedup2.is_duplicate(cid)

    def test_clear_resets_index(self, tmp_path):
        """clear() removes all tracked IDs."""
        dedup = ContentDeduplicator(working_dir=str(tmp_path / "wd"))
        cid = dedup.compute_content_id("data")
        dedup.mark_seen(cid)
        assert dedup.is_duplicate(cid)

        dedup.clear()
        assert not dedup.is_duplicate(cid)


# ===========================================================================
# 9.2  LLMCache and CachedLLMProvider
# ===========================================================================


class TestLLMCache:
    """Tests for pipeline.dx.cache.LLMCache."""

    def test_cache_miss_returns_none(self, tmp_path):
        cache = LLMCache(cache_dir=str(tmp_path / "cache"))
        assert cache.get("never seen") is None

    def test_put_then_get(self, tmp_path):
        cache = LLMCache(cache_dir=str(tmp_path / "cache"))
        cache.put("prompt-1", "response-1")
        assert cache.get("prompt-1") == "response-1"

    def test_clear_wipes_cache(self, tmp_path):
        cache = LLMCache(cache_dir=str(tmp_path / "cache"))
        cache.put("p", "r")
        cache.clear()
        assert cache.get("p") is None

    def test_corrupted_cache_treated_as_miss(self, tmp_path):
        """Invalid JSON in a cache file → treated as miss, not an error."""
        cache = LLMCache(cache_dir=str(tmp_path / "cache"))
        cache.put("prompt", "good-response")
        # Corrupt the file
        key_path = cache._path(
            __import__("hashlib")
            .md5(("prompt" + str(sorted({}.items()))).encode())
            .hexdigest()
        )
        with open(key_path, "w") as f:
            f.write("NOT VALID JSON{{{")
        assert cache.get("prompt") is None


class TestCachedLLMProvider:
    """Tests for pipeline.dx.cache.CachedLLMProvider."""

    def test_cache_hit_skips_inner(self, tmp_path):
        """On cache hit, the inner LLM is NOT called."""
        cache = LLMCache(cache_dir=str(tmp_path / "cache"))
        inner = StubLLMProvider()
        cached = CachedLLMProvider(inner=inner, cache=cache)

        # First call → miss → inner called
        r1 = cached.invoke("hello")
        assert r1 == "sync-response-for:hello"
        assert len(inner.invoke_calls) == 1

        # Second call → hit → inner NOT called again
        r2 = cached.invoke("hello")
        assert r2 == "sync-response-for:hello"
        assert len(inner.invoke_calls) == 1  # still 1

    def test_cache_miss_calls_inner(self, tmp_path):
        cache = LLMCache(cache_dir=str(tmp_path / "cache"))
        inner = StubLLMProvider()
        cached = CachedLLMProvider(inner=inner, cache=cache)
        result = cached.invoke("new prompt")
        assert result == "sync-response-for:new prompt"
        assert len(inner.invoke_calls) == 1

    def test_wrap_produces_valid_llm_provider(self, tmp_path):
        """LLMCache.wrap() returns a CachedLLMProvider that is an LLMProvider."""
        cache = LLMCache(cache_dir=str(tmp_path / "cache"))
        inner = StubLLMProvider()
        wrapped = cache.wrap(inner)
        assert isinstance(wrapped, LLMProvider)
        assert isinstance(wrapped, CachedLLMProvider)

    @pytest.mark.asyncio
    async def test_ainvoke_cache_hit(self, tmp_path):
        cache = LLMCache(cache_dir=str(tmp_path / "cache"))
        inner = StubLLMProvider()
        cached = CachedLLMProvider(inner=inner, cache=cache)

        r1 = await cached.ainvoke("async prompt")
        assert r1 == "async-response-for:async prompt"
        assert len(inner.ainvoke_calls) == 1

        r2 = await cached.ainvoke("async prompt")
        assert r2 == "async-response-for:async prompt"
        assert len(inner.ainvoke_calls) == 1  # not called again


# ===========================================================================
# 9.3  RateLimitedLLMProvider
# ===========================================================================


class TestRateLimitedLLMProvider:
    """Tests for pipeline.dx.limiter.RateLimitedLLMProvider."""

    def test_implements_llm_provider_abc(self):
        inner = StubLLMProvider()
        rl = RateLimitedLLMProvider(inner=inner, max_concurrent=4)
        assert isinstance(rl, LLMProvider)

    def test_invoke_passthrough(self):
        """Sync invoke() delegates directly to inner without semaphore."""
        inner = StubLLMProvider()
        rl = RateLimitedLLMProvider(inner=inner, max_concurrent=2)
        result = rl.invoke("sync prompt")
        assert result == "sync-response-for:sync prompt"
        assert inner.invoke_calls == ["sync prompt"]

    @pytest.mark.asyncio
    async def test_ainvoke_passthrough(self):
        inner = StubLLMProvider()
        rl = RateLimitedLLMProvider(inner=inner, max_concurrent=4)
        result = await rl.ainvoke("async prompt")
        assert result == "async-response-for:async prompt"
        assert inner.ainvoke_calls == ["async prompt"]

    @pytest.mark.asyncio
    async def test_concurrency_bound(self):
        """At most max_concurrent calls execute on the inner LLM simultaneously."""
        max_concurrent = 2
        peak_concurrent = 0
        current_concurrent = 0

        class SlowLLM(LLMProvider):
            def invoke(self, prompt: str, **kw: Any) -> str:
                return "ok"

            async def ainvoke(self, prompt: str, **kw: Any) -> str:
                nonlocal peak_concurrent, current_concurrent
                current_concurrent += 1
                peak_concurrent = max(peak_concurrent, current_concurrent)
                await asyncio.sleep(0.05)
                current_concurrent -= 1
                return "ok"

            async def extract(self, prompt: str, text: str) -> dict[str, Any]:
                return {"entities": [], "relationships": []}

        inner = SlowLLM()
        rl = RateLimitedLLMProvider(inner=inner, max_concurrent=max_concurrent)

        # Fire 6 concurrent calls
        await asyncio.gather(*(rl.ainvoke(f"p{i}") for i in range(6)))
        assert peak_concurrent <= max_concurrent


# ===========================================================================
# 9.4  BackendRegistry
# ===========================================================================


class TestBackendRegistry:
    """Tests for pipeline.dx.registry.BackendRegistry."""

    def test_string_shorthand_resolution(self):
        """String shorthand 'neo4j' resolves via the lazy loader."""
        with patch("pipeline.dx.registry._lazy_neo4j", return_value=StubGraphStore):
            store = BackendRegistry.resolve_graph_store("neo4j")
            assert isinstance(store, GraphStore)

    def test_class_reference_resolution(self):
        """Passing a GraphStore subclass instantiates it."""
        store = BackendRegistry.resolve_graph_store(StubGraphStore)
        assert isinstance(store, StubGraphStore)

    def test_instance_passthrough(self):
        """Passing a live instance returns it unchanged."""
        instance = StubGraphStore()
        result = BackendRegistry.resolve_graph_store(instance)
        assert result is instance

    def test_unknown_shorthand_raises(self):
        """Unrecognised string shorthand raises ValueError."""
        with pytest.raises(ValueError, match="Unknown graph_store"):
            BackendRegistry.resolve_graph_store("postgres")

    def test_register_custom_shorthand(self):
        """register() adds a new shorthand that can be resolved."""
        BackendRegistry.register("graph_store", "stub", StubGraphStore)
        try:
            store = BackendRegistry.resolve_graph_store("stub")
            assert isinstance(store, StubGraphStore)
        finally:
            # Clean up so other tests aren't affected
            from pipeline.dx.registry import _GRAPH_STORES

            _GRAPH_STORES.pop("stub", None)

    def test_resolve_llm_string(self):
        """LLM shorthand 'ollama' resolves via lazy loader."""
        with patch("pipeline.dx.registry._lazy_ollama", return_value=StubLLMProvider):
            llm = BackendRegistry.resolve_llm("ollama")
            assert isinstance(llm, LLMProvider)

    def test_resolve_embedding_class(self):
        """Embedding class reference resolves correctly."""
        emb = BackendRegistry.resolve_embedding(StubEmbeddingProvider)
        assert isinstance(emb, EmbeddingProvider)

    def test_resolve_job_store_instance(self):
        """JobStore instance passes through."""
        inst = StubJobStore()
        result = BackendRegistry.resolve_job_store(inst)
        assert result is inst


# ===========================================================================
# 9.5  LiteGraf.__post_init__
# ===========================================================================


class TestLiteGrafPostInit:
    """Tests for LiteGraf construction (with mocked backends)."""

    def _make_litegraf(self, **overrides: Any):
        """Construct a LiteGraf with all backends mocked to stubs."""
        from pipeline.litegraf import LiteGraf

        defaults = dict(
            graph_store=StubGraphStore(),
            embedding=StubEmbeddingProvider(),
            llm=StubLLMProvider(),
            job_store=StubJobStore(),
            enable_cache=False,
            enable_dedup=False,
        )
        defaults.update(overrides)
        return LiteGraf(**defaults)

    def test_default_construction(self, tmp_path):
        """LiteGraf with stub instances constructs without error."""
        kg = self._make_litegraf(working_dir=str(tmp_path / "wd"))
        assert kg._graph is not None
        assert kg._llm is not None
        assert kg._embedder is not None

    def test_custom_kwargs(self, tmp_path):
        """Custom kwargs are reflected on the instance."""
        kg = self._make_litegraf(
            chunk_token_size=256,
            chunk_overlap_tokens=32,
            working_dir=str(tmp_path / "wd"),
        )
        assert kg.chunk_token_size == 256
        assert kg.chunk_overlap_tokens == 32

    def test_enable_cache_wraps_llm(self, tmp_path):
        """When enable_cache=True, the resolved LLM is a CachedLLMProvider (or wrapped further)."""
        kg = self._make_litegraf(
            enable_cache=True,
            cache_dir=str(tmp_path / "cache"),
            max_async_calls=0,
            working_dir=str(tmp_path / "wd"),
        )
        assert isinstance(kg._llm, CachedLLMProvider)

    def test_max_async_calls_wraps_llm(self, tmp_path):
        """When max_async_calls > 0, the resolved LLM is a RateLimitedLLMProvider."""
        kg = self._make_litegraf(
            enable_cache=False,
            max_async_calls=8,
            working_dir=str(tmp_path / "wd"),
        )
        assert isinstance(kg._llm, RateLimitedLLMProvider)

    def test_error_empty_content_insert(self, tmp_path):
        """insert('') raises ValueError."""
        kg = self._make_litegraf(working_dir=str(tmp_path / "wd"))
        with pytest.raises(ValueError, match="non-empty"):
            kg.insert("")

    def test_error_unknown_shorthand(self, tmp_path):
        """Unknown graph_store shorthand raises ValueError during construction."""
        from pipeline.litegraf import LiteGraf

        with pytest.raises(ValueError, match="Unknown graph_store"):
            LiteGraf(
                graph_store="postgres",
                embedding=StubEmbeddingProvider(),
                llm=StubLLMProvider(),
                job_store=StubJobStore(),
                working_dir=str(tmp_path / "wd"),
            )

    def test_repr_redacts_password(self, tmp_path):
        """__repr__ does not leak graph_password."""
        kg = self._make_litegraf(
            graph_password="super-secret",
            working_dir=str(tmp_path / "wd"),
        )
        r = repr(kg)
        assert "super-secret" not in r
        assert "****" in r

    def test_context_manager(self, tmp_path):
        """LiteGraf can be used as a context manager; close() is called on exit."""
        stub_graph = StubGraphStore()
        kg = self._make_litegraf(
            graph_store=stub_graph,
            working_dir=str(tmp_path / "wd"),
        )
        with kg:
            pass
        assert stub_graph.closed


# ===========================================================================
# 9.6  run_sync()
# ===========================================================================


class TestRunSync:
    """Tests for pipeline.dx.sync_utils.run_sync."""

    def test_basic_coroutine_execution(self):
        """run_sync executes a simple coroutine and returns its result."""

        async def add(a: int, b: int) -> int:
            return a + b

        assert run_sync(add(2, 3)) == 5

    def test_error_propagation(self):
        """Exceptions raised inside the coroutine propagate out."""

        async def boom() -> None:
            raise RuntimeError("kaboom")

        with pytest.raises(RuntimeError, match="kaboom"):
            run_sync(boom())

    def test_returns_complex_value(self):
        """run_sync can return non-trivial objects."""

        async def make_dict() -> dict[str, int]:
            return {"a": 1, "b": 2}

        result = run_sync(make_dict())
        assert result == {"a": 1, "b": 2}
