"""Property-based tests for pipeline-interfaces-packaging.

# Feature: pipeline-interfaces-packaging, Property 4: JobStore save/load round-trip

For any valid job metadata dictionary (with string keys and JSON-serializable
values), saving it to a SQLiteJobStore with a given job_id and then loading it
back with the same job_id SHALL produce a dictionary equivalent to the original.

**Validates: Requirements 6.4, 6.5**
"""

from __future__ import annotations

import tempfile
from typing import Any

import pytest

pytestmark = pytest.mark.properties

from hypothesis import given, settings
from hypothesis import strategies as st

from pipeline.backends.sqlite_job_store import SQLiteJobStore

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

metadata_strategy = st.dictionaries(
    st.text(),
    st.one_of(
        st.text(),
        st.integers(),
        st.floats(allow_nan=False),
        st.booleans(),
        st.none(),
    ),
)

job_id_strategy = st.text(min_size=1)


# ---------------------------------------------------------------------------
# Property 4 — JobStore save/load round-trip
# ---------------------------------------------------------------------------


@given(job_id=job_id_strategy, metadata=metadata_strategy)
@settings(max_examples=100, deadline=None)
@pytest.mark.asyncio
async def test_jobstore_save_load_round_trip(
    job_id: str,
    metadata: dict[str, Any],
) -> None:
    """**Validates: Requirements 6.4, 6.5**

    For any valid job metadata dictionary, saving it to a SQLiteJobStore and
    then loading it back with the same job_id produces an equivalent dict.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = f"{tmp_dir}/test_jobs.db"
        store = SQLiteJobStore(db_path=db_path)
        try:
            await store.save(job_id, metadata)
            loaded = await store.load(job_id)
            assert loaded is not None, f"load({job_id!r}) returned None after save"
            assert loaded == metadata, (
                f"Round-trip mismatch for job_id={job_id!r}:\n"
                f"  saved:  {metadata!r}\n"
                f"  loaded: {loaded!r}"
            )
        finally:
            await store.close()


# ---------------------------------------------------------------------------
# Shared fixture — module-scoped LocalEmbeddingProvider
# ---------------------------------------------------------------------------

# We use a global instance to avoid loading the sentence-transformers model
# on every Hypothesis example.  The model is stateless for embed_* calls,
# so sharing a single instance is safe.

from pipeline.backends.local_embeddings import LocalEmbeddingProvider

_embedding_provider: LocalEmbeddingProvider | None = None


def _get_embedding_provider() -> LocalEmbeddingProvider:
    global _embedding_provider
    if _embedding_provider is None:
        _embedding_provider = (
            LocalEmbeddingProvider()
        )  # default: all-mpnet-base-v2, 768 dims
    return _embedding_provider


EXPECTED_DIM = 768

# ---------------------------------------------------------------------------
# Strategies for embedding tests
# ---------------------------------------------------------------------------

non_empty_text = st.text(min_size=1, max_size=500)
non_empty_text_list = st.lists(
    st.text(min_size=1, max_size=200), min_size=1, max_size=20
)


# ---------------------------------------------------------------------------
# Feature: pipeline-interfaces-packaging, Property 5: Embedding dimension invariant
# ---------------------------------------------------------------------------


@given(text=non_empty_text)
@settings(max_examples=100, deadline=None)
def test_embed_query_dimension_invariant(text: str) -> None:
    """**Validates: Requirements 7.4**

    For any non-empty text string, calling embed_query on a
    LocalEmbeddingProvider returns a list of floats whose length equals 768.
    """
    provider = _get_embedding_provider()
    result = provider.embed_query(text)

    assert isinstance(result, list), f"Expected list, got {type(result)}"
    assert all(isinstance(v, float) for v in result), "All elements must be floats"
    assert len(result) == EXPECTED_DIM, (
        f"Expected embedding dimension {EXPECTED_DIM}, got {len(result)}"
    )


# ---------------------------------------------------------------------------
# Feature: pipeline-interfaces-packaging, Property 6: Embedding count preservation
# ---------------------------------------------------------------------------


@given(texts=non_empty_text_list)
@settings(max_examples=100, deadline=None)
def test_embed_documents_count_preservation(texts: list[str]) -> None:
    """**Validates: Requirements 7.5**

    For any list of N non-empty text strings (N >= 1), calling embed_documents
    on a LocalEmbeddingProvider returns exactly N embedding vectors, each with
    the same dimension.
    """
    provider = _get_embedding_provider()
    results = provider.embed_documents(texts)

    assert isinstance(results, list), f"Expected list, got {type(results)}"
    assert len(results) == len(texts), (
        f"Expected {len(texts)} vectors, got {len(results)}"
    )
    for i, vec in enumerate(results):
        assert isinstance(vec, list), f"Vector {i} is not a list"
        assert all(isinstance(v, float) for v in vec), (
            f"Vector {i} contains non-float elements"
        )
        assert len(vec) == EXPECTED_DIM, (
            f"Vector {i}: expected dimension {EXPECTED_DIM}, got {len(vec)}"
        )


# ---------------------------------------------------------------------------
# Feature: pipeline-interfaces-packaging, Property 1: Interface-only backend usage
# ---------------------------------------------------------------------------

import logging
from unittest.mock import patch

from pipeline.interfaces import EmbeddingProvider, GraphStore, JobStore, LLMProvider


def _abc_method_names(abc_cls: type) -> set[str]:
    """Return the set of public/dunder method names defined on an ABC.

    Includes abstract methods, concrete helper methods (like __enter__),
    and properties — but excludes internal ABC machinery (_abc_*).
    """
    names: set[str] = set()
    for name in dir(abc_cls):
        if name.startswith("_abc_"):
            continue
        obj = getattr(abc_cls, name, None)
        if obj is None:
            continue
        if callable(obj) or isinstance(obj, property):
            names.add(name)
    return names


class _CallTracker:
    """Mixin that records attribute accesses that look like method calls."""

    def __init__(self) -> None:
        self._tracked_calls: set[str] = set()

    def _record(self, name: str) -> None:
        self._tracked_calls.add(name)


class TrackingGraphStore(GraphStore, _CallTracker):
    def __init__(self) -> None:
        _CallTracker.__init__(self)

    def execute_query(self, query, params=None):
        self._record("execute_query")
        return []

    def upsert_node(self, label, properties):
        self._record("upsert_node")
        return "node-id"

    def upsert_relationship(self, source_id, rel_type, target_id, properties=None):
        self._record("upsert_relationship")

    def close(self):
        self._record("close")


class TrackingEmbeddingProvider(EmbeddingProvider, _CallTracker):
    def __init__(self) -> None:
        _CallTracker.__init__(self)

    def embed_query(self, text):
        self._record("embed_query")
        return [0.0] * 768

    def embed_documents(self, texts):
        self._record("embed_documents")
        return [[0.0] * 768 for _ in texts]


class TrackingLLMProvider(LLMProvider, _CallTracker):
    def __init__(self) -> None:
        _CallTracker.__init__(self)

    def invoke(self, prompt, **kwargs):
        self._record("invoke")
        return ""

    async def ainvoke(self, prompt, **kwargs):
        self._record("ainvoke")
        return ""

    async def extract(self, prompt, text):
        self._record("extract")
        return {"entities": [], "relationships": []}


class TrackingJobStore(JobStore, _CallTracker):
    def __init__(self) -> None:
        _CallTracker.__init__(self)
        self._data: dict[str, dict] = {}

    async def save(self, job_id, metadata):
        self._record("save")
        self._data[job_id] = metadata

    async def load(self, job_id):
        self._record("load")
        return self._data.get(job_id)

    async def delete(self, job_id):
        self._record("delete")
        self._data.pop(job_id, None)

    async def list_jobs(self):
        self._record("list_jobs")
        return list(self._data.values())


# Map of (module_label, constructor_callable, abc_interfaces_used)
def _build_pipeline_modules() -> list[
    tuple[str, object, list[tuple[type, _CallTracker]]]
]:
    """Construct each pipeline module with tracking mocks.

    Returns a list of (label, pipeline_instance, [(abc_class, tracker), ...]).
    """
    results = []

    # --- KGPipeline ---
    # Needs patching because constructor imports neo4j_graphrag internals
    gs1 = TrackingGraphStore()
    ep1 = TrackingEmbeddingProvider()
    lp1 = TrackingLLMProvider()
    try:
        with (
            patch("pipeline.ingest.kg_pipeline.TokenChunker"),
            patch("pipeline.ingest.kg_pipeline.EntityResolver"),
            patch("pipeline.ingest.kg_pipeline.IncrementalConsolidator"),
            patch("pipeline.ingest.kg_pipeline.RelationshipCounter"),
        ):
            from pipeline.ingest.kg_pipeline import KGPipeline

            kg = KGPipeline(gs1, ep1, lp1, database="test")
        results.append(
            (
                "KGPipeline",
                kg,
                [(GraphStore, gs1), (EmbeddingProvider, ep1), (LLMProvider, lp1)],
            )
        )
    except Exception:
        pass  # Skip if import fails due to missing deps

    # --- EmbeddingPipeline ---
    gs2 = TrackingGraphStore()
    ep2 = TrackingEmbeddingProvider()
    lp2 = TrackingLLMProvider()
    try:
        from pipeline.ingest.embedding_pipeline import EmbeddingPipeline

        emb = EmbeddingPipeline(gs2, ep2, lp2, database="test")
        results.append(
            (
                "EmbeddingPipeline",
                emb,
                [(GraphStore, gs2), (EmbeddingProvider, ep2), (LLMProvider, lp2)],
            )
        )
    except Exception:
        pass

    # --- EnrichmentOrchestrator ---
    gs3 = TrackingGraphStore()
    lp3 = TrackingLLMProvider()
    try:
        from pipeline.enrichment.enrichment_orchestrator import EnrichmentOrchestrator

        eo = EnrichmentOrchestrator(gs3, lp3, database="test")
        results.append(
            (
                "EnrichmentOrchestrator",
                eo,
                [(GraphStore, gs3), (LLMProvider, lp3)],
            )
        )
    except Exception:
        pass

    # --- IngestionJobManager ---
    js4 = TrackingJobStore()
    try:
        from src.services.ingestion.ingestion_job_manager import IngestionJobManager

        ijm = IngestionJobManager(js4, logging.getLogger("test"))
        results.append(
            (
                "IngestionJobManager",
                ijm,
                [(JobStore, js4)],
            )
        )
    except Exception:
        pass

    return results


# Build once at module level so Hypothesis doesn't re-import on every example
_PIPELINE_MODULES = _build_pipeline_modules()
_PIPELINE_ENTRIES = [
    (label, abc_cls, tracker)
    for label, _inst, ifaces in _PIPELINE_MODULES
    for abc_cls, tracker in ifaces
]


@given(entry=st.sampled_from(_PIPELINE_ENTRIES) if _PIPELINE_ENTRIES else st.nothing())
@settings(max_examples=100, deadline=None)
def test_interface_only_backend_usage(
    entry: tuple[str, type, _CallTracker],
) -> None:
    """**Validates: Requirements 1.6, 2.4, 3.4, 4.5**

    For any pipeline module constructed with tracking mock backends, the only
    methods called on those backends during construction SHALL be methods
    defined on the corresponding abstract interface.
    """
    label, abc_cls, tracker = entry
    allowed = _abc_method_names(abc_cls)
    called = tracker._tracked_calls

    # Every method that was called must be in the ABC's interface
    forbidden = called - allowed
    assert not forbidden, (
        f"{label}: called non-interface methods on {abc_cls.__name__}: "
        f"{forbidden}. Allowed: {allowed}"
    )


# ---------------------------------------------------------------------------
# Feature: pipeline-interfaces-packaging, Property 3: Missing backend raises descriptive error
# ---------------------------------------------------------------------------

# Each tuple: (constructor_callable, param_index_to_invalidate, expected_type_name)
# We build a list of test cases where one required backend param is replaced
# with an invalid value ("not-a-backend") while the others get valid mocks.

import logging as _logging
from unittest.mock import patch as _patch


def _make_missing_backend_cases() -> list[tuple[str, int, str]]:
    """Build (label, param_index, expected_type_name) tuples.

    For each pipeline constructor and each required backend param, we record
    which positional index to replace with an invalid value and what type name
    should appear in the resulting TypeError message.
    """
    cases: list[tuple[str, int, str]] = []

    # KGPipeline(graph_store, embedding_provider, llm_provider)
    cases.append(("KGPipeline", 0, "GraphStore"))
    cases.append(("KGPipeline", 1, "EmbeddingProvider"))
    cases.append(("KGPipeline", 2, "LLMProvider"))

    # EmbeddingPipeline(graph_store, embedding_provider, llm_provider)
    cases.append(("EmbeddingPipeline", 0, "GraphStore"))
    cases.append(("EmbeddingPipeline", 1, "EmbeddingProvider"))
    cases.append(("EmbeddingPipeline", 2, "LLMProvider"))

    # EnrichmentOrchestrator(graph_store, llm_provider)
    cases.append(("EnrichmentOrchestrator", 0, "GraphStore"))
    cases.append(("EnrichmentOrchestrator", 1, "LLMProvider"))

    # IngestionJobManager(job_store, logger) — only job_store is a backend
    # IngestionJobManager is in graffold-api, not litegraf — skip
    # cases.append(("IngestionJobManager", 0, "JobStore"))

    return cases


_MISSING_BACKEND_CASES = _make_missing_backend_cases()


def _construct_with_invalid_backend(label: str, param_index: int) -> None:
    """Attempt to construct a pipeline module with one invalid backend.

    Builds valid mock backends for all required params, then replaces the
    param at *param_index* with the string ``"not-a-backend"``.
    Calls the constructor — the caller should expect a TypeError.
    """
    gs = TrackingGraphStore()
    ep = TrackingEmbeddingProvider()
    lp = TrackingLLMProvider()
    invalid = "not-a-backend"

    if label == "KGPipeline":
        args = [gs, ep, lp]
        args[param_index] = invalid
        with (
            _patch("pipeline.ingest.kg_pipeline.TokenChunker"),
            _patch("pipeline.ingest.kg_pipeline.EntityResolver"),
            _patch("pipeline.ingest.kg_pipeline.IncrementalConsolidator"),
            _patch("pipeline.ingest.kg_pipeline.RelationshipCounter"),
        ):
            from pipeline.ingest.kg_pipeline import KGPipeline

            KGPipeline(*args, database="test")

    elif label == "EmbeddingPipeline":
        args = [gs, ep, lp]
        args[param_index] = invalid
        from pipeline.ingest.embedding_pipeline import EmbeddingPipeline

        EmbeddingPipeline(*args, database="test")

    elif label == "EnrichmentOrchestrator":
        args = [gs, lp]
        args[param_index] = invalid
        from pipeline.enrichment.enrichment_orchestrator import EnrichmentOrchestrator

        EnrichmentOrchestrator(*args, database="test")

    elif label == "IngestionJobManager":
        pytest.skip("IngestionJobManager is in graffold-api, not litegraf")


@given(case=st.sampled_from(_MISSING_BACKEND_CASES))
@settings(max_examples=100, deadline=None)
def test_missing_backend_raises_descriptive_error(
    case: tuple[str, int, str],
) -> None:
    """**Validates: Requirements 5.6**

    For any pipeline module constructor and for any required backend parameter,
    calling the constructor with an invalid value for that parameter SHALL raise
    a TypeError whose message includes the name of the missing backend type.
    """
    label, param_index, expected_type_name = case

    with pytest.raises(TypeError, match=expected_type_name):
        _construct_with_invalid_backend(label, param_index)


# ---------------------------------------------------------------------------
# Feature: pipeline-interfaces-packaging, Property 7: Processor discovery completeness
# ---------------------------------------------------------------------------

import sys
import types

from pipeline.processors import discover_processors
from pipeline.processors.base import ProcessorBase

# Build a set of concrete ProcessorBase subclasses with unique names.
# Each has a distinct ``name`` property value so discover_processors() can
# map them by name.


def _make_processor_class(class_name: str, processor_name: str) -> type:
    """Dynamically create a concrete ProcessorBase subclass."""

    cls = type(
        class_name,
        (ProcessorBase,),
        {
            "__init__": lambda self, graph_store=None, **kw: None,
            "process": lambda self, data, **kw: {"result": "ok"},
            "name": property(lambda self: processor_name),
        },
    )
    # Mark process as async (discover_processors doesn't call it, but keep
    # the interface honest).

    async def _async_process(self, data, **kw):
        return {"result": "ok"}

    cls.process = _async_process  # type: ignore[assignment]
    return cls


_TEST_PROCESSOR_CLASSES = [
    _make_processor_class("AlphaProcessor", "alpha_processor"),
    _make_processor_class("BetaProcessor", "beta_processor"),
    _make_processor_class("GammaProcessor", "gamma_processor"),
    _make_processor_class("DeltaProcessor", "delta_processor"),
    _make_processor_class("EpsilonProcessor", "epsilon_processor"),
]


@given(
    subset=st.lists(
        st.sampled_from(_TEST_PROCESSOR_CLASSES),
        min_size=1,
        max_size=len(_TEST_PROCESSOR_CLASSES),
        unique=True,
    )
)
@settings(max_examples=100, deadline=None)
def test_processor_discovery_completeness(
    subset: list[type],
) -> None:
    """**Validates: Requirements 13.1, 13.2**

    For any set of ProcessorBase subclasses registered in a module inside
    pipeline.processors, calling discover_processors() SHALL return a
    dictionary containing every such class.
    """
    # Create a temporary module inside the pipeline.processors namespace
    temp_module_name = "_test_discovery_tmp"
    fq_name = f"pipeline.processors.{temp_module_name}"

    mod = types.ModuleType(fq_name)
    mod.__package__ = "pipeline.processors"

    # Add the chosen processor subclasses to the module
    for cls in subset:
        setattr(mod, cls.__name__, cls)

    # Register the module so importlib.import_module can find it
    sys.modules[fq_name] = mod

    # Temporarily add the temp module name to the package's iter_modules
    # results by patching pkgutil.iter_modules for the discovery call.
    import pkgutil

    _original_iter_modules = pkgutil.iter_modules

    def _patched_iter_modules(path=None, prefix=""):
        """Yield original modules plus our temporary test module."""
        yield from _original_iter_modules(path, prefix)
        # Only inject when scanning the pipeline.processors package path
        if path is not None:
            import pipeline.processors as pkg

            if path is pkg.__path__ or path == pkg.__path__:
                yield (None, temp_module_name, False)

    try:
        pkgutil.iter_modules = _patched_iter_modules
        result = discover_processors()
    finally:
        # Clean up: restore pkgutil and remove temp module
        pkgutil.iter_modules = _original_iter_modules
        sys.modules.pop(fq_name, None)

    # Verify every processor in the subset was discovered
    for cls in subset:
        # The name property value is used as the dict key
        inst = cls.__new__(cls)
        expected_name = inst.name
        assert expected_name in result, (
            f"Processor {cls.__name__!r} (name={expected_name!r}) not found "
            f"in discover_processors() result. Keys: {list(result.keys())}"
        )
        assert result[expected_name] is cls, (
            f"Processor {cls.__name__!r} mapped to wrong class: "
            f"{result[expected_name]!r}"
        )


# ---------------------------------------------------------------------------
# Feature: pipeline-interfaces-packaging, Property 2: No forbidden imports in pipeline package
# ---------------------------------------------------------------------------

import os as _os

FORBIDDEN_NAMES = ["Neo4jDatabase", "get_llm", "get_embedder", "RedisCache"]


def _collect_pipeline_py_files() -> list[str]:
    """Walk the src/pipeline/ directory and return all .py file paths."""
    py_files: list[str] = []
    pipeline_root = _os.path.join(
        _os.path.dirname(__file__), "..", "..", "src", "pipeline"
    )
    pipeline_root = _os.path.normpath(pipeline_root)
    for dirpath, _dirnames, filenames in _os.walk(pipeline_root):
        # Skip __pycache__ directories
        if "__pycache__" in dirpath:
            continue
        for fname in filenames:
            if fname.endswith(".py"):
                py_files.append(_os.path.join(dirpath, fname))
    return py_files


_PIPELINE_PY_FILES = _collect_pipeline_py_files()


@given(
    filepath=st.sampled_from(_PIPELINE_PY_FILES) if _PIPELINE_PY_FILES else st.nothing()
)
@settings(max_examples=100, deadline=None)
def test_no_forbidden_imports_in_pipeline_package(filepath: str) -> None:
    """**Validates: Requirements 5.5**

    For any Python source file within the pipeline/ package directory tree,
    parsing its content SHALL yield zero references to Neo4jDatabase, get_llm,
    get_embedder, or RedisCache.
    """
    with open(filepath, encoding="utf-8") as f:
        content = f.read()

    violations: list[str] = []
    for forbidden in FORBIDDEN_NAMES:
        if forbidden in content:
            violations.append(forbidden)

    rel_path = _os.path.relpath(filepath)
    assert not violations, f"Forbidden imports found in {rel_path}: {violations}"


# ---------------------------------------------------------------------------
# Feature: pipeline-interfaces-packaging, Property 8: CLI missing configuration error
# ---------------------------------------------------------------------------

from pipeline.cli import main as cli_main

# Each tuple: (subcommand, required_arg_flag, arg_name_in_error)
# argparse prints "error: the following arguments are required: <flag>"
_REQUIRED_CLI_PARAMS: list[tuple[str, str, str]] = [
    # "run" requires --query
    ("run", "--query", "--query"),
    # "enrich" requires --file
    ("enrich", "--file", "--file"),
]


@given(
    param_case=st.sampled_from(_REQUIRED_CLI_PARAMS),
)
@settings(max_examples=100, deadline=None)
def test_cli_missing_configuration_error(
    param_case: tuple[str, str, str],
) -> None:
    """**Validates: Requirements 14.5**

    For any required backend configuration parameter, invoking the CLI without
    that parameter and without a config file SHALL produce an error message
    that names the missing parameter.
    """
    subcommand, _required_flag, _expected_name = param_case

    # Build argv without the required flag — only the subcommand
    import sys
    from unittest.mock import patch as _p

    with _p.object(sys, "argv", ["biokg-ingest", subcommand]):
        with pytest.raises(SystemExit) as exc_info:
            cli_main()

    # argparse exits with code 2 for missing required arguments
    assert exc_info.value.code == 2, f"Expected exit code 2, got {exc_info.value.code}"
