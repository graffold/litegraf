# Contributing to the Benchmark Suite

This guide covers how to extend the benchmark suite with new datasets and competitor runners.

## Module Structure

```
benchmarks/
├── __init__.py
├── run_benchmark.py              # CLI entry point (--all, --axis, --competitors)
├── results/                      # Timestamped JSON output from runs
├── datasets/
│   ├── models.py                 # Shared data models (Entity, Relation, BenchmarkExample, BenchmarkDataset)
│   ├── loader.py                 # Registry + unified download/load CLI
│   ├── bc5cdr.py                 # Example: BC5CDR dataset module
│   ├── chemprot.py               # Example: ChemProt dataset module
│   ├── gad.py                    # Example: GAD dataset module
│   └── data/                     # Downloaded data (gitignored)
├── competitors/
│   ├── __init__.py
│   ├── lightrag_runner.py        # Example: LightRAG adapter
│   ├── nano_graphrag_runner.py   # Example: nano-graphrag adapter
│   ├── ms_graphrag_runner.py     # Example: MS GraphRAG adapter
│   └── README.md                 # Setup instructions per competitor
├── generators/                   # Synthetic dataset generators
│   ├── consolidation_stress.py
│   ├── contradiction_pairs.py
│   └── provenance_annotator.py
└── metrics/                      # Evaluation modules
    ├── extraction_metrics.py
    ├── kg_quality_metrics.py
    ├── query_metrics.py
    └── throughput_metrics.py
```

---

## Adding a New Dataset

Each dataset is a Python module under `benchmarks/datasets/` that exposes two functions: `download()` and `load()`.

### Step 1: Create the module

Create `benchmarks/datasets/your_dataset.py`:

```python
"""Your Dataset Name — brief description of the task.

Source: https://example.com/dataset
"""

from __future__ import annotations

import logging
from pathlib import Path

from .models import (
    BenchmarkDataset,
    BenchmarkExample,
    BenchmarkSplit,
    Entity,
    Relation,
)

logger = logging.getLogger(__name__)


def download(target_dir: str) -> None:
    """Download dataset files to target_dir. Skip if already present."""
    out = Path(target_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Use a sentinel file to detect prior download
    if (out / "test.tsv").exists():
        logger.info("Already downloaded: %s", out)
        return

    # Download and extract here...


def load(data_dir: str) -> BenchmarkDataset:
    """Load dataset from data_dir into BenchmarkDataset."""
    root = Path(data_dir)

    # Parse files into BenchmarkExample instances...
    examples = []

    return BenchmarkDataset(
        name="your_dataset",
        task="ner",  # or "re", "binary_re"
        entity_types=["Chemical", "Disease"],
        relation_types=["CID"],
        splits={"test": BenchmarkSplit(name="test", examples=examples)},
    )
```

### Step 2: Register in the loader

Edit `benchmarks/datasets/loader.py`:

```python
from . import bc5cdr, chemprot, gad, your_dataset  # add import

REGISTRY: dict[str, _DatasetEntry] = {
    # ... existing entries ...
    "your_dataset": {
        "module": your_dataset,
        "dir": "your_dataset",
        "description": "Your Dataset — brief description",
    },
}
```

### Step 3: Add a README (optional but recommended)

Create `benchmarks/datasets/your_dataset/README.md` with source links, citation, and format documentation.

### Data models reference

All datasets use shared models from `benchmarks/datasets/models.py`:

| Model | Purpose |
|-------|---------|
| `Entity` | Named entity mention (text, type, start/end offsets, optional mesh_id) |
| `Relation` | Relation between two entities (head, tail, relation_type) |
| `BenchmarkExample` | Single document with entities, relations, and optional label |
| `BenchmarkSplit` | Named split (train/dev/test) containing examples |
| `BenchmarkDataset` | Complete dataset with task type, entity/relation types, and splits |

### Conventions

- `download()` must be idempotent (skip if data exists)
- `load()` returns a `BenchmarkDataset` dataclass
- Downloaded data goes under `benchmarks/datasets/data/<name>/` (gitignored)
- Use `logging` for progress messages, not `print`

---

## Adding a New Competitor Runner

Each competitor is a module under `benchmarks/competitors/` that exposes three functions: `is_available()`, `run_extraction()`, and `run_query()`.

### Step 1: Create the module

Create `benchmarks/competitors/your_competitor_runner.py`:

```python
"""Your Competitor adapter for the Graffold benchmark suite.

Wraps the your-competitor library so it can be driven by the same
datasets and metrics used for Graffold. Skips gracefully when the
package is not installed.

CLI (smoke-test)::

    python -m benchmarks.competitors.your_competitor_runner
"""

from __future__ import annotations

import logging
import tempfile
import time
from typing import Any

from benchmarks.datasets.models import BenchmarkExample, Entity

logger = logging.getLogger(__name__)

_AVAILABLE: bool | None = None


def is_available() -> bool:
    """Return True if the competitor package is importable."""
    global _AVAILABLE
    if _AVAILABLE is None:
        try:
            import your_competitor  # noqa: F401
            _AVAILABLE = True
        except ImportError:
            _AVAILABLE = False
    return _AVAILABLE


def _require() -> None:
    if not is_available():
        msg = "your-competitor is not installed. Install with: pip install your-competitor"
        raise ImportError(msg)


def run_extraction(
    examples: list[BenchmarkExample],
    **kwargs: Any,
) -> list[list[Entity]]:
    """Run entity extraction on examples. Returns predicted entities per example."""
    _require()
    # Index texts, query for entities, parse results into Entity objects
    ...


def run_query(
    texts: list[str],
    questions: list[str],
    **kwargs: Any,
) -> dict[str, Any]:
    """Index texts and answer questions. Returns answers + timing."""
    _require()
    results: list[str] = []
    start = time.perf_counter()
    # Index and query...
    elapsed = time.perf_counter() - start
    return {"answers": results, "elapsed_seconds": elapsed}


if __name__ == "__main__":
    print(f"Available: {is_available()}")
```

### Step 2: Document setup

Add a section to `benchmarks/competitors/README.md` with install commands and verification steps.

### Interface contract

| Function | Signature | Purpose |
|----------|-----------|---------|
| `is_available()` | `() -> bool` | Import check — returns False when package missing |
| `run_extraction()` | `(examples, **kwargs) -> list[list[Entity]]` | NER benchmark adapter |
| `run_query()` | `(texts, questions, **kwargs) -> dict` | Query benchmark adapter |

### Conventions

- Use `is_available()` with a cached global bool for the import check
- `_require()` raises `ImportError` with install instructions when package is missing
- Use temp directories for working state (auto-cleaned)
- Accept optional LLM/embedding function injection via `**kwargs`
- Skip gracefully when the package is not installed — never crash the benchmark runner
- Async libraries need `asyncio.run()` wrappers for the sync interface

---

## Extension Points

| Extension point | Location | How to extend |
|-----------------|----------|---------------|
| Datasets | `benchmarks/datasets/` | New module + register in `loader.py` REGISTRY |
| Competitors | `benchmarks/competitors/` | New module implementing `is_available`/`run_extraction`/`run_query` |
| Metrics | `benchmarks/metrics/` | New module with evaluation functions + CLI `__main__` block |
| Generators | `benchmarks/generators/` | New module with `generate(seed=42)` function + `main()` CLI |
| Benchmark axes | `benchmarks/run_benchmark.py` | Add runner function to `AXIS_RUNNERS` dict |

---

## Running Quality Checks

```bash
# Type checking
uv run mypy benchmarks/

# Linting + formatting
uv run ruff check benchmarks/
uv run ruff format benchmarks/

# Run the full benchmark suite
uv run python benchmarks/run_benchmark.py --all
```
