# Competitor Runners

Adapter modules that wrap third-party GraphRAG systems so they can be driven
by the same benchmark datasets and metrics used for Graffold.

## LightRAG

**Package**: [`lightrag-hku`](https://pypi.org/project/lightrag-hku/) (MIT license)
**Paper**: [LightRAG: Simple and Fast Retrieval-Augmented Generation](https://arxiv.org/abs/2410.05779) (EMNLP 2025)

### Setup

```bash
# Install the LightRAG core library
pip install lightrag-hku

# Or with uv
uv pip install lightrag-hku
```

LightRAG requires an LLM and an embedding model. The simplest option is
OpenAI (set `OPENAI_API_KEY`), but Ollama, Azure, Gemini, and HuggingFace
are also supported. See the
[LightRAG docs](https://github.com/HKUDS/LightRAG/blob/main/docs/ProgramingWithCore.md)
for provider configuration.

### Verify installation

```bash
python -m benchmarks.competitors.lightrag_runner
# Prints "LightRAG is available ✓" if installed, or a skip message otherwise.
```

### Usage in benchmarks

The adapter skips gracefully when `lightrag-hku` is not installed — the
benchmark runner will report the competitor as unavailable rather than
crashing.

```python
from benchmarks.competitors.lightrag_runner import is_available, run_extraction

if is_available():
    results = run_extraction(gold_examples)
```

### Running the same datasets

LightRAG uses the same `BenchmarkExample` model as Graffold, so any dataset
from `benchmarks/datasets/` can be passed directly:

```python
from benchmarks.datasets.loader import load_dataset
from benchmarks.competitors.lightrag_runner import is_available, run_extraction
from benchmarks.metrics.extraction_metrics import evaluate_extraction

dataset = load_dataset("bc5cdr")
gold = dataset.splits["test"].examples

if is_available():
    result = run_extraction(gold, llm_model_func=my_llm, embedding_func=my_embed)
    scores = evaluate_extraction(gold, result["predictions"])
```

### Supported benchmark axes

| Axis | Function | Notes |
|------|----------|-------|
| Extraction | `run_extraction()` | Inserts texts, queries for entities, parses answer |
| Query | `run_query()` | Indexes texts, answers questions, returns timing |

### LLM / Embedding injection

Both `run_extraction()` and `run_query()` accept optional `llm_model_func`
and `embedding_func` keyword arguments. When omitted, LightRAG uses its
defaults (OpenAI `gpt-4o-mini` + `text-embedding-3-small`).

```python
from lightrag.llm.ollama import ollama_model_complete, ollama_embed

result = run_extraction(
    examples,
    llm_model_func=ollama_model_complete,
    embedding_func=ollama_embed,
)
```

---

## nano-graphrag

**Package**: [`nano-graphrag`](https://pypi.org/project/nano-graphrag/) (MIT license)
**Repo**: [gusye1234/nano-graphrag](https://github.com/gusye1234/nano-graphrag) — a small, hackable GraphRAG implementation (~1100 LOC)

### Setup

```bash
# Install nano-graphrag
pip install nano-graphrag

# Or with uv
uv pip install nano-graphrag
```

nano-graphrag requires an LLM and an embedding model. The simplest option is
OpenAI (set `OPENAI_API_KEY`), but Ollama, Amazon Bedrock, and custom
functions are also supported. See the
[nano-graphrag examples](https://github.com/gusye1234/nano-graphrag/tree/main/examples)
for provider configuration.

### Verify installation

```bash
python -m benchmarks.competitors.nano_graphrag_runner
# Prints "nano-graphrag is available ✓" if installed, or a skip message otherwise.
```

### Usage in benchmarks

The adapter skips gracefully when `nano-graphrag` is not installed — the
benchmark runner will report the competitor as unavailable rather than
crashing.

```python
from benchmarks.competitors.nano_graphrag_runner import is_available, run_extraction

if is_available():
    results = run_extraction(gold_examples)
```

### Running the same datasets

nano-graphrag uses the same `BenchmarkExample` model as Graffold, so any
dataset from `benchmarks/datasets/` can be passed directly:

```python
from benchmarks.datasets.loader import load_dataset
from benchmarks.competitors.nano_graphrag_runner import is_available, run_extraction
from benchmarks.metrics.extraction_metrics import evaluate_extraction

dataset = load_dataset("bc5cdr")
gold = dataset.splits["test"].examples

if is_available():
    result = run_extraction(gold)
    scores = evaluate_extraction(gold, result["predictions"])
```

### Supported benchmark axes

| Axis | Function | Notes |
|------|----------|-------|
| Extraction | `run_extraction()` | Inserts texts, queries for entities, parses answer |
| Query | `run_query()` | Indexes texts, answers questions, returns timing |

### LLM / Embedding injection

Both `run_extraction()` and `run_query()` accept optional `best_model_func`,
`cheap_model_func`, and `embedding_func` keyword arguments. When omitted,
nano-graphrag uses its defaults (OpenAI `gpt-4o` / `gpt-4o-mini` +
`text-embedding-3-small`).

```python
result = run_extraction(
    examples,
    best_model_func=my_llm_complete,
    cheap_model_func=my_cheap_llm,
    embedding_func=my_embed_func,
)
```

---

## Microsoft GraphRAG

**Package**: [`graphrag`](https://pypi.org/project/graphrag/) (MIT license)
**Paper**: [From Local to Global: A Graph RAG Approach to Query-Focused Summarization](https://arxiv.org/abs/2404.16130) (arXiv:2404.16130)
**Repo**: [microsoft/graphrag](https://github.com/microsoft/graphrag)

### Published results (always available)

MS GraphRAG's full indexing pipeline is heavyweight (Leiden community
detection → community summaries → Parquet artifacts). The adapter provides
**published benchmark numbers** from the paper without requiring installation:

```python
from benchmarks.competitors.ms_graphrag_runner import get_published_results

results = get_published_results()
# Includes global/local search win rates from the paper
```

### Setup (optional — for local runs)

```bash
# Install the graphrag package
pip install graphrag

# Or with uv
uv pip install graphrag
```

MS GraphRAG requires an OpenAI-compatible LLM and embedding model. See the
[graphrag docs](https://microsoft.github.io/graphrag/) for configuration.

### Verify installation

```bash
python -m benchmarks.competitors.ms_graphrag_runner
# Always prints published results; also shows local availability status.
```

### Usage in benchmarks

Published results are always available. Local runs require the `graphrag`
package — the adapter skips gracefully when not installed.

```python
from benchmarks.competitors.ms_graphrag_runner import (
    is_available,
    get_published_results,
    get_unsupported_capabilities,
    run_extraction,
)

# Published numbers (always works)
published = get_published_results()

# N/A capabilities
unsupported = get_unsupported_capabilities()

# Local run (requires graphrag)
if is_available():
    result = run_extraction(gold_examples)
```

### Supported benchmark axes

| Axis | Function | Notes |
|------|----------|-------|
| Extraction | `run_extraction()` | Published capabilities; local run placeholder |
| Query | `run_query()` | Published win rates from paper; local run placeholder |
| Published | `get_published_results()` | Always available — paper numbers |
| N/A | `get_unsupported_capabilities()` | Features MS GraphRAG lacks |

### N/A capabilities

MS GraphRAG does not support:
- Incremental updates (requires full re-indexing)
- Streaming responses
- Provenance tracking
- Contradiction detection
- Temporal validity windows
- Statistical edges
- Context-aware query routing
