# litegraf Benchmark Suite

Biomedical NER benchmark comparing LLM providers on gold-standard PubMed datasets,
with optional end-to-end knowledge graph construction.

📊 **[Live Dashboard](https://thegraffy.github.io/litegraf/)** — interactive charts & leaderboard

## Prerequisites

```bash
# AWS SSO login (models span eu-north-1 and us-east-1)
aws sso login

# For KG build axis: local Neo4j running on bolt://localhost:7687
```

## Quick Start

```bash
cd litegraf

# Extraction benchmark (all models, no Neo4j needed)
./bench.sh --docs 10

# Full suite including KG construction (requires Neo4j)
BENCH_MAX_DOCS=10 \
NEO4J_DATABASE=benchmark \
NEO4J_PASSWORD=<your-password> \
PYTHONPATH=src python -m pipeline.benchmarks.run_benchmark --all
```

## Run Options

### Extraction Only (compare_providers)

Compares entity extraction quality across all models against gold annotations.

```bash
./bench.sh --docs 10
./bench.sh --docs 50 --dataset chemprot
./bench.sh --dataset gad --docs 10

# Or directly
PYTHONPATH=src python -m pipeline.benchmarks.compare_providers --docs 10
```

Set `BENCH_TIMEOUT=90` to adjust per-call timeout (default 60s).

### Full 5-Axis Suite (run_benchmark)

```bash
BENCH_MAX_DOCS=10 \
NEO4J_DATABASE=benchmark \
NEO4J_PASSWORD=<your-password> \
PYTHONPATH=src python -m pipeline.benchmarks.run_benchmark --all
```

Axes: extraction, kg-quality, kg-build, query, throughput.

Run individual axes:
```bash
PYTHONPATH=src python -m pipeline.benchmarks.run_benchmark --axis extraction
PYTHONPATH=src python -m pipeline.benchmarks.run_benchmark --axis kg-build
PYTHONPATH=src python -m pipeline.benchmarks.run_benchmark --axis kg-quality
```

### KG Build Axis

Constructs a real Neo4j knowledge graph for **each model** in the `MODELS` list:
1. Clears the graph
2. Inserts benchmark docs via `LiteGraf.insert()` (chunking → LLM extraction → Neo4j upsert)
3. Queries the built graph with sample biomedical questions
4. Reports per-model: entities, relationships, graph size, throughput, query latency

```bash
BENCH_MAX_DOCS=10 \
NEO4J_DATABASE=benchmark \
NEO4J_PASSWORD=<your-password> \
PYTHONPATH=src python -m pipeline.benchmarks.run_benchmark --axis kg-build
```

Neo4j environment variables:
| Variable | Default |
|----------|---------|
| `NEO4J_URI` | `bolt://localhost:7687` |
| `NEO4J_USER` | `neo4j` |
| `NEO4J_PASSWORD` | `password` |
| `NEO4J_DATABASE` | `litegrafbench` |

## Datasets

| Dataset | Docs | Entities | Description |
|---------|------|----------|-------------|
| `bc5cdr` (default) | 1,500 | Chemical, Disease | BioCreative V — drug/disease NER from PubMed |
| `chemprot` | 2,432 | Chemical, Gene/Protein | BioCreative VI — chemical-protein interactions |
| `gad` | 5,330 | Gene, Disease | Genetic Association Database — gene-disease links |

Downloaded automatically on first run.

## Configured Models

12 active models across 6 labs, all via Bedrock serverless:

| Label | Lab | Model ID | Region |
|-------|-----|----------|--------|
| nova-micro | Amazon | eu.amazon.nova-micro-v1:0 | eu-north-1 |
| nova-lite | Amazon | eu.amazon.nova-lite-v1:0 | eu-north-1 |
| claude-haiku-4.5 | Anthropic | us.anthropic.claude-haiku-4-5-20251001-v1:0 | us-east-1 |
| claude-sonnet-4 | Anthropic | us.anthropic.claude-sonnet-4-20250514-v1:0 | us-east-1 |
| deepseek-v3 | DeepSeek | deepseek.v3-v1:0 | eu-north-1 |
| deepseek-v3.2 | DeepSeek | deepseek.v3.2 | eu-north-1 |
| gpt-oss-20b | OpenAI | openai.gpt-oss-20b-1:0 | eu-north-1 |
| gpt-oss-120b | OpenAI | openai.gpt-oss-120b-1:0 | eu-north-1 |
| qwen3-32b | Qwen | qwen.qwen3-32b-v1:0 | eu-north-1 |
| qwen3-coder-30b | Qwen | qwen.qwen3-coder-30b-a3b-v1:0 | eu-north-1 |
| llama3.1-8b | Meta | us.meta.llama3-1-8b-instruct-v1:0 | us-east-1 |
| llama3.2-3b | Meta | us.meta.llama3-2-3b-instruct-v1:0 | us-east-1 |

## Modify Models

Edit `src/pipeline/benchmarks/compare_providers.py` → `MODELS` list:

```python
{"label": "my-model", "provider": "bedrock", "model": "the.model-id", "region": "us-east-1"},
```

Comment out to disable, copy a line to add.

## Other Providers

```bash
# Google Gemini
GOOGLE_API_KEY=AIza... python -m pipeline.benchmarks.compare_providers --providers google

# OpenRouter
OPENAI_API_KEY=sk-or-... python -m pipeline.benchmarks.compare_providers --providers openai

# Local Ollama
python -m pipeline.benchmarks.compare_providers --providers ollama
```

## Output

Leaderboard to terminal, full JSON to `src/pipeline/benchmarks/results/`.

```
Dataset: bc5cdr (10 docs, test split)
====================================================================================================
Model                     Provider       Exact F1   Partial F1     Time   Errs
----------------------------------------------------------------------------------------------------
deepseek-v3.2             bedrock          0.7500       0.6800      4.8s     0
nano-graphrag             competitor       0.6200       0.5900     12.3s     0
nova-lite                 bedrock          0.5600       0.5100      1.5s     0
====================================================================================================
```

## Competitor Comparison

nano-graphrag runs automatically if installed:

```bash
pip install nano-graphrag

# It will appear in the leaderboard alongside Bedrock models
./bench.sh --docs 10

# Or explicitly in the full suite
PYTHONPATH=src python -m pipeline.benchmarks.run_benchmark --all --competitors nano-graphrag
```

## Dashboard

Interactive GitHub Pages dashboard with charts for F1, precision/recall, latency, and KG stats.

```bash
# After running benchmarks, publish results to the dashboard
./publish_dashboard.sh

# Then push to GitHub — dashboard auto-updates at:
# https://thegraffy.github.io/litegraf/
```

To enable GitHub Pages: repo Settings → Pages → Source: "Deploy from a branch" → Branch: `main`, folder: `/docs`.

## Troubleshooting

| Error | Fix |
|-------|-----|
| `Token has expired` | `aws sso login` |
| `ResourceNotFoundException` | Enable model access in Bedrock console for that region |
| `Invocation of model ID ... not supported` | Use geo prefix (`us.` for us-east-1, `eu.` for eu-north-1) |
| `Read timeout` / slow model | Increase `BENCH_TIMEOUT` (default 60s) or comment out model |
| `Model use case details` | Submit Anthropic use case form in AWS console |
| Neo4j connection refused | Start Neo4j locally or check `NEO4J_URI` |
