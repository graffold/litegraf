# litegraf Benchmark Suite

Component-level benchmarks for the litegraf knowledge graph pipeline.
No graph database required — benchmarks the LLM extraction, quality metrics,
query capabilities, and throughput independently.

📊 **[Live Dashboard](https://thegraffy.github.io/litegraf/)** — interactive charts & leaderboard

## Prerequisites

```bash
# AWS SSO login (models span eu-north-1 and us-east-1)
aws sso login
```

## Quick Start

```bash
cd litegraf

# Extraction benchmark (all models)
./bench.sh --docs 10

# Full 4-axis suite
BENCH_MAX_DOCS=10 PYTHONPATH=src python -m pipeline.benchmarks.run_benchmark --all

# Or use the interactive TUI
uv run litegraf-tui
```

## Benchmark Axes

| Axis | What it tests | Needs Neo4j? |
|------|--------------|:------------:|
| `extraction` | LLM NER quality against gold-standard datasets | No |
| `kg-quality` | Entity consolidation, contradiction detection, provenance validation | No |
| `query` | Mode routing accuracy, answer relevance, latency | No |
| `throughput` | Extraction speed (docs/min), embedding rate, memory, token cost | No |

> **Note:** End-to-end KG construction benchmarks (insert → graph → query) live in
> [graffold-api](https://github.com/graffold/graffold-api), which uses these
> pre-validated components to build and benchmark the full stack.

### Run Options

```bash
# Extraction only (compare_providers — model leaderboard)
./bench.sh --docs 10
./bench.sh --docs 50 --dataset chemprot

# Full suite
PYTHONPATH=src python -m pipeline.benchmarks.run_benchmark --all

# Individual axes
PYTHONPATH=src python -m pipeline.benchmarks.run_benchmark --axis extraction
PYTHONPATH=src python -m pipeline.benchmarks.run_benchmark --axis kg-quality
PYTHONPATH=src python -m pipeline.benchmarks.run_benchmark --axis query
PYTHONPATH=src python -m pipeline.benchmarks.run_benchmark --axis throughput

# With competitor comparison
PYTHONPATH=src python -m pipeline.benchmarks.run_benchmark --all --competitors nano-graphrag lightrag
```

Set `BENCH_TIMEOUT=90` to adjust per-call timeout (default 60s).

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

Edit `src/pipeline/benchmarks/compare_providers.py` → `MODELS` list to add/remove.

## Other Providers

```bash
# Google Gemini
GOOGLE_API_KEY=AIza... python -m pipeline.benchmarks.compare_providers --providers google

# OpenRouter
OPENAI_API_KEY=sk-or-... python -m pipeline.benchmarks.compare_providers --providers openai

# Local Ollama
python -m pipeline.benchmarks.compare_providers --providers ollama

# Cloudflare Workers AI
CF_ACCOUNT_ID=... CF_API_TOKEN=... python -m pipeline.benchmarks.compare_providers --providers cloudflare
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
./bench.sh --docs 10
```

## Dashboard

Interactive GitHub Pages dashboard with charts for F1, precision/recall, latency.

```bash
./publish_dashboard.sh
# Dashboard at: https://thegraffy.github.io/litegraf/
```

## Troubleshooting

| Error | Fix |
|-------|-----|
| `Token has expired` | `aws sso login` |
| `ResourceNotFoundException` | Enable model access in Bedrock console for that region |
| `Invocation of model ID ... not supported` | Use geo prefix (`us.` for us-east-1, `eu.` for eu-north-1) |
| `Read timeout` / slow model | Increase `BENCH_TIMEOUT` (default 60s) or comment out model |
| `Model use case details` | Submit Anthropic use case form in AWS console |
