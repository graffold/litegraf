# litegraf Benchmark Suite

Biomedical NER benchmark comparing LLM providers on gold-standard PubMed datasets.

## Prerequisites

```bash
# AWS SSO login (all models run via Bedrock in eu-north-1)
aws sso login
```

## Run

```bash
cd litegraf    # the litegraf repo root (contains src/, tests/, pyproject.toml)

# Easiest way
./bench.sh --docs 10

# Or with PYTHONPATH
PYTHONPATH=src python -m pipeline.benchmarks.compare_providers --docs 10

# Or install the package first (then no PYTHONPATH needed)
pip install -e .
python -m pipeline.benchmarks --docs 10
```

More examples:
```bash
./bench.sh --docs 50
./bench.sh --dataset chemprot --docs 10
./bench.sh --dataset gad --docs 10
```

## What It Does

1. Downloads a gold-standard NER dataset (PubMed abstracts with expert-annotated entities)
2. Sends each abstract to every configured LLM with an "extract entities" prompt
3. Compares LLM output to gold annotations
4. Prints a leaderboard sorted by F1 score

## Datasets

| Dataset | Docs | Entities | What it is |
|---------|------|----------|------------|
| `bc5cdr` (default) | 1,500 | Chemical, Disease | BioCreative V — drug/disease NER from PubMed |
| `chemprot` | 2,432 | Chemical, Gene/Protein | BioCreative VI — chemical-protein interactions |
| `gad` | 5,330 | Gene, Disease | Genetic Association Database — gene-disease links |

Downloaded automatically on first run.

## Configured Models

All run via Bedrock serverless in **eu-north-1**. 12 models, 6 labs:

| Label | Lab | Model ID |
|-------|-----|----------|
| nova-micro | Amazon | eu.amazon.nova-micro-v1:0 |
| nova-lite | Amazon | eu.amazon.nova-lite-v1:0 |
| claude-haiku-4.5 | Anthropic | eu.anthropic.claude-haiku-4-5-20251001-v1:0 |
| claude-sonnet-4 | Anthropic | eu.anthropic.claude-sonnet-4-20250514-v1:0 |
| deepseek-v3 | DeepSeek | deepseek.v3-v1:0 |
| deepseek-v3.2 | DeepSeek | deepseek.v3.2 |
| minimax-m2.1 | MiniMax | minimax.minimax-m2.1 |
| minimax-m2.5 | MiniMax | minimax.minimax-m2.5 |
| gpt-oss-20b | OpenAI | openai.gpt-oss-20b-1:0 |
| gpt-oss-120b | OpenAI | openai.gpt-oss-120b-1:0 |
| qwen3-32b | Qwen | qwen.qwen3-32b-v1:0 |
| qwen3-coder-30b | Qwen | qwen.qwen3-coder-30b-a3b-v1:0 |

## Modify Models

Open `src/pipeline/benchmarks/compare_providers.py` and edit the `MODELS` list:

```python
{"label": "my-model", "provider": "bedrock", "model": "the.model-id", "region": "eu-north-1"},
```

Comment out to disable, copy a line to add.

## Other Providers

Uncomment entries at the bottom of `MODELS` and set env vars:

```bash
# Google Gemini
GOOGLE_API_KEY=AIza... python -m pipeline.benchmarks.compare_providers --providers google

# Qwen/others via OpenRouter
OPENAI_API_KEY=sk-or-... python -m pipeline.benchmarks.compare_providers --providers openai

# Local Ollama
python -m pipeline.benchmarks.compare_providers --providers ollama

# Mix
GOOGLE_API_KEY=AIza... python -m pipeline.benchmarks.compare_providers --providers bedrock google
```

## Output

Leaderboard to terminal, full JSON to `src/pipeline/benchmarks/results/`.

```
Dataset: bc5cdr (10 docs, test split)
====================================================================================================
Model                     Provider       Exact F1   Partial F1     Time   Errs
----------------------------------------------------------------------------------------------------
deepseek-v3.2             bedrock          0.7500       0.6800      4.8s     0
nova-lite                 bedrock          0.5600       0.5100      1.5s     0
qwen3-32b                 bedrock          0.4550       0.4000      2.9s     0
====================================================================================================
```

## Full 4-Axis Suite (single model)

```bash
BENCH_LLM_SERVICE=bedrock \
BENCH_LLM_MODEL=eu.amazon.nova-lite-v1:0 \
AWS_REGION=eu-north-1 \
BENCH_MAX_DOCS=10 \
python -m pipeline.benchmarks.run_benchmark --all
```

Runs extraction + KG quality + query performance + throughput.

## Troubleshooting

| Error | Fix |
|-------|-----|
| `Token has expired` | `aws sso login` |
| `Invocation of model ID ... not supported` | Use `eu.` prefix (e.g. `eu.amazon.nova-micro-v1:0`) |
| `Read timeout` | Model is slow — shows as errors, others still run |
| `Model use case details` | Submit Anthropic use case form in AWS console |
