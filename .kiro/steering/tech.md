# Tech Stack

## Language & Runtime
- Python ≥ 3.11
- Async-first design using `asyncio` with sync wrappers via `run_sync()`

## Build System
- Build backend: Hatchling (`hatchling`)
- Package manager: uv (lockfile: `uv.lock`)
- Source layout: `src/pipeline/` (wheel package target)

## Core Dependencies
- `neo4j` / `neo4j-graphrag` — graph database
- `ollama` — local LLM inference
- `sentence-transformers` — local embeddings
- `aiosqlite` — async SQLite for job persistence
- `langgraph` — agent orchestration
- `httpx` / `aiohttp` — HTTP clients
- `orjson` — fast JSON serialization
- `rich` — terminal UI and formatting
- `biopython` / `obonet` — biomedical data parsing
- `boto3` / `langchain-aws` — AWS Bedrock LLM backend (optional)

## Testing
- `pytest` with `pytest-asyncio` (asyncio_mode = "auto")
- `pytest-timeout` for test timeouts
- `hypothesis` for property-based testing
- Linting: `ruff`
- Type checking: `mypy`

### Test Markers
- `integration` — requires live services (Neo4j, Ollama, etc.)
- `properties` — property-based tests using Hypothesis

### Hypothesis Profiles
- `ci` — max_examples=20 (auto-detected via `CI` or `GITHUB_ACTIONS` env vars)
- `dev` — max_examples=30 (default for local development)

## Common Commands
```bash
# Install (editable, all extras)
pip install -e ".[all]"

# Run unit tests (excludes integration tests)
pytest tests/ -m "not integration"

# Run property-based tests only
pytest tests/ -m "properties"

# Lint
ruff check src/

# Type check
mypy src/

# Run interactive TUI
uv run litegraf-tui

# Run benchmarks
uv run litegraf-tui bench --all
```

## Configuration
- Environment variables via `PipelineConfig` in `src/pipeline/config.py`
- `.env` file at project root for local secrets (Neo4j creds, API keys, etc.)
- No YAML/TOML runtime config — everything is env vars or constructor args
