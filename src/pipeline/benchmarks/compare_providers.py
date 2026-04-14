#!/usr/bin/env python3
"""Run the litegraf benchmark across multiple LLM providers.

Edit MODELS below to add/remove models. All models use AWS Bedrock in
eu-north-1 by default — no API keys needed, just ``aws sso login``.

Usage (from the pipeline/ directory):
    python -m benchmarks.compare_providers --docs 10
    python -m benchmarks.compare_providers --docs 50 --dataset chemprot
    python -m benchmarks.compare_providers --providers bedrock google --docs 10

Or from the project root:
    python pipeline/benchmarks/compare_providers.py --docs 10
"""

from __future__ import annotations

import argparse
import json
import os
import re as _re
import sys
import time
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Ensure src/ is on sys.path
_BENCH_DIR = Path(__file__).resolve().parent
_SRC_DIR = _BENCH_DIR.parent.parent  # src/
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

# ============================================================================
# MODEL REGISTRY — edit this to add/remove models
#
# All Bedrock models below are available in eu-north-1 (serverless).
# Grouped by lab, roughly ordered by size (smallest first).
# ============================================================================

MODELS: list[dict[str, Any]] = [
    # --- Amazon (eu-north-1) ---
    {"label": "nova-micro",           "provider": "bedrock", "model": "eu.amazon.nova-micro-v1:0",              "region": "eu-north-1"},
    {"label": "nova-lite",            "provider": "bedrock", "model": "eu.amazon.nova-lite-v1:0",               "region": "eu-north-1"},

    # --- Anthropic (us-east-1) ---
    {"label": "claude-haiku-4.5",     "provider": "bedrock", "model": "us.anthropic.claude-haiku-4-5-20251001-v1:0", "region": "us-east-1"},
    {"label": "claude-sonnet-4",      "provider": "bedrock", "model": "us.anthropic.claude-sonnet-4-20250514-v1:0",  "region": "us-east-1"},

    # --- DeepSeek (eu-north-1) ---
    {"label": "deepseek-v3",          "provider": "bedrock", "model": "deepseek.v3-v1:0",                   "region": "eu-north-1"},
    {"label": "deepseek-v3.2",        "provider": "bedrock", "model": "deepseek.v3.2",                      "region": "eu-north-1"},

    # --- MiniMax (eu-north-1) ---
    # {"label": "minimax-m2.1",         "provider": "bedrock", "model": "minimax.minimax-m2.1",               "region": "eu-north-1"},
    # {"label": "minimax-m2.5",         "provider": "bedrock", "model": "minimax.minimax-m2.5",               "region": "eu-north-1"},

    # --- OpenAI (eu-north-1) ---
    {"label": "gpt-oss-20b",          "provider": "bedrock", "model": "openai.gpt-oss-20b-1:0",             "region": "eu-north-1"},
    {"label": "gpt-oss-120b",         "provider": "bedrock", "model": "openai.gpt-oss-120b-1:0",            "region": "eu-north-1"},

    # --- Qwen (eu-north-1) ---
    {"label": "qwen3-32b",            "provider": "bedrock", "model": "qwen.qwen3-32b-v1:0",               "region": "eu-north-1"},
    {"label": "qwen3-coder-30b",      "provider": "bedrock", "model": "qwen.qwen3-coder-30b-a3b-v1:0",     "region": "eu-north-1"},

    # --- Meta (geo-dependent) ---
    {"label": "llama3.1-8b",          "provider": "bedrock", "model": "us.meta.llama3-1-8b-instruct-v1:0",  "region": "us-east-1"},
    {"label": "llama3.2-3b",          "provider": "bedrock", "model": "us.meta.llama3-2-3b-instruct-v1:0",  "region": "us-east-1"},

    # --- Mistral (us-east-1, geo-dependent) ---
    # {"label": "mistral-7b",         "provider": "bedrock", "model": "mistral.mistral-7b-instruct-v0:2",   "region": "us-east-1"},
    # {"label": "ministral-8b",       "provider": "bedrock", "model": "mistral.ministral-3-8b-instruct",    "region": "us-east-1"},

    # --- Direct API providers (set env vars) ---
    # {"label": "gemini-2.0-flash",   "provider": "google",    "model": "gemini-2.0-flash"},
    # {"label": "qwen3-8b-or",        "provider": "openai",    "model": "qwen/qwen3-8b"},
    # {"label": "claude-haiku-direct", "provider": "anthropic", "model": "claude-3-5-haiku-20241022"},

    # --- Cloudflare Workers AI (set CF_ACCOUNT_ID + CF_API_TOKEN) ---
    # {"label": "cf-llama3.1-8b",     "provider": "cloudflare", "model": "@cf/meta/llama-3.1-8b-instruct"},
    # {"label": "cf-mistral-7b",      "provider": "cloudflare", "model": "@cf/mistral/mistral-7b-instruct-v0.1"},

    # --- Local Ollama ---
    # {"label": "qwen3-8b-local",     "provider": "ollama",  "model": "qwen3:8b"},
]

# ============================================================================
# Provider dispatch
# ============================================================================

_bedrock_clients: dict[str, Any] = {}

# Per-doc timeout for Bedrock calls (seconds). Keeps slow models from blocking.
_BEDROCK_TIMEOUT = int(os.environ.get("BENCH_TIMEOUT", "60"))


def _call_bedrock(prompt: str, model: str, region: str = "eu-north-1", **_: Any) -> str:
    import boto3
    from botocore.config import Config
    if region not in _bedrock_clients:
        _bedrock_clients[region] = boto3.client(
            "bedrock-runtime", region_name=region,
            config=Config(read_timeout=_BEDROCK_TIMEOUT, connect_timeout=10, retries={"max_attempts": 0}),
        )
    resp = _bedrock_clients[region].converse(
        modelId=model,
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        inferenceConfig={"maxTokens": 2048, "temperature": 0.1},
    )
    # Handle varying response shapes across providers
    content = resp["output"]["message"]["content"]
    if isinstance(content, list) and content:
        return content[0].get("text", "") if isinstance(content[0], dict) else str(content[0])
    return str(content)


def _call_anthropic(prompt: str, model: str, **_: Any) -> str:
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        raise RuntimeError("Set ANTHROPIC_API_KEY")
    payload = json.dumps({
        "model": model, "max_tokens": 2048, "temperature": 0.1,
        "messages": [{"role": "user", "content": prompt}],
    }).encode()
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages", data=payload,
        headers={"Content-Type": "application/json", "x-api-key": key, "anthropic-version": "2023-06-01"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read())["content"][0]["text"]


def _call_google(prompt: str, model: str, **_: Any) -> str:
    key = os.environ.get("GOOGLE_API_KEY", "")
    if not key:
        raise RuntimeError("Set GOOGLE_API_KEY")
    payload = json.dumps({
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.1, "maxOutputTokens": 2048},
    }).encode()
    req = urllib.request.Request(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}",
        data=payload, headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read())["candidates"][0]["content"]["parts"][0]["text"]


def _call_openai(prompt: str, model: str, **_: Any) -> str:
    key = os.environ.get("OPENAI_API_KEY", "")
    if not key:
        raise RuntimeError("Set OPENAI_API_KEY")
    base = os.environ.get("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
    payload = json.dumps({
        "model": model, "max_tokens": 2048, "temperature": 0.1,
        "messages": [{"role": "user", "content": prompt}],
    }).encode()
    req = urllib.request.Request(
        f"{base}/chat/completions", data=payload,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read())["choices"][0]["message"]["content"]


def _call_ollama(prompt: str, model: str, **_: Any) -> str:
    url = os.environ.get("OLLAMA_URL", "http://localhost:11434")
    payload = json.dumps({"model": model, "prompt": prompt, "stream": False}).encode()
    req = urllib.request.Request(f"{url}/api/generate", data=payload, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        return json.loads(resp.read()).get("response", "")


def _call_cloudflare(prompt: str, model: str, **_: Any) -> str:
    account_id = os.environ.get("CF_ACCOUNT_ID", "")
    api_token = os.environ.get("CF_API_TOKEN", "")
    if not account_id or not api_token:
        raise RuntimeError("Set CF_ACCOUNT_ID and CF_API_TOKEN")
    payload = json.dumps({
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 2048, "temperature": 0.1,
    }).encode()
    req = urllib.request.Request(
        f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/{model}",
        data=payload,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_token}"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        body = json.loads(resp.read())
    if not body.get("success"):
        raise RuntimeError(f"Cloudflare Workers AI error: {body.get('errors', [])}")
    return body.get("result", {}).get("response", "")


PROVIDERS = {"bedrock": _call_bedrock, "anthropic": _call_anthropic, "google": _call_google, "openai": _call_openai, "ollama": _call_ollama, "cloudflare": _call_cloudflare}

# ============================================================================
# Benchmark runner
# ============================================================================


def _check_available(entry: dict) -> str | None:
    p = entry["provider"]
    if p == "anthropic" and not os.environ.get("ANTHROPIC_API_KEY"):
        return "ANTHROPIC_API_KEY not set"
    if p == "google" and not os.environ.get("GOOGLE_API_KEY"):
        return "GOOGLE_API_KEY not set"
    if p == "openai" and not os.environ.get("OPENAI_API_KEY"):
        return "OPENAI_API_KEY not set"
    if p == "ollama":
        try:
            urllib.request.urlopen(os.environ.get("OLLAMA_URL", "http://localhost:11434") + "/api/tags", timeout=2)
        except Exception:
            return "Ollama not running"
    if p == "cloudflare" and (not os.environ.get("CF_ACCOUNT_ID") or not os.environ.get("CF_API_TOKEN")):
        return "CF_ACCOUNT_ID / CF_API_TOKEN not set"
    return None


def run_comparison(models: list[dict], max_docs: int = 10, dataset: str = "bc5cdr", split: str = "test") -> dict[str, Any]:
    from pipeline.benchmarks.datasets.loader import download_dataset, load_dataset
    from pipeline.benchmarks.datasets.models import BenchmarkExample, Entity
    from pipeline.benchmarks.metrics.extraction_metrics import evaluate_extraction

    try:
        download_dataset(dataset)
    except Exception:
        pass
    ds = load_dataset(dataset)
    gold = ds.splits[split].examples[:max_docs]

    results: dict[str, Any] = {
        "metadata": {"timestamp": datetime.now(tz=UTC).isoformat(), "dataset": dataset, "split": split, "num_documents": len(gold)},
        "models": {},
    }

    for entry in models:
        label, provider, model = entry["label"], entry["provider"], entry["model"]
        call_fn = PROVIDERS[provider]

        err = _check_available(entry)
        if err:
            print(f"  SKIP {label}: {err}")
            results["models"][label] = {"skipped": err}
            continue

        print(f"  RUN  {label} ({provider}/{model})")
        predictions: list[BenchmarkExample] = []
        t0 = time.monotonic()
        errors = 0
        max_errors = max(3, len(gold) // 3)  # bail if >1/3 docs fail

        for i, ex in enumerate(gold):
            if errors >= max_errors:
                print(f"    BAIL — {errors} errors, skipping remaining docs")
                break
            prompt = (
                "Extract all named entities from the following biomedical text.\n"
                'Return a JSON array of objects with keys: "text", "entity_type".\n'
                f"Entity types: {', '.join(ds.entity_types)}.\n"
                "Only extract specific named entities, not generic terms.\n\n"
                f"Text: {ex.text[:2000]}\n\nReturn only valid JSON array:"
            )
            entities: list[Entity] = []
            try:
                raw = call_fn(prompt, **entry)
                # Strip markdown fences
                clean = _re.sub(r"```(?:json)?\s*", "", raw)
                m = _re.search(r"\[.*\]", clean, _re.DOTALL)
                if m:
                    json_str = m.group()
                    # Fix trailing commas
                    json_str = _re.sub(r",\s*([}\]])", r"\1", json_str)
                    for item in json.loads(json_str):
                        if isinstance(item, dict):
                            text = item.get("text") or item.get("name") or item.get("entity") or ""
                            etype = item.get("entity_type") or item.get("type") or item.get("label") or "Entity"
                            if text:
                                entities.append(Entity(text=text, entity_type=etype))
            except Exception as e:
                errors += 1
                if errors <= 2:
                    print(f"    ERR doc {i}: {e!s:.80}")
                elif errors == 3:
                    print(f"    ... suppressing further errors")
            predictions.append(BenchmarkExample(doc_id=ex.doc_id, text=ex.text, entities=entities))

        elapsed = time.monotonic() - t0
        metrics = evaluate_extraction(gold, predictions)
        results["models"][label] = {"provider": provider, "model": model, "region": entry.get("region", ""), "elapsed_sec": round(elapsed, 1), "errors": errors, **metrics}

    # --- Competitors (if installed) ---
    for comp_name, comp_module in [
        ("nano-graphrag", "nano_graphrag_runner"),
        ("lightrag", "lightrag_runner"),
        ("ms-graphrag", "ms_graphrag_runner"),
    ]:
        try:
            mod = __import__(f"pipeline.benchmarks.competitors.{comp_module}", fromlist=["is_available", "run_extraction"])
            if mod.is_available():
                import asyncio
                print(f"  RUN  {comp_name} (competitor)")
                t0 = time.monotonic()
                comp_preds = asyncio.run(mod.run_extraction(gold))
                elapsed = time.monotonic() - t0
                metrics = evaluate_extraction(gold, comp_preds)
                results["models"][comp_name] = {"provider": "competitor", "model": comp_name, "region": "", "elapsed_sec": round(elapsed, 1), "errors": 0, **metrics}
        except Exception as e:
            print(f"  SKIP {comp_name}: {e!s:.80}")

    return results


def print_table(results: dict[str, Any]) -> None:
    meta = results["metadata"]
    print(f"\nDataset: {meta['dataset']} ({meta['num_documents']} docs, {meta['split']} split)")
    print("=" * 100)
    print(f"{'Model':<25} {'Provider':<12} {'Exact F1':>10} {'Partial F1':>12} {'Time':>8} {'Errs':>6}")
    print("-" * 100)

    rows = []
    for label, r in results["models"].items():
        if "skipped" in r:
            print(f"{label:<25} {'—':<12} {'SKIPPED':>10} {r['skipped']}")
            continue
        exact_f1 = r.get("exact_match", {}).get("micro_avg", {}).get("f1", 0.0)
        partial_f1 = r.get("partial_match", {}).get("micro_avg", {}).get("f1", 0.0)
        rows.append((exact_f1, label, r))

    for exact_f1, label, r in sorted(rows, reverse=True):
        partial_f1 = r.get("partial_match", {}).get("micro_avg", {}).get("f1", 0.0)
        print(f"{label:<25} {r['provider']:<12} {exact_f1:>10.4f} {partial_f1:>12.4f} {r['elapsed_sec']:>7.1f}s {r['errors']:>5}")
    print("=" * 100)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare LLM providers on litegraf NER benchmarks")
    parser.add_argument("--docs", type=int, default=10, help="Documents per model (default: 10)")
    parser.add_argument("--dataset", default="bc5cdr", choices=["bc5cdr", "chemprot", "gad"])
    parser.add_argument("--providers", nargs="*", help="Filter to providers (bedrock, anthropic, google, openai, ollama, cloudflare)")
    parser.add_argument("-o", "--output", help="Output JSON path")
    args = parser.parse_args()

    models = MODELS
    if args.providers:
        models = [m for m in models if m["provider"] in args.providers]
    if not models:
        print("No models selected.")
        sys.exit(1)

    print(f"Running {len(models)} models × {args.docs} docs on {args.dataset}...\n")
    results = run_comparison(models, max_docs=args.docs, dataset=args.dataset)
    print_table(results)

    out_path = args.output or str(_BENCH_DIR / "results" / f"providers_{datetime.now(tz=UTC).strftime('%Y%m%dT%H%M%SZ')}.json")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nFull results: {out_path}")


if __name__ == "__main__":
    main()
