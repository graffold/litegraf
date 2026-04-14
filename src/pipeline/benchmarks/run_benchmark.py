"""Benchmark runner CLI — main entry point for the litegraf benchmark suite.

Runs benchmark axes (extraction, kg-quality, query, throughput) and writes
structured JSON results to ``benchmarks/results/``.

Usage::

    # Full suite (requires Ollama running)
    BENCH_LLM_MODEL=qwen3:8b python -m pipeline.benchmarks.run_benchmark --all

    # Single axis
    BENCH_LLM_MODEL=qwen3:8b python -m pipeline.benchmarks.run_benchmark --axis extraction

    # Quick test
    BENCH_MAX_DOCS=5 BENCH_LLM_MODEL=qwen3:8b python -m pipeline.benchmarks.run_benchmark --all

    # With competitor comparison
    BENCH_LLM_MODEL=qwen3:8b python -m pipeline.benchmarks.run_benchmark --all --competitors lightrag ms-graphrag

Environment variables::

    BENCH_LLM_MODEL   Ollama model name (default: llama3)
    BENCH_LLM_URL     Ollama server URL (default: http://localhost:11434)
    BENCH_DATASET      Dataset for extraction: bc5cdr, chemprot, gad (default: bc5cdr)
    BENCH_MAX_DOCS     Max documents per axis (default: 50)
    BENCH_SPLIT        Dataset split (default: test)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import re as _re
import resource
import sys
import time
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Ensure src/ is on sys.path when run as a script
_SRC_DIR = str(Path(__file__).resolve().parent.parent.parent)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

VALID_AXES = ("extraction", "kg-quality", "kg-build", "query", "throughput")

logger = logging.getLogger("benchmarks.run_benchmark")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _llm_generate(prompt: str, *, timeout: int = 180) -> str:
    """Call LLM via Ollama HTTP, AWS Bedrock, or Cloudflare Workers AI depending on BENCH_LLM_SERVICE."""
    service = os.environ.get("BENCH_LLM_SERVICE", "ollama")

    if service == "bedrock":
        return _bedrock_generate(prompt)
    if service == "cloudflare":
        return _cloudflare_generate(prompt)
    return _ollama_generate(prompt, timeout=timeout)


def _ollama_generate(prompt: str, *, timeout: int = 180) -> str:
    """Call Ollama HTTP API directly."""
    model = os.environ.get("BENCH_LLM_MODEL", "llama3")
    url = os.environ.get("BENCH_LLM_URL", "http://localhost:11434")

    payload = json.dumps({"model": model, "prompt": prompt, "stream": False}).encode()
    req = urllib.request.Request(
        f"{url}/api/generate", data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
        body = json.loads(resp.read())
    return body.get("response", "")


_bedrock_client = None
_bedrock_client_region = None


def _bedrock_generate(prompt: str) -> str:
    """Call AWS Bedrock Converse API (client cached per region)."""
    global _bedrock_client, _bedrock_client_region
    import boto3

    region = os.environ.get("AWS_REGION", "eu-west-1")
    model_id = os.environ.get("BENCH_LLM_MODEL", "eu.amazon.nova-micro-v1:0")

    if _bedrock_client is None or _bedrock_client_region != region:
        _bedrock_client = boto3.client("bedrock-runtime", region_name=region)
        _bedrock_client_region = region

    resp = _bedrock_client.converse(
        modelId=model_id,
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        inferenceConfig={"maxTokens": 2048, "temperature": 0.1},
    )
    return resp["output"]["message"]["content"][0]["text"]


def _cloudflare_generate(prompt: str) -> str:
    """Call Cloudflare Workers AI REST API."""
    account_id = os.environ.get("CF_ACCOUNT_ID", "")
    api_token = os.environ.get("CF_API_TOKEN", "")
    model = os.environ.get("BENCH_LLM_MODEL", "@cf/meta/llama-3.1-8b-instruct")
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


# Models comparable to Llama 3 8B — 2 per lab
COMPARE_MODELS = [
    {"model": "us.meta.llama3-1-8b-instruct-v1:0", "region": "us-east-1", "label": "meta-llama3.1-8b"},
    {"model": "eu.amazon.nova-micro-v1:0", "region": "eu-west-1", "label": "amazon-nova-micro"},
    {"model": "eu.amazon.nova-lite-v1:0", "region": "eu-west-1", "label": "amazon-nova-lite"},
    {"model": "mistral.mistral-7b-instruct-v0:2", "region": "us-east-1", "label": "mistral-7b-instruct"},
    {"model": "mistral.ministral-3-8b-instruct", "region": "us-east-1", "label": "mistral-ministral-8b"},
]


def _parse_json_array(text: str) -> list[dict]:
    """Extract the first JSON array from LLM output."""
    m = _re.search(r"\[.*\]", text, _re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except (ValueError, KeyError):
            pass
    return []


def _parse_json_object(text: str) -> dict:
    """Extract the first JSON object from LLM output."""
    m = _re.search(r"\{.*\}", text, _re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except (ValueError, KeyError):
            pass
    return {}


def _max_docs() -> int:
    return int(os.environ.get("BENCH_MAX_DOCS", "50"))


def _peak_rss_mb() -> float:
    """Current max RSS in MB (macOS/Linux)."""
    ru = resource.getrusage(resource.RUSAGE_SELF)
    # macOS reports bytes, Linux reports KB
    if platform.system() == "Darwin":
        return ru.ru_maxrss / (1024 * 1024)
    return ru.ru_maxrss / 1024


def _collect_metadata(competitors: list[str]) -> dict[str, Any]:
    """Build metadata block."""
    uname = platform.uname()
    hw: dict[str, Any] = {
        "system": uname.system,
        "machine": uname.machine,
        "processor": uname.processor or platform.processor(),
        "cpu_count": os.cpu_count(),
        "os_version": platform.platform(),
    }
    try:
        import subprocess
        mem = subprocess.check_output(
            ["sysctl", "-n", "hw.memsize"], text=True, timeout=5,
        ).strip()
        hw["ram_gb"] = round(int(mem) / (1024**3))
    except Exception:
        pass
    try:
        import subprocess
        hw["cpu_model"] = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True, timeout=5,
        ).strip()
    except Exception:
        pass

    pkg_versions: dict[str, str] = {}
    for pkg in ("neo4j", "tiktoken", "orjson", "pydantic", "granian"):
        try:
            mod = __import__(pkg)
            pkg_versions[pkg] = getattr(mod, "__version__", "installed")
        except ImportError:
            pass

    return {
        "run_date": datetime.now(tz=UTC).isoformat(),
        "graffold_version": _graffold_version(),
        "python_version": platform.python_version(),
        "hardware": hw,
        "package_versions": pkg_versions,
        "llm_model": os.environ.get("BENCH_LLM_MODEL", "llama3"),
        "llm_service": os.environ.get("BENCH_LLM_SERVICE", "ollama"),
        "competitors": competitors,
    }


def _graffold_version() -> str:
    toml_path = Path(__file__).resolve().parent.parent / "pyproject.toml"
    try:
        text = toml_path.read_text()
        for line in text.splitlines():
            if line.strip().startswith("version"):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    except OSError:
        pass
    return "unknown"


# ---------------------------------------------------------------------------
# Axis: extraction
# ---------------------------------------------------------------------------


def _run_extraction(competitors: list[str]) -> dict[str, Any]:
    """Run NER extraction against real biomedical datasets via LLM."""
    from pipeline.benchmarks.datasets.loader import download_dataset, load_dataset
    from pipeline.benchmarks.datasets.models import BenchmarkExample, Entity
    from pipeline.benchmarks.metrics.extraction_metrics import evaluate_extraction

    results: dict[str, Any] = {"competitors": competitors}

    dataset_name = os.environ.get("BENCH_DATASET", "bc5cdr")
    max_docs = _max_docs()
    split = os.environ.get("BENCH_SPLIT", "test")

    try:
        download_dataset(dataset_name)
    except Exception as e:
        logger.warning("Dataset download failed: %s", e)

    try:
        ds = load_dataset(dataset_name)
    except Exception as e:
        results["error"] = f"Dataset load failed: {e}"
        return results

    if split not in ds.splits:
        results["error"] = f"Split '{split}' not in {dataset_name}"
        return results

    gold_examples = ds.splits[split].examples[:max_docs]
    results["dataset"] = dataset_name
    results["split"] = split
    results["num_documents"] = len(gold_examples)

    # --- Run litegraf extraction ---
    predictions: list[BenchmarkExample] = []
    for i, ex in enumerate(gold_examples):
        logger.info("Extracting %d/%d: doc %s (%d chars)", i + 1, len(gold_examples), ex.doc_id, len(ex.text))
        entities: list[Entity] = []
        try:
            prompt = (
                "Extract all named entities from the following biomedical text.\n"
                'Return a JSON array of objects with keys: "text", "entity_type".\n'
                "Entity types: Chemical, Disease, Gene, Protein.\n"
                "Only extract specific named entities, not generic terms.\n\n"
                f"Text: {ex.text[:2000]}\n\nReturn only valid JSON array:"
            )
            raw = _llm_generate(prompt)
            for item in _parse_json_array(raw):
                entities.append(Entity(
                    text=item.get("text", ""),
                    entity_type=item.get("entity_type", "Entity"),
                    start=item.get("start", -1),
                    end=item.get("end", -1),
                ))
        except Exception as e:
            logger.warning("Extraction failed for %s: %s", ex.doc_id, e)
        predictions.append(BenchmarkExample(doc_id=ex.doc_id, text=ex.text, entities=entities))

    results["litegraf"] = evaluate_extraction(gold_examples, predictions)

    # --- Competitors ---
    for comp_name in competitors:
        try:
            mod = __import__(f"pipeline.benchmarks.competitors.{comp_name.replace('-', '_')}_runner", fromlist=["is_available", "run_extraction"])
            if mod.is_available():
                import asyncio
                comp_preds = asyncio.run(mod.run_extraction(gold_examples))
                results[comp_name] = evaluate_extraction(gold_examples, comp_preds)
            else:
                results[comp_name] = {"error": f"{comp_name} not installed"}
        except Exception as e:
            results[comp_name] = {"error": str(e)}

    return results


# ---------------------------------------------------------------------------
# Axis: kg-quality
# ---------------------------------------------------------------------------


def _run_kg_quality(competitors: list[str]) -> dict[str, Any]:
    """Run KG quality benchmarks: consolidation, provenance, contradiction detection via LLM."""
    from pipeline.benchmarks.generators import (
        consolidation_stress,
        contradiction_pairs,
        provenance_annotator,
    )
    from pipeline.benchmarks.metrics.kg_quality_metrics import full_report

    consol_gt = consolidation_stress.generate(seed=42)
    contra_gt = contradiction_pairs.generate(seed=42)
    prov_gt = provenance_annotator.generate(seed=42)

    # --- 1. Entity consolidation: ask LLM to cluster duplicate nodes ---
    nodes = consol_gt["nodes"]
    max_nodes = min(len(nodes), _max_docs() * 3)  # scale with BENCH_MAX_DOCS
    node_subset = nodes[:max_nodes]

    node_list_str = "\n".join(
        f"  {n['id']}: {n['name']} (type={n['type']}, uniprot={n.get('uniprot_id', '')}, synonyms={n.get('synonyms', [])})"
        for n in node_subset
    )
    prompt_consol = (
        "You are an entity resolution system. The following biomedical entities may contain duplicates "
        "(typos, synonyms, case variants, same UniProt/MONDO ID).\n"
        "Group them into clusters of entities that refer to the same real-world entity.\n"
        'Return a JSON object mapping cluster_id to list of node IDs, e.g. {"c0": ["n0", "n1"], "c1": ["n3"]}.\n\n'
        f"Entities:\n{node_list_str}\n\nReturn only valid JSON:"
    )

    logger.info("KG-quality: consolidation — %d nodes", len(node_subset))
    try:
        raw = _llm_generate(prompt_consol, timeout=300)
        predicted_clusters = _parse_json_object(raw)
        # Ensure values are lists of strings
        predicted_clusters = {
            str(k): [str(x) for x in v] if isinstance(v, list) else [str(v)]
            for k, v in predicted_clusters.items()
        }
    except Exception as e:
        logger.warning("Consolidation LLM call failed: %s", e)
        # Fallback: each node is its own cluster
        predicted_clusters = {n["id"]: [n["id"]] for n in node_subset}

    # --- 2. Contradiction detection: ask LLM to classify pairs ---
    pairs = contra_gt["pairs"]
    max_pairs = min(len(pairs), _max_docs())
    pair_subset = pairs[:max_pairs]
    predicted_contradictions: dict[str, bool] = {}

    logger.info("KG-quality: contradiction detection — %d pairs", len(pair_subset))
    for i, pair in enumerate(pair_subset):
        try:
            prompt_contra = (
                "Do these two biomedical claims contradict each other? "
                'Answer with a JSON object: {"contradictory": true} or {"contradictory": false}.\n\n'
                f"Claim A: {pair['abstract_a']['claim']}\n"
                f"Claim B: {pair['abstract_b']['claim']}\n\n"
                "Return only valid JSON:"
            )
            raw = _llm_generate(prompt_contra, timeout=60)
            obj = _parse_json_object(raw)
            predicted_contradictions[pair["pair_id"]] = bool(obj.get("contradictory", False))
        except Exception as e:
            logger.warning("Contradiction detection failed for %s: %s", pair["pair_id"], e)
            predicted_contradictions[pair["pair_id"]] = False

        if (i + 1) % 10 == 0:
            logger.info("  contradiction %d/%d", i + 1, len(pair_subset))

    # --- 3. Provenance validation: ask LLM to verify chains ---
    chains = prov_gt["chains"]
    max_chains = min(len(chains), _max_docs())
    chain_subset = chains[:max_chains]
    predicted_provenance: dict[str, bool] = {}

    logger.info("KG-quality: provenance validation — %d chains", len(chain_subset))
    for i, chain in enumerate(chain_subset):
        try:
            steps_str = "\n".join(
                f"  Step {s['step']}: {s['type']} — {json.dumps({k: v for k, v in s.items() if k not in ('step', 'type')})}"
                for s in chain["steps"]
            )
            prompt_prov = (
                "Does this provenance chain form a valid, complete trace from source document to entity node? "
                'Answer with JSON: {"valid": true} or {"valid": false}.\n\n'
                f"Entity: {chain['entity']['name']} ({chain['entity']['type']})\n"
                f"Chain steps:\n{steps_str}\n\n"
                "Return only valid JSON:"
            )
            raw = _llm_generate(prompt_prov, timeout=60)
            obj = _parse_json_object(raw)
            predicted_provenance[chain["chain_id"]] = bool(obj.get("valid", False))
        except Exception as e:
            logger.warning("Provenance validation failed for %s: %s", chain["chain_id"], e)
            predicted_provenance[chain["chain_id"]] = False

    # --- Build report ---
    report = full_report(
        consolidation_gt=consol_gt,
        predicted_clusters=predicted_clusters,
        nodes=nodes,
        provenance_gt=prov_gt,
        predicted_provenance=predicted_provenance,
        contradiction_gt=contra_gt,
        predicted_contradictions=predicted_contradictions,
    )
    report["competitors"] = competitors
    return report


# ---------------------------------------------------------------------------
# Axis: query
# ---------------------------------------------------------------------------

# Benchmark questions with expected properties for evaluation
_BENCH_QUESTIONS = [
    {"question": "What proteins are associated with cardiovascular disease?", "expected_mode": "local", "hop_depth": 1},
    {"question": "What is the relationship between IL-6 and diabetes?", "expected_mode": "local", "hop_depth": 1},
    {"question": "Which proteins interact with TNF-alpha in inflammatory pathways?", "expected_mode": "local", "hop_depth": 2},
    {"question": "Give me an overview of protein biomarkers in kidney disease", "expected_mode": "global", "hop_depth": 1},
    {"question": "What proteins are shared between Alzheimer's disease and type 2 diabetes?", "expected_mode": "hybrid", "hop_depth": 2},
    {"question": "How does VEGF relate to cancer progression through intermediate proteins?", "expected_mode": "local", "hop_depth": 3},
    {"question": "Compare protein biomarker profiles across autoimmune diseases", "expected_mode": "global", "hop_depth": 1},
    {"question": "What is the role of BRCA1 in breast cancer?", "expected_mode": "local", "hop_depth": 1},
    {"question": "Which diseases share the most protein associations in the knowledge graph?", "expected_mode": "global", "hop_depth": 2},
    {"question": "Trace the evidence chain for the association between CRP and heart failure", "expected_mode": "local", "hop_depth": 2},
]


def _run_query(competitors: list[str]) -> dict[str, Any]:
    """Run query benchmarks: mode classification, answer generation, and latency measurement."""
    from pipeline.benchmarks.metrics.query_metrics import (
        QueryRecord,
        evaluate_query_performance,
    )

    max_q = min(_max_docs(), len(_BENCH_QUESTIONS))
    questions = _BENCH_QUESTIONS[:max_q]
    records: list[QueryRecord] = []

    logger.info("Query benchmark: %d questions", len(questions))

    for i, q in enumerate(questions):
        question = q["question"]
        logger.info("Query %d/%d: %s", i + 1, len(questions), question[:60])

        # --- 1. Mode classification ---
        try:
            mode_prompt = (
                "Classify this knowledge graph query into one of these modes: local, global, hybrid, naive.\n"
                "- local: specific entity lookups (e.g. 'What proteins are associated with X?')\n"
                "- global: broad overviews requiring community summaries\n"
                "- hybrid: needs both local entity data and global context\n"
                "- naive: simple factual question, no graph needed\n\n"
                f"Query: {question}\n\n"
                'Return JSON: {{"mode": "local"}} (or global/hybrid/naive):'
            )
            t0 = time.monotonic()
            raw = _llm_generate(mode_prompt, timeout=60)
            mode_latency = (time.monotonic() - t0) * 1000
            obj = _parse_json_object(raw)
            predicted_mode = obj.get("mode", "local")
        except Exception as e:
            logger.warning("Mode classification failed: %s", e)
            predicted_mode = "local"
            mode_latency = 0.0

        # --- 2. Answer generation ---
        try:
            answer_prompt = (
                "You are a biomedical knowledge graph assistant. Answer the following question "
                "based on your knowledge of protein-disease relationships.\n\n"
                f"Question: {question}\n\n"
                "Provide a concise, informative answer mentioning specific proteins and diseases:"
            )
            t0 = time.monotonic()
            answer = _llm_generate(answer_prompt, timeout=180)
            answer_latency = (time.monotonic() - t0) * 1000
        except Exception as e:
            logger.warning("Answer generation failed: %s", e)
            answer = ""
            answer_latency = 0.0

        total_latency = mode_latency + answer_latency

        records.append(QueryRecord(
            question=question,
            answer=answer,
            expected_mode=q.get("expected_mode", ""),
            predicted_mode=predicted_mode,
            latency_ms=total_latency,
            success=bool(answer),
            cache_hit=False,
            hop_depth=q.get("hop_depth", 1),
            context=answer[:500],
        ))

    # --- 3. LLM-as-judge for answer relevance ---
    class _OllamaJudge:
        def score_relevance(self, question: str, answer: str, context: str) -> float:
            prompt = (
                "Rate how well this answer addresses the biomedical question on a scale of 1-5.\n"
                "1 = completely irrelevant, 5 = comprehensive and accurate.\n\n"
                f"Question: {question}\n"
                f"Answer: {answer[:1000]}\n\n"
                'Return JSON: {{"score": 3.5}}:'
            )
            try:
                raw = _llm_generate(prompt, timeout=60)
                obj = _parse_json_object(raw)
                return float(obj.get("score", 3.0))
            except Exception:
                return 3.0  # neutral default

    judge = _OllamaJudge()
    report = evaluate_query_performance(records, judge=judge)
    report["competitors"] = competitors
    return report


# ---------------------------------------------------------------------------
# Axis: throughput
# ---------------------------------------------------------------------------


def _run_throughput(competitors: list[str]) -> dict[str, Any]:
    """Run throughput benchmarks: measure extraction speed, embedding rate, memory."""
    from pipeline.benchmarks.datasets.loader import download_dataset, load_dataset
    from pipeline.benchmarks.metrics.throughput_metrics import (
        ThroughputRecord,
        evaluate_throughput,
    )

    dataset_name = os.environ.get("BENCH_DATASET", "bc5cdr")
    max_docs = _max_docs()
    split = os.environ.get("BENCH_SPLIT", "test")

    try:
        download_dataset(dataset_name)
    except Exception:
        pass

    try:
        ds = load_dataset(dataset_name)
    except Exception as e:
        return {"error": f"Dataset load failed: {e}", "competitors": competitors}

    if split not in ds.splits:
        return {"error": f"Split '{split}' not in {dataset_name}", "competitors": competitors}

    examples = ds.splits[split].examples[:max_docs]
    records: list[ThroughputRecord] = []

    # --- 1. Extraction throughput ---
    logger.info("Throughput: extraction — %d documents", len(examples))
    rss_before = _peak_rss_mb()
    t0 = time.monotonic()
    total_prompt_tokens = 0
    total_completion_tokens = 0
    successes = 0

    for i, ex in enumerate(examples):
        try:
            prompt = (
                "Extract entities and relationships from this biomedical text. "
                'Return JSON: {"nodes": [...], "relationships": [...]}\n\n'
                f"Text: {ex.text[:2000]}\n\nReturn only valid JSON:"
            )
            total_prompt_tokens += len(prompt.split())  # rough token estimate
            raw = _llm_generate(prompt)
            total_completion_tokens += len(raw.split())
            successes += 1
        except Exception as e:
            logger.warning("Throughput extraction failed for doc %d: %s", i, e)

        if (i + 1) % 10 == 0:
            elapsed = time.monotonic() - t0
            rate = (i + 1) / elapsed * 60
            logger.info("  %d/%d docs (%.1f docs/min)", i + 1, len(examples), rate)

    extraction_duration = time.monotonic() - t0
    rss_after = _peak_rss_mb()

    records.append(ThroughputRecord(
        source="pubmed",
        doc_count=successes,
        duration_sec=extraction_duration,
        prompt_tokens=total_prompt_tokens,
        completion_tokens=total_completion_tokens,
        llm_config=os.environ.get("BENCH_LLM_MODEL", "ollama"),
        peak_rss_mb=rss_after,
        pipeline_phase="extraction",
    ))

    # --- 2. Embedding throughput (if sentence-transformers available) ---
    embed_duration = 0.0
    try:
        from sentence_transformers import SentenceTransformer

        model_name = "all-mpnet-base-v2"
        logger.info("Throughput: embeddings — %d texts with %s", len(examples), model_name)
        model = SentenceTransformer(model_name)
        texts = [ex.text[:512] for ex in examples]

        t0 = time.monotonic()
        model.encode(texts, batch_size=32, show_progress_bar=False)
        embed_duration = time.monotonic() - t0

        records.append(ThroughputRecord(
            source="pubmed",
            doc_count=len(texts),
            duration_sec=embed_duration,
            embedding_count=len(texts),
            embedding_duration_sec=embed_duration,
            peak_rss_mb=_peak_rss_mb(),
            pipeline_phase="embedding",
        ))
    except ImportError:
        logger.info("Throughput: skipping embeddings (sentence-transformers not installed)")

    # --- 3. Full pipeline record ---
    records.append(ThroughputRecord(
        source="pubmed",
        doc_count=successes,
        duration_sec=extraction_duration + embed_duration,
        prompt_tokens=total_prompt_tokens,
        completion_tokens=total_completion_tokens,
        llm_config=os.environ.get("BENCH_LLM_MODEL", "ollama"),
        peak_rss_mb=_peak_rss_mb(),
        pipeline_phase="full",
    ))

    report = evaluate_throughput(records)
    report["competitors"] = competitors
    report["dataset"] = dataset_name
    report["extraction_success_rate"] = round(successes / len(examples), 4) if examples else 0.0
    return report


# ---------------------------------------------------------------------------
# Axis: kg-build (end-to-end graph construction + query)
# ---------------------------------------------------------------------------


def _run_kg_build(competitors: list[str]) -> dict[str, Any]:
    """Build a real KG from benchmark docs using LiteGraf for each model in MODELS."""
    from pipeline.benchmarks.datasets.loader import download_dataset, load_dataset
    from pipeline.litegraf import LiteGraf

    # Import MODELS from compare_providers
    from pipeline.benchmarks.compare_providers import MODELS

    dataset_name = os.environ.get("BENCH_DATASET", "bc5cdr")
    max_docs = _max_docs()
    split = os.environ.get("BENCH_SPLIT", "test")

    # Check Neo4j connectivity
    graph_uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    graph_user = os.environ.get("NEO4J_USER", "neo4j")
    graph_password = os.environ.get("NEO4J_PASSWORD", "password")
    graph_database = os.environ.get("NEO4J_DATABASE", "litegrafbench")

    try:
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver(graph_uri, auth=(graph_user, graph_password))
        driver.verify_connectivity()
        driver.close()
    except Exception as e:
        return {"skipped": f"Neo4j not reachable at {graph_uri}: {e}", "competitors": competitors}

    try:
        download_dataset(dataset_name)
    except Exception:
        pass

    ds = load_dataset(dataset_name)
    examples = ds.splits[split].examples[:max_docs]
    texts = [ex.text for ex in examples]

    results: dict[str, Any] = {
        "dataset": dataset_name,
        "num_documents": len(texts),
        "models": {},
        "competitors": competitors,
    }

    sample_questions = [
        "What proteins are associated with cardiovascular disease?",
        "Which chemicals interact with gene targets?",
        "What diseases share common protein biomarkers?",
    ]

    for entry in MODELS:
        label = entry["label"]
        provider = entry["provider"]
        model = entry["model"]
        region = entry.get("region", "eu-north-1")

        # Only bedrock, ollama, and cloudflare are supported as LiteGraf LLM backends
        if provider not in ("bedrock", "ollama", "cloudflare"):
            logger.info("kg-build: SKIP %s (provider %s not supported)", label, provider)
            results["models"][label] = {"skipped": f"provider '{provider}' not wired for LiteGraf"}
            continue

        logger.info("kg-build: === %s (%s/%s) ===", label, provider, model)

        # Clear the graph between models
        try:
            from neo4j import GraphDatabase as _GD
            _drv = _GD.driver(graph_uri, auth=(graph_user, graph_password))
            with _drv.session(database=graph_database) as sess:
                sess.run("MATCH (n) DETACH DELETE n")
            _drv.close()
        except Exception as e:
            logger.warning("kg-build: failed to clear graph: %s", e)

        # Build LiteGraf with this model
        llm_kwargs: dict[str, Any] = {"model": model}
        if provider == "bedrock":
            llm_kwargs["region_name"] = region
        elif provider == "ollama":
            llm_kwargs["base_url"] = os.environ.get("BENCH_LLM_URL", "http://localhost:11434")

        try:
            lg = LiteGraf(
                graph_store="neo4j",
                graph_uri=graph_uri,
                graph_user=graph_user,
                graph_password=graph_password,
                graph_database=graph_database,
                llm=provider,
                llm_model=model,
                llm_url=llm_kwargs.get("base_url", "http://localhost:11434"),
                enable_dedup=False,
            )
            # Override the LLM with region-aware Bedrock provider if needed
            if provider == "bedrock":
                from pipeline.backends.bedrock_llm import BedrockLLMProvider
                lg._llm = BedrockLLMProvider(model=model, region_name=region)
            elif provider == "cloudflare":
                from pipeline.backends.cloudflare_llm import CloudflareLLMProvider
                lg._llm = CloudflareLLMProvider(model=model)
        except Exception as e:
            logger.warning("kg-build: failed to init LiteGraf for %s: %s", label, e)
            results["models"][label] = {"error": str(e)}
            continue

        # Insert docs
        t0 = time.monotonic()
        total_entities = 0
        total_rels = 0
        errors = 0

        for i, text in enumerate(texts):
            try:
                res = lg.insert(text)
                total_entities += res.entities_extracted
                total_rels += res.relationships_extracted
            except Exception as e:
                errors += 1
                if errors <= 2:
                    logger.warning("  ERR doc %d: %s", i, str(e)[:80])
            if errors > max(3, len(texts) // 3):
                logger.warning("  BAIL — too many errors for %s", label)
                break

        insert_duration = time.monotonic() - t0

        # Query
        query_results: list[dict[str, Any]] = []
        t1 = time.monotonic()
        for q in sample_questions:
            try:
                qr = lg.query(q)
                query_results.append({
                    "question": q,
                    "answer_length": len(qr.answer) if qr.answer else 0,
                    "context_chunks": len(qr.context),
                    "duration_sec": qr.duration_seconds,
                })
            except Exception as e:
                query_results.append({"question": q, "error": str(e)})
        query_duration = time.monotonic() - t1

        # Graph stats
        try:
            node_count = lg._graph.execute_query("MATCH (n) RETURN count(n) AS c")[0]["c"]
            rel_count = lg._graph.execute_query("MATCH ()-[r]->() RETURN count(r) AS c")[0]["c"]
        except Exception:
            node_count = total_entities
            rel_count = total_rels

        lg.close()

        results["models"][label] = {
            "provider": provider,
            "model": model,
            "region": region,
            "insert_duration_sec": round(insert_duration, 2),
            "docs_per_minute": round(len(texts) / insert_duration * 60, 1) if insert_duration > 0 else 0,
            "entities_extracted": total_entities,
            "relationships_extracted": total_rels,
            "graph_nodes": node_count,
            "graph_relationships": rel_count,
            "insert_errors": errors,
            "query_duration_sec": round(query_duration, 2),
            "query_results": query_results,
        }

        print(f"  {label}: {total_entities} entities, {total_rels} rels, "
              f"{node_count} nodes, {round(insert_duration, 1)}s, {errors} errs")

    return results


# ---------------------------------------------------------------------------
# Axis registry
# ---------------------------------------------------------------------------

AXIS_RUNNERS: dict[str, Any] = {
    "extraction": _run_extraction,
    "kg-quality": _run_kg_quality,
    "kg-build": _run_kg_build,
    "query": _run_query,
    "throughput": _run_throughput,
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _collect_competitor_published(competitors: list[str]) -> dict[str, Any]:
    published: dict[str, Any] = {}
    if "ms-graphrag" in competitors:
        try:
            from pipeline.benchmarks.competitors.ms_graphrag_runner import (
                get_published_results,
                get_unsupported_capabilities,
            )
            published["ms-graphrag"] = {
                **get_published_results(),
                "unsupported": get_unsupported_capabilities(),
            }
        except Exception:
            logger.exception("Failed to collect MS GraphRAG published results")
    return published


def run(
    axes: list[str],
    competitors: list[str],
    output_dir: Path,
) -> dict[str, Any]:
    """Execute selected benchmark axes and return combined results."""
    results: dict[str, Any] = {
        "metadata": _collect_metadata(competitors),
        "axes": {},
    }

    published = _collect_competitor_published(competitors)
    if published:
        results["competitor_published"] = published

    for axis in axes:
        runner = AXIS_RUNNERS[axis]
        logger.info("Running axis: %s", axis)
        t0 = time.monotonic()
        try:
            axis_result = runner(competitors)
        except Exception:
            logger.exception("Axis %s failed", axis)
            axis_result = {"error": True, "message": f"Axis {axis} failed — see logs"}
        elapsed = time.monotonic() - t0
        axis_result["elapsed_sec"] = round(elapsed, 3)
        results["axes"][axis] = axis_result
        logger.info("Axis %s completed in %.2fs", axis, elapsed)

    # Write results
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    out_path = output_dir / f"benchmark_{ts}.json"
    out_path.write_text(json.dumps(results, indent=2) + "\n")
    logger.info("Results written to %s", out_path)

    return results


def _run_model_comparison(args: argparse.Namespace) -> None:
    """Run extraction axis across all COMPARE_MODELS and print a summary table."""
    global _bedrock_client, _bedrock_client_region

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    os.environ["BENCH_LLM_SERVICE"] = "bedrock"
    output_dir = args.output_dir
    all_results: dict[str, Any] = {"metadata": _collect_metadata([]), "models": {}}

    for entry in COMPARE_MODELS:
        label = entry["label"]
        os.environ["BENCH_LLM_MODEL"] = entry["model"]
        os.environ["AWS_REGION"] = entry["region"]
        _bedrock_client = None  # force new client for region change
        _bedrock_client_region = None

        logger.info("=== Model: %s (%s in %s) ===", label, entry["model"], entry["region"])
        t0 = time.monotonic()
        try:
            result = _run_extraction([])
        except Exception:
            logger.exception("Model %s failed", label)
            result = {"error": True}
        result["elapsed_sec"] = round(time.monotonic() - t0, 3)
        result["model_id"] = entry["model"]
        result["region"] = entry["region"]
        all_results["models"][label] = result

    # Write full results
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    out_path = output_dir / f"model_comparison_{ts}.json"
    out_path.write_text(json.dumps(all_results, indent=2) + "\n")
    logger.info("Results written to %s", out_path)

    # Print summary table
    print("\n" + "=" * 90)
    print(f"{'Model':<30} {'Exact F1':>10} {'Partial F1':>12} {'Time (s)':>10} {'Docs':>6}")
    print("-" * 90)
    for label, r in all_results["models"].items():
        lg = r.get("litegraf", {})
        exact_f1 = lg.get("exact_match", {}).get("micro_avg", {}).get("f1", 0.0)
        partial_f1 = lg.get("partial_match", {}).get("micro_avg", {}).get("f1", 0.0)
        elapsed = r.get("elapsed_sec", 0.0)
        n_docs = r.get("num_documents", 0)
        print(f"{label:<30} {exact_f1:>10.4f} {partial_f1:>12.4f} {elapsed:>10.1f} {n_docs:>6}")
    print("=" * 90)

    print(json.dumps(all_results, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="litegraf benchmark runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--all", action="store_true", help="Run all benchmark axes")
    parser.add_argument("--axis", action="append", choices=VALID_AXES, help="Run a specific axis (repeatable)")
    parser.add_argument("--competitors", nargs="*", default=[], help="Competitor systems to include")
    parser.add_argument("-o", "--output-dir", type=Path, default=Path(__file__).resolve().parent / "results", help="Output directory")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")
    parser.add_argument("--compare-models", action="store_true", help="Run extraction across all 6 Bedrock models (2 per lab)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.compare_models:
        _run_model_comparison(args)
        return

    if args.all:
        axes = list(VALID_AXES)
    elif args.axis:
        axes = list(dict.fromkeys(args.axis))
    else:
        parser.error("Specify --all or at least one --axis")

    results = run(axes, args.competitors, args.output_dir)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
