"""Graphical CLI for litegraf — rich terminal UI with benchmark focus.

Usage::

    litegraf-tui                                    # interactive mode
    litegraf-tui bench --all                        # run all 4 benchmark axes
    litegraf-tui bench --axis extraction --axis throughput
    litegraf-tui bench --compare-models --docs 10   # multi-model leaderboard
    litegraf-tui insert "TP53 is associated with breast cancer."
    litegraf-tui query "What cancers are associated with TP53?"
    litegraf-tui status
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rich.columns import Columns
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.prompt import Prompt
from rich.table import Table
from rich.text import Text
from rich.tree import Tree

console = Console()

# ── Brand palette (matches graffold.com) ──────────────────────────────────

BLUE = "bold #06B6D4"
VIOLET = "bold #A78BFA"
EMERALD = "#34D399"
TEXT = "#FAFAFA"
TEXT2 = "#A1A1AA"
HINT = "#71717A"
BOLD = f"bold {TEXT}"

BANNER_LITE = [
    "▀▀      ▀▀▀▀▀▀▀ ▀▀▀▀▀▀▀ ▀▀▀▀▀▀▀ ",
    "▀▀        ▀▀▀     ▀▀▀   ▀▀      ",
    "▀▀        ▀▀▀     ▀▀▀   ▀▀▀▀▀   ",
    "▀▀        ▀▀▀     ▀▀▀   ▀▀      ",
    "▀▀▀▀▀▀▀ ▀▀▀▀▀▀▀   ▀▀▀   ▀▀▀▀▀▀▀ ",
]

BANNER_GRAF = [
    " ▀▀▀▀▀  ▀▀▀▀▀▀   ▀▀▀▀▀  ▀▀▀▀▀▀▀",
    "▀▀      ▀▀   ▀▀ ▀▀   ▀▀ ▀▀     ",
    "▀▀  ▀▀▀ ▀▀▀▀▀▀  ▀▀▀▀▀▀▀ ▀▀▀▀▀  ",
    "▀▀   ▀▀ ▀▀  ▀▀  ▀▀   ▀▀ ▀▀     ",
    " ▀▀▀▀▀▀ ▀▀   ▀▀ ▀▀   ▀▀ ▀▀     ",
]


def _detect_status() -> dict[str, tuple[str, str]]:
    """Detect live service status — returns {label: (value, hint)}."""
    status: dict[str, tuple[str, str]] = {}

    # Ollama
    try:
        import urllib.request
        url = os.environ.get("OLLAMA_URL", "http://localhost:11434")
        with urllib.request.urlopen(f"{url}/api/tags", timeout=2) as resp:  # noqa: S310
            tags = json.loads(resp.read())
        n = len(tags.get("models", []))
        status["Ollama"] = (url, f"{n} models")
    except Exception:
        status["Ollama"] = ("not running", "")

    # Bedrock
    try:
        import boto3
        sts = boto3.client("sts")
        ident = sts.get_caller_identity()
        status["Bedrock"] = (ident.get("Account", "?"), ident.get("Arn", "").split("/")[-1])
    except Exception:
        status["Bedrock"] = ("no credentials", "aws sso login")

    # Cloudflare
    cf_acct = os.environ.get("CF_ACCOUNT_ID", "")
    cf_tok = os.environ.get("CF_API_TOKEN", "")
    if cf_acct and cf_tok:
        status["Cloudflare"] = (cf_acct[:8] + "…", "Workers AI")
    else:
        status["Cloudflare"] = ("not configured", "set CF_ACCOUNT_ID + CF_API_TOKEN")

    # Embeddings
    try:
        import sentence_transformers
        status["Embeddings"] = ("sentence-transformers", sentence_transformers.__version__)
    except ImportError:
        status["Embeddings"] = ("not installed", "pip install sentence-transformers")

    # Datasets
    data_dir = Path(__file__).resolve().parent / "benchmarks" / "data"
    present = [n for n in ("bc5cdr", "chemprot", "gad") if (data_dir / n).exists()]
    status["Datasets"] = (f"{len(present)}/3 cached", "auto-download on run" if len(present) < 3 else "ready")

    return status


def _print_banner() -> None:
    for l_line, g_line in zip(BANNER_LITE, BANNER_GRAF):
        console.print(f"[{BLUE}]{l_line}[/][{VIOLET}]{g_line}[/]")
    console.print(f"  [{BOLD}]v0.1.0[/]  [{TEXT2}]lightweight knowledge graph benchmark suite[/]")
    console.print()

    # Inline status block
    for label, (val, hint) in _detect_status().items():
        if hint:
            console.print(f"  [{TEXT}]{label:<14}[/][{BLUE}]{val}[/]  [{HINT}]({hint})[/]")
        else:
            console.print(f"  [{TEXT}]{label:<14}[/][{BLUE}]{val}[/]")
    console.print()


def _header() -> Panel:
    lines = "\n".join(
        f"[{BLUE}]{l}[/][{VIOLET}]{g}[/]"
        for l, g in zip(BANNER_LITE, BANNER_GRAF)
    )
    return Panel(lines, border_style="cyan", expand=False)


# ── Benchmark result renderers ────────────────────────────────────────────


def _f1_color(f1: float) -> str:
    if f1 >= 0.7:
        return "green"
    if f1 >= 0.4:
        return "yellow"
    return "red"


def _bar(value: float, width: int = 20) -> str:
    filled = int(value * width)
    return f"[green]{'█' * filled}[/green][dim]{'░' * (width - filled)}[/dim]"


def _render_extraction(data: dict[str, Any]) -> Table:
    """Render extraction metrics as a rich table."""
    t = Table(title="🔬 Extraction (NER)", title_style="bold magenta", border_style="dim")
    t.add_column("Model", style="bold")
    t.add_column("Exact F1", justify="right")
    t.add_column("Partial F1", justify="right")
    t.add_column("", min_width=22)  # bar
    t.add_column("Time", justify="right")
    t.add_column("Errs", justify="right")

    # litegraf's own result
    lg = data.get("litegraf", {})
    if lg:
        ef1 = lg.get("exact_match", {}).get("micro_avg", {}).get("f1", 0.0)
        pf1 = lg.get("partial_match", {}).get("micro_avg", {}).get("f1", 0.0)
        t.add_row(
            "litegraf",
            f"[{_f1_color(ef1)}]{ef1:.4f}[/]",
            f"[{_f1_color(pf1)}]{pf1:.4f}[/]",
            _bar(pf1),
            f"{data.get('elapsed_sec', 0):.1f}s",
            "",
        )

    # competitors
    for key in data.get("competitors", []):
        comp = data.get(key, {})
        if "error" in comp:
            t.add_row(key, f"[red]{comp['error'][:40]}[/]", "", "", "", "")
            continue
        ef1 = comp.get("exact_match", {}).get("micro_avg", {}).get("f1", 0.0)
        pf1 = comp.get("partial_match", {}).get("micro_avg", {}).get("f1", 0.0)
        t.add_row(key, f"[{_f1_color(ef1)}]{ef1:.4f}[/]", f"[{_f1_color(pf1)}]{pf1:.4f}[/]", _bar(pf1), "", "")

    return t


def _render_throughput(data: dict[str, Any]) -> Table:
    """Render throughput metrics."""
    t = Table(title="⚡ Throughput", title_style="bold magenta", border_style="dim")
    t.add_column("Metric", style="bold")
    t.add_column("Value", justify="right")

    ing = data.get("ingestion_throughput", {})
    if ing.get("pubmed"):
        t.add_row("PubMed docs/min", f"[cyan]{ing['pubmed'].get('docs_per_minute', 0):.1f}[/]")

    emb = data.get("embedding_rate", {})
    if emb.get("vectors_per_second"):
        t.add_row("Embedding vec/s", f"[cyan]{emb['vectors_per_second']:.1f}[/]")

    pipe = data.get("pipeline_time", {})
    if pipe.get("total_seconds"):
        t.add_row("Pipeline total", f"{pipe['total_seconds']:.1f}s")

    mem = data.get("memory_peak", {})
    if mem.get("peak_rss_mb"):
        t.add_row("Peak RSS", f"{mem['peak_rss_mb']:.0f} MB")

    cost = data.get("token_cost", {})
    if cost.get("estimated_cost_per_1k_docs"):
        t.add_row("Cost / 1k docs", f"${cost['estimated_cost_per_1k_docs']:.4f}")

    t.add_row("Success rate", f"{data.get('extraction_success_rate', 0) * 100:.1f}%")
    return t


def _render_query(data: dict[str, Any]) -> Table:
    """Render query performance metrics."""
    t = Table(title="🔍 Query Performance", title_style="bold magenta", border_style="dim")
    t.add_column("Metric", style="bold")
    t.add_column("Value", justify="right")

    rel = data.get("answer_relevance", {})
    if rel.get("mean_score"):
        score = rel["mean_score"]
        t.add_row("Answer relevance", f"[{_f1_color(score / 5)}]{score:.2f}[/] / 5.0")

    mode = data.get("mode_routing", {})
    if mode.get("accuracy") is not None:
        acc = mode["accuracy"]
        t.add_row("Mode routing acc", f"[{_f1_color(acc)}]{acc:.1%}[/]")

    hop = data.get("multi_hop_success", {})
    if hop.get("success_rate") is not None:
        t.add_row("Multi-hop success", f"{hop['success_rate']:.1%}")

    lat = data.get("latency", {})
    if lat.get("p50_ms"):
        t.add_row("Latency p50", f"{lat['p50_ms']:.0f} ms")
    if lat.get("p95_ms"):
        t.add_row("Latency p95", f"{lat['p95_ms']:.0f} ms")

    return t


def _render_kg_quality(data: dict[str, Any]) -> Table:
    """Render KG quality metrics."""
    t = Table(title="🧬 KG Quality", title_style="bold magenta", border_style="dim")
    t.add_column("Metric", style="bold")
    t.add_column("Value", justify="right")
    t.add_column("", min_width=22)

    consol = data.get("consolidation", {})
    if consol.get("accuracy") is not None:
        v = consol["accuracy"]
        t.add_row("Consolidation acc", f"[{_f1_color(v)}]{v:.1%}[/]", _bar(v))
    if consol.get("false_merge_rate") is not None:
        v = consol["false_merge_rate"]
        color = "green" if v < 0.1 else "yellow" if v < 0.3 else "red"
        t.add_row("False merge rate", f"[{color}]{v:.1%}[/]", "")

    prov = data.get("provenance", {})
    if prov.get("completeness") is not None:
        v = prov["completeness"]
        t.add_row("Provenance", f"[{_f1_color(v)}]{v:.1%}[/]", _bar(v))

    contra = data.get("contradiction_detection", {})
    if contra.get("recall") is not None:
        v = contra["recall"]
        t.add_row("Contradiction recall", f"[{_f1_color(v)}]{v:.1%}[/]", _bar(v))
    if contra.get("precision") is not None:
        v = contra["precision"]
        t.add_row("Contradiction prec", f"[{_f1_color(v)}]{v:.1%}[/]", "")

    mapping = data.get("mapping_rates", {})
    if mapping.get("uniprot_rate") is not None:
        t.add_row("UniProt mapped", f"{mapping['uniprot_rate']:.1%}", "")
    if mapping.get("mondo_rate") is not None:
        t.add_row("MONDO mapped", f"{mapping['mondo_rate']:.1%}", "")

    return t




def _render_model_comparison(data: dict[str, Any]) -> Table:
    """Render provider comparison table."""
    t = Table(title="🏆 Model Comparison", title_style="bold magenta", border_style="dim")
    t.add_column("Model", style="bold")
    t.add_column("Provider", style="dim")
    t.add_column("Exact F1", justify="right")
    t.add_column("Partial F1", justify="right")
    t.add_column("", min_width=22)
    t.add_column("Time", justify="right")
    t.add_column("Errs", justify="right")

    rows: list[tuple[float, str, dict]] = []
    for label, r in data.get("models", {}).items():
        if "skipped" in r:
            t.add_row(label, "", f"[dim]SKIP: {r['skipped'][:30]}[/]", "", "", "", "")
            continue
        ef1 = r.get("exact_match", {}).get("micro_avg", {}).get("f1", 0.0)
        rows.append((ef1, label, r))

    for ef1, label, r in sorted(rows, reverse=True):
        pf1 = r.get("partial_match", {}).get("micro_avg", {}).get("f1", 0.0)
        t.add_row(
            label,
            r.get("provider", ""),
            f"[{_f1_color(ef1)}]{ef1:.4f}[/]",
            f"[{_f1_color(pf1)}]{pf1:.4f}[/]",
            _bar(pf1),
            f"{r.get('elapsed_sec', 0):.1f}s",
            str(r.get("errors", 0)),
        )
    return t


AXIS_RENDERERS = {
    "extraction": _render_extraction,
    "throughput": _render_throughput,
    "query": _render_query,
    "kg-quality": _render_kg_quality,
}


# ── Benchmark commands ────────────────────────────────────────────────────


def _run_benchmark_live(axes: list[str], competitors: list[str], output_dir: Path, verbose: bool) -> None:
    """Run benchmark axes with live progress display."""
    import logging

    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    from pipeline.benchmarks.run_benchmark import AXIS_RUNNERS, _collect_metadata

    _print_banner()
    console.print()

    results: dict[str, Any] = {"metadata": _collect_metadata(competitors), "axes": {}}

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=30),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    )

    with progress:
        task = progress.add_task("Benchmark axes", total=len(axes))
        for axis in axes:
            progress.update(task, description=f"Running [cyan]{axis}[/cyan]")
            runner = AXIS_RUNNERS[axis]
            t0 = time.monotonic()
            try:
                axis_result = runner(competitors)
            except Exception as e:
                axis_result = {"error": True, "message": str(e)}
            axis_result["elapsed_sec"] = round(time.monotonic() - t0, 3)
            results["axes"][axis] = axis_result
            progress.advance(task)

    # Render results
    console.print()
    panels: list[Any] = []
    for axis, data in results["axes"].items():
        renderer = AXIS_RENDERERS.get(axis)
        if renderer and "error" not in data:
            panels.append(renderer(data))

    if panels:
        console.print(Columns(panels, equal=True, expand=True) if len(panels) <= 2 else Group(*panels))

    # Metadata footer
    meta = results["metadata"]
    hw = meta.get("hardware", {})
    footer = (
        f"[dim]litegraf {meta.get('graffold_version', '?')} · "
        f"Python {meta.get('python_version', '?')} · "
        f"{meta.get('llm_model', '?')} ({meta.get('llm_service', '?')}) · "
        f"{hw.get('cpu_model', hw.get('processor', '?'))} · "
        f"{hw.get('ram_gb', '?')} GB RAM[/dim]"
    )
    console.print(Panel(footer, border_style="dim"))

    # Write JSON
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    out_path = output_dir / f"benchmark_{ts}.json"
    out_path.write_text(json.dumps(results, indent=2) + "\n")
    console.print(f"\n[dim]Results → {out_path}[/dim]")


def _run_compare_models_live(docs: int, dataset: str, providers: list[str] | None) -> None:
    """Run model comparison with live progress."""
    from pipeline.benchmarks.compare_providers import MODELS, run_comparison

    _print_banner()
    console.print()

    models = MODELS
    if providers:
        models = [m for m in models if m["provider"] in providers]

    console.print(f"[bold]Comparing {len(models)} models × {docs} docs on {dataset}[/bold]\n")

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=30),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    )

    with progress:
        task = progress.add_task("Models", total=len(models))
        # run_comparison does its own iteration, so we just show a spinner
        progress.update(task, description="Running comparison…")
        results = run_comparison(models, max_docs=docs, dataset=dataset)
        progress.update(task, completed=len(models))

    console.print()
    console.print(_render_model_comparison(results))

    # Write JSON
    out_dir = Path(__file__).resolve().parent / "benchmarks" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    out_path = out_dir / f"providers_{ts}.json"
    out_path.write_text(json.dumps(results, indent=2) + "\n")
    console.print(f"\n[dim]Results → {out_path}[/dim]")


def _show_results(path: str) -> None:
    """Load and render a previous benchmark JSON result file."""
    data = json.loads(Path(path).read_text())
    _print_banner()
    console.print()

    # Full benchmark result
    if "axes" in data:
        for axis, axis_data in data["axes"].items():
            renderer = AXIS_RENDERERS.get(axis)
            if renderer and "error" not in axis_data:
                console.print(renderer(axis_data))
                console.print()
        return

    # Model comparison result
    if "models" in data:
        console.print(_render_model_comparison(data))
        return

    # PDF benchmark result
    if "tool_summaries" in data:
        console.print(_render_pdf_bench(data))
        return

    console.print("[yellow]Unrecognized result format[/yellow]")


# ── Insert / Query commands ───────────────────────────────────────────────


# ── Shared commands (used by both CLI and interactive) ────────────────────


def _do_insert(text: str) -> None:
    from pipeline.litegraf import LiteGraf

    with console.status(f"[{BLUE}]Inserting…[/]"):
        kg = LiteGraf()
        result = kg.insert(text)
        kg.close()

    t = Table(border_style="dim")
    t.add_column("Metric", style="bold")
    t.add_column("Value", justify="right")
    t.add_row("Chunks", str(result.chunks_processed))
    t.add_row("Entities", f"[{EMERALD}]{result.entities_extracted}[/]")
    t.add_row("Relationships", f"[{EMERALD}]{result.relationships_extracted}[/]")
    console.print(t)


def _do_query(question: str, only_context: bool = False) -> None:
    from pipeline.litegraf import LiteGraf

    with console.status(f"[{BLUE}]Querying…[/]"):
        kg = LiteGraf()
        result = kg.query(question, only_context=only_context)
        kg.close()

    if result.answer:
        console.print(Panel(result.answer, title="Answer", border_style="green"))
    if result.context:
        t = Table(title="Context", border_style="dim")
        t.add_column("#", style="dim", width=3)
        t.add_column("Score", justify="right", width=6)
        t.add_column("Text")
        for i, chunk in enumerate(result.context):
            score = getattr(chunk, "score", 0.0)
            text = getattr(chunk, "text", str(chunk))[:120] + "…"
            t.add_row(str(i + 1), f"{score:.2f}", text)
        console.print(t)


# ── Guided benchmark config builder ──────────────────────────────────────


def _pick_multi(label: str, options: dict[str, str], default: str = "") -> list[str]:
    for k, v in options.items():
        console.print(f"    [{VIOLET}]{k}[/] [{HINT}]{v}[/]")
    raw = Prompt.ask(f"  [{HINT}]{label} (comma-separated)[/]", default=default)
    keys = [k.strip() for k in raw.split(",")]
    return [options[k] for k in keys if k in options]


def _pick_one(label: str, options: dict[str, str], default: str = "") -> str:
    for k, v in options.items():
        console.print(f"    [{VIOLET}]{k}[/] [{HINT}]{v}[/]")
    raw = Prompt.ask(f"  [{HINT}]{label}[/]", default=default)
    return options.get(raw.strip(), options.get(default, ""))


def _configure_benchmark() -> dict[str, Any] | None:
    """Guided benchmark setup — returns config dict or None to cancel."""
    console.print(f"\n  [{BLUE}]Configure Benchmark[/]\n")

    # A) Mode
    console.print(f"  [{BLUE}]A) Benchmark type[/]")
    mode = _pick_one("Type", {
        "1": "axes",
        "2": "compare",
    }, default="1")

    if mode == "compare":
        # Model comparison — ask dataset + docs
        console.print(f"\n  [{BLUE}]B) Dataset[/]")
        dataset = _pick_one("Dataset", {
            "1": "bc5cdr",
            "2": "chemprot",
            "3": "gad",
        }, default="1")

        console.print(f"\n  [{BLUE}]C) Documents per model[/]")
        docs = int(Prompt.ask(f"  [{HINT}]Docs[/]", default="10"))

        console.print(f"\n  [{BLUE}]D) Providers[/]")
        providers = _pick_multi("Providers", {
            "1": "bedrock",
            "2": "cloudflare",
            "3": "ollama",
        }, default="1,2,3")

        console.print()
        return {"mode": "compare", "dataset": dataset, "docs": docs, "providers": providers or None}

    # Axis benchmark
    console.print(f"\n  [{BLUE}]B) Axes[/]")
    axes = _pick_multi("Axes", {
        "1": "extraction",
        "2": "kg-quality",
        "3": "query",
        "4": "throughput",
    }, default="1,2,3,4")
    if not axes:
        axes = ["extraction", "kg-quality", "query", "throughput"]

    console.print(f"\n  [{BLUE}]C) Dataset[/]")
    dataset = _pick_one("Dataset", {
        "1": "bc5cdr",
        "2": "chemprot",
        "3": "gad",
    }, default="1")

    console.print(f"\n  [{BLUE}]D) Max documents[/]")
    docs = int(Prompt.ask(f"  [{HINT}]Docs[/]", default="10"))

    console.print(f"\n  [{BLUE}]E) LLM service[/]")
    service = _pick_one("Service", {
        "1": "bedrock",
        "2": "ollama",
        "3": "cloudflare",
    }, default="1")

    console.print(f"\n  [{BLUE}]F) Competitors[/]")
    competitors = _pick_multi("Include", {
        "1": "nano-graphrag",
        "2": "lightrag",
        "3": "ms-graphrag",
    }, default="")

    console.print()
    return {
        "mode": "axes", "axes": axes, "dataset": dataset, "docs": docs,
        "service": service, "competitors": competitors,
    }


def _run_configured_benchmark(cfg: dict[str, Any]) -> None:
    """Execute a benchmark from the guided config."""
    output_dir = Path(__file__).resolve().parent / "benchmarks" / "results"

    if cfg["mode"] == "compare":
        provs = cfg.get("providers")
        _run_compare_models_live(cfg["docs"], cfg["dataset"], provs)
    else:
        os.environ["BENCH_MAX_DOCS"] = str(cfg["docs"])
        os.environ["BENCH_DATASET"] = cfg["dataset"]
        os.environ["BENCH_LLM_SERVICE"] = cfg.get("service", "bedrock")
        _run_benchmark_live(cfg["axes"], cfg.get("competitors", []), output_dir, verbose=False)


# ── PDF-to-Markdown benchmark ─────────────────────────────────────────────


def _render_pdf_bench(data: dict[str, Any]) -> Table:
    """Render PDF benchmark results as a rich Table."""
    t = Table(title="PDF → Markdown Tool Benchmark", border_style="dim", title_style=BLUE)
    t.add_column("Tool", style="bold")
    t.add_column("OK", justify="right")
    t.add_column("Avg Chars", justify="right")
    t.add_column("Headings", justify="right")
    t.add_column("Lists", justify="right")
    t.add_column("Tables", justify="right")
    t.add_column("Avg Time", justify="right")
    t.add_column("Total", justify="right")

    for _name, s in data.get("tool_summaries", {}).items():
        ok = f"[{EMERALD}]{s['pdfs_succeeded']}/{s['pdfs_attempted']}[/]"
        t.add_row(
            s["tool"],
            ok,
            f"{s['avg_chars']:,.0f}",
            f"{s['avg_headings']:.1f}",
            f"{s['avg_list_items']:.1f}",
            f"{s['avg_tables']:.1f}",
            f"{s['avg_elapsed_sec']:.2f}s",
            f"{s['total_elapsed_sec']:.1f}s",
        )
    return t


def _run_pdf_bench_live(pdf_paths: list[str], tools: list[str] | None = None) -> dict[str, Any]:
    """Run PDF benchmark with a live spinner."""
    from pipeline.benchmarks.pdf_bench import run_pdf_benchmark

    out_dir = Path(__file__).resolve().parent / "benchmarks" / "results"
    md_dir = out_dir / "pdfbench_output"

    with console.status(f"[{BLUE}]Running PDF benchmark on {len(pdf_paths)} file(s)…[/]"):
        results = run_pdf_benchmark(pdf_paths, tools=tools, output_dir=str(md_dir))

    console.print(_render_pdf_bench(results))

    # Save JSON results
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    out_path = out_dir / f"pdfbench_{ts}.json"
    out_path.write_text(json.dumps(results, indent=2) + "\n")
    console.print(f"\n  [{HINT}]Results  → {out_path}[/]")
    console.print(f"  [{HINT}]Markdown → {md_dir}/[/]")
    return results


def _do_pdf_benchmark_interactive() -> None:
    """Interactive PDF benchmark setup."""
    from pipeline.benchmarks.pdf_bench import available_tools, _TOOLS

    console.print(f"\n  [{BLUE}]PDF-to-Markdown Benchmark[/]\n")

    # Show tool availability
    avail = available_tools()
    for name, ok in avail.items():
        status = f"[{EMERALD}]✓[/]" if ok else f"[red]✗[/]  [{HINT}]{_TOOLS[name]['install']}[/]"
        console.print(f"    {name:<14} {status}")
    console.print()

    active = [t for t, ok in avail.items() if ok]
    if not active:
        console.print("[red]No PDF tools available. Install at least one.[/]")
        return

    # Get PDF paths
    raw = Prompt.ask(f"  [{HINT}]PDF file(s) — comma-separated paths or glob[/]")
    if not raw.strip():
        return

    import glob
    pdf_paths: list[str] = []
    for part in raw.split(","):
        part = part.strip()
        expanded = glob.glob(part)
        if expanded:
            pdf_paths.extend(expanded)
        elif Path(part).is_file():
            pdf_paths.append(part)
        else:
            console.print(f"  [yellow]Not found: {part}[/]")

    if not pdf_paths:
        console.print("[red]No valid PDF files found.[/]")
        return

    console.print(f"  [{HINT}]Found {len(pdf_paths)} PDF(s), running {len(active)} tool(s)…[/]\n")
    _run_pdf_bench_live(pdf_paths, tools=active)


# ── Interactive mode ──────────────────────────────────────────────────────

_MENU = {
    "1": ("status",    "Show system status"),
    "2": ("bench",     "Run benchmark (guided setup)"),
    "3": ("insert",    "Insert text into KG"),
    "4": ("query",     "Query the KG"),
    "5": ("show",      "Render a result JSON"),
    "6": ("dashboard", "Open HTML dashboard"),
    "7": ("pdfbench",  "PDF-to-Markdown tool benchmark"),
}


def _interactive() -> None:
    """Interactive REPL — stays alive until Ctrl+C."""
    _print_banner()

    console.print(f"[{VIOLET}]Interactive mode[/]  [{HINT}]Ctrl+C to exit[/]\n")
    for key, (_, desc) in _MENU.items():
        console.print(f"  [{VIOLET}]{key}[/]  [{TEXT2}]{desc}[/]")
    console.print()

    try:
        while True:
            try:
                choice = Prompt.ask(f"[{EMERALD}]litegraf ❯[/]", choices=[*_MENU, "q", "quit", "help"], show_choices=False, default="help")
            except EOFError:
                break

            if choice in ("q", "quit"):
                break

            if choice == "help":
                for key, (_, desc) in _MENU.items():
                    console.print(f"  [{VIOLET}]{key}[/]  [{TEXT2}]{desc}[/]")
                console.print(f"  [{VIOLET}]q[/]  [{TEXT2}]Quit[/]")
                continue

            cmd, _ = _MENU[choice]

            if cmd == "status":
                _print_banner()

            elif cmd == "bench":
                cfg = _configure_benchmark()
                if cfg:
                    _run_configured_benchmark(cfg)

            elif cmd == "insert":
                text = Prompt.ask(f"  [{HINT}]Text to insert[/]")
                if text:
                    _do_insert(text)

            elif cmd == "query":
                question = Prompt.ask(f"  [{HINT}]Question[/]")
                if question:
                    _do_query(question)

            elif cmd == "show":
                path = Prompt.ask(f"  [{HINT}]Path to result JSON[/]")
                if path:
                    try:
                        _show_results(path)
                    except Exception as e:
                        console.print(f"[red]{e}[/]")

            elif cmd == "dashboard":
                import subprocess
                html = Path(__file__).resolve().parent.parent.parent / "docs" / "index.html"
                if html.exists():
                    subprocess.run(["open", str(html)], check=False)
                    console.print(f"  [{EMERALD}]✓[/] Opened {html.name}")
                else:
                    console.print(f"[red]Dashboard not found at {html}[/]")

            elif cmd == "pdfbench":
                _do_pdf_benchmark_interactive()

            console.print()

    except KeyboardInterrupt:
        console.print(f"\n[{HINT}]bye[/]")


# ── CLI entrypoint ────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(prog="litegraf-tui", description="litegraf graphical CLI")
    subs = parser.add_subparsers(dest="command")

    # bench
    bp = subs.add_parser("bench", help="Run benchmarks")
    bp.add_argument("--all", action="store_true", help="Run all benchmark axes")
    bp.add_argument("--axis", action="append", choices=["extraction", "kg-quality", "query", "throughput"])
    bp.add_argument("--competitors", nargs="*", default=[])
    bp.add_argument("--compare-models", action="store_true", help="Multi-model provider comparison")
    bp.add_argument("--docs", type=int, default=10)
    bp.add_argument("--dataset", default="bc5cdr", choices=["bc5cdr", "chemprot", "gad"])
    bp.add_argument("--providers", nargs="*")
    bp.add_argument("-o", "--output-dir", type=Path, default=Path(__file__).resolve().parent / "benchmarks" / "results")
    bp.add_argument("-v", "--verbose", action="store_true")

    # show
    sp = subs.add_parser("show", help="Render a previous benchmark result JSON")
    sp.add_argument("file", help="Path to benchmark JSON file")

    # insert
    ip = subs.add_parser("insert", help="Insert text into the knowledge graph")
    ip.add_argument("text")

    # query
    qp = subs.add_parser("query", help="Query the knowledge graph")
    qp.add_argument("question")
    qp.add_argument("--only-context", action="store_true")

    # status
    subs.add_parser("status", help="Show system status")

    # pdfbench
    pb = subs.add_parser("pdfbench", help="PDF-to-Markdown tool benchmark")
    pb.add_argument("pdfs", nargs="+", help="PDF file paths or globs")
    pb.add_argument("--tools", nargs="*", help="Tool names to include (default: all available)")

    args = parser.parse_args()

    if not args.command:
        _interactive()
        return

    _print_banner()

    if args.command == "bench":
        if args.compare_models:
            _run_compare_models_live(args.docs, args.dataset, args.providers)
        elif args.all:
            _run_benchmark_live(["extraction", "kg-quality", "query", "throughput"], args.competitors, args.output_dir, args.verbose)
        elif args.axis:
            _run_benchmark_live(list(dict.fromkeys(args.axis)), args.competitors, args.output_dir, args.verbose)
        else:
            console.print("[red]Specify --all, --axis, or --compare-models[/red]")
    elif args.command == "show":
        _show_results(args.file)
    elif args.command == "insert":
        _do_insert(args.text)
    elif args.command == "query":
        _do_query(args.question, args.only_context)
    elif args.command == "status":
        _print_banner()
    elif args.command == "pdfbench":
        import glob as _glob
        pdf_paths: list[str] = []
        for p in args.pdfs:
            expanded = _glob.glob(p)
            pdf_paths.extend(expanded if expanded else [p])
        if not pdf_paths:
            console.print("[red]No PDF files found.[/]")
            sys.exit(1)
        _run_pdf_bench_live(pdf_paths, tools=args.tools)


if __name__ == "__main__":
    main()
