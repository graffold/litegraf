"""Graphical CLI for litegraf — rich terminal UI with benchmark focus.

Usage::

    litegraf-tui bench --all
    litegraf-tui bench --axis extraction --axis throughput
    litegraf-tui bench --compare-models --docs 10
    litegraf-tui bench --axis kg-build --competitors lightrag ms-graphrag
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


def _print_banner() -> None:
    for l_line, g_line in zip(BANNER_LITE, BANNER_GRAF):
        console.print(f"[{BLUE}]{l_line}[/][{VIOLET}]{g_line}[/]")
    console.print(f"  [{BOLD}]v0.1.0[/]  [{TEXT2}]lightweight knowledge graph benchmark suite[/]")
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


def _render_kg_build(data: dict[str, Any]) -> Table:
    """Render kg-build (multi-model graph construction) results."""
    t = Table(title="🏗️  KG Build (per model)", title_style="bold magenta", border_style="dim")
    t.add_column("Model", style="bold")
    t.add_column("Entities", justify="right")
    t.add_column("Rels", justify="right")
    t.add_column("Nodes", justify="right")
    t.add_column("docs/min", justify="right")
    t.add_column("Time", justify="right")
    t.add_column("Errs", justify="right")

    for label, m in data.get("models", {}).items():
        if "skipped" in m or "error" in m:
            t.add_row(label, f"[dim]{m.get('skipped') or m.get('error', '')[:30]}[/]", "", "", "", "", "")
            continue
        t.add_row(
            label,
            str(m.get("entities_extracted", 0)),
            str(m.get("relationships_extracted", 0)),
            str(m.get("graph_nodes", 0)),
            f"[cyan]{m.get('docs_per_minute', 0):.1f}[/]",
            f"{m.get('insert_duration_sec', 0):.1f}s",
            str(m.get("insert_errors", 0)),
        )
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

    console.print("[yellow]Unrecognized result format[/yellow]")


# ── Insert / Query commands ───────────────────────────────────────────────


def _cmd_insert(args: argparse.Namespace) -> None:
    from pipeline.litegraf import LiteGraf

    _print_banner()
    console.print()

    with console.status("[bold cyan]Inserting…"):
        kg = LiteGraf()
        result = kg.insert(args.text)
        kg.close()

    t = Table(title="Insert Result", border_style="dim")
    t.add_column("Metric", style="bold")
    t.add_column("Value", justify="right")
    t.add_row("Chunks processed", str(result.chunks_processed))
    t.add_row("Entities extracted", f"[cyan]{result.entities_extracted}[/]")
    t.add_row("Relationships", f"[cyan]{result.relationships_extracted}[/]")
    console.print(t)


def _cmd_query(args: argparse.Namespace) -> None:
    from pipeline.litegraf import LiteGraf

    _print_banner()
    console.print()

    with console.status("[bold cyan]Querying…"):
        kg = LiteGraf()
        result = kg.query(args.question, only_context=args.only_context)
        kg.close()

    if result.answer:
        console.print(Panel(result.answer, title="Answer", border_style="green"))

    if result.context:
        t = Table(title="Context Chunks", border_style="dim")
        t.add_column("#", style="dim", width=3)
        t.add_column("Score", justify="right", width=6)
        t.add_column("Text")
        for i, chunk in enumerate(result.context):
            score = getattr(chunk, "score", 0.0)
            text = getattr(chunk, "text", str(chunk))[:120] + "…"
            t.add_row(str(i + 1), f"{score:.2f}", text)
        console.print(t)


def _cmd_status(args: argparse.Namespace) -> None:
    """Show system status: backends, connectivity, versions."""
    _print_banner()
    console.print()

    tree = Tree("[bold]System Status[/bold]")

    # Python
    tree.add(f"Python {sys.version.split()[0]}")

    # Neo4j
    neo4j_branch = tree.add("Neo4j")
    try:
        from neo4j import GraphDatabase

        uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
        driver = GraphDatabase.driver(uri, auth=("neo4j", os.environ.get("NEO4J_PASSWORD", "password")))
        driver.verify_connectivity()
        driver.close()
        neo4j_branch.add(f"[green]✓[/green] Connected ({uri})")
    except Exception as e:
        neo4j_branch.add(f"[red]✗[/red] {e}")

    # Ollama
    ollama_branch = tree.add("Ollama")
    try:
        import urllib.request

        url = os.environ.get("OLLAMA_URL", "http://localhost:11434")
        with urllib.request.urlopen(f"{url}/api/tags", timeout=2) as resp:  # noqa: S310
            tags = json.loads(resp.read())
        models = [m["name"] for m in tags.get("models", [])][:5]
        ollama_branch.add(f"[green]✓[/green] {url} — {len(tags.get('models', []))} models")
        for m in models:
            ollama_branch.add(f"  {m}")
    except Exception as e:
        ollama_branch.add(f"[red]✗[/red] {e}")

    # Sentence-transformers
    st_branch = tree.add("Embeddings")
    try:
        import sentence_transformers

        st_branch.add(f"[green]✓[/green] sentence-transformers {sentence_transformers.__version__}")
    except ImportError:
        st_branch.add("[red]✗[/red] sentence-transformers not installed")

    # Benchmark datasets
    ds_branch = tree.add("Benchmark datasets")
    data_dir = Path(__file__).resolve().parent / "benchmarks" / "data"
    for name in ("bc5cdr", "chemprot", "gad"):
        p = data_dir / name
        if p.exists():
            ds_branch.add(f"[green]✓[/green] {name}")
        else:
            ds_branch.add(f"[dim]○[/dim] {name} (not downloaded)")

    console.print(tree)


# ── Interactive mode ───────────────────────────────────────────────────────

_MENU = {
    "1": ("status",              "Show system status"),
    "2": ("bench --all",         "Run all benchmarks"),
    "3": ("bench --axis extraction", "Benchmark: extraction only"),
    "4": ("bench --axis throughput", "Benchmark: throughput only"),
    "5": ("bench --compare-models",  "Compare LLM providers"),
    "6": ("insert",              "Insert text into KG"),
    "7": ("query",               "Query the KG"),
    "8": ("show",                "Render a result JSON"),
}


def _interactive() -> None:
    """Interactive REPL — stays alive until Ctrl+C."""
    _print_banner()

    # CLI subcommands in blue
    console.print(f"[{BLUE}]CLI[/]")
    cli_cmds = {
        "bench":  "Run benchmarks with graphical output",
        "show":   "Render a previous benchmark result JSON",
        "insert": "Insert text into the knowledge graph",
        "query":  "Query the knowledge graph",
        "status": "Show system status and connectivity",
    }
    for name, desc in cli_cmds.items():
        console.print(f"  [{BLUE}]{name:<10}[/] [{HINT}]{desc}[/]")

    # Interactive menu in violet
    console.print(f"\n[{VIOLET}]Interactive mode[/]  [{HINT}]Ctrl+C to exit[/]\n")
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

            cmd_str, _ = _MENU[choice]

            if cmd_str == "insert":
                text = Prompt.ask(f"  [{TEXT2}]Text to insert[/]")
                if not text:
                    continue
                _do_insert(text)

            elif cmd_str == "query":
                question = Prompt.ask(f"  [{TEXT2}]Question[/]")
                if not question:
                    continue
                _do_query(question)

            elif cmd_str == "show":
                path = Prompt.ask(f"  [{TEXT2}]Path to result JSON[/]")
                if not path:
                    continue
                try:
                    _show_results(path)
                except Exception as e:
                    console.print(f"[red]{e}[/]")

            elif cmd_str == "status":
                _cmd_status(argparse.Namespace())

            elif cmd_str.startswith("bench"):
                parts = cmd_str.split()
                ns = argparse.Namespace(
                    all="--all" in parts,
                    axis=[p for prev, p in zip(parts, parts[1:]) if prev == "--axis"] or None,
                    compare_models="--compare-models" in parts,
                    competitors=[],
                    docs=10,
                    dataset="bc5cdr",
                    providers=None,
                    output_dir=Path(__file__).resolve().parent / "benchmarks" / "results",
                    verbose=False,
                )
                if ns.compare_models:
                    _run_compare_models_live(ns.docs, ns.dataset, ns.providers)
                elif ns.all:
                    _run_benchmark_live(
                        ["extraction", "kg-quality", "kg-build", "query", "throughput"],
                        ns.competitors, ns.output_dir, ns.verbose,
                    )
                elif ns.axis:
                    _run_benchmark_live(ns.axis, ns.competitors, ns.output_dir, ns.verbose)

            console.print()

    except KeyboardInterrupt:
        console.print(f"\n[{HINT}]bye[/]")


def _do_insert(text: str) -> None:
    """Insert text — used by interactive mode."""
    from pipeline.litegraf import LiteGraf

    with console.status("[bold cyan]Inserting…"):
        kg = LiteGraf()
        result = kg.insert(text)
        kg.close()

    t = Table(border_style="dim")
    t.add_column("Metric", style="bold")
    t.add_column("Value", justify="right")
    t.add_row("Chunks", str(result.chunks_processed))
    t.add_row("Entities", f"[cyan]{result.entities_extracted}[/]")
    t.add_row("Relationships", f"[cyan]{result.relationships_extracted}[/]")
    console.print(t)


def _do_query(question: str, only_context: bool = False) -> None:
    """Query KG — used by interactive mode."""
    from pipeline.litegraf import LiteGraf

    with console.status("[bold cyan]Querying…"):
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


# ── Entrypoint ────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(prog="litegraf-tui", description="litegraf graphical CLI")
    subs = parser.add_subparsers(dest="command")

    # bench
    bp = subs.add_parser("bench", help="Run benchmarks with graphical output")
    bp.add_argument("--all", action="store_true", help="Run all benchmark axes")
    bp.add_argument("--axis", action="append", choices=["extraction", "kg-quality", "query", "throughput"])
    bp.add_argument("--competitors", nargs="*", default=[])
    bp.add_argument("--compare-models", action="store_true", help="Run multi-model provider comparison")
    bp.add_argument("--docs", type=int, default=10, help="Documents per model (for --compare-models)")
    bp.add_argument("--dataset", default="bc5cdr", choices=["bc5cdr", "chemprot", "gad"])
    bp.add_argument("--providers", nargs="*", help="Filter providers (for --compare-models)")
    bp.add_argument("-o", "--output-dir", type=Path, default=Path(__file__).resolve().parent / "benchmarks" / "results")
    bp.add_argument("-v", "--verbose", action="store_true")

    # show (render previous results)
    sp = subs.add_parser("show", help="Render a previous benchmark result JSON")
    sp.add_argument("file", help="Path to benchmark JSON file")

    # insert
    ip = subs.add_parser("insert", help="Insert text into the knowledge graph")
    ip.add_argument("text", help="Text to insert")

    # query
    qp = subs.add_parser("query", help="Query the knowledge graph")
    qp.add_argument("question", help="Question to ask")
    qp.add_argument("--only-context", action="store_true")

    # status
    subs.add_parser("status", help="Show system status and connectivity")

    args = parser.parse_args()

    if not args.command:
        _interactive()
        return

    if args.command == "bench":
        if args.compare_models:
            _run_compare_models_live(args.docs, args.dataset, args.providers)
        elif args.all:
            _run_benchmark_live(
                ["extraction", "kg-quality", "query", "throughput"],
                args.competitors, args.output_dir, args.verbose,
            )
        elif args.axis:
            _run_benchmark_live(list(dict.fromkeys(args.axis)), args.competitors, args.output_dir, args.verbose)
        else:
            console.print("[red]Specify --all, --axis, or --compare-models[/red]")
    elif args.command == "show":
        _show_results(args.file)
    elif args.command == "insert":
        _cmd_insert(args)
    elif args.command == "query":
        _cmd_query(args)
    elif args.command == "status":
        _cmd_status(args)


if __name__ == "__main__":
    main()
