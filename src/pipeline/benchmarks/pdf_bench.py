"""PDF-to-Markdown tool benchmark.

Compares PDF extraction tools on biomedical papers, measuring:
- Extraction quality (character yield, markdown structure)
- Speed (seconds per page)
- Availability (is the tool installed?)

Supported tools:
- **pymupdf** — baseline, always available (``pip install pymupdf``)
- **markitdown** — Microsoft MarkItDown (``pip install markitdown``)
- **pspdfkit** — Nutrient / PSPDFKit CLI (``npm i -g @pspdfkit/pdf-to-markdown``)

Usage::

    # From TUI interactive menu
    litegraf-tui  →  option 7 (PDF benchmark)

    # CLI
    litegraf-tui pdfbench paper1.pdf paper2.pdf

    # Programmatic
    from pipeline.benchmarks.pdf_bench import run_pdf_benchmark
    results = run_pdf_benchmark(pdf_paths=["paper1.pdf", "paper2.pdf"])
"""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result model
# ---------------------------------------------------------------------------


@dataclass
class ToolResult:
    """Result from a single tool x single PDF."""

    tool: str
    pdf: str
    success: bool = False
    markdown: str = ""
    char_count: int = 0
    heading_count: int = 0
    list_item_count: int = 0
    table_count: int = 0
    elapsed_sec: float = 0.0
    error: str = ""


@dataclass
class ToolSummary:
    """Aggregate metrics for one tool across all PDFs."""

    tool: str
    available: bool = True
    pdfs_attempted: int = 0
    pdfs_succeeded: int = 0
    avg_chars: float = 0.0
    avg_headings: float = 0.0
    avg_list_items: float = 0.0
    avg_tables: float = 0.0
    avg_elapsed_sec: float = 0.0
    total_elapsed_sec: float = 0.0
    errors: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Markdown quality helpers
# ---------------------------------------------------------------------------


def _count_md_features(text: str) -> dict[str, int]:
    """Count markdown structural features in extracted text."""
    return {
        "headings": len(re.findall(r"^#{1,6}\s", text, re.MULTILINE)),
        "list_items": len(re.findall(
            r"^[\s]*[-*+]\s|^\s*\d+\.\s", text, re.MULTILINE
        )),
        "tables": text.count("|---") + text.count("| ---"),
    }


# ---------------------------------------------------------------------------
# Tool runners
# ---------------------------------------------------------------------------


def _run_pymupdf(pdf_path: str) -> str:
    """Baseline: PyMuPDF plain-text extraction."""
    import fitz  # type: ignore[import-untyped]

    doc = fitz.open(pdf_path)
    pages = [page.get_text() for page in doc]
    doc.close()
    return "\n\n".join(pages)


def _run_markitdown(pdf_path: str) -> str:
    """Microsoft MarkItDown."""
    from markitdown import MarkItDown  # type: ignore[import-untyped]

    md = MarkItDown()
    result = md.convert(pdf_path)
    return result.text_content


def _run_pspdfkit(pdf_path: str) -> str:
    """PSPDFKit / Nutrient pdf-to-markdown CLI.

    Shells out to ``pdf-to-markdown`` (or ``npx @pspdfkit/pdf-to-markdown``).
    """
    with tempfile.NamedTemporaryFile(suffix=".md", delete=False) as tmp:
        out_path = tmp.name

    try:
        # Prefer the globally-installed binary, fall back to npx
        bin_path = shutil.which("pdf-to-markdown")
        if bin_path:
            cmd = [bin_path, pdf_path, out_path]
        else:
            cmd = ["npx", "--yes", "@pspdfkit/pdf-to-markdown", pdf_path, out_path]

        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                proc.stderr.strip()[:300] or f"exit code {proc.returncode}"
            )
        return Path(out_path).read_text()
    finally:
        Path(out_path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------


def _try_import(module: str) -> bool:
    try:
        __import__(module)
        return True
    except ImportError:
        return False


def _check_pspdfkit() -> bool:
    """Check if pdf-to-markdown CLI is available."""
    if shutil.which("pdf-to-markdown"):
        return True
    # Check if npx can resolve it (fast check — just --help)
    try:
        proc = subprocess.run(
            ["npx", "--yes", "@pspdfkit/pdf-to-markdown", "--help"],
            capture_output=True, timeout=30,
        )
        return proc.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


_TOOLS: dict[str, dict[str, Any]] = {
    "pymupdf": {
        "fn": _run_pymupdf,
        "check": lambda: _try_import("fitz"),
        "install": "pip install pymupdf",
    },
    "markitdown": {
        "fn": _run_markitdown,
        "check": lambda: _try_import("markitdown"),
        "install": "pip install markitdown",
    },
    "pspdfkit": {
        "fn": _run_pspdfkit,
        "check": _check_pspdfkit,
        "install": "npm i -g @pspdfkit/pdf-to-markdown",
    },
}


def available_tools() -> dict[str, bool]:
    """Return {tool_name: is_available} for all registered tools."""
    return {name: info["check"]() for name, info in _TOOLS.items()}


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------


def run_pdf_benchmark(
    pdf_paths: list[str],
    tools: list[str] | None = None,
    output_dir: str | None = None,
) -> dict[str, Any]:
    """Run the PDF-to-markdown benchmark.

    Parameters
    ----------
    pdf_paths:
        List of PDF file paths to benchmark.
    tools:
        Tool names to include.  ``None`` means all available.
    output_dir:
        Directory to save extracted markdown files.  ``None`` skips saving.
        Files are written as ``<output_dir>/<tool>/<pdf_stem>.md``.

    Returns
    -------
    dict with ``"tool_summaries"``, ``"per_pdf"``, and ``"output_dir"`` keys.
    """
    avail = available_tools()
    tool_names = tools or [t for t, ok in avail.items() if ok]

    all_results: list[ToolResult] = []

    for tool_name in tool_names:
        if not avail.get(tool_name, False):
            logger.warning("Tool %s not available, skipping", tool_name)
            continue

        fn = _TOOLS[tool_name]["fn"]

        for pdf_path in pdf_paths:
            tr = ToolResult(tool=tool_name, pdf=Path(pdf_path).name)
            t0 = time.monotonic()
            try:
                md = fn(pdf_path)
                tr.success = True
                tr.markdown = md
                tr.char_count = len(md)
                feats = _count_md_features(md)
                tr.heading_count = feats["headings"]
                tr.list_item_count = feats["list_items"]
                tr.table_count = feats["tables"]
            except Exception as e:
                tr.error = str(e)[:200]
                logger.warning(
                    "Tool %s failed on %s: %s", tool_name, pdf_path, tr.error
                )
            tr.elapsed_sec = round(time.monotonic() - t0, 3)

            # Save extracted markdown to disk
            if tr.success and output_dir:
                tool_dir = Path(output_dir) / tool_name
                tool_dir.mkdir(parents=True, exist_ok=True)
                stem = Path(pdf_path).stem
                out_path = tool_dir / f"{stem}.md"
                out_path.write_text(tr.markdown, encoding="utf-8")

            all_results.append(tr)

    # Build summaries
    summaries: dict[str, ToolSummary] = {}
    for tool_name in tool_names:
        tool_results = [r for r in all_results if r.tool == tool_name]
        succeeded = [r for r in tool_results if r.success]
        n = len(succeeded) or 1
        summaries[tool_name] = ToolSummary(
            tool=tool_name,
            available=avail.get(tool_name, False),
            pdfs_attempted=len(tool_results),
            pdfs_succeeded=len(succeeded),
            avg_chars=round(sum(r.char_count for r in succeeded) / n, 1),
            avg_headings=round(
                sum(r.heading_count for r in succeeded) / n, 1
            ),
            avg_list_items=round(
                sum(r.list_item_count for r in succeeded) / n, 1
            ),
            avg_tables=round(sum(r.table_count for r in succeeded) / n, 1),
            avg_elapsed_sec=round(
                sum(r.elapsed_sec for r in succeeded) / n, 3
            ),
            total_elapsed_sec=round(
                sum(r.elapsed_sec for r in tool_results), 3
            ),
            errors=[r.error for r in tool_results if r.error],
        )

    return {
        "tool_summaries": {
            k: _summary_to_dict(v) for k, v in summaries.items()
        },
        "per_pdf": [_result_to_dict(r) for r in all_results],
        "available_tools": avail,
        "output_dir": output_dir,
    }


def _summary_to_dict(s: ToolSummary) -> dict[str, Any]:
    return {
        "tool": s.tool,
        "available": s.available,
        "pdfs_attempted": s.pdfs_attempted,
        "pdfs_succeeded": s.pdfs_succeeded,
        "avg_chars": s.avg_chars,
        "avg_headings": s.avg_headings,
        "avg_list_items": s.avg_list_items,
        "avg_tables": s.avg_tables,
        "avg_elapsed_sec": s.avg_elapsed_sec,
        "total_elapsed_sec": s.total_elapsed_sec,
        "errors": s.errors,
    }


def _result_to_dict(r: ToolResult) -> dict[str, Any]:
    return {
        "tool": r.tool,
        "pdf": r.pdf,
        "success": r.success,
        "char_count": r.char_count,
        "heading_count": r.heading_count,
        "list_item_count": r.list_item_count,
        "table_count": r.table_count,
        "elapsed_sec": r.elapsed_sec,
        "error": r.error,
    }
