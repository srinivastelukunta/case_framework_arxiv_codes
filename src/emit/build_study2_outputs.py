"""Build results/study2_table.tex and the study2 section of
results/stats_summary.json from config/tools.yaml.

Scrapes each approved tool's documentation (robots-aware, cached), codes
Full/Partial/None per CASE layer against config/mechanism_classes.yaml
(cache-first; API only for uncached tools), and emits the tool-by-layer
matrix plus coverage stats.

Usage: python -m src.emit.build_study2_outputs [--dry-run]
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import yaml

from bs4 import BeautifulSoup

from src.collectors.fetch import Fetcher, fetcher_from_config
from src.emit.latex_tables import emit_study2_table
from src.emit.stats_summary import study2_stats
from src.ledger import upsert_ledger_entries
from src.study2.code_tools import code_tool
from src.llm.structured import StructuredRefused

ROOT = Path(__file__).resolve().parents[2]
TOOLS_YAML = ROOT / "config" / "tools.yaml"
MECHANISMS_YAML = ROOT / "config" / "mechanism_classes.yaml"
SOURCES_YAML = ROOT / "config" / "sources.yaml"
TABLE_TEX = ROOT / "results" / "study2_table.tex"
STATS_JSON = ROOT / "results" / "stats_summary.json"
EVIDENCE_CSV = ROOT / "data" / "study2_tool_coding.csv"
LEDGER_CSV = ROOT / "data" / "citation_ledger.csv"
CODING_CACHE = ROOT / "data" / "raw" / "tool_coding"


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _extract_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "header", "footer", "aside"]):
        tag.decompose()
    body = soup.body or soup
    return " ".join(body.get_text(separator=" ").split())


def _doc_text(tool: dict, fetcher: Fetcher, cache_root: Path, subdir: str,
              refresh: bool) -> tuple[str, list[str]]:
    """Concatenated documentation text for a tool; returns (text, failures).

    Prefers a pinned rendered-markdown cache (data/raw/tool_docs/{id}.md).
    Many vendor docs are JS-rendered SPAs whose static HTML carries almost no
    text, so those are captured once as rendered markdown and committed as the
    reproducible source (the same pinning approach as the dataset snapshots).
    Falls back to robots-aware static HTML scraping when no .md exists.
    """
    md = cache_root / subdir / f"{tool['id']}.md"
    if md.exists():
        return md.read_text(encoding="utf-8"), []

    parts, failures = [], []
    for i, url in enumerate(tool["doc_urls"]):
        cache_file = cache_root / subdir / f"{tool['id']}_{i}.html"
        try:
            path = fetcher.fetch(url, cache_file, refresh=refresh)
            parts.append(f"[source: {url}]\n" + _extract_text(
                path.read_text(encoding="utf-8", errors="replace")
            ))
        except Exception as exc:  # noqa: BLE001 — one bad page must not kill the run
            failures.append(f"{tool['id']} {url}: {type(exc).__name__}: {exc}")
    return "\n\n".join(parts), failures


def main(dry_run: bool = False) -> int:
    _load_dotenv(ROOT / ".env")
    tools_cfg = yaml.safe_load(TOOLS_YAML.read_text(encoding="utf-8"))
    mechanisms = yaml.safe_load(MECHANISMS_YAML.read_text(encoding="utf-8"))
    defaults = yaml.safe_load(SOURCES_YAML.read_text(encoding="utf-8"))["defaults"]
    subdir = tools_cfg["defaults"]["cache_subdir"]

    tools = [t for t in tools_cfg["tools"] if t.get("status") == "approved"]
    print(f"approved tools to code: {len(tools)}")

    fetcher = fetcher_from_config(defaults)
    doc_texts, all_failures = {}, []
    for tool in tools:
        text, failures = _doc_text(tool, fetcher, ROOT / defaults["cache_dir"],
                                   subdir, refresh=False)
        doc_texts[tool["id"]] = text
        all_failures.extend(failures)
    for f in all_failures:
        print(f"  WARN doc fetch failed: {f}")

    uncached = [
        t for t in tools if not (CODING_CACHE / f"{t['id']}.json").exists()
    ]
    print(f"uncached (needs API): {len(uncached)}")
    if dry_run:
        print("dry run: no API calls made")
        return 0
    if uncached and not (
        os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")
    ):
        print("ERROR: uncached tools remain but no ANTHROPIC_API_KEY is set.")
        return 1

    coded, refused = [], []
    for tool in tools:
        text = doc_texts[tool["id"]]
        if not text and not (CODING_CACHE / f"{tool['id']}.json").exists():
            refused.append(f"{tool['id']}: no doc text fetched")
            continue
        try:
            cells = code_tool(tool, text, mechanisms, CODING_CACHE)
        except StructuredRefused:
            refused.append(f"{tool['id']}: model refused")
            continue
        coded.append({"id": tool["id"], "name": tool["name"],
                      "category": tool["category"], "cells": cells})

    for r in refused:
        print(f"  SKIPPED: {r}")

    _write_evidence_csv(coded)
    _write_ledger(coded)
    TABLE_TEX.parent.mkdir(parents=True, exist_ok=True)
    TABLE_TEX.write_text(emit_study2_table(coded), encoding="utf-8")

    stats = json.loads(STATS_JSON.read_text(encoding="utf-8")) if STATS_JSON.exists() else {}
    stats["study2"] = study2_stats(coded)
    STATS_JSON.write_text(
        json.dumps(stats, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    cov = stats["study2"]["full_coverage_share"]
    print(f"\nwrote {TABLE_TEX.relative_to(ROOT)} and {STATS_JSON.relative_to(ROOT)}")
    print(f"  tools coded : {len(coded)}")
    print("  Full coverage: " + ", ".join(
        f"{lyr} {round(100 * cov[lyr])}%" for lyr in ("L1", "L2", "L3", "L4")))
    return 0


def _write_evidence_csv(coded: list[dict]) -> None:
    import csv
    with open(EVIDENCE_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["tool_id", "tool", "category", "layer", "verdict",
                    "evidence", "evidence_url"])
        for t in coded:
            for layer in ("L1", "L2", "L3", "L4"):
                c = t["cells"][layer]
                w.writerow([t["id"], t["name"], t["category"], layer,
                            c["verdict"], c["evidence"], c["evidence_url"]])


def _write_ledger(coded: list[dict]) -> None:
    entries = []
    for t in coded:
        for layer in ("L1", "L2", "L3", "L4"):
            c = t["cells"][layer]
            if c["verdict"] != "None" and c["evidence_url"]:
                entries.append({
                    "item_id": f"study2-{t['id']}-{layer}",
                    "study": "study2",
                    "source": "tool_docs",
                    "title": f"{t['name']} — {layer} {c['verdict']}: {c['evidence'][:120]}",
                    "url": c["evidence_url"],
                    "accessed_date": "",
                    "license_note": "vendor/project documentation",
                })
    if entries:
        upsert_ledger_entries(entries, LEDGER_CSV)


if __name__ == "__main__":
    sys.exit(main(dry_run="--dry-run" in sys.argv[1:]))
