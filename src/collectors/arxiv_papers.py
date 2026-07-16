"""arXiv collector for Study 1.

Queries the arXiv Atom API for agent failure-mode papers (search terms in
config/sources.yaml) and screens entries into candidates. Per the guide,
the goal is concrete incidents documented WITHIN these papers; the coding
stage (T5) reads the papers behind screened-in candidates and harvests the
incidents. Raw Atom responses are cached per query in data/raw/arxiv/.

The arXiv API is a public programmatic interface (fetched robots_exempt);
its terms ask for a 3-second courtesy delay, which the caller enforces via
a dedicated Fetcher with min_interval >= 3.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import quote

from src.collectors.fetch import Fetcher
from src.collectors.screen import screen_row

ATOM_NS = "{http://www.w3.org/2005/Atom}"
MAX_RESULTS_PER_QUERY = 100


def query_url(base_url: str, term: str, max_results: int = MAX_RESULTS_PER_QUERY) -> str:
    search = quote(f'all:"{term}"')
    return (
        f"{base_url}?search_query={search}&start=0&max_results={max_results}"
        f"&sortBy=submittedDate&sortOrder=descending"
    )


def parse_atom_feed(xml_text: str) -> list[dict]:
    root = ET.fromstring(xml_text)
    entries = []
    for entry in root.findall(f"{ATOM_NS}entry"):
        raw_id = entry.findtext(f"{ATOM_NS}id", default="")
        m = re.search(r"abs/([0-9.]+?)(v\d+)?$", raw_id)
        if not m:
            continue
        arxiv_id = m.group(1)
        entries.append(
            {
                "arxiv_id": arxiv_id,
                "url": f"https://arxiv.org/abs/{arxiv_id}",
                "date": entry.findtext(f"{ATOM_NS}published", default="")[:10],
                "title": entry.findtext(f"{ATOM_NS}title", default=""),
                "summary": entry.findtext(f"{ATOM_NS}summary", default=""),
            }
        )
    return entries


def screen_arxiv_entries(
    entries: list[dict], min_date: str, terms: tuple[str, ...]
) -> list[dict]:
    rows, seen = [], set()
    for e in entries:
        if e["arxiv_id"] in seen:
            continue  # same paper returned by multiple search terms
        seen.add(e["arxiv_id"])
        rows.append(
            screen_row(
                candidate_id=f"arxiv-{e['arxiv_id']}",
                source="arxiv",
                source_ref=e["arxiv_id"],
                url=e["url"],
                date=e["date"],
                title=e["title"],
                text=e["summary"],
                terms=terms,
                min_date=min_date,
            )
        )
    return rows


def _slug(term: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", term.lower()).strip("_")


def collect(
    fetcher: Fetcher,
    cfg: dict,
    cache_root: Path,
    terms: tuple[str, ...],
    refresh: bool = False,
) -> list[dict]:
    entries: list[dict] = []
    for term in cfg["search_terms"]:
        cache_file = cache_root / cfg["cache_subdir"] / f"query_{_slug(term)}.xml"
        path = fetcher.fetch(
            query_url(cfg["base_url"], term),
            cache_file,
            refresh=refresh,
            robots_exempt=True,  # public programmatic API per arXiv ToU
        )
        entries.extend(parse_atom_feed(path.read_text(encoding="utf-8")))
    return screen_arxiv_entries(
        entries, min_date=cfg.get("min_date", "2023-01-01"), terms=terms
    )
