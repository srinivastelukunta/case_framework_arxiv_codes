"""Collection-stage screening shared by all Study 1 collectors.

This is the keyword/date screen that builds the candidate pool
(data/study1_candidates.csv). It is deliberately coarse: the full
Appendix B inclusion filter runs at coding time via
src/coding/protocol.py. Every screened candidate keeps its decision and
reason so the pool is auditable.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

AGENT_TERMS_DEFAULT = (
    "agent",
    "autonomous",
    "tool use",
    "multi-step",
    "orchestration",
    "copilot action",
)

CANDIDATE_COLUMNS = (
    "candidate_id",
    "source",
    "source_ref",
    "url",
    "date",
    "title",
    "text_snippet",
    "matched_terms",
    "screen_included",
    "screen_exclusion_reason",
)

SNIPPET_CHARS = 300


def _term_pattern(term: str) -> re.Pattern:
    # Word-boundary at the start so 'agent' matches 'agents'/'agentic' but
    # not 'reagent'; spaces in multi-word terms match any whitespace.
    escaped = re.escape(term).replace(r"\ ", r"\s+")
    return re.compile(rf"\b{escaped}", re.IGNORECASE)


def match_agent_terms(text: str, terms: tuple[str, ...]) -> list[str]:
    return [t for t in terms if _term_pattern(t).search(text or "")]


def screen_row(
    *,
    candidate_id: str,
    source: str,
    source_ref: str,
    url: str,
    date: str,
    title: str,
    text: str,
    terms: tuple[str, ...],
    min_date: str,
) -> dict:
    """Screen one candidate: date gate first, then agent-term gate."""
    matched = match_agent_terms(f"{title}\n{text}", terms)
    included, reason = True, ""
    if not date or str(date) < str(min_date):
        included, reason = False, "pre_2023"
    elif not matched:
        included, reason = False, "no_agent_terms"
    snippet = re.sub(r"\s+", " ", str(text or "")).strip()[:SNIPPET_CHARS]
    return {
        "candidate_id": candidate_id,
        "source": source,
        "source_ref": source_ref,
        "url": url,
        "date": date,
        "title": re.sub(r"\s+", " ", str(title or "")).strip(),
        "text_snippet": snippet,
        "matched_terms": "; ".join(matched),
        "screen_included": included,
        "screen_exclusion_reason": reason,
    }


def write_candidates_csv(rows: list[dict], path: Path) -> None:
    """Deterministic candidate CSV: fixed columns, sorted by candidate_id.

    Refuses to write an empty pool (no-fabrication guardrail).
    """
    if not rows:
        raise ValueError("candidate pool is empty; refusing to write CSV")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(rows, key=lambda r: r["candidate_id"])
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CANDIDATE_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for row in ordered:
            writer.writerow({col: row.get(col, "") for col in CANDIDATE_COLUMNS})
