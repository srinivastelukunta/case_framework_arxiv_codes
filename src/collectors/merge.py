"""Merged-pool deduplication for Study 1 candidates.

Collection-stage dedup is exact-URL only (the same source document reached
via two collectors). Incident-level dedup by (system, date) happens at
coding time via src/coding/protocol.py, where system and date have been
extracted from the text.
"""

from __future__ import annotations


def dedup_by_url(
    rows: list[dict], skip_sources: frozenset[str] = frozenset()
) -> tuple[list[dict], dict[str, str]]:
    """Keep the first row per URL (input order = source priority).

    skip_sources lists sources whose URL is a collection landing page
    rather than a per-document link (e.g. mit_risk_repo rows all cite the
    repository itself); their rows are never URL-deduped.

    Returns (kept rows, {dropped_candidate_id: kept_candidate_id}).
    """
    kept: list[dict] = []
    by_url: dict[str, str] = {}
    dropped: dict[str, str] = {}
    for row in rows:
        url = row.get("url", "")
        if row.get("source") in skip_sources or not url:
            kept.append(row)
            continue
        if url in by_url:
            dropped[row["candidate_id"]] = by_url[url]
            continue
        by_url[url] = row["candidate_id"]
        kept.append(row)
    return kept, dropped
