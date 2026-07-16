"""Build the Study 1 candidate pool from all four collectors:
AIID, MIT Risk Repository, arXiv, and curated web reports.

Usage: python -m src.collectors.build_candidates [--refresh]

Writes data/study1_candidates.csv (every screened row with its decision
and reason), data/study1_candidates_summary.json (pool statistics for the
paper's methods text), and source-dataset rows in the citation ledger.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

from src.collectors import aiid, arxiv_papers, mit_risk_repo, web_reports
from src.collectors.fetch import Fetcher, fetcher_from_config
from src.collectors.merge import dedup_by_url
from src.collectors.screen import write_candidates_csv
from src.ledger import upsert_ledger_entries

ROOT = Path(__file__).resolve().parents[2]
CANDIDATES_CSV = ROOT / "data" / "study1_candidates.csv"
SUMMARY_JSON = ROOT / "data" / "study1_candidates_summary.json"
LEDGER_CSV = ROOT / "data" / "citation_ledger.csv"
ARXIV_MIN_INTERVAL = 3.0  # arXiv API terms ask for a 3s courtesy delay


def _accessed_date(cache_root: Path, subdir: str, filename: str) -> str:
    manifest = cache_root / subdir / "manifest.json"
    if manifest.exists():
        entries = json.loads(manifest.read_text(encoding="utf-8"))
        if filename in entries:
            return entries[filename]["accessed_utc"]
    return ""


def _source_ledger_entries(sources: dict, cache_root: Path) -> list[dict]:
    aiid_cfg, mit_cfg = sources["aiid"], sources["mit_risk_repo"]
    snapshot_name = aiid_cfg["snapshot_url"].rsplit("/", 1)[-1]
    return [
        # url = human-readable landing page; the exact payload URL and
        # sha256 of what we parsed are pinned in data/raw/*/manifest.json
        {
            "item_id": "dataset-aiid-snapshot",
            "study": "study1",
            "source": "aiid",
            "title": f"AI Incident Database weekly snapshot "
            f"({aiid_cfg['snapshot_date']}, {snapshot_name}; direct download "
            f"URL in data/raw/aiid/manifest.json)",
            "url": "https://incidentdatabase.ai/research/snapshots/",
            "accessed_date": _accessed_date(
                cache_root, aiid_cfg["cache_subdir"], snapshot_name
            ),
            "license_note": "Public research snapshot; see "
            "incidentdatabase.ai/terms-of-use",
        },
        {
            "item_id": "dataset-mit-risk-repository",
            "study": "study1",
            "source": "mit_risk_repo",
            "title": "MIT AI Risk Repository (public spreadsheet release; "
            "xlsx export URL in data/raw/mit_risk_repo/manifest.json)",
            "url": "https://airisk.mit.edu/risks",
            "accessed_date": _accessed_date(
                cache_root, mit_cfg["cache_subdir"], "ai_risk_repository.xlsx"
            ),
            "license_note": mit_cfg["license_note"],
        },
    ]


def main(refresh: bool = False) -> int:
    with open(ROOT / "config" / "sources.yaml", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    defaults = config["defaults"]
    sources = config["sources"]
    cache_root = ROOT / defaults["cache_dir"]
    fetcher = fetcher_from_config(defaults)
    arxiv_fetcher = Fetcher(
        min_interval=max(ARXIV_MIN_INTERVAL, float(defaults["min_seconds_between_requests"])),
        user_agent=str(defaults["user_agent"]).strip(),
        timeout=int(defaults.get("timeout_seconds", 30)),
    )
    # Per-source screen terms (pool expansion 2026-07-16): AIID and the
    # curated web reports use the widened list; MIT and arXiv keep the
    # narrow base list (zero includes in the v2 coding run — widening them
    # would only add coding cost).
    wide_terms = tuple(sources["aiid"]["filter_terms"])
    mit_terms = tuple(sources["mit_risk_repo"]["filter_terms"])
    arxiv_terms = tuple(sources["arxiv"]["filter_terms"])

    if refresh:
        print("refresh: re-downloading source payloads")
        aiid_cfg = sources["aiid"]
        fetcher.fetch(
            aiid_cfg["snapshot_url"],
            cache_root / aiid_cfg["cache_subdir"]
            / aiid_cfg["snapshot_url"].rsplit("/", 1)[-1],
            refresh=True,
        )
        mit_cfg = sources["mit_risk_repo"]
        fetcher.fetch(
            mit_cfg["export_url"],
            cache_root / mit_cfg["cache_subdir"] / "ai_risk_repository.xlsx",
            refresh=True,
            robots_exempt=bool(mit_cfg.get("explicit_dataset_link", False)),
        )

    # Order matters: dedup_by_url keeps the first occurrence, so richer
    # incident sources take priority over paper/report carriers.
    print("collecting: AIID")
    rows = aiid.collect(fetcher, sources["aiid"], cache_root)
    counts = {"aiid": len(rows)}
    print(f"  {counts['aiid']} rows screened")

    print("collecting: curated web reports")
    web_rows, web_failures = web_reports.collect(
        fetcher, sources["web_reports"], cache_root, wide_terms, refresh=refresh
    )
    rows.extend(web_rows)
    counts["web_reports"] = len(web_rows)
    print(f"  {len(web_rows)} rows screened")
    for failure in web_failures:
        print(f"  WARN fetch failed, screened on metadata: {failure}")

    print("collecting: arXiv")
    arxiv_rows = arxiv_papers.collect(
        arxiv_fetcher, sources["arxiv"], cache_root, arxiv_terms, refresh=refresh
    )
    rows.extend(arxiv_rows)
    counts["arxiv"] = len(arxiv_rows)
    print(f"  {len(arxiv_rows)} rows screened")

    print("collecting: MIT AI Risk Repository")
    mit_rows = mit_risk_repo.collect(
        fetcher, sources["mit_risk_repo"], cache_root, mit_terms
    )
    rows.extend(mit_rows)
    counts["mit_risk_repo"] = len(mit_rows)
    print(f"  {len(mit_rows)} rows screened")

    # MIT rows all cite the repository landing page, not per-row documents
    rows, url_dups = dedup_by_url(rows, skip_sources=frozenset({"mit_risk_repo"}))
    if url_dups:
        print(f"  url-dedup dropped {len(url_dups)} duplicate-URL rows")

    write_candidates_csv(rows, CANDIDATES_CSV)
    upsert_ledger_entries(_source_ledger_entries(sources, cache_root), LEDGER_CSV)

    included = [r for r in rows if r["screen_included"]]
    summary = {
        "screened_total": len(rows),
        "screen_included_total": len(included),
        "screened_by_source": counts,
        "included_by_source": {
            s: sum(1 for r in included if r["source"] == s) for s in counts
        },
        "excluded_by_reason": {
            reason: sum(
                1 for r in rows if r["screen_exclusion_reason"] == reason
            )
            for reason in ("pre_2023", "no_agent_terms")
        },
        "url_duplicates_dropped": url_dups,
        "web_fetch_failures": web_failures,
    }
    SUMMARY_JSON.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print(f"\nwrote {CANDIDATES_CSV.relative_to(ROOT)}")
    print(f"wrote {SUMMARY_JSON.relative_to(ROOT)}")
    print(f"  candidates screened : {len(rows)}")
    print(f"  screen-included     : {len(included)}")
    for source, n in summary["included_by_source"].items():
        print(f"    {source:15s} : {n}")
    return 0


if __name__ == "__main__":
    sys.exit(main(refresh="--refresh" in sys.argv[1:]))
