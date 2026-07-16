"""Study 1 dual-coding orchestrator.

Usage: python -m src.coding.run_coding [--dry-run]

Reads screen-included candidates from data/study1_candidates.csv, runs
coding passes A and B (cache-first; API calls only for uncached
candidates), computes Cohen's kappa on primary codes before adjudication,
exports disagreements for human review, and — once all disagreements are
adjudicated — writes data/study1_incidents.csv.

--dry-run reports what would be coded without calling the API.
"""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

import pandas as pd

from src.coding.adjudicate import (
    build_final_dataset,
    dedup_incident_twins,
    export_disagreements,
    load_adjudications,
    merge_passes,
)
from src.coding.kappa import cohens_kappa
from src.coding.llm_precoder import run_pass

ROOT = Path(__file__).resolve().parents[2]


def _load_dotenv(path: Path) -> None:
    """Minimal .env loader: KEY=value lines into os.environ (no override).

    Keeps the API key out of shell history and the repo (.env is
    gitignored); no python-dotenv dependency needed.
    """
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
CANDIDATES_CSV = ROOT / "data" / "study1_candidates.csv"
CODING_CACHE = ROOT / "data" / "raw" / "coding"
DISAGREEMENTS_CSV = ROOT / "data" / "study1_disagreements.csv"
ADJUDICATIONS_CSV = ROOT / "data" / "study1_adjudications.csv"
INCIDENTS_CSV = ROOT / "data" / "study1_incidents.csv"
KAPPA_JSON = ROOT / "data" / "study1_kappa.json"
REFUSED_CSV = ROOT / "data" / "study1_refused.csv"

# Dedup priority: richer incident sources first (first-in wins at dedup).
SOURCE_ORDER = {"aiid": 0, "web_reports": 1, "arxiv": 2, "mit_risk_repo": 3}

INCIDENT_COLUMNS = (
    "incident_id",
    "source",
    "url",
    "date",
    "system_description",
    "included",
    "exclusion_reason",
    "primary_layer",
    "secondary_layers",
    "mechanism_phrase",
    "rationale",
    "coder_a",
    "coder_b",
    "adjudicated",
)


def _overlap_map() -> dict[str, str]:
    """web candidate_id -> aiid candidate_id, from config aiid_overlap.

    The author-curated dedup key for curated/AIID incident twins.
    """
    import yaml

    with open(ROOT / "config" / "sources.yaml", encoding="utf-8") as f:
        items = yaml.safe_load(f)["sources"]["web_reports"].get("items", [])
    return {
        f"web-{it['id']}": f"aiid-{it['aiid_overlap']}"
        for it in items
        if it.get("aiid_overlap")
    }


def load_candidates() -> list[dict]:
    df = pd.read_csv(CANDIDATES_CSV).fillna("")
    pool = df[df.screen_included].to_dict("records")
    pool.sort(
        key=lambda r: (SOURCE_ORDER.get(r["source"], 9), str(r["candidate_id"]))
    )
    return pool


def _uncached(candidates: list[dict], pass_name: str) -> list[str]:
    pass_dir = CODING_CACHE / f"pass{pass_name}"
    return [
        c["candidate_id"]
        for c in candidates
        if not (pass_dir / f"{c['candidate_id']}.json").exists()
    ]


def main(dry_run: bool = False) -> int:
    _load_dotenv(ROOT / ".env")
    from src.coding.source_text import enrich_candidates

    candidates = enrich_candidates(load_candidates(), ROOT / "data" / "raw")
    lengths = sorted(len(c["text_snippet"]) for c in candidates)
    print(f"screen-included candidates to code: {len(candidates)}")
    print(f"coding text chars: median={lengths[len(lengths) // 2]}, "
          f"min={lengths[0]}, max={lengths[-1]}")

    todo = {p: _uncached(candidates, p) for p in ("A", "B")}
    n_calls = sum(len(v) for v in todo.values())
    print(f"uncached (needs API): pass A={len(todo['A'])}, pass B={len(todo['B'])}")

    if dry_run:
        print("dry run: no API calls made")
        return 0

    if n_calls and not (
        os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")
    ):
        print(
            "ERROR: uncached candidates remain but no ANTHROPIC_API_KEY / "
            "ANTHROPIC_AUTH_TOKEN is set. Set a key (see README) or restore "
            "the coding cache."
        )
        return 1

    rows_a, refused_a = run_pass("A", candidates, CODING_CACHE, progress=print)
    rows_b, refused_b = run_pass("B", candidates, CODING_CACHE, progress=print)

    # A candidate the model refused in EITHER pass has no valid dual coding,
    # so drop it from the coded set (both passes must align for kappa/merge)
    # and report it for human coding. Never fabricate a code for a refusal.
    refused = sorted(set(refused_a) | set(refused_b))
    if refused:
        REFUSED_CSV.write_text(
            "candidate_id\n" + "\n".join(refused) + "\n", encoding="utf-8"
        )
        keep = {r["candidate_id"] for r in rows_a} & {r["candidate_id"] for r in rows_b}
        rows_a = [r for r in rows_a if r["candidate_id"] in keep]
        rows_b = [r for r in rows_b if r["candidate_id"] in keep]
        print(
            f"\nMODEL REFUSED {len(refused)} candidate(s) (cyber safety "
            f"classifier) -> {REFUSED_CSV.name}; excluded from auto-coding, "
            "route to human coding:"
        )
        for cid in refused:
            print(f"    {cid}")

    kappa = cohens_kappa([r["label"] for r in rows_a], [r["label"] for r in rows_b])
    merged = merge_passes(rows_a, rows_b)
    n_disagreements = export_disagreements(merged, DISAGREEMENTS_CSV)
    KAPPA_JSON.write_text(
        json.dumps(
            {
                "cohens_kappa_primary_pre_adjudication": kappa,
                "n_units": len(merged),
                "n_disagreements": n_disagreements,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Cohen's kappa (primary codes, pre-adjudication): {kappa:.3f}")
    print(f"disagreements: {n_disagreements} -> {DISAGREEMENTS_CSV.name}")

    adjudications = load_adjudications(ADJUDICATIONS_CSV)
    try:
        final = build_final_dataset(merged, adjudications)
    except ValueError as exc:
        print(f"\nSTOPPED before final dataset: {exc}")
        print("Review notebooks/adjudication_review.ipynb, fill "
              f"{ADJUDICATIONS_CSV.name}, then re-run.")
        return 2

    final, twin_dups = dedup_incident_twins(final, _overlap_map())
    for row in twin_dups:
        print(f"  incident dedup: {row['incident_id']} -> "
              f"{row['exclusion_reason']}")

    with open(INCIDENTS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=INCIDENT_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for row in sorted(final, key=lambda r: r["incident_id"]):
            writer.writerow(row)
    n_inc = sum(1 for r in final if r["included"])
    print(f"\nwrote {INCIDENTS_CSV.relative_to(ROOT)}: {len(final)} rows, "
          f"{n_inc} included")
    return 0


if __name__ == "__main__":
    sys.exit(main(dry_run="--dry-run" in sys.argv[1:]))
