"""Citation ledger: every evidence item with URL and accessed date.

Project rule (guide sec 6): every evidence item enters
data/citation_ledger.csv. Upsert semantics keyed on item_id so pipeline
re-runs stay deterministic; rows are sorted for stable diffs.
"""

from __future__ import annotations

import csv
from pathlib import Path

LEDGER_COLUMNS = (
    "item_id",
    "study",
    "source",
    "title",
    "url",
    "accessed_date",
    "license_note",
)


def upsert_ledger_entries(entries: list[dict], path: Path) -> None:
    path = Path(path)
    existing: dict[str, dict] = {}
    if path.exists():
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                existing[row["item_id"]] = row

    for e in entries:
        if not e.get("item_id"):
            raise ValueError(f"ledger entry missing item_id: {e}")
        if not e.get("url"):
            raise ValueError(f"ledger entry {e['item_id']!r} missing url")
        existing[e["item_id"]] = {col: str(e.get(col, "")) for col in LEDGER_COLUMNS}

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=LEDGER_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for item_id in sorted(existing):
            writer.writerow(existing[item_id])
