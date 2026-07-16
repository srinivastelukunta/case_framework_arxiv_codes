"""Dual-coder merge, disagreement export, and final dataset build.

Pass A and pass B rows are merged by candidate_id. Agreements become final
directly (secondaries = union of both passes). Disagreements require a
human adjudication row (the author's review of the adjudication notebook
is the human adjudication the paper claims); building the final dataset
with unresolved disagreements fails loudly.
"""

from __future__ import annotations

import csv
from pathlib import Path

LAYER_ORDER = ("L1", "L2", "L3", "L4")


def _union_secondaries(a: str, b: str, primary: str) -> str:
    layers = {s.strip() for s in f"{a};{b}".split(";") if s.strip()}
    layers.discard(primary)
    return "; ".join(l for l in LAYER_ORDER if l in layers)


def merge_passes(rows_a: list[dict], rows_b: list[dict]) -> list[dict]:
    by_id_b = {r["candidate_id"]: r for r in rows_b}
    missing = [r["candidate_id"] for r in rows_a if r["candidate_id"] not in by_id_b]
    if missing or len(rows_a) != len(rows_b):
        raise ValueError(f"pass A/B candidate sets differ (missing in B: {missing})")

    merged = []
    for a in rows_a:
        b = by_id_b[a["candidate_id"]]
        agreed = a["label"] == b["label"]
        merged.append(
            {
                "candidate_id": a["candidate_id"],
                "source": a["source"],
                "url": a["url"],
                "date": a["date"] or b["date"],
                "system": a["system"] or b["system"],
                "coder_a": a["label"],
                "coder_b": b["label"],
                "agreed": agreed,
                "final_label": a["label"] if agreed else None,
                "exclusion_reason": a["exclusion_reason"] if agreed else "",
                "secondary_layers": _union_secondaries(
                    a["secondary_layers"], b["secondary_layers"], a["label"]
                )
                if agreed
                else "",
                "mechanism_phrase": a["mechanism_phrase"] or b["mechanism_phrase"],
                "rationale_a": a["rationale"],
                "rationale_b": b["rationale"],
            }
        )
    return merged


def export_disagreements(merged: list[dict], path: Path) -> int:
    disagreements = [m for m in merged if not m["agreed"]]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = (
        "candidate_id",
        "source",
        "url",
        "system",
        "date",
        "coder_a",
        "coder_b",
        "rationale_a",
        "rationale_b",
    )
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for m in disagreements:
            writer.writerow({c: m[c] for c in columns})
    return len(disagreements)


def load_adjudications(path: Path) -> dict[str, dict]:
    """data/study1_adjudications.csv: candidate_id, final_label, rationale.

    final_label is L1..L4 or 'excluded'; rationale is the author's written
    reason (mandatory per project rules).
    """
    path = Path(path)
    if not path.exists():
        return {}
    out = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if not row.get("final_label"):
                continue
            if not row.get("rationale", "").strip():
                raise ValueError(
                    f"adjudication for {row['candidate_id']} missing rationale"
                )
            out[row["candidate_id"]] = {
                "final_label": row["final_label"].strip(),
                "rationale": row["rationale"].strip(),
            }
    return out


def build_final_dataset(
    merged: list[dict], adjudications: dict[str, dict]
) -> list[dict]:
    """Final per-candidate rows for data/study1_incidents.csv."""
    unresolved = [
        m["candidate_id"]
        for m in merged
        if not m["agreed"] and m["candidate_id"] not in adjudications
    ]
    if unresolved:
        raise ValueError(
            f"{len(unresolved)} unresolved disagreement(s): {unresolved[:10]} — "
            "adjudicate them in notebooks/adjudication_review.ipynb first"
        )

    final = _build_rows(merged, adjudications)
    return final


def dedup_incident_twins(
    final: list[dict], overlap_map: dict[str, str]
) -> tuple[list[dict], list[dict]]:
    """Collapse curated/AIID twins of the same incident.

    A curated web report may cover the same incident as an AIID entry
    (declared via `aiid_overlap` in config/sources.yaml). Coding runs one
    unit per candidate, so when BOTH the web twin and its AIID counterpart
    are included, the incident is counted twice. Keep the AIID entry as the
    canonical incident (stable cite ID) and mark the web twin excluded.

    overlap_map: web candidate_id -> aiid candidate_id.
    Returns (final, dropped_rows). Deterministic; touches only rows where
    both twins are currently included.
    """
    by_id = {r["incident_id"]: r for r in final}
    dropped = []
    for web_id, aiid_id in sorted(overlap_map.items()):
        web, aiid = by_id.get(web_id), by_id.get(aiid_id)
        if web and aiid and web["included"] and aiid["included"]:
            web["included"] = False
            web["exclusion_reason"] = f"duplicate_of_{aiid_id}"
            web["primary_layer"] = ""
            web["secondary_layers"] = ""
            web["mechanism_phrase"] = ""
            web["rationale"] = (
                f"Incident-level dedup: same incident as {aiid_id} "
                f"(aiid_overlap); AIID entry retained as canonical."
            )
            web["adjudicated"] = True
            dropped.append(web)
    return final, dropped


def _build_rows(merged: list[dict], adjudications: dict[str, dict]) -> list[dict]:
    final = []
    for m in merged:
        if m["agreed"]:
            label, rationale, adjudicated = (
                m["final_label"],
                m["rationale_a"],
                False,
            )
        else:
            adj = adjudications[m["candidate_id"]]
            label, rationale, adjudicated = (
                adj["final_label"],
                adj["rationale"],
                True,
            )
        included = label != "excluded"
        final.append(
            {
                "incident_id": m["candidate_id"],
                "source": m["source"],
                "url": m["url"],
                "date": m["date"],
                "system_description": m["system"],
                "included": included,
                "exclusion_reason": "" if included else (m["exclusion_reason"] or "adjudicated_excluded"),
                "primary_layer": label if included else "",
                "secondary_layers": m["secondary_layers"] if included else "",
                "mechanism_phrase": m["mechanism_phrase"] if included else "",
                "rationale": rationale,
                "coder_a": m["coder_a"],
                "coder_b": m["coder_b"],
                "adjudicated": adjudicated,
            }
        )
    return final
