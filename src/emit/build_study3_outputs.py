"""Build results/figure1.pdf and the study3 section of stats_summary.json
from the scored deployment corpus.

Corpus source: data/raw/study3/part_*.json (research-scored, author-adjudicated)
is merged into data/study3_deployments.csv (the auditable coded dataset).
The composite index, figure, and stats are then computed deterministically.

Usage: python -m src.emit.build_study3_outputs
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

from src.emit.figure1 import render_figure1
from src.emit.stats_summary import study3_stats
from src.study3.index import DEFAULT_ALPHA, LAYERS
from src.ledger import upsert_ledger_entries

ROOT = Path(__file__).resolve().parents[2]
CONFIG_YAML = ROOT / "config" / "study3.yaml"
PARTS_DIR = ROOT / "data" / "raw" / "study3"
DEPLOYMENTS_CSV = ROOT / "data" / "study3_deployments.csv"
FIGURE_PDF = ROOT / "results" / "figure1.pdf"
STATS_JSON = ROOT / "results" / "stats_summary.json"
LEDGER_CSV = ROOT / "data" / "citation_ledger.csv"
MIN_N = 30

CSV_COLUMNS = (
    ["deployment", "organization", "sector"]
    + [f"m_{l}" for l in LAYERS]
    + [f"evidence_{l}" for l in LAYERS]
    + [f"url_{l}" for l in LAYERS]
)


def load_corpus() -> list[dict]:
    """Prefer the committed CSV; otherwise merge the research JSON parts."""
    if DEPLOYMENTS_CSV.exists():
        return _load_csv()
    return _merge_parts()


def _org_key(org: str) -> str:
    """Normalized company key for dedup: drop parentheticals, lowercase.

    'Klarna (powered by OpenAI)' and 'Klarna (built with OpenAI)' -> 'klarna'
    so the same deployment surfacing in two sector batches counts once. Each
    organization contributes one deployment-level maturity profile.
    """
    import re
    return re.sub(r"\(.*?\)", "", str(org)).strip().lower()


def _merge_parts() -> list[dict]:
    rows = []
    seen = set()
    for part in sorted(PARTS_DIR.glob("part_*.json")):
        for d in json.loads(part.read_text(encoding="utf-8")):
            key = _org_key(d.get("organization", ""))
            if key in seen:
                continue
            seen.add(key)
            rows.append(_normalize(d))
    return rows


def _normalize(d: dict) -> dict:
    scores = {l: float(d["scores"][l]) for l in LAYERS}
    ev = d.get("evidence", {})
    row = {
        "deployment": d.get("deployment", ""),
        "organization": d.get("organization", ""),
        "sector": d.get("sector", ""),
    }
    for l in LAYERS:
        row[f"m_{l}"] = scores[l]
        row[f"evidence_{l}"] = (ev.get(l, {}) or {}).get("text", "")
        row[f"url_{l}"] = (ev.get(l, {}) or {}).get("url", "")
    return row


def _load_csv() -> list[dict]:
    with open(DEPLOYMENTS_CSV, newline="", encoding="utf-8") as f:
        out = []
        for r in csv.DictReader(f):
            for l in LAYERS:
                r[f"m_{l}"] = float(r[f"m_{l}"])
            out.append(r)
        return out


def _write_csv(rows: list[dict]) -> None:
    ordered = sorted(rows, key=lambda r: (r["sector"], r["organization"],
                                          r["deployment"]))
    with open(DEPLOYMENTS_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS, lineterminator="\n")
        w.writeheader()
        for r in ordered:
            w.writerow({c: r.get(c, "") for c in CSV_COLUMNS})


def _deployments_for_stats(rows: list[dict]) -> list[dict]:
    return [{"deployment": r["deployment"],
             "scores": {l: r[f"m_{l}"] for l in LAYERS}} for r in rows]


def _load_alpha() -> float:
    """Bottleneck weight alpha from config/study3.yaml (default 0.6)."""
    if CONFIG_YAML.exists():
        import yaml
        cfg = yaml.safe_load(CONFIG_YAML.read_text(encoding="utf-8")) or {}
        return float(cfg.get("index", {}).get("alpha", DEFAULT_ALPHA))
    return DEFAULT_ALPHA


def main() -> int:
    rows = load_corpus()
    if len(rows) < MIN_N:
        print(f"study3: {len(rows)} deployments < minimum {MIN_N}; refusing "
              "to emit (no-fabrication guardrail). Merge more into "
              f"{PARTS_DIR.relative_to(ROOT)} or {DEPLOYMENTS_CSV.name}.")
        return 1

    if not DEPLOYMENTS_CSV.exists():
        _write_csv(rows)
        print(f"wrote {DEPLOYMENTS_CSV.relative_to(ROOT)} ({len(rows)} rows)")

    deployments = _deployments_for_stats(rows)
    alpha = _load_alpha()
    render_figure1(deployments, FIGURE_PDF, alpha=alpha)
    _write_ledger(rows)

    stats = json.loads(STATS_JSON.read_text(encoding="utf-8")) if STATS_JSON.exists() else {}
    stats["study3"] = study3_stats(deployments, alpha=alpha)
    STATS_JSON.write_text(
        json.dumps(stats, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    s = stats["study3"]
    idx = s["index"]
    levels = idx["certified_level_distribution"]
    print(f"wrote {FIGURE_PDF.relative_to(ROOT)} and study3 stats")
    print(f"  N deployments        : {s['n_deployments']}")
    print(f"  per-layer mean       : "
          + ", ".join(f"{l} {s['per_layer_mean'][l]:.2f}" for l in LAYERS))
    print(f"  bottleneck (a={idx['alpha']}) : "
          f"mean {idx['bottleneck_mean']:.3f}, "
          f"median {idx['bottleneck_median']:.3f}, "
          f"min {idx['bottleneck_min']:.3f}, max {idx['bottleneck_max']:.3f}")
    print(f"  certified levels     : "
          + ", ".join(f"{name.split()[0]} {count}"
                      for name, count in levels.items()))
    print(f"  geometric limit      : all composites zero = "
          f"{idx['geometric_robustness_limit']['all_composites_zero']}")
    print(f"  modal weakest layer  : {s['modal_weakest_layer']}")
    return 0


def _write_ledger(rows: list[dict]) -> None:
    entries = []
    for r in rows:
        for l in LAYERS:
            url = r.get(f"url_{l}", "")
            if r[f"m_{l}"] > 0 and url:
                entries.append({
                    "item_id": f"study3-{_slug(r['organization'])}-{l}",
                    "study": "study3",
                    "source": "deployment_evidence",
                    "title": f"{r['deployment']} ({r['organization']}) {l} "
                             f"m={r[f'm_{l}']}: {r.get(f'evidence_{l}','')[:100]}",
                    "url": url,
                    "accessed_date": "",
                    "license_note": "public case study / disclosure",
                })
    if entries:
        upsert_ledger_entries(entries, LEDGER_CSV)


def _slug(text: str) -> str:
    import re
    return re.sub(r"[^a-z0-9]+", "-", str(text).lower()).strip("-") or "org"


if __name__ == "__main__":
    sys.exit(main())
