"""Build results/study1_table.tex and the study1 section of
results/stats_summary.json from data/study1_incidents.csv.

Usage: python -m src.emit.build_study1_outputs
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

from src.emit.latex_tables import emit_study1_table
from src.emit.stats_summary import study1_stats

ROOT = Path(__file__).resolve().parents[2]
INCIDENTS_CSV = ROOT / "data" / "study1_incidents.csv"
KAPPA_JSON = ROOT / "data" / "study1_kappa.json"
TABLE_TEX = ROOT / "results" / "study1_table.tex"
STATS_JSON = ROOT / "results" / "stats_summary.json"


def main() -> int:
    if not INCIDENTS_CSV.exists():
        print(f"missing {INCIDENTS_CSV.relative_to(ROOT)} — run "
              "python -m src.coding.run_coding first")
        return 1
    incidents = pd.read_csv(INCIDENTS_CSV).fillna("")
    kappa_data = json.loads(KAPPA_JSON.read_text(encoding="utf-8"))
    kappa = kappa_data["cohens_kappa_primary_pre_adjudication"]

    tex = emit_study1_table(incidents)
    TABLE_TEX.parent.mkdir(parents=True, exist_ok=True)
    TABLE_TEX.write_text(tex, encoding="utf-8")

    stats = {}
    if STATS_JSON.exists():
        stats = json.loads(STATS_JSON.read_text(encoding="utf-8"))
    stats["study1"] = study1_stats(incidents, kappa=kappa)
    stats["study1"]["kappa_detail"] = kappa_data
    STATS_JSON.write_text(
        json.dumps(stats, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    s = stats["study1"]
    print(f"wrote {TABLE_TEX.relative_to(ROOT)} and {STATS_JSON.relative_to(ROOT)}")
    print(f"  N included : {s['n_included']}")
    print(f"  per layer  : {s['per_layer_counts']}")
    print(f"  L2+L3 share: {s['l2_l3_share']:.1%}")
    print(f"  kappa      : {s['cohens_kappa']:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
