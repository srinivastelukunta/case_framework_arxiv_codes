"""LaTeX emitter tests (T6: study1; T7 adds study2).

study1_table.tex must slot into the paper's tab:failures tabular
(p{2.6cm}p{7.2cm}cc): one row per layer L1..L4 in order, columns
Primary layer / Representative failure modes observed / Count / Share.
No-fabrication guardrail: emitters refuse empty or undersized inputs.
"""

import json

import pandas as pd
import pytest

from src.emit.latex_tables import emit_study1_table
from src.emit.stats_summary import study1_stats

MIN_N = 50


def make_incidents(n_l1=20, n_l2=15, n_l3=10, n_l4=8, n_excluded=5):
    rows = []
    layer_counts = {"L1": n_l1, "L2": n_l2, "L3": n_l3, "L4": n_l4}
    phrases = {
        "L1": ["runaway retry loop", "goal drift", "runaway retry loop"],
        "L2": ["shared-memory contamination", "cascade amplification"],
        "L3": ["approval fatigue", "oversight bypass"],
        "L4": ["no rollback path", "unbounded cost"],
    }
    i = 0
    for layer, count in layer_counts.items():
        for j in range(count):
            rows.append(
                {
                    "incident_id": f"inc-{i}",
                    "included": True,
                    "primary_layer": layer,
                    "secondary_layers": "L4" if layer == "L2" and j < 5 else "",
                    "mechanism_phrase": phrases[layer][j % len(phrases[layer])],
                }
            )
            i += 1
    for j in range(n_excluded):
        rows.append(
            {
                "incident_id": f"exc-{j}",
                "included": False,
                "primary_layer": "",
                "secondary_layers": "",
                "mechanism_phrase": "",
            }
        )
    return pd.DataFrame(rows)


class TestStudy1Table:
    def test_four_rows_in_layer_order(self):
        tex = emit_study1_table(make_incidents())
        lines = [l for l in tex.strip().splitlines() if l.strip()]
        assert len(lines) == 4
        for line, label in zip(
            lines,
            ("L1 Control", "L2 Adaptive systems", "L3 Supervisory",
             "L4 Engineering ops"),
        ):
            assert line.startswith(label), line
            assert line.rstrip().endswith(r"\\")

    def test_counts_and_shares(self):
        tex = emit_study1_table(make_incidents(20, 15, 10, 8))
        l1 = tex.strip().splitlines()[0]
        cells = [c.strip() for c in l1.split("&")]
        assert cells[2] == "20"
        assert cells[3].rstrip("\\").strip() == r"38\%"  # 20/53

    def test_shares_sum_to_100(self):
        tex = emit_study1_table(make_incidents())
        shares = []
        for line in tex.strip().splitlines():
            cell = line.split("&")[3].strip()
            cell = cell.replace("\\\\", "").replace("\\%", "").strip()
            shares.append(int(cell))
        assert sum(shares) in (99, 100, 101)

    def test_representative_modes_are_most_frequent(self):
        tex = emit_study1_table(make_incidents())
        l1 = tex.strip().splitlines()[0]
        modes_cell = l1.split("&")[1]
        # 'runaway retry loop' appears twice per cycle vs once for others
        assert modes_cell.strip().startswith("runaway retry loop")

    def test_latex_special_chars_escaped(self):
        df = make_incidents()
        df.loc[df.index[0], "mechanism_phrase"] = "cost & token % overrun"
        # make it the most frequent phrase for L1
        l1_idx = df[df.primary_layer == "L1"].index
        df.loc[l1_idx, "mechanism_phrase"] = "cost & token % overrun"
        tex = emit_study1_table(df)
        assert r"cost \& token \% overrun" in tex

    def test_undersized_input_raises(self):
        with pytest.raises(ValueError, match=r"N=13.*50"):
            emit_study1_table(make_incidents(5, 4, 2, 2))

    def test_empty_input_raises(self):
        with pytest.raises(ValueError):
            emit_study1_table(pd.DataFrame())


class TestStudy1Stats:
    def test_headline_numbers(self):
        stats = study1_stats(make_incidents(20, 15, 10, 8), kappa=0.72)
        assert stats["n_included"] == 53
        assert stats["n_screened"] == 58
        assert stats["per_layer_counts"] == {"L1": 20, "L2": 15, "L3": 10, "L4": 8}
        assert abs(stats["l2_l3_share"] - 25 / 53) < 1e-9
        assert stats["cohens_kappa"] == 0.72

    def test_cooccurrence_counts_secondaries(self):
        stats = study1_stats(make_incidents(), kappa=0.5)
        # 5 L2-primary incidents carry L4 secondary
        assert stats["cooccurrence"]["L2"]["L4"] == 5
        assert stats["cooccurrence"]["L1"]["L3"] == 0

    def test_json_serializable(self):
        stats = study1_stats(make_incidents(), kappa=0.5)
        json.dumps(stats)

    def test_coupling_fields(self):
        # Give L1-primary incidents secondary codes so the coupling headline
        # (most L1-primary incidents also carry L3/L4 secondaries) is testable.
        rows = [
            {"incident_id": f"a{i}", "included": True, "primary_layer": "L1",
             "secondary_layers": sec, "mechanism_phrase": "x"}
            for i, sec in enumerate(["L3; L4", "L3", "L4", ""])  # 4 L1
        ]
        stats = study1_stats(pd.DataFrame(rows), kappa=0.5)
        c = stats["coupling"]
        assert c["l1_primary_count"] == 4
        assert c["l1_primary_with_l3_secondary"] == 2   # rows 0,1
        assert c["l1_primary_with_l4_secondary"] == 2   # rows 0,2
        assert abs(c["l1_primary_with_l3_secondary_share"] - 0.5) < 1e-9
        assert c["n_with_any_secondary"] == 3           # rows 0,1,2
        assert abs(c["share_with_any_secondary"] - 0.75) < 1e-9
