"""Study 2: tool-capability coder + emitter tests."""

import json

import pytest
import yaml

from src.emit.latex_tables import emit_study2_table
from src.emit.stats_summary import study2_stats
from src.study2.code_tools import (
    TOOL_SCHEMA,
    build_prompt,
    code_tool,
)

MECHANISMS = {
    "layers": {
        "L1": {"name": "Control", "classes": [
            {"mechanism": "Budget caps"}, {"mechanism": "Circuit breakers"}]},
        "L2": {"name": "Adaptive", "classes": [
            {"mechanism": "Contagion controls"}]},
        "L3": {"name": "Supervisory", "classes": [
            {"mechanism": "Approval gates"}]},
        "L4": {"name": "Ops", "classes": [{"mechanism": "Evals in CI/CD"}]},
    }
}

TOOL = {
    "id": "acme",
    "name": "Acme Guard",
    "vendor": "Acme",
    "doc_urls": ["https://acme.example/docs"],
}


def canned(l1="Full", l2="None", l3="Partial", l4="Full"):
    def cell(v):
        return {"verdict": v,
                "evidence": "" if v == "None" else "documented mechanism",
                "evidence_url": "" if v == "None" else "https://acme.example/docs"}
    return json.dumps({"L1": cell(l1), "L2": cell(l2), "L3": cell(l3), "L4": cell(l4)})


class TestCoder:
    def test_prompt_includes_frame_and_doc(self):
        p = build_prompt(TOOL, "Acme enforces token budget caps.", MECHANISMS)
        assert "Budget caps" in p and "Approval gates" in p  # mechanism frame
        assert "token budget caps" in p                       # doc text
        assert "Acme Guard" in p

    def test_schema_forbids_extra_and_enumerates_verdicts(self):
        assert TOOL_SCHEMA["additionalProperties"] is False
        assert set(TOOL_SCHEMA["properties"]) == {"L1", "L2", "L3", "L4"}
        enum = TOOL_SCHEMA["properties"]["L1"]["properties"]["verdict"]["enum"]
        assert enum == ["Full", "Partial", "None"]

    def test_code_tool_caches_and_normalizes(self, tmp_path):
        calls = []

        def fake(prompt):
            calls.append(prompt)
            return canned()

        cells = code_tool(TOOL, "doc", MECHANISMS, tmp_path, call_model=fake)
        assert cells["L1"]["verdict"] == "Full"
        assert cells["L2"] == {"verdict": "None", "evidence": "", "evidence_url": ""}
        assert (tmp_path / "acme.json").exists()

        cells2 = code_tool(TOOL, "doc", MECHANISMS, tmp_path, call_model=fake)
        assert cells2 == cells
        assert len(calls) == 1, "second call must be served from cache"

    def test_none_cell_clears_evidence(self, tmp_path):
        # Even if the model returns stray evidence on a None cell, normalize it.
        raw = json.dumps({
            "L1": {"verdict": "None", "evidence": "leftover", "evidence_url": "x"},
            "L2": {"verdict": "None", "evidence": "", "evidence_url": ""},
            "L3": {"verdict": "None", "evidence": "", "evidence_url": ""},
            "L4": {"verdict": "None", "evidence": "", "evidence_url": ""},
        })
        cells = code_tool(TOOL, "doc", MECHANISMS, tmp_path, call_model=lambda p: raw)
        assert cells["L1"] == {"verdict": "None", "evidence": "", "evidence_url": ""}


def make_coded(n_full_l1=12, n=20):
    """n tools; n_full_l1 have L1=Full, rest L1=None; L2 always None;
    L3 Full on first 2; L4 Full on all."""
    tools = []
    for i in range(n):
        tools.append({
            "id": f"t{i}", "name": f"Tool {i}", "category": "cat",
            "cells": {
                "L1": {"verdict": "Full" if i < n_full_l1 else "None",
                       "evidence": "", "evidence_url": ""},
                "L2": {"verdict": "None", "evidence": "", "evidence_url": ""},
                "L3": {"verdict": "Full" if i < 2 else "None",
                       "evidence": "", "evidence_url": ""},
                "L4": {"verdict": "Full", "evidence": "", "evidence_url": ""},
            },
        })
    return tools


class TestStudy2Table:
    def test_rows_and_coverage_footer(self):
        tex = emit_study2_table(make_coded(n_full_l1=10, n=20))
        lines = [l for l in tex.strip().splitlines() if l.strip()]
        # 20 tool rows + \midrule + footer
        assert lines[0].startswith("Tool 0 &")
        assert lines[0].count("&") == 4  # Tool + 4 layer cells
        assert r"\midrule" in tex
        footer = lines[-1]
        assert footer.startswith(r"\textbf{Column coverage (Full)}")
        # L1 Full 10/20=50%, L2 0%, L3 2/20=10%, L4 100%
        assert "50\\%" in footer and "0\\%" in footer and "10\\%" in footer and "100\\%" in footer

    def test_undersized_registry_raises(self):
        with pytest.raises(ValueError, match="below the minimum"):
            emit_study2_table(make_coded(n=10))

    def test_tool_name_latex_escaped(self):
        coded = make_coded(n=15)
        coded[0]["name"] = "A&B_Tool"
        tex = emit_study2_table(coded)
        assert r"A\&B\_Tool" in tex


class TestStudy2Stats:
    def test_coverage_counts_and_shares(self):
        stats = study2_stats(make_coded(n_full_l1=12, n=20))
        assert stats["n_tools"] == 20
        l1 = stats["per_layer_coverage"]["L1"]
        assert l1["Full"] == 12 and l1["None"] == 8
        assert abs(l1["full_share"] - 0.6) < 1e-9
        assert stats["per_layer_coverage"]["L2"]["Full"] == 0
        assert stats["full_coverage_share"]["L4"] == 1.0

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="no coded tools"):
            study2_stats([])

    def test_registry_config_valid(self):
        from pathlib import Path
        cfg = yaml.safe_load(
            (Path(__file__).resolve().parents[1] / "config" / "tools.yaml")
            .read_text(encoding="utf-8"))
        approved = [t for t in cfg["tools"] if t.get("status") == "approved"]
        assert 15 <= len(approved) <= 25
        for t in approved:
            assert t["id"] and t["name"] and t["doc_urls"] and t["category"]
        ids = [t["id"] for t in cfg["tools"]]
        assert len(ids) == len(set(ids))
