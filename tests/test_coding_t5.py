"""Tests for T5: kappa, LLM pre-coder plumbing, and adjudication merge.

All offline: the API call is injectable; tests use canned JSON responses.
"""

import json

import pytest

from src.coding.adjudicate import (
    build_final_dataset,
    dedup_incident_twins,
    merge_passes,
)
from src.coding.kappa import cohens_kappa
from src.coding.llm_precoder import (
    CODING_SCHEMA,
    CodingRefused,
    build_prompt,
    features_from_response,
    label_for,
    run_pass,
)


class TestKappa:
    def test_perfect_agreement_is_one(self):
        labels = ["L1", "L2", "excluded", "L3"]
        assert cohens_kappa(labels, labels) == 1.0

    def test_known_value(self):
        # Classic 2-category example: po=0.7, pe=0.5 -> kappa=0.4
        a = ["x"] * 35 + ["x"] * 15 + ["y"] * 15 + ["y"] * 35
        b = ["x"] * 35 + ["y"] * 15 + ["x"] * 15 + ["y"] * 35
        assert abs(cohens_kappa(a, b) - 0.4) < 1e-9

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            cohens_kappa(["L1"], ["L1", "L2"])


def canned_response(**overrides):
    base = {
        "codeable": True,
        "exclusion_reason": None,
        "system": "Test Agent",
        "date": "2025-03-01",
        "autonomous_multistep": True,
        "mechanism_detail": True,
        "independent_reporting": True,
        "self_reported_with_specifics": False,
        "single_agent_loop_would_prevent": True,
        "required_interagent_or_shared_state": False,
        "oversight_point_existed": False,
        "oversight_deficient": False,
        "operational_mechanism": False,
        "mechanism_phrase": "runaway retry loop",
        "rationale": "Budget caps would have stopped it.",
        "confidence": 0.9,
    }
    base.update(overrides)
    return base


CANDIDATE = {
    "candidate_id": "aiid-900",
    "source": "aiid",
    "url": "https://incidentdatabase.ai/cite/900",
    "date": "2025-03-01",
    "title": "Agent deletes database",
    "text_snippet": "An agent ran a destructive command.",
}


class TestPrecoder:
    def test_prompts_differ_between_passes(self):
        a = build_prompt("A", CANDIDATE)
        b = build_prompt("B", CANDIDATE)
        assert a != b
        for prompt in (a, b):
            assert "aiid-900" not in prompt  # no id leakage to bias coding
            assert "Agent deletes database" in prompt
            assert "earliest" in prompt.lower()  # core decision rule present

    def test_schema_forbids_extra_properties(self):
        assert CODING_SCHEMA["additionalProperties"] is False
        assert set(CODING_SCHEMA["required"]) == set(CODING_SCHEMA["properties"])

    def test_features_and_label(self):
        feats = features_from_response(canned_response(), CANDIDATE)
        assert feats.incident_id == "aiid-900"
        assert feats.system == "Test Agent"
        assert label_for(canned_response(), CANDIDATE) == "L1"

    def test_label_excluded_when_not_codeable(self):
        resp = canned_response(codeable=False, exclusion_reason="not_agentic")
        assert label_for(resp, CANDIDATE) == "excluded"

    def test_run_pass_caches_responses(self, tmp_path):
        calls = []

        def fake_call(prompt):
            calls.append(prompt)
            return json.dumps(canned_response())

        rows, refused = run_pass(
            "A", [CANDIDATE], cache_dir=tmp_path, call_model=fake_call
        )
        assert len(rows) == 1 and rows[0]["label"] == "L1"
        assert refused == []
        assert (tmp_path / "passA" / "aiid-900.json").exists()

        rows2, _ = run_pass("A", [CANDIDATE], cache_dir=tmp_path, call_model=fake_call)
        assert rows2 == rows
        assert len(calls) == 1, "second run must be served from cache"

    def test_run_pass_invalid_json_raises(self, tmp_path):
        with pytest.raises(ValueError, match="aiid-900"):
            run_pass(
                "A", [CANDIDATE], cache_dir=tmp_path, call_model=lambda p: "not json"
            )

    def test_run_pass_records_refusal_and_continues(self, tmp_path):
        """A refusal on one candidate is recorded and skipped; the batch's
        other candidates still code, and no cache/row is written for the
        refused one (never fabricate a code for a refusal)."""
        ok = dict(CANDIDATE, candidate_id="aiid-900")
        bad = dict(CANDIDATE, candidate_id="web-cyber-exploit")

        def fake_call(prompt):
            if "web-cyber-exploit" in prompt or "cyber-exploit" in prompt:
                raise CodingRefused("refused")
            return json.dumps(canned_response())

        # route by title so the refusal targets the right candidate
        ok["title"] = "benign agent incident"
        bad["title"] = "web-cyber-exploit RCE"
        rows, refused = run_pass(
            "A", [ok, bad], cache_dir=tmp_path, call_model=fake_call
        )
        assert [r["candidate_id"] for r in rows] == ["aiid-900"]
        assert refused == ["web-cyber-exploit"]
        assert not (tmp_path / "passA" / "web-cyber-exploit.json").exists()


def coded_row(candidate_id, label, secondaries="", rationale="r", **extra):
    row = {
        "candidate_id": candidate_id,
        "source": "aiid",
        "url": f"https://example.org/{candidate_id}",
        "date": "2025-03-01",
        "system": f"sys-{candidate_id}",
        "label": label,
        "exclusion_reason": "" if label != "excluded" else "not_agentic",
        "secondary_layers": secondaries,
        "mechanism_phrase": "phrase",
        "rationale": rationale,
        "confidence": 0.9,
    }
    row.update(extra)
    return row


class TestAdjudication:
    def test_agreement_needs_no_adjudication(self):
        merged = merge_passes(
            [coded_row("c1", "L1")], [coded_row("c1", "L1", secondaries="L4")]
        )
        (row,) = merged
        assert row["agreed"] is True
        assert row["final_label"] == "L1"
        # union of secondaries across passes
        assert row["secondary_layers"] == "L4"

    def test_disagreement_flagged(self):
        merged = merge_passes([coded_row("c1", "L1")], [coded_row("c1", "L3")])
        (row,) = merged
        assert row["agreed"] is False
        assert row["final_label"] is None

    def test_final_dataset_applies_adjudications(self):
        merged = merge_passes([coded_row("c1", "L1")], [coded_row("c1", "L3")])
        final = build_final_dataset(
            merged,
            adjudications={"c1": {"final_label": "L3", "rationale": "author: variety deficit"}},
        )
        (row,) = final
        assert row["primary_layer"] == "L3"
        assert row["adjudicated"] is True
        assert "author" in row["rationale"]

    def test_unresolved_disagreement_raises(self):
        merged = merge_passes([coded_row("c1", "L1")], [coded_row("c1", "L3")])
        with pytest.raises(ValueError, match="unresolved"):
            build_final_dataset(merged, adjudications={})

    def test_agreed_exclusion_produces_excluded_row(self):
        merged = merge_passes(
            [coded_row("c2", "excluded")], [coded_row("c2", "excluded")]
        )
        final = build_final_dataset(merged, adjudications={})
        (row,) = final
        assert row["included"] is False
        assert row["primary_layer"] == ""


class TestIncidentTwinDedup:
    def _final(self, *pairs):
        """pairs of (candidate_id, label) -> merged agreement final rows."""
        rows_a = [coded_row(cid, lbl) for cid, lbl in pairs]
        rows_b = [coded_row(cid, lbl) for cid, lbl in pairs]
        return build_final_dataset(merge_passes(rows_a, rows_b), adjudications={})

    def test_both_twins_included_drops_web_keeps_aiid(self):
        final = self._final(("web-x", "L1"), ("aiid-9", "L1"))
        final, dropped = dedup_incident_twins(final, {"web-x": "aiid-9"})
        by = {r["incident_id"]: r for r in final}
        assert by["aiid-9"]["included"] is True
        assert by["web-x"]["included"] is False
        assert by["web-x"]["exclusion_reason"] == "duplicate_of_aiid-9"
        assert [r["incident_id"] for r in dropped] == ["web-x"]

    def test_no_drop_when_aiid_twin_excluded(self):
        # web twin is the only included copy -> must be kept (no double count)
        final = self._final(("web-x", "L1"), ("aiid-9", "excluded"))
        final, dropped = dedup_incident_twins(final, {"web-x": "aiid-9"})
        by = {r["incident_id"]: r for r in final}
        assert by["web-x"]["included"] is True
        assert dropped == []

    def test_no_overlap_is_noop(self):
        final = self._final(("web-x", "L1"), ("aiid-9", "L1"))
        _, dropped = dedup_incident_twins(final, {})
        assert dropped == []
