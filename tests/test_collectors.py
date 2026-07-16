"""Tests for the AIID and MIT collectors' screening logic (T3).

Screening is the collection-stage keyword/date filter that builds the
candidate pool; the full Appendix B inclusion filter runs at coding time
(T5) via src/coding/protocol.py.
"""

from pathlib import Path

import pandas as pd
import yaml

from src.collectors.aiid import screen_aiid_incidents
from src.collectors.mit_risk_repo import screen_mit_rows
from src.collectors.screen import (
    AGENT_TERMS_DEFAULT,
    match_agent_terms,
    write_candidates_csv,
)

ROOT = Path(__file__).resolve().parents[1]


class TestTermMatching:
    def test_matches_agent_and_variants(self):
        assert "agent" in match_agent_terms(
            "An autonomous agentic system deleted data", AGENT_TERMS_DEFAULT
        )

    def test_matches_multiword_terms(self):
        text = "the model used tool use and multi-step planning"
        found = match_agent_terms(text, AGENT_TERMS_DEFAULT)
        assert "tool use" in found and "multi-step" in found

    def test_no_match_returns_empty(self):
        assert match_agent_terms("a simple chatbot reply", AGENT_TERMS_DEFAULT) == []

    def test_agent_does_not_match_inside_words(self):
        # 'reagent' must not count as 'agent'
        assert match_agent_terms("chemical reagent spill", AGENT_TERMS_DEFAULT) == []

    def test_jailbreak_needs_explicit_variant(self):
        """Prefix matching means 'jailbreak' does NOT cover 'jailbroken';
        the widened AIID list must carry both variants."""
        assert match_agent_terms("Jailbroken Lovable AI", ("jailbreak",)) == []
        assert match_agent_terms("Jailbroken Lovable AI", ("jailbroken",)) == [
            "jailbroken"
        ]


class TestScreenTermsConfig:
    """Pool-expansion decision (author approved 2026-07-16): the AIID screen
    is widened, while MIT / arXiv keep the narrow base terms — both yielded
    zero included incidents in the v2 coding run, so widening them would
    only add coding cost with no expected includes."""

    def load_sources(self):
        with open(ROOT / "config" / "sources.yaml", encoding="utf-8") as f:
            return yaml.safe_load(f)["sources"]

    def test_aiid_terms_widened(self):
        terms = self.load_sources()["aiid"]["filter_terms"]
        for base in AGENT_TERMS_DEFAULT:
            assert base in terms, f"base term {base!r} dropped from AIID screen"
        for added in (
            "assistant",
            "copilot",
            "chatbot",
            "prompt injection",
            "jailbreak",
            "jailbroken",
            "automated system",
        ):
            assert added in terms, f"widened term {added!r} missing"

    def test_mit_and_arxiv_keep_narrow_terms(self):
        sources = self.load_sources()
        for src in ("mit_risk_repo", "arxiv"):
            assert tuple(sources[src]["filter_terms"]) == AGENT_TERMS_DEFAULT, (
                f"{src} must keep the narrow base terms (zero-yield source)"
            )


class TestAiidScreening:
    def make_df(self):
        return pd.DataFrame(
            [
                {  # included: recent + agent terms
                    "incident_id": 900,
                    "date": "2025-03-10",
                    "title": "Autonomous agent deletes production database",
                    "description": "An LLM agent with tool use ran a drop table.",
                },
                {  # excluded: pre-2023
                    "incident_id": 12,
                    "date": "2019-05-01",
                    "title": "Autonomous agent malfunction",
                    "description": "agentic loop failure",
                },
                {  # excluded: no agent terms
                    "incident_id": 901,
                    "date": "2024-01-01",
                    "title": "Chatbot gives wrong answer",
                    "description": "A model output contained an error.",
                },
            ]
        )

    def test_screening_branches(self):
        cands = screen_aiid_incidents(
            self.make_df(),
            min_date="2023-01-01",
            terms=AGENT_TERMS_DEFAULT,
            url_template="https://incidentdatabase.ai/cite/{incident_id}",
        )
        by_id = {c["candidate_id"]: c for c in cands}
        assert by_id["aiid-900"]["screen_included"] is True
        assert by_id["aiid-900"]["url"] == "https://incidentdatabase.ai/cite/900"
        assert "agent" in by_id["aiid-900"]["matched_terms"]
        assert by_id["aiid-12"]["screen_included"] is False
        assert by_id["aiid-12"]["screen_exclusion_reason"] == "pre_2023"
        assert by_id["aiid-901"]["screen_included"] is False
        assert by_id["aiid-901"]["screen_exclusion_reason"] == "no_agent_terms"

    def test_pre_2023_checked_before_terms(self):
        """Date gate first: an old incident with terms codes pre_2023."""
        cands = screen_aiid_incidents(
            self.make_df(),
            min_date="2023-01-01",
            terms=AGENT_TERMS_DEFAULT,
            url_template="x{incident_id}",
        )
        old = next(c for c in cands if c["candidate_id"] == "aiid-12")
        assert old["screen_exclusion_reason"] == "pre_2023"

    def test_force_include_overrides_term_gate(self):
        """A vetted id with no agent terms is force-included (term gate only)."""
        cands = screen_aiid_incidents(
            self.make_df(),
            min_date="2023-01-01",
            terms=AGENT_TERMS_DEFAULT,
            url_template="x{incident_id}",
            force_include_ids=frozenset({"901"}),
        )
        forced = next(c for c in cands if c["candidate_id"] == "aiid-901")
        assert forced["screen_included"] is True
        assert forced["matched_terms"] == "curated_targeted"

    def test_force_include_does_not_override_date_gate(self):
        """A forced id before min_date stays excluded (date scope is firm)."""
        cands = screen_aiid_incidents(
            self.make_df(),
            min_date="2023-01-01",
            terms=AGENT_TERMS_DEFAULT,
            url_template="x{incident_id}",
            force_include_ids=frozenset({"12"}),
        )
        forced = next(c for c in cands if c["candidate_id"] == "aiid-12")
        assert forced["screen_included"] is False
        assert forced["screen_exclusion_reason"] == "pre_2023"


class TestMitScreening:
    def make_df(self):
        return pd.DataFrame(
            [
                {
                    "QuickRef": "Smith2025",
                    "Ev_ID": "R1.2",
                    "Title": "Multi-agent cascade risk",
                    "Description": "Agents propagate errors through shared memory.",
                    "Date": "2025",
                },
                {
                    "QuickRef": "Old2019",
                    "Ev_ID": "R0.1",
                    "Title": "Agent misalignment",
                    "Description": "agentic drift",
                    "Date": "2019",
                },
                {
                    "QuickRef": "Plain2024",
                    "Ev_ID": "R2.9",
                    "Title": "Bias in image classifier",
                    "Description": "Training data skew.",
                    "Date": "2024",
                },
            ]
        )

    def test_screening_branches(self):
        cands = screen_mit_rows(
            self.make_df(), min_year=2023, terms=AGENT_TERMS_DEFAULT
        )
        by_id = {c["candidate_id"]: c for c in cands}
        assert by_id["mit-R1.2"]["screen_included"] is True
        assert by_id["mit-R0.1"]["screen_included"] is False
        assert by_id["mit-R0.1"]["screen_exclusion_reason"] == "pre_2023"
        assert by_id["mit-R2.9"]["screen_included"] is False
        assert by_id["mit-R2.9"]["screen_exclusion_reason"] == "no_agent_terms"


class TestCandidatesCsv:
    def test_write_is_deterministic_and_complete(self, tmp_path):
        rows = [
            {
                "candidate_id": "aiid-2",
                "source": "aiid",
                "source_ref": "2",
                "url": "https://incidentdatabase.ai/cite/2",
                "date": "2025-01-01",
                "title": "B",
                "text_snippet": "agent",
                "matched_terms": "agent",
                "screen_included": True,
                "screen_exclusion_reason": "",
            },
            {
                "candidate_id": "aiid-1",
                "source": "aiid",
                "source_ref": "1",
                "url": "https://incidentdatabase.ai/cite/1",
                "date": "2025-01-02",
                "title": "A",
                "text_snippet": "agent",
                "matched_terms": "agent",
                "screen_included": False,
                "screen_exclusion_reason": "no_agent_terms",
            },
        ]
        out = tmp_path / "cands.csv"
        write_candidates_csv(rows, out)
        first = out.read_bytes()
        df = pd.read_csv(out)
        # sorted by candidate_id, every row keeps its exclusion decision
        assert list(df["candidate_id"]) == ["aiid-1", "aiid-2"]
        assert df.loc[0, "screen_exclusion_reason"] == "no_agent_terms"
        # byte-identical on rewrite (determinism contract)
        write_candidates_csv(list(reversed(rows)), out)
        assert out.read_bytes() == first

    def test_empty_candidate_set_raises(self, tmp_path):
        """No-fabrication guardrail: refuse to write an empty pool."""
        import pytest

        with pytest.raises(ValueError, match="empty"):
            write_candidates_csv([], tmp_path / "cands.csv")
