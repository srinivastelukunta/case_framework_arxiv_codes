"""Tests for the T4 collectors: arXiv API and curated web reports, plus
merged-pool URL deduplication."""

import pandas as pd
import pytest

from src.collectors.arxiv_papers import parse_atom_feed, screen_arxiv_entries
from src.collectors.merge import dedup_by_url
from src.collectors.screen import AGENT_TERMS_DEFAULT
from src.collectors.web_reports import extract_page_text, screen_web_items

ATOM_FIXTURE = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title>ArXiv Query</title>
  <entry>
    <id>http://arxiv.org/abs/2501.11111v2</id>
    <published>2025-01-20T10:00:00Z</published>
    <title>Why Do Multi-Agent LLM Systems Fail?</title>
    <summary>We analyze failures of LLM agent systems with tool use
    across multi-step tasks and document concrete incidents.</summary>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/2201.22222v1</id>
    <published>2022-02-01T10:00:00Z</published>
    <title>Old agent safety paper</title>
    <summary>autonomous agents survey</summary>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/2503.33333v1</id>
    <published>2025-03-01T10:00:00Z</published>
    <title>Image classifier robustness</title>
    <summary>A benchmark for perturbation robustness.</summary>
  </entry>
</feed>
"""


class TestArxiv:
    def test_parse_atom_feed(self):
        entries = parse_atom_feed(ATOM_FIXTURE)
        assert len(entries) == 3
        first = entries[0]
        assert first["arxiv_id"] == "2501.11111"
        assert first["url"] == "https://arxiv.org/abs/2501.11111"
        assert first["date"] == "2025-01-20"
        assert "Multi-Agent" in first["title"]

    def test_screening_branches(self):
        entries = parse_atom_feed(ATOM_FIXTURE)
        cands = screen_arxiv_entries(
            entries, min_date="2023-01-01", terms=AGENT_TERMS_DEFAULT
        )
        by_id = {c["candidate_id"]: c for c in cands}
        assert by_id["arxiv-2501.11111"]["screen_included"] is True
        assert by_id["arxiv-2201.22222"]["screen_exclusion_reason"] == "pre_2023"
        assert by_id["arxiv-2503.33333"]["screen_exclusion_reason"] == "no_agent_terms"

    def test_duplicate_arxiv_ids_across_queries_collapse(self):
        """The same paper returned by two search terms yields one entry."""
        entries = parse_atom_feed(ATOM_FIXTURE) + parse_atom_feed(ATOM_FIXTURE)
        cands = screen_arxiv_entries(
            entries, min_date="2023-01-01", terms=AGENT_TERMS_DEFAULT
        )
        ids = [c["candidate_id"] for c in cands]
        assert len(ids) == len(set(ids)) == 3


class TestWebReports:
    HTML = """<html><head><title>Postmortem: agent deleted database</title></head>
    <body><nav>menu</nav><p>The autonomous agent executed a destructive
    command during a code freeze after multi-step tool use.</p></body></html>"""

    def test_extract_page_text(self):
        title, text = extract_page_text(self.HTML)
        assert "Postmortem" in title
        assert "destructive" in text
        assert "menu" not in text  # nav/script stripped

    def test_screen_web_items_uses_page_text(self):
        items = [
            {
                "id": "replit_db_deletion_2025",
                "title": "Replit incident",
                "date": "2025-07",
                "status": "seed",
                "url": "https://example.org/replit",
            }
        ]
        cands = screen_web_items(
            items,
            page_texts={"replit_db_deletion_2025": self.HTML},
            min_date="2023-01-01",
            terms=AGENT_TERMS_DEFAULT,
        )
        assert cands[0]["screen_included"] is True
        assert cands[0]["candidate_id"] == "web-replit_db_deletion_2025"

    def test_rejected_items_are_skipped(self):
        items = [
            {"id": "x", "title": "t", "date": "2025-01", "status": "rejected",
             "url": "https://example.org/x"},
        ]
        assert screen_web_items(items, {}, "2023-01-01", AGENT_TERMS_DEFAULT) == []

    def test_fetch_failure_falls_back_to_metadata(self):
        """No page text -> screen on curated title so the candidate row
        still exists with its evidence URL."""
        items = [
            {"id": "y", "title": "Autonomous agent cost overrun postmortem",
             "date": "2025-02", "status": "proposed",
             "url": "https://example.org/y"},
        ]
        cands = screen_web_items(items, {}, "2023-01-01", AGENT_TERMS_DEFAULT)
        assert cands[0]["screen_included"] is True

    def test_item_without_url_raises(self):
        items = [{"id": "z", "title": "t", "date": "2025-01", "status": "seed",
                  "url": None}]
        with pytest.raises(ValueError, match="url"):
            screen_web_items(items, {}, "2023-01-01", AGENT_TERMS_DEFAULT)


class TestMergeDedup:
    def test_dedup_by_url_keeps_first_and_reports(self):
        rows = [
            {"candidate_id": "aiid-1", "source": "aiid", "url": "https://a.example/x"},
            {"candidate_id": "web-1", "source": "web_reports", "url": "https://a.example/x"},
            {"candidate_id": "arxiv-1", "source": "arxiv", "url": "https://b.example/y"},
        ]
        kept, dropped = dedup_by_url(rows)
        assert [r["candidate_id"] for r in kept] == ["aiid-1", "arxiv-1"]
        assert dropped == {"web-1": "aiid-1"}

    def test_landing_page_sources_are_never_url_deduped(self):
        """Regression: all MIT rows cite the repository landing page; they
        must not collapse into one candidate."""
        rows = [
            {"candidate_id": f"mit-{i}", "source": "mit_risk_repo",
             "url": "https://airisk.mit.edu/risks"}
            for i in range(3)
        ]
        kept, dropped = dedup_by_url(
            rows, skip_sources=frozenset({"mit_risk_repo"})
        )
        assert len(kept) == 3 and dropped == {}
