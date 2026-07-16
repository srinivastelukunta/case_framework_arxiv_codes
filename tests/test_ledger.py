"""Tests for the citation ledger (data/citation_ledger.csv)."""

import pandas as pd
import pytest

from src.ledger import LEDGER_COLUMNS, upsert_ledger_entries


def entry(item_id, url="https://example.org/x", **overrides):
    base = {
        "item_id": item_id,
        "study": "study1",
        "source": "aiid",
        "title": "Some evidence item",
        "url": url,
        "accessed_date": "2026-07-15",
        "license_note": "public source",
    }
    base.update(overrides)
    return base


class TestLedger:
    def test_creates_file_with_columns(self, tmp_path):
        path = tmp_path / "citation_ledger.csv"
        upsert_ledger_entries([entry("a")], path)
        df = pd.read_csv(path)
        assert list(df.columns) == list(LEDGER_COLUMNS)
        assert len(df) == 1

    def test_upsert_replaces_same_item_id(self, tmp_path):
        path = tmp_path / "citation_ledger.csv"
        upsert_ledger_entries([entry("a", title="old")], path)
        upsert_ledger_entries([entry("a", title="new")], path)
        df = pd.read_csv(path)
        assert len(df) == 1
        assert df.loc[0, "title"] == "new"

    def test_appends_new_items_and_sorts(self, tmp_path):
        path = tmp_path / "citation_ledger.csv"
        upsert_ledger_entries([entry("b")], path)
        upsert_ledger_entries([entry("a")], path)
        df = pd.read_csv(path)
        assert list(df["item_id"]) == ["a", "b"]

    def test_entry_without_url_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="url"):
            upsert_ledger_entries(
                [entry("a", url="")], tmp_path / "citation_ledger.csv"
            )
