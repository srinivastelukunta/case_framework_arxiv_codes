"""Study 3: composite index, maturity levels, stats, and figure emitter."""

import pytest

from src.emit.stats_summary import study3_stats
from src.study3.index import (
    geometric_mean_index,
    maturity_level,
)


class TestIndex:
    def test_all_ones_is_one(self):
        assert geometric_mean_index({"L1": 1, "L2": 1, "L3": 1, "L4": 1}) == 1.0

    def test_zero_anywhere_collapses_to_zero(self):
        # The non-compensatory property (Eq. 9): a single zero -> composite 0.
        assert geometric_mean_index({"L1": 1, "L2": 1, "L3": 0.0, "L4": 1}) == 0.0
        assert geometric_mean_index([0.0, 0.9, 0.9, 0.9]) == 0.0

    def test_all_half_is_half(self):
        assert abs(geometric_mean_index([0.5, 0.5, 0.5, 0.5]) - 0.5) < 1e-12

    def test_known_geometric_mean(self):
        # (0.25 * 0.5 * 0.5 * 1.0)^(1/4)
        expected = (0.25 * 0.5 * 0.5 * 1.0) ** 0.25
        assert abs(geometric_mean_index([0.25, 0.5, 0.5, 1.0]) - expected) < 1e-12

    def test_accepts_list_and_dict_equally(self):
        d = geometric_mean_index({"L1": 0.5, "L2": 0.25, "L3": 0.75, "L4": 1.0})
        li = geometric_mean_index([0.5, 0.25, 0.75, 1.0])
        assert abs(d - li) < 1e-12

    def test_out_of_range_raises(self):
        with pytest.raises(ValueError):
            geometric_mean_index([1.2, 0.5, 0.5, 0.5])
        with pytest.raises(ValueError):
            geometric_mean_index([0.5, 0.5, 0.5])  # wrong length

    def test_level_boundaries(self):
        assert maturity_level(0.0) == 0
        assert maturity_level(0.19) == 0
        assert maturity_level(0.2) == 1
        assert maturity_level(0.39) == 1
        assert maturity_level(0.4) == 2
        assert maturity_level(0.6) == 3
        assert maturity_level(0.8) == 4
        assert maturity_level(1.0) == 4


def make_deployments():
    # 5 deployments; most have L2=0 (no interaction monitoring) -> composite 0.
    specs = [
        {"L1": 0.5, "L2": 0.0, "L3": 0.25, "L4": 0.5},   # composite 0 (L2)
        {"L1": 0.25, "L2": 0.0, "L3": 0.0, "L4": 0.25},  # composite 0 (L2,L3 tie)
        {"L1": 0.5, "L2": 0.25, "L3": 0.25, "L4": 0.5},  # composite > 0, level L1
        {"L1": 0.25, "L2": 0.0, "L3": 0.0, "L4": 0.5},   # composite 0
        {"L1": 0.75, "L2": 0.5, "L3": 0.5, "L4": 0.75},  # higher
    ]
    return [{"deployment": f"d{i}", "scores": s} for i, s in enumerate(specs)]


class TestStats:
    def test_headline_stats(self):
        stats = study3_stats(make_deployments())
        assert stats["n_deployments"] == 5
        # 4 of 5 have composite < 0.4 (three collapse to 0, plus the small L1)
        assert stats["share_composite_le_L1"] >= 0.6
        # L2 is the most frequent weakest layer here (0 in 4 of 5)
        assert stats["modal_weakest_layer"] == "L2"
        assert set(stats["per_layer_mean"]) == {"L1", "L2", "L3", "L4"}

    def test_weakest_frequency_counts_ties(self):
        # deployment with L2=L3=0 both counted as weakest
        stats = study3_stats([{"deployment": "x",
                               "scores": {"L1": 0.5, "L2": 0.0, "L3": 0.0, "L4": 0.5}}])
        assert stats["weakest_layer_frequency"]["L2"] == 1
        assert stats["weakest_layer_frequency"]["L3"] == 1
        assert stats["weakest_layer_frequency"]["L1"] == 0

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="no deployments"):
            study3_stats([])


class TestFigure:
    def test_renders_pdf(self, tmp_path):
        from src.emit.figure1 import render_figure1
        out = render_figure1(make_deployments(), tmp_path / "figure1.pdf")
        assert out.exists()
        head = out.read_bytes()[:5]
        assert head == b"%PDF-", "figure1 must be a valid PDF"
