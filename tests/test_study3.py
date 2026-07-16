"""Study 3: two-part maturity instrument, stats, and figure emitter."""

import pytest

from src.emit.stats_summary import study3_stats
from src.study3.index import (
    bottleneck_index,
    certified_level,
    geometric_mean_index,
)


class TestGeometricRobustnessLimit:
    def test_all_ones_is_one(self):
        assert geometric_mean_index({"L1": 1, "L2": 1, "L3": 1, "L4": 1}) == 1.0

    def test_zero_anywhere_collapses_to_zero(self):
        # The limiting non-compensatory case: a single zero -> composite 0.
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


def make_deployments():
    # 5 deployments; most have L2=0 (no interaction monitoring) -> certify L0.
    specs = [
        {"L1": 0.5, "L2": 0.0, "L3": 0.25, "L4": 0.5},   # M=0.125, L0 (L2 gate)
        {"L1": 0.25, "L2": 0.0, "L3": 0.0, "L4": 0.25},  # M=0.05,  L0 (L2,L3 tie)
        {"L1": 0.5, "L2": 0.25, "L3": 0.25, "L4": 0.5},  # M=0.30,  L1
        {"L1": 0.25, "L2": 0.0, "L3": 0.0, "L4": 0.5},   # M=0.075, L0
        {"L1": 0.75, "L2": 0.5, "L3": 0.5, "L4": 0.75},  # M=0.55,  L2
    ]
    return [{"deployment": f"d{i}", "scores": s} for i, s in enumerate(specs)]


class TestStats:
    def test_headline_stats(self):
        stats = study3_stats(make_deployments())
        assert stats["n_deployments"] == 5
        # L2 is the most frequent weakest layer here (0 in 3 of 5)
        assert stats["modal_weakest_layer"] == "L2"
        assert set(stats["per_layer_mean"]) == {"L1", "L2", "L3", "L4"}

    def test_index_block(self):
        idx = study3_stats(make_deployments())["index"]
        assert idx["alpha"] == 0.6
        assert idx["bottleneck_mean"] == pytest.approx(0.22)
        assert idx["bottleneck_median"] == pytest.approx(0.125)
        assert idx["bottleneck_min"] == pytest.approx(0.05)
        assert idx["bottleneck_max"] == pytest.approx(0.55)
        assert idx["scores"] == pytest.approx([0.125, 0.05, 0.30, 0.075, 0.55])
        assert idx["certified_level_distribution"] == {
            "L0 Ad hoc": 3, "L1 Controlled": 1, "L2 Emergence-aware": 1,
            "L3 Requisite": 0, "L4 Autonomic": 0,
        }
        # d2 and d4 have no zero layer -> geometric composites nonzero
        assert idx["geometric_robustness_limit"]["all_composites_zero"] is False

    def test_all_zero_layer_population_hits_geometric_limit(self):
        deployments = [{"deployment": "x",
                        "scores": {"L1": 0.5, "L2": 0.0, "L3": 0.25, "L4": 0.5}}]
        idx = study3_stats(deployments)["index"]
        assert idx["geometric_robustness_limit"]["all_composites_zero"] is True
        assert idx["certified_level_distribution"]["L0 Ad hoc"] == 1
        # bottleneck stays informative where the geometric limit is all-zero
        assert idx["bottleneck_mean"] > 0

    def test_alpha_is_threaded_through(self):
        deployments = make_deployments()
        lo = study3_stats(deployments, alpha=0.5)["index"]
        hi = study3_stats(deployments, alpha=0.75)["index"]
        assert lo["alpha"] == 0.5 and hi["alpha"] == 0.75
        # heavier min-weighting pulls this min-dominated population down
        assert hi["bottleneck_mean"] < lo["bottleneck_mean"]

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


class TestConsistency:
    def test_stats_agree_with_index_functions(self):
        deployments = make_deployments()
        idx = study3_stats(deployments)["index"]
        assert idx["scores"] == pytest.approx(
            [bottleneck_index(d["scores"]) for d in deployments])
        n_l0 = sum(1 for d in deployments if certified_level(d["scores"]) == 0)
        assert idx["certified_level_distribution"]["L0 Ad hoc"] == n_l0


class TestFigure:
    def test_renders_pdf(self, tmp_path):
        from src.emit.figure1 import render_figure1
        out = render_figure1(make_deployments(), tmp_path / "figure1.pdf")
        assert out.exists()
        head = out.read_bytes()[:5]
        assert head == b"%PDF-", "figure1 must be a valid PDF"
