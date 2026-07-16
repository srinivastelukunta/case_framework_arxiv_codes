"""Two-part maturity instrument tests (paper v7, Sec. 6.2).

Covers the bottleneck score (Eq. 9), the certification gate (Eq. 10),
the Appendix A worked example, and the geometric-mean composite retained
as the robustness limit.
"""

from pathlib import Path

import pytest
import yaml

from src.study3.index import (
    DEFAULT_ALPHA,
    TAU,
    bottleneck_index,
    certified_level,
    geometric_mean_index,
)

ROOT = Path(__file__).resolve().parents[1]


class TestWorkedExampleAppendixA:
    def test_base_profile(self):
        m = [0.75, 0.25, 0.50, 0.75]
        assert bottleneck_index(m) == pytest.approx(0.375)
        assert certified_level(m) == 1

    def test_raising_the_minimum_lifts_score_and_level(self):
        m = [0.75, 0.50, 0.50, 0.75]
        assert bottleneck_index(m) == pytest.approx(0.55)
        assert certified_level(m) == 2

    def test_raising_a_non_minimum_cannot_lift_the_level(self):
        # From the base profile, maxing m1 moves M only through the mean
        # term; the minimum still gates certification at L1.
        m = [1.00, 0.25, 0.50, 0.75]
        assert bottleneck_index(m) == pytest.approx(0.40)
        assert certified_level(m) == 1


class TestEdgeCases:
    def test_all_zeros(self):
        assert bottleneck_index([0.0, 0.0, 0.0, 0.0]) == 0.0
        assert certified_level([0.0, 0.0, 0.0, 0.0]) == 0

    def test_all_ones(self):
        assert bottleneck_index([1.0, 1.0, 1.0, 1.0]) == pytest.approx(1.0)
        assert certified_level([1.0, 1.0, 1.0, 1.0]) == 4

    def test_tied_minimum_min_term_moves_only_when_minimum_rises(self):
        # alpha=1 isolates the min term of Eq. 9. Raising one of two tied
        # minima leaves it unchanged; it moves once the minimum itself rises.
        assert bottleneck_index([0.25, 0.25, 1.0, 1.0], alpha=1.0) == pytest.approx(0.25)
        assert bottleneck_index([0.25, 1.00, 1.0, 1.0], alpha=1.0) == pytest.approx(0.25)
        assert bottleneck_index([0.50, 0.50, 1.0, 1.0], alpha=1.0) == pytest.approx(0.50)

    def test_gate_boundary_exact_quarters_certify_exactly_L1(self):
        assert certified_level([0.25, 0.25, 0.25, 0.25]) == 1


class TestGateThresholds:
    def test_tau_matches_paper(self):
        assert TAU == {1: 0.25, 2: 0.50, 3: 0.75, 4: 1.00}

    @pytest.mark.parametrize("floor,level", [(0.0, 0), (0.2, 0), (0.25, 1),
                                             (0.5, 2), (0.75, 3), (1.0, 4)])
    def test_uniform_profiles_certify_by_floor(self, floor, level):
        assert certified_level([floor] * 4) == level


class TestInterface:
    def test_accepts_dict_and_list_equally(self):
        d = bottleneck_index({"L1": 0.75, "L2": 0.25, "L3": 0.50, "L4": 0.75})
        li = bottleneck_index([0.75, 0.25, 0.50, 0.75])
        assert d == pytest.approx(li)
        assert certified_level({"L1": 0.75, "L2": 0.25, "L3": 0.50, "L4": 0.75}) == 1

    def test_out_of_range_or_wrong_length_raises(self):
        with pytest.raises(ValueError):
            bottleneck_index([1.2, 0.5, 0.5, 0.5])
        with pytest.raises(ValueError):
            bottleneck_index([0.5, 0.5, 0.5])
        with pytest.raises(ValueError):
            certified_level([-0.1, 0.5, 0.5, 0.5])
        with pytest.raises(ValueError):
            certified_level([0.5, 0.5, 0.5, 0.5, 0.5])

    def test_alpha_out_of_range_raises(self):
        with pytest.raises(ValueError):
            bottleneck_index([0.5, 0.5, 0.5, 0.5], alpha=1.5)


class TestConfigAndRobustnessLimit:
    def test_config_alpha_matches_default(self):
        with open(ROOT / "config" / "study3.yaml", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        assert cfg["index"]["alpha"] == DEFAULT_ALPHA == 0.6
        lo, hi = cfg["index"]["alpha_sensitivity_range"]
        assert (lo, hi) == (0.5, 0.75)

    def test_geometric_mean_retained_and_collapses_on_zero(self):
        # Kept as the robustness limit (Sec. 6.2, Study 3): any zero layer
        # zeroes the composite, which excellence elsewhere cannot buy back.
        assert geometric_mean_index([0.0, 0.9, 0.9, 0.9]) == 0.0
        assert geometric_mean_index([0.5, 0.5, 0.5, 0.5]) == pytest.approx(0.5)
