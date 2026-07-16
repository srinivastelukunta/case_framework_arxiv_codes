"""Maturity index tests for src/study3/maturity_scorer.py (T9).

T9 will replace these skips with tests for the geometric-mean composite
(Eq. 9 in the paper), band assignment {0, .25, .5, .75, 1.0}, and edge cases
(zero layer score collapses the index; conservative coding).
"""

import pytest

pytestmark = pytest.mark.skip(reason="maturity_scorer.py is implemented in T9")


def test_geometric_mean_index():
    ...


def test_zero_layer_collapses_composite():
    ...


def test_band_assignment_edges():
    ...
