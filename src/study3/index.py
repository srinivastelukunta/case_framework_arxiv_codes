"""CASE composite maturity index (Equation 9) and level mapping for Study 3.

The composite is the GEOMETRIC MEAN of the four layer maturities m_l in [0,1]:
    M_CASE = (m_L1 * m_L2 * m_L3 * m_L4) ** (1/4)
This is deliberately non-compensatory (Sec. 4.4 / N4): a zero on any layer
drives the composite to zero, so excellence at three layers cannot buy back
an absent fourth. The five maturity levels L0..L4 (Table tab:levels) partition
the composite with bin edges at 0.2/0.4/0.6/0.8.
"""

from __future__ import annotations

from math import prod

LAYERS = ("L1", "L2", "L3", "L4")
LEVEL_EDGES = (0.2, 0.4, 0.6, 0.8)  # M_CASE cut points between L0..L4
LEVEL_NAMES = ("L0 Ad hoc", "L1 Controlled", "L2 Emergence-aware",
               "L3 Requisite", "L4 Autonomic")


def geometric_mean_index(scores: dict[str, float] | list[float]) -> float:
    """Composite M_CASE = geometric mean of the four layer scores.

    Accepts a {layer: score} dict or a 4-element list. Each score must be in
    [0,1]. Returns 0.0 if any layer is 0 (the non-compensatory property).
    """
    values = ([scores[l] for l in LAYERS] if isinstance(scores, dict)
              else list(scores))
    if len(values) != 4:
        raise ValueError(f"expected 4 layer scores, got {len(values)}")
    for v in values:
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"layer score {v} out of range [0,1]")
    if any(v == 0.0 for v in values):
        return 0.0
    return prod(values) ** (1.0 / 4.0)


def maturity_level(m: float) -> int:
    """Map a composite index in [0,1] to a maturity level 0..4."""
    if not (0.0 <= m <= 1.0):
        raise ValueError(f"composite {m} out of range [0,1]")
    level = 0
    for edge in LEVEL_EDGES:
        if m >= edge:
            level += 1
    return level
