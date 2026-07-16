"""CASE two-part maturity instrument for Study 3 (paper v7, Sec. 6.2).

Part 1 — bottleneck score (Eq. 9):
    M_CASE = alpha * min(m) + (1 - alpha) * mean(m),   alpha = 0.6 default.
Weakest-layer dominated but continuous: an absent layer no longer collapses
the score to exactly zero, yet excellence at three layers cannot buy back
the missing fourth at full value.

Part 2 — certification gate (Eq. 10):
    level(m) = max{k : m_l >= tau_k for all layers}, else L0,
so the certified level L0..L4 is set by the weakest layer alone.

The former geometric-mean composite (pre-v7 Eq. 9) is retained as the
robustness limit reported alongside the instrument (Sec. 6.2, Study 3).
"""

from __future__ import annotations

from math import prod

LAYERS = ("L1", "L2", "L3", "L4")
LEVEL_NAMES = ("L0 Ad hoc", "L1 Controlled", "L2 Emergence-aware",
               "L3 Requisite", "L4 Autonomic")
DEFAULT_ALPHA = 0.6  # bottleneck weight; sensitivity range [0.5, 0.75] (config/study3.yaml)
TAU = {1: 0.25, 2: 0.50, 3: 0.75, 4: 1.00}  # certification thresholds (Eq. 10)


def _layer_values(scores: dict[str, float] | list[float]) -> list[float]:
    """Validate and order a {layer: score} dict or 4-element list."""
    values = ([scores[l] for l in LAYERS] if isinstance(scores, dict)
              else list(scores))
    if len(values) != 4:
        raise ValueError(f"expected 4 layer scores, got {len(values)}")
    for v in values:
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"layer score {v} out of range [0,1]")
    return values


def bottleneck_index(scores: dict[str, float] | list[float],
                     alpha: float = DEFAULT_ALPHA) -> float:
    """Bottleneck score M_CASE (Eq. 9): alpha*min(m) + (1-alpha)*mean(m).

    Accepts a {layer: score} dict or a 4-element list; scores in [0,1].
    """
    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"alpha {alpha} out of range [0,1]")
    values = _layer_values(scores)
    return alpha * min(values) + (1.0 - alpha) * (sum(values) / len(values))


def certified_level(scores: dict[str, float] | list[float]) -> int:
    """Certified maturity level (Eq. 10): max k with every layer >= tau_k.

    Returns 0 (L0) when no threshold is met by all four layers. The gate is
    non-compensatory by construction: only the minimum layer matters.
    """
    floor = min(_layer_values(scores))
    level = 0
    for k, tau_k in sorted(TAU.items()):
        if floor >= tau_k:
            level = k
    return level


def geometric_mean_index(scores: dict[str, float] | list[float]) -> float:
    """Geometric-mean composite, retained as the robustness limit.

    (m_L1 * m_L2 * m_L3 * m_L4) ** (1/4). Any zero layer drives it to zero,
    which on the scored corpus zeroes every composite — the paper reports
    this as the limiting non-compensatory case (Sec. 6.2, Study 3).
    """
    values = _layer_values(scores)
    if any(v == 0.0 for v in values):
        return 0.0
    return prod(values) ** (1.0 / 4.0)
