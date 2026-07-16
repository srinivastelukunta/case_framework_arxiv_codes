"""Study 3 Figure 1: two-panel CASE maturity figure (grayscale-safe, vector).

Panel A: histogram of the composite index M_CASE across scored deployments,
with bin edges at the maturity levels (L0..L4).
Panel B: per-layer mean maturity bars (L1..L4).

Rendered to a vector PDF with fonts embedded (pdf.fonttype=42), grayscale
fills so it survives black-and-white printing.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.study3.index import (  # noqa: E402
    LAYERS,
    geometric_mean_index,
    maturity_level,
)

# Embed TrueType fonts (type 42) rather than type-3 bitmaps: required for
# arXiv/publication-grade vector PDFs.
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

_BIN_EDGES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
_LEVEL_TICK_LABELS = ["L0", "L1", "L2", "L3", "L4"]


def render_figure1(deployments: list[dict], out_path: Path) -> Path:
    """Write the two-panel figure to out_path (PDF). Returns the path."""
    composites = [geometric_mean_index(d["scores"]) for d in deployments]
    n = len(composites)
    per_layer_mean = [
        sum(d["scores"][l] for d in deployments) / n for l in LAYERS
    ]

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.2, 3.0))

    # Panel A: composite histogram with maturity-level bins
    axA.hist(composites, bins=_BIN_EDGES, color="0.55", edgecolor="black",
             linewidth=0.8)
    axA.set_xlim(0, 1)
    axA.set_xticks([(a + b) / 2 for a, b in zip(_BIN_EDGES[:-1], _BIN_EDGES[1:])])
    axA.set_xticklabels(_LEVEL_TICK_LABELS)
    for edge in _BIN_EDGES[1:-1]:
        axA.axvline(edge, color="0.3", linewidth=0.5, linestyle=":")
    axA.set_xlabel(r"Composite maturity $M_{\mathrm{CASE}}$ (by level)")
    axA.set_ylabel("Deployments")
    axA.set_title(f"(A) Composite maturity distribution ($N={n}$)",
                  fontsize=9)

    # Panel B: per-layer mean bars
    xs = range(len(LAYERS))
    axB.bar(xs, per_layer_mean, color="0.55", edgecolor="black", linewidth=0.8)
    axB.set_xticks(list(xs))
    axB.set_xticklabels(list(LAYERS))
    axB.set_ylim(0, 1)
    axB.set_ylabel("Mean layer maturity")
    axB.set_title("(B) Per-layer mean maturity", fontsize=9)
    for x, v in zip(xs, per_layer_mean):
        axB.text(x, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    return out_path


# expose for tests / callers that only need the level of one composite
__all__ = ["render_figure1", "geometric_mean_index", "maturity_level"]
