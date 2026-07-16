"""Study 3 Figure 1: two-panel CASE maturity figure (grayscale-safe, vector).

Panel A: histogram of the bottleneck score M_CASE (Eq. 9) across scored
deployments, with a dashed mean line and a certification-gate annotation.
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
    DEFAULT_ALPHA,
    LAYERS,
    bottleneck_index,
    certified_level,
    geometric_mean_index,
)

# Embed TrueType fonts (type 42) rather than type-3 bitmaps: required for
# arXiv/publication-grade vector PDFs.
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

_BIN_WIDTH = 0.025
_BIN_EDGES = [i * _BIN_WIDTH for i in range(8)]  # [0, 0.175]


def _gate_annotation(deployments: list[dict], levels: list[int],
                     geometric_all_zero: bool) -> str:
    """Certification summary for Panel A, derived from the data only."""
    n = len(levels)
    n_l0 = levels.count(0)
    if n_l0 == n:
        # Layers at zero in every deployment gate certification outright.
        zero_gates = [l for l in LAYERS
                      if all(d["scores"][l] == 0.0 for d in deployments)]
        gate = f" (gated by $m_{{{zero_gates[0]}}}=0$)" if zero_gates else ""
        first = f"All {n} certify L0{gate}"
    else:
        first = f"{n_l0} of {n} certify L0"
    if geometric_all_zero:
        return first + "\ngeometric limit: all composites 0"
    return first


def render_figure1(deployments: list[dict], out_path: Path,
                   alpha: float = DEFAULT_ALPHA) -> Path:
    """Write the two-panel figure to out_path (PDF). Returns the path."""
    scores = [bottleneck_index(d["scores"], alpha) for d in deployments]
    levels = [certified_level(d["scores"]) for d in deployments]
    geometric_all_zero = all(
        geometric_mean_index(d["scores"]) == 0.0 for d in deployments
    )
    n = len(scores)
    mean_score = sum(scores) / n
    per_layer_mean = [
        sum(d["scores"][l] for d in deployments) / n for l in LAYERS
    ]

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.2, 3.0))

    # Panel A: bottleneck-score histogram with dashed mean line
    counts, _, _ = axA.hist(scores, bins=_BIN_EDGES, color="0.55",
                            edgecolor="black", linewidth=0.8)
    axA.set_xlim(_BIN_EDGES[0], _BIN_EDGES[-1])
    axA.set_xticks([0.0, 0.05, 0.10, 0.15])
    axA.set_ylim(0, max(counts) * 1.35)  # headroom for the annotation
    axA.axvline(mean_score, ymax=0.82, color="black", linewidth=1.0,
                linestyle="--")
    axA.text(mean_score - 0.003, 0.02, f"mean {mean_score:.3f}",
             transform=axA.get_xaxis_transform(), ha="right", va="bottom",
             rotation=90, fontsize=7)
    axA.annotate(_gate_annotation(deployments, levels, geometric_all_zero),
                 xy=(0.97, 0.97), xycoords="axes fraction",
                 ha="right", va="top", fontsize=7)
    axA.set_xlabel(r"Bottleneck score $M_{\mathrm{CASE}}$")
    axA.set_ylabel("Deployments")
    axA.set_title(f"(A) Bottleneck maturity distribution ($N={n}$)",
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


# expose for tests / callers that only need one deployment's score or level
__all__ = ["render_figure1", "bottleneck_index", "certified_level",
           "geometric_mean_index"]
