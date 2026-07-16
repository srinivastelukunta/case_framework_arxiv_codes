"""LaTeX table emitters.

Emitted files contain ONLY table body rows, designed to be \\input{} inside
the paper's existing tabular skeletons so the paper compiles without column
changes (project rule). No-fabrication guardrail: emitters fail loudly on
empty or undersized inputs — they never emit placeholder numbers.
"""

from __future__ import annotations

from collections import Counter

import pandas as pd

STUDY1_MIN_N = 50
STUDY1_MAX_N = 100

LAYER_LABELS = {
    "L1": "L1 Control",
    "L2": "L2 Adaptive systems",
    "L3": "L3 Supervisory",
    "L4": "L4 Engineering ops",
}

_LATEX_ESCAPES = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


def latex_escape(text: str) -> str:
    return "".join(_LATEX_ESCAPES.get(ch, ch) for ch in str(text))


def _representative_modes(phrases: pd.Series, top: int = 3) -> str:
    counts = Counter(p.strip().lower() for p in phrases if str(p).strip())
    ordered = [phrase for phrase, _ in counts.most_common(top)]
    return "; ".join(latex_escape(p) for p in ordered)


STUDY2_MIN_TOOLS = 15
STUDY2_LAYER_HEADERS = ("L1", "L2", "L3", "L4")


def emit_study2_table(coded_tools: list[dict]) -> str:
    """Body rows + coverage footer for tab:tooling.

    Each coded tool is {"name": str, "cells": {layer: {"verdict": ...}}}.
    Emits one row per tool (Tool & L1 & L2 & L3 & L4 with Full/Partial/None
    cells) followed by a \\midrule and a 'Column coverage (Full)' footer row
    giving the share of tools coded Full for each layer.
    """
    n = len(coded_tools)
    if n < STUDY2_MIN_TOOLS:
        raise ValueError(
            f"study2 emitter: {n} tools is below the minimum of "
            f"{STUDY2_MIN_TOOLS}; refusing to emit (no-fabrication guardrail)"
        )
    rows = []
    for tool in coded_tools:
        cells = tool["cells"]
        verdicts = " & ".join(
            cells[layer]["verdict"] for layer in STUDY2_LAYER_HEADERS
        )
        rows.append(f"{latex_escape(tool['name'])} & {verdicts} \\\\")

    full = {
        layer: sum(
            1 for t in coded_tools if t["cells"][layer]["verdict"] == "Full"
        )
        for layer in STUDY2_LAYER_HEADERS
    }
    footer_cells = " & ".join(
        f"{round(100 * full[layer] / n)}\\%" for layer in STUDY2_LAYER_HEADERS
    )
    footer = f"\\textbf{{Column coverage (Full)}} & {footer_cells} \\\\"
    return "\n".join(rows) + "\n\\midrule\n" + footer + "\n"


def emit_study1_table(incidents: pd.DataFrame) -> str:
    """Body rows for tab:failures: Primary layer & modes & Count & Share."""
    if incidents.empty or "included" not in incidents.columns:
        raise ValueError("study1 emitter: input is empty or malformed")
    included = incidents[incidents["included"] == True]  # noqa: E712
    n = len(included)
    if n < STUDY1_MIN_N:
        raise ValueError(
            f"study1 emitter: N={n} included incidents is below the minimum "
            f"of {STUDY1_MIN_N}; refusing to emit (no-fabrication guardrail)"
        )

    lines = []
    for layer in ("L1", "L2", "L3", "L4"):
        subset = included[included["primary_layer"] == layer]
        count = len(subset)
        share = round(100 * count / n)
        modes = _representative_modes(subset["mechanism_phrase"])
        lines.append(
            f"{LAYER_LABELS[layer]} & {modes} & {count} & {share}\\% \\\\"
        )
    return "\n".join(lines) + "\n"
