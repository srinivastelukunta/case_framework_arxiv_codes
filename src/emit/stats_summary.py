"""Headline statistics for results/stats_summary.json.

Every statistic quoted in the paper text must come from this file
(definition of done); study2/study3 sections are added by their emitters.
"""

from __future__ import annotations

import pandas as pd

LAYERS = ("L1", "L2", "L3", "L4")


VERDICTS = ("Full", "Partial", "None")


def study3_stats(deployments: list[dict], alpha: float | None = None) -> dict:
    """Headline statistics for Study 3's maturity distribution.

    deployments: [{"deployment", "scores": {L1..L4}}]. The two-part
    instrument (Sec. 6.2): bottleneck score M_CASE (Eq. 9) and certified
    level via the tau-gate (Eq. 10); the geometric-mean composite is
    reported only as the robustness limit. The paper also cites the modal
    weakest layer and per-layer means.
    """
    from statistics import median

    from src.study3.index import (
        DEFAULT_ALPHA,
        LAYERS,
        LEVEL_NAMES,
        bottleneck_index,
        certified_level,
        geometric_mean_index,
    )

    if alpha is None:
        alpha = DEFAULT_ALPHA
    n = len(deployments)
    if n == 0:
        raise ValueError("study3 stats: no deployments")

    scores = [bottleneck_index(d["scores"], alpha) for d in deployments]
    levels = [certified_level(d["scores"]) for d in deployments]
    geometric = [geometric_mean_index(d["scores"]) for d in deployments]
    level_counts = {name: 0 for name in LEVEL_NAMES}
    for lv in levels:
        level_counts[LEVEL_NAMES[lv]] += 1

    # Weakest-layer frequency: every layer tied for a deployment's minimum
    # counts (many deployments have several zero layers).
    weakest_freq = {l: 0 for l in LAYERS}
    for d in deployments:
        lo = min(d["scores"][l] for l in LAYERS)
        for l in LAYERS:
            if d["scores"][l] == lo:
                weakest_freq[l] += 1
    modal_weakest = max(weakest_freq, key=weakest_freq.get)

    per_layer_mean = {
        l: sum(d["scores"][l] for d in deployments) / n for l in LAYERS
    }

    return {
        "n_deployments": n,
        "per_layer_mean": per_layer_mean,
        "modal_weakest_layer": modal_weakest,
        "weakest_layer_frequency": weakest_freq,
        "index": {
            "alpha": alpha,
            "bottleneck_mean": sum(scores) / n,
            "bottleneck_median": median(scores),
            "bottleneck_min": min(scores),
            "bottleneck_max": max(scores),
            "scores": scores,
            "certified_level_distribution": level_counts,
            "geometric_robustness_limit": {
                "all_composites_zero": all(g == 0.0 for g in geometric),
                "composite_mean": sum(geometric) / n,
            },
        },
    }


def study2_stats(coded_tools: list[dict]) -> dict:
    """Coverage statistics for Study 2's tool-by-layer matrix.

    coded_tools: [{"name", "category", "cells": {layer: {"verdict", ...}}}].
    Reports per-layer Full/Partial/None counts and the Full and Full-or-Partial
    coverage shares that the paper cites for the L1/L4-dense, L2/L3-sparse
    hypothesis.
    """
    n = len(coded_tools)
    if n == 0:
        raise ValueError("study2 stats: no coded tools")
    per_layer = {}
    for layer in LAYERS:
        counts = {v: 0 for v in VERDICTS}
        for t in coded_tools:
            counts[t["cells"][layer]["verdict"]] += 1
        per_layer[layer] = {
            "Full": counts["Full"],
            "Partial": counts["Partial"],
            "None": counts["None"],
            "full_share": counts["Full"] / n,
            "full_or_partial_share": (counts["Full"] + counts["Partial"]) / n,
        }
    # Per-category Full coverage: shows that L3 (oversight) coverage, where it
    # exists, concentrates in dedicated orchestration/HITL tools and is absent
    # from the observability and guardrails tooling that dominates adoption.
    categories = sorted({t.get("category", "") for t in coded_tools})
    by_category = {}
    for cat in categories:
        group = [t for t in coded_tools if t.get("category") == cat]
        by_category[cat] = {
            "n_tools": len(group),
            "full_coverage_share": {
                lyr: sum(
                    1 for t in group if t["cells"][lyr]["verdict"] == "Full"
                ) / len(group)
                for lyr in LAYERS
            },
        }

    return {
        "n_tools": n,
        "per_layer_coverage": per_layer,
        "full_coverage_share": {lyr: per_layer[lyr]["full_share"] for lyr in LAYERS},
        "by_category": by_category,
    }


def study1_stats(incidents: pd.DataFrame, kappa: float) -> dict:
    included = incidents[incidents["included"] == True]  # noqa: E712
    n = len(included)
    if n == 0:
        raise ValueError("study1 stats: no included incidents")

    per_layer = {
        layer: int((included["primary_layer"] == layer).sum()) for layer in LAYERS
    }
    by_source = {
        src: int((included["source"] == src).sum())
        for src in sorted(included["source"].unique())
    } if "source" in included.columns else {}
    cooccurrence = {a: {b: 0 for b in LAYERS} for a in LAYERS}
    n_with_any_secondary = 0
    for _, row in included.iterrows():
        primary = row["primary_layer"]
        secondaries = [
            s.strip() for s in str(row.get("secondary_layers", "")).split(";")
            if s.strip() in LAYERS
        ]
        if secondaries:
            n_with_any_secondary += 1
        for sec in secondaries:
            cooccurrence[primary][sec] += 1

    # Coupling / co-occurrence framing (Study 1 headline as of 2026-07-16):
    # the primary distribution is L1-heavy, but most L1-primary incidents also
    # implicate oversight (L3) and engineering-ops (L4) mechanisms as secondary
    # codes. These derived fields are what the paper text cites, so they stay
    # reproducible from this file (every quoted statistic lives here).
    def _share(count: int, of: int) -> float:
        return count / of if of else 0.0

    n_l1 = per_layer["L1"]
    coupling = {
        "n_with_any_secondary": n_with_any_secondary,
        "share_with_any_secondary": _share(n_with_any_secondary, n),
        "l1_primary_count": n_l1,
        "l1_primary_with_l3_secondary": cooccurrence["L1"]["L3"],
        "l1_primary_with_l3_secondary_share": _share(cooccurrence["L1"]["L3"], n_l1),
        "l1_primary_with_l4_secondary": cooccurrence["L1"]["L4"],
        "l1_primary_with_l4_secondary_share": _share(cooccurrence["L1"]["L4"], n_l1),
    }

    return {
        "n_screened": int(len(incidents)),
        "n_included": n,
        "included_by_source": by_source,
        "per_layer_counts": per_layer,
        "per_layer_shares": {k: v / n for k, v in per_layer.items()},
        "l2_l3_share": (per_layer["L2"] + per_layer["L3"]) / n,
        "cohens_kappa": kappa,
        "cooccurrence": cooccurrence,
        "coupling": coupling,
    }
