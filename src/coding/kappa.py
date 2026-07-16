"""Cohen's kappa on primary codes, computed before adjudication.

Labels are the five-way category {L1, L2, L3, L4, excluded} per candidate,
aligned across the two coding passes by candidate_id.
"""

from __future__ import annotations

from sklearn.metrics import cohen_kappa_score


def cohens_kappa(labels_a: list[str], labels_b: list[str]) -> float:
    if len(labels_a) != len(labels_b):
        raise ValueError(
            f"label lists differ in length: {len(labels_a)} vs {len(labels_b)}"
        )
    if not labels_a:
        raise ValueError("cannot compute kappa on empty label lists")
    return float(cohen_kappa_score(labels_a, labels_b))
