"""Study 1 inclusion filter and layer-assignment decision tree.

Deterministic implementation of the Appendix B coding protocol
(config/coding_protocol.yaml). Operates on structured incident features;
extracting those features from incident text is the job of the coding
passes (T5), which then delegate the actual layer decision here so both
coders share one rule base.

Primary code = the earliest CASE layer (L1..L4 order) whose correct
functioning would have interrupted the failure trajectory. All other
contributing layers are secondary codes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

LAYERS = ("L1", "L2", "L3", "L4")

EXCLUSION_REASONS = (
    "not_agentic",
    "insufficient_mechanism",
    "unverified_account",
    "duplicate",
)


@dataclass(frozen=True)
class IncidentFeatures:
    """Structured coding features for one candidate incident."""

    incident_id: str
    system: str
    date: str  # ISO date or YYYY-MM; part of the dedup key

    # Inclusion criteria (Appendix B)
    autonomous_multistep: bool          # planning + tool execution
    mechanism_detail: bool              # described well enough to code
    independent_reporting: bool
    self_reported_with_specifics: bool

    # Layer contributions: would the layer's correct functioning have
    # interrupted the trajectory / did its mechanism contribute?
    single_agent_loop_would_prevent: bool       # L1
    required_interagent_or_shared_state: bool   # L2
    oversight_point_existed: bool               # L3 (with deficiency)
    oversight_deficient: bool                   # lacked capacity/model/authority
    operational_mechanism: bool                 # L4


@dataclass(frozen=True)
class InclusionDecision:
    included: bool
    exclusion_reason: str | None = None


@dataclass(frozen=True)
class LayerAssignment:
    primary: str
    secondary: tuple[str, ...]


@dataclass(frozen=True)
class CodingResult:
    incident_id: str
    included: bool
    exclusion_reason: str | None
    primary: str | None
    secondary: tuple[str, ...]


def dedup_key(inc: IncidentFeatures) -> tuple[str, str]:
    """(system, date) key, whitespace- and case-normalized."""
    system = re.sub(r"\s+", " ", inc.system).strip().lower()
    return (system, inc.date.strip())


def check_inclusion(
    inc: IncidentFeatures, seen_keys: set[tuple[str, str]]
) -> InclusionDecision:
    """Apply the three Appendix B criteria, then deduplication."""
    if not inc.autonomous_multistep:
        return InclusionDecision(False, "not_agentic")
    if not inc.mechanism_detail:
        return InclusionDecision(False, "insufficient_mechanism")
    if not (inc.independent_reporting or inc.self_reported_with_specifics):
        return InclusionDecision(False, "unverified_account")
    if dedup_key(inc) in seen_keys:
        return InclusionDecision(False, "duplicate")
    return InclusionDecision(True)


def _contributing_layers(inc: IncidentFeatures) -> tuple[str, ...]:
    contributions = {
        "L1": inc.single_agent_loop_would_prevent,
        "L2": inc.required_interagent_or_shared_state,
        "L3": inc.oversight_point_existed and inc.oversight_deficient,
        "L4": inc.operational_mechanism,
    }
    return tuple(layer for layer in LAYERS if contributions[layer])


def assign_layers(inc: IncidentFeatures) -> LayerAssignment:
    """Primary = earliest contributing layer; the rest are secondaries."""
    contributing = _contributing_layers(inc)
    if not contributing:
        raise ValueError(
            f"incident {inc.incident_id!r}: no contributing layer; an "
            "includable incident must code to at least one layer "
            "(mechanism_detail should have excluded it otherwise)"
        )
    return LayerAssignment(primary=contributing[0], secondary=contributing[1:])


def code_incident(
    inc: IncidentFeatures, seen_keys: set[tuple[str, str]]
) -> CodingResult:
    """Inclusion + layer assignment; registers the dedup key only for
    included incidents so a rejected candidate cannot shadow a better
    account of the same (system, date)."""
    decision = check_inclusion(inc, seen_keys)
    if not decision.included:
        return CodingResult(
            incident_id=inc.incident_id,
            included=False,
            exclusion_reason=decision.exclusion_reason,
            primary=None,
            secondary=(),
        )
    assignment = assign_layers(inc)
    seen_keys.add(dedup_key(inc))
    return CodingResult(
        incident_id=inc.incident_id,
        included=True,
        exclusion_reason=None,
        primary=assignment.primary,
        secondary=assignment.secondary,
    )
