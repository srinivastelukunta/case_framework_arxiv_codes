"""Decision-tree unit tests for src/coding/protocol.py (T2).

Synthetic incidents exercise every branch of the Appendix B protocol:
- inclusion filter: 3 criteria + (system, date) deduplication
- primary code: earliest layer L1..L4 whose correct functioning would have
  interrupted the failure trajectory
- secondary codes: all other contributing layers
"""

from pathlib import Path

import pytest
import yaml

from src.coding.protocol import (
    EXCLUSION_REASONS,
    IncidentFeatures,
    check_inclusion,
    assign_layers,
    code_incident,
    dedup_key,
)

ROOT = Path(__file__).resolve().parents[1]


def make_incident(**overrides) -> IncidentFeatures:
    """A baseline includable incident; tests override single features."""
    base = dict(
        incident_id="inc-000",
        system="Test Agent System",
        date="2025-01-15",
        autonomous_multistep=True,
        mechanism_detail=True,
        independent_reporting=True,
        self_reported_with_specifics=False,
        single_agent_loop_would_prevent=False,
        required_interagent_or_shared_state=False,
        oversight_point_existed=False,
        oversight_deficient=False,
        operational_mechanism=False,
    )
    base.update(overrides)
    return IncidentFeatures(**base)


# ---------------------------------------------------------------- inclusion

class TestInclusionFilter:
    def test_excluded_not_agentic(self):
        """Branch: criterion 1 fails — single-completion output, no agency."""
        inc = make_incident(
            incident_id="inc-001",
            system="Plain Chatbot",
            autonomous_multistep=False,
            single_agent_loop_would_prevent=True,
        )
        decision = check_inclusion(inc, seen_keys=set())
        assert decision.included is False
        assert decision.exclusion_reason == "not_agentic"

    def test_excluded_insufficient_mechanism(self):
        """Branch: criterion 2 fails — no mechanism-level description."""
        inc = make_incident(
            incident_id="inc-002",
            system="Vague News Story Agent",
            mechanism_detail=False,
        )
        decision = check_inclusion(inc, seen_keys=set())
        assert decision.included is False
        assert decision.exclusion_reason == "insufficient_mechanism"

    def test_excluded_unverified_account(self):
        """Branch: criterion 3 fails — neither independent nor self-reported
        with technical specifics."""
        inc = make_incident(
            incident_id="inc-003",
            system="Rumored Trading Agent",
            independent_reporting=False,
            self_reported_with_specifics=False,
            single_agent_loop_would_prevent=True,
        )
        decision = check_inclusion(inc, seen_keys=set())
        assert decision.included is False
        assert decision.exclusion_reason == "unverified_account"

    def test_included_via_self_report_with_specifics(self):
        """Branch: criterion 3 passes on the self-reported path."""
        inc = make_incident(
            incident_id="inc-004",
            system="Vendor Postmortem Agent",
            independent_reporting=False,
            self_reported_with_specifics=True,
            single_agent_loop_would_prevent=True,
        )
        decision = check_inclusion(inc, seen_keys=set())
        assert decision.included is True
        assert decision.exclusion_reason is None

    def test_excluded_duplicate_same_system_and_date(self):
        """Branch: dedup by (system, date) across sources, case/space
        insensitive."""
        first = make_incident(
            incident_id="inc-005a",
            system="Replit Agent",
            date="2025-07-18",
            single_agent_loop_would_prevent=True,
        )
        seen = {dedup_key(first)}
        dup = make_incident(
            incident_id="inc-005b",
            system="  replit  agent ",
            date="2025-07-18",
            single_agent_loop_would_prevent=True,
        )
        decision = check_inclusion(dup, seen_keys=seen)
        assert decision.included is False
        assert decision.exclusion_reason == "duplicate"

    def test_same_system_different_date_not_duplicate(self):
        first = make_incident(
            incident_id="inc-006a", system="Replit Agent", date="2025-07-18",
            single_agent_loop_would_prevent=True,
        )
        seen = {dedup_key(first)}
        other = make_incident(
            incident_id="inc-006b", system="Replit Agent", date="2025-09-02",
            single_agent_loop_would_prevent=True,
        )
        assert check_inclusion(other, seen_keys=seen).included is True

    def test_exclusion_reasons_match_config(self):
        with open(ROOT / "config" / "coding_protocol.yaml", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        assert set(EXCLUSION_REASONS) == set(config["exclusion_reasons"])


# ------------------------------------------------------------- primary code

class TestPrimaryCodeDecisionTree:
    def test_l1_primary_runaway_loop(self):
        """L1 branch: a correct single-agent closed loop (budget caps,
        iteration limits) would have prevented a runaway retry loop."""
        inc = make_incident(
            incident_id="inc-101",
            system="Unbounded Retry Agent",
            single_agent_loop_would_prevent=True,
        )
        result = assign_layers(inc)
        assert result.primary == "L1"
        assert result.secondary == ()

    def test_l2_primary_shared_memory_contamination(self):
        """L2 branch: mechanism required shared state between agents; a
        single-agent loop would not have caught it."""
        inc = make_incident(
            incident_id="inc-102",
            system="Shared-Memory Fleet",
            required_interagent_or_shared_state=True,
        )
        result = assign_layers(inc)
        assert result.primary == "L2"
        assert result.secondary == ()

    def test_l3_primary_approval_fatigue(self):
        """L3 branch: an oversight point existed but the human lacked
        capacity to absorb the variety."""
        inc = make_incident(
            incident_id="inc-103",
            system="Rubber-Stamp Review Pipeline",
            oversight_point_existed=True,
            oversight_deficient=True,
        )
        result = assign_layers(inc)
        assert result.primary == "L3"
        assert result.secondary == ()

    def test_l4_primary_no_rollback(self):
        """L4 branch: operational mechanism — deployment without a rollback
        path."""
        inc = make_incident(
            incident_id="inc-104",
            system="No-Rollback Deployer",
            operational_mechanism=True,
        )
        result = assign_layers(inc)
        assert result.primary == "L4"
        assert result.secondary == ()

    def test_earliest_layer_wins_l1_over_l2_l4(self):
        """Multi-layer trajectory: L1 fix would have interrupted earliest,
        so L1 is primary; L2 and L4 contributions become secondaries."""
        inc = make_incident(
            incident_id="inc-105",
            system="Cascading Cost Overrun Fleet",
            single_agent_loop_would_prevent=True,
            required_interagent_or_shared_state=True,
            operational_mechanism=True,
        )
        result = assign_layers(inc)
        assert result.primary == "L1"
        assert result.secondary == ("L2", "L4")

    def test_earliest_layer_wins_l2_over_l3(self):
        inc = make_incident(
            incident_id="inc-106",
            system="Convention Lock-In Swarm",
            required_interagent_or_shared_state=True,
            oversight_point_existed=True,
            oversight_deficient=True,
        )
        result = assign_layers(inc)
        assert result.primary == "L2"
        assert result.secondary == ("L3",)

    def test_oversight_existed_but_functioned_is_not_l3(self):
        """L3 contributes only when the oversight point was deficient; an
        oversight point that worked does not code L3."""
        inc = make_incident(
            incident_id="inc-107",
            system="Well-Overseen But Unmonitored Deployer",
            oversight_point_existed=True,
            oversight_deficient=False,
            operational_mechanism=True,
        )
        result = assign_layers(inc)
        assert result.primary == "L4"
        assert result.secondary == ()

    def test_no_contributing_layer_raises(self):
        """An includable incident with no codeable mechanism is a protocol
        violation (mechanism_detail should have failed): fail loudly."""
        inc = make_incident(incident_id="inc-108", system="Uncodeable")
        with pytest.raises(ValueError, match="no contributing layer"):
            assign_layers(inc)


# ------------------------------------------------------------ full pipeline

class TestCodeIncident:
    def test_included_incident_gets_codes_and_registers_key(self):
        seen: set = set()
        inc = make_incident(
            incident_id="inc-201",
            system="Fleet A",
            date="2025-03-01",
            single_agent_loop_would_prevent=True,
            operational_mechanism=True,
        )
        result = code_incident(inc, seen_keys=seen)
        assert result.included is True
        assert result.primary == "L1"
        assert result.secondary == ("L4",)
        assert dedup_key(inc) in seen

    def test_excluded_incident_has_no_codes(self):
        result = code_incident(
            make_incident(incident_id="inc-202", autonomous_multistep=False),
            seen_keys=set(),
        )
        assert result.included is False
        assert result.exclusion_reason == "not_agentic"
        assert result.primary is None
        assert result.secondary == ()

    def test_excluded_incident_does_not_register_dedup_key(self):
        """An excluded candidate must not block a later, better-documented
        account of the same (system, date) from inclusion."""
        seen: set = set()
        rejected = make_incident(
            incident_id="inc-203a",
            system="Fleet B",
            date="2025-04-01",
            mechanism_detail=False,
        )
        code_incident(rejected, seen_keys=seen)
        better = make_incident(
            incident_id="inc-203b",
            system="Fleet B",
            date="2025-04-01",
            single_agent_loop_would_prevent=True,
        )
        result = code_incident(better, seen_keys=seen)
        assert result.included is True
