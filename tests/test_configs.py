"""T1 scaffold tests: config files exist, parse, and match the paper.

The mechanism-class counts are fixed by Table 3 (tab:mechanisms) of
../paper/main.tex: 9 L1, 8 L2, 9 L3, 10 L4. The coding protocol must encode
the Appendix B inclusion rules and the four-layer decision order.
"""

from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config"

EXPECTED_CLASS_COUNTS = {"L1": 9, "L2": 8, "L3": 9, "L4": 10}


@pytest.fixture(scope="module")
def mechanism_classes():
    with open(CONFIG / "mechanism_classes.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def coding_protocol():
    with open(CONFIG / "coding_protocol.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def sources():
    with open(CONFIG / "sources.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


class TestMechanismClasses:
    def test_all_four_layers_present(self, mechanism_classes):
        assert set(mechanism_classes["layers"].keys()) == {"L1", "L2", "L3", "L4"}

    @pytest.mark.parametrize("layer,count", sorted(EXPECTED_CLASS_COUNTS.items()))
    def test_class_counts_match_paper_table3(self, mechanism_classes, layer, count):
        classes = mechanism_classes["layers"][layer]["classes"]
        assert len(classes) == count, (
            f"{layer} must have exactly {count} mechanism classes per Table 3"
        )

    def test_every_class_has_required_fields(self, mechanism_classes):
        for layer in mechanism_classes["layers"].values():
            for cls in layer["classes"]:
                assert cls.get("id"), f"class missing id: {cls}"
                assert cls.get("mechanism"), f"{cls['id']} missing mechanism"
                assert cls.get("classical_construct"), (
                    f"{cls['id']} missing classical_construct"
                )

    def test_class_ids_unique(self, mechanism_classes):
        ids = [
            cls["id"]
            for layer in mechanism_classes["layers"].values()
            for cls in layer["classes"]
        ]
        assert len(ids) == len(set(ids))


class TestCodingProtocol:
    def test_three_inclusion_criteria(self, coding_protocol):
        assert len(coding_protocol["inclusion_criteria"]) == 3

    def test_decision_order_is_l1_to_l4(self, coding_protocol):
        order = [rule["layer"] for rule in coding_protocol["primary_code_rules"]]
        assert order == ["L1", "L2", "L3", "L4"], (
            "primary code = earliest layer that would have interrupted the "
            "failure trajectory; rules must be ordered L1..L4"
        )

    def test_dedup_key_is_system_and_date(self, coding_protocol):
        assert sorted(coding_protocol["deduplication"]["key"]) == ["date", "system"]

    def test_secondary_codes_enabled(self, coding_protocol):
        assert coding_protocol["secondary_codes"]["record"] is True


class TestSources:
    def test_default_rate_limit_at_least_2s(self, sources):
        assert sources["defaults"]["min_seconds_between_requests"] >= 2

    def test_robots_txt_respected(self, sources):
        assert sources["defaults"]["respect_robots_txt"] is True

    def test_required_source_groups_present(self, sources):
        for key in ("aiid", "mit_risk_repo", "arxiv", "web_reports"):
            assert key in sources["sources"], f"missing source group: {key}"


class TestRepoLayout:
    @pytest.mark.parametrize(
        "relpath",
        [
            "Makefile",
            "requirements.txt",
            "README.md",
            "LICENSE",
            "DATA_LICENSE",
            ".gitignore",
            "src/collectors/__init__.py",
            "src/coding/__init__.py",
            "src/study2/__init__.py",
            "src/study3/__init__.py",
            "src/emit/__init__.py",
        ],
    )
    def test_scaffold_file_exists(self, relpath):
        assert (ROOT / relpath).is_file(), f"missing scaffold file: {relpath}"

    def test_packages_import(self):
        import src.coding
        import src.collectors
        import src.emit
        import src.study2
        import src.study3  # noqa: F401
