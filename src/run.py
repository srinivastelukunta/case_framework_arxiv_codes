"""Pipeline entry points used by the Makefile.

Each target composes the underlying module mains. Table/figure targets read
the cached data under data/ and emit into results/; the coding targets call
the model API only for UNCACHED candidates/tools (set ANTHROPIC_API_KEY; see
README). With the shipped caches in place, every target runs offline.
"""

import sys

TARGETS = ("candidates", "refresh", "study1", "study2", "study3",
           "tables", "figures")


def main() -> int:
    if len(sys.argv) != 2 or sys.argv[1] not in TARGETS:
        print(f"usage: python -m src.run [{'|'.join(TARGETS)}]")
        return 2
    return globals()[f"_{sys.argv[1]}"]()


def _candidates() -> int:
    from src.collectors.build_candidates import main as build
    return build()


def _refresh() -> int:
    from src.collectors.build_candidates import main as build
    return build(refresh=True)


def _study1() -> int:
    from src.coding.run_coding import main as code
    rc = code()
    if rc != 0:
        return rc
    from src.emit.build_study1_outputs import main as emit
    return emit()


def _study2() -> int:
    from src.emit.build_study2_outputs import main as build
    return build()


def _study3() -> int:
    from src.emit.build_study3_outputs import main as build
    return build()


def _tables() -> int:
    from src.emit.build_study1_outputs import main as s1
    from src.emit.build_study2_outputs import main as s2
    return s1() or s2()


def _figures() -> int:
    from src.emit.build_study3_outputs import main as build
    return build()


if __name__ == "__main__":
    sys.exit(main())
