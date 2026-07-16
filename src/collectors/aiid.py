"""AI Incident Database collector.

Downloads the pinned weekly snapshot (tar.bz2 containing CSV/JSON dumps),
extracts the incidents table, and screens rows into Study 1 candidates.
Snapshot URL and date are pinned in config/sources.yaml for
reproducibility.
"""

from __future__ import annotations

import tarfile
from pathlib import Path

import pandas as pd

from src.collectors.fetch import Fetcher
from src.collectors.screen import screen_row


def download_snapshot(fetcher: Fetcher, cfg: dict, cache_root: Path) -> Path:
    url = cfg["snapshot_url"]
    dest = cache_root / cfg["cache_subdir"] / url.rsplit("/", 1)[-1]
    return fetcher.fetch(url, dest)


def extract_incidents_table(snapshot_path: Path, dest_dir: Path) -> Path:
    """Pull the incidents CSV out of the snapshot archive (selective
    extraction; the archive holds full MongoDB dumps we don't need)."""
    dest = Path(dest_dir) / "incidents.csv"
    if dest.exists():
        return dest
    with tarfile.open(snapshot_path, "r:bz2") as tar:
        member = next(
            (
                m
                for m in tar
                if m.name.lower().endswith("incidents.csv")
            ),
            None,
        )
        if member is None:
            raise FileNotFoundError(
                f"no incidents.csv member found in {snapshot_path.name}"
            )
        fileobj = tar.extractfile(member)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(fileobj.read())
    return dest


def load_incidents(incidents_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(incidents_csv, dtype=str).fillna("")
    required = {"incident_id", "date", "title", "description"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"AIID incidents table missing columns: {sorted(missing)}")
    return df


def screen_aiid_incidents(
    df: pd.DataFrame,
    min_date: str,
    terms: tuple[str, ...],
    url_template: str,
    force_include_ids: frozenset[str] = frozenset(),
) -> list[dict]:
    """Screen AIID incidents on date + agent terms.

    force_include_ids are author-vetted incident_ids added to overcome the
    coarse keyword screen (targeted L2/L3 additions). They bypass ONLY the
    term gate, never the date gate — a forced id before min_date stays
    excluded (no-fabrication: the date scope is not negotiable).
    """
    rows = []
    for _, rec in df.iterrows():
        incident_id = str(rec["incident_id"])
        row = screen_row(
            candidate_id=f"aiid-{incident_id}",
            source="aiid",
            source_ref=incident_id,
            url=url_template.format(incident_id=incident_id),
            date=str(rec["date"]),
            title=str(rec["title"]),
            text=str(rec["description"]),
            terms=terms,
            min_date=min_date,
        )
        if (
            incident_id in force_include_ids
            and not row["screen_included"]
            and row["screen_exclusion_reason"] == "no_agent_terms"
        ):
            row["screen_included"] = True
            row["screen_exclusion_reason"] = ""
            row["matched_terms"] = "curated_targeted"
        rows.append(row)
    return rows


def collect(fetcher: Fetcher, cfg: dict, cache_root: Path) -> list[dict]:
    snapshot = download_snapshot(fetcher, cfg, cache_root)
    incidents_csv = extract_incidents_table(
        snapshot, cache_root / cfg["cache_subdir"]
    )
    df = load_incidents(incidents_csv)
    return screen_aiid_incidents(
        df,
        min_date=cfg["min_date"],
        terms=tuple(cfg["filter_terms"]),
        url_template=cfg["incident_url_template"],
        force_include_ids=frozenset(str(i) for i in cfg.get("force_include_ids", [])),
    )
