"""MIT AI Risk Repository collector.

Downloads the public Google Sheet (CC BY 4.0, published for copying at
airisk.mit.edu) as xlsx and screens the risk-database rows into Study 1
candidates. Rows here are risk excerpts extracted from source documents;
the coding stage (T5) decides whether an excerpt documents a concrete
codeable incident.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from src.collectors.fetch import Fetcher
from src.collectors.screen import screen_row


def download_workbook(fetcher: Fetcher, cfg: dict, cache_root: Path) -> Path:
    dest = cache_root / cfg["cache_subdir"] / "ai_risk_repository.xlsx"
    return fetcher.fetch(
        cfg["export_url"],
        dest,
        robots_exempt=bool(cfg.get("explicit_dataset_link", False)),
    )


def load_risk_database(workbook_path: Path) -> pd.DataFrame:
    """Locate the risk-database sheet and normalize its header row.

    The workbook has intro/tabulation sheets; the database sheet is the one
    whose header contains Title and Description columns (header may not be
    on the first row).
    """
    book = pd.read_excel(workbook_path, sheet_name=None, header=None, dtype=str)
    for name, raw in book.items():
        if "database" not in name.lower():
            continue
        for header_idx in range(min(5, len(raw))):
            header = [str(v).strip() for v in raw.iloc[header_idx]]
            if "Title" in header and "Description" in header:
                df = raw.iloc[header_idx + 1 :].copy()
                df.columns = header
                return df.fillna("")
    raise ValueError(
        f"could not locate a risk-database sheet with Title/Description "
        f"columns in {workbook_path.name}; sheets: {list(book)}"
    )


def _row_year(rec: pd.Series) -> str:
    for col in ("Date", "Year", "QuickRef"):
        if col in rec.index:
            m = re.search(r"(19|20)\d{2}", str(rec[col]))
            if m:
                return m.group(0)
    return ""


def screen_mit_rows(
    df: pd.DataFrame, min_year: int, terms: tuple[str, ...]
) -> list[dict]:
    rows = []
    for idx, rec in df.iterrows():
        ev_id = str(rec.get("Ev_ID", "") or f"row{idx}").strip() or f"row{idx}"
        year = _row_year(rec)
        rows.append(
            screen_row(
                candidate_id=f"mit-{ev_id}",
                source="mit_risk_repo",
                source_ref=ev_id,
                url="https://airisk.mit.edu/risks",
                date=year,
                title=str(rec.get("Title", "")),
                text=str(rec.get("Description", "")),
                terms=terms,
                min_date=str(min_year),
            )
        )
    return rows


def collect(
    fetcher: Fetcher, cfg: dict, cache_root: Path, terms: tuple[str, ...]
) -> list[dict]:
    workbook = download_workbook(fetcher, cfg, cache_root)
    df = load_risk_database(workbook)
    return screen_mit_rows(df, min_year=2023, terms=terms)
