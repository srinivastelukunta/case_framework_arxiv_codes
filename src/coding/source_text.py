"""Full-text resolution for coding.

The screening snippet (300 chars) exists for pool auditability only;
coding requires mechanism-level detail, so each candidate's full text is
resolved from the raw cache at coding time:

- aiid:        incident description + full text of the first linked
               report(s) from the snapshot's reports table
- web_reports: extracted text of the cached page
- arxiv:       full abstract from the cached Atom responses
- mit:         full Description cell from the cached workbook

Everything comes from data/raw/ — no network at coding time.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

MAX_CHARS = 6000
AIID_REPORTS_PER_INCIDENT = 2


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


class SourceTextResolver:
    def __init__(self, raw_root: Path):
        self.raw_root = Path(raw_root)
        self._aiid_incidents: pd.DataFrame | None = None
        self._aiid_reports: dict[str, str] | None = None
        self._arxiv_abstracts: dict[str, str] | None = None
        self._mit_desc: dict[str, str] | None = None

    # ---------------------------------------------------------------- aiid

    def _load_aiid(self) -> None:
        if self._aiid_incidents is not None:
            return
        self._aiid_incidents = pd.read_csv(
            self.raw_root / "aiid" / "incidents.csv", dtype=str
        ).fillna("")
        reports_csv = self.raw_root / "aiid" / "reports.csv"
        self._aiid_reports = {}
        if reports_csv.exists():
            rep = pd.read_csv(
                reports_csv,
                dtype=str,
                usecols=lambda c: c in ("report_number", "text"),
            ).fillna("")
            self._aiid_reports = dict(zip(rep.report_number, rep.text))

    def _aiid_text(self, incident_id: str) -> str:
        self._load_aiid()
        rows = self._aiid_incidents[
            self._aiid_incidents.incident_id == str(incident_id)
        ]
        if rows.empty:
            return ""
        row = rows.iloc[0]
        parts = [_clean(row.get("description", ""))]
        report_ids = re.findall(r"\d+", str(row.get("reports", "")))
        for rid in report_ids[:AIID_REPORTS_PER_INCIDENT]:
            text = _clean(self._aiid_reports.get(rid, ""))
            if text:
                parts.append(f"[linked report {rid}] {text}")
        return " ".join(parts)

    # ----------------------------------------------------------------- web

    def _web_text(self, item_id: str) -> str:
        from src.collectors.web_reports import extract_page_text

        cache_file = self.raw_root / "web_reports" / f"{item_id}.html"
        if not cache_file.exists():
            return ""
        html = cache_file.read_text(encoding="utf-8", errors="replace")
        _, text = extract_page_text(html)
        return _clean(text)

    # --------------------------------------------------------------- arxiv

    def _load_arxiv(self) -> None:
        if self._arxiv_abstracts is not None:
            return
        from src.collectors.arxiv_papers import parse_atom_feed

        self._arxiv_abstracts = {}
        arxiv_dir = self.raw_root / "arxiv"
        if arxiv_dir.exists():
            for xml_file in sorted(arxiv_dir.glob("query_*.xml")):
                for entry in parse_atom_feed(
                    xml_file.read_text(encoding="utf-8")
                ):
                    self._arxiv_abstracts.setdefault(
                        entry["arxiv_id"], _clean(entry["summary"])
                    )

    def _arxiv_text(self, arxiv_id: str) -> str:
        self._load_arxiv()
        return self._arxiv_abstracts.get(arxiv_id, "")

    # ----------------------------------------------------------------- mit

    def _load_mit(self) -> None:
        if self._mit_desc is not None:
            return
        from src.collectors.mit_risk_repo import load_risk_database

        self._mit_desc = {}
        workbook = self.raw_root / "mit_risk_repo" / "ai_risk_repository.xlsx"
        if workbook.exists():
            df = load_risk_database(workbook)
            for _, row in df.iterrows():
                ev_id = str(row.get("Ev_ID", "")).strip()
                if ev_id:
                    self._mit_desc[ev_id] = _clean(
                        f"{row.get('Title', '')}. {row.get('Description', '')} "
                        f"{row.get('Additional ev.', '') or row.get('Additional evidence', '')}"
                    )

    def _mit_text(self, ev_id: str) -> str:
        self._load_mit()
        return self._mit_desc.get(ev_id, "")

    # ------------------------------------------------------------- resolve

    def full_text(self, candidate: dict) -> str:
        source, ref = candidate.get("source", ""), str(candidate.get("source_ref", ""))
        if source == "aiid":
            text = self._aiid_text(ref)
        elif source == "web_reports":
            text = self._web_text(ref)
        elif source == "arxiv":
            text = self._arxiv_text(ref)
        elif source == "mit_risk_repo":
            text = self._mit_text(ref)
        else:
            text = ""
        text = text[:MAX_CHARS]
        # never worse than the screening snippet
        return text if len(text) >= len(str(candidate.get("text_snippet", ""))) else str(
            candidate.get("text_snippet", "")
        )


def enrich_candidates(candidates: list[dict], raw_root: Path) -> list[dict]:
    """Return copies of candidates with text_snippet replaced by full text.

    Fails loudly if a material share of candidates could not be enriched
    beyond the 300-char snippet (that was the v1 defect)."""
    resolver = SourceTextResolver(raw_root)
    out = []
    n_thin = 0
    for cand in candidates:
        text = resolver.full_text(cand)
        if len(text) <= 320:
            n_thin += 1
        enriched = dict(cand)
        enriched["text_snippet"] = text
        out.append(enriched)
    if candidates and n_thin / len(candidates) > 0.5:
        raise ValueError(
            f"full-text enrichment failed for {n_thin}/{len(candidates)} "
            "candidates — raw caches missing?"
        )
    return out
