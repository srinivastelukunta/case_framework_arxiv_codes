"""Curated postmortem / field-study collector for Study 1.

Items live in config/sources.yaml under web_reports.items with a status
lifecycle: seed | proposed | approved | rejected. Rejected items are
skipped; everything else is fetched (robots-aware), cached, and screened.
The author must approve proposed items before T5 coding uses them.

If a page cannot be fetched (paywall interstitial, 403, robots), the
candidate row is still built from the curated metadata so the evidence URL
is preserved; the failure is reported to the caller.
"""

from __future__ import annotations

from pathlib import Path

from bs4 import BeautifulSoup

from src.collectors.fetch import Fetcher
from src.collectors.screen import screen_row

ACTIVE_STATUSES = ("seed", "proposed", "approved")


def extract_page_text(html: str) -> tuple[str, str]:
    """(title, main text) with nav/script/style/header/footer stripped."""
    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.get_text(strip=True) if soup.title else ""
    for tag in soup(["script", "style", "nav", "header", "footer", "aside"]):
        tag.decompose()
    body = soup.body or soup
    text = " ".join(body.get_text(separator=" ").split())
    return title, text


def screen_web_items(
    items: list[dict],
    page_texts: dict[str, str],
    min_date: str,
    terms: tuple[str, ...],
) -> list[dict]:
    rows = []
    for item in items:
        status = item.get("status", "proposed")
        if status == "rejected":
            continue
        if not item.get("url"):
            raise ValueError(f"web_reports item {item.get('id')!r} has no url")
        html = page_texts.get(item["id"])
        if html:
            page_title, text = extract_page_text(html)
            title = item.get("title") or page_title
        else:
            title, text = item.get("title", ""), ""
        rows.append(
            screen_row(
                candidate_id=f"web-{item['id']}",
                source="web_reports",
                source_ref=item["id"],
                url=item["url"],
                date=str(item.get("date", "")),
                title=title,
                text=text or title,
                terms=terms,
                min_date=min_date,
            )
        )
    return rows


def collect(
    fetcher: Fetcher,
    cfg: dict,
    cache_root: Path,
    terms: tuple[str, ...],
    refresh: bool = False,
) -> tuple[list[dict], list[str]]:
    """Returns (candidate rows, ids of items whose page fetch failed)."""
    items = [
        i for i in cfg.get("items", []) if i.get("status") in ACTIVE_STATUSES
    ]
    page_texts: dict[str, str] = {}
    failures: list[str] = []
    for item in items:
        cache_file = cache_root / cfg["cache_subdir"] / f"{item['id']}.html"
        try:
            path = fetcher.fetch(item["url"], cache_file, refresh=refresh)
            page_texts[item["id"]] = path.read_text(
                encoding="utf-8", errors="replace"
            )
        except Exception as exc:  # noqa: BLE001 — one bad page must not kill the run
            failures.append(f"{item['id']}: {type(exc).__name__}: {exc}")
    rows = screen_web_items(items, page_texts, "2023-01-01", terms)
    return rows, failures
