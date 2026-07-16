"""Cached, rate-limited, robots-aware fetch layer for all collectors.

Compliance rules (guide sec 6): public sources only, robots.txt respected
for crawl-type fetching, per-domain rate limit, every payload cached under
data/raw/<source>/ with a tracked manifest.json (url, sha256, size,
accessed date) so the gitignored payloads stay reproducible and citable.
"""

from __future__ import annotations

import hashlib
import json
import time
import urllib.robotparser
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable
from urllib.parse import urlsplit

import requests


class RobotsDisallowed(RuntimeError):
    """The target URL is disallowed for us by the site's robots.txt."""


def _requests_download(url: str, dest: Path, user_agent: str, timeout: int) -> None:
    with requests.get(
        url, headers={"User-Agent": user_agent}, stream=True, timeout=timeout
    ) as resp:
        resp.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                f.write(chunk)


def _requests_robots(robots_url: str, user_agent: str, timeout: int) -> str | None:
    try:
        resp = requests.get(
            robots_url, headers={"User-Agent": user_agent}, timeout=timeout
        )
    except requests.RequestException:
        return None
    if resp.status_code >= 400:
        return None  # absent/unreadable robots.txt -> allow
    return resp.text


class Fetcher:
    """Downloads URLs into a cache directory, once.

    Transport, clock, and sleep are injectable for offline tests.
    """

    def __init__(
        self,
        min_interval: float,
        user_agent: str,
        timeout: int = 30,
        download: Callable[[str, Path], None] | None = None,
        robots_fetch: Callable[[str], str | None] | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.min_interval = min_interval
        self.user_agent = user_agent
        self._download = download or (
            lambda url, dest: _requests_download(url, dest, user_agent, timeout)
        )
        self._robots_fetch = robots_fetch or (
            lambda robots_url: _requests_robots(robots_url, user_agent, timeout)
        )
        self._clock = clock
        self._sleep = sleep
        self._last_request: dict[str, float] = {}
        self._robots_cache: dict[str, urllib.robotparser.RobotFileParser | None] = {}

    # ------------------------------------------------------------- policy

    def _robots_allows(self, url: str) -> bool:
        domain = urlsplit(url).netloc
        if domain not in self._robots_cache:
            robots_url = f"https://{domain}/robots.txt"
            text = self._robots_fetch(robots_url)
            if text is None:
                self._robots_cache[domain] = None
            else:
                parser = urllib.robotparser.RobotFileParser()
                parser.parse(text.splitlines())
                self._robots_cache[domain] = parser
        parser = self._robots_cache[domain]
        return True if parser is None else parser.can_fetch(self.user_agent, url)

    def _rate_limit(self, url: str) -> None:
        domain = urlsplit(url).netloc
        now = self._clock()
        last = self._last_request.get(domain)
        if last is not None:
            wait = self.min_interval - (now - last)
            if wait > 0:
                self._sleep(wait)
                now = last + self.min_interval
        self._last_request[domain] = now

    # -------------------------------------------------------------- fetch

    def fetch(
        self,
        url: str,
        cache_path: Path,
        refresh: bool = False,
        robots_exempt: bool = False,
    ) -> Path:
        """Return the cached file for url, downloading it if needed.

        robots_exempt is reserved for author-published explicit dataset
        links (flagged in config/sources.yaml); everything else goes
        through the robots.txt check.
        """
        cache_path = Path(cache_path)
        if cache_path.exists() and not refresh and not self._url_changed(
            url, cache_path
        ):
            return cache_path

        if not robots_exempt and not self._robots_allows(url):
            raise RobotsDisallowed(f"robots.txt disallows {url}")

        self._rate_limit(url)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache_path.with_suffix(cache_path.suffix + ".part")
        self._download(url, tmp)
        tmp.replace(cache_path)
        self._write_manifest_entry(url, cache_path)
        return cache_path

    def _url_changed(self, url: str, cache_path: Path) -> bool:
        """A cached file fetched from a different URL is stale (the config
        pointed this cache slot somewhere new)."""
        manifest_path = cache_path.parent / "manifest.json"
        if not manifest_path.exists():
            return False
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        entry = manifest.get(cache_path.name)
        return bool(entry) and entry.get("url") != url

    def _write_manifest_entry(self, url: str, cache_path: Path) -> None:
        manifest_path = cache_path.parent / "manifest.json"
        manifest = {}
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        digest = hashlib.sha256(cache_path.read_bytes()).hexdigest()
        manifest[cache_path.name] = {
            "url": url,
            "sha256": digest,
            "bytes": cache_path.stat().st_size,
            "accessed_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        }
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )


def fetcher_from_config(defaults: dict) -> Fetcher:
    return Fetcher(
        min_interval=float(defaults["min_seconds_between_requests"]),
        user_agent=str(defaults["user_agent"]).strip(),
        timeout=int(defaults.get("timeout_seconds", 30)),
    )
