"""Tests for src/collectors/fetch.py: caching, manifest, rate limit, robots."""

import json

import pytest

from src.collectors.fetch import Fetcher, RobotsDisallowed


def make_fetcher(tmp_path, downloads, *, robots_txt=None, clock=None, sleeps=None):
    """Fetcher with injected transport and clock; no real network or sleep."""

    def download(url, dest):
        dest.write_bytes(downloads[url])

    def robots_fetch(robots_url):
        return robots_txt  # None -> treated as absent (allow all)

    times = clock if clock is not None else iter(range(1000))

    return Fetcher(
        min_interval=2.0,
        user_agent="test-agent/0.1",
        download=download,
        robots_fetch=robots_fetch,
        clock=lambda: next(times),
        sleep=(sleeps.append if sleeps is not None else lambda s: None),
    )


class TestCacheAndManifest:
    def test_downloads_once_then_serves_from_cache(self, tmp_path):
        url = "https://example.org/data.csv"
        calls = []
        downloads = {url: b"a,b\n1,2\n"}

        def counting_download(u, dest):
            calls.append(u)
            dest.write_bytes(downloads[u])

        f = Fetcher(
            min_interval=0,
            user_agent="t",
            download=counting_download,
            robots_fetch=lambda u: None,
            clock=lambda: 0.0,
            sleep=lambda s: None,
        )
        p1 = f.fetch(url, tmp_path / "src" / "data.csv")
        p2 = f.fetch(url, tmp_path / "src" / "data.csv")
        assert p1 == p2
        assert p1.read_bytes() == b"a,b\n1,2\n"
        assert calls == [url], "second fetch must hit the cache, not the network"

    def test_url_change_invalidates_cache(self, tmp_path):
        """Same cache slot, new URL in config -> stale file is refetched."""
        calls = []

        def counting_download(u, dest):
            calls.append(u)
            dest.write_bytes(u.encode())

        f = Fetcher(
            min_interval=0,
            user_agent="t",
            download=counting_download,
            robots_fetch=lambda u: None,
            clock=lambda: 0.0,
            sleep=lambda s: None,
        )
        dest = tmp_path / "page.html"
        f.fetch("https://old.example/a", dest)
        p = f.fetch("https://new.example/b", dest)
        assert calls == ["https://old.example/a", "https://new.example/b"]
        assert p.read_bytes() == b"https://new.example/b"

    def test_refresh_forces_redownload(self, tmp_path):
        url = "https://example.org/data.csv"
        calls = []

        def counting_download(u, dest):
            calls.append(u)
            dest.write_bytes(b"x")

        f = Fetcher(
            min_interval=0,
            user_agent="t",
            download=counting_download,
            robots_fetch=lambda u: None,
            clock=lambda: 0.0,
            sleep=lambda s: None,
        )
        f.fetch(url, tmp_path / "d.bin")
        f.fetch(url, tmp_path / "d.bin", refresh=True)
        assert len(calls) == 2

    def test_manifest_records_url_sha256_size_and_date(self, tmp_path):
        url = "https://example.org/data.csv"
        f = make_fetcher(tmp_path, {url: b"payload"})
        dest = tmp_path / "aiid" / "data.csv"
        f.fetch(url, dest)
        manifest = json.loads((tmp_path / "aiid" / "manifest.json").read_text())
        entry = manifest["data.csv"]
        assert entry["url"] == url
        assert entry["bytes"] == len(b"payload")
        # sha256 of b"payload"
        assert entry["sha256"] == (
            "239f59ed55e737c77147cf55ad0c1b030b6d7ee748a7426952f9b852d5a935e5"
        )
        assert entry["accessed_utc"]


class TestRateLimit:
    def test_second_request_same_domain_waits(self, tmp_path):
        urls = {
            "https://example.org/a": b"a",
            "https://example.org/b": b"b",
        }
        sleeps = []
        # clock: t=0 at first request, t=0.5 at second -> must sleep ~1.5
        f = make_fetcher(tmp_path, urls, clock=iter([0.0, 0.5]), sleeps=sleeps)
        f.fetch("https://example.org/a", tmp_path / "a")
        f.fetch("https://example.org/b", tmp_path / "b")
        assert sleeps and abs(sleeps[0] - 1.5) < 1e-9

    def test_different_domains_do_not_wait(self, tmp_path):
        urls = {
            "https://one.example/a": b"a",
            "https://two.example/b": b"b",
        }
        sleeps = []
        f = make_fetcher(tmp_path, urls, clock=iter([0.0, 0.1]), sleeps=sleeps)
        f.fetch("https://one.example/a", tmp_path / "a")
        f.fetch("https://two.example/b", tmp_path / "b")
        assert sleeps == []


class TestRobots:
    ROBOTS = "User-agent: *\nDisallow: /private/\n"

    def test_disallowed_path_raises(self, tmp_path):
        url = "https://example.org/private/x.html"
        f = make_fetcher(tmp_path, {url: b"x"}, robots_txt=self.ROBOTS)
        with pytest.raises(RobotsDisallowed):
            f.fetch(url, tmp_path / "x.html")

    def test_allowed_path_fetches(self, tmp_path):
        url = "https://example.org/public/x.html"
        f = make_fetcher(tmp_path, {url: b"x"}, robots_txt=self.ROBOTS)
        assert f.fetch(url, tmp_path / "x.html").read_bytes() == b"x"

    def test_missing_robots_txt_allows(self, tmp_path):
        url = "https://example.org/anything"
        f = make_fetcher(tmp_path, {url: b"x"}, robots_txt=None)
        assert f.fetch(url, tmp_path / "y").read_bytes() == b"x"

    def test_robots_exempt_skips_check(self, tmp_path):
        """Explicit author-published dataset links bypass the crawl check."""
        url = "https://example.org/private/dataset.xlsx"
        f = make_fetcher(tmp_path, {url: b"x"}, robots_txt=self.ROBOTS)
        p = f.fetch(url, tmp_path / "d.xlsx", robots_exempt=True)
        assert p.read_bytes() == b"x"
