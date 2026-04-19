"""Crawl depth semantics with mocked HTTP — no real network requests."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from core.crawler import CrawlerJob
from core.index_store import IndexStore


def _isolated_store(tmp: Path) -> IndexStore:
    storage_dir = tmp / "data" / "storage"
    storage_dir.mkdir(parents=True, exist_ok=True)
    storage_file = storage_dir / "p.data"
    with patch("core.index_store.STORAGE_DIR", storage_dir), patch(
        "core.index_store.STORAGE_FILE", storage_file
    ):
        return IndexStore()


def _fast_queue_get(queue_obj):
    """Shrink Queue.get timeout so idle worker exit does not wait ~5s."""
    orig = queue_obj.get

    def fast_get(block=True, timeout=None):
        t = min(timeout, 0.02) if timeout is not None else None
        return orig(block=block, timeout=t)

    queue_obj.get = fast_get  # type: ignore[method-assign]


class TestCrawlerDepthSemantics(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.store = _isolated_store(Path(self._tmp.name))

    def tearDown(self):
        self._tmp.cleanup()

    def _run_job(self, job: CrawlerJob, fetch_map: dict[str, str]):
        calls: list[str] = []

        def fake_fetch(url: str) -> str | None:
            calls.append(url)
            return fetch_map.get(url)

        _fast_queue_get(job._url_queue)

        with patch.object(job, "_fetch", side_effect=fake_fetch), patch.object(
            job, "_wait_rate_limit"
        ):
            job.run()

        return calls

    def test_max_depth_zero_fetches_only_origin(self):
        origin = "https://depth0.example/"
        child = "https://other.example/child"
        html_origin = f'<html><body><a href="{child}">go</a><p>Hi</p></body></html>'
        fetch_map = {origin: html_origin}

        job = CrawlerJob(
            crawler_id="d0",
            origin=origin,
            max_depth=0,
            index_store=self.store,
            num_workers=1,
            queue_size=50,
            max_rate=0,
        )
        calls = self._run_job(job, fetch_map)
        self.assertEqual(calls, [origin])
        self.assertNotIn(child, calls)

    def test_max_depth_one_fetches_origin_and_direct_links_only(self):
        origin = "https://tier.example/"
        link_a = "https://tier.example/a"
        link_b = "https://tier.example/b"
        deep = "https://elsewhere.example/deep"

        html_origin = (
            f'<html><body>'
            f'<a href="{link_a}">A</a><a href="{link_b}">B</a>'
            f'<p>text here</p></body></html>'
        )
        html_a = (
            f'<html><body><a href="{deep}">Deeper</a><p>aa words</p></body></html>'
        )
        html_b = "<html><body><p>bb words</p></body></html>"

        fetch_map = {
            origin: html_origin,
            link_a: html_a,
            link_b: html_b,
        }

        job = CrawlerJob(
            crawler_id="d1",
            origin=origin,
            max_depth=1,
            index_store=self.store,
            num_workers=2,
            queue_size=50,
            max_rate=0,
        )
        calls = self._run_job(job, fetch_map)
        self.assertIn(origin, calls)
        self.assertIn(link_a, calls)
        self.assertIn(link_b, calls)
        self.assertNotIn(deep, calls)
        self.assertEqual(len(calls), 3)


if __name__ == "__main__":
    unittest.main()
