"""Unit tests for IndexStore: insert, search, stats, concurrent writes."""

import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from core.index_store import CrawlerStats, IndexStore


def _isolated_store(tmp: Path) -> IndexStore:
    storage_dir = tmp / "data" / "storage"
    storage_dir.mkdir(parents=True, exist_ok=True)
    storage_file = storage_dir / "p.data"
    with patch("core.index_store.STORAGE_DIR", storage_dir), patch(
        "core.index_store.STORAGE_FILE", storage_file
    ):
        return IndexStore()


class TestIndexStoreAddAndSearch(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.store = _isolated_store(Path(self._tmp.name))

    def tearDown(self):
        self._tmp.cleanup()

    def test_add_page_inserts_postings(self):
        self.store.add_page(
            url="https://a.com/p1",
            origin="https://a.com/",
            depth=0,
            word_freqs={"hello": 2, "world": 1},
            title_words={"hello"},
            crawler_id="c1",
        )
        hello_entries = [e for e in self.store.search(["hello"]) if e.url == "https://a.com/p1"]
        self.assertTrue(hello_entries)
        self.assertTrue(any(e.in_title for e in hello_entries))

    def test_search_exact_word(self):
        self.store.add_page(
            url="https://b.com/x",
            origin="https://b.com/",
            depth=1,
            word_freqs={"alpha": 3},
            title_words=set(),
            crawler_id="c1",
        )
        hits = self.store.search(["alpha"])
        self.assertTrue(any(e.word == "alpha" for e in hits))

    def test_search_prefix_expands(self):
        self.store.add_page(
            url="https://c.com/y",
            origin="https://c.com/",
            depth=0,
            word_freqs={"running": 1},
            title_words=set(),
            crawler_id="c1",
        )
        hits = self.store.search(["run"])
        self.assertTrue(any(e.word == "running" for e in hits))

    def test_total_pages_and_words(self):
        self.assertEqual(self.store.total_pages_indexed(), 0)
        self.assertEqual(self.store.total_words_indexed(), 0)
        self.store.add_page(
            url="https://d.com/z",
            origin="https://d.com/",
            depth=0,
            word_freqs={"one": 1, "two": 1},
            title_words=set(),
            crawler_id="c1",
        )
        self.assertEqual(self.store.total_pages_indexed(), 1)
        self.assertGreaterEqual(self.store.total_words_indexed(), 2)

    def test_register_and_get_stats(self):
        stats = CrawlerStats(
            crawler_id="id1",
            origin="https://o.com/",
            max_depth=2,
            queue_capacity=10,
            total_workers=3,
        )
        self.store.register_crawler(stats)
        got = self.store.get_stats("id1")
        self.assertIsNotNone(got)
        assert got is not None
        self.assertEqual(got.origin, "https://o.com/")
        self.assertEqual(got.max_depth, 2)

    def test_update_stats(self):
        self.store.register_crawler(
            CrawlerStats(
                crawler_id="u1",
                origin="https://u.com/",
                max_depth=1,
            )
        )
        self.store.update_stats("u1", urls_visited=5, backpressure_active=True)
        s = self.store.get_stats("u1")
        self.assertIsNotNone(s)
        assert s is not None
        self.assertEqual(s.urls_visited, 5)
        self.assertTrue(s.backpressure_active)


class TestIndexStoreConcurrency(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.store = _isolated_store(Path(self._tmp.name))

    def tearDown(self):
        self._tmp.cleanup()

    def test_concurrent_add_page(self):
        errors: list[BaseException] = []
        barrier = threading.Barrier(8)

        def worker(i: int):
            try:
                barrier.wait()
                self.store.add_page(
                    url=f"https://conc.example/page{i}",
                    origin="https://conc.example/",
                    depth=0,
                    word_freqs={f"word{i}": 1, "shared": 1},
                    title_words=set(),
                    crawler_id="cx",
                )
            except BaseException as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)
            self.assertFalse(t.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(self.store.total_pages_indexed(), 8)
        shared_hits = self.store.search(["shared"])
        urls = {e.url for e in shared_hits}
        self.assertEqual(len(urls), 8)


class TestIndexStoreThreadSafetyReadDuringWrite(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.store = _isolated_store(Path(self._tmp.name))

    def tearDown(self):
        self._tmp.cleanup()

    def test_search_while_adding(self):
        stop = threading.Event()
        failures: list[BaseException] = []

        def writer():
            i = 0
            try:
                while not stop.is_set() and i < 200:
                    self.store.add_page(
                        url=f"https://w.example/{i}",
                        origin="https://w.example/",
                        depth=0,
                        word_freqs={"token": 1},
                        title_words=set(),
                        crawler_id="w",
                    )
                    i += 1
            except BaseException as e:
                failures.append(e)

        def reader():
            try:
                while not stop.is_set():
                    _ = self.store.search(["token"])
                    _ = self.store.total_pages_indexed()
            except BaseException as e:
                failures.append(e)

        wt = threading.Thread(target=writer)
        rt = threading.Thread(target=reader)
        wt.start()
        rt.start()
        wt.join(timeout=15)
        stop.set()
        rt.join(timeout=5)
        self.assertEqual(failures, [])


if __name__ == "__main__":
    unittest.main()
