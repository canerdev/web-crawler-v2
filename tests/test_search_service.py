"""Unit tests for SearchService relevancy scoring (_score_entries) and ranking."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from core.index_store import IndexEntry, IndexStore
from services.search_service import SearchService, _score_entries


def _isolated_store(tmp: Path) -> IndexStore:
    storage_dir = tmp / "data" / "storage"
    storage_dir.mkdir(parents=True, exist_ok=True)
    storage_file = storage_dir / "p.data"
    with patch("core.index_store.STORAGE_DIR", storage_dir), patch(
        "core.index_store.STORAGE_FILE", storage_file
    ):
        return IndexStore()


class TestScoreEntries(unittest.TestCase):
    def test_known_score_exact_match_low_depth(self):
        entry = IndexEntry(
            word="python",
            url="https://ex.com/a",
            origin="https://ex.com/",
            depth=0,
            frequency=2,
            in_title=False,
        )
        scored = _score_entries([entry], ["python"])
        self.assertEqual(len(scored), 1)
        score, out = scored[0]
        # freq*10 + 1000 - depth*5 = 20 + 1000 - 0 = 1020
        self.assertEqual(score, 1020.0)
        self.assertIs(out, entry)

    def test_prefix_query_no_exact_word_bonus(self):
        entry = IndexEntry(
            word="running",
            url="https://ex.com/b",
            origin="https://ex.com/",
            depth=1,
            frequency=1,
            in_title=False,
        )
        scored = _score_entries([entry], ["run"])
        score, _ = scored[0]
        # 10 - 5 = 5 (no +1000 because "running" not in query set)
        self.assertEqual(score, 5.0)

    def test_depth_penalty(self):
        e0 = IndexEntry(
            word="cat",
            url="https://ex.com/c",
            origin="https://ex.com/",
            depth=0,
            frequency=1,
            in_title=False,
        )
        e2 = IndexEntry(
            word="cat",
            url="https://ex.com/d",
            origin="https://ex.com/",
            depth=2,
            frequency=1,
            in_title=False,
        )
        s0 = _score_entries([e0], ["cat"])[0][0]
        s2 = _score_entries([e2], ["cat"])[0][0]
        self.assertGreater(s0, s2)
        self.assertEqual(s0 - s2, 10.0)  # two depth steps * 5


class TestSearchServiceIntegration(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.store = _isolated_store(Path(self._tmp.name))
        self.svc = SearchService(self.store)

    def tearDown(self):
        self._tmp.cleanup()

    def test_dedupe_by_url_keeps_best_score(self):
        self.store.add_page(
            url="https://rank.example/u",
            origin="https://rank.example/",
            depth=0,
            word_freqs={"queryterm": 1},
            title_words=set(),
            crawler_id="r",
        )
        self.store.add_page(
            url="https://rank.example/u",
            origin="https://rank.example/",
            depth=0,
            word_freqs={"queryterm": 10},
            title_words=set(),
            crawler_id="r",
        )
        results = self.svc.search("queryterm", limit=10)
        urls = [r["relevant_url"] for r in results]
        self.assertEqual(urls.count("https://rank.example/u"), 1)

    def test_sort_by_depth(self):
        self.store.add_page(
            url="https://s.example/deep",
            origin="https://s.example/",
            depth=5,
            word_freqs={"zzword": 100},
            title_words=set(),
            crawler_id="s",
        )
        self.store.add_page(
            url="https://s.example/shallow",
            origin="https://s.example/",
            depth=0,
            word_freqs={"zzword": 1},
            title_words=set(),
            crawler_id="s",
        )
        out = self.svc.search("zzword", limit=10, sort_by="depth")
        depths = [r["depth"] for r in out]
        self.assertEqual(depths, sorted(depths))


if __name__ == "__main__":
    unittest.main()
