import threading
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple


class IndexEntry(NamedTuple):
    word: str
    url: str
    origin: str
    depth: int
    frequency: int
    in_title: bool


@dataclass
class CrawlerStats:
    crawler_id: str
    origin: str
    max_depth: int
    urls_visited: int = 0
    urls_queued: int = 0
    queue_capacity: int = 0
    active_workers: int = 0
    total_workers: int = 0
    is_running: bool = True
    is_paused: bool = False
    is_finished: bool = False
    backpressure_active: bool = False


STORAGE_DIR = Path(__file__).resolve().parent.parent / "data" / "storage"
STORAGE_FILE = STORAGE_DIR / "p.data"


class IndexStore:
    """Thread-safe in-memory inverted index shared between crawlers and search."""

    def __init__(self):
        self._lock = threading.RLock()
        self._index: dict[str, list[IndexEntry]] = {}
        self._stats: dict[str, CrawlerStats] = {}
        self._pages: set[str] = set()
        self._ensure_storage_dir()
        self._load_from_disk()

    def _ensure_storage_dir(self):
        STORAGE_DIR.mkdir(parents=True, exist_ok=True)

    def _load_from_disk(self):
        if not STORAGE_FILE.exists():
            return
        with open(STORAGE_FILE, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) >= 5:
                    word, url, origin, depth_str, freq_str = parts[:5]
                    try:
                        entry = IndexEntry(
                            word=word,
                            url=url,
                            origin=origin,
                            depth=int(depth_str),
                            frequency=int(freq_str),
                            in_title=False,
                        )
                        if word not in self._index:
                            self._index[word] = []
                        self._index[word].append(entry)
                        self._pages.add(url)
                    except ValueError:
                        continue

    def _append_to_disk(self, entry: IndexEntry):
        with open(STORAGE_FILE, "a", encoding="utf-8") as f:
            f.write(
                f"{entry.word}\t{entry.url}\t{entry.origin}\t{entry.depth}\t{entry.frequency}\n"
            )

    def add_page(
        self,
        url: str,
        origin: str,
        depth: int,
        word_freqs: dict[str, int],
        title_words: set[str],
        crawler_id: str,
    ):
        with self._lock:
            self._pages.add(url)
            for word, freq in word_freqs.items():
                entry = IndexEntry(
                    word=word,
                    url=url,
                    origin=origin,
                    depth=depth,
                    frequency=freq,
                    in_title=(word in title_words),
                )
                if word not in self._index:
                    self._index[word] = []
                self._index[word].append(entry)
                self._append_to_disk(entry)

    def search(self, query_words: list[str]) -> list[IndexEntry]:
        """Return all entries matching any of the query words (exact + prefix)."""
        with self._lock:
            results: list[IndexEntry] = []
            for qw in query_words:
                if qw in self._index:
                    results.extend(self._index[qw])
                for stored_word, entries in self._index.items():
                    if stored_word != qw and stored_word.startswith(qw):
                        results.extend(entries)
            return results

    def register_crawler(self, stats: CrawlerStats):
        with self._lock:
            self._stats[stats.crawler_id] = stats

    def update_stats(self, crawler_id: str, **kwargs):
        with self._lock:
            if crawler_id in self._stats:
                stats = self._stats[crawler_id]
                for key, value in kwargs.items():
                    if hasattr(stats, key):
                        setattr(stats, key, value)

    def get_stats(self, crawler_id: str) -> CrawlerStats | None:
        with self._lock:
            s = self._stats.get(crawler_id)
            if s is None:
                return None
            return CrawlerStats(
                crawler_id=s.crawler_id,
                origin=s.origin,
                max_depth=s.max_depth,
                urls_visited=s.urls_visited,
                urls_queued=s.urls_queued,
                queue_capacity=s.queue_capacity,
                active_workers=s.active_workers,
                total_workers=s.total_workers,
                is_running=s.is_running,
                is_paused=s.is_paused,
                is_finished=s.is_finished,
                backpressure_active=s.backpressure_active,
            )

    def get_all_stats(self) -> list[CrawlerStats]:
        with self._lock:
            return [self.get_stats(cid) for cid in self._stats]

    def total_pages_indexed(self) -> int:
        with self._lock:
            return len(self._pages)

    def total_words_indexed(self) -> int:
        with self._lock:
            return len(self._index)
