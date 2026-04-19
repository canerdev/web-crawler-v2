import threading
import uuid

from core.crawler import CrawlerJob
from core.index_store import IndexStore


class CrawlerService:
    def __init__(self, index_store: IndexStore):
        self._store = index_store
        self._crawlers: dict[str, CrawlerJob] = {}
        self._lock = threading.Lock()

    def start_crawl(self, origin: str, k: int) -> str:
        crawler_id = uuid.uuid4().hex[:12]
        job = CrawlerJob(
            crawler_id=crawler_id,
            origin=origin,
            max_depth=k,
            index_store=self._store,
        )
        with self._lock:
            self._crawlers[crawler_id] = job
        job.start()
        return crawler_id

    def stop_crawl(self, crawler_id: str) -> bool:
        with self._lock:
            job = self._crawlers.get(crawler_id)
        if job is None:
            return False
        job.stop()
        return True

    def pause_crawl(self, crawler_id: str) -> bool:
        with self._lock:
            job = self._crawlers.get(crawler_id)
        if job is None or job.is_stopped:
            return False
        job.pause()
        return True

    def resume_crawl(self, crawler_id: str) -> bool:
        with self._lock:
            job = self._crawlers.get(crawler_id)
        if job is None or job.is_stopped:
            return False
        job.resume()
        return True

    def get_status(self, crawler_id: str):
        stats = self._store.get_stats(crawler_id)
        if stats is None:
            return None
        return _stats_to_dict(stats)

    def list_crawlers(self) -> list[dict]:
        all_stats = self._store.get_all_stats()
        return [_stats_to_dict(s) for s in all_stats if s is not None]

    def summary(self) -> dict:
        return {
            "total_crawlers": len(self._crawlers),
            "total_pages_indexed": self._store.total_pages_indexed(),
            "total_words_indexed": self._store.total_words_indexed(),
            "crawlers": self.list_crawlers(),
        }


def _stats_to_dict(s) -> dict:
    return {
        "crawler_id": s.crawler_id,
        "origin": s.origin,
        "max_depth": s.max_depth,
        "urls_visited": s.urls_visited,
        "urls_queued": s.urls_queued,
        "queue_capacity": s.queue_capacity,
        "active_workers": s.active_workers,
        "total_workers": s.total_workers,
        "is_running": s.is_running,
        "is_paused": s.is_paused,
        "is_finished": s.is_finished,
        "backpressure_active": s.backpressure_active,
    }
