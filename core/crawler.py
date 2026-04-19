import logging
import queue
import re
import ssl
import threading
import time
import urllib.error
import urllib.request
from collections import Counter
from urllib.parse import urlparse

from core.index_store import CrawlerStats, IndexStore
from core.parser import parse_html

logger = logging.getLogger(__name__)

_WORD_RE = re.compile(r"\b[a-zA-Z]{2,}\b")
_DEFAULT_TIMEOUT = 10
_DEFAULT_WORKERS = 5
_DEFAULT_QUEUE_SIZE = 1000
_DEFAULT_RATE = 10


def _make_ssl_context(verify: bool = True) -> ssl.SSLContext:
    if verify:
        return ssl.create_default_context()
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


class CrawlerJob(threading.Thread):
    """BFS web crawler that runs in a background thread with a pool of workers."""

    def __init__(
        self,
        crawler_id: str,
        origin: str,
        max_depth: int,
        index_store: IndexStore,
        num_workers: int = _DEFAULT_WORKERS,
        queue_size: int = _DEFAULT_QUEUE_SIZE,
        max_rate: float = _DEFAULT_RATE,
    ):
        super().__init__(daemon=True, name=f"crawler-{crawler_id}")
        self.crawler_id = crawler_id
        self.origin = origin
        self.max_depth = max_depth
        self.index_store = index_store
        self.num_workers = num_workers
        self.max_rate = max_rate

        self._url_queue: queue.Queue[tuple[str, int]] = queue.Queue(maxsize=queue_size)
        self._visited: set[str] = set()
        self._visited_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._pause_event.set()
        self._active_count = 0
        self._active_lock = threading.Lock()

        self._rate_interval = 1.0 / max_rate if max_rate > 0 else 0
        self._rate_lock = threading.Lock()
        self._last_request_time = 0.0

        self.index_store.register_crawler(
            CrawlerStats(
                crawler_id=crawler_id,
                origin=origin,
                max_depth=max_depth,
                queue_capacity=queue_size,
                total_workers=num_workers,
            )
        )

    def run(self):
        self._url_queue.put((self.origin, 0))
        self._sync_stats()

        workers: list[threading.Thread] = []
        for i in range(self.num_workers):
            t = threading.Thread(
                target=self._worker_loop,
                daemon=True,
                name=f"worker-{self.crawler_id}-{i}",
            )
            t.start()
            workers.append(t)

        for t in workers:
            t.join()

        finished_naturally = not self._stop_event.is_set()
        self.index_store.update_stats(
            self.crawler_id,
            is_running=False,
            is_finished=finished_naturally,
            active_workers=0,
        )
        logger.info(
            "Crawler %s %s",
            self.crawler_id,
            "finished" if finished_naturally else "stopped",
        )

    def stop(self):
        self._pause_event.set()
        self._stop_event.set()
        self.index_store.update_stats(self.crawler_id, is_running=False, is_paused=False)

    def pause(self):
        self._pause_event.clear()
        self.index_store.update_stats(self.crawler_id, is_paused=True)

    def resume(self):
        self._pause_event.set()
        self.index_store.update_stats(self.crawler_id, is_paused=False)

    @property
    def is_stopped(self) -> bool:
        return self._stop_event.is_set()

    @property
    def is_paused(self) -> bool:
        return not self._pause_event.is_set()

    def _worker_loop(self):
        idle_rounds = 0
        max_idle = 10

        while not self._stop_event.is_set():
            try:
                url, depth = self._url_queue.get(timeout=0.5)
            except queue.Empty:
                idle_rounds += 1
                if idle_rounds >= max_idle and self._all_idle():
                    break
                continue

            idle_rounds = 0
            self._pause_event.wait()
            if self._stop_event.is_set():
                self._url_queue.task_done()
                break

            with self._active_lock:
                self._active_count += 1

            try:
                self._process_url(url, depth)
            except Exception:
                logger.exception("Error processing %s", url)
            finally:
                self._url_queue.task_done()
                with self._active_lock:
                    self._active_count -= 1
                self._sync_stats()

    def _all_idle(self) -> bool:
        with self._active_lock:
            return self._active_count == 0 and self._url_queue.empty()

    def _process_url(self, url: str, depth: int):
        with self._visited_lock:
            if url in self._visited:
                return
            self._visited.add(url)

        self._wait_rate_limit()
        html = self._fetch(url)
        if html is None:
            return

        result = parse_html(html, url)

        word_counts = Counter(w.lower() for w in _WORD_RE.findall(result.text))
        title_words = set(w.lower() for w in _WORD_RE.findall(result.title))

        if word_counts:
            self.index_store.add_page(
                url=url,
                origin=self.origin,
                depth=depth,
                word_freqs=dict(word_counts),
                title_words=title_words,
                crawler_id=self.crawler_id,
            )

        if depth < self.max_depth:
            for link in result.links:
                if self._stop_event.is_set():
                    break
                with self._visited_lock:
                    if link in self._visited:
                        continue
                try:
                    self._url_queue.put_nowait((link, depth + 1))
                except queue.Full:
                    self.index_store.update_stats(self.crawler_id, backpressure_active=True)
                    break

    def _fetch(self, url: str) -> str | None:
        for verify in (True, False):
            try:
                ctx = _make_ssl_context(verify=verify)
                req = urllib.request.Request(
                    url,
                    headers={"User-Agent": "MidasCrawler/1.0"},
                )
                with urllib.request.urlopen(req, timeout=_DEFAULT_TIMEOUT, context=ctx) as resp:
                    content_type = resp.headers.get("Content-Type", "")
                    if "text/html" not in content_type:
                        return None
                    raw = resp.read(5 * 1024 * 1024)
                    charset = resp.headers.get_content_charset() or "utf-8"
                    try:
                        return raw.decode(charset)
                    except (UnicodeDecodeError, LookupError):
                        return raw.decode("latin-1")
            except ssl.SSLError:
                if not verify:
                    return None
                continue
            except urllib.error.URLError as e:
                if isinstance(e.reason, ssl.SSLError) and verify:
                    continue
                return None
            except (OSError, ValueError):
                return None
        return None

    def _wait_rate_limit(self):
        if self._rate_interval <= 0:
            return
        with self._rate_lock:
            now = time.monotonic()
            elapsed = now - self._last_request_time
            if elapsed < self._rate_interval:
                time.sleep(self._rate_interval - elapsed)
            self._last_request_time = time.monotonic()

    def _sync_stats(self):
        with self._visited_lock:
            visited = len(self._visited)
        with self._active_lock:
            active = self._active_count
        bp = self._url_queue.full()
        self.index_store.update_stats(
            self.crawler_id,
            urls_visited=visited,
            urls_queued=self._url_queue.qsize(),
            active_workers=active,
            backpressure_active=bp,
        )
