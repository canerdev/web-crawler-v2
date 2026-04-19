# Architect design artifacts

**Status:** Approved for implementation  
**Inputs:** `PRD.md` v2.0  
**Reference parity:** System design matches **Project 1** (`../web-crawler`): same responsibilities, HTTP surface, crawler architecture (worker pool + bounded queue + rate limit), inverted-index shape, search scoring shape, and depth/back-pressure semantics. v2 adds the course multi-agent workflow and documentation deliverables only; behavior is not a redesign.

**Stack:** Python 3.11+, Flask, `urllib` + `html.parser` for core fetch/parse (no Scrapy / Beautiful Soup).

---

## 1. Component diagram (ASCII)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Presentation (UI Agent) — SPA static + API             │
└───────────────────────────────────────┬─────────────────────────────────┘
                                        │ JSON / HTTP (CORS: * per P1)
                                        ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         Flask app (Backend)                               │
│  POST /index   GET /search   GET /status   GET /status/<crawler_id>      │
│  POST /stop|pause|resume/<crawler_id>                                    │
└─────────┬───────────────────┬───────────────────────┬───────────────────┘
          │                   │                       │
          ▼                   ▼                       ▼
┌─────────────────┐ ┌─────────────────┐ ┌───────────────────────────────┐
│ CrawlerService  │ │ SearchService   │ │ Status: crawler_service.      │
│ • registry of   │ │ • query → ranked│ │   summary() / get_status(id)  │
│   CrawlerJob    │ │   URL dicts     │ │                               │
└────────┬────────┘ └────────┬────────┘ └───────────────────────────────┘
         │                   │
         │                   │ reads
         ▼                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ IndexStore (shared, thread-safe)                                         │
│ • Inverted index: word → list[IndexEntry(word,url,origin,depth,freq,…)] │
│ • CrawlerStats per crawler_id; optional disk append (parity with P1)     │
└──────────────────────────────────────▲────────────────────────────────────┘
                                       │ writes
┌──────────────────────────────────────┴──────────────────────────────────┐
│ CrawlerJob (extends threading.Thread)                                    │
│   Main thread: starts N worker threads, joins them, marks job finished    │
│   Workers: dequeue (url, depth) → visit → fetch → parse → index → enqueue│
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────────┐             │
│  │ queue.Queue  │──▶│ worker × N   │──▶│ parse_html       │             │
│  │ (bounded)    │   │ + rate limit │   │ (html.parser)    │             │
│  └──────────────┘   └──────────────┘   └────────┬─────────┘             │
│         ▲                              Visited set + enqueue new links   │
│         └────────────────────────────────────────────────────────────────┘
```

**Data flow (same as Project 1)**

- **Index:** `CrawlerService.start_crawl` builds a `CrawlerJob`, registers stats, starts the job thread. Job seeds `(origin, 0)` on the queue; workers run `_process_url` → fetch (after rate limit) → `IndexStore.add_page` → enqueue links per depth rules.
- **Search:** `SearchService.search` reads `IndexStore` only; ranks/deduplicates by URL.
- **Status:** Per-crawler stats live in `IndexStore` (`CrawlerStats`); global summary aggregates totals and lists crawlers.

---

## 2. API contracts

### 2.1 HTTP (Flask) — external API (aligned with Project 1)

| Method | Path | Request | Success response | Errors |
|--------|------|---------|------------------|--------|
| `POST` | `/index` | JSON `{"origin": string, "k": int}` — **`k` must be a positive integer (≥ 1)** | **201** `{"crawler_id": string, "status": "started"}` | **400** if `origin` missing or not a string, or `k` missing / not int / `< 1` |
| `GET` | `/search` | Query **`query`** (required), optional **`limit`** (int, default 50), **`sortBy`** (`relevance` \| `depth`, default `relevance`) | `{"query", "count", "results": [...]}` each result includes at least `relevant_url`, `origin_url`, `depth`, plus scoring fields per P1 | **400** if `query` empty |
| `GET` | `/status` | — | Summary: `total_crawlers`, `total_pages_indexed`, `total_words_indexed`, `crawlers`: list of per-crawler status dicts | — |
| `GET` | `/status/<crawler_id>` | — | Same fields as one element of `crawlers` (see below) | **404** if unknown id |
| `POST` | `/stop/<crawler_id>` | — | `{"status": "stopping", "crawler_id"}` | **404** if not found |
| `POST` | `/pause/<crawler_id>` | — | `{"status": "paused", "crawler_id"}` | **404** if not found or stopped |
| `POST` | `/resume/<crawler_id>` | — | `{"status": "resumed", "crawler_id"}` | **404** if not found or stopped |

**Per-crawler status payload (`CrawlerStats` projection — match P1)**

| Field | Type | Meaning |
|-------|------|---------|
| `crawler_id` | `str` | Job id returned from `POST /index` |
| `origin` | `str` | Seed URL |
| `max_depth` | `int` | Same as `k` for this job |
| `urls_visited` | `int` | Size of visited set (approximation of progress) |
| `urls_queued` | `int` | `queue.qsize()` |
| `queue_capacity` | `int` | Bounded queue `maxsize` |
| `active_workers` | `int` | Workers currently in `_process_url` |
| `total_workers` | `int` | Pool size `N` |
| `is_running` | `bool` | |
| `is_paused` | `bool` | |
| `is_finished` | `bool` | Natural completion vs stop |
| `backpressure_active` | `bool` | **True** when frontier queue is full (see §4) |

`OPTIONS` preflight: CORS headers as in P1 (`Access-Control-Allow-Origin: *`, etc.).

---

### 2.2 Internal module contracts (logical; names match Project 1)

#### `CrawlerService`

| Operation | Parameters | Returns |
|-----------|------------|---------|
| `start_crawl` | `origin: str`, `k: int` | `crawler_id: str` — starts `CrawlerJob` thread |
| `stop_crawl` / `pause_crawl` / `resume_crawl` | `crawler_id: str` | `bool` |
| `get_status` | `crawler_id: str` | dict or `None` |
| `summary` | — | global summary dict for `GET /status` |

Multiple concurrent crawls are **allowed** (registry keyed by `crawler_id`), same as P1.

#### `CrawlerJob` (`threading.Thread`)

- **Fields:** `crawler_id`, `origin`, `max_depth` (= `k`), `index_store`, configurable `num_workers`, `queue_size`, `max_rate`.
- **Queue:** `queue.Queue(maxsize=queue_size)` of `(url: str, depth: int)`.
- **Visited:** `set[str]` + `threading.Lock` — check/add at start of `_process_url` (same ordering as P1).
- **Workers:** `num_workers` daemon threads running `_worker_loop` (deque, pause/stop events, idle exit when queue empty and all workers idle).
- **Rate limiting:** global minimum interval between fetches **per worker** via shared timing + lock (`_wait_rate_limit`), same semantics as P1.

#### `parse_html` (parser module)

- **Signature:** `(html_content: str, base_url: str) -> ParseResult` with `title`, `text`, `links` (absolute http/https, fragment stripped, deduped).

#### `IndexStore`

| Operation | Parameters | Returns |
|-----------|------------|---------|
| `register_crawler` | `CrawlerStats` | — |
| `update_stats` | `crawler_id`, `**kwargs` | — |
| `add_page` | `url`, `origin` (seed), `depth`, `word_freqs: dict[str,int]`, `title_words: set[str]`, `crawler_id` | — — builds inverted postings per word |
| `search` | `query_words: list[str]` | `list[IndexEntry]` — prefix + exact match expansion as P1 |
| `get_stats` / `get_all_stats` | | |
| `total_pages_indexed` / `total_words_indexed` | | |

**`IndexEntry` fields:** `word`, `url`, `origin`, `depth`, `frequency`, `in_title` (or equivalent).

#### `SearchService`

- `search(query: str, limit: int = 50, sort_by: str = "relevance") -> list[dict]`
- Scoring: **same family as P1** — frequency-based score, exact-match bonus, depth penalty, dedupe by URL keeping best score; `sortBy` `relevance` vs `depth`.

---

## 3. Concurrency design table

**Model:** One `CrawlerJob` thread owns **N worker threads**; all share one bounded queue and one `IndexStore`.

| Shared resource | Protect with | Writers | Readers |
|-----------------|--------------|---------|---------|
| `IndexStore` inverted index + `_pages` + `_stats` | `threading.RLock` | Workers (`add_page`, `update_stats`, `register_crawler`) | `SearchService`, `get_status`, summary |
| `Visited` set | `threading.Lock` | Workers (`_process_url` add-after-check) | — |
| `queue.Queue` | thread-safe API | Workers `get` / `put_nowait` | `update_stats` (`qsize`) |
| Rate limiter state (`_last_request_time`) | `threading.Lock` | Workers | — |
| `_active_count` (workers in URL processing) | `threading.Lock` | Workers | stats sync |
| `CrawlerService._crawlers` | `threading.Lock` | `start_crawl` | stop/pause/resume |

**AC-3 / S-3:** Search holds `IndexStore` lock only for the duration of `search()` / reads; indexer releases lock after each `add_page`. No nested lock order that waits on the crawl pipeline.

**Deadlock note:** Workers must **not** hold `IndexStore` lock while waiting on rate limit or network (P1: fetch outside index lock). Enqueue uses `put_nowait`; on failure, update back-pressure and **stop enqueuing links from that page** (do not block holding index lock).

---

## 4. Back-pressure design decision (Project 1 behavior)

**Two mechanisms — both present:**

| Mechanism | Behavior |
|-----------|----------|
| **Bounded `queue.Queue(maxsize=queue_size)`** | Enqueue with **`put_nowait((link, depth+1))`**. On **`queue.Full`**, set **`backpressure_active`** on crawler stats and **break** out of the link loop for that page (no unbounded growth). Summary: back-pressure is **non-blocking enqueue with drop/stop-enqueue for remaining links on that page**, not a blocking `put` that stalls the worker forever. |
| **Per-worker rate limit** | Minimum interval between HTTP fetches (global cap across workers via shared clock), implemented in the worker before `_fetch`. |

**Status:** `backpressure_active` is driven by **`queue.full()`** (and/or sticky flag updated in stats sync) so the UI shows when the frontier is saturated.

**Constants (document in `readme.md`):** default `queue_size`, `num_workers`, `max_rate` — same order of magnitude as P1 unless you document a change.

---

## 5. Crawl depth semantics (exact match to Project 1)

- **Seed:** enqueue `(origin, 0)` before workers start.
- **`k` / `max_depth`:** positive integer; **same field** as `POST /index` body `k`.
- **Outgoing links:** from a page fetched at **`depth`**, enqueue `(link, depth + 1)` **only if `depth < max_depth`**.
- **Equivalently:** pages are fetched at depths **`0, 1, …, max_depth`**; from depth **`max_depth`**, no new URLs are enqueued.

**Visited:** URL skipped if already in visited set at the start of `_process_url` (before fetch). Links: skip if already visited before `put_nowait`.

**Provenance (S-2):** For each indexed page, `origin_url` stored in the index is the **crawl seed** `origin` (not the referring page). `depth` is the **depth integer** passed through the queue for that fetch.

---

## 6. Handoff notes

- **Backend Agent:** Mirror `web-crawler` layout: `app.py`, `core/crawler.py`, `core/index_store.py`, `core/parser.py`, `services/crawler_service.py`, `services/search_service.py`, `static/` unless told otherwise.
- **UI Agent:** Use **`query`** for search; display **`crawler_id`** after index; poll **`/status` or `/status/<id>`** for progress, `urls_queued`, `backpressure_active`.
- **QA Agent:** Map tests to PRD AC-* and to P1 behaviors (depth, concurrent search, queue full).

*End of architect artifacts.*
