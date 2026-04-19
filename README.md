# Web Crawler & Search Engine (Project 2)

**Repository:** `https://github.com/canerdev/web-crawler-v2`

This is the **Homework 2 / Project 2** version: same crawler and search behavior as Project 1, developed with a **documented multi-agent workflow** (see `multi_agent_workflow.md` when present, and `agents/`). The running system is a normal Python app — agents refer to how the work was organized, not a runtime multi-agent engine.

A web crawling and search platform built with Python. Crawl sites to a configurable depth, then search indexed content by keyword. Everything runs in a **single process** using threads — no external database or message queue for the core exercise.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Open **http://127.0.0.1:3600** (or `http://localhost:3600`) in your browser.

## How it works

Two operations: **index** and **search**.

### `index(origin, k)`

Starts a crawl from `origin`, following links up to **`k` hops** from the seed (`k` must be a **positive integer**, ≥ 1).

```http
POST /index
Content-Type: application/json

{"origin": "https://example.com", "k": 2}
```

**Response:** `201 Created` with `{"crawler_id": "<id>", "status": "started"}`.

When indexing runs:

1. A **coordinator** thread starts a pool of **worker** threads (default: 5).
2. The origin is enqueued at **depth 0** in a **bounded** URL queue.
3. Workers dequeue `(url, depth)`, fetch with `urllib`, parse with `html.parser`, tokenize text, and append postings into an **in-memory inverted index**.
4. Outgoing links are enqueued at **depth + 1** only while **`depth < k`** (see `architect_design.md` for exact semantics).
5. When the queue is empty and workers are idle, the crawl finishes (unless stopped).

The same URL is never processed twice: a **visited** set per job (lock-protected) deduplicates across workers.

### `search(query)`

Keyword search over indexed pages with ranking.

```http
GET /search?query=example
```

Optional query parameters: **`limit`** (default 50), **`sortBy`** (`relevance` or `depth`, default `relevance`).

**Response shape:**

```json
{
  "query": "example",
  "count": 1,
  "results": [
    {
      "relevant_url": "https://example.com/",
      "origin_url": "https://example.com/",
      "depth": 0,
      "frequency": 3,
      "relevance_score": 1030.0
    }
  ]
}
```

- **`relevant_url`** — page where the term matched  
- **`origin_url`** — seed URL of the crawl that indexed that page  
- **`depth`** — hop count from that seed  
- **`frequency`** / **`relevance_score`** — scoring fields used for ranking  

Search can run **while a crawl is in progress**; new pages become searchable after they are indexed.

### Relevancy scoring

Results are scored in the spirit of:

`score ≈ (frequency × 10) + (large bonus for exact word match) − (depth × 5)`

Entries are **deduplicated by URL** (best score wins), then sorted by `sortBy`.

## Back-pressure

1. **Bounded queue** — The per-crawl URL queue has a maximum size (default 1000). When full, further enqueues from that page are not accepted (`put_nowait`); status reports **`backpressure_active`** when the queue is full.  
2. **Rate limiting** — A shared limiter caps fetch rate (default **10 requests per second** across workers) before each HTTP request.

## Concurrency (overview)

```
Flask (main thread)
  |
  |-- POST /index --> CrawlerService --> CrawlerJob (thread)
  |                                        +-- workers (pool) --> fetch / parse / IndexStore.add_page
  |
  |-- GET /search --> SearchService --> IndexStore (read)
  |-- GET /status --> CrawlerService / IndexStore stats
```

Shared data uses locks (`IndexStore` RLock, per-job visited / rate / active counters). The URL queue is `queue.Queue`.

## Crawler lifecycle

States: **running** → **finished** (queue drained), or **paused** ↔ **running**, or **stopping** → **stopped**. Use **`POST /pause/<id>`**, **`POST /resume/<id>`**, **`POST /stop/<id>`** with the `crawler_id` from `POST /index`.

## Project structure

```
web-crawler-v2/
├── PRD.md                  Product requirements (course)
├── architect_design.md     System design / API contract for agents
├── app.py                  Flask API (routes, CORS, static)
├── requirements.txt
├── core/
│   ├── parser.py           HTML extraction (stdlib html.parser)
│   ├── index_store.py      Thread-safe inverted index + optional disk append
│   └── crawler.py          Multi-worker BFS crawler
├── services/
│   ├── crawler_service.py  Start / stop / pause / resume
│   └── search_service.py   Query, score, rank
├── data/storage/           Local persistence (see below)
├── static/index.html       Web UI
├── agents/                 Agent role definitions (course)
└── tests/                  unittest suite (no network in unit tests)
```

## API reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `POST /index` | JSON `{"origin", "k"}` (`k` ≥ 1) | Start a crawl. Returns **`201`** + `crawler_id`. |
| `GET /search?query=...` | | Search. Optional `limit`, `sortBy`. |
| `GET /status` | | All crawlers + aggregates. |
| `GET /status/<id>` | | One crawler. |
| `POST /pause/<id>` | | Pause. |
| `POST /resume/<id>` | | Resume. |
| `POST /stop/<id>` | | Stop. |

## Running tests

From the project root with the venv activated:

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
```

Core tests use the **stdlib only** (no live network in typical unit tests).

## Data storage

Indexed postings may be **appended** to `data/storage/p.data` (tab-separated). On startup, existing lines can be loaded into memory. **Do not commit large `p.data` files** to Git; add `data/storage/p.data` to `.gitignore` if needed. If you do not ship a compressed sample, omit any `gunzip` step — only decompress if you have a checked-in `p.data.gz`.

## Design notes

- **stdlib for core crawl/parse** — `urllib.request`, `html.parser`. No Scrapy / Beautiful Soup for the core pipeline. Flask is the web layer.  
- **Thread pool per crawl job** — Parallelizes I/O; shared rate limiter limits total fetch rate.  
- **SSL** — Fetcher tries verified TLS first, then may retry with verification disabled (same pattern as Project 1).  
- **Parity** — Behavior is aligned with Project 1 (`web-crawler`); see `PRD.md` §2.5 and `architect_design.md`.
