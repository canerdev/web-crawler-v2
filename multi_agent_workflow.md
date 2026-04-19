# Multi-Agent Workflow
## Web Crawler & Search Engine

---

## 1. Overview

This project was built using a multi-agent AI workflow. Rather than prompting a single AI session to produce the entire codebase, the work was divided across four specialized agents — each with a defined role, a scoped set of responsibilities, and a clear set of inputs and outputs. The human developer acted as the orchestrator: passing artifacts between agents, reviewing outputs, making final decisions, and resolving conflicts when agents produced incompatible results.

The agents do not communicate directly. All handoffs are mediated by the developer.

---

## 2. Agents

### 2.1 Architect Agent

**Role:** System designer. Produces all design artifacts before any code is written. Does not write implementation code.

**Responsibilities:**
- Define component boundaries and module ownership
- Define API contracts between all modules (function signatures, parameter types, return types)
- Design the concurrency model: which data structures are shared, which locks protect them, how indexer and searcher coexist safely
- Design the back-pressure mechanism
- Define crawl depth semantics precisely

**Inputs:**
- `PRD.md`

**Outputs:**
- Component diagram
- API contracts for all modules
- Concurrency design table
- Back-pressure design decision

**Consumed by:** Backend Agent, UI Agent, QA Agent

---

### 2.2 Backend Agent

**Role:** Backend engineer. Implements all server-side Python code from the Architect Agent's design artifacts.

**Responsibilities:**
- Implement the HTML parser using Python stdlib only (`html.parser`, `urllib`)
- Implement the thread-safe inverted index with the locking strategy specified by the Architect Agent
- Implement the BFS crawler with worker thread pool, visited set, depth tracking, and back-pressure
- Implement Flask routes: `POST /index`, `GET /search`, `GET /status`, `POST /stop`
- Implement crawler lifecycle and search service

**Inputs:**
- `PRD.md`
- Architect Agent outputs (component diagram, API contracts, concurrency design)

**Outputs:**
- `core/parser.py`
- `core/index_store.py`
- `core/crawler.py`
- `services/crawler_service.py`
- `services/search_service.py`
- `app.py`
- `requirements.txt`

**Consumed by:** UI Agent (depends on live routes), QA Agent (tests these modules directly)

---

### 2.3 UI Agent

**Role:** Frontend engineer. Implements a single-page web UI as a self-contained HTML file.

**Responsibilities:**
- Build the indexing panel: origin URL input, depth input, start button
- Build the search panel: query input, submit button, results table showing `relevant_url`, `origin_url`, `depth`
- Build the system state panel: crawl status, queue depth, back-pressure status, visited count — auto-refreshed while a crawl is running
- All API calls must match the contracts defined by the Architect Agent exactly

**Inputs:**
- Architect Agent outputs (API contracts)
- Running Backend Agent server (for testing)

**Outputs:**
- `static/index.html`

**Consumed by:** QA Agent (verifies UI covers all required user actions)

---

### 2.4 QA Agent

**Role:** Quality assurance engineer. Writes tests and reviews all other agents' outputs against the PRD acceptance criteria.

**Responsibilities:**
- Write unit tests for the HTML parser
- Write unit tests for the inverted index, including a concurrent write test
- Write unit tests for the relevancy scoring function
- Review Backend Agent outputs against PRD acceptance criteria and flag gaps
- Verify no prohibited libraries appear in the codebase

**Inputs:**
- `PRD.md`
- Architect Agent outputs
- All Backend Agent outputs
- UI Agent output

**Outputs:**
- `tests/test_parser.py`
- `tests/test_index_store.py`
- `tests/test_search_service.py` (relevancy scoring + `SearchService` integration)
- `tests/test_crawler_depth.py` (depth semantics; mocked HTTP)
- `tests/test_policy_no_banned_libraries.py` (PRD AC-7 substring scan on `core/`, `services/`, `app.py`)
- `tests/qa_prd_review.py` (written PRD coverage notes)
- `tests/__init__.py`
- Written review of PRD coverage (also embedded in `tests/qa_prd_review.py`)

**Consumed by:** Backend Agent (fixes flagged issues), human reviewer (assesses coverage)

---

## 3. Handoff Diagram

```
PRD.md
      |
      v
[Architect Agent]
      |
      | component diagram
      | API contracts
      | concurrency design
      | back-pressure decision
      |
      +----------------+-----------------+
      |                |                 |
      v                v                 v
[Backend Agent]  [UI Agent]        [QA Agent]
      |                |                 ^
      | core/          | static/         |
      | services/      | index.html      |
      | app.py         |                 |
      |                |                 |
      +----------------+-----------------+
                       |
                  all outputs
                       |
                       v
              [QA Agent review]
                       |
              failures reported back
                       |
                       v
              [Backend Agent fixes]
```

---

## 4. Prompts & Interactions

### 4.1 Architect Agent Session

Two Architect interactions are documented: (1) initial design artifacts from `PRD.md` and `agents/architect_agent.md`, and (2) a revision pass to lock **functional parity** with the existing Project 1 codebase (`../web-crawler`).

**Prompt used — initial design artifacts:**

```
@web-crawler-v2/agents/architect_agent.md @web-crawler-v2/PRD.md okay you are architect agent. "Produce your design artifacts: component diagram, API contracts, concurrency design table, and back-pressure decision."
```

**Output produced — initial design:**

- **Deliverable file:** `architect_design.md` (single source of truth for Backend / UI / QA).
- **Component diagram (ASCII):** Presentation (SPA) → Flask (`POST /index`, `GET /search`, `GET /status`) → `CrawlController`, `SearchService`, `StatusProvider` → shared **`IndexStore`**; crawl pipeline **`FrontierQueue` (bounded)** → **`FetchClient` + rate limiter** → **`LinkExtractor`** (`html.parser`), with **`VisitedSet`** feedback into the queue.
- **HTTP API contract:** JSON `POST /index` with `origin` and `k`; `GET /search` with query parameter documented initially as **`q`** in one row (later corrected in parity pass — see below); `GET /status` with indexing flag, pages fetched, queue depth/capacity, **`backpressure`** object; optional `409` if crawl already running (single-crawl policy noted as an interpretation).
- **Internal module contracts (logical):** `CrawlController.start_crawl` / `is_crawling`; bounded **`FrontierQueue`** + **`CrawlTask(url, depth)`**; **`VisitedSet.try_mark_visited`**; **`FetchClient.fetch`** → `FetchResult`; **`LinkExtractor.extract_links`**; **`IndexStore.upsert_page`** / **`search`** returning `(relevant_url, origin_url, depth)` triples; **`SearchService`**; **`StatusProvider.snapshot`**.
- **Concurrency design table:** `IndexStore` under **`threading.RLock`**; visited set lock; **`queue.Queue`** for frontier; metrics and job lifecycle locks; **deadlock note** — do not hold the index lock while blocked on a **blocking** `queue.put`.
- **Back-pressure decision:** **Bounded queue + token-bucket / rate limit on fetches**; rationale: PRD allows either bound; dual mechanism caps memory and request rate; configurable **`FRONTIER_MAXSIZE`**, **`MAX_FETCHES_PER_SECOND`**.
- **Crawl depth semantics:** Seed at **`depth = 0`**; enqueue child at **`d+1` only if `d+1 ≤ k`**; provenance rule for duplicate graph paths (recommend **first successful ingest** depth).
- **Assumptions stated in-doc:** Python 3.11+, `urllib` / `html.parser`; where PRD is silent, optional single active crawl and deterministic relevancy heuristic.

**Prompt used — Project 1 parity:**

```
make sure the functionality, i.e. system design, is the same with project 1@web-crawler
```

**Output produced — parity revision:**

- **Updated `architect_design.md`** so the **runtime design matches Project 1** (`app.py`, `core/crawler.py`, `services/*`), not a greenfield redesign:
  - **HTTP:** `GET /search` uses query param **`query`** (not `q`); **`POST /index` → `201`** with **`crawler_id`**; **`k` validation as positive integer (≥ 1)**; full route set: **`GET /status`**, **`GET /status/<crawler_id>`**, **`POST /stop|pause|resume/<crawler_id>`**; CORS as in P1.
  - **Architecture:** **`CrawlerService`** registry + **`CrawlerJob`** as **`threading.Thread`** with a **worker pool** (not a single “one crawler thread” only); **`IndexStore.add_page`** with **word frequencies + title words** and inverted **`IndexEntry`** postings (not only a generic text `upsert_page`).
  - **Back-pressure:** Documented **actual P1 behavior** — **`put_nowait`** on enqueue; on **`queue.Full`**, set **`backpressure_active`** and **break** out of the link loop for that page; **`queue.full()`** reflected in status — rather than emphasizing blocking `put` as the primary mechanism.
  - **Depth:** **`depth < max_depth`** to enqueue children; seed **`(origin, 0)`** — equivalent to the first draft’s hop bound but aligned with **`CrawlerJob`**’s field names and loop.
  - **Concurrency table** updated for **multiple worker threads** and the same locks as P1 (`_visited_lock`, `_active_lock`, `_rate_lock`).
- **Updated `PRD.md`:** new **§2.5 Parity with Project 1 (reference system)** — points to `architect_design.md` as authoritative and states **`k` ≥ 1** for API validation consistency.

**Decisions made / changes from output:**

- **Orchestrator:** Requested a **second pass** after the initial architect delivery so v2 **does not drift** from the graded/reference Project 1 implementation; accepted file-level deliverable **`architect_design.md`** (no separate diagram image — text/ASCII only per agent brief).
- **Architect Agent (session):** Initial pass used slightly generic module names (`CrawlController`, `upsert_page`); parity pass **renamed concepts** to match the codebase (`CrawlerService` / `CrawlerJob`, `add_page`) so Backend could follow **one** document without ambiguity.
- **Rejected / not carried forward:** Optional **single-crawl-only `409`** and **blocking `put`** as the main back-pressure story were **not** required for P1 parity — final doc follows **multi-crawl registry** and **non-blocking enqueue + break** on full queue.
- **Human clarification:** Course deliverable naming (`product_prd.md` vs `PRD.md`) was left as **`PRD.md`** in-repo; no separate `product_prd.md` was created in this workflow.

**Prompt used — workflow documentation (Architect subsection, this session):**

```
@web-crawler-v2/multi_agent_workflow.md okay now fill your part in this document including this prompt and its output
```

**Output produced — workflow documentation:**

- **§4.1** (this subsection) fully populated with the **exact Architect prompts** (initial design + Project 1 parity), **summarized outputs** (`architect_design.md` contents, `PRD.md` §2.5), and **orchestrator / agent decisions** — so readers can trace what was asked and what was produced without opening chat logs. The sentence you are reading is part of that output.

---

### 4.2 Backend Agent Session

One session implemented the full backend stack in module order. The workflow template originally assumed separate prompts per file; in practice a **single orchestrator prompt** referenced the agent brief and design artifacts and listed all deliverables.

**Prompt used:**

```
@web-crawler-v2/backend_agent.md @web-crawler-v2/architect_design.md @web-crawler-v2/PRD.md you are backend agent on this project. implement the following modules in order

core/parser.py
core/index_store.py
core/crawler.py
services/crawler_service.py + services/search_service.py
app.py + requirements.txt
```

**Output produced:**

- **`core/parser.py`:** `ParseResult` dataclass; `parse_html(html_content, base_url)` using `html.parser` only—title, visible text (skips `script` / `style` / `noscript`), `a[href]` resolved with `urllib.parse.urljoin`, absolute **http/https** only, fragments stripped, links deduped.
- **`core/index_store.py`:** `IndexEntry` (`NamedTuple`), `CrawlerStats`, `IndexStore` with **`threading.RLock`**; `add_page`, `search` (exact token + **prefix** expansion on stored words), crawler stats registry, `total_pages_indexed` / `total_words_indexed`; optional **append + load** from `data/storage/p.data` for parity with Project 1.
- **`core/crawler.py`:** `CrawlerJob(threading.Thread)`—BFS via bounded **`queue.Queue`**, visited **`set`** + lock, **N worker** threads, pause/stop events, **`put_nowait`** for children with **`queue.Full`** → `backpressure_active` and stop enqueueing remaining links for that page; **shared rate limit** before fetch; fetch via **`urllib.request`** with HTML content-type check, SSL retry, 5 MB read cap; tokenization aligned with search (`\b[a-zA-Z]{2,}\b`); **`MidasCrawler/1.0`** User-Agent (same as reference `web-crawler`).
- **`services/crawler_service.py`:** UUID-based `crawler_id`, registry + **`threading.Lock`**, `start_crawl` / `stop_crawl` / `pause_crawl` / `resume_crawl`, `get_status`, `summary` with architect-specified status fields (`urls_visited`, `urls_queued`, `queue_capacity`, `active_workers`, `total_workers`, `backpressure_active`, run/pause/finish flags).
- **`services/search_service.py`:** `SearchService.search(query, limit, sort_by)`—index lookup, scoring **family matching P1** (frequency × 10, +1000 exact query-word match, −5 × depth), **dedupe by URL** keeping best score, **`sortBy`** `relevance` | `depth`.
- **`app.py`:** Flask app, **`after_request` CORS** (`Access-Control-Allow-Origin: *`, etc.), routes **`POST /index`**, **`GET /search`**, **`GET /status`**, **`GET /status/<crawler_id>`**, **`POST /stop|pause|resume/<crawler_id>`**, **`/`** → `static/index.html`; validation: `origin` string, **`k` int ≥ 1**, non-empty `query`; **`201`** on successful index start.
- **`requirements.txt`:** **`flask>=3.0,<4`** as the only declared non-stdlib dependency for the API layer.
- **Supporting:** `core/__init__.py` and `services/__init__.py` for clean imports; smoke check: venv + `pip install -r requirements.txt` + import `app`.

**Decisions made / changes from output:**

- **Orchestrator / user:** Chose **one consolidated prompt** listing modules in dependency order instead of four separate backend prompts; no revision loop on this session.
- **Backend Agent:** Implementation **mirrored the existing reference codebase** `../web-crawler` where the architect doc explicitly requires **Project 1 parity** (routes, JSON shapes, worker pool, bounded queue, rate limit, index/scoring behavior). No API contract changes beyond what `architect_design.md` already specified (including **`POST /pause`** and **`POST /resume`**, which extend the shorter list in `agents/backend_agent.md`).
- **Minor:** `IndexStore.add_page(..., crawler_id=...)` accepts `crawler_id` per internal contract but does not yet use it in postings (same as P1); disk persistence stores word/url/origin/depth/frequency only—`in_title` not persisted on reload.

---

### 4.3 UI Agent Session

Two UI sessions occurred: an initial implementation against `architect_design.md`, `PRD.md`, and `agents/ui_agent.md`, followed by a revision pass to match the reference Project 1 (`web-crawler`) UI exactly.

**Prompt used — initial implementation:**

```
@web-crawler-v2/architect_design.md @web-crawler-v2/PRD.md @web-crawler-v2/agents/ui_agent.md you are ui agent. read all the documents given and implement your part in detail. do not rush. go step by step. take a deep breath and then go
```

**Output produced — initial implementation:**

- Added `static/index.html` as a **single self-contained file** (HTML + `<style>` + `<script>`), per UI agent constraints: no frameworks, no separate CSS/JS files, no external CSS libraries.
- **Indexing:** Form for `origin` and `k` (≥ 1); `POST /index` with `Content-Type: application/json`; success message showing **`crawler_id`**; validation aligned with API (400 errors surfaced).
- **Search:** `GET /search` with `query`, `limit`, and `sortBy` (`relevance` | `depth`); results table with `relevant_url` (links), `origin_url`, `depth`, and scoring column (`relevance_score` / frequency); optional checkbox to **poll search** while any crawl is running so results update as the index grows (PRD S-3).
- **System state:** `GET /status` with summary stats and per-crawler cards; fields including **`urls_visited`**, **`urls_queued`** / **`queue_capacity`**, **`backpressure_active`**, worker counts, run/pause/finish flags; **auto-poll** of status while crawls are active (~1.5s); **Pause / Resume / Stop** calling `POST /pause|resume|stop/<crawler_id>`; visual highlight for the last-started crawler id.
- Styling: distinct dark theme (slate/indigo) and card layout—**not** the same as Project 1 at this stage.

**Prompt used — parity with Project 1:**

```
ui is not the same with the project 1. make sure everything is identical, like statuses, ui, layout, ux, etc everything
```

**Output produced — Project 1 parity:**

- Replaced the custom UI with a structure and stylesheet **aligned to `web-crawler/static/index.html`**: same header (**“Web Crawler & Search Engine”** + **`v1.0`**), vertical card stack (Index → Search → System Status), same color tokens (`#0f1117`, `#161822`, `#1e2030`, indigo/orange/green/red buttons), same **crawler row** grid and badge set (Running, Paused, Finished, Stopped, Stopping…, Backpressure), same **`stoppingSet`** behavior for “Stopping…”, same **3s** `setInterval` for `refreshStatus`, same search **`limit=30`**, same empty/searching/no-results copy, same Index success behavior (clear origin, reset depth to `2`).
- **Small API alignment:** result meta line labels **`score:`**; values read from **`relevance_score`** (and `score` if ever present)—Project 1’s static HTML referenced `r.score` while the backend returns `relevance_score`, so the parity UI displays the numeric score correctly without changing the backend.

**Decisions made / changes from output:**

- **Orchestrator / user:** Requested full **visual and UX parity** with Project 1 after the first delivery; the first UI was discarded in favor of cloning P1’s layout and interactions.
- **UI Agent:** No backend changes; no split of `index.html` into multiple files (constraints preserved).
- **Documented trade-off:** First build added extra controls (`sortBy`, live search polling, richer status cards); second build removed those to match P1’s minimal surface—acceptable because PRD/AC-6 are still satisfied by the P1-identical UI (index, search, status with queue and back-pressure).

---

### 4.4 QA Agent Session

Two QA interactions are documented: (1) implementation of the test suite against `agents/qa_agent.md`, and (2) recording that work in this workflow file.

---

**Prompt used — test implementation:**

```
@web-crawler-v2/agents/qa_agent.md @web-crawler-v2/architect_design.md @web-crawler-v2/PRD.md you are qa agent. implement your tests step by step
```

**Output produced — test implementation:**

- **Framework:** Python **`unittest` only** (stdlib); no pytest or other test dependencies, per `agents/qa_agent.md`.
- **`tests/test_parser.py`:** `parse_html` — absolute/relative links, fragment stripping, http/https-only filtering, dedupe; title and visible text; script/style/noscript skipped; empty HTML, malformed markup, anchors without `href`.
- **`tests/test_index_store.py`:** `add_page` / `search` (exact + prefix per `IndexStore.search`), stats (`register_crawler`, `update_stats`, `get_stats`), `total_pages_indexed` / `total_words_indexed`; **`threading` concurrency** — eight threads barrier-aligned on `add_page`; **read-while-write** thread calling `search` / `total_pages_indexed` while another repeatedly `add_page`s. **`STORAGE_DIR` / `STORAGE_FILE` patched** to a temp dir so tests do not read/write the project’s `data/storage/p.data`.
- **`tests/test_search_service.py`:** **`_score_entries`** — known numeric expectations (frequency × 10 + 1000 exact match − 5 × depth), prefix query without exact-word bonus, depth penalty delta; **`SearchService.search`** — dedupe by URL, `sort_by="depth"`.
- **`tests/test_crawler_depth.py`:** **`CrawlerJob`** with **`_fetch` mocked** (no network), **`_wait_rate_limit` mocked**, queue **`get` timeout shortened** so idle workers exit quickly — **`max_depth=0`** fetches only the seed URL (links not followed); **`max_depth=1`** fetches seed + direct links only (second hop not fetched).
- **`tests/test_policy_no_banned_libraries.py`:** scans **`core/`**, **`services/`**, and **`app.py`** for substrings **`scrapy`**, **`beautifulsoup`**, **`bs4`** (PRD AC-7 / QA responsibility to verify no prohibited stacks).
- **`tests/qa_prd_review.py`:** maps PRD AC-* to tests; notes partial coverage (e.g. full live **index + search** pipeline for AC-3, **queue-full back-pressure** for AC-4) and defers **UI** to manual review.
- **Run command:** `PYTHONPATH=. python -m unittest discover -s tests -p 'test_*.py' -v` — **28 tests**, all passing (~0.5s in a typical run).

**Decisions made / changes from output — test implementation:**

- **Orchestrator / QA Agent:** Delivered **additional** modules beyond the minimal two filenames in the original agent brief (`test_search_service.py`, `test_crawler_depth.py`, `test_policy_no_banned_libraries.py`) so relevancy scoring, depth semantics, and AC-7 are explicitly covered.
- **Note on depth-0:** **`max_depth=0`** is asserted on **`CrawlerJob`** directly; **`POST /index`** still requires **`k` ≥ 1**, so “seed only” is not exposed as `k=0` via the API — the test documents **engine** semantics, not the HTTP contract.
- **Backend loop:** No failing tests were reported back to the Backend Agent from this suite on first run; no revision cycle was opened from QA test failures in this session.

**Note:** The meta-prompt used to record the **Architect** session (including the request to embed “this prompt and its output”) is documented at the **end of §4.1**, not here — this subsection is only the QA test-implementation thread.

---

## 5. Key Decisions

The Architect-first approach paid off clearly. Because the API contracts and concurrency design were settled before any code was written, the Backend and UI agents had an unambiguous spec to work from. The Backend Agent produced all six modules in a single session with no interface mismatches between them — a direct result of having a complete design document rather than discovering the interface incrementally during implementation. Keeping agents scoped to a single concern also made review straightforward: when the UI was wrong, it was obviously the UI Agent's output that needed replacing, not a systemic issue, and the fix was isolated to one session.

---

## 6. What Worked and What Didn't

The Architect Agent's first pass drifted from the reference implementation in ways that weren't immediately obvious — generic naming, a wrong query parameter, a different back-pressure mechanism. This required a second session and a more specific prompt referencing Project 1 directly. The root cause was that the agent brief was written before reviewing the reference codebase. The UI Agent had the same problem: the brief said nothing about visual parity with Project 1, so the first delivery was wasted work. Both issues point to the same lesson — agent briefs should be written after reviewing any reference or constraint material, not before. If doing this again, each agent file would include explicit parity constraints and reference file paths from the start, so the first delivery lands closer to the target.

---

*End of Multi-Agent Workflow*
