# Product Requirements Document
## Web Crawler & Search Engine

---

## 1. Overview

Build a web crawling and search platform that exposes two distinct capabilities: `index` and `search`. The system must use language-native functionality for the core work (e.g., `net/http` or `urllib`) rather than high-level libraries that do the core work out of the box (e.g., Scrapy, Beautiful Soup). The scale of the crawl is assumed to be very large but does not require multiple machines.

The system must be built using a multi-agent AI workflow. Each agent is responsible for a distinct part of the system. The development process must demonstrate clear agent collaboration, with handoffs and decisions documented in `multi_agent_workflow.md`.

---

## 2. Functional Requirements

### 2.1 `index(origin, k)` — The Crawler

| # | Requirement |
|---|---|
| F-1 | Accept two parameters: `origin` (a URL from which to initiate the crawl) and `k` (the number of hops between `origin` and a newly discovered link — i.e., the maximum crawl depth). |
| F-2 | Perform a recursive crawl starting from `origin` to a maximum depth of `k`. |
| F-3 | Maintain a **Visited set** to ensure no page is crawled twice. |
| F-4 | Use language-native HTTP and HTML functionality for fetching and parsing. High-level libraries that perform the core crawling or parsing work out of the box are prohibited. |
| F-5 | Include **back-pressure**: the system must manage its own load via a maximum rate of work or a bounded queue depth, so load is controlled rather than unbounded. |

### 2.2 `search(query)` — The Query Engine

| # | Requirement |
|---|---|
| S-1 | Accept a single string parameter `query`. |
| S-2 | Return a list of triples of the form `(relevant_url, origin_url, depth)`, where `relevant_url` is the URL of an indexed page relevant to the query, and `origin_url` and `depth` are the parameters from the `/index` call under which `relevant_url` was discovered. |
| S-3 | Must be able to run while the indexer is still active, reflecting new results as they are discovered. |
| S-4 | Define and apply a simple relevancy heuristic (e.g., keyword frequency or title matching) to determine which URLs are relevant and to rank results. |

### 2.3 Concurrency

| # | Requirement |
|---|---|
| C-1 | Use thread-safe data structures (e.g., Mutexes, Channels, or Concurrent Maps) to prevent data corruption when the indexer and searcher operate concurrently. |

### 2.4 UI / CLI

| # | Requirement |
|---|---|
| U-1 | Provide a simple UI or CLI that allows the user to initiate indexing. |
| U-2 | Provide a simple UI or CLI that allows the user to initiate search. |
| U-3 | Provide a simple UI or CLI that allows the user to view the state of the system, including: indexing progress, queue depth, and back-pressure status. |

### 2.5 Parity with Project 1 (reference system)

Functional behavior and system design **match the existing Project 1 implementation** (`web-crawler`): same Flask routes and JSON shapes (including `POST /index` returning `crawler_id`, `GET /search?query=…`, per-crawler `/status/<crawler_id>`, stop/pause/resume), **multi-worker `CrawlerJob` with bounded frontier queue + rate limiting**, inverted index with word postings and the same depth and back-pressure semantics. See `architect_design.md` for the authoritative contract. **Validation:** `k` is a **positive integer** (≥ 1), consistent with Project 1.

---

## 3. Multi-Agent Requirement

The system must be built using a multi-agent AI workflow. The final codebase does not need to run as agents — the requirement is that the development process demonstrates clear agent collaboration.

| Agent | Responsibility |
|---|---|
| **Architect** | System design, component boundaries, API contracts, concurrency model |
| **Backend** | Crawler, parser, inverted index, Flask routes |
| **UI** | Single-page web frontend |
| **QA** | Tests and review of other agents' outputs against this PRD |

Agent definitions are documented in the `/agents` directory. Handoffs, decisions, and interactions between agents are documented in `multi_agent_workflow.md`.

---

## 4. Acceptance Criteria

| ID | Criterion |
|---|---|
| AC-1 | `index(origin, k)` crawls from `origin` recursively to depth `k` and never visits the same URL more than once. |
| AC-2 | `search(query)` returns a list of `(relevant_url, origin_url, depth)` triples for pages relevant to `query`, ranked by the defined relevancy heuristic. |
| AC-3 | `search(query)` can be called while `index` is running and returns results from pages indexed so far without deadlock or crash. |
| AC-4 | The back-pressure mechanism prevents the system from growing its work queue or request rate unboundedly. |
| AC-5 | Concurrent access to shared data structures does not produce data corruption. |
| AC-6 | The UI or CLI supports initiating indexing, initiating search, and viewing system state (progress, queue depth, back-pressure status). |
| AC-7 | No high-level crawling or parsing library (e.g., Scrapy, Beautiful Soup) is used for the core work. |

---

## 5. Required Deliverables

| File | Description |
|---|---|
| `PRD.md` | This document |
| `readme.md` | How the project works and how to run it |
| `recommendation.md` | 1–2 paragraphs on production deployment considerations |
| `multi_agent_workflow.md` | Explanation of agents, prompts used, interactions, and decisions made |
| `/agents` | One definition file per agent |
| GitHub repository | Working codebase |

---

*End of PRD*
