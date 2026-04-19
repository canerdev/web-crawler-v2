"""
QA written review vs PRD acceptance criteria (manual / automated coverage).

AC-1 (crawl to depth k, visited once): Covered in tests/test_crawler_depth.py with mocked fetch.
AC-2 (search triples + ranking): SearchService returns relevant_url, origin_url, depth; scoring tested in tests/test_search_service.py.
AC-3 (search while indexing): Partially covered by tests/test_index_store.py concurrent read/write; full pipeline concurrency is not integration-tested here (no live crawler + search in one test).
AC-4 (back-pressure): Not unit-tested in isolation (would need queue-full simulation or bounded queue test against CrawlerJob).
AC-5 (no corruption under concurrency): tests/test_index_store.py concurrent add + read-during-write smoke tests.
AC-6 (UI/CLI): Manual / out of scope for stdlib-only unit tests; static UI exists per project layout.
AC-7 (no Scrapy/BS4): tests/test_policy_no_banned_libraries.py scans core/, services/, app.py.

API vs UI: Routes in app.py align with architect_design (POST /index, GET /search, /status, stop/pause/resume). UI contract review is left to human inspection of static/index.html.
"""
